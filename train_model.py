from transformers import BertTokenizer, EncoderDecoderModel, TrainingArguments, Trainer
import torch
from sklearn.model_selection import train_test_split
from dataset import EDI_Dataset, load_data_with_errors
from evaluate import load as load_metric
import numpy as np
import json
import os

# Charger les données
segments_with_errors, correct_segments, labels = load_data_with_errors("synthetic_edifact_data_with_errors.json")
train_erroneous, val_erroneous, train_correct, val_correct = train_test_split(
    segments_with_errors, correct_segments, test_size=0.2
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_erroneous, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_erroneous, truncation=True, padding=True, max_length=128)

train_correct_encodings = tokenizer(train_correct, truncation=True, padding=True, max_length=128)
val_correct_encodings = tokenizer(val_correct, truncation=True, padding=True, max_length=128)

# Dataset
train_dataset = EDI_Dataset(train_encodings, train_correct_encodings)
val_dataset = EDI_Dataset(val_encodings, val_correct_encodings)

# Modèle Encoder-Decoder
model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

# Configurations du modèle
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

# Chargement de la métrique BLEU
bleu_metric = load_metric("bleu")

# Fonction de calcul des métriques
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if predictions.ndim == 3:
        predictions = predictions.argmax(axis=-1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [label.strip() for label in decoded_labels]

    preds_for_bleu = [" ".join(pred.split()) for pred in decoded_preds]
    labels_for_bleu = [[" ".join(label.split())] for label in decoded_labels]

    bleu = bleu_metric.compute(predictions=preds_for_bleu, references=labels_for_bleu)["bleu"]

    exact_matches = sum([1 for pred, label in zip(decoded_preds, decoded_labels) if pred == label])
    accuracy = exact_matches / len(decoded_labels) if decoded_labels else 0.0

    return {"accuracy": accuracy, "bleu": bleu}

# Arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Entraîneur
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Entraînement
trainer.train()

# Sauvegarde du modèle
model.save_pretrained("./trained_model_corrector")
tokenizer.save_pretrained("./trained_model_corrector")

# Amélioration 1 : Export des résultats d'évaluation
def export_evaluation_results():
    """
    Sauvegarde les résultats d'évaluation dans un fichier JSON.
    """
    eval_results = trainer.evaluate()
    eval_results_file = "evaluation_results.json"
    with open(eval_results_file, "w") as file:
        json.dump(eval_results, file, indent=4)
    print(f"Résultats d'évaluation sauvegardés dans {eval_results_file}.")

# Appel de la fonction d'export des résultats d'évaluation
export_evaluation_results()

# Amélioration 2 : Export des fichiers EDIFACT et comparaison automatique
def export_edifact_files():
    """
    Transforme les prédictions en fichiers EDIFACT lisibles et compare avec les références.
    """
    predictions_file = "predictions.json"
    output_dir = "output_edifact"

    # Vérifiez si le dataset d'évaluation est défini
    dataset_to_use = val_dataset

    if not os.path.exists(predictions_file) or os.path.getsize(predictions_file) == 0:
        print(f"Fichier {predictions_file} non trouvé ou vide. Génération des prédictions...")
        
        predictions = trainer.predict(dataset_to_use).predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        with open(predictions_file, "w") as pred_file:
            json.dump(predictions.tolist(), pred_file)
        print(f"Prédictions sauvegardées dans {predictions_file}.")

    with open(predictions_file, "r") as pred_file:
        predictions = json.load(pred_file)

    os.makedirs(output_dir, exist_ok=True)

    for idx, prediction in enumerate(predictions):
        edifact_content = f"UNA:+.? '\nUNB+UNOC:3+SENDER+RECEIVER+20231117:1259+{idx}'\n"
        edifact_content += f"UNH+{idx}+INVOIC:D:96A:UN'\n"
        edifact_content += f"{' '.join(map(str, prediction))}\n"
        edifact_content += "UNT+1+INVOIC'\nUNZ+1+00001'"

        output_file = os.path.join(output_dir, f"edifact_message_{idx + 1}.edi")
        with open(output_file, "w") as edifact_file:
            edifact_file.write(edifact_content)

    print(f"Transformation réussie ! Les fichiers EDIFACT ont été sauvegardés dans {output_dir}.")

    # Comparaison automatique
    analyze_edifact_predictions(output_dir)

def analyze_edifact_predictions(output_dir="output_edifact"):
    """
    Analyse automatiquement les prédictions EDIFACT et compare avec les références.
    """
    with open("synthetic_edifact_data_with_errors.json", "r") as file:
        data = json.load(file)

    correct_segments = data["correct_segments"]

    accuracy_count = 0
    total_files = 0

    for idx, correct_segment in enumerate(correct_segments):
        edifact_file = os.path.join(output_dir, f"edifact_message_{idx + 1}.edi")
        if os.path.exists(edifact_file):
            with open(edifact_file, "r") as file:
                prediction = file.read()

            if correct_segment in prediction:
                accuracy_count += 1
            total_files += 1

    accuracy = accuracy_count / total_files if total_files > 0 else 0.0
    print(f"Accuracy sur les fichiers EDIFACT : {accuracy:.2%}")

# Appel de la fonction d'export des fichiers EDIFACT
export_edifact_files()
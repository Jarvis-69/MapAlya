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
    """
    Calcule les métriques d'évaluation, y compris BLEU et exact match.
    """
    predictions, labels = eval_pred

    # Vérifiez et ajustez les prédictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if predictions.ndim == 3:  # Si les prédictions sont de dimension (batch_size, seq_len, vocab_size)
        predictions = predictions.argmax(axis=-1)

    # Décodage des prédictions et labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [label.strip() for label in decoded_labels]

    # Ajustement pour le calcul de BLEU
    preds_for_bleu = [" ".join(pred.split()) for pred in decoded_preds]  # Transforme en séquences valides
    labels_for_bleu = [[" ".join(label.split())] for label in decoded_labels]  # Références en format correct

    # Vérifiez les formats pour diagnostiquer
    print("=== DIAGNOSTIC ===")
    print(f"Predictions BLEU format: {preds_for_bleu[:3]}")
    print(f"References BLEU format: {labels_for_bleu[:3]}")

    # Calcul du score BLEU
    bleu = bleu_metric.compute(predictions=preds_for_bleu, references=labels_for_bleu)["bleu"]

    # Calcul de l'accuracy
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

def export_edifact_files():
    """
    Fonction pour exporter les prédictions en fichier EDIFACT.
    """
    predictions_file = "predictions.json"

    # Vérifiez si le dataset d'évaluation est défini
    dataset_to_use = eval_dataset if 'eval_dataset' in globals() else val_dataset
    if dataset_to_use is None:
        raise ValueError("Ni `eval_dataset` ni `val_dataset` ne sont définis. Assurez-vous qu'un jeu de données est disponible pour l'évaluation.")
    
    if not os.path.exists(predictions_file) or os.path.getsize(predictions_file) == 0:
        print(f"Fichier {predictions_file} non trouvé ou vide. Génération des prédictions...")
        
        # Générez les prédictions
        predictions = trainer.predict(dataset_to_use).predictions
        
        # Si predictions est un tuple, extraire le premier élément
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Sauvegarder les prédictions
        with open(predictions_file, "w") as pred_file:
            json.dump(predictions.tolist(), pred_file)
        print(f"Prédictions sauvegardées dans {predictions_file}.")

    # Lecture des prédictions
    if os.path.getsize(predictions_file) > 0:  # Assurez-vous que le fichier n'est pas vide avant de lire
        with open(predictions_file, "r") as pred_file:
            predictions = json.load(pred_file)
            print(f"Prédictions chargées depuis {predictions_file}.")
    else:
        raise ValueError(f"Le fichier {predictions_file} est vide après tentative d'écriture. Vérifiez les prédictions générées.")
    
    # Ajoutez ici la logique pour convertir en EDIFACT
    print("Traitement des prédictions pour générer des fichiers EDIFACT...")

# Appel de la fonction d'export
export_edifact_files()
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
from sklearn.model_selection import train_test_split

# Charger les données
def load_data():
    data = {
        "data": [
            {
                "incorrect_mapping": [
                    {"segment": "BGM", "element": "Numéro de document", "xml": "<CODE_CLIENT_CHEZ_FRN>"},
                    {"segment": "DTM+137", "element": "Code client", "xml": "<DATE_DOCUMENT>"},
                    {"segment": "NAD+BY", "element": "Numéro de document", "xml": "<NO_DOCUMENT>"},
                    {"segment": "NAD+SU", "element": "Code fournisseur", "xml": "<CODE_FOURNISSEUR>"},
                    {"segment": "MOA+79", "element": "Montant HT", "xml": "<MONTANT_HT>"},
                    {"segment": "TAX+7 et MOA+124", "element": "TVA (25%)", "xml": "<TAUX_TVA>, <MONTANT_TVA>"},
                    {"segment": "LIN", "element": "Numéro de ligne", "xml": "<NUM_LIG>"},
                    {"segment": "LIN", "element": "Référence article", "xml": "<REF_NART>"},
                    {"segment": "QTY+47", "element": "Quantité", "xml": "<QUANTITE_FACTUREE>"},
                    {"segment": "PRI+AAA", "element": "Montant HT", "xml": "<PRIX_NET_PAR_UNITE>"}
                ],
                "correct_mapping": [
                    {"segment": "BGM", "element": "Numéro de document", "xml": "<NO_DOCUMENT>"},
                    {"segment": "DTM+137", "element": "Date de document", "xml": "<DATE_DOCUMENT>"},
                    {"segment": "NAD+BY", "element": "Code client", "xml": "<CODE_CLIENT_CHEZ_FRN>"},
                    {"segment": "NAD+SU", "element": "Code fournisseur", "xml": "<CODE_FOURNISSEUR>"},
                    {"segment": "MOA+79", "element": "Montant HT", "xml": "<MONTANT_HT>"},
                    {"segment": "TAX+7 et MOA+124", "element": "TVA (25%)", "xml": "<TAUX_TVA>, <MONTANT_TVA>"},
                    {"segment": "LIN", "element": "Numéro de ligne", "xml": "<NUM_LIG>"},
                    {"segment": "LIN", "element": "Référence article", "xml": "<REF_NART>"},
                    {"segment": "QTY+47", "element": "Quantité", "xml": "<QUANTITE_FACTUREE>"},
                    {"segment": "PRI+AAA", "element": "Prix unitaire", "xml": "<PRIX_NET_PAR_UNITE>"}
                ]
            }
        ]
    }

    incorrect_mappings = []
    correct_mappings = []

    for entry in data["data"]:
        incorrect = " ".join([f"{item['segment']}:{item['xml']}" for item in entry["incorrect_mapping"]])
        correct = " ".join([f"{item['segment']}:{item['xml']}" for item in entry["correct_mapping"]])
        incorrect_mappings.append(incorrect)
        correct_mappings.append(correct)

    # Si les données sont insuffisantes pour un split, tout utiliser pour l'entraînement
    if len(incorrect_mappings) <= 1:
        return incorrect_mappings, [], correct_mappings, []

    return train_test_split(incorrect_mappings, correct_mappings, test_size=0.2, random_state=42)

# Charger les données
train_erroneous, val_erroneous, train_correct, val_correct = load_data()

# Initialisation du tokenizer et du modèle
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Préparation des données
def encode_data(data, tokenizer, max_length=128):
    return tokenizer(data, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

train_encodings = encode_data(train_erroneous, tokenizer)
val_encodings = encode_data(val_erroneous, tokenizer) if val_erroneous else None
train_labels = encode_data(train_correct, tokenizer)
val_labels = encode_data(val_correct, tokenizer) if val_correct else None

class EDI_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels["input_ids"][idx]
        }

train_dataset = EDI_Dataset(train_encodings, train_labels)
val_dataset = EDI_Dataset(val_encodings, val_labels) if val_encodings else None

# Ajustements du modèle
model.resize_token_embeddings(len(tokenizer))

# Utiliser le CPU
device = torch.device("cpu")

# Entraînement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch" if val_dataset else "no",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=3,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()

# Sauvegarde du modèle
model.save_pretrained("./trained_model_corrector")
tokenizer.save_pretrained("./trained_model_corrector")

# Test avec tous les exemples
test_input = train_erroneous + val_erroneous
test_encodings = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True).to(device)

# Génération et analyse
test_output = model.generate(
    test_encodings["input_ids"],
    max_length=50,
    num_beams=8,
    no_repeat_ngram_size=2,
    repetition_penalty=3.0,<
    early_stopping=True
)

decoded_output = tokenizer.batch_decode(test_output, skip_special_tokens=True)
print("\n=== Analyse Étape par Étape ===")
for i, input_text in enumerate(test_input):
    print(f"\nEntrée {i + 1} : {input_text}")
    print(f"Sortie générée : {decoded_output[i]}")

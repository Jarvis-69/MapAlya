import random
import json

# Définir les classes et leurs correspondances EDIFACT - XML
edifact_xml_mapping = [
    {"segment": "BGM+{}", "xml_element": "<NO_DOCUMENT>", "label": 0},
    {"segment": "DTM+137:{}", "xml_element": "<DATE_DOCUMENT>", "label": 1},
    {"segment": "NAD+BY+{}", "xml_element": "<CODE_CLIENT_CHEZ_FRN>", "label": 2},
    {"segment": "NAD+SU+{}", "xml_element": "<CODE_FOURNISSEUR>", "label": 3},
    {"segment": "MOA+79:{}", "xml_element": "<MONTANT_HT>", "label": 4},
    {"segment": "TAX+7+{}:MOA+124:{}", "xml_element": "<TAUX_TVA>, <MONTANT_TVA>", "label": 5},
    {"segment": "LIN+{}", "xml_element": "<NUM_LIG>", "label": 6},
]

# Fonctions pour générer des valeurs valides
def generate_document_number():
    return str(random.randint(100000, 999999))

def generate_date():
    year = random.randint(2020, 2025)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}{month:02d}{day:02d}"

def generate_code():
    return str(random.randint(100000000, 999999999))

def generate_amount():
    return f"{random.uniform(100.0, 10000.0):.2f}"

# Fonction pour introduire des erreurs dans les segments
def introduce_error(segment):
    errors = [
        lambda s: s.replace("+", ""),  # Supprimer un délimiteur "+"
        lambda s: s.replace(":", ";"), # Remplacer ":" par ";"
        lambda s: s[:-1],              # Supprimer le dernier caractère
        lambda s: random.choice(["UNA", "ZZZ"]) + "+" + s, # Ajouter un segment aléatoire au début
    ]
    if random.random() < 0.3:  # 30% des segments contiennent des erreurs
        return random.choice(errors)(segment)
    return segment

# Génération de données synthétiques avec erreurs
def generate_synthetic_data_with_errors(num_samples=1000):
    data = []
    for _ in range(num_samples):
        mapping = random.choice(edifact_xml_mapping)
        if mapping["label"] == 0:
            segment = mapping["segment"].format(generate_document_number())
        elif mapping["label"] == 1:
            segment = mapping["segment"].format(generate_date())
        elif mapping["label"] in [2, 3]:
            segment = mapping["segment"].format(generate_code())
        elif mapping["label"] == 4:
            segment = mapping["segment"].format(generate_amount())
        elif mapping["label"] == 5:
            segment = mapping["segment"].format("25", generate_amount())
        elif mapping["label"] == 6:
            segment = mapping["segment"].format(random.randint(1, 100))

        # Ajouter des erreurs dans le segment
        erroneous_segment = introduce_error(segment)

        data.append({
            "original_segment": segment,       # Segment original (correct)
            "erroneous_segment": erroneous_segment,  # Segment avec erreurs
            "xml_element": mapping["xml_element"],
            "label": mapping["label"]
        })
    
    with open("synthetic_edifact_data_with_errors.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    generate_synthetic_data_with_errors(1000)
    print("Jeu de données avec erreurs généré et sauvegardé dans 'synthetic_edifact_data_with_errors.json'")
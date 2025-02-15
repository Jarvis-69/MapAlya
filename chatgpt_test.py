from g4f.client import Client
from sklearn.model_selection import train_test_split
import asyncio
import platform
import json
from termcolor import colored  # Importer termcolor pour la coloration du texte

# Configuration pour Windows (boucle d'événement asynchrone)
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
            },
        ]
    }

    incorrect_mappings = []
    correct_mappings = []

    for entry in data["data"]:
        incorrect = " ".join([f"{item['segment']}:{item['xml']}" for item in entry["incorrect_mapping"]])
        correct = " ".join([f"{item['segment']}:{item['xml']}" for item in entry["correct_mapping"]])
        incorrect_mappings.append(incorrect)
        correct_mappings.append(correct)

    return incorrect_mappings, correct_mappings

# Charger les données
incorrect_mappings, correct_mappings = load_data()

# Utilisation du client `g4f.client`
def generate_corrections(incorrect_mappings):
    client = Client()
    outputs = []
    token_counts = []
    for input_text in incorrect_mappings:
        prompt = (
            "Corrige uniquement les éléments incorrects dans le mappage EDIFACT suivant. "
            "Ne modifie pas les segments qui sont déjà corrects.\n\n"
            "Retourne le mappage corrigé au format JSON sans explications supplémentaires.\n\n"
            f"Incorrect : {input_text}\n"
            "Correct :"
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                n=1,
                temperature=0.0
            )
            # Vérifier que la réponse contient des choix
            if response.choices:
                generated_output = response.choices[0].message.content.strip()
                # Tenter de parser la sortie en JSON
                try:
                    generated_json = json.loads(generated_output)
                    outputs.append(generated_json)
                except json.JSONDecodeError:
                    # Si le parsing échoue, ajouter la sortie brute
                    outputs.append(generated_output)
                # Comptage des tokens (si disponible)
                usage = getattr(response, 'usage', None)
                if usage and 'total_tokens' in usage:
                    token_counts.append(usage['total_tokens'])
                else:
                    # Estimation simple du nombre de tokens (par mots)
                    token_counts.append(len(generated_output.split()))
            else:
                outputs.append("Erreur : Aucune réponse générée.")
                token_counts.append(0)
        except Exception as e:
            outputs.append(f"Erreur : {str(e)}")
            token_counts.append(0)

    return outputs, token_counts

# Valider la sortie générée
def validate_output(output):
    if isinstance(output, dict):
        # Supposons que le JSON a une clé 'correct_mapping'
        return output.get("correct_mapping", "Erreur : Clé 'correct_mapping' manquante.")
    elif isinstance(output, str):
        # Tenter de diviser la chaîne si possible
        try:
            output = output.split("Correct :")[1].strip()
            return output
        except:
            return "Erreur : Impossible de valider la sortie."
    else:
        return "Erreur : Format de sortie inconnu."

# Fonction pour mettre en évidence les différences
def highlight_differences(incorrect, correct):
    # Convertir les mappings en dictionnaires pour une comparaison plus facile
    def parse_mapping(mapping_str):
        segments = mapping_str.split(" ")
        mapping_dict = {}
        for seg in segments:
            try:
                key, value = seg.split(":", 1)  # Split only on the first colon
                mapping_dict[key] = value
            except ValueError:
                continue
        return mapping_dict

    incorrect_dict = parse_mapping(incorrect)
    correct_dict = parse_mapping(correct)

    highlighted_output = ""
    for segment, xml in incorrect_dict.items():
        correct_xml = correct_dict.get(segment, None)
        if correct_xml and xml != correct_xml:
            # Mettre en évidence le segment incorrect en rouge
            highlighted_output += colored(f"{segment}:{xml} ", 'red')
        else:
            # Segment correct en vert
            highlighted_output += colored(f"{segment}:{xml} ", 'green')
    return highlighted_output.strip()

# Génération des sorties
generated_outputs, token_counts = generate_corrections(incorrect_mappings)

# Validation et affichage des résultats avec mise en avant des segments incorrects
for i, input_text in enumerate(incorrect_mappings):
    expected_output = correct_mappings[i]
    generated_output = generated_outputs[i]
    tokens_used = token_counts[i]

    # Valider la sortie générée
    validated_output = validate_output(generated_output)

    # Utiliser la fonction de mise en évidence
    highlighted = highlight_differences(input_text, expected_output)

    print(f"\nEntrée {i + 1} : {input_text}")
    print("----------------------------------------------------------------")
    print("Segments après correction (les segments incorrects sont en rouge) :")
    print(highlighted)
    print("----------------------------------------------------------------")
    print(f"Sortie attendue : {expected_output}")
    print("----------------------------------------------------------------")
    print(f"Sortie générée : {validated_output}")
    print("----------------------------------------------------------------")
    print(f"Tokens utilisés : {tokens_used}")
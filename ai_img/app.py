from flask import Flask, request, jsonify
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import io

app = Flask(__name__)

# Charger le modèle MobileNetV2
model = MobileNetV2(weights="imagenet")

def prepare_image(image, target_size):
    image = image.resize(target_size)  # Redimensionner l'image
    image = img_to_array(image)  # Convertir en tableau
    image = preprocess_input(image)  # Prétraitement pour MobileNetV2
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension supplémentaire
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Obtenez le fichier depuis la requête
    file = request.files["file"]

    # Convertir le fichier en BytesIO pour le rendre compatible avec load_img
    image = load_img(io.BytesIO(file.read()), target_size=(224, 224))

    # Préparer l'image pour le modèle
    image = prepare_image(image, target_size=(224, 224))

    # Faire la prédiction
    preds = model.predict(image)
    results = decode_predictions(preds, top=1)[0]  # Retourner la meilleure prédiction
    
    # Formater les résultats
    predictions = [{"label": label, "score": float(score)} for _, label, score in results]

    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(port=5000, debug=True)

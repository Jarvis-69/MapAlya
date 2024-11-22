import tensorflow as tf

# Charger un modèle pré-entraîné (ex: MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Préparer le modèle pour les prédictions
model.summary()
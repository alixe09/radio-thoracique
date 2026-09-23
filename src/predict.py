"""Chargement du modèle entraîné et prédiction sur une image unique."""

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from preprocessing import IMG_SIZE
from train import MODEL_PATH

CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIE"}


def load_trained_model(model_path=MODEL_PATH):
    return load_model(model_path)


def preprocess_image(image: Image.Image, img_size=IMG_SIZE):
    image = image.convert("RGB").resize(img_size)
    array = np.array(image, dtype="float32")
    return np.expand_dims(array, axis=0)


def predict(model, image: Image.Image, threshold=0.5):
    img_array = preprocess_image(image)
    proba = float(model.predict(img_array, verbose=0)[0][0])
    label = CLASS_NAMES[int(proba > threshold)]
    return label, proba, img_array

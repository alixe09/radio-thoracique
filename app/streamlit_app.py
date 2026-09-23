import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from gradcam import make_gradcam_heatmap, overlay_heatmap
from predict import load_trained_model, predict

st.set_page_config(page_title="Détection de pneumonie sur radio thoracique", page_icon="🫁")

st.title("🫁 Détection de pneumonie sur radio thoracique")
st.caption(
    "Projet de deep learning appliqué à l'imagerie médicale — "
    "CNN par transfer learning (EfficientNetB0) + Grad-CAM."
)

st.warning(
    "⚠️ Outil pédagogique réalisé dans un cadre de projet personnel / recherche de "
    "stage. Ce n'est **pas** un dispositif médical et ne doit pas être utilisé pour "
    "un diagnostic réel.",
    icon="⚠️",
)


@st.cache_resource
def get_model():
    return load_trained_model()


threshold = st.slider(
    "Seuil de détection de la pneumonie",
    min_value=0.3,
    max_value=0.95,
    value=0.5,
    step=0.05,
    help="Plus le seuil est bas, plus le modèle détecte de pneumonies (sensibilité "
    "élevée) mais plus il génère de fausses alertes sur des radios normales.",
)

st.info(
    "Le modèle est entraîné sur des radios **pédiatriques** (1 à 5 ans) d'un seul "
    "hôpital : sur d'autres types d'images (adultes, autres appareils, photos), "
    "il a tendance à répondre « pneumonie » avec un score élevé, à tort."
)

uploaded_file = st.file_uploader(
    "Charger une radio thoracique (JPEG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    try:
        model = get_model()
    except (OSError, ValueError):
        st.error(
            "Modèle introuvable. Entraîne-le d'abord avec `python src/train.py` "
            "(voir le README)."
        )
        st.stop()

    label, proba, img_array = predict(model, image, threshold)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image d'origine")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Zones d'attention du modèle (Grad-CAM)")
        heatmap = make_gradcam_heatmap(img_array, model)
        overlay = overlay_heatmap(img_array[0], heatmap)
        st.image(overlay, use_container_width=True)

    if label == "PNEUMONIE":
        st.error(f"**Prédiction : {label}** (score de pneumonie : {proba:.1%})")
    else:
        st.success(f"**Prédiction : {label}** (score de pneumonie : {proba:.1%})")

    st.caption(
        "La heatmap Grad-CAM met en évidence les régions de l'image qui ont le "
        "plus influencé la prédiction du modèle."
    )
else:
    st.info("Charge une image de radio thoracique pour lancer une prédiction.")

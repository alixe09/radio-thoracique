"""Chargement des images de radios thoraciques en générateurs Keras.

Le dossier `val/` officiel du dataset Kaggle ne contient que 16 images
(8 par classe), trop peu pour un suivi fiable pendant l'entraînement.
On reconstruit donc une validation à partir d'un split stratifié de `train/`
(validation_split), et on garde `test/` intact comme jeu de test final.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15

DATA_DIR = "data/raw/chest_xray"


def get_generators(
    data_dir=DATA_DIR,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False,
        validation_split=validation_split,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        color_mode="rgb",
        subset="training",
        shuffle=True,
        seed=42,
    )
    val_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        color_mode="rgb",
        subset="validation",
        shuffle=False,
        seed=42,
    )
    test_gen = eval_datagen.flow_from_directory(
        f"{data_dir}/test",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        color_mode="rgb",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen

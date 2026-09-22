"""Entraînement d'un classifieur pneumonie/normal par transfer learning."""

import json

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from preprocessing import IMG_SIZE, get_generators

MODEL_PATH = "models/best_model.keras"
HISTORY_PATH = "models/history.json"
EPOCHS_HEAD = 10
EPOCHS_FINE_TUNE = 5


def build_model(img_size=IMG_SIZE):
    base_model = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*img_size, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def get_class_weights(train_gen):
    classes = train_gen.classes
    weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(classes), y=classes
    )
    return dict(enumerate(weights))


def main():
    train_gen, val_gen, test_gen = get_generators()
    class_weights = get_class_weights(train_gen)
    print("Classes :", train_gen.class_indices)
    print("Poids de classes :", class_weights)

    model, base_model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True),
        EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    print("\n--- Phase 1 : entraînement de la tête (base gelée) ---")
    history_head = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print("\n--- Phase 2 : fine-tuning (dégel des dernières couches) ---")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINE_TUNE,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    history = {
        k: history_head.history[k] + history_fine.history[k] for k in history_head.history
    }
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

    print("\n--- Évaluation sur le jeu de test ---")
    y_true = test_gen.classes
    y_pred_proba = model.predict(test_gen).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)

    print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices)))
    print("ROC-AUC :", roc_auc_score(y_true, y_pred_proba))


if __name__ == "__main__":
    main()

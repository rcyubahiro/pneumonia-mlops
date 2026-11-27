"""Model creation, training and testing (offline). Run:  python -m src.model --epochs 15"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

from src import config
from src.preprocessing import list_images, make_dataset, split_train_val


def build_model(lr: float = 1e-3, pretrained: bool | None = None):
    """Transfer learning: ImageNet-pretrained MobileNetV2 + a small trainable head.
    Optimisation techniques: pretrained backbone, two-phase fine-tuning, augmentation,
    dropout, L2, Adam, early stopping, LR scheduling, class weights."""
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    pretrained = config.PRETRAINED if pretrained is None else pretrained
    l2 = regularizers.l2(1e-4)
    inputs = keras.Input(shape=(*config.IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)           # augmentation (train-time only)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)       # MobileNetV2 expects inputs in [-1, 1]
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet" if pretrained else None,
                                          input_shape=(*config.IMG_SIZE, 3))
    base.trainable = False                                # phase 1: train only the new head
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)    # P(PNEUMONIA)
    model = keras.Model(inputs, outputs, name="pneumonia_cnn")
    compile_model(model, lr)
    return model


def unfreeze_top(model, n_layers: int = 60):
    """Phase 2: unfreeze the top of the backbone (BatchNorm stays frozen) for fine-tuning."""
    from tensorflow import keras

    base = next(l for l in model.layers if isinstance(l, keras.Model))
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def compile_model(model, lr: float):
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )


def fit_model(model, train_paths, train_labels, val_paths, val_labels, epochs: int):
    from tensorflow import keras

    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array(train_labels))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    history = model.fit(
        make_dataset(train_paths, train_labels, True),
        validation_data=make_dataset(val_paths, val_labels, False),
        epochs=epochs,
        class_weight={0: float(weights[0]), 1: float(weights[1])},
        callbacks=callbacks,
        verbose=2,
    )
    return {k: [float(v) for v in vals] for k, vals in history.history.items()}


def evaluate_model(model, paths, labels) -> dict:
    """Testing: accuracy, loss, precision, recall, F1, ROC-AUC, confusion matrix.
    Precision / recall / F1 treat 'PNEUMONIA' (label 1) as the positive class,
    because missing a pneumonia case is the costly error."""
    probs = model.predict(make_dataset(paths, labels, False), verbose=0).ravel()
    y_true = np.asarray(labels)
    y_pred = (probs >= 0.5).astype(int)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": float(log_loss(y_true, np.clip(probs, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def save_metrics(kind: str, test: dict, history: dict, n_train: int, n_val: int, previous: dict | None = None):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": kind,
        "trained_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_train": n_train,
        "n_val": n_val,
        "test": test,
        "previous_test": previous,
        "history": history,
    }
    config.METRICS_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def train_initial(epochs: int = 15):
    paths, labels = list_images(config.TRAIN_DIR)
    if not paths:
        raise SystemExit(f"No images found in {config.TRAIN_DIR}. Run scripts/download_data.py first.")
    tr_p, va_p, tr_l, va_l = split_train_val(paths, labels)
    model = build_model()
    head_epochs = max(3, epochs // 3)
    h1 = fit_model(model, tr_p, tr_l, va_p, va_l, head_epochs)            # phase 1: head only
    unfreeze_top(model)
    compile_model(model, 1e-5)                                            # phase 2: low-LR fine-tuning
    h2 = fit_model(model, tr_p, tr_l, va_p, va_l, max(1, epochs - head_epochs))
    history = {k: h1.get(k, []) + h2.get(k, []) for k in h1}

    te_p, te_l = list_images(config.TEST_DIR)
    test = evaluate_model(model, te_p, te_l)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.MODEL_PATH)
    save_metrics("initial", test, history, len(tr_p), len(va_p))
    print(json.dumps(test, indent=2))
    return test


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    train_initial(ap.parse_args().epochs)

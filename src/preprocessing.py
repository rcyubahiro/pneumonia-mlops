"""Data processing: listing, decoding, resizing, splitting and simple image features.

The SAME decode+resize function is used for training and inference so the model
never sees a different preprocessing at prediction time. Pixel rescaling (0-255 -> 0-1)
lives inside the model itself (Rescaling layer).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src import config

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


# ---------- listing / validation ----------
def list_images(root) -> tuple[list[str], list[int]]:
    """Read root/<ClassName>/* and return (paths, integer labels)."""
    root = Path(root)
    paths, labels = [], []
    for idx, name in enumerate(config.CLASS_NAMES):
        for p in sorted((root / name).glob("*")):
            if p.suffix.lower() in VALID_EXT:
                paths.append(str(p))
                labels.append(idx)
    return paths, labels


def validate_image(data: bytes) -> bool:
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def split_train_val(paths, labels, val_size=0.15, seed=config.SEED):
    return train_test_split(paths, labels, test_size=val_size, stratify=labels, random_state=seed)


# ---------- decoding / resizing (TensorFlow, shared by train + inference) ----------
def decode_resize(raw):
    import tensorflow as tf

    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, config.IMG_SIZE)
    img.set_shape((*config.IMG_SIZE, 3))
    return tf.cast(img, tf.float32)


def preprocess_bytes(data: bytes) -> np.ndarray:
    """Raw image bytes -> (1, H, W, 3) float32 array in 0-255 range."""
    return decode_resize(data).numpy()[None, ...]


def make_dataset(paths, labels, training: bool):
    import tensorflow as tf

    def _load(path, label):
        return decode_resize(tf.io.read_file(path)), tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((list(map(str, paths)), np.asarray(labels, dtype=np.float32)))
    if training:
        ds = ds.shuffle(len(paths), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        # contrast/brightness jitter for robustness to different X-ray sources
        # (done in the data pipeline, not as model layers, so the saved .h5 stays reloadable)
        def _augment(img, label):
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_brightness(img, 20.0)
            return tf.clip_by_value(img, 0.0, 255.0), label

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------- interpretable features for the visualisations ----------
def image_features(path) -> dict:
    """Three human-interpretable features per chest X-ray."""
    img = Image.open(path).convert("RGB").resize(config.IMG_SIZE)
    rgb = np.asarray(img, dtype=np.float32)
    gray = rgb.mean(axis=2)
    return {
        "brightness": float(gray.mean()),                  # overall radiographic density
        "contrast": float(gray.std()),                     # spread of grey levels (sharpness of structures)
        "bright_pixel_fraction": float((gray > 200).mean()),  # share of very white pixels (opacities)
    }

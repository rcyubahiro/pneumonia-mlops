"""Prediction: thread-safe model holder that can be hot-reloaded after retraining."""
from __future__ import annotations

import threading

from src import config
from src.preprocessing import preprocess_bytes


class Predictor:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        if not config.MODEL_PATH.exists():
            return False
        from tensorflow import keras

        model = keras.models.load_model(config.MODEL_PATH, compile=False)
        with self._lock:
            self._model = model
        return True

    def predict_bytes(self, data: bytes) -> dict:
        if self._model is None and not self.load():
            raise RuntimeError("Model file not found. Train the model first.")
        x = preprocess_bytes(data)
        with self._lock:
            p_pneumonia = float(self._model(x, training=False).numpy().ravel()[0])
        idx = int(p_pneumonia >= 0.5)
        return {
            "label": config.CLASS_NAMES[idx],
            "confidence": round(p_pneumonia if idx == 1 else 1 - p_pneumonia, 4),
            "probabilities": {
                config.CLASS_NAMES[0]: round(1 - p_pneumonia, 4),
                config.CLASS_NAMES[1]: round(p_pneumonia, 4),
            },
        }


predictor = Predictor()

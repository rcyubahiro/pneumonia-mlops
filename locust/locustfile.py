"""Flood the prediction endpoint.

Web UI:    locust -f locust/locustfile.py --host http://localhost:8080
Headless:  locust -f locust/locustfile.py --host http://localhost:8080 \
                  --headless -u 100 -r 10 -t 60s --csv results/run
"""
import io
import os
import random
from pathlib import Path

from locust import HttpUser, between, task
from PIL import Image

SAMPLE_DIR = Path(os.getenv("SAMPLE_DIR", Path(__file__).resolve().parent.parent / "data" / "test"))
SAMPLES = [p for p in SAMPLE_DIR.rglob("*.jpeg")][:50]


def _random_image() -> bytes:
    """Fallback when no real test images are present."""
    buf = io.BytesIO()
    Image.effect_noise((128, 128), 60).convert("RGB").save(buf, "JPEG")
    return buf.getvalue()


IMAGES = [p.read_bytes() for p in SAMPLES] or [_random_image()]


class PredictionUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        self.client.post("/predict", files={"file": ("xray.jpeg", random.choice(IMAGES), "image/jpeg")},
                         name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")

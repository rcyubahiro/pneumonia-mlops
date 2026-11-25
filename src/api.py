"""FastAPI service.  Run:  uvicorn src.api:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import json
import random
import socket
import time
import uuid
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request

from src import config, database, retrain
from src.prediction import predictor
from src.preprocessing import image_features, list_images, validate_image

app = FastAPI(title="Chest X-ray Pneumonia Classifier API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

START_TIME = time.time()
HOST = socket.gethostname()
counters = {"requests": 0, "predictions": 0, "errors": 0, "total_latency_ms": 0.0}
_insights_cache: dict | None = None


@app.middleware("http")
async def track_requests(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        counters["errors"] += 1
        raise
    ms = (time.perf_counter() - t0) * 1000
    counters["requests"] += 1
    counters["total_latency_ms"] += ms
    if response.status_code >= 500:
        counters["errors"] += 1
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    response.headers["X-Served-By"] = HOST      # shows which container answered
    return response


@app.on_event("startup")
def startup():
    for d in (config.UPLOAD_DIR / c for c in config.CLASS_NAMES):
        d.mkdir(parents=True, exist_ok=True)
    database.init_db()
    predictor.load()


# ---------------- monitoring ----------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_loaded": predictor.loaded,
        "container": HOST,
    }


@app.get("/stats")
def stats():
    n = max(counters["requests"], 1)
    train_p, train_l = list_images(config.TRAIN_DIR)
    test_p, _ = list_images(config.TEST_DIR)
    return {
        **counters,
        "avg_latency_ms": round(counters["total_latency_ms"] / n, 2),
        "train_images": len(train_p),
        "test_images": len(test_p),
        "uploaded_images": database.total_count(),
        "pending_for_retrain": database.pending_count(),
        "retrain_threshold": config.RETRAIN_THRESHOLD,
        "auto_retrain": config.AUTO_RETRAIN,
    }


@app.get("/metrics")
def metrics():
    if not config.METRICS_PATH.exists():
        raise HTTPException(404, "No metrics yet - train the model first.")
    return json.loads(config.METRICS_PATH.read_text())


@app.get("/insights")
def insights(refresh: bool = False, per_class: int = 200):
    """Feature distributions per class for the Visualizations page."""
    global _insights_cache
    if _insights_cache is not None and not refresh:
        return _insights_cache
    rng = random.Random(config.SEED)
    out = {"class_counts": {}, "features": {}}
    paths, labels = list_images(config.TRAIN_DIR)
    for i, name in enumerate(config.CLASS_NAMES):
        class_paths = [p for p, l in zip(paths, labels) if l == i]
        out["class_counts"][name] = len(class_paths)
        for p in rng.sample(class_paths, min(per_class, len(class_paths))):
            for feat, val in image_features(p).items():
                out["features"].setdefault(feat, {}).setdefault(name, []).append(round(val, 4))
    _insights_cache = out
    return out


# ---------------- prediction ----------------
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read()
    if not validate_image(data):
        raise HTTPException(400, "Uploaded file is not a valid image.")
    try:
        result = predictor.predict_bytes(data)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    counters["predictions"] += 1
    return {**result, "container": HOST}


# ---------------- upload + retrain ----------------
@app.post("/upload")
def upload(files: List[UploadFile] = File(...), label: str = Form(...)):
    """Bulk upload labelled images. Files go to disk (data/uploads/<label>/) and SQLite."""
    if label not in config.CLASS_NAMES:
        raise HTTPException(400, f"label must be one of {config.CLASS_NAMES}")
    dest = config.UPLOAD_DIR / label
    dest.mkdir(parents=True, exist_ok=True)
    saved, rejected = 0, []
    for f in files:
        data = f.file.read()
        if not validate_image(data):
            rejected.append(f.filename)
            continue
        ext = (f.filename or "x.png").rsplit(".", 1)[-1].lower()
        path = dest / f"{uuid.uuid4().hex}.{ext if ext in ('png', 'jpg', 'jpeg', 'bmp') else 'png'}"
        path.write_bytes(data)
        database.add_upload(f.filename, label, str(path))
        saved += 1

    auto_started = False
    if config.AUTO_RETRAIN and retrain.should_retrain():
        auto_started = retrain.start_retrain_async()
    return {"saved": saved, "rejected": rejected, "pending_for_retrain": database.pending_count(),
            "auto_retrain_started": auto_started}


@app.post("/retrain")
def trigger_retrain(epochs: int = config.RETRAIN_EPOCHS):
    if database.total_count() == 0:
        raise HTTPException(400, "Upload new images before retraining.")
    if not retrain.start_retrain_async(epochs):
        raise HTTPException(409, "Retraining already running.")
    return {"status": "started"}


@app.get("/retrain/status")
def retrain_status():
    return {**retrain.state, "pending_for_retrain": database.pending_count()}

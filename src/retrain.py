"""Retraining pipeline + trigger.

Pipeline:  uploaded files (already saved to disk + SQLite by the API)
           -> preprocessing (same decode/resize as training)
           -> fine-tune the EXISTING saved model (pre-trained) at a low learning rate
           -> evaluate on the untouched test set, compare with the previous model
           -> save new model, back up the old one, hot-reload the API.

Trigger:   POST /retrain (UI button)  OR  automatically once RETRAIN_THRESHOLD new
           images are waiting (AUTO_RETRAIN=true).  Or from a terminal: python -m src.retrain
"""
from __future__ import annotations

import datetime as dt
import random
import shutil
import threading
import traceback

from src import config, database
from src.preprocessing import list_images, split_train_val

_lock = threading.Lock()
state = {"status": "idle", "message": "", "started_at": None, "finished_at": None, "result": None}


def should_retrain() -> bool:
    """The retraining trigger condition."""
    return database.pending_count() >= config.RETRAIN_THRESHOLD


def retrain(epochs: int = config.RETRAIN_EPOCHS, lr: float = 1e-4) -> dict:
    from tensorflow import keras

    from src.model import compile_model, evaluate_model, fit_model, save_metrics

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("No pre-trained model found - run python -m src.model first.")

    up_paths, up_labels = list_images(config.UPLOAD_DIR)
    if not up_paths:
        raise ValueError("No uploaded images to retrain on.")

    # replay a sample of the original data so the model does not forget it
    base_paths, base_labels = list_images(config.TRAIN_DIR)
    rng = random.Random(config.SEED)
    idx = list(range(len(base_paths)))
    rng.shuffle(idx)
    idx = idx[: config.RETRAIN_BASE_SAMPLES]
    paths = [base_paths[i] for i in idx] + up_paths
    labels = [base_labels[i] for i in idx] + up_labels
    tr_p, va_p, tr_l, va_l = split_train_val(paths, labels)

    te_p, te_l = list_images(config.TEST_DIR)
    model = keras.models.load_model(config.MODEL_PATH, compile=False)   # custom pre-trained model
    compile_model(model, lr)
    previous = evaluate_model(model, te_p, te_l) if te_p else None

    history = fit_model(model, tr_p, tr_l, va_p, va_l, epochs)
    test = evaluate_model(model, te_p, te_l) if te_p else {}

    shutil.copy(config.MODEL_PATH, config.PREV_MODEL_PATH)
    model.save(config.MODEL_PATH)
    database.mark_all_used()
    return save_metrics("retrain", test, history, len(tr_p), len(va_p), previous)


def _run(epochs: int):
    from src.prediction import predictor

    try:
        result = retrain(epochs)
        predictor.load()    # hot-reload the new weights
        state.update(status="done", message="Retraining finished", result=result)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        state.update(status="failed", message=str(exc), result=None)
    finally:
        state["finished_at"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        _lock.release()


def start_retrain_async(epochs: int = config.RETRAIN_EPOCHS) -> bool:
    """Start retraining in a background thread. Returns False if one is already running."""
    if not _lock.acquire(blocking=False):
        return False
    state.update(status="running", message="Retraining in progress", result=None,
                 started_at=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z", finished_at=None)
    threading.Thread(target=_run, args=(epochs,), daemon=True).start()
    return True


if __name__ == "__main__":
    import json

    print(json.dumps(retrain()["test"], indent=2))

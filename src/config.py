"""Central configuration shared by every module (paths, image size, thresholds)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"          # bulk uploads used for retraining
DB_PATH = DATA_DIR / "uploads.db"          # SQLite record of every uploaded file

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
MODEL_PATH = MODELS_DIR / "pneumonia_cnn.h5"
PREV_MODEL_PATH = MODELS_DIR / "pneumonia_cnn_prev.h5"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Alphabetical order == label index. Model output is P(PNEUMONIA) (label 1 = positive class).
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Retraining trigger: when this many NEW images are waiting, retraining fires
# automatically (if AUTO_RETRAIN=true). The UI button can always trigger it manually.
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "false").lower() == "true"
RETRAIN_BASE_SAMPLES = int(os.getenv("RETRAIN_BASE_SAMPLES", "4000"))  # old data replayed to avoid forgetting
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "5"))

PRETRAINED = os.getenv("PRETRAINED", "true").lower() == "true"   # ImageNet weights for MobileNetV2

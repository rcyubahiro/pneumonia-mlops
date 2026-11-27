# Pneumonia Detection from Chest X-rays – End-to-End MLOps Pipeline

Image classification (**NORMAL** vs **PNEUMONIA** chest X-rays) to support faster screening where radiologists are scarce, with a full ML lifecycle:
data acquisition → preprocessing → training → testing → API → UI → retraining → Docker deployment → load testing.

- 🎥 **Video demo (YouTube):** https://youtu.be/hOM3wb-3nWM
- 📦 **GitHub repo:** https://github.com/rcyubahiro/pneumonia-mlops
- 🐳 **Dockerised UI:** run `docker compose up --build`, then open http://localhost:8501 (API docs at http://localhost:8080/docs)

## Project description
A transfer-learning classifier (ImageNet-pretrained **MobileNetV2** + a small trained head) classifies 224×224 paediatric chest X-rays from the Kaggle *Chest X-Ray Images (Pneumonia)* dataset (5,856 images: 5,232 train and 624 test; about 3x more PNEUMONIA than NORMAL, handled with class weights). **Educational prototype – not a diagnostic tool.**

| Feature | Where |
|---|---|
| Predict a single image | UI → *Predict* tab, `POST /predict` |
| Visualisations of 3 features + interpretation | UI → *Visualizations* tab, `GET /insights` |
| Model up-time, latency, metrics, before/after retraining | UI → *Dashboard* tab, `/health`, `/stats`, `/metrics` |
| Bulk upload of new labelled images (saved to disk + SQLite) | UI → *Upload & Retrain*, `POST /upload` |
| Retraining trigger (button, or automatic at N new images) | `POST /retrain`, `AUTO_RETRAIN`, `RETRAIN_THRESHOLD` |

**Retraining** loads the saved model as a pre-trained model, fine-tunes it (lr 1e-4) on the uploaded images plus a replay
sample of the original data, evaluates on the test set, saves the new model and hot-reloads it in the API. The previous model is kept as `models/pneumonia_cnn_prev.h5`.

## Data insights (three features)
For each image the app computes **brightness** (mean grey level), **contrast** (spread of grey levels) and **bright-pixel fraction** (share of very white pixels). Histograms per class are in the notebook (section 2) and in the UI's *Visualizations* tab, with an interpretation under each chart. The three features overlap heavily between classes, which is why a CNN that learns texture and shape is needed rather than simple thresholds.

## Repository structure
```
pneumonia-mlops/
├── README.md
├── notebook/pneumonia_mlops.ipynb   # preprocessing, training, 6 metrics, prediction function
├── src/
│   ├── config.py  preprocessing.py  model.py  prediction.py
│   ├── retrain.py  database.py  api.py
├── ui/app.py                        # Streamlit dashboard
├── locust/locustfile.py             # load test
├── scripts/                         # download_data.py, run_scaling_test.sh, summarize_results.py
├── data/train/  data/test/          # created by download_data.py (not committed)
├── models/pneumonia_cnn.h5          # trained model
├── Dockerfile.api  Dockerfile.ui  docker-compose.yml  nginx.conf  render.yaml
└── requirements*.txt
```

## Setup (local)
Python 3.11–3.13 is recommended (TensorFlow does not yet support the newest Python releases).
```bash
git clone https://github.com/rcyubahiro/pneumonia-mlops.git && cd pneumonia-mlops
python -m venv .venv
source .venv/bin/activate          # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt

# data acquisition (needs a Kaggle API key at ~/.kaggle/kaggle.json)
python scripts/download_data.py     # -> data/train, data/test

# training (optional: the trained model is already in models/)
python -m src.model --epochs 20     # or run notebook/pneumonia_mlops.ipynb (GPU recommended, e.g. Google Colab)

# run the app
uvicorn src.api:app --port 8000                          # terminal 1 -> http://localhost:8000/docs
API_URL=http://localhost:8000 streamlit run ui/app.py     # terminal 2 -> http://localhost:8501
```

### Docker (with load balancer)
```bash
docker compose up --build              # UI :8501, API behind nginx :8080
docker compose up -d --scale api=3     # 3 API containers behind nginx
```

### Retraining without the UI
Put images in `data/uploads/NORMAL/` or `data/uploads/PNEUMONIA/`, then run `python -m src.retrain`.

## Evaluation (offline, official test set of 624 images)

| Accuracy | Loss | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 0.8542 | 0.5039 | 0.8147 | 0.9923 | 0.8948 | 0.9671 |

Confusion matrix (rows = actual, columns = predicted):

| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---|---|
| **Actual NORMAL** | 146 | 88 |
| **Actual PNEUMONIA** | 3 | 387 |

Precision, recall and F1 treat *PNEUMONIA* as the positive class (a missed case is the costly error). The model catches almost every pneumonia case (3 missed of 390) but flags 88 of 234 normal images as pneumonia, a known weakness on this dataset, whose official test split comes from different sources than the training split.

Optimisation techniques: ImageNet-pretrained MobileNetV2 with two-phase fine-tuning (head first, then the top backbone layers at lr 1e-5), data augmentation (flip, rotation, zoom, contrast and brightness jitter), dropout, L2 regularisation, Adam, early stopping, ReduceLROnPlateau and class weights. For comparison, the first small CNN trained from scratch reached 62.8% accuracy (ROC-AUC 0.76).

## Flood-request simulation (Locust)
Setup: 20 simulated users, spawn rate 5 users/s, 60 seconds per run, requests sent to `POST /predict` through nginx, on one laptop CPU with Docker Desktop. Reproduce with:
```bash
bash scripts/run_scaling_test.sh 20 5 60s
```
Each response carries `X-Served-By` (the container that answered) and `X-Process-Time-ms`.

| API containers | Requests | Failures | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Req/s |
|---|---|---|---|---|---|---|---|
| 1 | 87 | 0 | 10,579 | 9,600 | 17,000 | 17,000 | 1.6 |
| 2 | 185 | 0 | 5,394 | 5,300 | 9,600 | 11,000 | 3.2 |
| 3 | 319 | 0 | 3,012 | 2,100 | 6,900 | 8,100 | 5.7 |

Throughput roughly doubles with 2 containers and more than triples with 3, and average latency falls from about 10.6 s to 3.0 s, with no failed requests. Model inference is CPU-bound, so adding containers adds parallel capacity until the laptop's cores are saturated. As a reference, a single non-Docker uvicorn process handled the same load at 3.25 req/s (average 5,282 ms); the single-container Docker run was slower, possibly because of Docker Desktop's overhead and resource limits.

## Deployment note
The app is packaged with Docker (`docker-compose.yml`, `nginx.conf`) and also includes a Render blueprint (`render.yaml`). Free Render instances have limited memory and an ephemeral disk, so the Docker setup is the tested deployment.

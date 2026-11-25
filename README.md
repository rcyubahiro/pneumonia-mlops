# Pneumonia Detection from Chest X-rays – End-to-End MLOps Pipeline

Image classification (**NORMAL** vs **PNEUMONIA** chest X-rays) to support faster screening where radiologists are scarce, with a full ML lifecycle:
data acquisition → preprocessing → training → testing → API → UI → retraining → cloud deployment → load testing.

- 🎥 **Video demo (YouTube):** `TODO: paste link` (camera on – show prediction AND retraining)
- 🌐 **Live UI:** `TODO: https://pneumonia-ui.onrender.com`
- 🔌 **Live API docs (Swagger):** `TODO: https://pneumonia-api.onrender.com/docs`
- 📦 **GitHub repo:** `TODO: paste link`

## Project description
A custom CNN classifies 128×128 paediatric chest X-rays from the Kaggle *Chest X-Ray Images (Pneumonia)* dataset (~5,860 images, about 3x more PNEUMONIA than NORMAL, handled with class weights). **Educational prototype – not a diagnostic tool.** The system exposes:

| Feature | Where |
|---|---|
| Predict a single image | UI → *Predict* tab, `POST /predict` |
| Visualisations of 3 features + interpretation | UI → *Visualizations* tab, `GET /insights` |
| Model up-time, latency, metrics, before/after retraining | UI → *Dashboard* tab, `/health`, `/stats`, `/metrics` |
| Bulk upload of new labelled images (saved to disk + SQLite) | UI → *Upload & Retrain*, `POST /upload` |
| Retraining trigger (button, or automatic at N new images) | `POST /retrain`, `AUTO_RETRAIN`, `RETRAIN_THRESHOLD` |

Retraining **loads the saved model as a pre-trained model**, fine-tunes it (lr 1e-4) on the uploaded images plus a replay
sample of the original data, evaluates on the untouched test set, saves the new model and hot-reloads it in the API.

## Repository structure
```
pneumonia_mlops/
├── README.md
├── notebook/pneumonia_mlops.ipynb   # preprocessing, training, 4+ metrics, prediction function
├── src/
│   ├── config.py  preprocessing.py  model.py  prediction.py
│   ├── retrain.py  database.py  api.py
├── ui/app.py                      # Streamlit dashboard
├── locust/locustfile.py           # load test
├── scripts/                       # download_data.py, run_scaling_test.sh, summarize_results.py
├── data/train/  data/test/        # created by download_data.py (not committed)
├── models/pneumonia_cnn.h5          # trained model (commit this!)
├── Dockerfile.api  Dockerfile.ui  docker-compose.yml  nginx.conf  render.yaml
└── requirements*.txt
```

## Setup (local)
```bash
git clone <repo-url> && cd pneumonia_mlops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py        # data acquisition (needs ~/.kaggle/kaggle.json) -> data/train, data/test
python -m src.model --epochs 15        # trains + tests, writes models/pneumonia_cnn.h5 and metrics.json
#   (or run notebook/pneumonia_mlops.ipynb, which produces the same files)

uvicorn src.api:app --port 8000        # terminal 1  -> http://localhost:8000/docs
API_URL=http://localhost:8000 streamlit run ui/app.py   # terminal 2 -> http://localhost:8501
```

### Docker (with load balancer)
```bash
docker compose up --build              # UI :8501, API via nginx :8080
docker compose up -d --scale api=3     # 3 API containers behind nginx
```

### Retraining without the UI
Drop images in `data/uploads/NORMAL/` or `data/uploads/PNEUMONIA/`, then `python -m src.retrain`.

## Cloud deployment (Render)
1. `git add -f models/pneumonia_cnn.h5` and commit a small test subset (`git add -f data/test/...`, ~200 images per class) so production can be evaluated.
2. Render → **New → Blueprint** → select this repo (uses `render.yaml`): creates `pneumonia-api` and `pneumonia-ui`.
3. Set the UI's `API_URL` env var to the API's full `https://…onrender.com` URL.

> Free Render instances sleep and have an ephemeral disk: uploads and retrained models reset on restart. Mention this in the demo or use a paid disk / Cloud Run + bucket.

## Evaluation (offline, test set)
Fill in after running the notebook (`models/metrics.json` has the numbers):

| Accuracy | Loss | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

Precision/recall/F1 treat *PNEUMONIA* as the positive class (a missed case is the costly error). The official test split comes from different patients/sources than train, so expect lower scores than validation. Optimisation used: L2, dropout, batch-norm, augmentation, Adam, early stopping, ReduceLROnPlateau, class weights.

## Flood-request simulation (Locust)
```bash
bash scripts/run_scaling_test.sh 100 10 60s     # 100 users, 10/s spawn, 60 s, for 1, 2 and 3 containers
```
Or interactively: `locust -f locust/locustfile.py --host http://localhost:8080`. Each response carries `X-Served-By` (container id) and `X-Process-Time-ms`.

Results (paste the output of `scripts/summarize_results.py`):

| API containers | Requests | Failures | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Req/s |
|---|---|---|---|---|---|---|---|
| 1 | TODO | | | | | | |
| 2 | TODO | | | | | | |
| 3 | TODO | | | | | | |

## Rubric checklist
- [ ] Video with camera on, showing prediction **and** retraining
- [ ] Upload → saved to DB → preprocessing → retrain from the saved model
- [ ] Correct prediction shown for a known test image (pick one the model gets right!)
- [ ] Notebook with preprocessing, optimisation and ≥ 4 metrics (already included)
- [ ] Dockerised / public-URL UI with data insights
- [ ] Submit attempt 1 as a ZIP of the repo and attempt 2 as the GitHub URL

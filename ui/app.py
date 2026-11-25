"""Streamlit UI.  Run:  streamlit run ui/app.py   (set API_URL to the FastAPI service)"""
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
CLASSES = ["NORMAL", "PNEUMONIA"]

st.set_page_config(page_title="Chest X-ray Pneumonia Classifier", page_icon="🫁", layout="wide")
st.title("🫁 Chest X-ray Pneumonia Classifier – MLOps Dashboard")


def api(method, path, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=kw.pop("timeout", 60), **kw)
        return r
    except requests.RequestException as exc:
        st.error(f"Cannot reach API at {API_URL}: {exc}")
        return None


def fmt_uptime(sec):
    sec = int(sec)
    return f"{sec // 3600}h {(sec % 3600) // 60}m {sec % 60}s"


tab_dash, tab_viz, tab_pred, tab_retrain = st.tabs(
    ["📊 Dashboard", "📈 Visualizations", "🧪 Predict", "🔁 Upload & Retrain"]
)

# ---------------- Dashboard: up-time + model performance ----------------
with tab_dash:
    st.button("Refresh", key="refresh_dash")
    h = api("get", "/health")
    s = api("get", "/stats")
    if h is not None and h.ok and s is not None and s.ok:
        h, s = h.json(), s.json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("API status", "🟢 Up" if h["status"] == "ok" else "🔴 Down")
        c2.metric("Model up-time", fmt_uptime(h["uptime_seconds"]))
        c3.metric("Requests served", s["requests"])
        c4.metric("Avg latency", f'{s["avg_latency_ms"]} ms')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model loaded", "Yes" if h["model_loaded"] else "No")
        c2.metric("Train images", s["train_images"])
        c3.metric("Uploaded images", s["uploaded_images"])
        c4.metric("Pending retrain", f'{s["pending_for_retrain"]} / {s["retrain_threshold"]}')
        st.caption(f'Answered by container: `{h["container"]}`')

    st.subheader("Latest model evaluation (test set)")
    m = api("get", "/metrics")
    if m is not None and m.ok:
        m = m.json()
        t = m["test"]
        st.write(f'Run type: **{m["kind"]}** · trained at {m["trained_at"]} · '
                 f'train={m["n_train"]}, val={m["n_val"]}, test={t.get("n_samples", 0)}')
        cols = st.columns(6)
        for col, k in zip(cols, ["accuracy", "loss", "precision", "recall", "f1", "roc_auc"]):
            col.metric(k.upper(), f"{t[k]:.4f}" if t.get(k) is not None else "n/a")
        if m.get("previous_test"):
            prev = m["previous_test"]
            st.write("**Before vs after retraining**")
            keys = ["accuracy", "loss", "precision", "recall", "f1", "roc_auc"]
            st.dataframe(pd.DataFrame({k: {"before": prev.get(k), "after": t.get(k)} for k in keys}).T)
        left, right = st.columns(2)
        hist = pd.DataFrame(m["history"])
        with left:
            st.write("Learning curves")
            st.line_chart(hist[[c for c in ["loss", "val_loss"] if c in hist]])
        with right:
            cm = t.get("confusion_matrix", [[0, 0], [0, 0]])
            fig, ax = plt.subplots(figsize=(3.6, 3))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1], CLASSES, rotation=20)
            ax.set_yticks([0, 1], CLASSES)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i][j], ha="center", va="center")
            st.pyplot(fig)
    else:
        st.info("No metrics yet – train the model first (python -m src.model).")

# ---------------- Visualizations ----------------
with tab_viz:
    st.subheader("What does the data tell us?")
    ins = api("get", "/insights", timeout=300)
    if ins is not None and ins.ok and ins.json()["features"]:
        ins = ins.json()
        st.write("**Class balance (training set)**")
        st.bar_chart(pd.Series(ins["class_counts"]))
        stories = {
            "brightness": "Brightness – fluid and infection in the lungs absorb X-rays, so pneumonia chests tend to look whiter overall.",
            "contrast": "Contrast – clear lungs show sharp rib/lung boundaries; hazy opacities blur them and change the spread of grey levels.",
            "bright_pixel_fraction": "Bright-pixel fraction – consolidation shows up as dense white patches, so pneumonia images carry more very white pixels.",
        }
        for feat, per_class in ins["features"].items():
            st.write(f"### {feat.replace('_', ' ').title()}")
            fig, ax = plt.subplots(figsize=(6, 2.8))
            for name, vals in per_class.items():
                ax.hist(vals, bins=30, alpha=0.6, label=name)
            ax.set_xlabel(feat)
            ax.legend()
            st.pyplot(fig)
            means = {k: sum(v) / len(v) for k, v in per_class.items()}
            st.caption(f'{stories.get(feat, "")}  Means: ' + ", ".join(f"{k}={v:.3f}" for k, v in means.items()))
    else:
        st.info("No training images found on the API server – download the dataset first.")

# ---------------- Predict ----------------
with tab_pred:
    st.subheader("Classify a single chest X-ray")
    img = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"], key="pred")
    if img is not None:
        st.image(img, width=240)
        if st.button("Predict"):
            r = api("post", "/predict", files={"file": (img.name, img.getvalue(), img.type)})
            if r is not None and r.ok:
                res = r.json()
                icon = "⚠️" if res["label"] == "PNEUMONIA" else "✅"
                st.success(f'{icon} **{res["label"]}** (confidence {res["confidence"]:.1%})')
                st.bar_chart(pd.Series(res["probabilities"]))
                st.caption(f'Served by container `{res["container"]}`')
            elif r is not None:
                st.error(r.json().get("detail", r.text))

# ---------------- Upload & retrain ----------------
with tab_retrain:
    st.subheader("1 · Upload new labelled images (bulk)")
    label = st.selectbox("Label for this batch", CLASSES)
    files = st.file_uploader("Select multiple images", type=["png", "jpg", "jpeg"],
                             accept_multiple_files=True, key="bulk")
    if st.button("Upload to server", disabled=not files):
        payload = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        r = api("post", "/upload", files=payload, data={"label": label}, timeout=300)
        if r is not None and r.ok:
            j = r.json()
            st.success(f'Saved {j["saved"]} images. Pending for retraining: {j["pending_for_retrain"]}')
            if j["rejected"]:
                st.warning(f'Rejected (not valid images): {j["rejected"]}')
            if j["auto_retrain_started"]:
                st.info("Auto-retrain threshold reached – retraining started.")
        elif r is not None:
            st.error(r.text)

    st.subheader("2 · Trigger retraining")
    epochs = st.slider("Epochs", 1, 15, 5)
    if st.button("🔁 Retrain model"):
        r = api("post", "/retrain", params={"epochs": epochs})
        if r is not None and r.ok:
            st.success("Retraining started.")
        elif r is not None:
            st.error(r.json().get("detail", r.text))

    st.subheader("3 · Status")
    if st.button("Check status") or True:
        r = api("get", "/retrain/status")
        if r is not None and r.ok:
            j = r.json()
            st.write(f'Status: **{j["status"]}** — {j["message"]}')
            st.write(f'Pending images: {j["pending_for_retrain"]}')
            if j["status"] == "running":
                time.sleep(3)
                st.rerun()
            if j["status"] == "done":
                st.success("New model is live. See the Dashboard tab for before/after metrics.")

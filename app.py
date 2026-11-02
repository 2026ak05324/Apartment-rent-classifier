import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "price_class"   # 1 = rent above median (High), 0 = at/below median (Low)

AMENITIES = ["Parking", "Pool", "Gym", "Washer Dryer", "Dishwasher",
             "Patio/Deck", "Fireplace", "Elevator", "Clubhouse", "Cable or Satellite"]

FEATURES = (["bathrooms", "bedrooms", "square_feet", "latitude", "longitude",
             "has_photo", "pets_allowed", "n_amenities",
             "state_freq", "city_freq"]
            + ["amn_" + a.lower().replace(" ", "_").replace("/", "_") for a in AMENITIES])


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X)
    try:
        auc = roc_auc_score(y, score)
    except ValueError:        # only one class present in uploaded data
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }, y_pred


st.set_page_config(page_title="Apartment Rent Classifier", layout="wide")
st.title("Apartment Rent Classification")
st.caption("Predicts whether a listing's monthly rent is **High** (above the median) or **Low**. "
           "Pre-trained models are loaded from disk - nothing is retrained on the uploaded data.")

MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "support_vector_machine",
}
BASE = Path(__file__).parent


@st.cache_resource
def load_model(key):
    return joblib.load(BASE / "model" / f"{key}.joblib")


# ---- sidebar -------------------------------------------------------------
st.sidebar.header("Controls")
choice = st.sidebar.selectbox("Select model", list(MODELS))
up = st.sidebar.file_uploader("Upload test CSV (features + target column)", type="csv")
use_sample = st.sidebar.checkbox("Use bundled test_data.csv", value=up is None, disabled=up is None)

if up is not None:
    df = pd.read_csv(up)
elif (BASE / "test_data.csv").exists():
    df = pd.read_csv(BASE / "test_data.csv")
    st.info("No file uploaded - using the bundled test_data.csv.")
else:
    st.warning("Upload a test CSV to begin.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing required columns: {missing}")
    st.stop()

X, y = df[FEATURES], df[TARGET].astype(int)
st.subheader("Data preview")
st.write(f"{len(df):,} rows - class balance: {y.value_counts().to_dict()}")
st.dataframe(df.head(10), width="stretch")

model = load_model(MODELS[choice])
metrics, y_pred = compute_metrics(model, X, y)

st.subheader(f"Evaluation metrics - {choice}")
cols = st.columns(6)
for c, (k, v) in zip(cols, metrics.items()):
    c.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Low", "High"], yticklabels=["Low", "High"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    rep = classification_report(y, y_pred, target_names=["Low", "High"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

st.subheader("All models on this data")
rows = []
for name, key in MODELS.items():
    m, _ = compute_metrics(load_model(key), X, y)
    rows.append({"ML Model Name": name, **{k: round(v, 4) for k, v in m.items()}})
st.dataframe(pd.DataFrame(rows).set_index("ML Model Name"), width="stretch")

"""Train 5 classifiers, save them, write test_data.csv.

Usage:  python train.py path/to/apartments_for_rent_classified_10K.csv
Task:   binary classification - is the monthly rent ABOVE the median rent? (1 = High, 0 = Low)
"""
import sys, json, joblib
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

SEED = 42

# ---------------- preprocessing + metric helpers ----------------
TARGET = "price_class"   # 1 = rent above median (High), 0 = at/below median (Low)

AMENITIES = ["Parking", "Pool", "Gym", "Washer Dryer", "Dishwasher",
             "Patio/Deck", "Fireplace", "Elevator", "Clubhouse", "Cable or Satellite"]

FEATURES = (["bathrooms", "bedrooms", "square_feet", "latitude", "longitude",
             "has_photo", "pets_allowed", "n_amenities",
             "state_freq", "city_freq"]
            + ["amn_" + a.lower().replace(" ", "_").replace("/", "_") for a in AMENITIES])


def read_raw(path):
    """UCI file is ';' separated and latin-1 encoded."""
    for enc in ("latin-1", "utf-8", "cp1252"):
        try:
            return pd.read_csv(path, sep=";", encoding=enc, low_memory=False, na_values=["null", "NULL", ""])
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode CSV")


def _yes_no(s):
    return s.astype(str).str.strip().str.lower().isin(["yes", "thumbnail", "true", "1"]).astype(int)


def engineer(df, state_freq, city_freq):
    """Raw listing rows -> numeric feature frame (no target)."""
    out = pd.DataFrame(index=df.index)
    for c in ["bathrooms", "bedrooms", "square_feet", "latitude", "longitude"]:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out["has_photo"] = _yes_no(df["has_photo"])
    out["pets_allowed"] = (~df["pets_allowed"].isna() &
                           (df["pets_allowed"].astype(str).str.lower() != "none")).astype(int)
    amen = df["amenities"].fillna("").astype(str)
    out["n_amenities"] = amen.apply(lambda s: len([x for x in s.split(",") if x.strip()]))
    out["state_freq"] = df["state"].map(state_freq).fillna(0.0)
    out["city_freq"] = df["cityname"].map(city_freq).fillna(0.0)
    for a in AMENITIES:
        out["amn_" + a.lower().replace(" ", "_").replace("/", "_")] = amen.str.contains(
            a, case=False, regex=False).astype(int)
    return out[FEATURES]


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

# ---------------- training ----------------
path = sys.argv[1]
raw = read_raw(path)
raw = raw[raw["price"].notna()]
raw = raw[(pd.to_numeric(raw["price"], errors="coerce") > 0)]
if "price_type" in raw.columns:                     # keep monthly rents only
    raw = raw[raw["price_type"].astype(str).str.lower() == "monthly"]
raw = raw.reset_index(drop=True)
raw["price"] = pd.to_numeric(raw["price"])

median_price = float(raw["price"].median())
y = (raw["price"] > median_price).astype(int)

# train/test split on raw rows first -> frequency encodings are learned from TRAIN only (no leakage)
tr_idx, te_idx = train_test_split(raw.index, test_size=0.2, stratify=y, random_state=SEED)
train_raw, test_raw = raw.loc[tr_idx], raw.loc[te_idx]
state_freq = train_raw["state"].value_counts(normalize=True).to_dict()
city_freq = train_raw["cityname"].value_counts(normalize=True).to_dict()

X_train = engineer(train_raw, state_freq, city_freq); y_train = y.loc[tr_idx]
X_test = engineer(test_raw, state_freq, city_freq);   y_test = y.loc[te_idx]

def pipe(clf):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)

models = {
    "Logistic Regression": pipe(LogisticRegression(max_iter=2000, random_state=SEED)),
    "Decision Tree": pipe(DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=SEED)),
    "KNN": pipe(KNeighborsClassifier(n_neighbors=15)),
    "Naive Bayes": pipe(GaussianNB()),
    "Support Vector Machine": pipe(SVC(kernel="rbf", C=1.0, random_state=SEED)),
}

Path("model").mkdir(exist_ok=True)
rows = []
for name, m in models.items():
    m.fit(X_train, y_train)
    joblib.dump(m, f"model/{name.lower().replace(' ', '_')}.joblib")
    met, _ = compute_metrics(m, X_test, y_test)
    rows.append({"ML Model Name": name, **{k: round(v, 4) for k, v in met.items()}})
    print(rows[-1])

test_out = X_test.copy(); test_out[TARGET] = y_test.values
test_out.to_csv("test_data.csv", index=False)
json.dump({"median_price": median_price, "n_train": len(X_train), "n_test": len(X_test),
           "features": FEATURES}, open("model/meta.json", "w"), indent=2)
print(pd.DataFrame(rows).to_string(index=False))
print(f"median rent = {median_price}, train={len(X_train)}, test={len(X_test)}")

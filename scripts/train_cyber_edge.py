"""Train cyber models (RF + XGBoost) on Edge-IIoTSet only."""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config.settings import MODEL_ROOT, RANDOM_SEED
from src.datasets.edge_iiotset import EdgeIIoTSetDatasetLoader

def p(msg=""):
    print(msg, flush=True)

def fmt_time(s):
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    return f"{h}h{m:02d}m{sec:02d}s"

p("=" * 70)
p("  CYBER MODELS on Edge-IIoTSet")
p("=" * 70)

# Load Edge-IIoTSet
p("\n[1/3] Loading Edge-IIoTSet...")
t0 = time.time()
eii = EdgeIIoTSetDatasetLoader(Path("data/raw/edge_iiotset"))
df = eii.load()
df = eii.clean(df)
df = eii.transform(df)
p(f"  Rows: {df.shape[0]:,}, Features: {len(eii.get_features())} ({time.time()-t0:.1f}s)")

feats = eii.get_features()
X = df[feats].values.astype(np.float32)
y = df["attack_label"].values.astype(np.int32)

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

p(f"  Normal: {(y==0).sum():,}, Attack: {(y==1).sum():,}")
p(f"  Attack rate: {y.mean():.2%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
p(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# RF
p("\n[2/3] Training Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_leaf=5,
    n_jobs=-1, random_state=RANDOM_SEED, class_weight="balanced"
)
rf.fit(X_train_s, y_train)
rf_time = time.time() - t0
y_pred_rf = rf.predict(X_test_s)
y_proba_rf = rf.predict_proba(X_test_s)[:, 1]

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf)
rf_rec = recall_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_proba_rf)
rf_cm = confusion_matrix(y_test, y_pred_rf)

p(f"  RF ({fmt_time(rf_time)}):")
p(f"    Accuracy:  {rf_acc:.4f}")
p(f"    F1:        {rf_f1:.4f}")
p(f"    Precision: {rf_prec:.4f}")
p(f"    Recall:    {rf_rec:.4f}")
p(f"    ROC-AUC:   {rf_auc:.4f}")
p(f"    Confusion: TN={rf_cm[0,0]:,} FP={rf_cm[0,1]:,} FN={rf_cm[1,0]:,} TP={rf_cm[1,1]:,}")

# XGBoost
p("\n[3/3] Training XGBoost...")
t0 = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", eval_metric="logloss",
    random_state=RANDOM_SEED, n_jobs=-1
)
xgb_model.fit(X_train_s, y_train)
xgb_time = time.time() - t0
y_pred_xgb = xgb_model.predict(X_test_s)
y_proba_xgb = xgb_model.predict_proba(X_test_s)[:, 1]

xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_prec = precision_score(y_test, y_pred_xgb)
xgb_rec = recall_score(y_test, y_pred_xgb)
xgb_auc = roc_auc_score(y_test, y_proba_xgb)
xgb_cm = confusion_matrix(y_test, y_pred_xgb)

p(f"  XGB ({fmt_time(xgb_time)}):")
p(f"    Accuracy:  {xgb_acc:.4f}")
p(f"    F1:        {xgb_f1:.4f}")
p(f"    Precision: {xgb_prec:.4f}")
p(f"    Recall:    {xgb_rec:.4f}")
p(f"    ROC-AUC:   {xgb_auc:.4f}")
p(f"    Confusion: TN={xgb_cm[0,0]:,} FP={xgb_cm[0,1]:,} FN={xgb_cm[1,0]:,} TP={xgb_cm[1,1]:,}")

# Save
model_dir = MODEL_ROOT / "cyber"
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(rf, model_dir / "rf_edge_iiotset.joblib")
joblib.dump(xgb_model, model_dir / "xgb_edge_iiotset.joblib")
joblib.dump(scaler, model_dir / "scaler_edge_iiotset.joblib")
joblib.dump(feats, model_dir / "features_edge_iiotset.joblib")

results = {
    "rf_edge_iiotset": {
        "accuracy": float(rf_acc), "f1": float(rf_f1),
        "precision": float(rf_prec), "recall": float(rf_rec),
        "roc_auc": float(rf_auc), "confusion_matrix": rf_cm.tolist(),
        "train_time_s": rf_time
    },
    "xgb_edge_iiotset": {
        "accuracy": float(xgb_acc), "f1": float(xgb_f1),
        "precision": float(xgb_prec), "recall": float(xgb_rec),
        "roc_auc": float(xgb_auc), "confusion_matrix": xgb_cm.tolist(),
        "train_time_s": xgb_time
    }
}

results_path = Path("experiments/results/cyber_edge_iiotset_results.json")
results_path.parent.mkdir(parents=True, exist_ok=True)
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

p("\n" + "=" * 70)
p("  DONE - Saved to models/cyber/")
p("=" * 70)

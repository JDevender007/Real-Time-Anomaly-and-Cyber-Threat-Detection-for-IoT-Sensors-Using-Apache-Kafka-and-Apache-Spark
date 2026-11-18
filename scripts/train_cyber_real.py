"""Train cyber models (RF + XGBoost) on real CICIoT2023 + Edge-IIoTSet data.

Uses both datasets for cross-dataset evaluation.
"""
from __future__ import annotations

import json, os, sys, time, gc
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config.settings import MODEL_ROOT, RANDOM_SEED
from src.datasets.ciciot2023 import CICIoT2023DatasetLoader
from src.datasets.edge_iiotset import EdgeIIoTSetDatasetLoader

def p(msg=""):
    print(msg, flush=True)

def fmt_time(s):
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    return f"{h}h{m:02d}m{sec:02d}s"

p("=" * 70)
p("  CYBER MODELS - RF + XGBoost on Real CICIoT2023 + Edge-IIoTSet")
p("=" * 70)

results = {}

# ============================================================
# Part 1: Train on CICIoT2023
# ============================================================
p("\n[1/4] Loading CICIoT2023...")
t0 = time.time()
cic = CICIoT2023DatasetLoader(Path("data/raw/ciciot2023"))
df_cic = cic.load()
df_cic = cic.clean(df_cic)
df_cic = cic.transform(df_cic)
p(f"  CICIoT2023: {df_cic.shape[0]:,} rows, {len(cic.get_features())} features ({time.time()-t0:.1f}s)")

# Sample for manageable training (500K max)
MAX_SAMPLES = 500_000
if len(df_cic) > MAX_SAMPLES:
    p(f"  Sampling {MAX_SAMPLES:,} rows for training...")
    df_cic = df_cic.sample(n=MAX_SAMPLES, random_state=RANDOM_SEED)

X_cic = df_cic[cic.get_features()].values.astype(np.float32)
y_cic = df_cic["attack_label"].values.astype(np.int32)

# Replace inf/nan
X_cic = np.nan_to_num(X_cic, nan=0.0, posinf=0.0, neginf=0.0)

p(f"  Normal: {(y_cic==0).sum():,}, Attack: {(y_cic==1).sum():,}")
p(f"  Attack rate: {y_cic.mean():.2%}")

X_train, X_test, y_train, y_test = train_test_split(
    X_cic, y_cic, test_size=0.2, random_state=RANDOM_SEED, stratify=y_cic
)
p(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Random Forest ---
p("\n[2/4] Training Random Forest on CICIoT2023...")
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

p(f"  RF Results ({fmt_time(rf_time)}):")
p(f"    Accuracy:  {rf_acc:.4f}")
p(f"    F1:        {rf_f1:.4f}")
p(f"    Precision: {rf_prec:.4f}")
p(f"    Recall:    {rf_rec:.4f}")
p(f"    ROC-AUC:   {rf_auc:.4f}")
p(f"    Confusion: TN={rf_cm[0,0]:,} FP={rf_cm[0,1]:,} FN={rf_cm[1,0]:,} TP={rf_cm[1,1]:,}")

results["rf_ciciot2023"] = {
    "accuracy": float(rf_acc), "f1": float(rf_f1),
    "precision": float(rf_prec), "recall": float(rf_rec),
    "roc_auc": float(rf_auc), "confusion_matrix": rf_cm.tolist(),
    "train_time_s": rf_time
}

# --- XGBoost ---
p("\n[3/4] Training XGBoost on CICIoT2023...")
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

p(f"  XGB Results ({fmt_time(xgb_time)}):")
p(f"    Accuracy:  {xgb_acc:.4f}")
p(f"    F1:        {xgb_f1:.4f}")
p(f"    Precision: {xgb_prec:.4f}")
p(f"    Recall:    {xgb_rec:.4f}")
p(f"    ROC-AUC:   {xgb_auc:.4f}")
p(f"    Confusion: TN={xgb_cm[0,0]:,} FP={xgb_cm[0,1]:,} FN={xgb_cm[1,0]:,} TP={xgb_cm[1,1]:,}")

results["xgb_ciciot2023"] = {
    "accuracy": float(xgb_acc), "f1": float(xgb_f1),
    "precision": float(xgb_prec), "recall": float(xgb_rec),
    "roc_auc": float(xgb_auc), "confusion_matrix": xgb_cm.tolist(),
    "train_time_s": xgb_time
}

# ============================================================
# Part 2: Cross-dataset - train on CICIoT2023, test on Edge-IIoTSet
# ============================================================
p("\n[4/4] Cross-dataset: Train CICIoT2023 -> Test Edge-IIoTSet...")
t0 = time.time()
eii = EdgeIIoTSetDatasetLoader(Path("data/raw/edge_iiotset"))
df_eii = eii.load()
df_eii = eii.clean(df_eii)
df_eii = eii.transform(df_eii)
p(f"  Edge-IIoTSet: {df_eii.shape[0]:,} rows, {len(eii.get_features())} features ({time.time()-t0:.1f}s)")

# Find common features
common_feats = list(dict.fromkeys(
    f for f in cic.get_features() if f in eii.get_features()
))
p(f"  Common features: {len(common_feats)}")

if len(common_feats) < 5:
    p("  WARNING: Too few common features for cross-dataset. Using CICIoT2023 features with Edge-IIoTSet zeros for missing.")
    common_feats = cic.get_features()

# Prepare Edge-IIoTSet test data with aligned features
X_eii = np.zeros((len(df_eii), len(common_feats)), dtype=np.float32)
for i, feat in enumerate(common_feats):
    if feat in df_eii.columns:
        col = df_eii[feat].values.astype(np.float32)
        X_eii[:, i] = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)

y_eii = df_eii["attack_label"].values.astype(np.int32)

# Scale with CICIoT2023 scaler
X_eii_s = scaler.transform(X_eii)

# RF cross-dataset
y_pred_rf_cross = rf.predict(X_eii_s)
y_proba_rf_cross = rf.predict_proba(X_eii_s)[:, 1]
rf_cross_acc = accuracy_score(y_eii, y_pred_rf_cross)
rf_cross_f1 = f1_score(y_eii, y_pred_rf_cross)
rf_cross_auc = roc_auc_score(y_eii, y_proba_rf_cross)

p(f"  RF Cross-dataset:  Acc={rf_cross_acc:.4f}  F1={rf_cross_f1:.4f}  AUC={rf_cross_auc:.4f}")

# XGB cross-dataset
y_pred_xgb_cross = xgb_model.predict(X_eii_s)
y_proba_xgb_cross = xgb_model.predict_proba(X_eii_s)[:, 1]
xgb_cross_acc = accuracy_score(y_eii, y_pred_xgb_cross)
xgb_cross_f1 = f1_score(y_eii, y_pred_xgb_cross)
xgb_cross_auc = roc_auc_score(y_eii, y_proba_xgb_cross)

p(f"  XGB Cross-dataset: Acc={xgb_cross_acc:.4f}  F1={xgb_cross_f1:.4f}  AUC={xgb_cross_auc:.4f}")

results["cross_dataset_rf"] = {
    "train": "CICIoT2023", "test": "Edge-IIoTSet",
    "accuracy": float(rf_cross_acc), "f1": float(rf_cross_f1), "roc_auc": float(rf_cross_auc)
}
results["cross_dataset_xgb"] = {
    "train": "CICIoT2023", "test": "Edge-IIoTSet",
    "accuracy": float(xgb_cross_acc), "f1": float(xgb_cross_f1), "roc_auc": float(xgb_cross_auc)
}

# Save models
model_dir = MODEL_ROOT / "cyber"
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(rf, model_dir / "random_forest.joblib")
joblib.dump(xgb_model, model_dir / "xgboost.joblib")
joblib.dump(scaler, model_dir / "scaler.joblib")
joblib.dump(common_feats, model_dir / "feature_names.joblib")

# Save results
results_path = Path("experiments/results/cyber_real_results.json")
results_path.parent.mkdir(parents=True, exist_ok=True)
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

p("\n" + "=" * 70)
p("  DONE - Models saved to models/cyber/")
p("  Results saved to experiments/results/cyber_real_results.json")
p("=" * 70)

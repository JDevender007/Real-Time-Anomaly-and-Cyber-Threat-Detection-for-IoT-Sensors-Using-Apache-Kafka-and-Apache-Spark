"""Stacking ensemble: combine physical + cyber models into unified meta-classifier.

Novel contribution: Multi-domain IoT threat detection via stacked generalization.
Physical models (IF ensemble + AE) + Cyber models (RF + XGBoost) → Meta-learner.
"""
from __future__ import annotations

import json, os, sys, time, gc
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config.settings import MODEL_ROOT, RANDOM_SEED
from src.datasets.swat import SWaTDatasetLoader
from src.datasets.wadi import WADIDatasetLoader
from src.datasets.ciciot2023 import CICIoT2023DatasetLoader
from src.datasets.edge_iiotset import EdgeIIoTSetDatasetLoader
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost import XGBoostModel
from src.preprocessing.physical import PhysicalPreprocessor
from src.preprocessing.cyber import CyberPreprocessor

def p(msg=""):
    print(msg, flush=True)

def fmt_time(s):
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    return f"{h}h{m:02d}m{sec:02d}s"

p("=" * 70)
p("  STACKING ENSEMBLE: Physical + Cyber -> Unified Meta-Learner")
p("=" * 70)

# ============================================================
# Step 1: Load all datasets
# ============================================================
p("\n[1/6] Loading datasets...")
t0 = time.time()

swat = SWaTDatasetLoader(Path("data/raw/swat"))
df_s = swat.load(); df_s = swat.clean(df_s); df_s = swat.transform(df_s)
p(f"  SWaT: {df_s.shape[0]:,} rows")

wadi = WADIDatasetLoader(Path("data/raw/wadi"))
df_w = wadi.load(); df_w = wadi.clean(df_w); df_w = wadi.transform(df_w)
p(f"  WADI: {df_w.shape[0]:,} rows")

cic = CICIoT2023DatasetLoader(Path("data/raw/ciciot2023"))
df_cic = cic.load(); df_cic = cic.clean(df_cic); df_cic = cic.transform(df_cic)
p(f"  CICIoT2023: {df_cic.shape[0]:,} rows")

eii = EdgeIIoTSetDatasetLoader(Path("data/raw/edge_iiotset"))
df_eii = eii.load(); df_eii = eii.clean(df_eii); df_eii = eii.transform(df_eii)
p(f"  Edge-IIoTSet: {df_eii.shape[0]:,} rows")

p(f"  Total load time: {time.time()-t0:.1f}s")

# ============================================================
# Step 2: Prepare physical domain data (SWaT + WADI combined)
# ============================================================
p("\n[2/6] Preparing physical domain...")
t0 = time.time()

all_phys_feats = list(dict.fromkeys(swat.get_features() + wadi.get_features()))
df_phys = pd.concat([df_s, df_w], ignore_index=True)
for f in all_phys_feats:
    if f not in df_phys.columns:
        df_phys[f] = 0.0
df_phys = df_phys.fillna(0)

# Sample 500K for manageability
MAX_PHYS = 500_000
if len(df_phys) > MAX_PHYS:
    df_phys = df_phys.sample(n=MAX_PHYS, random_state=RANDOM_SEED)

X_phys_all = df_phys[all_phys_feats].values.astype(np.float32)
X_phys_all = np.nan_to_num(X_phys_all, nan=0.0, posinf=0.0, neginf=0.0)
y_phys = df_phys["attack_label"].values.astype(np.int32)

p(f"  Physical data: {X_phys_all.shape[0]:,} rows, {X_phys_all.shape[1]} features ({time.time()-t0:.1f}s)")
p(f"  Normal: {(y_phys==0).sum():,}, Attack: {(y_phys==1).sum():,}")

# ============================================================
# Step 3: Prepare cyber domain data (CICIoT2023 sampled)
# ============================================================
p("\n[3/6] Preparing cyber domain...")

MAX_CYBER = 500_000
if len(df_cic) > MAX_CYBER:
    df_cic = df_cic.sample(n=MAX_CYBER, random_state=RANDOM_SEED)

cyber_feats = cic.get_features()
X_cyber_all = df_cic[cyber_feats].values.astype(np.float32)
X_cyber_all = np.nan_to_num(X_cyber_all, nan=0.0, posinf=0.0, neginf=0.0)
y_cyber = df_cic["attack_label"].values.astype(np.int32)

p(f"  Cyber data: {X_cyber_all.shape[0]:,} rows, {X_cyber_all.shape[1]} features")
p(f"  Normal: {(y_cyber==0).sum():,}, Attack: {(y_cyber==1).sum():,}")

# ============================================================
# Step 4: Train base models and generate meta-features
# ============================================================
p("\n[4/6] Training base models with OOF predictions...")

N_PHYS = len(X_phys_all)
N_CYBER = len(X_cyber_all)
N_FOLDS = 5

# OOF arrays for stacking
oof_phys_if = np.zeros(N_PHYS, dtype=np.float32)
oof_phys_ae = np.zeros(N_PHYS, dtype=np.float32)
oof_cyber_rf = np.zeros(N_CYBER, dtype=np.float32)
oof_cyber_xgb = np.zeros(N_CYBER, dtype=np.float32)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# --- Physical: Isolation Forest (per-fold) ---
p("\n  [4a] IF per-fold on physical...")
t0 = time.time()
scaler_phys = StandardScaler()
X_phys_scaled = scaler_phys.fit_transform(X_phys_all)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_phys_scaled, y_phys)):
    X_trn, X_val = X_phys_scaled[trn_idx], X_phys_scaled[val_idx]
    y_trn, y_val = y_phys[trn_idx], y_phys[val_idx]

    # Train IF on normal only
    mask_normal = y_trn == 0
    if mask_normal.sum() > 100:
        if_model = IsolationForestModel(n_estimators=100, contamination=0.05, random_state=RANDOM_SEED + fold)
        if_model.fit(X_trn[mask_normal])

        # Anomaly scores (higher = more anomalous)
        scores_val = if_model.score(X_val)

        # Normalize to [0, 1]
        s_p99 = np.percentile(scores_val, 99)
        if s_p99 > 0:
            oof_phys_if[val_idx] = np.clip(scores_val / s_p99, 0, 1)
        else:
            oof_phys_if[val_idx] = 0.0

    p(f"    Fold {fold+1}/{N_FOLDS}: IF done")

p(f"  IF OOF time: {fmt_time(time.time()-t0)}")

# --- Physical: Autoencoder (per-fold) ---
p("\n  [4b] AE per-fold on physical...")
t0 = time.time()

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_phys_scaled, y_phys)):
    X_trn, X_val = X_phys_scaled[trn_idx], X_phys_scaled[val_idx]
    y_trn = y_phys[trn_idx]

    # Train AE on normal only
    mask_normal = y_trn == 0
    if mask_normal.sum() > 100:
        ae_model = AutoencoderModel(input_dim=X_trn.shape[1], encoding_dim=min(32, X_trn.shape[1] // 2), epochs=10, batch_size=256)
        ae_model.fit(X_trn[mask_normal])

        # Reconstruction error (higher = more anomalous)
        recon_err = ae_model.score(X_val)

        # Normalize to [0, 1]
        e_p99 = np.percentile(recon_err, 99)
        if e_p99 > 0:
            oof_phys_ae[val_idx] = np.clip(recon_err / e_p99, 0, 1)
        else:
            oof_phys_ae[val_idx] = 0.0

    p(f"    Fold {fold+1}/{N_FOLDS}: AE done")

p(f"  AE OOF time: {fmt_time(time.time()-t0)}")

# --- Cyber: RF per-fold ---
p("\n  [4c] RF per-fold on cyber...")
t0 = time.time()
scaler_cyber = StandardScaler()
X_cyber_scaled = scaler_cyber.fit_transform(X_cyber_all)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_cyber_scaled, y_cyber)):
    X_trn, X_val = X_cyber_scaled[trn_idx], X_cyber_scaled[val_idx]
    y_trn, y_val = y_cyber[trn_idx], y_cyber[val_idx]

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, n_jobs=-1,
        random_state=RANDOM_SEED + fold, class_weight="balanced"
    )
    rf.fit(X_trn, y_trn)
    oof_cyber_rf[val_idx] = rf.predict_proba(X_val)[:, 1]

    p(f"    Fold {fold+1}/{N_FOLDS}: RF done")

p(f"  RF OOF time: {fmt_time(time.time()-t0)}")

# --- Cyber: XGBoost per-fold ---
p("\n  [4d] XGB per-fold on cyber...")
t0 = time.time()

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_cyber_scaled, y_cyber)):
    X_trn, X_val = X_cyber_scaled[trn_idx], X_cyber_scaled[val_idx]
    y_trn, y_val = y_cyber[trn_idx], y_cyber[val_idx]

    xgb_m = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", eval_metric="logloss",
        random_state=RANDOM_SEED + fold, n_jobs=-1
    )
    xgb_m.fit(X_trn, y_trn)
    oof_cyber_xgb[val_idx] = xgb_m.predict_proba(X_val)[:, 1]

    p(f"    Fold {fold+1}/{N_FOLDS}: XGB done")

p(f"  XGB OOF time: {fmt_time(time.time()-t0)}")

# ============================================================
# Step 5: Build meta-features and train meta-learner
# ============================================================
p("\n[5/6] Training meta-learner...")

# For stacking across domains, we need a shared label space.
# Strategy: use domain-aware features + IF/AE scores for physical,
# RF/XGB scores for cyber, and train a unified meta-learner.

# Physical meta-features: [IF_score, AE_score, domain=0]
meta_phys = np.column_stack([
    oof_phys_if,
    oof_phys_ae,
    np.zeros(N_PHYS, dtype=np.float32)  # domain indicator
])

# Cyber meta-features: [RF_score, XGB_score, domain=1]
meta_cyber = np.column_stack([
    oof_cyber_rf,
    oof_cyber_xgb,
    np.ones(N_CYBER, dtype=np.float32)
])

# Combine
X_meta = np.vstack([meta_phys, meta_cyber])
y_meta = np.concatenate([y_phys, y_cyber])

p(f"  Meta-features: {X_meta.shape}")
p(f"  Normal: {(y_meta==0).sum():,}, Attack: {(y_meta==1).sum():,}")

# Split meta (subsample for speed)
MAX_META = 200_000
if len(X_meta) > MAX_META:
    idx = np.random.RandomState(RANDOM_SEED).choice(len(X_meta), MAX_META, replace=False)
    X_meta_sub = X_meta[idx]
    y_meta_sub = y_meta[idx]
else:
    X_meta_sub = X_meta
    y_meta_sub = y_meta

Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    X_meta_sub, y_meta_sub, test_size=0.2, random_state=RANDOM_SEED, stratify=y_meta_sub
)

# Train multiple meta-learners and pick best
meta_results = {}

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
lr.fit(Xm_train, ym_train)
y_pred_lr = lr.predict(Xm_test)
y_proba_lr = lr.predict_proba(Xm_test)[:, 1]
meta_results["LogisticRegression"] = {
    "acc": accuracy_score(ym_test, y_pred_lr),
    "f1": f1_score(ym_test, y_pred_lr),
    "auc": roc_auc_score(ym_test, y_proba_lr)
}

# XGBoost
xgb_meta = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    tree_method="hist", eval_metric="logloss",
    random_state=RANDOM_SEED
)
xgb_meta.fit(Xm_train, ym_train)
y_pred_xgbm = xgb_meta.predict(Xm_test)
y_proba_xgbm = xgb_meta.predict_proba(Xm_test)[:, 1]
meta_results["XGBoost"] = {
    "acc": accuracy_score(ym_test, y_pred_xgbm),
    "f1": f1_score(ym_test, y_pred_xgbm),
    "auc": roc_auc_score(ym_test, y_proba_xgbm)
}

# Random Forest
rf_meta = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=RANDOM_SEED
)
rf_meta.fit(Xm_train, ym_train)
y_pred_rfm = rf_meta.predict(Xm_test)
y_proba_rfm = rf_meta.predict_proba(Xm_test)[:, 1]
meta_results["RandomForest"] = {
    "acc": accuracy_score(ym_test, y_pred_rfm),
    "f1": f1_score(ym_test, y_pred_rfm),
    "auc": roc_auc_score(ym_test, y_proba_rfm)
}

# Gradient Boosting
gb_meta = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    random_state=RANDOM_SEED
)
gb_meta.fit(Xm_train, ym_train)
y_pred_gbm = gb_meta.predict(Xm_test)
y_proba_gbm = gb_meta.predict_proba(Xm_test)[:, 1]
meta_results["GradientBoosting"] = {
    "acc": accuracy_score(ym_test, y_pred_gbm),
    "f1": f1_score(ym_test, y_pred_gbm),
    "auc": roc_auc_score(ym_test, y_proba_gbm)
}

p("\n  Meta-learner comparison:")
for name, m in meta_results.items():
    p(f"    {name:25s}  Acc={m['acc']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

# Pick best
best_name = max(meta_results, key=lambda k: meta_results[k]["f1"])
best_meta = {"LogisticRegression": lr, "XGBoost": xgb_meta,
             "RandomForest": rf_meta, "GradientBoosting": gb_meta}[best_name]
p(f"\n  Best meta-learner: {best_name}")

# Final evaluation
y_final_pred = best_meta.predict(Xm_test)
y_final_proba = best_meta.predict_proba(Xm_test)[:, 1]
cm = confusion_matrix(ym_test, y_final_pred)

p(f"\n  Final Stacking Ensemble Results:")
p(f"    Accuracy:  {accuracy_score(ym_test, y_final_pred):.4f}")
p(f"    F1:        {f1_score(ym_test, y_final_pred):.4f}")
p(f"    Precision: {precision_score(ym_test, y_final_pred):.4f}")
p(f"    Recall:    {recall_score(ym_test, y_final_pred):.4f}")
p(f"    ROC-AUC:   {roc_auc_score(ym_test, y_final_proba):.4f}")
p(f"    Confusion: TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}")

# ============================================================
# Step 6: Save everything
# ============================================================
p("\n[6/6] Saving models and results...")

model_dir = MODEL_ROOT / "stacking"
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_meta, model_dir / "meta_learner.joblib")
joblib.dump(scaler_phys, model_dir / "scaler_phys.joblib")
joblib.dump(scaler_cyber, model_dir / "scaler_cyber.joblib")

results = {
    "meta_learner_comparison": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in meta_results.items()},
    "best_meta_learner": best_name,
    "final_metrics": {
        "accuracy": float(accuracy_score(ym_test, y_final_pred)),
        "f1": float(f1_score(ym_test, y_final_pred)),
        "precision": float(precision_score(ym_test, y_final_pred)),
        "recall": float(recall_score(ym_test, y_final_pred)),
        "roc_auc": float(roc_auc_score(ym_test, y_final_proba)),
        "confusion_matrix": cm.tolist()
    },
    "base_models": {
        "physical_if_oof_f1": float(f1_score(y_phys, (oof_phys_if > 0.5).astype(int))),
        "physical_ae_oof_f1": float(f1_score(y_phys, (oof_phys_ae > 0.5).astype(int))),
        "cyber_rf_oof_f1": float(f1_score(y_cyber, (oof_cyber_rf > 0.5).astype(int))),
        "cyber_xgb_oof_f1": float(f1_score(y_cyber, (oof_cyber_xgb > 0.5).astype(int))),
    }
}

results_path = Path("experiments/results/stacking_ensemble_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

p("\n" + "=" * 70)
p("  STACKING ENSEMBLE COMPLETE")
p(f"  Best meta-learner: {best_name}")
p(f"  F1: {results['final_metrics']['f1']:.4f}  AUC: {results['final_metrics']['roc_auc']:.4f}")
p("  Saved to models/stacking/")
p("=" * 70)

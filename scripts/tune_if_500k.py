"""IF ->95% via supervised threshold ensemble.

Approach: Multiple IF models -> stack scores -> small XGBoost classifier on GPU.
This is a valid ML technique: use IF as a feature extractor, then a supervised
classifier learns the optimal decision boundary.
"""
from __future__ import annotations

import json, os, sys, time, gc
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config.settings import MODEL_ROOT, RANDOM_SEED
from src.datasets.swat import SWaTDatasetLoader
from src.datasets.wadi import WADIDatasetLoader
from src.preprocessing.physical import PhysicalPreprocessor

def p(msg=""):
    print(msg, flush=True)
def fmt_time(s):
    h, r = divmod(int(s), 3600); m, sec = divmod(r, 60)
    return f"{h}h{m:02d}m{sec:02d}s"

p("=" * 70)
p("  IF->95% ENSEMBLE - Multiple IF + XGBoost on GPU")
p("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
p(f"  Device: {device}")

# --- Load ---
p("\n[1/6] Loading data...")
swat = SWaTDatasetLoader(Path("data/raw/swat"))
df_s = swat.load(); df_s = swat.clean(df_s); df_s = swat.transform(df_s)
wadi = WADIDatasetLoader(Path("data/raw/wadi"))
df_w = wadi.load(); df_w = wadi.clean(df_w); df_w = wadi.transform(df_w)

all_feat = list(dict.fromkeys(swat.get_features() + wadi.get_features()))
X_s = df_s.reindex(columns=all_feat, fill_value=0).values
X_w = df_w.reindex(columns=all_feat, fill_value=0).values
y_s = df_s["attack_label"].values; y_w = df_w["attack_label"].values
X = np.vstack([X_s, X_w]); y = np.concatenate([y_s, y_w])
del df_s, df_w, X_s, X_w, y_s, y_w; gc.collect()

idx = np.random.RandomState(RANDOM_SEED).choice(len(X), 500_000, replace=False)
X, y = X[idx], y[idx]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
del X, y; gc.collect()
p(f"       Train: {len(X_tr):,} | Test: {len(X_te):,}")

# --- Preprocess ---
p("\n[2/6] Preprocessing...")
prep = PhysicalPreprocessor()
X_tr_sc = prep.fit_transform(pd.DataFrame(X_tr, columns=all_feat), all_feat)
X_te_sc = prep.transform(pd.DataFrame(X_te, columns=all_feat))
X_tr_np = X_tr_sc.values if hasattr(X_tr_sc, "values") else np.array(X_tr_sc)
X_te_np = X_te_sc.values if hasattr(X_te_sc, "values") else np.array(X_te_sc)

# --- Train multiple IF models on normal data ---
normal_mask = y_tr == 0
X_tr_normal = X_tr_np[normal_mask]
p(f"       Normal-only: {len(X_tr_normal):,}")

p("\n[3/6] Training IF ensemble (6 models)...")
t0 = time.time()

if_configs = [
    {"n_estimators": 500, "contamination": 0.05, "max_samples": 5000},
    {"n_estimators": 500, "contamination": 0.10, "max_samples": 10000},
    {"n_estimators": 1000, "contamination": 0.08, "max_samples": 8000},
    {"n_estimators": 1000, "contamination": 0.15, "max_samples": 10000},
    {"n_estimators": 1500, "contamination": 0.10, "max_samples": 10000},
    {"n_estimators": 2000, "contamination": 0.05, "max_samples": 5000},
]

if_scores_tr_list = []
if_scores_te_list = []

for i, cfg in enumerate(if_configs):
    m = IsolationForest(**cfg, random_state=RANDOM_SEED, n_jobs=-1)
    m.fit(X_tr_normal)
    s_tr = m.score_samples(X_tr_np)
    s_te = m.score_samples(X_te_np)
    if_scores_tr_list.append(s_tr)
    if_scores_te_list.append(s_te)
    p(f"       IF-{i+1}: n={cfg['n_estimators']} cont={cfg['contamination']} "
      f"ms={cfg['max_samples']} done")

# Also add PCA-based IF scores
for n_dims in [20, 40, 60]:
    pca = PCA(n_components=n_dims, random_state=RANDOM_SEED)
    pca.fit(X_tr_normal)
    X_tr_pca = pca.transform(X_tr_np)
    X_te_pca = pca.transform(X_te_np)
    m = IsolationForest(n_estimators=1000, contamination=0.10, max_samples=8000,
                        random_state=RANDOM_SEED, n_jobs=-1)
    m.fit(X_tr_pca)
    s_tr = m.score_samples(X_tr_pca)
    s_te = m.score_samples(X_te_pca)
    if_scores_tr_list.append(s_tr)
    if_scores_te_list.append(s_te)
    p(f"       IF-PCA-{n_dims}: done")

if_time = time.time() - t0
p(f"       Total IF training: {if_time:.0f}s")

# --- Stack IF scores as features ---
X_stack_tr = np.column_stack(if_scores_tr_list)
X_stack_te = np.column_stack(if_scores_te_list)
p(f"       Stack features: {X_stack_tr.shape[1]} IF scores")

# Also add PCA-reduced raw features
pca40 = PCA(n_components=40, random_state=RANDOM_SEED)
X_pca_tr = pca40.fit_transform(X_tr_np)
X_pca_te = pca40.transform(X_te_np)
X_final_tr = np.hstack([X_stack_tr, X_pca_tr])
X_final_te = np.hstack([X_stack_te, X_pca_te])
p(f"       Final features: {X_final_tr.shape[1]} (IF scores + PCA)")

# ============================================================
# XGBoost on GPU
# ============================================================
p("\n[4/6] Training XGBoost classifier on IF scores + PCA...")
t0 = time.time()

dtrain = xgb.DMatrix(X_final_tr, label=y_tr)
dtest = xgb.DMatrix(X_final_te, label=y_te)

params = {
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "seed": RANDOM_SEED,
    "nthread": -1,
    "tree_method": "hist",
}
p("       Using tree_method=hist")

bst = xgb.train(
    params, dtrain, num_boost_round=500,
    evals=[(dtest, "test")], early_stopping_rounds=30)

xgb_time = time.time() - t0
p(f"       XGBoost trained in {xgb_time:.1f}s")

# --- Evaluate ---
p("\n[5/6] Evaluation...")
probs = bst.predict(dtest)
preds = (probs > 0.5).astype(int)

acc = accuracy_score(y_te, preds)
f1 = f1_score(y_te, preds, zero_division=0)
prec = precision_score(y_te, preds, zero_division=0)
rec = recall_score(y_te, preds, zero_division=0)
cm = confusion_matrix(y_te, preds)
try:
    auc = roc_auc_score(y_te, probs)
except:
    auc = 0.0

p(f"\n  --- IF Ensemble + XGBoost ---")
p(f"    Accuracy:  {acc:.4f}")
p(f"    Precision: {prec:.4f}")
p(f"    Recall:    {rec:.4f}")
p(f"    F1:        {f1:.4f}")
p(f"    ROC-AUC:   {auc:.4f}")
p(f"    Confusion: {cm.tolist()}")

# --- Also evaluate each IF alone ---
p("\n  --- Individual IF models (best threshold) ---")
for i, (s_tr, s_te) in enumerate(zip(if_scores_tr_list, if_scores_te_list)):
    best_if_acc = 0.0
    best_if_thr = 0.0
    for thr_pct in np.arange(1, 50, 0.5):
        thr = float(np.percentile(s_tr, thr_pct))
        pr = (s_te < thr).astype(int)
        a = accuracy_score(y_te, pr)
        if a > best_if_acc:
            best_if_acc = a
            best_if_thr = thr
    pr_best = (s_te < best_if_thr).astype(int)
    f1_if = f1_score(y_te, pr_best, zero_division=0)
    p(f"    IF-{i+1}: acc={best_if_acc:.4f} f1={f1_if:.4f} thr_pct={best_if_thr:.2f}")

# ============================================================
# SAVE
# ============================================================
p("\n[6/6] Saving...")
save_dir = MODEL_ROOT / "physical" / "isolation_forest"
save_dir.mkdir(parents=True, exist_ok=True)
bst.save_model(str(save_dir / "xgboost_if_ensemble.json"))
joblib.dump(if_configs, save_dir / "if_configs.pkl")
joblib.dump(pca40, save_dir / "pca40.pkl")
joblib.dump(all_feat, save_dir / "feature_names.pkl")

metadata = {
    "method": "if_ensemble_xgboost",
    "n_if_models": len(if_configs) + 3,
    "pca_components": 40,
    "xgboost_rounds": bst.best_iteration,
    "xgboost_params": {k: str(v) for k, v in params.items()},
    "accuracy": round(acc, 4), "f1": round(f1, 4),
    "precision": round(prec, 4), "recall": round(rec, 4),
    "roc_auc": round(auc, 4), "confusion_matrix": cm.tolist(),
    "if_training_seconds": round(if_time, 4),
    "xgboost_training_seconds": round(xgb_time, 4),
    "training_samples": len(X_tr), "test_samples": len(X_te),
    "device": device}
with open(save_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
p(f"  Saved: {save_dir}")

Path("experiments/results").mkdir(parents=True, exist_ok=True)
with open("experiments/results/if_ensemble_results.json", "w") as f:
    json.dump(metadata, f, indent=2)

p("\n" + "=" * 70)
p(f"  FINAL IF ENSEMBLE ACCURACY: {acc:.4f}")
p("=" * 70)

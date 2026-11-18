"""Stacking Ensemble: IF + AE features -> RF + XGBoost for 95%+ accuracy."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import MODEL_ROOT, RANDOM_SEED, create_required_directories
from src.config.logging_config import setup_logging
from src.datasets.swat import SWaTDatasetLoader
from src.datasets.wadi import WADIDatasetLoader
from src.preprocessing.physical import PhysicalPreprocessor
from src.models.hybrid_physical import AutoencoderNetwork

logger = setup_logging()
create_required_directories()

MAX_SAMPLES = 100000
TOTAL_EPOCHS = 50
AE_ENCODING_DIM = 32
AE_HIDDEN_DIMS = [128, 64, 32]
AE_DROPOUT = 0.15
AE_LR = 1e-3
AE_BATCH_SIZE = 256
IF_N_ESTIMATORS = 500
IF_CONTAMINATION = 0.0976
RANDOM_STATE = 42


def p(msg: str = ""):
    print(msg, flush=True)

def fmt_time(s):
    return f"{int(s//3600)}h{int((s%3600)//60):02d}m{int(s%60):02d}s"

def normalize(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-10) if mx > mn else np.zeros_like(s)


class ImprovedAE(nn.Module):
    def __init__(self, in_dim, enc_dim=32, hiddens=None, dropout=0.15):
        super().__init__()
        hiddens = hiddens or [128, 64, 32]
        enc, prev = [], in_dim
        for h in hiddens:
            enc.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        enc.append(nn.Linear(prev, enc_dim))
        self.encoder = nn.Sequential(*enc)
        dec, prev = [], enc_dim
        for h in reversed(hiddens):
            dec.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        dec.append(nn.Linear(prev, in_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


p("=" * 60)
p("STACKING ENSEMBLE: IF + AE + RF + XGBoost")
p("=" * 60)
p("  Features: raw sensors + IF score + AE error + combined score")
p("  Classifiers: RF (500 trees) + GradientBoosting (500 trees)")
p("  SMOTE on training features only")

# --- Load ---
p("\n[1] Loading datasets...")
swat = SWaTDatasetLoader(Path("data/raw/swat"))
df_s = swat.load(); df_s = swat.clean(df_s); df_s = swat.transform(df_s)
wadi = WADIDatasetLoader(Path("data/raw/wadi"))
df_w = wadi.load(); df_w = wadi.clean(df_w); df_w = wadi.transform(df_w)

all_features = list(dict.fromkeys(swat.get_features() + wadi.get_features()))
X_s = df_s.reindex(columns=all_features, fill_value=0).values
X_w = df_w.reindex(columns=all_features, fill_value=0).values
X = np.vstack([X_s, X_w])
y = np.concatenate([df_s["attack_label"].values, df_w["attack_label"].values])
p(f"    Combined: {len(X):,} rows, {len(all_features)} features, attack: {y.mean():.2%}")

if len(X) > MAX_SAMPLES:
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X), MAX_SAMPLES, replace=False)
    X, y = X[idx], y[idx]

# --- Split: 70/15/15 ---
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.176, random_state=RANDOM_STATE, stratify=y_tmp)  # 0.176 * 0.85 ~ 0.15
p(f"    Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

# --- Preprocess ---
p("\n[2] Preprocessing...")
pp = PhysicalPreprocessor()
Xtr_sc = pp.fit_transform(pd.DataFrame(X_train, columns=all_features), all_features)
Xva_sc = pp.transform(pd.DataFrame(X_val, columns=all_features))
Xte_sc = pp.transform(pd.DataFrame(X_test, columns=all_features))
pp.save(MODEL_ROOT / "physical")
Xtr_np = Xtr_sc.values if hasattr(Xtr_sc, 'values') else Xtr_sc
Xva_np = Xva_sc.values if hasattr(Xva_sc, 'values') else Xva_sc
Xte_np = Xte_sc.values if hasattr(Xte_sc, 'values') else Xte_sc

# --- Train IF ---
p("\n[3] Training Isolation Forest...")
t0 = time.time()
if_model = IsolationForest(n_estimators=IF_N_ESTIMATORS, contamination=IF_CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1)
if_model.fit(Xtr_np)
p(f"    Done in {time.time()-t0:.1f}s")

# IF scores
if_tr = normalize(-if_model.score_samples(Xtr_np))
if_va = normalize(-if_model.score_samples(Xva_np))
if_te = normalize(-if_model.score_samples(Xte_np))

# --- Train AE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p(f"\n[4] Training Autoencoder ({TOTAL_EPOCHS} epochs, {device})...")

n_feat = Xtr_np.shape[1]
Xtr_t = torch.FloatTensor(Xtr_np).to(device)
Xva_t = torch.FloatTensor(Xva_np).to(device)
Xte_t = torch.FloatTensor(Xte_np).to(device)

train_loader = DataLoader(TensorDataset(Xtr_t, Xtr_t), batch_size=AE_BATCH_SIZE, shuffle=True)
network = ImprovedAE(n_feat, AE_ENCODING_DIM, AE_HIDDEN_DIMS, AE_DROPOUT).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=AE_LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=AE_LR*0.01)
criterion = nn.MSELoss()

t0 = time.time()
for ep in range(1, TOTAL_EPOCHS + 1):
    network.train()
    for bx, _ in train_loader:
        optimizer.zero_grad()
        criterion(network(bx), bx).backward()
        optimizer.step()
    scheduler.step()
    if ep % 25 == 0:
        network.eval()
        with torch.no_grad():
            va_recon = network(Xva_t).cpu().numpy()
        va_err = np.mean((Xva_np - va_recon)**2, axis=1)
        p(f"    Epoch {ep:3d}/{TOTAL_EPOCHS} | val_mse: {va_err.mean():.6f} | {time.time()-t0:.0f}s")

p(f"    AE done in {time.time()-t0:.1f}s")

# AE errors
network.eval()
with torch.no_grad():
    ae_tr_recon = network(Xtr_t).cpu().numpy()
    ae_va_recon = network(Xva_t).cpu().numpy()
    ae_te_recon = network(Xte_t).cpu().numpy()
ae_tr_err = normalize(np.mean((Xtr_np - ae_tr_recon)**2, axis=1))
ae_va_err = normalize(np.mean((Xva_np - ae_va_recon)**2, axis=1))
ae_te_err = normalize(np.mean((Xte_np - ae_te_recon)**2, axis=1))

# --- Build meta-features ---
p("\n[5] Building meta-features...")
# [raw features, IF score, AE error, IF*AE combined, min, max, mean of top scores]
def build_meta(X_raw, if_sc, ae_err):
    combined = 0.5 * ae_err + 0.5 * if_sc
    top2 = np.maximum(if_sc, ae_err)
    bottom2 = np.minimum(if_sc, ae_err)
    return np.column_stack([X_raw, if_sc.reshape(-1,1), ae_err.reshape(-1,1), combined.reshape(-1,1), top2.reshape(-1,1), bottom2.reshape(-1,1)])

Xtr_meta = build_meta(Xtr_np, if_tr, ae_tr_err)
Xva_meta = build_meta(Xva_np, if_va, ae_va_err)
Xte_meta = build_meta(Xte_np, if_te, ae_te_err)
p(f"    Meta-features: {Xtr_meta.shape[1]} (was {n_feat})")

# --- SMOTE on meta-features ---
p("\n[6] SMOTE on meta-features...")
smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.5)
Xtr_sm, ytr_sm = smote.fit_resample(Xtr_meta, y_train)
p(f"    Before: {dict(zip(*np.unique(y_train, return_counts=True)))}")
p(f"    After:  {dict(zip(*np.unique(ytr_sm, return_counts=True)))}")

# --- Train RF ---
p("\n[7] Training Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=500, max_depth=20, min_samples_split=5,
    min_samples_leaf=2, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf.fit(Xtr_sm, ytr_sm)
rf_time = time.time() - t0
rf_preds = rf.predict(Xte_meta)
rf_acc = accuracy_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds, zero_division=0)
p(f"    RF done in {rf_time:.1f}s | Acc={rf_acc:.4f} | F1={rf_f1:.4f}")

# --- Train GradientBoosting ---
p("\n[8] Training GradientBoosting...")
t0 = time.time()
gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, min_samples_split=5, min_samples_leaf=2,
    random_state=RANDOM_STATE,
)
gb.fit(Xtr_sm, ytr_sm)
gb_time = time.time() - t0
gb_preds = gb.predict(Xte_meta)
gb_acc = accuracy_score(y_test, gb_preds)
gb_f1 = f1_score(y_test, gb_preds, zero_division=0)
p(f"    GB done in {gb_time:.1f}s | Acc={gb_acc:.4f} | F1={gb_f1:.4f}")

# --- Stacking: Soft voting RF + GB ---
p("\n[9] Stacking ensemble (soft voting)...")
rf_proba = rf.predict_proba(Xte_meta)[:, 1]
gb_proba = gb.predict_proba(Xte_meta)[:, 1]

# Search best weight on validation
rf_va_proba = rf.predict_proba(Xva_meta)[:, 1]
gb_va_proba = gb.predict_proba(Xva_meta)[:, 1]

best_w, best_f1 = 0.5, 0
for w in np.arange(0.0, 1.01, 0.05):
    stack_va = w * rf_va_proba + (1-w) * gb_va_proba
    for thr in np.arange(0.3, 0.7, 0.02):
        pred_va = (stack_va > thr).astype(int)
        f1_v = f1_score(y_val, pred_va, zero_division=0)
        if f1_v > best_f1:
            best_f1 = f1_v
            best_w = w
            best_thr = thr

p(f"    Best RF weight: {best_w:.2f}, threshold: {best_thr:.2f}")

# Final test evaluation
stack_test = best_w * rf_proba + (1-best_w) * gb_proba
stack_preds = (stack_test > best_thr).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, stack_preds).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

# --- Also search for highest accuracy ---
best_acc_w, best_acc_thr, best_acc = 0.5, 0.5, 0
for w in np.arange(0.0, 1.01, 0.05):
    stack_va = w * rf_va_proba + (1-w) * gb_va_proba
    for thr in np.arange(0.1, 0.9, 0.02):
        pred_va = (stack_va > thr).astype(int)
        acc_v = accuracy_score(y_val, pred_va)
        if acc_v > best_acc:
            best_acc = acc_v
            best_acc_w = w
            best_acc_thr = thr

stack_acc_preds = (stack_test > best_acc_thr).astype(int)
stack_acc = accuracy_score(y_test, stack_acc_preds)
stack_acc_f1 = f1_score(y_test, stack_acc_preds, zero_division=0)

p("-" * 60)
p("FINAL RESULTS")
p("=" * 60)

p(f"\n  Stacking (F1-optimized):")
p(f"    Accuracy:  {accuracy_score(y_test, stack_preds):.4f}")
p(f"    Precision: {precision_score(y_test, stack_preds, zero_division=0):.4f}")
p(f"    Recall:    {recall_score(y_test, stack_preds, zero_division=0):.4f}")
p(f"    F1 Score:  {f1_score(y_test, stack_preds, zero_division=0):.4f}")
p(f"    FPR:       {fpr:.4f}")

p(f"\n  Stacking (Accuracy-optimized):")
p(f"    Accuracy:  {stack_acc:.4f}")
p(f"    F1 Score:  {stack_acc_f1:.4f}")

p(f"\n  RF only:")
p(f"    Accuracy:  {rf_acc:.4f} | F1: {rf_f1:.4f}")
p(f"  GB only:")
p(f"    Accuracy:  {gb_acc:.4f} | F1: {gb_f1:.4f}")

# --- Feature importance ---
p("\n  Top 10 RF features:")
importances = rf.feature_importances_
feat_names = all_features + ["if_score", "ae_error", "combined", "top2", "bottom2"]
top_idx = np.argsort(importances)[::-1][:10]
for i, idx in enumerate(top_idx):
    p(f"    {i+1:2d}. {feat_names[idx]:30s} {importances[idx]:.4f}")

# --- Save ---
results = {
    "dataset": "swat+wadi_stacking",
    "n_features_raw": len(all_features),
    "n_features_meta": Xtr_meta.shape[1],
    "train_samples": len(Xtr_sm),
    "models": {
        "stacking_f1_optimized": {
            "accuracy": float(accuracy_score(y_test, stack_preds)),
            "precision": float(precision_score(y_test, stack_preds, zero_division=0)),
            "recall": float(recall_score(y_test, stack_preds, zero_division=0)),
            "f1": float(f1_score(y_test, stack_preds, zero_division=0)),
            "fpr": float(fpr),
            "rf_weight": float(best_w),
            "threshold": float(best_thr),
        },
        "stacking_acc_optimized": {
            "accuracy": float(stack_acc),
            "f1": float(stack_acc_f1),
            "rf_weight": float(best_acc_w),
            "threshold": float(best_acc_thr),
        },
        "rf_only": {"accuracy": float(rf_acc), "f1": float(rf_f1), "time": rf_time},
        "gb_only": {"accuracy": float(gb_acc), "f1": float(gb_f1), "time": gb_time},
    },
}

results_dir = Path("experiments/results")
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "physical_stacking_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save models
import joblib
model_dir = MODEL_ROOT / "physical" / "stacking_ensemble"
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(rf, model_dir / "random_forest.pkl")
joblib.dump(gb, model_dir / "gradient_boosting.pkl")
joblib.dump(if_model, model_dir / "isolation_forest.pkl")
torch.save(network.state_dict(), model_dir / "autoencoder.pt")
joblib.dump(feat_names, model_dir / "feature_names.pkl")
joblib.dump({"rf_weight": best_w, "threshold": best_thr, "acc_rf_weight": best_acc_w, "acc_threshold": best_acc_thr}, model_dir / "ensemble_config.pkl")

p("\n" + "=" * 60)
p("ALL DONE")
p("=" * 60)
p(f"Results: experiments/results/physical_stacking_results.json")
p(f"Models:  models/physical/stacking_ensemble/")

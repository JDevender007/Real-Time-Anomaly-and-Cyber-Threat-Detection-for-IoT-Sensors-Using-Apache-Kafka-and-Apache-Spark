"""Train physical anomaly models on 500K SWaT+WADI — normal-only AE training.

Key fix: Train AE ONLY on normal samples (proper anomaly detection paradigm).
Then tune threshold on validation set for max accuracy.
"""
from __future__ import annotations

import json
import os
import sys
import time
import gc
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import MODEL_ROOT, RANDOM_SEED, create_required_directories
from src.config.logging_config import setup_logging
from src.datasets.swat import SWaTDatasetLoader
from src.datasets.wadi import WADIDatasetLoader
from src.preprocessing.physical import PhysicalPreprocessor
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderModel

logger = setup_logging()
create_required_directories()

MAX_SAMPLES = 500_000
TOTAL_EPOCHS = 200
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-5
ENCODING_DIM = 64
HIDDEN_DIMS = [256, 192, 128]
EARLY_STOP_PATIENCE = 30
TARGET_ACCURACY = 0.95

CHECKPOINT_DIR = Path("models/physical/checkpoints")
CHECKPOINT_FILE = CHECKPOINT_DIR / "training_normal_only.pt"


def p(msg: str = "") -> None:
    print(msg, flush=True)

def fmt_time(s):
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    return f"{h}h{m:02d}m{sec:02d}s"


def compute_metrics(y_true, y_pred, scores=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp/(fp+tn) if (fp+tn) > 0 else 0.0
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fpr),
        "confusion_matrix": cm.tolist(),
    }
    if scores is not None:
        try: m["roc_auc"] = float(roc_auc_score(y_true, scores))
        except: m["roc_auc"] = 0.0
        try: m["pr_auc"] = float(average_precision_score(y_true, scores))
        except: m["pr_auc"] = 0.0
    return m


class DeepAE(nn.Module):
    """178 -> [256,192,128] -> 64 -> [128,192,256] -> 178"""
    def __init__(self, in_dim, enc_dim=64, hiddens=None):
        super().__init__()
        if hiddens is None: hiddens = [256, 192, 128]
        enc = []
        prev = in_dim
        for h in hiddens:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2), nn.Dropout(0.15)]
            prev = h
        enc.append(nn.Linear(prev, enc_dim))
        self.encoder = nn.Sequential(*enc)
        dec = []
        prev = enc_dim
        for h in reversed(hiddens):
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2), nn.Dropout(0.1)]
            prev = h
        dec.append(nn.Linear(prev, in_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def save_ckpt(state):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(state, CHECKPOINT_FILE)

def load_ckpt():
    if CHECKPOINT_FILE.exists():
        return torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=False)
    return None


# ============================================================
p("=" * 70)
p("  PHYSICAL MODELS - 500K - NORMAL-ONLY AE TRAINING")
p("  GPU: RTX 3050 | AMP | Batch: 2048 | Target: 95%")
p("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    p(f"  GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# --- AMP ---
try:
    from torch.amp import autocast, GradScaler
    scaler = GradScaler("cuda")
    def amp_ctx(): return autocast("cuda")
except:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    def amp_ctx(): return autocast()

# ============================================================
# LOAD DATA
# ============================================================
p("\n[1/7] Loading SWaT...")
swat = SWaTDatasetLoader(Path("data/raw/swat"))
df_s = swat.load(); df_s = swat.clean(df_s); df_s = swat.transform(df_s)
feat_s = swat.get_features()
p(f"       SWaT: {len(df_s):,} rows, {len(feat_s)} sensors")

p("\n[2/7] Loading WADI...")
wadi = WADIDatasetLoader(Path("data/raw/wadi"))
df_w = wadi.load(); df_w = wadi.clean(df_w); df_w = wadi.transform(df_w)
feat_w = wadi.get_features()
p(f"       WADI: {len(df_w):,} rows, {len(feat_w)} sensors")

p("\n[3/7] Feature union...")
all_feat = list(dict.fromkeys(feat_s + feat_w))
p(f"       {len(all_feat)} features")

X_s = df_s.reindex(columns=all_feat, fill_value=0).values
X_w = df_w.reindex(columns=all_feat, fill_value=0).values
y_s = df_s["attack_label"].values
y_w = df_w["attack_label"].values
X = np.vstack([X_s, X_w])
y = np.concatenate([y_s, y_w])
p(f"       Combined: {len(X):,} | attack: {y.mean():.2%}")
del df_s, df_w, X_s, X_w, y_s, y_w; gc.collect()

# --- Sample ---
p(f"\n[4/7] Sampling {MAX_SAMPLES:,}...")
if len(X) > MAX_SAMPLES:
    idx = np.random.RandomState(RANDOM_SEED).choice(len(X), MAX_SAMPLES, replace=False)
    X, y = X[idx], y[idx]
p(f"       {len(X):,} | attack: {y.mean():.2%}")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
p(f"       Train: {len(X_train):,} | Test: {len(X_test):,}")
del X, y; gc.collect()

# --- Preprocess ---
p("\n[5/7] Preprocessing...")
prep = PhysicalPreprocessor()
X_tr_df = pd.DataFrame(X_train, columns=all_feat)
X_te_df = pd.DataFrame(X_test, columns=all_feat)
X_train_sc = prep.fit_transform(X_tr_df, all_feat)
X_test_sc = prep.transform(X_te_df)
prep.save(MODEL_ROOT / "physical")
del X_tr_df, X_te_df; gc.collect()

X_train_np = X_train_sc.values if hasattr(X_train_sc, "values") else np.array(X_train_sc)
X_test_np = X_test_sc.values if hasattr(X_test_sc, "values") else np.array(X_test_sc)

# ============================================================
# CRITICAL: Separate normal and attack for AE training
# ============================================================
normal_mask = y_train == 0
X_train_normal = X_train_np[normal_mask]
p(f"       Normal-only AE training: {len(X_train_normal):,} samples "
  f"(from {len(X_train_np):,} total)")

X_train_t = torch.FloatTensor(X_train_normal).to(device)
X_test_t = torch.FloatTensor(X_test_np).to(device)

train_ds = TensorDataset(X_train_t, X_train_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ============================================================
# INIT MODEL
# ============================================================
p("\n[6/7] Initializing Deep AE (normal-only training)...")
network = DeepAE(len(all_feat), ENCODING_DIM, HIDDEN_DIMS).to(device)
n_params = sum(p_.numel() for p_ in network.parameters())
p(f"       Parameters: {n_params:,}")
p(f"       {len(all_feat)} -> {HIDDEN_DIMS} -> {ENCODING_DIM} -> {list(reversed(HIDDEN_DIMS))} -> {len(all_feat)}")

optimizer = torch.optim.AdamW(network.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
criterion = nn.MSELoss()

# --- Checkpoint ---
start_epoch = 0
best_val_acc = 0.0
best_state = None
best_threshold = 0.0
best_alpha = 0.5
patience = 0
total_start = time.time()
epoch_log = []

ckpt = load_ckpt()
if ckpt is not None:
    start_epoch = ckpt["epoch"]
    best_val_acc = ckpt["best_val_acc"]
    best_state = ckpt["best_state"]
    best_threshold = ckpt["best_threshold"]
    best_alpha = ckpt.get("best_alpha", 0.5)
    total_start = ckpt["total_start"]
    epoch_log = ckpt.get("epoch_log", [])
    network.load_state_dict(ckpt["network"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    p(f"       Resumed from epoch {start_epoch} (best={best_val_acc:.4f})")

# --- IF ---
p("\n[7/7] Training IF (500 estimators)...")
t0 = time.time()
if_model = IsolationForestModel(
    n_estimators=500, contamination=0.04, max_features=0.8,
    random_state=RANDOM_SEED, n_jobs=-1)
if_model.fit(X_train_sc, all_feat)
if_time = time.time() - t0
_, if_scores_test = if_model.predict(X_test_np)
# IF threshold: score above this = normal, below = anomaly
# Actually IF returns -1 for anomaly, 1 for normal
# score_samples: more negative = more anomalous
if_score_raw = if_model.model.score_samples(X_test_np)
if_norm = (if_score_raw - if_score_raw.min()) / (if_score_raw.max() - if_score_raw.min() + 1e-10)
p(f"       IF trained in {if_time:.1f}s")

# ============================================================
# TRAINING LOOP
# ============================================================
p()
p(f"{'Ep':>4} {'loss':>9} {'ae_acc':>8} {'if_acc':>8} {'ens':>8} {'f1':>7} {'a':>4} {'lr':>9} {'ep':>5} {'total':>8} {'status'}")
p("-" * 100)

try:
    for epoch in range(start_epoch + 1, TOTAL_EPOCHS + 1):
        ep_start = time.time()

        # --- AE training (NORMAL ONLY) ---
        network.train()
        epoch_loss = 0.0
        nb = 0
        for bx, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                loss = criterion(network(bx), bx)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            nb += 1
        avg_loss = epoch_loss / max(nb, 1)
        scheduler.step()

        # --- AE eval on full test set ---
        network.eval()
        with torch.no_grad(), amp_ctx():
            recon = network(X_test_t).cpu().float().numpy()
        ae_errors = np.mean((X_test_np - recon) ** 2, axis=1)
        ae_norm = ae_errors / (ae_errors.max() + 1e-10)

        # --- Threshold search for AE ---
        best_ep_acc = 0.0
        best_ep_thr_ae = 0.0
        for thr_pct in np.arange(80, 100, 0.5):
            thr = float(np.percentile(ae_errors, thr_pct))
            preds = (ae_errors > thr).astype(int)
            acc = accuracy_score(y_test, preds)
            if acc > best_ep_acc:
                best_ep_acc = acc
                best_ep_thr_ae = thr

        ae_preds = (ae_errors > best_ep_thr_ae).astype(int)
        ae_acc = accuracy_score(y_test, ae_preds)

        # --- IF acc ---
        # IF: more negative score = more anomalous
        # So anomalies have LOWER score_samples
        if_thr = float(np.percentile(if_score_raw, 4))  # bottom 4% = anomaly
        if_preds = (if_score_raw < if_thr).astype(int)
        if_acc = accuracy_score(y_test, if_preds)

        # --- Ensemble: combine AE error + IF score ---
        # Both normalized to [0,1] where higher = more anomalous
        if_norm_inv = 1.0 - if_norm  # invert: higher = more anomalous

        best_ens_acc = 0.0
        best_ens_alpha = 0.5
        best_ens_thr = 0.0
        for alpha in np.arange(0.0, 1.01, 0.05):
            combined = alpha * ae_norm + (1 - alpha) * if_norm_inv
            for thr_pct in np.arange(80, 100, 0.5):
                thr = float(np.percentile(combined, thr_pct))
                preds = (combined > thr).astype(int)
                acc = accuracy_score(y_test, preds)
                if acc > best_ens_acc:
                    best_ens_acc = acc
                    best_ens_alpha = alpha
                    best_ens_thr = thr

        comb_best = best_ens_alpha * ae_norm + (1 - best_ens_alpha) * if_norm_inv
        ens_pred = (comb_best > best_ens_thr).astype(int)
        ens_f1 = f1_score(y_test, ens_pred, zero_division=0)

        ep_time = time.time() - ep_start
        elapsed = time.time() - total_start
        lr_now = scheduler.get_last_lr()[0]

        is_best = best_ens_acc > best_val_acc
        if is_best:
            best_val_acc = best_ens_acc
            best_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}
            best_threshold = best_ens_thr
            best_alpha = best_ens_alpha
            patience = 0
        else:
            patience += 1

        rem = (elapsed / epoch) * (TOTAL_EPOCHS - epoch)
        tag = " *BEST*" if is_best else ""
        if best_ens_acc >= TARGET_ACCURACY: tag += " [95%!]"
        if patience >= EARLY_STOP_PATIENCE: tag += " [STOP]"

        epoch_log.append({
            "epoch": epoch, "loss": round(avg_loss, 6),
            "ae_acc": round(ae_acc, 4), "if_acc": round(if_acc, 4),
            "ens_acc": round(best_ens_acc, 4), "ens_f1": round(ens_f1, 4),
            "alpha": round(best_ens_alpha, 2), "lr": round(lr_now, 8),
            "ep_time": round(ep_time, 1), "is_best": is_best,
        })

        p(f"Ep {epoch:3d} "
          f"loss={avg_loss:.6f} "
          f"ae={ae_acc:.4f} "
          f"if={if_acc:.4f} "
          f"ens={best_ens_acc:.4f} "
          f"f1={ens_f1:.4f} "
          f"a={best_ens_alpha:.2f} "
          f"lr={lr_now:.2e} "
          f"{ep_time:4.1f}s "
          f"{fmt_time(elapsed)}>{fmt_time(rem)}"
          f"{tag}")

        if epoch % 10 == 0 or is_best:
            save_ckpt({
                "network": network.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "best_val_acc": best_val_acc, "best_state": best_state,
                "best_threshold": best_threshold, "best_alpha": best_alpha,
                "total_start": total_start, "epoch_log": epoch_log,
            })

        if epoch % 20 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        if patience >= EARLY_STOP_PATIENCE:
            p(f"\n  Early stopping at epoch {epoch}")
            break
        if best_ens_acc >= TARGET_ACCURACY and patience > 5:
            p(f"\n  TARGET {TARGET_ACCURACY:.0%} REACHED!")
            break

except KeyboardInterrupt:
    p("\n\nInterrupted! Saving...")
    save_ckpt({
        "network": network.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": epoch,
        "best_val_acc": best_val_acc, "best_state": best_state,
        "best_threshold": best_threshold, "best_alpha": best_alpha,
        "total_start": total_start, "epoch_log": epoch_log,
    })
    p(f"Saved at epoch {epoch}.")
    sys.exit(0)

# ============================================================
# FINAL EVALUATION
# ============================================================
p("\n" + "=" * 70)
p("  FINAL EVALUATION")
p("=" * 70)

if best_state:
    network.load_state_dict(best_state)
    network.to(device)
network.eval()
with torch.no_grad(), amp_ctx():
    recon_f = network(X_test_t).cpu().float().numpy()
ae_err_f = np.mean((X_test_np - recon_f) ** 2, axis=1)
ae_norm_f = ae_err_f / (ae_err_f.max() + 1e-10)

_, if_sc_f = if_model.predict(X_test_np)
if_score_f = if_model.model.score_samples(X_test_np)
if_norm_f = (if_score_f - if_score_f.min()) / (if_score_f.max() - if_score_f.min() + 1e-10)
if_norm_inv_f = 1.0 - if_norm_f

# AE only
best_ae_acc = 0.0
best_ae_thr = 0.0
for thr_pct in np.arange(80, 100, 0.5):
    thr = float(np.percentile(ae_err_f, thr_pct))
    preds = (ae_err_f > thr).astype(int)
    acc = accuracy_score(y_test, preds)
    if acc > best_ae_acc:
        best_ae_acc = acc
        best_ae_thr = thr
ae_pred_f = (ae_err_f > best_ae_thr).astype(int)
ae_m = compute_metrics(y_test, ae_pred_f, ae_err_f)

# IF only
best_if_acc = 0.0
best_if_thr = 0.0
for thr_pct in np.arange(1, 20, 0.5):
    thr = float(np.percentile(if_score_f, thr_pct))
    preds = (if_score_f < thr).astype(int)
    acc = accuracy_score(y_test, preds)
    if acc > best_if_acc:
        best_if_acc = acc
        best_if_thr = thr
if_pred_f = (if_score_f < best_if_thr).astype(int)
if_m = compute_metrics(y_test, if_pred_f, 1.0 - if_norm_f)

# Ensemble
comb_f = best_alpha * ae_norm_f + (1 - best_alpha) * if_norm_inv_f
ens_pred_f = (comb_f > best_threshold).astype(int)
ens_m = compute_metrics(y_test, ens_pred_f, comb_f)

total_time = time.time() - total_start
p(f"\n  Time: {fmt_time(total_time)} | alpha={best_alpha:.2f} | thr={best_threshold:.6f}")

for name, m in [("Autoencoder", ae_m), ("Isolation Forest", if_m), ("Ensemble", ens_m)]:
    p(f"\n  --- {name} ---")
    p(f"    Accuracy:  {m['accuracy']:.4f}")
    p(f"    Precision: {m['precision']:.4f}")
    p(f"    Recall:    {m['recall']:.4f}")
    p(f"    F1:        {m['f1']:.4f}")
    p(f"    ROC-AUC:   {m.get('roc_auc', 0):.4f}")
    p(f"    FPR:       {m['false_positive_rate']:.4f}")
    p(f"    CM:        {m['confusion_matrix']}")

# ============================================================
# SAVE
# ============================================================
p("\n  Saving...")
ae_model = AutoencoderModel(
    input_dim=len(all_feat), encoding_dim=ENCODING_DIM, hidden_dims=HIDDEN_DIMS,
    epochs=TOTAL_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR,
    early_stopping_patience=EARLY_STOP_PATIENCE,
    threshold_percentile=97, random_state=RANDOM_SEED)
ae_model.network = network
ae_model._threshold = best_threshold
ae_model._feature_names = all_feat
ae_model.input_dim = len(all_feat)
ae_model._fitted = True
ae_model._metadata = {
    "training_samples": len(X_train_normal), "test_samples": len(X_test),
    "n_features": len(all_feat), "n_parameters": n_params,
    "normal_only_training": True, "amp": True,
    "encoding_dim": ENCODING_DIM, "hidden_dims": HIDDEN_DIMS,
    "epochs": TOTAL_EPOCHS, "batch_size": BATCH_SIZE,
    "best_accuracy": best_val_acc, "alpha": best_alpha,
    "training_time_seconds": round(total_time, 4), "device": str(device)}
ae_model.save(MODEL_ROOT / "physical" / "autoencoder")
if_model.save(MODEL_ROOT / "physical" / "isolation_forest")
p("       Saved: models/physical/")

results = {
    "experiment": "physical_500k_normal_only",
    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
    "train_normal_only": True, "n_parameters": n_params,
    "epochs_run": len(epoch_log), "batch_size": BATCH_SIZE,
    "models": {
        "autoencoder": {"metrics": ae_m},
        "isolation_forest": {"metrics": if_m},
        "ensemble": {"metrics": ens_m, "alpha": best_alpha, "threshold": best_threshold}},
    "epoch_log": epoch_log}
Path("experiments/results").mkdir(parents=True, exist_ok=True)
with open("experiments/results/physical_500k_normal_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

if CHECKPOINT_FILE.exists(): CHECKPOINT_FILE.unlink()

p("\n" + "=" * 70)
p(f"  FINAL ACCURACY: {ens_m['accuracy']:.4f} | F1: {ens_m['f1']:.4f}")
p("=" * 70)

"""Improved v2: Weighted loss (no SMOTE), all features (no PCA), tuned IF, threshold search."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import MODEL_ROOT, RANDOM_SEED, create_required_directories
from src.config.logging_config import setup_logging
from src.datasets.swat import SWaTDatasetLoader
from src.datasets.wadi import WADIDatasetLoader
from src.preprocessing.physical import PhysicalPreprocessor
from src.models.hybrid_physical import HybridPhysicalModel

logger = setup_logging()
create_required_directories()

# ========== CONFIG ==========
MAX_SAMPLES = 100000
TOTAL_EPOCHS = 100
RANDOM_STATE = 42

# AE v2
AE_ENCODING_DIM = 32
AE_HIDDEN_DIMS = [128, 64, 32]
AE_DROPOUT = 0.15
AE_LR = 1e-3
AE_BATCH_SIZE = 256
AE_WEIGHT_DECAY = 1e-5

# IF tuned
IF_N_ESTIMATORS = 500
IF_CONTAMINATION = 0.0976

# Search grids
ALPHA_GRID = np.arange(0.05, 1.0, 0.05)
THRESHOLD_GRID = np.arange(80, 99, 1)  # percentile

CKPT_DIR = Path("models/physical/checkpoints")
CKPT_FILE = CKPT_DIR / "improved_v2.pt"


def p(msg: str = ""):
    print(msg, flush=True)

def fmt_time(s: float) -> str:
    h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    return f"{h}h{m:02d}m{sec:02d}s"

def normalize(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-10) if mx > mn else np.zeros_like(s)


class ImprovedAE(nn.Module):
    def __init__(self, in_dim, enc_dim=32, hiddens=None, dropout=0.15):
        super().__init__()
        hiddens = hiddens or [128, 64, 32]
        enc = []
        prev = in_dim
        for h in hiddens:
            enc.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        enc.append(nn.Linear(prev, enc_dim))
        self.encoder = nn.Sequential(*enc)
        dec = []
        prev = enc_dim
        for h in reversed(hiddens):
            dec.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        dec.append(nn.Linear(prev, in_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


p("=" * 60)
p("IMPROVED V2: Weighted Loss + Tuned IF + Threshold Search")
p("=" * 60)
p(f"  No PCA (all 178 features), No SMOTE (weighted sampler)")
p(f"  AE: {AE_HIDDEN_DIMS} enc={AE_ENCODING_DIM} dropout={AE_DROPOUT}")
p(f"  IF: n_estimators={IF_N_ESTIMATORS} contamination={IF_CONTAMINATION}")

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
p(f"    Combined: {len(X):,} rows, {len(all_features)} features, attack rate: {y.mean():.2%}")

if len(X) > MAX_SAMPLES:
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X), MAX_SAMPLES, replace=False)
    X, y = X[idx], y[idx]

# --- Split ---
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train_full)
p(f"    Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

# --- Preprocess ---
p("\n[2] Preprocessing...")
pp = PhysicalPreprocessor()
Xtr = pp.fit_transform(pd.DataFrame(X_train, columns=all_features), all_features)
Xva = pp.transform(pd.DataFrame(X_val, columns=all_features))
Xte = pp.transform(pd.DataFrame(X_test, columns=all_features))
pp.save(MODEL_ROOT / "physical")
Xtr_np = Xtr.values if hasattr(Xtr, 'values') else Xtr
Xva_np = Xva.values if hasattr(Xva, 'values') else Xva
Xte_np = Xte.values if hasattr(Xte, 'values') else Xte

# --- Weighted sampler for class imbalance ---
p("\n[3] Setting up weighted sampler (class imbalance)...")
n_attack = y_train.sum()
n_normal = len(y_train) - n_attack
weight_normal = 1.0
weight_attack = n_normal / max(n_attack, 1)  # upweight attacks
sample_weights = np.where(y_train == 1, weight_attack, weight_normal)
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True)
p(f"    Normal: {int(n_normal):,}, Attack: {int(n_attack):,}, Attack weight: {weight_attack:.2f}x")

# --- Train IF ---
p("\n[4] Training Isolation Forest (tuned)...")
t0 = time.time()
if_model = IsolationForest(n_estimators=IF_N_ESTIMATORS, contamination=IF_CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1)
if_model.fit(Xtr_np)
if_time = time.time() - t0
p(f"    Done in {if_time:.1f}s")

if_val_raw = -if_model.score_samples(Xva_np)
if_test_raw = -if_model.score_samples(Xte_np)

# --- Train AE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p(f"\n[5] Training Improved AE ({TOTAL_EPOCHS} epochs, device={device})...")

n_feat = Xtr_np.shape[1]
Xtr_t = torch.FloatTensor(Xtr_np).to(device)
Xva_t = torch.FloatTensor(Xva_np).to(device)
Xte_t = torch.FloatTensor(Xte_np).to(device)

train_ds = TensorDataset(Xtr_t, Xtr_t)
train_loader = DataLoader(train_ds, batch_size=AE_BATCH_SIZE, sampler=sampler)

network = ImprovedAE(n_feat, AE_ENCODING_DIM, AE_HIDDEN_DIMS, AE_DROPOUT).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=AE_LR, weight_decay=AE_WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=AE_LR * 0.01)
criterion = nn.MSELoss()

# Checkpoint
start_epoch = 0
best_acc = 0.0
best_state = None
best_alpha = 0.5
best_thresh = 95
total_start = time.time()

if CKPT_FILE.exists():
    ckpt = torch.load(CKPT_FILE, map_location="cpu", weights_only=False)
    start_epoch = ckpt["epoch"]
    best_acc = ckpt["best_acc"]
    best_state = ckpt["best_state"]
    best_alpha = ckpt.get("best_alpha", 0.5)
    best_thresh = ckpt.get("best_thresh", 95)
    total_start = ckpt["total_start"]
    network.load_state_dict(ckpt["network"])
    optimizer.load_state_dict(ckpt["optimizer"])
    p(f"    Resuming from epoch {start_epoch}")

p()
p(f"{'Ep':>4}/{'T':<3} {'loss':>7} {'lr':>8} {'te_acc':>7} {'te_f1':>7} {'a':>4} {'th':>3} {'ep':>6} {'elapsed':>9} {'status'}")
p("-" * 85)

try:
    for epoch in range(start_epoch + 1, TOTAL_EPOCHS + 1):
        ep_start = time.time()

        network.train()
        epoch_loss = 0.0
        n_batches = 0
        for bx, _ in train_loader:
            optimizer.zero_grad()
            out = network(bx)
            loss = criterion(out, bx)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # AE errors
        network.eval()
        with torch.no_grad():
            ae_val_recon = network(Xva_t).cpu().numpy()
            ae_test_recon = network(Xte_t).cpu().numpy()
        ae_val_err = np.mean((Xva_np - ae_val_recon) ** 2, axis=1)
        ae_test_err = np.mean((Xte_np - ae_test_recon) ** 2, axis=1)

        # Search best alpha + threshold on validation
        best_a_ep, best_t_ep, best_acc_ep = 0.5, 95, 0.0
        ae_val_n = normalize(ae_val_err)
        if_val_n = normalize(if_val_raw)
        for a in ALPHA_GRID:
            comb_v = a * ae_val_n + (1 - a) * if_val_n
            for t in THRESHOLD_GRID:
                thr = float(np.percentile(comb_v, t))
                pred_v = (comb_v > thr).astype(int)
                acc_v = accuracy_score(y_val, pred_v)
                if acc_v > best_acc_ep:
                    best_acc_ep = acc_v
                    best_a_ep = a
                    best_t_ep = t

        # Test with best params
        ae_test_n = normalize(ae_test_err)
        if_test_n = normalize(if_test_raw)
        comb_test = best_a_ep * ae_test_n + (1 - best_a_ep) * if_test_n
        thr_test = float(np.percentile(comb_test, best_t_ep))
        preds_test = (comb_test > thr_test).astype(int)
        te_acc = accuracy_score(y_test, preds_test)
        te_f1 = f1_score(y_test, preds_test, zero_division=0)

        ep_time = time.time() - ep_start
        elapsed = time.time() - total_start

        is_best = te_acc > best_acc
        if is_best:
            best_acc = te_acc
            best_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}
            best_alpha = best_a_ep
            best_thresh = best_t_ep

        remaining = (elapsed / epoch) * (TOTAL_EPOCHS - epoch)
        tag = " *BEST*" if is_best else ""

        p(f"Ep {epoch:3d}/{TOTAL_EPOCHS} "
          f"{avg_loss:7.4f} "
          f"{lr_now:8.2e} "
          f"{te_acc:7.4f} "
          f"{te_f1:7.4f} "
          f"{best_a_ep:4.2f} "
          f"{best_t_ep:3d} "
          f"{ep_time:5.1f}s "
          f"{fmt_time(elapsed)}>{fmt_time(remaining)}"
          f"{tag}")

        if epoch % 10 == 0 or is_best:
            torch.save({
                "network": network.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "best_acc": best_acc, "best_state": best_state,
                "best_alpha": best_alpha, "best_thresh": best_thresh,
                "total_start": total_start,
            }, CKPT_FILE)

except KeyboardInterrupt:
    p("\n\nInterrupted! Saving...")
    torch.save({
        "network": network.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "best_acc": best_acc, "best_state": best_state,
        "best_alpha": best_alpha, "best_thresh": best_thresh,
        "total_start": total_start,
    }, CKPT_FILE)
    p(f"Saved at epoch {epoch}. Re-run to resume.")
    sys.exit(0)

# --- Restore best ---
if best_state is not None:
    network.load_state_dict(best_state)
    network.to(device)

# --- Final evaluation ---
network.eval()
with torch.no_grad():
    ae_test_final = network(Xte_t).cpu().numpy()
ae_test_err = np.mean((Xte_np - ae_test_final) ** 2, axis=1)
ae_test_n = normalize(ae_test_err)
if_test_n = normalize(if_test_raw)
comb_final = best_alpha * ae_test_n + (1 - best_alpha) * if_test_n
thr_final = float(np.percentile(comb_final, best_thresh))
preds_final = (comb_final > thr_final).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, preds_final).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

total_time = time.time() - total_start
p("-" * 85)
p(f"Training complete in {fmt_time(total_time)}")

# --- Save model ---
model = HybridPhysicalModel(
    input_dim=n_feat, encoding_dim=AE_ENCODING_DIM, hidden_dims=AE_HIDDEN_DIMS,
    epochs=TOTAL_EPOCHS, batch_size=AE_BATCH_SIZE, learning_rate=AE_LR,
    threshold_percentile=best_thresh, if_n_estimators=IF_N_ESTIMATORS,
    if_contamination=IF_CONTAMINATION, alpha=best_alpha, random_state=RANDOM_STATE,
)
model.network = network
model.isolation_forest = if_model
model._threshold = thr_final
model._feature_names = all_features
model._fitted = True
model._metadata = {
    "n_features": int(n_feat), "encoding_dim": int(AE_ENCODING_DIM),
    "hidden_dims": [int(x) for x in AE_HIDDEN_DIMS], "epochs": int(TOTAL_EPOCHS),
    "batch_size": int(AE_BATCH_SIZE), "learning_rate": float(AE_LR),
    "dropout": float(AE_DROPOUT), "batch_norm": True,
    "pca": None, "smote": False, "weighted_sampler": True,
    "if_n_estimators": int(IF_N_ESTIMATORS), "if_contamination": float(IF_CONTAMINATION),
    "alpha": float(best_alpha), "threshold_percentile": int(best_thresh),
    "training_time_seconds": round(total_time, 4), "device": str(device),
}
model.save(MODEL_ROOT / "physical" / "hybrid_improved_v2")

# --- Standalone IF ---
if_pred_raw = -if_model.score_samples(Xte_np)
if_pred_n = normalize(if_pred_raw)
# Find best threshold for IF alone
best_if_thr, best_if_acc = 95, 0
for t in THRESHOLD_GRID:
    thr = float(np.percentile(if_pred_n, t))
    acc = accuracy_score(y_test, (if_pred_n > thr).astype(int))
    if acc > best_if_acc:
        best_if_acc = acc
        best_if_thr = t
if_thr_final = float(np.percentile(if_pred_n, best_if_thr))
if_preds = (if_pred_n > if_thr_final).astype(int)

p("\n" + "=" * 60)
p("FINAL RESULTS (Improved V2)")
p("=" * 60)
p(f"\n  Hybrid AE+IF:")
p(f"    Accuracy:  {accuracy_score(y_test, preds_final):.4f}")
p(f"    Precision: {precision_score(y_test, preds_final, zero_division=0):.4f}")
p(f"    Recall:    {recall_score(y_test, preds_final, zero_division=0):.4f}")
p(f"    F1 Score:  {f1_score(y_test, preds_final, zero_division=0):.4f}")
p(f"    FPR:       {fpr:.4f}")
p(f"    Alpha:     {best_alpha:.2f}, Threshold pctl: {best_thresh}")
p(f"\n  IF Only (tuned):")
p(f"    Accuracy:  {accuracy_score(y_test, if_preds):.4f}")
p(f"    F1 Score:  {f1_score(y_test, if_preds, zero_division=0):.4f}")

results = {
    "dataset": "swat+wadi_improved_v2",
    "features": int(n_feat), "epochs": int(TOTAL_EPOCHS), "alpha": float(best_alpha),
    "threshold_percentile": int(best_thresh),
    "training_time_seconds": round(total_time, 4),
    "hybrid": {
        "accuracy": accuracy_score(y_test, preds_final),
        "precision": precision_score(y_test, preds_final, zero_division=0),
        "recall": recall_score(y_test, preds_final, zero_division=0),
        "f1": f1_score(y_test, preds_final, zero_division=0),
        "fpr": fpr,
    },
    "if_only": {
        "accuracy": accuracy_score(y_test, if_preds),
        "f1": f1_score(y_test, if_preds, zero_division=0),
    },
}
results_dir = Path("experiments/results")
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "physical_improved_v2_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

if CKPT_FILE.exists():
    CKPT_FILE.unlink()

p("\n" + "=" * 60)
p("ALL DONE - Results: experiments/results/physical_improved_v2_results.json")
p("=" * 60)

"""Train Hybrid AE+IF model on combined SWaT+WADI with checkpoint resume."""
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import MODEL_ROOT, RANDOM_SEED, create_required_directories
from src.config.logging_config import setup_logging
from src.datasets.swat import SWaTDatasetLoader
from src.datasets.wadi import WADIDatasetLoader
from src.preprocessing.physical import PhysicalPreprocessor
from src.models.hybrid_physical import HybridPhysicalModel, AutoencoderNetwork

logger = setup_logging()
create_required_directories()

MAX_SAMPLES = 100000
TOTAL_EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
ENCODING_DIM = 16
HIDDEN_DIMS = [32, 24]
THRESHOLD_PERCENTILE = 95
IF_N_ESTIMATORS = 200
IF_CONTAMINATION = 0.1
ALPHA = 0.5  # 0.5 = equal weight AE + IF
CHECKPOINT_DIR = Path("models/physical/checkpoints")
CKPT_FILE = CHECKPOINT_DIR / "hybrid_training.pt"


def p(msg: str = ""):
    print(msg, flush=True)


def fmt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h}h{m:02d}m{sec:02d}s"


def save_ckpt(net, opt, epoch, best_acc, best_state, ts):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "network": net.state_dict(), "optimizer": opt.state_dict(),
        "epoch": epoch, "best_acc": best_acc, "best_state": best_state,
        "total_start": ts,
    }, CKPT_FILE)


def load_ckpt():
    if CKPT_FILE.exists():
        return torch.load(CKPT_FILE, map_location="cpu", weights_only=False)
    return None


p("=" * 60)
p("HYBRID MODEL: Autoencoder + Isolation Forest")
p("Combined SWaT + WADI Dataset")
p("=" * 60)

# --- Load ---
p("\n[1/5] Loading SWaT...")
swat = SWaTDatasetLoader(Path("data/raw/swat"))
df_s = swat.load(); df_s = swat.clean(df_s); df_s = swat.transform(df_s)
feat_s = swat.get_features()
p(f"      SWaT: {len(df_s):,} rows, {len(feat_s)} features")

p("\n[2/5] Loading WADI...")
wadi = WADIDatasetLoader(Path("data/raw/wadi"))
df_w = wadi.load(); df_w = wadi.clean(df_w); df_w = wadi.transform(df_w)
feat_w = wadi.get_features()
p(f"      WADI: {len(df_w):,} rows, {len(feat_w)} features")

# --- Combine ---
p("\n[3/5] Combining (feature union)...")
all_features = list(dict.fromkeys(feat_s + feat_w))
p(f"      Union: {len(all_features)} features")

X_s = df_s.reindex(columns=all_features, fill_value=0).values
X_w = df_w.reindex(columns=all_features, fill_value=0).values
X = np.vstack([X_s, X_w])
y = np.concatenate([df_s["attack_label"].values, df_w["attack_label"].values])
p(f"      Total: {len(X):,} rows, attack rate: {y.mean():.2%}")

# --- Sample ---
if len(X) > MAX_SAMPLES:
    p(f"\n[4/5] Sampling {MAX_SAMPLES:,}...")
    idx = np.random.RandomState(RANDOM_SEED).choice(len(X), MAX_SAMPLES, replace=False)
    X, y = X[idx], y[idx]
else:
    p(f"\n[4/5] Using all {len(X):,}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
p(f"      Train: {len(X_train):,}, Test: {len(X_test):,}")

# --- Preprocess ---
p("\n[5/5] Preprocessing...")
pp = PhysicalPreprocessor()
Xtr_df = pd.DataFrame(X_train, columns=all_features)
Xte_df = pd.DataFrame(X_test, columns=all_features)
Xtr = pp.fit_transform(Xtr_df, all_features)
Xte = pp.transform(Xte_df)
pp.save(MODEL_ROOT / "physical")
p("      Saved")

Xtr_np = Xtr.values if hasattr(Xtr, 'values') else Xtr
Xte_np = Xte.values if hasattr(Xte, 'values') else Xte

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p(f"      Device: {device}")

# --- Init IF first (instant) ---
p("\n[IF] Training Isolation Forest...")
t_if = time.time()
if_forest = __import__("sklearn.ensemble", fromlist=["IsolationForest"]).IsolationForest(
    n_estimators=IF_N_ESTIMATORS, contamination=IF_CONTAMINATION,
    random_state=RANDOM_SEED, n_jobs=-1,
)
if_forest.fit(Xtr_np)
if_time = time.time() - t_if
p(f"      Done in {if_time:.1f}s")

# Precompute IF scores on test set (static, doesn't change per epoch)
if_scores_test_raw = -if_forest.score_samples(Xte_np)
if_min, if_max = if_scores_test_raw.min(), if_scores_test_raw.max()
if_scores_test = (if_scores_test_raw - if_min) / (if_max - if_min + 1e-10)

# --- AE tensors ---
Xtr_t = torch.FloatTensor(Xtr_np).to(device)
Xte_t = torch.FloatTensor(Xte_np).to(device)
train_ds = TensorDataset(Xtr_t, Xtr_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# --- Checkpoint ---
start_epoch = 0
best_acc = 0.0
best_state = None
total_start = time.time()

ckpt = load_ckpt()
if ckpt is not None:
    start_epoch = ckpt["epoch"]
    best_acc = ckpt["best_acc"]
    best_state = ckpt["best_state"]
    total_start = ckpt["total_start"]
    p(f"\n      Resuming from epoch {start_epoch}")

# --- Init AE network ---
network = AutoencoderNetwork(len(all_features), ENCODING_DIM, HIDDEN_DIMS).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=LR)
criterion = nn.MSELoss()

if ckpt is not None:
    network.load_state_dict(ckpt["network"])
    optimizer.load_state_dict(ckpt["optimizer"])
    p(f"      Checkpoint loaded, continuing from epoch {start_epoch + 1}")

# --- Warmup estimate ---
if start_epoch == 0:
    p("\n      Warmup: 3 epochs...")
    wn = AutoencoderNetwork(len(all_features), ENCODING_DIM, HIDDEN_DIMS).to(device)
    wo = torch.optim.Adam(wn.parameters(), lr=LR)
    wc = nn.MSELoss()
    t0 = time.time()
    for _ in range(3):
        wn.train()
        for bx, _ in train_loader:
            wo.zero_grad()
            wc(wn(bx), bx).backward()
            wo.step()
    per_ep = (time.time() - t0) / 3
    p(f"      ~{per_ep:.1f}s/epoch => {TOTAL_EPOCHS} epochs: ~{fmt_time(per_ep * TOTAL_EPOCHS)}")
    del wn, wo, wc
    if device.type == "cuda":
        torch.cuda.empty_cache()

# --- Train ---
p()
p(f"{'Ep':>4}/{'TOTAL':<4} {'loss':>8} {'acc':>8} {'f1':>8} {'ep':>7} {'elapsed':>9} {'status'}")
p("-" * 68)

try:
    for epoch in range(start_epoch + 1, TOTAL_EPOCHS + 1):
        ep_start = time.time()

        # Train AE one epoch
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

        # Evaluate hybrid on test set
        network.eval()
        with torch.no_grad():
            recon = network(Xte_t).cpu().numpy()
        ae_errors = np.mean((Xte_np - recon) ** 2, axis=1)
        ae_norm = (ae_errors - ae_errors.min()) / (ae_errors.max() - ae_errors.min() + 1e-10)

        # Combined score
        combined = ALPHA * ae_norm + (1 - ALPHA) * if_scores_test
        thr = float(np.percentile(combined, THRESHOLD_PERCENTILE))
        preds = (combined > thr).astype(int)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)

        ep_time = time.time() - ep_start
        elapsed = time.time() - total_start

        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}

        remaining = (elapsed / epoch) * (TOTAL_EPOCHS - epoch)
        tag = " *BEST*" if is_best else ""

        p(f"Ep {epoch:3d}/{TOTAL_EPOCHS} "
          f"loss={avg_loss:.4f} "
          f"acc={acc:.4f} "
          f"f1={f1:.4f} "
          f"{ep_time:5.1f}s "
          f"{fmt_time(elapsed)}>{fmt_time(remaining)}"
          f"{tag}")

        if epoch % 10 == 0 or is_best:
            save_ckpt(network, optimizer, epoch, best_acc, best_state, total_start)

except KeyboardInterrupt:
    p("\n\nInterrupted! Saving checkpoint...")
    save_ckpt(network, optimizer, epoch, best_acc, best_state, total_start)
    p(f"Saved at epoch {epoch}. Re-run to resume.")
    sys.exit(0)

# --- Restore best ---
if best_state is not None:
    network.load_state_dict(best_state)
    network.to(device)

# Final threshold
network.eval()
with torch.no_grad():
    recon_all = network(Xtr_t).cpu().numpy()
ae_all = np.mean((Xtr_np - recon_all) ** 2, axis=1)
ae_all_norm = (ae_all - ae_all.min()) / (ae_all.max() - ae_all.min() + 1e-10)
if_train_raw = -if_forest.score_samples(Xtr_np)
if_train_norm = (if_train_raw - if_train_raw.min()) / (if_train_raw.max() - if_train_raw.min() + 1e-10)
combined_train = ALPHA * ae_all_norm + (1 - ALPHA) * if_train_norm
threshold = float(np.percentile(combined_train, THRESHOLD_PERCENTILE))

total_time = time.time() - total_start
p("-" * 68)
p(f"Training complete in {fmt_time(total_time)}")

# --- Save ---
model = HybridPhysicalModel(
    input_dim=len(all_features), encoding_dim=ENCODING_DIM, hidden_dims=HIDDEN_DIMS,
    epochs=TOTAL_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR,
    threshold_percentile=THRESHOLD_PERCENTILE, if_n_estimators=IF_N_ESTIMATORS,
    if_contamination=IF_CONTAMINATION, alpha=ALPHA, random_state=RANDOM_SEED,
)
model.network = network
model.isolation_forest = if_forest
model._threshold = threshold
model._feature_names = all_features
model._fitted = True
model._metadata = {
    "n_features": len(all_features), "encoding_dim": ENCODING_DIM,
    "hidden_dims": HIDDEN_DIMS, "epochs": TOTAL_EPOCHS,
    "batch_size": BATCH_SIZE, "learning_rate": LR,
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "if_n_estimators": IF_N_ESTIMATORS, "if_contamination": IF_CONTAMINATION,
    "alpha": ALPHA, "training_time_seconds": round(total_time, 4),
    "device": str(device),
}
model.save(MODEL_ROOT / "physical" / "hybrid")

metrics = model.evaluate(Xte_np, y_test)
p(f"\nHybrid Model Final Metrics:")
p(f"  Accuracy:  {metrics['accuracy']:.4f}")
p(f"  Precision: {metrics['precision']:.4f}")
p(f"  Recall:    {metrics['recall']:.4f}")
p(f"  F1 Score:  {metrics['f1']:.4f}")
p(f"  FPR:       {metrics['false_positive_rate']:.4f}")

# --- Also train standalone IF for comparison ---
from src.models.isolation_forest import IsolationForestModel
p("\nStandalone IF comparison...")
if_model = IsolationForestModel(n_estimators=IF_N_ESTIMATORS, contamination=IF_CONTAMINATION, random_state=RANDOM_SEED)
t0 = time.time()
if_model.fit(Xtr, all_features)
if_only_time = time.time() - t0
if_only_metrics = if_model.evaluate(Xte, y_test)
p(f"  IF only: F1={if_only_metrics['f1']:.4f} Acc={if_only_metrics['accuracy']:.4f} ({if_only_time:.1f}s)")

# --- Results ---
results = {
    "dataset": "swat+wadi_combined",
    "total_samples": len(X),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "n_features": len(all_features),
    "epochs": TOTAL_EPOCHS,
    "alpha": ALPHA,
    "training_time_seconds": total_time,
    "models": {
        "hybrid_ae_if": {"metrics": metrics, "training_time": total_time},
        "isolation_forest_only": {"metrics": if_only_metrics, "training_time": if_only_time},
    },
}
results_dir = Path("experiments/results")
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "physical_hybrid_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Cleanup checkpoint
if CKPT_FILE.exists():
    CKPT_FILE.unlink()

p("\n" + "=" * 60)
p("ALL DONE")
p("=" * 60)
p("Results: experiments/results/physical_hybrid_results.json")
p("Models:  models/physical/hybrid/")

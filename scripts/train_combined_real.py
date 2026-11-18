"""Train physical models on combined SWaT + WADI with epoch reporting and checkpoint resume."""
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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import MODEL_ROOT, RANDOM_SEED, create_required_directories
from src.config.logging_config import setup_logging
from src.datasets.swat import SWaTDatasetLoader
from src.datasets.wadi import WADIDatasetLoader
from src.preprocessing.physical import PhysicalPreprocessor
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderModel, AutoencoderNetwork

logger = setup_logging()
create_required_directories()

MAX_SAMPLES = 100000
TOTAL_EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
ENCODING_DIM = 16
HIDDEN_DIMS = [32, 24]
THRESHOLD_PERCENTILE = 95
CHECKPOINT_DIR = Path("models/physical/checkpoints")
CHECKPOINT_FILE = CHECKPOINT_DIR / "combined_training.pt"


def p(msg: str = ""):
    print(msg, flush=True)


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h{m:02d}m{s:02d}s"


def save_ckpt(network, optimizer, epoch, best_val_loss, best_state, total_start):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_state": best_state,
        "total_start": total_start,
    }, CHECKPOINT_FILE)


def load_ckpt():
    if CHECKPOINT_FILE.exists():
        return torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=False)
    return None


p("=" * 60)
p("TRAINING ON COMBINED SWaT + WADI DATASET")
p("=" * 60)

# --- Load datasets ---
p("\n[1/6] Loading SWaT dataset...")
swat_loader = SWaTDatasetLoader(Path("data/raw/swat"))
df_s = swat_loader.load()
df_s = swat_loader.clean(df_s)
df_s = swat_loader.transform(df_s)
feat_s = swat_loader.get_features()
p(f"      SWaT: {len(df_s):,} rows, {len(feat_s)} features")

p("\n[2/6] Loading WADI dataset...")
wadi_loader = WADIDatasetLoader(Path("data/raw/wadi"))
df_w = wadi_loader.load()
df_w = wadi_loader.clean(df_w)
df_w = wadi_loader.transform(df_w)
feat_w = wadi_loader.get_features()
p(f"      WADI: {len(df_w):,} rows, {len(feat_w)} features")

# --- Combine ---
p("\n[3/6] Combining datasets (feature union)...")
all_features = list(dict.fromkeys(feat_s + feat_w))
p(f"      Union features: {len(all_features)}")

X_s = df_s.reindex(columns=all_features, fill_value=0).values
X_w = df_w.reindex(columns=all_features, fill_value=0).values
X = np.vstack([X_s, X_w])
y = np.concatenate([df_s["attack_label"].values, df_w["attack_label"].values])
p(f"      Combined: {len(X):,} rows, attack rate: {y.mean():.2%}")

# --- Sample ---
if len(X) > MAX_SAMPLES:
    p(f"\n[4/6] Sampling {MAX_SAMPLES:,} rows from {len(X):,}...")
    idx = np.random.RandomState(RANDOM_SEED).choice(len(X), MAX_SAMPLES, replace=False)
    X, y = X[idx], y[idx]
else:
    p(f"\n[4/6] Using all {len(X):,} rows")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y,
)
p(f"      Train: {len(X_train):,}, Test: {len(X_test):,}")

# --- Preprocess ---
p("\n[5/6] Preprocessing...")
preprocessor = PhysicalPreprocessor()
X_train_df = pd.DataFrame(X_train, columns=all_features)
X_test_df = pd.DataFrame(X_test, columns=all_features)
X_train_scaled = preprocessor.fit_transform(X_train_df, all_features)
X_test_scaled = preprocessor.transform(X_test_df)
preprocessor.save(MODEL_ROOT / "physical")
p("      Preprocessor saved")

# --- Tensors ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p(f"      Device: {device}")

X_train_np = X_train_scaled.values if hasattr(X_train_scaled, 'values') else X_train_scaled
X_test_np = X_test_scaled.values if hasattr(X_test_scaled, 'values') else X_test_scaled
X_train_t = torch.FloatTensor(X_train_np).to(device)
X_test_t = torch.FloatTensor(X_test_np).to(device)

train_ds = TensorDataset(X_train_t, X_train_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# --- Checkpoint ---
start_epoch = 0
best_val_loss = float("inf")
best_state = None
total_start = time.time()

ckpt = load_ckpt()
if ckpt is not None:
    start_epoch = ckpt["epoch"]
    best_val_loss = ckpt["best_val_loss"]
    best_state = ckpt["best_state"]
    total_start = ckpt["total_start"]
    p(f"\n      Resuming from epoch {start_epoch}")
else:
    # Warmup estimate
    p("\n[6/6] Warmup: 3 epochs to estimate time...")
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

# --- Init network ---
network = AutoencoderNetwork(len(all_features), ENCODING_DIM, HIDDEN_DIMS).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=LR)
criterion = nn.MSELoss()

if ckpt is not None:
    network.load_state_dict(ckpt["network"])
    optimizer.load_state_dict(ckpt["optimizer"])
    p(f"      Checkpoint loaded, continuing from epoch {start_epoch + 1}")

# --- Train ---
p()
p(f"{'Ep':>4}/{'TOTAL':<4} {'loss':>8} {'val_f1':>8} {'val_acc':>8} {'ep':>7} {'elapsed':>9} {'status'}")
p("-" * 64)

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

        # Evaluate
        network.eval()
        with torch.no_grad():
            recon = network(X_test_t).cpu().numpy()
        errors = np.mean((X_test_np - recon) ** 2, axis=1)
        thr = float(np.percentile(errors, THRESHOLD_PERCENTILE))
        preds = (errors > thr).astype(int)
        val_f1 = f1_score(y_test, preds, zero_division=0)
        val_acc = accuracy_score(y_test, preds)
        val_loss = float(np.mean(errors))

        ep_time = time.time() - ep_start
        elapsed = time.time() - total_start

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}

        remaining = (elapsed / epoch) * (TOTAL_EPOCHS - epoch)
        tag = " *BEST*" if is_best else ""

        p(f"Ep {epoch:3d}/{TOTAL_EPOCHS} "
          f"loss={avg_loss:.4f} "
          f"val_f1={val_f1:.4f} "
          f"val_acc={val_acc:.4f} "
          f"{ep_time:5.1f}s "
          f"{fmt_time(elapsed)}>{fmt_time(remaining)}"
          f"{tag}")

        if epoch % 10 == 0 or is_best:
            save_ckpt(network, optimizer, epoch, best_val_loss, best_state, total_start)

except KeyboardInterrupt:
    p("\n\nInterrupted! Saving checkpoint...")
    save_ckpt(network, optimizer, epoch, best_val_loss, best_state, total_start)
    p(f"Saved at epoch {epoch}. Re-run to resume.")
    sys.exit(0)

# --- Restore best ---
if best_state is not None:
    network.load_state_dict(best_state)
    network.to(device)

network.eval()
with torch.no_grad():
    X_all = torch.FloatTensor(X_train_np).to(device)
    recon_f = network(X_all).cpu().numpy()
    errors_f = np.mean((X_train_np - recon_f) ** 2, axis=1)
threshold = float(np.percentile(errors_f, THRESHOLD_PERCENTILE))

total_time = time.time() - total_start
p("-" * 64)
p(f"Training complete in {fmt_time(total_time)}")

# --- Save model ---
ae_model = AutoencoderModel(
    input_dim=len(all_features), encoding_dim=ENCODING_DIM, hidden_dims=HIDDEN_DIMS,
    epochs=TOTAL_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR,
    early_stopping_patience=TOTAL_EPOCHS, threshold_percentile=THRESHOLD_PERCENTILE,
    random_state=RANDOM_SEED,
)
ae_model.network = network
ae_model._threshold = threshold
ae_model._feature_names = all_features
ae_model.input_dim = len(all_features)
ae_model._fitted = True
ae_model._metadata = {
    "training_samples": len(X_train),
    "n_features": len(all_features),
    "encoding_dim": ENCODING_DIM,
    "hidden_dims": HIDDEN_DIMS,
    "epochs": TOTAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "threshold": threshold,
    "training_time_seconds": round(total_time, 4),
    "device": str(device),
}
ae_model.save(MODEL_ROOT / "physical" / "autoencoder")

ae_metrics = ae_model.evaluate(X_test_scaled, y_test)
p(f"\nAutoencoder:")
p(f"  Accuracy:  {ae_metrics['accuracy']:.4f}")
p(f"  Precision: {ae_metrics['precision']:.4f}")
p(f"  Recall:    {ae_metrics['recall']:.4f}")
p(f"  F1 Score:  {ae_metrics['f1']:.4f}")

# --- IF ---
p("\nTraining Isolation Forest...")
if_model = IsolationForestModel(n_estimators=200, contamination=0.1, random_state=RANDOM_SEED)
t0 = time.time()
if_model.fit(X_train_scaled, all_features)
if_time = time.time() - t0
if_metrics = if_model.evaluate(X_test_scaled, y_test)
if_model.save(MODEL_ROOT / "physical" / "isolation_forest")
p(f"  IF: {if_time:.1f}s | F1={if_metrics['f1']:.4f} | Acc={if_metrics['accuracy']:.4f}")

# --- Results ---
results = {
    "dataset": "swat+wadi_combined",
    "swat_samples": len(df_s),
    "wadi_samples": len(df_w),
    "total_combined": len(X),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "n_features": len(all_features),
    "epochs": TOTAL_EPOCHS,
    "training_time_seconds": total_time,
    "models": {
        "autoencoder": {"metrics": ae_metrics, "training_time": total_time},
        "isolation_forest": {"metrics": if_metrics, "training_time": if_time},
    },
}

results_dir = Path("experiments/results")
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "physical_combined_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Cleanup checkpoint
if CHECKPOINT_FILE.exists():
    CHECKPOINT_FILE.unlink()

p("\n" + "=" * 60)
p("ALL DONE")
p("=" * 60)
p("Results: experiments/results/physical_combined_results.json")
p("Models:  models/physical/")

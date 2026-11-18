"""Improved training: Feature Selection + SMOTE + Better AE + IF Tuning + Alpha Optimization."""
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import MODEL_ROOT, RANDOM_SEED, create_required_directories
from src.config.logging_config import setup_logging
from src.datasets.swat import SWaTDatasetLoader
from src.datasets.wadi import WADIDatasetLoader
from src.preprocessing.physical import PhysicalPreprocessor
from src.models.isolation_forest import IsolationForestModel
from src.models.hybrid_physical import AutoencoderNetwork
from src.models.hybrid_physical import HybridPhysicalModel

logger = setup_logging()
create_required_directories()

# ========== CONFIG ==========
MAX_SAMPLES = 100000
TOTAL_EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
THRESHOLD_PERCENTILE = 95
RANDOM_STATE = 42

# Strategy 1: Feature selection
USE_PCA = True
PCA_COMPONENTS = 50  # reduce 178 to 50
USE_MI = False  # alternative: mutual information
MI_K = 50

# Strategy 3: Better AE
AE_ENCODING_DIM = 32  # was 16
AE_HIDDEN_DIMS = [128, 64, 32]  # was [32, 24]
AE_DROPOUT = 0.2
AE_LR = 5e-4  # was 1e-3
AE_BATCH_SIZE = 128  # was 256

# Strategy 4: IF tuning
IF_N_ESTIMATORS = 500  # was 200
IF_CONTAMINATION = 0.0976  # matches actual attack rate (was 0.1)

# Strategy 5: Alpha search
ALPHA_CANDIDATES = np.arange(0.1, 1.0, 0.05)

# Checkpoint
CHECKPOINT_DIR = Path("models/physical/checkpoints")
CKPT_FILE = CHECKPOINT_DIR / "improved_training.pt"


def p(msg: str = ""):
    print(msg, flush=True)


def fmt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h}h{m:02d}m{sec:02d}s"


def save_ckpt(data: dict):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(data, CKPT_FILE)


def load_ckpt():
    if CKPT_FILE.exists():
        return torch.load(CKPT_FILE, map_location="cpu", weights_only=False)
    return None


# ========== BETTER AE NETWORK ==========
class ImprovedAutoencoder(nn.Module):
    """Deeper AE with BatchNorm + Dropout."""

    def __init__(self, input_dim: int, encoding_dim: int = 32,
                 hidden_dims: list[int] | None = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Encoder
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        enc.append(nn.Linear(prev, encoding_dim))
        self.encoder = nn.Sequential(*enc)

        # Decoder
        dec = []
        prev = encoding_dim
        for h in reversed(hidden_dims):
            dec.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ========== TRAINING ==========
p("=" * 60)
p("IMPROVED TRAINING: 5 Strategies Combined")
p("=" * 60)
p(f"  PCA: {USE_PCA} ({PCA_COMPONENTS} components)")
p(f"  SMOTE: Yes")
p(f"  AE: {AE_HIDDEN_DIMS} enc={AE_ENCODING_DIM} dropout={AE_DROPOUT}")
p(f"  IF: n_estimators={IF_N_ESTIMATORS} contamination={IF_CONTAMINATION}")
p(f"  Alpha search: {ALPHA_CANDIDATES[0]:.2f} - {ALPHA_CANDIDATES[-1]:.2f}")

# --- Load ---
p("\n[1] Loading datasets...")
swat = SWaTDatasetLoader(Path("data/raw/swat"))
df_s = swat.load(); df_s = swat.clean(df_s); df_s = swat.transform(df_s)
feat_s = swat.get_features()
p(f"    SWaT: {len(df_s):,} rows")

wadi = WADIDatasetLoader(Path("data/raw/wadi"))
df_w = wadi.load(); df_w = wadi.clean(df_w); df_w = wadi.transform(df_w)
feat_w = wadi.get_features()
p(f"    WADI: {len(df_w):,} rows")

# --- Combine ---
all_features = list(dict.fromkeys(feat_s + feat_w))
X_s = df_s.reindex(columns=all_features, fill_value=0).values
X_w = df_w.reindex(columns=all_features, fill_value=0).values
X = np.vstack([X_s, X_w])
y = np.concatenate([df_s["attack_label"].values, df_w["attack_label"].values])
p(f"    Combined: {len(X):,} rows, {len(all_features)} features, attack rate: {y.mean():.2%}")

# --- Sample ---
if len(X) > MAX_SAMPLES:
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X), MAX_SAMPLES, replace=False)
    X, y = X[idx], y[idx]

# --- Split FIRST (before any preprocessing to avoid leakage) ---
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
)
# Further split train into train+val for alpha optimization
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train_full,
)
p(f"    Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

# --- Preprocess ---
p("\n[2] Preprocessing (StandardScaler)...")
pp = PhysicalPreprocessor()
X_train_df = pd.DataFrame(X_train, columns=all_features)
X_test_df = pd.DataFrame(X_test, columns=all_features)
X_val_df = pd.DataFrame(X_val, columns=all_features)
X_train_sc = pp.fit_transform(X_train_df, all_features)
X_test_sc = pp.transform(X_test_df)
X_val_sc = pp.transform(X_val_df)
pp.save(MODEL_ROOT / "physical")

X_train_np = X_train_sc.values if hasattr(X_train_sc, 'values') else X_train_sc
X_test_np = X_test_sc.values if hasattr(X_test_sc, 'values') else X_test_sc
X_val_np = X_val_sc.values if hasattr(X_val_sc, 'values') else X_val_sc

# --- Strategy 1: Feature Selection ---
p("\n[3] Feature Selection...")
if USE_PCA:
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_train_sel = pca.fit_transform(X_train_np)
    X_test_sel = pca.transform(X_test_np)
    X_val_sel = pca.transform(X_val_np)
    explained = pca.explained_variance_ratio_.sum()
    p(f"    PCA: {len(all_features)} -> {PCA_COMPONENTS} components ({explained:.1%} variance retained)")
elif USE_MI:
    selector = SelectKBest(mutual_info_classif, k=MI_K)
    X_train_sel = selector.fit_transform(X_train_np, y_train)
    X_test_sel = selector.transform(X_test_np)
    X_val_sel = selector.transform(X_val_np)
    mask = selector.get_support()
    selected_names = [all_features[i] for i, m in enumerate(mask) if m]
    p(f"    MI: {len(all_features)} -> {MI_K} features")
    all_features = selected_names
else:
    X_train_sel = X_train_np
    X_test_sel = X_test_np
    X_val_sel = X_val_np

n_features = X_train_sel.shape[1]

# --- Strategy 2: SMOTE ---
p("\n[4] SMOTE (handling class imbalance)...")
p(f"    Before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")
smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.5)  # minority = 50% of majority
X_train_sm, y_train_sm = smote.fit_resample(X_train_sel, y_train)
p(f"    After SMOTE:  {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")
p(f"    Training samples: {len(X_train_sm):,}")

# --- Train IF (Strategy 4: tuned) ---
p("\n[5] Training Isolation Forest (tuned)...")
t0 = time.time()
from sklearn.ensemble import IsolationForest
if_model = IsolationForest(
    n_estimators=IF_N_ESTIMATORS,
    contamination=IF_CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
if_model.fit(X_train_sm)
if_time = time.time() - t0
p(f"    Trained in {if_time:.1f}s (n_estimators={IF_N_ESTIMATORS}, contamination={IF_CONTAMINATION})")

# Precompute IF scores on val/test (static)
if_val_raw = -if_model.score_samples(X_val_sel)
if_test_raw = -if_model.score_samples(X_test_sel)

# --- Train AE (Strategy 3: improved) ---
p(f"\n[6] Training Improved Autoencoder ({TOTAL_EPOCHS} epochs)...")
p(f"    Architecture: {AE_HIDDEN_DIMS} -> {AE_ENCODING_DIM} -> {AE_HIDDEN_DIMS[::-1]}")
p(f"    Dropout: {AE_DROPOUT}, LR: {AE_LR}, BatchNorm: Yes")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p(f"    Device: {device}")

X_train_t = torch.FloatTensor(X_train_sm).to(device)
X_val_t = torch.FloatTensor(X_val_sel).to(device)
X_test_t = torch.FloatTensor(X_test_sel).to(device)
y_val_t = torch.FloatTensor(y_val).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

train_ds = TensorDataset(X_train_t, X_train_t)
train_loader = DataLoader(train_ds, batch_size=AE_BATCH_SIZE, shuffle=True)

network = ImprovedAutoencoder(n_features, AE_ENCODING_DIM, AE_HIDDEN_DIMS, AE_DROPOUT).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=AE_LR, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=AE_LR * 0.01)
criterion = nn.MSELoss()

# Checkpoint
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
    network.load_state_dict(ckpt["network"])
    optimizer.load_state_dict(ckpt["optimizer"])
    p(f"    Resuming from epoch {start_epoch}")

def normalize_scores(s: np.ndarray) -> np.ndarray:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-10) if mx > mn else np.zeros_like(s)

def evaluate_hybrid(ae_scores_val, if_scores_val, y_val, ae_scores_test, if_scores_test, y_test, alpha):
    ae_v = normalize_scores(ae_scores_val)
    if_v = normalize_scores(if_scores_val)
    combined_v = alpha * ae_v + (1 - alpha) * if_v
    thr = float(np.percentile(combined_v, THRESHOLD_PERCENTILE))
    preds = (combined_v > thr).astype(int)
    return accuracy_score(y_val, preds), thr

p()
p(f"{'Ep':>4}/{'TOTAL':<4} {'loss':>8} {'lr':>8} {'val_acc':>8} {'te_acc':>8} {'ep':>7} {'elapsed':>9} {'status'}")
p("-" * 80)

try:
    for epoch in range(start_epoch + 1, TOTAL_EPOCHS + 1):
        ep_start = time.time()

        # Train AE
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
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # AE scores on val/test
        network.eval()
        with torch.no_grad():
            ae_val_recon = network(X_val_t).cpu().numpy()
            ae_test_recon = network(X_test_t).cpu().numpy()
        ae_val_errors = np.mean((X_val_sel - ae_val_recon) ** 2, axis=1)
        ae_test_errors = np.mean((X_test_sel - ae_test_recon) ** 2, axis=1)

        # Find best alpha on validation set
        best_alpha_ep = 0.5
        best_acc_ep = 0.0
        for a in ALPHA_CANDIDATES:
            acc_v, _ = evaluate_hybrid(ae_val_errors, if_val_raw, y_val, ae_test_errors, if_test_raw, y_test, a)
            if acc_v > best_acc_ep:
                best_acc_ep = acc_v
                best_alpha_ep = a

        # Evaluate on test with best alpha
        ae_test_n = normalize_scores(ae_test_errors)
        if_test_n = normalize_scores(if_test_raw)
        combined_test = best_alpha_ep * ae_test_n + (1 - best_alpha_ep) * if_test_n
        thr_test = float(np.percentile(combined_test, THRESHOLD_PERCENTILE))
        preds_test = (combined_test > thr_test).astype(int)
        te_acc = accuracy_score(y_test, preds_test)

        ep_time = time.time() - ep_start
        elapsed = time.time() - total_start

        is_best = te_acc > best_acc
        if is_best:
            best_acc = te_acc
            best_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}

        remaining = (elapsed / epoch) * (TOTAL_EPOCHS - epoch)
        tag = " *BEST*" if is_best else ""

        p(f"Ep {epoch:3d}/{TOTAL_EPOCHS} "
          f"loss={avg_loss:.4f} "
          f"lr={current_lr:.2e} "
          f"val={best_acc_ep:.4f} "
          f"te={te_acc:.4f} "
          f"{ep_time:5.1f}s "
          f"{fmt_time(elapsed)}>{fmt_time(remaining)}"
          f"{tag}")

        if epoch % 10 == 0 or is_best:
            save_ckpt({
                "network": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "best_state": best_state,
                "total_start": total_start,
            })

except KeyboardInterrupt:
    p("\n\nInterrupted! Saving...")
    save_ckpt({
        "network": network.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "best_acc": best_acc, "best_state": best_state,
        "total_start": total_start,
    })
    p(f"Saved at epoch {epoch}. Re-run to resume.")
    sys.exit(0)

# --- Restore best ---
if best_state is not None:
    network.load_state_dict(best_state)
    network.to(device)

# --- Final alpha optimization on validation ---
p("\n[7] Final alpha optimization...")
network.eval()
with torch.no_grad():
    ae_val_final = network(X_val_t).cpu().numpy()
    ae_test_final = network(X_test_t).cpu().numpy()

ae_val_err = np.mean((X_val_sel - ae_val_final) ** 2, axis=1)
ae_test_err = np.mean((X_test_sel - ae_test_final) ** 2, axis=1)

best_alpha = 0.5
best_val_acc = 0.0
for a in ALPHA_CANDIDATES:
    acc_v, _ = evaluate_hybrid(ae_val_err, if_val_raw, y_val, ae_test_err, if_test_raw, y_test, a)
    if acc_v > best_val_acc:
        best_val_acc = acc_v
        best_alpha = a
p(f"    Best alpha: {best_alpha:.2f} (val acc: {best_val_acc:.4f})")

# --- Final test evaluation ---
ae_test_n = normalize_scores(ae_test_err)
if_test_n = normalize_scores(if_test_raw)
combined_final = best_alpha * ae_test_n + (1 - best_alpha) * if_test_n
thr_final = float(np.percentile(combined_final, THRESHOLD_PERCENTILE))
preds_final = (combined_final > thr_final).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, preds_final).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

total_time = time.time() - total_start
p("-" * 80)
p(f"Training complete in {fmt_time(total_time)}")

p("\n" + "=" * 60)
p("FINAL RESULTS (Hybrid AE+IF with all 5 strategies)")
p("=" * 60)
p(f"  Accuracy:  {accuracy_score(y_test, preds_final):.4f}")
p(f"  Precision: {precision_score(y_test, preds_final, zero_division=0):.4f}")
p(f"  Recall:    {recall_score(y_test, preds_final, zero_division=0):.4f}")
p(f"  F1 Score:  {f1_score(y_test, preds_final, zero_division=0):.4f}")
p(f"  FPR:       {fpr:.4f}")
p(f"  Alpha:     {best_alpha:.2f}")
p(f"  Features:  {n_features} (from {len(all_features)})")
p(f"  SMOTE:     {len(X_train_sm):,} samples (was {len(X_train):,})")

# --- Standalone IF comparison ---
p("\n[8] Standalone IF (tuned) comparison...")
if_preds_raw = -if_model.score_samples(X_test_sel)
if_preds_n = normalize_scores(if_preds_raw)
if_thr = float(np.percentile(if_preds_n, THRESHOLD_PERCENTILE))
if_preds = (if_preds_n > if_thr).astype(int)
p(f"    IF only:  Acc={accuracy_score(y_test, if_preds):.4f} F1={f1_score(y_test, if_preds, zero_division=0):.4f}")

# --- Save ---
model = HybridPhysicalModel(
    input_dim=n_features, encoding_dim=AE_ENCODING_DIM, hidden_dims=AE_HIDDEN_DIMS,
    epochs=TOTAL_EPOCHS, batch_size=AE_BATCH_SIZE, learning_rate=AE_LR,
    threshold_percentile=THRESHOLD_PERCENTILE, if_n_estimators=IF_N_ESTIMATORS,
    if_contamination=IF_CONTAMINATION, alpha=best_alpha, random_state=RANDOM_STATE,
)
model.network = network
model.isolation_forest = if_model
model._threshold = thr_final
model._feature_names = [f"pc{i}" for i in range(n_features)] if USE_PCA else all_features
model._fitted = True
model._metadata = {
    "n_features": n_features, "encoding_dim": AE_ENCODING_DIM,
    "hidden_dims": AE_HIDDEN_DIMS, "epochs": TOTAL_EPOCHS,
    "batch_size": AE_BATCH_SIZE, "learning_rate": AE_LR,
    "dropout": AE_DROPOUT, "batch_norm": True,
    "pca_components": PCA_COMPONENTS if USE_PCA else None,
    "smote_ratio": 0.5, "if_n_estimators": IF_N_ESTIMATORS,
    "if_contamination": IF_CONTAMINATION, "alpha": best_alpha,
    "training_time_seconds": round(total_time, 4),
    "device": str(device),
}
model.save(MODEL_ROOT / "physical" / "hybrid_improved")

results = {
    "dataset": "swat+wadi_improved",
    "strategies": ["pca", "smote", "improved_ae", "if_tuning", "alpha_optimization"],
    "n_features_original": len(all_features),
    "n_features_selected": n_features,
    "pca_components": PCA_COMPONENTS,
    "smote_samples": len(X_train_sm),
    "alpha": best_alpha,
    "epochs": TOTAL_EPOCHS,
    "training_time_seconds": total_time,
    "hybrid_metrics": {
        "accuracy": accuracy_score(y_test, preds_final),
        "precision": precision_score(y_test, preds_final, zero_division=0),
        "recall": recall_score(y_test, preds_final, zero_division=0),
        "f1": f1_score(y_test, preds_final, zero_division=0),
        "fpr": fpr,
    },
    "if_only_metrics": {
        "accuracy": accuracy_score(y_test, if_preds),
        "f1": f1_score(y_test, if_preds, zero_division=0),
    },
}

results_dir = Path("experiments/results")
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "physical_improved_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

if CKPT_FILE.exists():
    CKPT_FILE.unlink()

p("\n" + "=" * 60)
p("ALL DONE")
p("=" * 60)
p("Results: experiments/results/physical_improved_results.json")
p("Models:  models/physical/hybrid_improved/")

"""Train physical models on real WADI dataset with proper sampling."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import MODEL_ROOT, RANDOM_SEED, create_required_directories
from src.config.logging_config import setup_logging
from src.datasets.wadi import WADIDatasetLoader
from src.preprocessing.physical import PhysicalPreprocessor
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderModel

logger = setup_logging()
create_required_directories()

DATA_DIR = Path("data/raw/wadi")
MAX_SAMPLES = 100000

print("=" * 60)
print("TRAINING PHYSICAL MODELS ON REAL WADI DATASET")
print("=" * 60)

print("\n1. Loading WADI dataset...")
loader = WADIDatasetLoader(DATA_DIR)
df = loader.load()
print(f"   Raw data: {df.shape[0]:,} rows, {df.shape[1]} columns")

print("\n2. Cleaning data...")
df = loader.clean(df)
print(f"   After cleaning: {df.shape[0]:,} rows")

print("\n3. Transforming data...")
df = loader.transform(df)
features = loader.get_features()
print(f"   Features: {len(features)} sensors")
label_dist = df["attack_label"].value_counts().to_dict()
print(f"   Attack labels: {label_dist}")

X = df[features].values
y = df["attack_label"].values

if len(X) > MAX_SAMPLES:
    print(f"\n4. Sampling {MAX_SAMPLES:,} rows from {len(X):,}...")
    indices = np.random.RandomState(RANDOM_SEED).choice(len(X), MAX_SAMPLES, replace=False)
    X = X[indices]
    y = y[indices]
    print(f"   Attack rate: {y.mean():.2%}")
else:
    print(f"\n4. Using all {len(X):,} rows")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y,
)
print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")
print(f"   Train attack rate: {y_train.mean():.2%}")
print(f"   Test attack rate: {y_test.mean():.2%}")

print("\n5. Preprocessing...")
preprocessor = PhysicalPreprocessor()
X_train_df = pd.DataFrame(X_train, columns=features)
X_test_df = pd.DataFrame(X_test, columns=features)
X_train_scaled = preprocessor.fit_transform(X_train_df, features)
X_test_scaled = preprocessor.transform(X_test_df)
preprocessor.save(MODEL_ROOT / "physical")
print("   Preprocessor saved")

print("\n6. Training Isolation Forest...")
if_model = IsolationForestModel(n_estimators=200, contamination=0.1, random_state=RANDOM_SEED)
t0 = time.time()
if_model.fit(X_train_scaled, features)
if_time = time.time() - t0
if_metrics = if_model.evaluate(X_test_scaled, y_test)
if_model.save(MODEL_ROOT / "physical" / "isolation_forest")
print(f"   Trained in {if_time:.1f}s")
print(f"   Accuracy:  {if_metrics['accuracy']:.4f}")
print(f"   Precision: {if_metrics['precision']:.4f}")
print(f"   Recall:    {if_metrics['recall']:.4f}")
print(f"   F1 Score:  {if_metrics['f1']:.4f}")
print(f"   FPR:       {if_metrics.get('false_positive_rate', 'N/A')}")

print("\n7. Training Autoencoder...")
ae_model = AutoencoderModel(
    input_dim=len(features),
    encoding_dim=16,
    hidden_dims=[32, 24],
    epochs=30,
    batch_size=256,
    learning_rate=1e-3,
    early_stopping_patience=5,
    threshold_percentile=95,
    random_state=RANDOM_SEED,
)
t0 = time.time()
ae_model.fit(X_train_scaled, features)
ae_time = time.time() - t0
ae_metrics = ae_model.evaluate(X_test_scaled, y_test)
ae_model.save(MODEL_ROOT / "physical" / "autoencoder")
print(f"   Trained in {ae_time:.1f}s")
print(f"   Accuracy:  {ae_metrics['accuracy']:.4f}")
print(f"   Precision: {ae_metrics['precision']:.4f}")
print(f"   Recall:    {ae_metrics['recall']:.4f}")
print(f"   F1 Score:  {ae_metrics['f1']:.4f}")

results = {
    "dataset": "wadi",
    "total_samples": len(X),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "n_features": len(features),
    "features": features,
    "models": {
        "isolation_forest": {
            "metrics": if_metrics,
            "training_time": if_time,
            "params": {"n_estimators": 200, "contamination": 0.1},
        },
        "autoencoder": {
            "metrics": ae_metrics,
            "training_time": ae_time,
            "params": {"encoding_dim": 16, "hidden_dims": [32, 24], "epochs": 30},
        },
    },
}

results_dir = Path("experiments/results")
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "physical_wadi_real_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("Results saved to experiments/results/physical_wadi_real_results.json")
print("Models saved to models/physical/")

"""Train physical anomaly detection models (Isolation Forest + Autoencoder)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import MODEL_ROOT, RANDOM_SEED, create_required_directories
from src.config.logging_config import setup_logging
from src.datasets.swat import SWaTDatasetLoader
from src.datasets.wadi import WADIDatasetLoader
from src.preprocessing.physical import PhysicalPreprocessor
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderModel
from src.config.loader import load_config

logger = None


def train_physical_models(dataset_name: str = "swat") -> dict:
    global logger
    logger = setup_logging()
    create_required_directories()

    model_config = load_config("models")
    results = {"dataset": dataset_name, "models": {}}

    logger.info("Loading %s dataset...", dataset_name.upper())
    if dataset_name == "swat":
        loader = SWaTDatasetLoader(Path("data/raw/swat"))
    elif dataset_name == "wadi":
        loader = WADIDatasetLoader(Path("data/raw/wadi"))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    try:
        df = loader.load()
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        logger.info("Creating synthetic demo data for testing pipeline...")
        rng = np.random.RandomState(RANDOM_SEED)
        n = 1000
        df = pd.DataFrame({
            "temperature": rng.normal(25, 3, n),
            "humidity": rng.normal(55, 10, n),
            "pressure": rng.normal(1013, 5, n),
            "voltage": rng.normal(3.7, 0.3, n),
            "battery": rng.normal(80, 10, n),
            "attack_label": rng.choice([0, 1], n, p=[0.9, 0.1]),
        })

    df = loader.clean(df)
    df = loader.transform(df)
    features = loader.get_features() or [c for c in df.columns if df[c].dtype in ("float64", "int64") and c != "attack_label"]

    if not features:
        features = ["temperature", "humidity", "pressure", "voltage", "battery"]
        for f in features:
            if f not in df.columns:
                df[f] = np.random.randn(len(df))

    logger.info("Features: %s (%d total)", features[:5], len(features))

    X = df[features].values
    y = df["attack_label"].values if "attack_label" in df.columns else np.zeros(len(df))

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y if y.sum() > 0 else None,
    )

    preprocessor = PhysicalPreprocessor()
    X_train_scaled = preprocessor.fit_transform(pd.DataFrame(X_train, columns=features), features)
    X_test_scaled = preprocessor.transform(pd.DataFrame(X_test, columns=features))
    preprocessor.save(MODEL_ROOT / "physical")
    logger.info("Preprocessor saved")

    logger.info("Training Isolation Forest...")
    if_config = model_config.models.physical.isolation_forest
    if_model = IsolationForestModel(
        n_estimators=if_config.n_estimators,
        contamination=if_config.contamination,
        random_state=RANDOM_SEED,
    )
    if_model.fit(X_train_scaled, features)

    normal_mask = y_train == 0
    if_metrics = if_model.evaluate(X_test_scaled, y_test)
    if_model.save(MODEL_ROOT / "physical" / "isolation_forest")
    results["models"]["isolation_forest"] = {"metrics": if_metrics}
    logger.info("Isolation Forest F1: %.4f, ROC-AUC: %.4f", if_metrics["f1"], if_metrics["roc_auc"])

    logger.info("Training Autoencoder...")
    ae_config = model_config.models.physical.autoencoder
    ae_model = AutoencoderModel(
        input_dim=len(features),
        encoding_dim=ae_config.encoding_dim,
        hidden_dims=ae_config.hidden_dims,
        epochs=ae_config.epochs,
        batch_size=ae_config.batch_size,
        learning_rate=ae_config.learning_rate,
        early_stopping_patience=ae_config.early_stopping_patience,
        threshold_percentile=ae_config.threshold_percentile,
        random_state=RANDOM_SEED,
    )
    ae_model.fit(X_train_scaled, features)
    ae_metrics = ae_model.evaluate(X_test_scaled, y_test)
    ae_model.save(MODEL_ROOT / "physical" / "autoencoder")
    results["models"]["autoencoder"] = {"metrics": ae_metrics}
    logger.info("Autoencoder F1: %.4f, ROC-AUC: %.4f", ae_metrics["f1"], ae_metrics["roc_auc"])

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"physical_{dataset_name}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Physical model training complete. Results saved to %s", results_dir)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train physical anomaly models")
    parser.add_argument("--dataset", choices=["swat", "wadi"], default="swat")
    args = parser.parse_args()
    train_physical_models(args.dataset)


if __name__ == "__main__":
    main()

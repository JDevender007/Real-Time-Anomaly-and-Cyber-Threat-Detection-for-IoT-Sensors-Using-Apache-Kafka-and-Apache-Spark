"""Train cyber threat detection models (Random Forest + XGBoost)."""
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
from src.datasets.ciciot2023 import CICIoT2023DatasetLoader
from src.datasets.edge_iiotset import EdgeIIoTSetDatasetLoader
from src.preprocessing.cyber import CyberPreprocessor
from src.models.random_forest import RandomForestModel
from src.models.xgboost import XGBoostModel
from src.config.loader import load_config

logger = None


def train_cyber_models(dataset_name: str = "ciciot2023") -> dict:
    global logger
    logger = setup_logging()
    create_required_directories()

    model_config = load_config("models")
    results = {"dataset": dataset_name, "models": {}}

    logger.info("Loading %s dataset...", dataset_name.upper())
    if dataset_name == "ciciot2023":
        loader = CICIoT2023DatasetLoader(Path("data/raw/ciciot2023"))
    elif dataset_name == "edge_iiotset":
        loader = EdgeIIoTSetDatasetLoader(Path("data/raw/edge_iiotset"))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    try:
        df = loader.load()
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        logger.info("Creating synthetic demo data for testing pipeline...")
        rng = np.random.RandomState(RANDOM_SEED)
        n = 1000
        labels = rng.choice(["Benign", "DDoS", "BruteForce", "PortScan"], n, p=[0.7, 0.15, 0.1, 0.05])
        df = pd.DataFrame({
            "packet_count": rng.randint(1, 1000, n),
            "packet_rate": rng.exponential(20, n),
            "bytes": rng.randint(100, 50000, n),
            "flow_duration": rng.exponential(5, n),
            "source_port": rng.randint(1024, 65535, n),
            "destination_port": rng.choice([80, 443, 22, 53, 8080], n),
            "label": labels,
        })

    df = loader.clean(df)
    df = loader.transform(df)
    features = loader.get_features() or [c for c in df.columns if df[c].dtype in ("float64", "int64", "int32", "float32")]

    if not features:
        features = ["packet_count", "packet_rate", "bytes", "flow_duration",
                     "source_port", "destination_port"]
        for f in features:
            if f not in df.columns:
                df[f] = np.random.randn(len(df))

    label_col = "attack_label"
    if label_col not in df.columns:
        for c in df.columns:
            if "label" in c.lower() or "attack" in c.lower():
                label_col = c
                break

    logger.info("Features: %s (%d total)", features[:5], len(features))
    logger.info("Label column: %s", label_col)

    preprocessor = CyberPreprocessor()
    X_scaled, y_encoded = preprocessor.fit_transform(df, features, label_col)
    label_classes = preprocessor.label_classes
    preprocessor.save(MODEL_ROOT / "cyber")
    logger.info("Preprocessor saved. Classes: %s", label_classes)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=RANDOM_SEED,
        stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None,
    )

    logger.info("Training Random Forest...")
    rf_config = model_config.models.cyber.random_forest
    rf_model = RandomForestModel(
        n_estimators=rf_config.n_estimators,
        max_depth=rf_config.max_depth,
        min_samples_split=rf_config.min_samples_split,
        min_samples_leaf=rf_config.min_samples_leaf,
        class_weight=rf_config.class_weight,
        random_state=RANDOM_SEED,
    )
    rf_model.fit(X_train, y_train, feature_names=features, label_classes=label_classes)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    rf_model.save(MODEL_ROOT / "cyber" / "random_forest")
    results["models"]["random_forest"] = {"metrics": rf_metrics}
    logger.info("Random Forest Accuracy: %.4f, F1: %.4f", rf_metrics["accuracy"], rf_metrics["f1"])

    logger.info("Training XGBoost...")
    xgb_config = model_config.models.cyber.xgboost
    xgb_model = XGBoostModel(
        n_estimators=xgb_config.n_estimators,
        max_depth=xgb_config.max_depth,
        learning_rate=xgb_config.learning_rate,
        subsample=xgb_config.subsample,
        colsample_bytree=xgb_config.colsample_bytree,
        random_state=RANDOM_SEED,
    )
    xgb_model.fit(X_train, y_train, feature_names=features, label_classes=label_classes)
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    xgb_model.save(MODEL_ROOT / "cyber" / "xgboost")
    results["models"]["xgboost"] = {"metrics": xgb_metrics}
    logger.info("XGBoost Accuracy: %.4f, F1: %.4f", xgb_metrics["accuracy"], xgb_metrics["f1"])

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"cyber_{dataset_name}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Cyber model training complete. Results saved to %s", results_dir)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train cyber threat models")
    parser.add_argument("--dataset", choices=["ciciot2023", "edge_iiotset"], default="ciciot2023")
    args = parser.parse_args()
    train_cyber_models(args.dataset)


if __name__ == "__main__":
    main()

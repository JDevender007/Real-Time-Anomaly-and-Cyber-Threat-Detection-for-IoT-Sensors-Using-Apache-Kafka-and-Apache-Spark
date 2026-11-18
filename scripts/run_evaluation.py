"""Run research experiments end-to-end."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import RANDOM_SEED, DATA_ROOT, MODEL_ROOT
from src.config.logging_config import setup_logging
from src.datasets.swat import SWaTDatasetLoader
from src.datasets.ciciot2023 import CICIoT2023DatasetLoader
from src.preprocessing.physical import PhysicalPreprocessor
from src.preprocessing.cyber import CyberPreprocessor
from src.research.experiments import CrossDatasetValidator, AblationStudy, ExperimentSuite
from src.research.figures import (
    plot_model_comparison, plot_confusion_matrix,
    plot_ablation_results, plot_training_time_comparison,
    set_style,
)

logger = None


def load_or_synthetic(loader_class, data_dir, name):
    try:
        loader = loader_class(data_dir)
        df = loader.load()
        df = loader.clean(df)
        df = loader.transform(df)
        return df, loader
    except Exception as e:
        logger.warning("Could not load %s: %s. Using synthetic data.", name, e)
        rng = np.random.RandomState(RANDOM_SEED)
        n = 5000
        if name in ("swat", "wadi"):
            labels = np.zeros(n, dtype=int)
            labels[1000:1100] = 1
            labels[3500:3550] = 1
            df = pd.DataFrame({
                "FIT101": rng.normal(2.5, 0.3, n),
                "LIT101": rng.normal(500, 20, n),
                "FIT102": rng.normal(2.4, 0.3, n),
                "LIT102": rng.normal(300, 15, n),
                "FIT103": rng.normal(1.2, 0.2, n),
                "LIT103": rng.normal(600, 25, n),
                "FIT104": rng.normal(2.3, 0.3, n),
                "LIT104": rng.normal(400, 18, n),
                "P101": rng.choice([0, 1], n),
                "P102": rng.choice([0, 1], n),
                "MV101": rng.choice([0, 1], n),
                "MV201": rng.choice([0, 1], n),
                "attack_label": labels,
            })
        else:
            labels = rng.choice(
                ["Benign", "DDoS", "DoS", "BruteForce"], n, p=[0.5, 0.2, 0.2, 0.1],
            )
            df = pd.DataFrame({
                "flow_duration": np.abs(rng.exponential(5, n)),
                "packet_count": rng.randint(1, 10000, n),
                "byte_count": rng.randint(100, 1000000, n),
                "packet_rate": np.abs(rng.exponential(50, n)),
                "byte_rate": np.abs(rng.exponential(5000, n)),
                "flow_iat_mean": np.abs(rng.exponential(1, n)),
                "flow_iat_std": np.abs(rng.exponential(0.5, n)),
                "fwd_iat_mean": np.abs(rng.exponential(1.2, n)),
                "bwd_iat_mean": np.abs(rng.exponential(1.5, n)),
                "fwd_pkt_len_mean": rng.uniform(20, 1500, n),
                "bwd_pkt_len_mean": rng.uniform(20, 1500, n),
                "src_port": rng.randint(1024, 65535, n),
                "dst_port": rng.choice([80, 443, 22, 53, 8080], n),
                "attack_label": labels,
            })
        return df, None


def run_cross_dataset_validation():
    global logger
    logger = setup_logging()
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("experiments/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CROSS-DATASET VALIDATION EXPERIMENT")
    logger.info("=" * 60)

    physical_datasets = [
        ("swat", SWaTDatasetLoader, DATA_ROOT / "raw" / "swat"),
        ("wadi", WADIDatasetLoader, DATA_ROOT / "raw" / "wadi"),
    ]
    cyber_datasets = [
        ("ciciot2023", CICIoT2023DatasetLoader, DATA_ROOT / "raw" / "ciciot2023"),
    ]

    all_results = ExperimentSuite(name="cross_dataset_all")

    for name, loader_class, data_dir in physical_datasets:
        logger.info("\n--- Physical Dataset: %s ---", name.upper())
        df, loader = load_or_synthetic(loader_class, data_dir, name)

        phys_features = [c for c in df.columns
                        if df[c].dtype in ("float64", "int64")
                        and c != "attack_label"]
        X = df[phys_features].values
        y = df["attack_label"].values

        validator = CrossDatasetValidator(n_folds=3)
        suite = validator.run_physical_validation(X, y, phys_features, name)
        all_results.results.extend(suite.results)
        logger.info("Completed %s: %d experiments", name, len(suite.results))

    for name, loader_class, data_dir in cyber_datasets:
        logger.info("\n--- Cyber Dataset: %s ---", name.upper())
        df, loader = load_or_synthetic(loader_class, data_dir, name)

        cyber_features = [c for c in df.columns
                         if df[c].dtype in ("float64", "int64")
                         and c != "attack_label"]
        preprocessor = CyberPreprocessor()
        X_scaled, y_encoded = preprocessor.fit_transform(df, cyber_features, "attack_label")

        validator = CrossDatasetValidator(n_folds=3)
        suite = validator.run_cyber_validation(
            X_scaled, y_encoded, cyber_features, preprocessor.label_classes, name,
        )
        all_results.results.extend(suite.results)
        logger.info("Completed %s: %d experiments", name, len(suite.results))

    all_results.save(results_dir / "cross_dataset_validation.json")
    logger.info("Results saved to experiments/results/cross_dataset_validation.json")

    summary_df = all_results.summary()
    if not summary_df.empty and "metrics" in summary_df.columns:
        metrics_rows = []
        for _, row in summary_df.iterrows():
            m = row["metrics"]
            metrics_rows.append({
                "dataset": row["dataset"],
                "model": row["model"],
                "fold": row["fold"],
                "accuracy": m.get("accuracy", 0),
                "f1": m.get("f1", 0),
                "roc_auc": m.get("roc_auc", 0),
                "training_time": row.get("training_time_s", 0),
            })
        metrics_df = pd.DataFrame(metrics_rows)
        agg = metrics_df.groupby(["dataset", "model"]).agg({
            "accuracy": ["mean", "std"],
            "f1": ["mean", "std"],
            "roc_auc": ["mean", "std"],
            "training_time": "mean",
        }).round(4)
        logger.info("\nAggregated Results:\n%s", agg)
        agg.to_csv(results_dir / "cross_dataset_aggregated.csv")

    return all_results


def run_ablation_study():
    global logger
    logger = setup_logging()
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ABLATION STUDY")
    logger.info("=" * 60)

    df, _ = load_or_synthetic(SWaTDatasetLoader, DATA_ROOT / "raw" / "swat", "swat")

    phys_features = [c for c in df.columns
                    if df[c].dtype in ("float64", "int64")
                    and c != "attack_label"]
    X = df[phys_features].values
    y = df["attack_label"].values

    ablation = AblationStudy()
    feature_suite = ablation.physical_feature_ablation(X, y, phys_features, "swat")
    feature_suite.save(results_dir / "ablation_features.json")
    logger.info("Feature ablation: %d experiments", len(feature_suite.results))

    summary = feature_suite.summary()
    if not summary.empty:
        param_data = []
        for _, row in summary.iterrows():
            params = row.get("params", {})
            if isinstance(params, dict):
                param_data.append({
                    "n_features": params.get("n_features", 0),
                    "accuracy": row.get("metrics", {}).get("accuracy", 0),
                    "f1": row.get("metrics", {}).get("f1", 0),
                    "roc_auc": row.get("metrics", {}).get("roc_auc", 0),
                    "training_time": row.get("training_time_s", 0),
                })
        if param_data:
            param_df = pd.DataFrame(param_data)
            param_df.to_csv(results_dir / "ablation_features_summary.csv", index=False)
            logger.info("Feature ablation summary:\n%s", param_df)

    return feature_suite


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run research experiments")
    parser.add_argument("--experiment", choices=["cross-dataset", "ablation", "all"],
                        default="all")
    args = parser.parse_args()

    if args.experiment in ("cross-dataset", "all"):
        run_cross_dataset_validation()

    if args.experiment in ("ablation", "all"):
        run_ablation_study()

    logger.info("\nAll experiments complete. Results in experiments/results/")


if __name__ == "__main__":
    main()

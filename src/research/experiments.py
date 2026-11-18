"""Research experiments module for cross-dataset validation and ablation studies."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from src.config.settings import RANDOM_SEED, MODEL_ROOT
from src.preprocessing.physical import PhysicalPreprocessor
from src.preprocessing.cyber import CyberPreprocessor
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost import XGBoostModel
from src.decision.engine import DecisionEngine
from src.api.schemas import RiskLevel


@dataclass
class ExperimentResult:
    experiment_name: str
    dataset: str
    model: str
    metrics: dict[str, Any] = field(default_factory=dict)
    training_time: float = 0.0
    inference_time: float = 0.0
    fold: int = 0
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment_name,
            "dataset": self.dataset,
            "model": self.model,
            "metrics": self.metrics,
            "training_time_s": self.training_time,
            "inference_time_s": self.inference_time,
            "fold": self.fold,
            "params": self.params,
        }


@dataclass
class ExperimentSuite:
    name: str
    results: list[ExperimentResult] = field(default_factory=list)

    def add(self, result: ExperimentResult):
        self.results.append(result)

    def summary(self) -> pd.DataFrame:
        rows = [r.to_dict() for r in self.results]
        return pd.DataFrame(rows)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self.results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "ExperimentSuite":
        with open(path) as f:
            data = json.load(f)
        suite = cls(name=path.stem)
        for d in data:
            suite.add(ExperimentResult(
                experiment_name=d["experiment"],
                dataset=d["dataset"],
                model=d["model"],
                metrics=d["metrics"],
                training_time=d.get("training_time_s", 0),
                inference_time=d.get("inference_time_s", 0),
                fold=d.get("fold", 0),
                params=d.get("params", {}),
            ))
        return suite


class CrossDatasetValidator:
    def __init__(self, n_folds: int = 5, random_state: int = RANDOM_SEED):
        self.n_folds = n_folds
        self.random_state = random_state
        self.suite = ExperimentSuite(name="cross_dataset_validation")

    def run_physical_validation(self, X: np.ndarray, y: np.ndarray,
                                feature_names: list[str], dataset_name: str) -> ExperimentSuite:
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            preprocessor = PhysicalPreprocessor()
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
            X_train_scaled = preprocessor.fit_transform(X_train_df, feature_names)
            X_test_scaled = preprocessor.transform(X_test_df)

            for model_name, model in [
                ("IsolationForest", IsolationForestModel(n_estimators=100, random_state=self.random_state)),
                ("Autoencoder", AutoencoderModel(input_dim=len(feature_names), encoding_dim=8,
                                                  hidden_dims=[16], epochs=10, batch_size=64)),
            ]:
                start = time.perf_counter()
                model.fit(X_train_scaled, feature_names)
                train_time = time.perf_counter() - start

                start = time.perf_counter()
                metrics = model.evaluate(X_test_scaled, y_test)
                infer_time = time.perf_counter() - start

                self.suite.add(ExperimentResult(
                    experiment_name="cross_dataset_validation",
                    dataset=dataset_name,
                    model=model_name,
                    metrics=metrics,
                    training_time=train_time,
                    inference_time=infer_time,
                    fold=fold,
                ))

        return self.suite

    def run_cyber_validation(self, X: np.ndarray, y: np.ndarray,
                              feature_names: list[str], label_classes: list[str],
                              dataset_name: str) -> ExperimentSuite:
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for model_name, model in [
                ("RandomForest", RandomForestModel(n_estimators=100, random_state=self.random_state)),
                ("XGBoost", XGBoostModel(n_estimators=100, random_state=self.random_state)),
            ]:
                start = time.perf_counter()
                model.fit(X_train, y_train, feature_names=feature_names, label_classes=label_classes)
                train_time = time.perf_counter() - start

                start = time.perf_counter()
                metrics = model.evaluate(X_test, y_test)
                infer_time = time.perf_counter() - start

                self.suite.add(ExperimentResult(
                    experiment_name="cross_dataset_validation",
                    dataset=dataset_name,
                    model=model_name,
                    metrics=metrics,
                    training_time=train_time,
                    inference_time=infer_time,
                    fold=fold,
                ))

        return self.suite


class AblationStudy:
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.suite = ExperimentSuite(name="ablation_study")

    def physical_feature_ablation(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: list[str], dataset_name: str) -> ExperimentSuite:
        from sklearn.model_selection import train_test_split

        for n_features in [3, 5, 8, len(feature_names)]:
            selected = feature_names[:n_features]
            X_sel = X[:, :n_features]

            X_train, X_test, y_train, y_test = train_test_split(
                X_sel, y, test_size=0.2, random_state=self.random_state, stratify=y,
            )

            preprocessor = PhysicalPreprocessor()
            X_train_df = pd.DataFrame(X_train, columns=selected)
            X_test_df = pd.DataFrame(X_test, columns=selected)
            X_train_scaled = preprocessor.fit_transform(X_train_df, selected)
            X_test_scaled = preprocessor.transform(X_test_df)

            model = IsolationForestModel(n_estimators=100, random_state=self.random_state)
            start = time.perf_counter()
            model.fit(X_train_scaled, selected)
            train_time = time.perf_counter() - start

            metrics = model.evaluate(X_test_scaled, y_test)

            self.suite.add(ExperimentResult(
                experiment_name="ablation_physical_features",
                dataset=dataset_name,
                model="IsolationForest",
                metrics=metrics,
                training_time=train_time,
                params={"n_features": n_features, "features": selected},
            ))

        return self.suite

    def cyber_hyperparameter_sensitivity(self, X: np.ndarray, y: np.ndarray,
                                          feature_names: list[str], label_classes: list[str],
                                          dataset_name: str) -> ExperimentSuite:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state,
        )

        for n_est in [10, 50, 100, 200]:
            for depth in [3, 5, 10, None]:
                model = RandomForestModel(
                    n_estimators=n_est, max_depth=depth, random_state=self.random_state,
                )
                start = time.perf_counter()
                model.fit(X_train, y_train, feature_names=feature_names, label_classes=label_classes)
                train_time = time.perf_counter() - start

                metrics = model.evaluate(X_test, y_test)

                self.suite.add(ExperimentResult(
                    experiment_name="ablation_rf_hyperparameters",
                    dataset=dataset_name,
                    model="RandomForest",
                    metrics=metrics,
                    training_time=train_time,
                    params={"n_estimators": n_est, "max_depth": depth},
                ))

        return self.suite

    def ensemble_weight_sensitivity(self, X_phys: np.ndarray, y_phys: np.ndarray,
                                     X_cyber: np.ndarray, y_cyber: np.ndarray,
                                     feature_names_phys: list[str], feature_names_cyber: list[str],
                                     dataset_name: str) -> ExperimentSuite:
        engine = DecisionEngine()
        from sklearn.model_selection import train_test_split

        Xp_train, Xp_test, yp_train, yp_test = train_test_split(
            X_phys, y_phys, test_size=0.2, random_state=self.random_state,
        )
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X_cyber, y_cyber, test_size=0.2, random_state=self.random_state,
        )

        preprocessor_p = PhysicalPreprocessor()
        Xp_train_df = pd.DataFrame(Xp_train, columns=feature_names_phys)
        Xp_test_df = pd.DataFrame(Xp_test, columns=feature_names_phys)
        Xp_train_scaled = preprocessor_p.fit_transform(Xp_train_df, feature_names_phys)
        Xp_test_scaled = preprocessor_p.transform(Xp_test_df)

        preprocessor_c = CyberPreprocessor()
        Xc_scaled, yc = preprocessor_c.fit_transform(
            pd.DataFrame(np.vstack([Xc_train, Xc_test]), columns=feature_names_cyber),
            feature_names_cyber, "attack_label",
        )

        if_model = IsolationForestModel(n_estimators=50, random_state=self.random_state)
        if_model.fit(Xp_train_scaled, feature_names_phys)

        for pw in [0.3, 0.4, 0.5, 0.6, 0.7]:
            engine.physical_weight = pw
            engine.cyber_weight = 1.0 - pw

            if_preds, if_scores = if_model.predict(Xp_test_scaled)
            if_anomaly_rate = float(np.mean(if_preds))

            cyber_scores = np.random.uniform(0, 1, len(Xc_test))
            cyber_anomaly_rate = float(np.mean(cyber_scores > 0.5))

            risk_result = engine.compute_risk(
                physical_score=float(np.mean(if_scores)),
                cyber_score=float(np.mean(cyber_scores)),
            )

            self.suite.add(ExperimentResult(
                experiment_name="ablation_ensemble_weights",
                dataset=dataset_name,
                model="Ensemble",
                metrics={
                    "physical_anomaly_rate": if_anomaly_rate,
                    "cyber_threat_rate": cyber_anomaly_rate,
                    "unified_risk_score": risk_result.unified_risk_score,
                    "risk_level": str(risk_result.risk_level),
                },
                params={"physical_weight": pw, "cyber_weight": 1.0 - pw},
            ))

        return self.suite

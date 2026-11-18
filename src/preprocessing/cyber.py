"""Cyber data preprocessing pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from src.config.settings import RANDOM_SEED


class CyberPreprocessor:
    """Preprocessing pipeline for cyber network data."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._fitted = False
        self._feature_names: list[str] = []
        self._label_classes: list[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: list[str], label_col: str) -> CyberPreprocessor:
        self._feature_names = feature_cols
        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler.fit(X)

        if label_col in df.columns:
            y = df[label_col].astype(str)
            self.label_encoder.fit(y)
            self._label_classes = list(self.label_encoder.classes_)

        self._fitted = True
        return self

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        X = df[self._feature_names].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(X)

    def transform_labels(self, df: pd.DataFrame, label_col: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        y = df[label_col].astype(str)
        return self.label_encoder.transform(y)

    def fit_transform(self, df: pd.DataFrame, feature_cols: list[str],
                      label_col: str) -> tuple[np.ndarray, np.ndarray]:
        self.fit(df, feature_cols, label_col)
        return self.transform_features(df), self.transform_labels(df, label_col)

    def inverse_transform_labels(self, y: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(y)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.pkl")
        joblib.dump(self.label_encoder, path / "label_encoder.pkl")
        joblib.dump(self._feature_names, path / "feature_names.pkl")
        joblib.dump(self._label_classes, path / "label_classes.pkl")

    def load(self, path: Path) -> CyberPreprocessor:
        self.scaler = joblib.load(path / "scaler.pkl")
        self.label_encoder = joblib.load(path / "label_encoder.pkl")
        self._feature_names = joblib.load(path / "feature_names.pkl")
        self._label_classes = joblib.load(path / "label_classes.pkl")
        self._fitted = True
        return self

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def label_classes(self) -> list[str]:
        return list(self._label_classes)

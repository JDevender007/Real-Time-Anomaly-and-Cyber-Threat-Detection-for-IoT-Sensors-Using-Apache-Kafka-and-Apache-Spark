"""Physical data preprocessing pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from src.config.settings import RANDOM_SEED


class PhysicalPreprocessor:
    """Preprocessing pipeline for physical sensor data."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self._fitted = False
        self._feature_names: list[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> PhysicalPreprocessor:
        self._feature_names = feature_cols
        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        X = df[self._feature_names].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(X)

    def fit_transform(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        self.fit(df, feature_cols)
        return self.transform(df)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.pkl")
        joblib.dump(self._feature_names, path / "feature_names.pkl")

    def load(self, path: Path) -> PhysicalPreprocessor:
        self.scaler = joblib.load(path / "scaler.pkl")
        self._feature_names = joblib.load(path / "feature_names.pkl")
        self._fitted = True
        return self

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

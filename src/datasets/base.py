"""Base dataset loader interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseDatasetLoader(ABC):
    """Abstract base class for all dataset loaders."""

    def __init__(self, data_dir: Path, name: str, dataset_type: str) -> None:
        self.data_dir = data_dir
        self.name = name
        self.dataset_type = dataset_type  # "physical" or "cyber"
        self._metadata: dict[str, Any] = {}

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load the raw dataset."""

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate the loaded dataset has expected columns."""

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset (handle missing values, duplicates, etc.)."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features into ML-ready format."""

    @abstractmethod
    def get_features(self) -> list[str]:
        """Return list of feature column names."""

    @abstractmethod
    def get_labels(self) -> list[str]:
        """Return list of label column names."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return dataset metadata."""

    def split(self, df: pd.DataFrame, test_size: float = 0.2,
              random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/test."""
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=test_size, random_state=random_state)

    def get_raw_path(self) -> Path:
        return self.data_dir

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.dataset_type,
            "data_dir": str(self.data_dir),
            "exists": self.data_dir.exists(),
        }

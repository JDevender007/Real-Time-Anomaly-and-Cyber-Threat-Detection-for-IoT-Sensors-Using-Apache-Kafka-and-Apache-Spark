"""CICIoT2023 dataset loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.datasets.base import BaseDatasetLoader
from src.utils.exceptions import DatasetLoadError, DatasetValidationError


class CICIoT2023DatasetLoader(BaseDatasetLoader):
    """Loader for CICIoT2023 cyber threat dataset.

    CICIoT2023 contains network traffic from IoT devices with various attack types:
    DDoS, DoS, Brute Force, Reconnaissance, Spoofing, Botnet, etc.

    Expected files:
        - pcap-based CSV or preprocessed CSV files
        - Network flow features with attack labels

    Reference: https://www.unb.ca/cic/datasets/iot-2023.html
    """

    LABEL_COLUMN = "label"

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, name="CICIoT2023", dataset_type="cyber")
        self._features: list[str] = []
        self._label_map: dict[str, str] = {}

    def load(self) -> pd.DataFrame:
        csv_files = [f for f in self.data_dir.rglob("*.csv")
                     if "synthetic" not in f.name.lower()]
        if not csv_files:
            csv_files = list(self.data_dir.rglob("*.csv"))
        if not csv_files:
            raise DatasetLoadError(
                f"No CSV files found in {self.data_dir}. "
                f"Please download CICIoT2023 dataset and place CSV files in: {self.data_dir}"
            )

        frames = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, encoding="utf-8", low_memory=False)
                frames.append(df)
            except Exception as e:
                raise DatasetLoadError(f"Failed to load {f}: {e}")

        combined = pd.concat(frames, ignore_index=True)
        self._metadata = {
            "source_files": [f.name for f in csv_files],
            "rows": len(combined),
            "columns": list(combined.columns),
        }
        return combined

    def validate(self, df: pd.DataFrame) -> bool:
        label_found = False
        for col in df.columns:
            if col.lower() in ("label", "attack", "class", "category"):
                label_found = True
                self.LABEL_COLUMN = col
                break

        if not label_found:
            raise DatasetValidationError(
                f"CICIoT2023 dataset missing label column. "
                f"Available columns: {list(df.columns)}"
            )
        return True

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(how="all")
        df = df.drop_duplicates()

        for col in df.columns:
            if col.lower() in ("label", "attack", "class", "category", "timestamp", "time", "attack_label"):
                continue
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass

        if "attack_label" in df.columns:
            pass
        elif self.LABEL_COLUMN in df.columns:
            unique_labels = df[self.LABEL_COLUMN].unique()
            self._label_map = {str(l): str(l) for l in unique_labels}
            df["attack_label"] = df[self.LABEL_COLUMN].apply(
                lambda x: 0 if "benign" in str(x).strip().lower() or "normal" in str(x).strip().lower() else 1
            )

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        exclude_cols = {"Timestamp", "Time", "timestamp", "time", "label", "attack",
                         "class", "category", "attack_label", "Label"}
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and df[c].dtype in ("int64", "float64", "int32", "float32")
        ]
        if not feature_cols:
            feature_cols = [c for c in df.columns if c not in exclude_cols]
        self._features = feature_cols

        numeric_cols = [c for c in feature_cols if df[c].dtype in ("int64", "float64", "int32", "float32")]
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            inf_cols = df[numeric_cols].columns[df[numeric_cols].isin([float("inf"), float("-inf")]).any()]
            for col in inf_cols:
                df[col] = df[col].replace([float("inf"), float("-inf")], float("nan"))
                df[col] = df[col].fillna(df[col].median())

        return df

    def get_features(self) -> list[str]:
        return self._features

    def get_labels(self) -> list[str]:
        return ["attack_label"]

    def get_label_taxonomy(self) -> dict[str, str]:
        return self._label_map

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.dataset_type,
            "description": "CICIoT2023 - IoT network traffic with attack labels",
            "label_column": self.LABEL_COLUMN,
            "attack_types": list(self._label_map.values()),
            "expected_files": "CSV files with network flow features",
            **self._metadata,
        }

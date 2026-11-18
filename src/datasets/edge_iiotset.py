"""Edge-IIoTSet dataset loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.datasets.base import BaseDatasetLoader
from src.utils.exceptions import DatasetLoadError, DatasetValidationError


class EdgeIIoTSetDatasetLoader(BaseDatasetLoader):
    """Loader for Edge-IIoTSet cyber threat dataset.

    Edge-IIoTSet contains network traffic from Edge-IIoT devices with
    various attack categories relevant to industrial IoT.

    Expected files:
        - CSV files with network flow features
        - Attack type labels

    Reference: https://research.aimaster.ir/
    """

    LABEL_COLUMN = "Attack_label"

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, name="Edge-IIoTSet", dataset_type="cyber")
        self._features: list[str] = []
        self._label_map: dict[str, str] = {}

    def load(self) -> pd.DataFrame:
        ml_file = self.data_dir / "ML-EdgeIIoT-dataset.csv"
        if ml_file.exists():
            csv_files = [ml_file]
        else:
            csv_files = [f for f in self.data_dir.glob("*.csv")
                         if "synthetic" not in f.name.lower()
                         and "live" not in f.name.lower()
                         and "dnn" not in f.name.lower()]
        if not csv_files:
            raise DatasetLoadError(
                f"No CSV files found in {self.data_dir}. "
                f"Please download Edge-IIoTSet dataset and place CSV files in: {self.data_dir}"
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
        for col in df.columns:
            if col.lower() in ("attack_label", "attack_type", "attack", "label", "class"):
                self.LABEL_COLUMN = col
                break
        if self.LABEL_COLUMN not in df.columns:
            raise DatasetValidationError(
                f"Edge-IIoTSet dataset missing label column. "
                f"Available columns: {list(df.columns)}"
            )
        return True

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(how="all")
        df = df.drop_duplicates()

        exclude_cols = {"attack_type", "attack_label", "attack", "label", "class", "timestamp", "time",
                        "frame.time", "ip.src_host", "ip.dst_host"}
        for col in df.columns:
            if col.lower() in exclude_cols or col in exclude_cols:
                continue
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass

        if self.LABEL_COLUMN in df.columns:
            unique_labels = df[self.LABEL_COLUMN].unique()
            self._label_map = {str(l): str(l) for l in unique_labels}
            if self.LABEL_COLUMN == "Attack_label":
                df["attack_label"] = df[self.LABEL_COLUMN].astype(int)
            else:
                df["attack_label"] = df[self.LABEL_COLUMN].apply(
                    lambda x: 0 if str(x).strip().lower() in ("normal", "benign", "0", "") else 1
                )

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        exclude_cols = {"Timestamp", "Time", "timestamp", "time", "label", "attack",
                        "attack_type", "class", "category", "attack_label", "Label",
                        "frame.time", "ip.src_host", "ip.dst_host", "mqtt.protoname", "mqtt.topic"}
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and df[c].dtype in ("int64", "float64", "int32", "float32")
        ]
        self._features = feature_cols

        if feature_cols:
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

        inf_cols = df[feature_cols].columns[df[feature_cols].isin([float("inf"), float("-inf")]).any()]
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
            "description": "Edge-IIoTSet - Edge IoT network traffic with attack labels",
            "label_column": self.LABEL_COLUMN,
            "attack_types": list(self._label_map.values()),
            "expected_files": "CSV files with network flow features",
            **self._metadata,
        }

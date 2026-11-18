"""SWaT (Secure Water Treatment) dataset loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.datasets.base import BaseDatasetLoader
from src.utils.exceptions import DatasetLoadError, DatasetValidationError


class SWaTDatasetLoader(BaseDatasetLoader):
    """Loader for SWaT physical anomaly dataset.

    SWaT is a real-time cyber-physical testbed for water treatment.
    Contains 51 sensors measuring temperature, flow, pressure, level, etc.

    Expected files:
        - SWaT.Dataset.csv or similar CSV files
        - Physical sensor readings with attack labels

    Reference: https://www.s3.nd.edu/swat/
    """

    EXPECTED_COLUMNS = [
        "Timestamp", "FIT101", "LIT101", "MV101", "P101", "P102",
        "AIT201", "AIT202", "AIT203", "FIT201", "MV201", "MV202",
        "MV203", "MV204", "P201", "P202", "P203", "P204", "P205",
        "P206", "DPIT301", "FIT301", "LIT301", "MV301", "MV302",
        "MV303", "P301", "P302", "AIT401", "AIT402", "FIT401",
        "LIT401", "P401", "P402", "P403", "P404",
        "AIT501", "AIT502", "AIT503", "AIT504", "FIT501", "FIT502",
        "FIT503", "FIT504", "P501", "P502",
        "FIT601", "P601", "P602", "P603",
    ]

    LABEL_COLUMN = "Normal/Attack"

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, name="SWaT", dataset_type="physical")
        self._features: list[str] = []

    def load(self) -> pd.DataFrame:
        csv_files = [f for f in self.data_dir.glob("*.csv")
                     if "synthetic" not in f.name.lower() and "merged" not in f.name.lower()]
        if not csv_files:
            csv_files = [f for f in self.data_dir.glob("*.csv") if "synthetic" not in f.name.lower()]
        if not csv_files:
            csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise DatasetLoadError(
                f"No CSV files found in {self.data_dir}. "
                f"Please download SWaT dataset and place CSV files in: {self.data_dir}"
            )

        frames = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, encoding="utf-8", low_memory=False)
                df.columns = df.columns.str.strip()
                frames.append(df)
            except Exception as e:
                raise DatasetLoadError(f"Failed to load {f}: {e}")

        combined = pd.concat(frames, ignore_index=True)
        if "Normal/Attack" not in combined.columns and "attack_label" in combined.columns:
            pass
        self._metadata = {
            "source_files": [f.name for f in csv_files],
            "rows": len(combined),
            "columns": list(combined.columns),
        }
        return combined

    def validate(self, df: pd.DataFrame) -> bool:
        missing = [c for c in self.EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            raise DatasetValidationError(
                f"SWaT dataset missing columns: {missing}. "
                f"Available: {list(df.columns)}"
            )
        return True

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "Normal/Attack" in df.columns:
            df["attack_label"] = df["Normal/Attack"].apply(
                lambda x: 0 if str(x).strip().lower() == "normal" else 1
            )
        elif "attack_label" not in df.columns:
            df["attack_label"] = 0

        df = df.dropna(how="all")
        df = df.drop_duplicates()

        exclude_cols = {"Timestamp", "Normal/Attack", "attack_label"}
        for col in df.columns:
            if col in exclude_cols:
                continue
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        exclude = {"Timestamp", "Normal/Attack", "attack_label"}
        sensor_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "int64", "int32", "float32")]
        if not sensor_cols:
            sensor_cols = [c for c in df.columns if c not in exclude]
        self._features = sensor_cols

        for col in sensor_cols:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        numeric_cols = [c for c in sensor_cols if df[c].dtype in ("float64", "int64", "int32", "float32")]
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df

    def get_features(self) -> list[str]:
        return self._features

    def get_labels(self) -> list[str]:
        return ["attack_label"]

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.dataset_type,
            "description": "SWaT - Secure Water Treatment testbed",
            "sensors": len(self._features),
            "label_column": self.LABEL_COLUMN,
            "binary_labels": "Normal (0) vs Attack (1)",
            "expected_files": "SWaT.Dataset.csv or similar",
            **self._metadata,
        }

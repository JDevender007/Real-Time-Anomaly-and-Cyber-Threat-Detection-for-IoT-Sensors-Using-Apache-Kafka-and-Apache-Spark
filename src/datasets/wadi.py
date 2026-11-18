"""WADI (Water Distribution) dataset loader."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.datasets.base import BaseDatasetLoader
from src.utils.exceptions import DatasetLoadError, DatasetValidationError


class WADIDatasetLoader(BaseDatasetLoader):
    """Loader for WADI physical anomaly dataset.

    WADI is a water distribution testbed companion to SWaT.
    Contains 127 flow, pressure, and quality sensors over 16 days.

    Expected files:
        - WADI_14days.csv (14 days normal + 2 days attack, skip 4 header lines)
        - WADI_attackdata.csv (2 days attack data only)

    Reference: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
    """

    LABEL_COLUMN = "attack_label"

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, name="WADI", dataset_type="physical")
        self._features: list[str] = []

    def load(self) -> pd.DataFrame:
        frames = []

        # WADI_14days.csv has 4 metadata header lines before the real CSV header
        main_file = self.data_dir / "WADI_14days.csv"
        if main_file.exists():
            df_main = pd.read_csv(main_file, skiprows=4, encoding="utf-8", low_memory=False)
            df_main.columns = df_main.columns.str.strip()
            df_main["attack_label"] = 0  # 14-day file is mostly normal
            frames.append(df_main)

        # WADI_attackdata.csv has no metadata header, direct CSV
        attack_file = self.data_dir / "WADI_attackdata.csv"
        if attack_file.exists():
            df_attack = pd.read_csv(attack_file, encoding="utf-8", low_memory=False)
            df_attack.columns = df_attack.columns.str.strip()
            df_attack["attack_label"] = 1  # Attack data
            frames.append(df_attack)

        if not frames:
            raise DatasetLoadError(
                f"No WADI CSV files found in {self.data_dir}. "
                f"Expected: WADI_14days.csv, WADI_attackdata.csv"
            )

        combined = pd.concat(frames, ignore_index=True)

        # Clean column names: strip UNC path prefix
        new_cols = []
        for col in combined.columns:
            match = re.search(r"LOG_DATA\\(.+)$", col)
            if match:
                new_cols.append(match.group(1))
            else:
                new_cols.append(col)
        combined.columns = new_cols

        self._metadata = {
            "source_files": [f.name for f in self.data_dir.glob("*.csv")
                            if "synthetic" not in f.name.lower()],
            "rows": len(combined),
            "columns": list(combined.columns),
        }
        return combined

    def validate(self, df: pd.DataFrame) -> bool:
        if self.LABEL_COLUMN not in df.columns:
            alt_labels = [c for c in df.columns if "attack" in c.lower()]
            if not alt_labels:
                raise DatasetValidationError(
                    f"WADI dataset missing label column '{self.LABEL_COLUMN}'. "
                    f"Available: {list(df.columns)}"
                )
        return True

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "attack_label" not in df.columns:
            df["attack_label"] = 0

        df = df.dropna(how="all")
        df = df.drop_duplicates()

        exclude_cols = {"Timestamp", "Date", "Time", "Row", "attack_label"}
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
        exclude_cols = {"Row", "Date", "Time", "attack_label"}
        sensor_cols = [c for c in df.columns if c not in exclude_cols]
        self._features = sensor_cols

        for col in sensor_cols:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df[sensor_cols] = df[sensor_cols].fillna(df[sensor_cols].median())
        return df

    def get_features(self) -> list[str]:
        return self._features

    def get_labels(self) -> list[str]:
        return ["attack_label"]

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.dataset_type,
            "description": "WADI - Water Distribution testbed (127 sensors, 16 days)",
            "label_column": self.LABEL_COLUMN,
            "binary_labels": "Normal (0) vs Attack (1)",
            "expected_files": "WADI_14days.csv, WADI_attackdata.csv",
            **self._metadata,
        }

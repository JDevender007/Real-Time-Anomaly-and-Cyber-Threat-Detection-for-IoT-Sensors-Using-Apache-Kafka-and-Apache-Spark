"""Physical sensor feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


def compute_rolling_features(df: pd.DataFrame, sensor_cols: list[str],
                             window: int = 10) -> pd.DataFrame:
    df = df.copy()
    for col in sensor_cols:
        if col in df.columns:
            df[f"{col}_rolling_mean"] = df[col].rolling(window=window, min_periods=1).mean()
            df[f"{col}_rolling_std"] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
    return df


def compute_rate_of_change(df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in sensor_cols:
        if col in df.columns:
            df[f"{col}_roc"] = df[col].diff().fillna(0)
    return df


def compute_deviation_from_baseline(df: pd.DataFrame, sensor_cols: list[str],
                                     baseline: dict[str, float] | None = None) -> pd.DataFrame:
    df = df.copy()
    for col in sensor_cols:
        if col in df.columns:
            if baseline and col in baseline:
                base = baseline[col]
            else:
                base = df[col].mean()
            df[f"{col}_deviation"] = df[col] - base
    return df


def compute_sensor_relationships(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sensor_cols = [c for c in df.columns if df[c].dtype in ("float64", "int64")]
    if len(sensor_cols) >= 2:
        df["sensor_mean"] = df[sensor_cols].mean(axis=1)
        df["sensor_std"] = df[sensor_cols].std(axis=1).fillna(0)
        df["sensor_range"] = df[sensor_cols].max(axis=1) - df[sensor_cols].min(axis=1)
    return df


def engineer_physical_features(df: pd.DataFrame, sensor_cols: list[str],
                                window: int = 10) -> pd.DataFrame:
    df = compute_rolling_features(df, sensor_cols, window)
    df = compute_rate_of_change(df, sensor_cols)
    df = compute_deviation_from_baseline(df, sensor_cols)
    df = compute_sensor_relationships(df)
    return df

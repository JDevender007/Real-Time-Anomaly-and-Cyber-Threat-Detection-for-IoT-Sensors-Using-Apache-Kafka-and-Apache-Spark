"""Cyber network feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cyber_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "bytes" in df.columns and "packet_count" in df.columns:
        df["bytes_per_packet"] = np.where(
            df["packet_count"] > 0,
            df["bytes"] / df["packet_count"],
            0,
        )
    if "packet_count" in df.columns and "flow_duration" in df.columns:
        df["packets_per_second"] = np.where(
            df["flow_duration"] > 0,
            df["packet_count"] / df["flow_duration"],
            0,
        )
    if "bytes" in df.columns and "flow_duration" in df.columns:
        df["bytes_per_second"] = np.where(
            df["flow_duration"] > 0,
            df["bytes"] / df["flow_duration"],
            0,
        )
    return df


def engineer_cyber_features(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_cyber_features(df)
    return df

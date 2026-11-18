"""Data drift monitoring using KS test and PSI."""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class DriftAlert:
    feature: str
    metric: str
    value: float
    threshold: float
    timestamp: float
    severity: str = "warning"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "severity": self.severity,
        }


class DataDriftMonitor:
    def __init__(self, window_size: int = 1000, psi_threshold: float = 0.2,
                 ks_threshold: float = 0.05):
        self.window_size = window_size
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self._reference_data: dict[str, np.ndarray] = {}
        self._current_windows: dict[str, deque] = {}
        self._drift_alerts: list[DriftAlert] = []
        self._drift_history: list[dict[str, Any]] = []

    def set_reference(self, data: dict[str, np.ndarray]):
        self._reference_data = {k: np.array(v) for k, v in data.items()}
        for k in self._reference_data:
            self._current_windows[k] = deque(maxlen=self.window_size)

    def update(self, data: dict[str, float]):
        for k, v in data.items():
            if k in self._current_windows:
                self._current_windows[k].append(v)

    def compute_psi(self, expected: np.ndarray, actual: np.ndarray,
                    n_bins: int = 10) -> float:
        eps = 1e-6
        expected_hist, bin_edges = np.histogram(expected, bins=n_bins, density=True)
        actual_hist, _ = np.histogram(actual, bins=bin_edges, density=True)
        expected_pct = np.clip(expected_hist / expected_hist.sum(), eps, None)
        actual_pct = np.clip(actual_hist / actual_hist.sum(), eps, None)
        psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
        return psi

    def compute_ks_statistic(self, reference: np.ndarray, current: np.ndarray) -> float:
        stat, _ = scipy_stats.ks_2samp(reference, current)
        return float(stat)

    def check_drift(self) -> list[DriftAlert]:
        alerts = []
        now = time.time()

        for feature, ref_data in self._reference_data.items():
            if feature not in self._current_windows:
                continue
            current_data = np.array(list(self._current_windows[feature]))
            if len(current_data) < 30:
                continue

            psi = self.compute_psi(ref_data, current_data)
            if psi > self.psi_threshold:
                severity = "critical" if psi > self.psi_threshold * 2 else "warning"
                alert = DriftAlert(
                    feature=feature, metric="PSI", value=psi,
                    threshold=self.psi_threshold, timestamp=now, severity=severity,
                )
                alerts.append(alert)

            ks_stat = self.compute_ks_statistic(ref_data, current_data)
            if ks_stat > self.ks_threshold:
                severity = "critical" if ks_stat > self.ks_threshold * 2 else "warning"
                alert = DriftAlert(
                    feature=feature, metric="KS", value=ks_stat,
                    threshold=self.ks_threshold, timestamp=now, severity=severity,
                )
                alerts.append(alert)

            ref_mean, ref_std = float(np.mean(ref_data)), float(np.std(ref_data))
            cur_mean, cur_std = float(np.mean(current_data)), float(np.std(current_data))
            mean_shift = abs(cur_mean - ref_mean) / (ref_std + 1e-6)
            if mean_shift > 2.0:
                alert = DriftAlert(
                    feature=feature, metric="mean_shift", value=mean_shift,
                    threshold=2.0, timestamp=now, severity="warning",
                )
                alerts.append(alert)

        self._drift_alerts.extend(alerts)
        return alerts

    def get_drift_summary(self) -> dict[str, Any]:
        feature_drift = {}
        for alert in self._drift_alerts:
            if alert.feature not in feature_drift:
                feature_drift[alert.feature] = {"psi": [], "ks": [], "mean_shift": []}
            feature_drift[alert.feature][alert.metric].append(alert.value)

        summary = {}
        for feature, metrics in feature_drift.items():
            summary[feature] = {
                metric: {"max": max(values), "count": len(values)}
                for metric, values in metrics.items() if values
            }

        return {
            "total_alerts": len(self._drift_alerts),
            "features_monitored": len(self._reference_data),
            "features_with_drift": len(summary),
            "feature_details": summary,
        }

    def get_current_stats(self) -> dict[str, dict[str, float]]:
        stats = {}
        for feature, window in self._current_windows.items():
            if len(window) > 0:
                data = np.array(list(window))
                stats[feature] = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "count": len(data),
                }
        return stats

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "reference": {k: v.tolist() for k, v in self._reference_data.items()},
            "drift_alerts": [a.to_dict() for a in self._drift_alerts],
            "config": {
                "window_size": self.window_size,
                "psi_threshold": self.psi_threshold,
                "ks_threshold": self.ks_threshold,
            },
        }
        with open(path / "drift_monitor.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> DataDriftMonitor:
        with open(path / "drift_monitor.json") as f:
            data = json.load(f)
        config = data.get("config", {})
        monitor = cls(
            window_size=config.get("window_size", 1000),
            psi_threshold=config.get("psi_threshold", 0.2),
            ks_threshold=config.get("ks_threshold", 0.05),
        )
        monitor._reference_data = {k: np.array(v) for k, v in data.get("reference", {}).items()}
        for k in monitor._reference_data:
            monitor._current_windows[k] = deque(maxlen=monitor.window_size)
        for alert_data in data.get("drift_alerts", []):
            monitor._drift_alerts.append(DriftAlert(
                feature=alert_data["feature"],
                metric=alert_data["metric"],
                value=alert_data["value"],
                threshold=alert_data["threshold"],
                timestamp=alert_data["timestamp"],
                severity=alert_data.get("severity", "warning"),
            ))
        return monitor

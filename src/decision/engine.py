"""Unified decision engine combining physical and cyber scores."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.api.schemas import RiskLevel, UnifiedRiskResult
from src.config.loader import load_config

logger = logging.getLogger(__name__)


@dataclass
class DecisionEngine:
    """Central decision engine combining physical anomaly and cyber threat scores."""

    physical_weight: float = 0.5
    cyber_weight: float = 0.5
    low_threshold: float = 0.3
    medium_threshold: float = 0.6
    high_threshold: float = 0.85

    def __post_init__(self):
        try:
            config = load_config("risk")
            self.low_threshold = config.thresholds.low
            self.medium_threshold = config.thresholds.medium
            self.high_threshold = config.thresholds.high
        except Exception:
            pass

    def compute_risk(
        self,
        physical_score: float = 0.0,
        cyber_score: float = 0.0,
        device_health: float = 100.0,
        model_confidence: float = 0.0,
    ) -> UnifiedRiskResult:
        health_factor = device_health / 100.0
        adjusted_physical = physical_score * health_factor

        # Renormalize weights over domains that produced a score, so a single
        # evaluated domain can express its full severity (unified == domain).
        active: list[tuple[float, float]] = []
        if adjusted_physical > 0:
            active.append((self.physical_weight, adjusted_physical))
        if cyber_score > 0:
            active.append((self.cyber_weight, cyber_score))

        if active:
            total_weight = sum(w for w, _ in active)
            unified_score = sum(w * s for w, s in active) / total_weight
        else:
            unified_score = 0.0
        unified_score = min(1.0, max(0.0, unified_score))

        risk_level = self._score_to_level(unified_score)
        action = self._recommend_action(risk_level, physical_score, cyber_score)

        return UnifiedRiskResult(
            device_id="",
            timestamp="",
            physical_anomaly_score=physical_score,
            cyber_threat_score=cyber_score,
            unified_risk_score=unified_score,
            risk_level=risk_level,
            recommended_action=action,
            device_health_score=device_health,
        )

    def _score_to_level(self, score: float) -> RiskLevel:
        if score >= self.high_threshold:
            return RiskLevel.CRITICAL
        elif score >= self.medium_threshold:
            return RiskLevel.HIGH
        elif score >= self.low_threshold:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _recommend_action(self, level: RiskLevel, physical: float, cyber: float) -> str:
        actions = {
            RiskLevel.LOW: "MONITOR",
            RiskLevel.MEDIUM: "ALERT_ADMIN",
            RiskLevel.HIGH: "ISOLATE_DEVICE",
            RiskLevel.CRITICAL: "BLOCK_AND_ALERT",
        }
        return actions.get(level, "MONITOR")

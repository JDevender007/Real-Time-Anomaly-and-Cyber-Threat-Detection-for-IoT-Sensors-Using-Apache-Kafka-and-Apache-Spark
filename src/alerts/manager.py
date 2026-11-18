"""Alert manager for generating and managing alerts."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from src.api.schemas import Alert, AlertStatus, AlertType, RiskLevel

logger = logging.getLogger(__name__)


class AlertManager:
    """Centralized alert generation and management."""

    def __init__(self) -> None:
        self._alerts: list[Alert] = []
        self._alert_handlers: list[Any] = []

    def register_handler(self, handler) -> None:
        self._alert_handlers.append(handler)

    def create_alert(
        self,
        device_id: str,
        alert_type: AlertType,
        severity: RiskLevel,
        physical_score: float = 0.0,
        cyber_score: float = 0.0,
        risk_score: float = 0.0,
        model_name: str = "",
        predicted_class: str = "",
        confidence: float = 0.0,
        explanation: str = "",
    ) -> Alert:
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            device_id=device_id,
            alert_type=alert_type,
            severity=severity,
            physical_score=physical_score,
            cyber_score=cyber_score,
            risk_score=risk_score,
            model_name=model_name,
            predicted_class=predicted_class,
            confidence=confidence,
            explanation=explanation,
            status=AlertStatus.NEW,
        )
        self._alerts.append(alert)
        logger.info("Alert created: %s for device %s (severity=%s)", alert.alert_id, device_id, severity.value)

        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error("Alert handler error: %s", e)

        return alert

    def get_alerts(self, device_id: str | None = None,
                   severity: RiskLevel | None = None,
                   status: AlertStatus | None = None) -> list[Alert]:
        results = self._alerts
        if device_id:
            results = [a for a in results if a.device_id == device_id]
        if severity:
            results = [a for a in results if a.severity == severity]
        if status:
            results = [a for a in results if a.status == status]
        return results

    def update_status(self, alert_id: str, status: AlertStatus) -> bool:
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.status = status
                return True
        return False

    def get_statistics(self) -> dict[str, Any]:
        total = len(self._alerts)
        by_severity = {}
        by_status = {}
        for a in self._alerts:
            by_severity[a.severity.value] = by_severity.get(a.severity.value, 0) + 1
            by_status[a.status.value] = by_status.get(a.status.value, 0) + 1
        return {
            "total": total,
            "by_severity": by_severity,
            "by_status": by_status,
        }

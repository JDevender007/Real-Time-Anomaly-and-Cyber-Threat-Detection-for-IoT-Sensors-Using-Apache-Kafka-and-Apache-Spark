"""Database repository layer."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from src.database.models import (
    Device, SensorReading, NetworkFlow, PhysicalAnomaly,
    CyberThreat, Alert, ModelPrediction, SystemMetric, AttackEvent,
    get_session, init_db,
)
from src.api.schemas import RiskLevel

logger = logging.getLogger(__name__)


class DeviceRepository:
    def __init__(self):
        init_db()

    def upsert(self, device_id: str, device_type: str = "iot_sensor",
               status: str = "online", health_score: float = 100.0) -> Device:
        session = get_session()
        try:
            device = session.query(Device).filter_by(device_id=device_id).first()
            if device:
                device.status = status
                device.health_score = health_score
                device.last_seen = datetime.now(timezone.utc)
            else:
                device = Device(
                    device_id=device_id, device_type=device_type,
                    status=status, health_score=health_score,
                )
                session.add(device)
            session.commit()
            session.refresh(device)
            return device
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_all(self) -> list[dict[str, Any]]:
        session = get_session()
        try:
            devices = session.query(Device).all()
            return [
                {"device_id": d.device_id, "status": d.status,
                 "health_score": d.health_score, "last_seen": str(d.last_seen)}
                for d in devices
            ]
        finally:
            session.close()

    def get_by_id(self, device_id: str) -> dict[str, Any] | None:
        session = get_session()
        try:
            d = session.query(Device).filter_by(device_id=device_id).first()
            if d:
                return {"device_id": d.device_id, "status": d.status,
                        "health_score": d.health_score, "last_seen": str(d.last_seen)}
            return None
        finally:
            session.close()


class AlertRepository:
    def __init__(self):
        init_db()

    def create(self, alert_id: str, device_id: str, alert_type: str,
               severity: str, physical_score: float = 0.0, cyber_score: float = 0.0,
               risk_score: float = 0.0, model_name: str = "", predicted_class: str = "",
               confidence: float = 0.0, explanation: str = "") -> Alert:
        session = get_session()
        try:
            alert = Alert(
                alert_id=alert_id, device_id=device_id, alert_type=alert_type,
                severity=severity, physical_score=physical_score, cyber_score=cyber_score,
                risk_score=risk_score, model_name=model_name, predicted_class=predicted_class,
                confidence=confidence, explanation=explanation,
                timestamp=datetime.now(timezone.utc),
            )
            session.add(alert)
            session.commit()
            session.refresh(alert)
            return alert
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_all(self, limit: int = 100) -> list[dict[str, Any]]:
        session = get_session()
        try:
            alerts = session.query(Alert).order_by(Alert.id.desc()).limit(limit).all()
            return [
                {"alert_id": a.alert_id, "device_id": a.device_id,
                 "alert_type": a.alert_type, "severity": a.severity,
                 "risk_score": a.risk_score, "status": a.status,
                 "timestamp": str(a.timestamp)}
                for a in alerts
            ]
        finally:
            session.close()

    def update_status(self, alert_id: str, status: str) -> bool:
        session = get_session()
        try:
            alert = session.query(Alert).filter_by(alert_id=alert_id).first()
            if alert:
                alert.status = status
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


    def get_statistics(self) -> dict[str, Any]:
        session = get_session()
        try:
            total = session.query(Alert).count()
            by_severity = {}
            for sev in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                count = session.query(Alert).filter_by(severity=sev).count()
                by_severity[sev.lower()] = count
            by_status = {}
            for st in ["NEW", "ACKNOWLEDGED", "RESOLVED", "FALSE_POSITIVE"]:
                count = session.query(Alert).filter_by(status=st).count()
                by_status[st.lower()] = count
            return {
                "total": total,
                "by_severity": by_severity,
                "by_status": by_status,
            }
        finally:
            session.close()


class MetricRepository:
    def __init__(self):
        init_db()

    def record(self, metric_name: str, metric_value: float,
               metadata_json: str = "{}") -> None:
        session = get_session()
        try:
            metric = SystemMetric(
                metric_name=metric_name, metric_value=metric_value,
                metadata_json=metadata_json,
                timestamp=datetime.now(timezone.utc),
            )
            session.add(metric)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Failed to record metric: %s", e)
        finally:
            session.close()

    def get_recent(self, metric_name: str, limit: int = 100) -> list[dict[str, Any]]:
        session = get_session()
        try:
            metrics = (
                session.query(SystemMetric)
                .filter_by(metric_name=metric_name)
                .order_by(SystemMetric.id.desc())
                .limit(limit)
                .all()
            )
            return [
                {"metric_name": m.metric_name, "metric_value": m.metric_value,
                 "timestamp": str(m.timestamp)}
                for m in metrics
            ]
        finally:
            session.close()


class SensorReadingRepository:
    def __init__(self):
        init_db()

    def store(self, device_id: str, timestamp: datetime, **kwargs) -> None:
        session = get_session()
        try:
            reading = SensorReading(device_id=device_id, timestamp=timestamp, **kwargs)
            session.add(reading)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Failed to store sensor reading: %s", e)
        finally:
            session.close()


class PhysicalAnomalyRepository:
    def __init__(self):
        init_db()

    def store(self, device_id: str, timestamp: datetime, **kwargs) -> None:
        session = get_session()
        try:
            anomaly = PhysicalAnomaly(device_id=device_id, timestamp=timestamp, **kwargs)
            session.add(anomaly)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Failed to store physical anomaly: %s", e)
        finally:
            session.close()

    def get_all(self, limit: int = 100) -> list[dict[str, Any]]:
        session = get_session()
        try:
            items = session.query(PhysicalAnomaly).order_by(
                PhysicalAnomaly.id.desc()
            ).limit(limit).all()
            return [
                {"device_id": a.device_id, "timestamp": str(a.timestamp),
                 "combined_score": a.combined_score, "confidence": a.confidence}
                for a in items
            ]
        finally:
            session.close()


class CyberThreatRepository:
    def __init__(self):
        init_db()

    def store(self, device_id: str, timestamp: datetime, **kwargs) -> None:
        session = get_session()
        try:
            threat = CyberThreat(device_id=device_id, timestamp=timestamp, **kwargs)
            session.add(threat)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Failed to store cyber threat: %s", e)
        finally:
            session.close()

    def get_all(self, limit: int = 100) -> list[dict[str, Any]]:
        session = get_session()
        try:
            items = session.query(CyberThreat).order_by(
                CyberThreat.id.desc()
            ).limit(limit).all()
            return [
                {"device_id": t.device_id, "timestamp": str(t.timestamp),
                 "rf_prediction": t.rf_prediction, "xgb_prediction": t.xgb_prediction,
                 "combined_threat_score": t.combined_threat_score}
                for t in items
            ]
        finally:
            session.close()

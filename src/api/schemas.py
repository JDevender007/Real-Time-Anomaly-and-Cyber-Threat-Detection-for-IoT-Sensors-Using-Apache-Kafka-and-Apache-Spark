"""Pydantic schemas for Kafka messages and API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertStatus(str, Enum):
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class AlertType(str, Enum):
    PHYSICAL_ANOMALY = "PHYSICAL_ANOMALY"
    CYBER_THREAT = "CYBER_THREAT"
    UNIFIED_RISK = "UNIFIED_RISK"
    DEVICE_HEALTH = "DEVICE_HEALTH"


class PhysicalSensorMessage(BaseModel):
    device_id: str
    timestamp: str
    temperature: float = 0.0
    humidity: float = 0.0
    pressure: float = 0.0
    voltage: float = 0.0
    battery: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0


class NetworkMessage(BaseModel):
    device_id: str
    timestamp: str
    source_ip: str = "0.0.0.0"
    destination_ip: str = "0.0.0.0"
    source_port: int = 0
    destination_port: int = 0
    protocol: str = "TCP"
    packet_count: int = 0
    packet_rate: float = 0.0
    bytes: int = 0
    network_bytes: int = 0


class PhysicalAnomalyResult(BaseModel):
    device_id: str
    timestamp: str
    if_prediction: int = 0
    if_score: float = 0.0
    ae_prediction: int = 0
    ae_score: float = 0.0
    reconstruction_error: float = 0.0
    combined_score: float = 0.0
    confidence: float = 0.0


class CyberThreatResult(BaseModel):
    device_id: str
    timestamp: str
    rf_prediction: str = "Benign"
    rf_probability: float = 0.0
    rf_confidence: float = 0.0
    xgb_prediction: str = "Benign"
    xgb_probability: float = 0.0
    xgb_confidence: float = 0.0
    combined_threat_score: float = 0.0


class UnifiedRiskResult(BaseModel):
    device_id: str
    timestamp: str
    physical_anomaly_score: float = 0.0
    cyber_threat_score: float = 0.0
    unified_risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    recommended_action: str = "MONITOR"
    device_health_score: float = 100.0


class Alert(BaseModel):
    alert_id: str = ""
    timestamp: str = ""
    device_id: str = ""
    alert_type: AlertType = AlertType.PHYSICAL_ANOMALY
    severity: RiskLevel = RiskLevel.LOW
    physical_score: float = 0.0
    cyber_score: float = 0.0
    risk_score: float = 0.0
    model_name: str = ""
    predicted_class: str = ""
    confidence: float = 0.0
    explanation: str = ""
    status: AlertStatus = AlertStatus.NEW


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    kafka: str = "unknown"
    spark: str = "unknown"
    database: str = "unknown"
    models: dict[str, str] = {}


class DeviceInfo(BaseModel):
    device_id: str
    status: str = "online"
    health_score: float = 100.0
    last_seen: str = ""
    anomaly_count: int = 0
    threat_count: int = 0

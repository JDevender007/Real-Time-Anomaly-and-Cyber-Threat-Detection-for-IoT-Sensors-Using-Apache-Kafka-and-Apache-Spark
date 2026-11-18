"""Database models using SQLAlchemy."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime, Boolean, Index, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config.settings import DATABASE_URL


class Base(DeclarativeBase):
    pass


class Device(Base):
    __tablename__ = "devices"
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(64), unique=True, nullable=False, index=True)
    device_type = Column(String(32), default="iot_sensor")
    status = Column(String(16), default="online")
    health_score = Column(Float, default=100.0)
    last_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SensorReading(Base):
    __tablename__ = "sensor_readings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    temperature = Column(Float, default=0.0)
    humidity = Column(Float, default=0.0)
    pressure = Column(Float, default=0.0)
    voltage = Column(Float, default=0.0)
    battery = Column(Float, default=0.0)
    cpu_usage = Column(Float, default=0.0)
    memory_usage = Column(Float, default=0.0)


class NetworkFlow(Base):
    __tablename__ = "network_flows"
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    source_ip = Column(String(45), default="")
    destination_ip = Column(String(45), default="")
    source_port = Column(Integer, default=0)
    destination_port = Column(Integer, default=0)
    protocol = Column(String(16), default="TCP")
    packet_count = Column(Integer, default=0)
    packet_rate = Column(Float, default=0.0)
    bytes_transferred = Column(Integer, default=0)


class PhysicalAnomaly(Base):
    __tablename__ = "physical_anomalies"
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    if_prediction = Column(Integer, default=0)
    if_score = Column(Float, default=0.0)
    ae_prediction = Column(Integer, default=0)
    ae_score = Column(Float, default=0.0)
    reconstruction_error = Column(Float, default=0.0)
    combined_score = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0)


class CyberThreat(Base):
    __tablename__ = "cyber_threats"
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    rf_prediction = Column(String(64), default="Benign")
    rf_probability = Column(Float, default=0.0)
    xgb_prediction = Column(String(64), default="Benign")
    xgb_probability = Column(Float, default=0.0)
    combined_threat_score = Column(Float, default=0.0)


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(64), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    device_id = Column(String(64), nullable=False, index=True)
    alert_type = Column(String(32), nullable=False)
    severity = Column(String(16), nullable=False, index=True)
    physical_score = Column(Float, default=0.0)
    cyber_score = Column(Float, default=0.0)
    risk_score = Column(Float, default=0.0)
    model_name = Column(String(64), default="")
    predicted_class = Column(String(64), default="")
    confidence = Column(Float, default=0.0)
    explanation = Column(Text, default="")
    status = Column(String(16), default="NEW", index=True)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    model_name = Column(String(64), nullable=False)
    prediction = Column(String(64), default="")
    confidence = Column(Float, default=0.0)
    probability = Column(Float, default=0.0)


class SystemMetric(Base):
    __tablename__ = "system_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    metric_name = Column(String(64), nullable=False)
    metric_value = Column(Float, default=0.0)
    metadata_json = Column(Text, default="{}")


class AttackEvent(Base):
    __tablename__ = "attack_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    attack_type = Column(String(64), nullable=False, index=True)
    severity = Column(String(16), default="LOW", index=True)
    risk_score = Column(Float, default=0.0)
    details = Column(Text, default="")


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, echo=False)
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    factory = get_session_factory()
    return factory()

"""Unit tests for Kafka, decision engine, alerts, simulation, features, monitoring."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.kafka.serializers import serialize_message, deserialize_message, extract_payload
from src.decision.engine import DecisionEngine
from src.alerts.manager import AlertManager
from src.api.schemas import AlertType, RiskLevel, AlertStatus
from src.simulation.iot_simulator import VirtualIoTDevice, IoTSimulator
from src.simulation.attack_simulator import CyberAttackSimulator
from src.features.physical import (
    compute_rolling_features, compute_rate_of_change,
    compute_deviation_from_baseline, engineer_physical_features,
)
from src.features.cyber import compute_cyber_features, engineer_cyber_features
from src.monitoring.performance import PerformanceMonitor


class TestSerializers:
    def test_roundtrip(self):
        data = {"device_id": "d1", "temperature": 25.0}
        raw = serialize_message(data)
        result = deserialize_message(raw)
        assert result["payload"]["device_id"] == "d1"

    def test_extract_payload(self):
        envelope = {"event_id": "123", "payload": {"key": "value"}}
        assert extract_payload(envelope) == {"key": "value"}


class TestDecisionEngine:
    def test_low_risk(self):
        engine = DecisionEngine()
        result = engine.compute_risk(physical_score=0.1, cyber_score=0.1)
        assert result.risk_level == RiskLevel.LOW

    def test_high_risk(self):
        engine = DecisionEngine()
        result = engine.compute_risk(physical_score=0.9, cyber_score=0.9)
        assert result.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_unified_score_range(self):
        engine = DecisionEngine()
        result = engine.compute_risk(physical_score=0.5, cyber_score=0.5)
        assert 0.0 <= result.unified_risk_score <= 1.0


class TestAlertManager:
    def test_create_alert(self):
        mgr = AlertManager()
        alert = mgr.create_alert(
            device_id="d1",
            alert_type=AlertType.PHYSICAL_ANOMALY,
            severity=RiskLevel.HIGH,
        )
        assert alert.status == AlertStatus.NEW
        assert len(mgr.get_alerts()) == 1

    def test_filter_by_device(self):
        mgr = AlertManager()
        mgr.create_alert(device_id="d1", alert_type=AlertType.PHYSICAL_ANOMALY, severity=RiskLevel.LOW)
        mgr.create_alert(device_id="d2", alert_type=AlertType.CYBER_THREAT, severity=RiskLevel.HIGH)
        assert len(mgr.get_alerts(device_id="d1")) == 1

    def test_update_status(self):
        mgr = AlertManager()
        alert = mgr.create_alert(device_id="d1", alert_type=AlertType.PHYSICAL_ANOMALY, severity=RiskLevel.LOW)
        mgr.update_status(alert.alert_id, AlertStatus.ACKNOWLEDGED)
        updated = mgr.get_alerts(device_id="d1")[0]
        assert updated.status == AlertStatus.ACKNOWLEDGED

    def test_statistics(self):
        mgr = AlertManager()
        mgr.create_alert(device_id="d1", alert_type=AlertType.PHYSICAL_ANOMALY, severity=RiskLevel.LOW)
        mgr.create_alert(device_id="d1", alert_type=AlertType.CYBER_THREAT, severity=RiskLevel.CRITICAL)
        stats = mgr.get_statistics()
        assert stats["total"] == 2


class TestIoTSimulator:
    def test_device_generation(self):
        device = VirtualIoTDevice("device_001")
        reading = device.generate_sensor_reading()
        assert reading.device_id == "device_001"
        assert reading.temperature > 0

    def test_network_event(self):
        device = VirtualIoTDevice("device_001")
        event = device.generate_network_event()
        assert event.device_id == "device_001"
        assert event.protocol in ("TCP", "UDP")

    def test_simulator_creates_devices(self):
        sim = IoTSimulator(num_devices=5)
        assert len(sim.devices) == 5

    def test_simulator_generate_once(self):
        sim = IoTSimulator(num_devices=3)
        sensors, network = sim.generate_once()
        assert len(sensors) == 3
        assert len(network) == 3


class TestAttackSimulator:
    def test_ddos(self):
        sim = CyberAttackSimulator()
        msg = sim.simulate_ddoS("d1")
        assert msg.packet_count > 100

    def test_spoofing(self):
        sim = CyberAttackSimulator()
        msg = sim.simulate_spoofing("d1")
        assert "192.168" in msg.source_ip

    def test_brute_force(self):
        sim = CyberAttackSimulator()
        msg = sim.simulate_brute_force("d1")
        assert msg.destination_port == 22

    def test_port_scan(self):
        sim = CyberAttackSimulator()
        msg = sim.simulate_port_scan("d1")
        assert msg.packet_count < 100

    def test_random_attack(self):
        sim = CyberAttackSimulator()
        attack_type, msg = sim.generate_random_attack("d1")
        assert attack_type in CyberAttackSimulator.ATTACK_TYPES


class TestPhysicalFeatures:
    def test_rolling_features(self):
        df = pd.DataFrame({"temp": [1, 2, 3, 4, 5]})
        result = compute_rolling_features(df, ["temp"], window=3)
        assert "temp_rolling_mean" in result.columns
        assert "temp_rolling_std" in result.columns

    def test_rate_of_change(self):
        df = pd.DataFrame({"temp": [1, 2, 4, 7]})
        result = compute_rate_of_change(df, ["temp"])
        assert "temp_roc" in result.columns
        assert result["temp_roc"].iloc[1] == 1.0

    def test_deviation(self):
        df = pd.DataFrame({"temp": [10, 20, 30]})
        result = compute_deviation_from_baseline(df, ["temp"], {"temp": 20})
        assert result["temp_deviation"].iloc[0] == -10.0

    def test_engineer_physical_features(self):
        df = pd.DataFrame({"temp": range(20), "humidity": range(20)})
        result = engineer_physical_features(df, ["temp", "humidity"])
        assert result.shape[1] > 2


class TestCyberFeatures:
    def test_compute_cyber_features(self):
        df = pd.DataFrame({
            "bytes": [1000, 2000],
            "packet_count": [10, 20],
            "flow_duration": [5.0, 10.0],
        })
        result = compute_cyber_features(df)
        assert "bytes_per_packet" in result.columns
        assert "packets_per_second" in result.columns
        assert result["bytes_per_packet"].iloc[0] == 100.0


class TestPerformanceMonitor:
    def test_latency_tracking(self):
        mon = PerformanceMonitor()
        mon.record_latency(10.0)
        mon.record_latency(20.0)
        stats = mon.get_latency_stats()
        assert stats["avg_ms"] == 15.0

    def test_message_count(self):
        mon = PerformanceMonitor()
        mon.record_message()
        mon.record_message()
        summary = mon.get_summary()
        assert summary["total_messages"] == 2

"""Unit tests for configuration and schemas."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config.settings import (
    PROJECT_ROOT, DATA_ROOT, MODEL_ROOT, RANDOM_SEED,
    PHYSICAL_SENSORS, CYBER_FEATURES, create_required_directories,
)
from src.config.loader import load_config, ConfigNamespace
from src.api.schemas import (
    PhysicalSensorMessage, NetworkMessage, PhysicalAnomalyResult,
    CyberThreatResult, UnifiedRiskResult, Alert, RiskLevel, AlertStatus,
)
from src.utils.exceptions import (
    IoTSecurityError, DatasetLoadError, ModelLoadError, KafkaConnectionError,
)
from src.utils.helpers import generate_id, utc_now, safe_divide, chunk_list


class TestSettings:
    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_data_root_exists(self):
        assert DATA_ROOT.exists()

    def test_model_root_exists(self):
        assert MODEL_ROOT.exists()

    def test_random_seed_default(self):
        assert RANDOM_SEED == 42

    def test_physical_sensors_defined(self):
        assert len(PHYSICAL_SENSORS) == 5
        assert "temperature" in PHYSICAL_SENSORS

    def test_cyber_features_defined(self):
        assert len(CYBER_FEATURES) == 12
        assert "packet_count" in CYBER_FEATURES

    def test_create_directories(self):
        create_required_directories()
        assert DATA_ROOT.exists()
        assert MODEL_ROOT.exists()


class TestConfigLoader:
    def test_load_app_config(self):
        config = load_config("app")
        assert isinstance(config, ConfigNamespace)
        assert config.project.name == "iot-cyber-physical-security"

    def test_load_kafka_config(self):
        config = load_config("kafka")
        assert config.kafka.bootstrap_servers == "localhost:9092"

    def test_config_namespace_get(self):
        config = load_config("app")
        assert config.get("nonexistent", "default") == "default"

    def test_config_to_dict(self):
        config = load_config("app")
        d = config.to_dict()
        assert isinstance(d, dict)


class TestSchemas:
    def test_physical_sensor_message(self):
        msg = PhysicalSensorMessage(device_id="d1", timestamp="2026-01-01T00:00:00Z")
        assert msg.device_id == "d1"
        assert msg.temperature == 0.0

    def test_network_message(self):
        msg = NetworkMessage(device_id="d1", timestamp="2026-01-01T00:00:00Z")
        assert msg.protocol == "TCP"

    def test_physical_anomaly_result(self):
        result = PhysicalAnomalyResult(device_id="d1", timestamp="2026-01-01T00:00:00Z")
        assert result.if_prediction == 0
        assert result.ae_prediction == 0

    def test_cyber_threat_result(self):
        result = CyberThreatResult(device_id="d1", timestamp="2026-01-01T00:00:00Z")
        assert result.rf_prediction == "Benign"
        assert result.xgb_prediction == "Benign"

    def test_unified_risk_result(self):
        result = UnifiedRiskResult(device_id="d1", timestamp="2026-01-01T00:00:00Z")
        assert result.risk_level == RiskLevel.LOW
        assert result.device_health_score == 100.0

    def test_alert(self):
        alert = Alert(alert_id="a1", device_id="d1")
        assert alert.status == AlertStatus.NEW

    def test_risk_level_enum(self):
        assert RiskLevel.LOW == "LOW"
        assert RiskLevel.CRITICAL == "CRITICAL"


class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(DatasetLoadError, IoTSecurityError)
        assert issubclass(ModelLoadError, IoTSecurityError)
        assert issubclass(KafkaConnectionError, IoTSecurityError)

    def test_raise_catch(self):
        with pytest.raises(IoTSecurityError):
            raise DatasetLoadError("test")


class TestHelpers:
    def test_generate_id(self):
        id1 = generate_id()
        id2 = generate_id()
        assert id1 != id2

    def test_utc_now(self):
        now = utc_now()
        assert now.tzinfo is not None

    def test_safe_divide(self):
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=-1) == -1

    def test_chunk_list(self):
        result = chunk_list([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

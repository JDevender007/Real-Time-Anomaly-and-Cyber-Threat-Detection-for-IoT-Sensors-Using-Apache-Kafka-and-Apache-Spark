"""Unit tests for database repositories."""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.database.models import init_db, get_session, Base, get_engine
from src.database.repositories import (
    DeviceRepository, AlertRepository, MetricRepository,
)


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    import src.database.models as models
    models._engine = None
    models._SessionLocal = None
    from src.config import settings
    settings.DATABASE_URL = f"sqlite:///{tmp_path / 'test.db'}"
    models._engine = None
    models._SessionLocal = None
    init_db()
    yield
    if models._SessionLocal:
        models._SessionLocal().close()
    models._engine = None
    models._SessionLocal = None


class TestDeviceRepository:
    def test_upsert_new(self):
        repo = DeviceRepository()
        device = repo.upsert("device_001", "iot_sensor", "online", 95.0)
        assert device.device_id == "device_001"
        assert device.health_score == 95.0

    def test_upsert_existing(self):
        repo = DeviceRepository()
        repo.upsert("device_001", "iot_sensor", "online", 95.0)
        updated = repo.upsert("device_001", "iot_sensor", "offline", 50.0)
        assert updated.health_score == 50.0

    def test_get_all(self):
        import uuid
        repo = DeviceRepository()
        id1 = f"dev_{uuid.uuid4().hex[:8]}"
        id2 = f"dev_{uuid.uuid4().hex[:8]}"
        repo.upsert(id1)
        repo.upsert(id2)
        devices = repo.get_all()
        assert len(devices) >= 2

    def test_get_by_id(self):
        repo = DeviceRepository()
        repo.upsert("device_001")
        device = repo.get_by_id("device_001")
        assert device is not None
        assert device["device_id"] == "device_001"

    def test_get_by_id_missing(self):
        repo = DeviceRepository()
        device = repo.get_by_id("nonexistent")
        assert device is None


class TestAlertRepository:
    def test_create_alert(self):
        import uuid
        repo = AlertRepository()
        uid = f"alert_{uuid.uuid4().hex[:8]}"
        alert = repo.create(
            alert_id=uid, device_id="d1",
            alert_type="PHYSICAL_ANOMALY", severity="HIGH",
        )
        assert alert.alert_id == uid

    def test_get_all(self):
        import uuid
        repo = AlertRepository()
        id1 = f"a1_{uuid.uuid4().hex[:8]}"
        id2 = f"a2_{uuid.uuid4().hex[:8]}"
        repo.create(id1, "d1", "PHYSICAL_ANOMALY", "LOW")
        repo.create(id2, "d2", "CYBER_THREAT", "CRITICAL")
        alerts = repo.get_all()
        assert len(alerts) >= 2

    def test_update_status(self):
        import uuid
        repo = AlertRepository()
        uid = f"a1_{uuid.uuid4().hex[:8]}"
        repo.create(uid, "d1", "PHYSICAL_ANOMALY", "LOW")
        result = repo.update_status(uid, "ACKNOWLEDGED")
        assert result is True

    def test_update_nonexistent(self):
        repo = AlertRepository()
        result = repo.update_status("nonexistent", "ACKNOWLEDGED")
        assert result is False


class TestMetricRepository:
    def test_record(self):
        repo = MetricRepository()
        repo.record("kafka_throughput_metric_test", 123.45)
        metrics = repo.get_recent("kafka_throughput_metric_test")
        assert len(metrics) >= 1
        assert metrics[0]["metric_value"] == 123.45

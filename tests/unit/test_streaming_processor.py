"""Unit tests for the streaming processor."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.streaming.processor import StreamingProcessor, PipelineStats


@pytest.fixture
def processor(tmp_path):
    return StreamingProcessor(model_dir=tmp_path)


class TestPipelineStats:
    def test_initial_state(self):
        stats = PipelineStats()
        assert stats.total_processed == 0
        assert stats.avg_latency_ms == 0.0

    def test_to_dict(self):
        stats = PipelineStats(total_processed=10, errors=2)
        d = stats.to_dict()
        assert d["total_processed"] == 10
        assert d["errors"] == 2

    def test_avg_latency(self):
        stats = PipelineStats(total_processed=100, total_latency_ms=500.0)
        assert stats.avg_latency_ms == 5.0


class TestStreamingProcessor:
    def test_init(self, processor):
        assert processor.stats.total_processed == 0

    def test_process_sensor_message(self, processor):
        data = {
            "device_id": "device_001",
            "timestamp": "2026-09-07T12:00:00Z",
            "temperature": 25.0,
            "humidity": 55.0,
            "pressure": 1013.0,
            "voltage": 3.7,
            "battery": 85.0,
        }
        result = processor.process_sensor_message(data)
        assert result is not None
        assert result["device_id"] == "device_001"
        assert "physical" in result
        assert "risk" in result
        assert "latency_ms" in result
        assert processor.stats.total_processed == 1

    def test_process_network_message(self, processor):
        data = {
            "device_id": "device_001",
            "timestamp": "2026-09-07T12:00:00Z",
            "source_ip": "10.0.0.10",
            "destination_ip": "10.0.0.20",
            "source_port": 4500,
            "destination_port": 80,
            "protocol": "TCP",
            "packet_count": 120,
            "packet_rate": 25.4,
            "bytes": 15400,
        }
        result = processor.process_network_message(data)
        assert result is not None
        assert result["device_id"] == "device_001"
        assert "cyber" in result
        assert processor.stats.total_processed == 1

    def test_process_message_routes_to_sensor(self, processor):
        data = {
            "device_id": "d1",
            "timestamp": "2026-01-01T00:00:00Z",
            "temperature": 25.0,
            "humidity": 50.0,
            "pressure": 1013.0,
            "voltage": 3.7,
            "battery": 80.0,
        }
        result = processor.process_message(data)
        assert result is not None
        assert "physical" in result

    def test_process_message_routes_to_network(self, processor):
        data = {
            "device_id": "d1",
            "timestamp": "2026-01-01T00:00:00Z",
            "source_ip": "10.0.0.1",
            "destination_ip": "10.0.0.2",
            "source_port": 1234,
            "destination_port": 80,
            "protocol": "TCP",
            "packet_count": 50,
            "packet_rate": 10.0,
            "bytes": 5000,
        }
        result = processor.process_message(data)
        assert result is not None
        assert "cyber" in result

    def test_process_message_with_payload(self, processor):
        data = {
            "payload": {
                "device_id": "d1",
                "timestamp": "2026-01-01T00:00:00Z",
                "temperature": 25.0,
                "humidity": 50.0,
                "pressure": 1013.0,
                "voltage": 3.7,
                "battery": 80.0,
            }
        }
        result = processor.process_message(data)
        assert result is not None

    def test_process_unknown_message(self, processor):
        data = {"unknown_field": "value"}
        result = processor.process_message(data)
        assert result is None

    def test_get_stats(self, processor):
        processor.process_sensor_message({
            "device_id": "d1", "timestamp": "2026-01-01T00:00:00Z",
            "temperature": 25.0, "humidity": 50.0, "pressure": 1013.0,
            "voltage": 3.7, "battery": 80.0,
        })
        stats = processor.get_stats()
        assert stats["total_processed"] == 1

    def test_reset_stats(self, processor):
        processor.process_sensor_message({
            "device_id": "d1", "timestamp": "2026-01-01T00:00:00Z",
            "temperature": 25.0, "humidity": 50.0, "pressure": 1013.0,
            "voltage": 3.7, "battery": 80.0,
        })
        processor.reset_stats()
        assert processor.stats.total_processed == 0

    def test_latency_measured(self, processor):
        result = processor.process_sensor_message({
            "device_id": "d1", "timestamp": "2026-01-01T00:00:00Z",
            "temperature": 25.0, "humidity": 50.0, "pressure": 1013.0,
            "voltage": 3.7, "battery": 80.0,
        })
        assert result["latency_ms"] >= 0.0

    def test_handles_invalid_data_gracefully(self, processor):
        result = processor.process_sensor_message({
            "device_id": "d1",
            "timestamp": "2026-01-01T00:00:00Z",
        })
        assert result is not None

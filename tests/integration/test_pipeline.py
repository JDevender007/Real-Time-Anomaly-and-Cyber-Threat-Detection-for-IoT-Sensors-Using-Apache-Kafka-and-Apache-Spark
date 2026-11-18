"""Integration tests for the IoT security platform.

Tests the full pipeline without requiring external services (Kafka, Spark, PostgreSQL).
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config.settings import RANDOM_SEED, MODEL_ROOT, DATA_ROOT
from src.config.loader import load_config
from src.datasets.swat import SWaTDatasetLoader
from src.datasets.wadi import WADIDatasetLoader
from src.datasets.ciciot2023 import CICIoT2023DatasetLoader
from src.datasets.edge_iiotset import EdgeIIoTSetDatasetLoader
from src.datasets.base import BaseDatasetLoader
from src.datasets.registry import DatasetRegistry
from src.preprocessing.physical import PhysicalPreprocessor
from src.preprocessing.cyber import CyberPreprocessor
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost import XGBoostModel
from src.decision.engine import DecisionEngine
from src.alerts.manager import AlertManager
from src.monitoring.performance import PerformanceMonitor
from src.streaming.processor import StreamingProcessor
from src.database.repositories import DeviceRepository
from src.api.schemas import RiskLevel, AlertType, AlertStatus


@pytest.fixture
def sample_swat_df():
    rng = np.random.RandomState(42)
    n = 2000
    labels = np.zeros(n, dtype=int)
    labels[500:600] = 1
    labels[1500:1550] = 1
    return pd.DataFrame({
        "Timestamp": pd.date_range("2015-12-22", periods=n, freq="s"),
        "FIT101": rng.normal(2.5, 0.3, n),
        "LIT101": rng.normal(500, 20, n),
        "FIT102": rng.normal(2.4, 0.3, n),
        "LIT102": rng.normal(300, 15, n),
        "FIT103": rng.normal(1.2, 0.2, n),
        "LIT103": rng.normal(600, 25, n),
        "FIT104": rng.normal(2.3, 0.3, n),
        "LIT104": rng.normal(400, 18, n),
        "P101": rng.choice([0, 1], n),
        "P102": rng.choice([0, 1], n),
        "MV101": rng.choice([0, 1], n),
        "MV201": rng.choice([0, 1], n),
        "MV301": rng.choice([0, 1], n),
        "MV401": rng.choice([0, 1], n),
        "P101_Freq": rng.uniform(45, 55, n),
        "P102_Freq": rng.uniform(45, 55, n),
        "attack_label": labels,
    })


@pytest.fixture
def sample_cyber_df():
    rng = np.random.RandomState(42)
    n = 2000
    labels = rng.choice(
        ["Benign", "DDoS", "DoS", "BruteForce"],
        n, p=[0.5, 0.2, 0.2, 0.1],
    )
    return pd.DataFrame({
        "flow_duration": np.abs(rng.exponential(5, n)),
        "packet_count": rng.randint(1, 10000, n),
        "byte_count": rng.randint(100, 1000000, n),
        "packet_rate": np.abs(rng.exponential(50, n)),
        "byte_rate": np.abs(rng.exponential(5000, n)),
        "flow_iat_mean": np.abs(rng.exponential(1, n)),
        "flow_iat_std": np.abs(rng.exponential(0.5, n)),
        "fwd_iat_mean": np.abs(rng.exponential(1.2, n)),
        "bwd_iat_mean": np.abs(rng.exponential(1.5, n)),
        "fwd_pkt_len_mean": rng.uniform(20, 1500, n),
        "bwd_pkt_len_mean": rng.uniform(20, 1500, n),
        "fwd_header_len": rng.randint(20, 200, n),
        "bwd_header_len": rng.randint(20, 200, n),
        "src_port": rng.randint(1024, 65535, n),
        "dst_port": rng.choice([80, 443, 22, 53, 8080], n),
        "attack_label": labels,
    })


class TestDatasetIntegration:
    def test_swat_load_and_clean(self, sample_swat_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "SWaT-A2.csv"
            sample_swat_df.to_csv(csv_path, index=False)
            loader = SWaTDatasetLoader(Path(tmpdir))
            df = loader.load()
            assert len(df) == 2000
            df = loader.clean(df)
            assert "attack_label" in df.columns
            df = loader.transform(df)
            features = loader.get_features()
            assert len(features) > 0
            assert loader.get_labels() == ["attack_label"]

    def test_ciciot_load_and_clean(self, sample_cyber_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "3-combined.csv"
            sample_cyber_df.to_csv(csv_path, index=False)
            loader = CICIoT2023DatasetLoader(Path(tmpdir))
            df = loader.load()
            assert len(df) == 2000
            df = loader.clean(df)
            assert "attack_label" in df.columns
            df = loader.transform(df)
            features = loader.get_features()
            assert len(features) > 0
            assert loader.get_labels() == ["attack_label"]

    def test_registry_integration(self):
        registry = DatasetRegistry()
        registry.register("swat", SWaTDatasetLoader(DATA_ROOT / "raw" / "swat"))
        registry.register("ciciot", CICIoT2023DatasetLoader(DATA_ROOT / "raw" / "ciciot2023"))
        datasets = registry.list_datasets()
        assert len(datasets) >= 2
        assert "swat" in datasets


class TestPhysicalMLIntegration:
    def test_if_pipeline(self, sample_swat_df):
        features = [c for c in sample_swat_df.columns
                    if sample_swat_df[c].dtype in ("float64", "int64")
                    and c != "attack_label"]
        X = sample_swat_df[features].values
        y = sample_swat_df["attack_label"].values

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        preprocessor = PhysicalPreprocessor()
        X_train_df = pd.DataFrame(X_train, columns=features)
        X_test_df = pd.DataFrame(X_test, columns=features)
        X_train_scaled = preprocessor.fit_transform(X_train_df, features)
        X_test_scaled = preprocessor.transform(X_test_df)

        model = IsolationForestModel(n_estimators=100, contamination=0.1, random_state=42)
        model.fit(X_train_scaled, features)
        predictions, scores = model.predict(X_test_scaled)
        assert len(predictions) == len(X_test_scaled)
        assert len(scores) == len(X_test_scaled)

        metrics = model.evaluate(X_test_scaled, y_test)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir))
            loaded = IsolationForestModel.load(Path(tmpdir))
            preds2, scores2 = loaded.predict(X_test_scaled)
            assert len(preds2) == len(X_test_scaled)

    def test_ae_pipeline(self, sample_swat_df):
        features = [c for c in sample_swat_df.columns
                    if sample_swat_df[c].dtype in ("float64", "int64")
                    and c != "attack_label"][:6]
        X = sample_swat_df[features].values
        y = sample_swat_df["attack_label"].values

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        preprocessor = PhysicalPreprocessor()
        X_train_df = pd.DataFrame(X_train, columns=features)
        X_test_df = pd.DataFrame(X_test, columns=features)
        X_train_scaled = preprocessor.fit_transform(X_train_df, features)
        X_test_scaled = preprocessor.transform(X_test_df)

        model = AutoencoderModel(
            input_dim=len(features), encoding_dim=8,
            hidden_dims=[16], epochs=5, batch_size=64,
        )
        model.fit(X_train_scaled, features)
        metrics = model.evaluate(X_test_scaled, y_test)
        assert "accuracy" in metrics
        assert "f1" in metrics


class TestCyberMLIntegration:
    def test_rf_pipeline(self, sample_cyber_df):
        features = [c for c in sample_cyber_df.columns
                    if sample_cyber_df[c].dtype in ("float64", "int64")
                    and c != "attack_label"]
        preprocessor = CyberPreprocessor()
        df_scaled, y_encoded = preprocessor.fit_transform(
            sample_cyber_df, features, "attack_label",
        )
        X = df_scaled
        y = y_encoded

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )

        model = RandomForestModel(n_estimators=50, random_state=42)
        model.fit(X_train, y_train, feature_names=features,
                  label_classes=preprocessor.label_classes)
        metrics = model.evaluate(X_test, y_test)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_xgb_pipeline(self, sample_cyber_df):
        features = [c for c in sample_cyber_df.columns
                    if sample_cyber_df[c].dtype in ("float64", "int64")
                    and c != "attack_label"]
        preprocessor = CyberPreprocessor()
        df_scaled, y_encoded = preprocessor.fit_transform(
            sample_cyber_df, features, "attack_label",
        )
        X = df_scaled
        y = y_encoded

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )

        model = XGBoostModel(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X_train, y_train, feature_names=features,
                  label_classes=preprocessor.label_classes)
        metrics = model.evaluate(X_test, y_test)
        assert "accuracy" in metrics
        assert "f1" in metrics


class TestDecisionEngineIntegration:
    def test_full_decision_flow(self):
        engine = DecisionEngine()
        result = engine.compute_risk(
            physical_score=0.8,
            cyber_score=0.6,
            device_health=80.0,
            model_confidence=0.9,
        )
        assert hasattr(result, "risk_level")
        assert hasattr(result, "unified_risk_score")
        assert result.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_various_scores(self):
        engine = DecisionEngine()
        for phys, cyber in [(0.0, 0.0), (0.3, 0.3), (0.5, 0.5), (0.7, 0.7), (0.9, 0.9)]:
            result = engine.compute_risk(physical_score=phys, cyber_score=cyber)
            assert hasattr(result, "risk_level")
            assert hasattr(result, "unified_risk_score")


class TestAlertManagerIntegration:
    def test_create_and_query_alerts(self):
        manager = AlertManager()
        alert = manager.create_alert(
            device_id="sensor_001",
            alert_type=AlertType.PHYSICAL_ANOMALY,
            severity=RiskLevel.HIGH,
            physical_score=0.9,
            cyber_score=0.0,
            risk_score=0.9,
            model_name="isolation_forest",
            predicted_class="anomaly",
            confidence=0.95,
            explanation="Temperature exceeded threshold",
        )
        assert alert.alert_id is not None
        assert alert.device_id == "sensor_001"
        alerts = manager.get_alerts(device_id="sensor_001")
        assert len(alerts) >= 1
        stats = manager.get_statistics()
        assert "total" in stats


class TestStreamingProcessorIntegration:
    def test_sensor_message_processing(self):
        processor = StreamingProcessor()
        msg = {
            "device_id": "sensor_001",
            "sensor_type": "temperature",
            "value": 25.5,
            "unit": "celsius",
            "timestamp": "2024-01-01T12:00:00Z",
            "location": "building_A",
            "message_id": "msg_001",
        }
        result = processor.process_sensor_message(msg)
        assert result is not None
        assert "device_id" in result

    def test_network_message_processing(self):
        processor = StreamingProcessor()
        msg = {
            "device_id": "sensor_001",
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.50",
            "src_port": 54321,
            "dst_port": 443,
            "protocol": "TCP",
            "packet_size": 1024,
            "flow_duration": 0.5,
            "timestamp": "2024-01-01T12:00:00Z",
            "message_id": "msg_002",
        }
        result = processor.process_network_message(msg)
        assert result is not None
        assert "device_id" in result

    def test_stats_tracking(self):
        processor = StreamingProcessor()
        msg = {
            "device_id": "sensor_001",
            "sensor_type": "temperature",
            "value": 25.5,
            "unit": "celsius",
            "timestamp": "2024-01-01T12:00:00Z",
            "location": "building_A",
            "message_id": "msg_001",
        }
        processor.process_sensor_message(msg)
        stats = processor.get_stats()
        assert isinstance(stats, dict)
        assert "total_processed" in stats


class TestMonitoringIntegration:
    def test_performance_monitor(self):
        monitor = PerformanceMonitor(window_size=100)
        for i in range(50):
            monitor.record_latency(np.random.uniform(5, 50))
            monitor.record_message()
        stats = monitor.get_latency_stats()
        assert "avg_ms" in stats
        assert "p95_ms" in stats
        throughput = monitor.get_throughput()
        assert throughput >= 0
        summary = monitor.get_summary()
        assert summary["total_messages"] == 50


class TestEndToEndPipeline:
    def test_synthetic_data_through_full_pipeline(self, sample_swat_df):
        features = [c for c in sample_swat_df.columns
                    if sample_swat_df[c].dtype in ("float64", "int64")
                    and c != "attack_label"]

        preprocessor = PhysicalPreprocessor()
        X = preprocessor.fit_transform(
            sample_swat_df[features], features,
        )
        assert X.shape[1] == len(features)

        model = IsolationForestModel(n_estimators=50, random_state=42)
        model.fit(X, features)
        predictions, scores = model.predict(X)
        assert len(predictions) == len(X)

        processor = StreamingProcessor()
        for i in range(5):
            msg = {
                "device_id": f"sensor_{i:03d}",
                "sensor_type": "temperature",
                "value": float(sample_swat_df.iloc[i]["FIT101"]),
                "unit": "celsius",
                "timestamp": "2024-01-01T12:00:00Z",
                "location": "building_A",
                "message_id": f"msg_{i:03d}",
            }
            result = processor.process_sensor_message(msg)
            assert result is not None

        stats = processor.get_stats()
        assert stats["total_processed"] >= 5

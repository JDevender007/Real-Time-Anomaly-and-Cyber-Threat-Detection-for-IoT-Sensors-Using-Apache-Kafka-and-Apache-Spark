"""Performance tests for the IoT security platform.

Benchmarks throughput, latency, and resource usage of the full pipeline.
"""
from __future__ import annotations

import time
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.settings import RANDOM_SEED
from src.preprocessing.physical import PhysicalPreprocessor
from src.preprocessing.cyber import CyberPreprocessor
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost import XGBoostModel
from src.streaming.processor import StreamingProcessor
from src.decision.engine import DecisionEngine
from src.monitoring.performance import PerformanceMonitor


@pytest.fixture
def large_swat_df():
    rng = np.random.RandomState(RANDOM_SEED)
    n = 10000
    labels = np.zeros(n, dtype=int)
    labels[2000:2200] = 1
    labels[7000:7100] = 1
    return pd.DataFrame({
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
def large_cyber_df():
    rng = np.random.RandomState(RANDOM_SEED)
    n = 10000
    labels = rng.choice(
        ["Benign", "DDoS", "DoS", "BruteForce", "PortScan"],
        n, p=[0.5, 0.15, 0.15, 0.1, 0.1],
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
        "src_port": rng.randint(1024, 65535, n),
        "dst_port": rng.choice([80, 443, 22, 53, 8080], n),
        "attack_label": labels,
    })


class TestModelTrainingPerformance:
    def test_if_training_time(self, large_swat_df):
        features = [c for c in large_swat_df.columns
                    if large_swat_df[c].dtype in ("float64", "int64")
                    and c != "attack_label"]
        X = large_swat_df[features].values
        y = large_swat_df["attack_label"].values

        preprocessor = PhysicalPreprocessor()
        X_df = pd.DataFrame(X, columns=features)
        X_scaled = preprocessor.fit_transform(X_df, features)

        model = IsolationForestModel(n_estimators=100, random_state=RANDOM_SEED)
        start = time.perf_counter()
        model.fit(X_scaled, features)
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        predictions, scores = model.predict(X_scaled)
        infer_time = time.perf_counter() - start

        assert train_time < 30, f"IF training took {train_time:.1f}s (>30s)"
        assert infer_time < 5, f"IF inference took {infer_time:.1f}s (>5s)"
        assert len(predictions) == len(X_scaled)
        print(f"IF: train={train_time:.2f}s, infer={infer_time:.3f}s, "
              f"throughput={len(X_scaled)/infer_time:.0f} msg/s")

    def test_rf_training_time(self, large_cyber_df):
        features = [c for c in large_cyber_df.columns
                    if large_cyber_df[c].dtype in ("float64", "int64")
                    and c != "attack_label"]
        preprocessor = CyberPreprocessor()
        df_scaled, y_encoded = preprocessor.fit_transform(
            large_cyber_df, features, "attack_label",
        )

        model = RandomForestModel(n_estimators=100, random_state=RANDOM_SEED)
        start = time.perf_counter()
        model.fit(df_scaled, y_encoded, feature_names=features,
                  label_classes=preprocessor.label_classes)
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        predictions, probabilities, confidence = model.predict(df_scaled)
        infer_time = time.perf_counter() - start

        assert train_time < 60, f"RF training took {train_time:.1f}s (>60s)"
        assert infer_time < 5, f"RF inference took {infer_time:.1f}s (>5s)"
        print(f"RF: train={train_time:.2f}s, infer={infer_time:.3f}s, "
              f"throughput={len(df_scaled)/infer_time:.0f} msg/s")

    def test_xgb_training_time(self, large_cyber_df):
        features = [c for c in large_cyber_df.columns
                    if large_cyber_df[c].dtype in ("float64", "int64")
                    and c != "attack_label"]
        preprocessor = CyberPreprocessor()
        df_scaled, y_encoded = preprocessor.fit_transform(
            large_cyber_df, features, "attack_label",
        )

        model = XGBoostModel(n_estimators=100, random_state=RANDOM_SEED)
        start = time.perf_counter()
        model.fit(df_scaled, y_encoded, feature_names=features,
                  label_classes=preprocessor.label_classes)
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        predictions, probabilities, confidence = model.predict(df_scaled)
        infer_time = time.perf_counter() - start

        assert train_time < 60, f"XGB training took {train_time:.1f}s (>60s)"
        assert infer_time < 5, f"XGB inference took {infer_time:.1f}s (>5s)"
        print(f"XGB: train={train_time:.2f}s, infer={infer_time:.3f}s, "
              f"throughput={len(df_scaled)/infer_time:.0f} msg/s")


class TestStreamingProcessorPerformance:
    def test_single_message_latency(self):
        processor = StreamingProcessor()
        latencies = []
        for i in range(100):
            msg = {
                "device_id": f"sensor_{i:04d}",
                "sensor_type": "temperature",
                "value": float(np.random.normal(25, 3)),
                "unit": "celsius",
                "timestamp": "2024-01-01T12:00:00Z",
                "location": "building_A",
                "message_id": f"msg_{i:04d}",
            }
            start = time.perf_counter()
            result = processor.process_sensor_message(msg)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
            assert result is not None

        avg_lat = statistics.mean(latencies)
        p95_lat = sorted(latencies)[int(0.95 * len(latencies))]
        p99_lat = sorted(latencies)[int(0.99 * len(latencies))]
        assert p95_lat < 500, f"P95 latency {p95_lat:.1f}ms (>500ms)"
        assert p99_lat < 1000, f"P99 latency {p99_lat:.1f}ms (>1000ms)"
        print(f"Streaming: avg={avg_lat:.1f}ms, p95={p95_lat:.1f}ms, p99={p99_lat:.1f}ms")

    def test_throughput_sustained(self):
        processor = StreamingProcessor()
        n_messages = 500
        start = time.perf_counter()
        for i in range(n_messages):
            msg = {
                "device_id": f"sensor_{i:04d}",
                "sensor_type": "temperature",
                "value": float(np.random.normal(25, 3)),
                "unit": "celsius",
                "timestamp": "2024-01-01T12:00:00Z",
                "location": "building_A",
                "message_id": f"msg_{i:04d}",
            }
            processor.process_sensor_message(msg)
        elapsed = time.perf_counter() - start
        throughput = n_messages / elapsed
        assert throughput > 10, f"Throughput {throughput:.0f} msg/s (<10 msg/s)"
        print(f"Sustained throughput: {throughput:.0f} msg/s ({n_messages} in {elapsed:.2f}s)")

    def test_network_message_throughput(self):
        processor = StreamingProcessor()
        n_messages = 500
        start = time.perf_counter()
        for i in range(n_messages):
            msg = {
                "device_id": f"sensor_{i:04d}",
                "src_ip": "192.168.1.100",
                "dst_ip": "10.0.0.50",
                "src_port": 54321,
                "dst_port": 443,
                "protocol": "TCP",
                "packet_size": int(np.random.exponential(500)),
                "flow_duration": float(np.random.exponential(0.5)),
                "timestamp": "2024-01-01T12:00:00Z",
                "message_id": f"msg_{i:04d}",
            }
            processor.process_network_message(msg)
        elapsed = time.perf_counter() - start
        throughput = n_messages / elapsed
        print(f"Network throughput: {throughput:.0f} msg/s")


class TestDecisionEnginePerformance:
    def test_decision_latency(self):
        engine = DecisionEngine()
        latencies = []
        for _ in range(1000):
            phys = float(np.random.uniform(0, 1))
            cyber = float(np.random.uniform(0, 1))
            start = time.perf_counter()
            result = engine.compute_risk(physical_score=phys, cyber_score=cyber)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        avg_lat = statistics.mean(latencies)
        p99_lat = sorted(latencies)[int(0.99 * len(latencies))]
        assert p99_lat < 10, f"P99 decision latency {p99_lat:.2f}ms (>10ms)"
        print(f"Decision engine: avg={avg_lat:.3f}ms, p99={p99_lat:.3f}ms")


class TestMemoryAndResourceUsage:
    def test_pipeline_memory_footprint(self):
        import sys
        processor = StreamingProcessor()
        base_size = sys.getsizeof(processor)

        for i in range(100):
            msg = {
                "device_id": f"sensor_{i:04d}",
                "sensor_type": "temperature",
                "value": float(np.random.normal(25, 3)),
                "unit": "celsius",
                "timestamp": "2024-01-01T12:00:00Z",
                "location": "building_A",
                "message_id": f"msg_{i:04d}",
            }
            processor.process_sensor_message(msg)

        final_size = sys.getsizeof(processor)
        growth = final_size - base_size
        print(f"Processor memory: base={base_size}, final={final_size}, growth={growth}")


class TestPreprocessingPerformance:
    def test_physical_preprocessing_throughput(self, large_swat_df):
        features = [c for c in large_swat_df.columns
                    if large_swat_df[c].dtype in ("float64", "int64")
                    and c != "attack_label"]
        n_rows = len(large_swat_df)

        preprocessor = PhysicalPreprocessor()
        X_df = pd.DataFrame(large_swat_df[features].values, columns=features)

        start = time.perf_counter()
        X_scaled = preprocessor.fit_transform(X_df, features)
        fit_time = time.perf_counter() - start

        start = time.perf_counter()
        X_transformed = preprocessor.transform(X_df)
        transform_time = time.perf_counter() - start

        assert transform_time < 5, f"Transform took {transform_time:.1f}s (>5s)"
        print(f"Physical preprocessing: fit={fit_time:.3f}s, transform={transform_time:.3f}s, "
              f"throughput={n_rows/transform_time:.0f} rows/s")

    def test_cyber_preprocessing_throughput(self, large_cyber_df):
        features = [c for c in large_cyber_df.columns
                    if large_cyber_df[c].dtype in ("float64", "int64")
                    and c != "attack_label"]
        n_rows = len(large_cyber_df)

        preprocessor = CyberPreprocessor()
        start = time.perf_counter()
        X_scaled, y = preprocessor.fit_transform(large_cyber_df, features, "attack_label")
        fit_time = time.perf_counter() - start

        start = time.perf_counter()
        X_transformed = preprocessor.transform_features(large_cyber_df)
        transform_time = time.perf_counter() - start

        assert transform_time < 5, f"Transform took {transform_time:.1f}s (>5s)"
        print(f"Cyber preprocessing: fit={fit_time:.3f}s, transform={transform_time:.3f}s, "
              f"throughput={n_rows/transform_time:.0f} rows/s")

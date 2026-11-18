"""End-to-end streaming processor: Kafka -> ML Inference -> Decision Engine -> DB."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.api.schemas import (
    PhysicalSensorMessage, NetworkMessage, PhysicalAnomalyResult,
    CyberThreatResult, UnifiedRiskResult, AlertType, RiskLevel,
)
from src.config.loader import load_config
from src.config.settings import MODEL_ROOT, RANDOM_SEED
from src.decision.engine import DecisionEngine
from src.alerts.manager import AlertManager
from src.database.repositories import (
    DeviceRepository, AlertRepository, SensorReadingRepository,
    PhysicalAnomalyRepository, CyberThreatRepository, MetricRepository,
)
from src.features.physical import engineer_physical_features
from src.features.cyber import engineer_cyber_features
from src.utils.helpers import utc_now, set_global_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineStats:
    total_processed: int = 0
    physical_anomalies: int = 0
    cyber_threats: int = 0
    alerts_generated: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    replay_physical: int = 0
    replay_physical_correct: int = 0
    replay_cyber: int = 0
    replay_cyber_correct: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.total_processed, 1)

    @property
    def replay_physical_accuracy(self) -> float | None:
        if self.replay_physical == 0:
            return None
        return self.replay_physical_correct / self.replay_physical

    @property
    def replay_cyber_accuracy(self) -> float | None:
        if self.replay_cyber == 0:
            return None
        return self.replay_cyber_correct / self.replay_cyber

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_processed": self.total_processed,
            "physical_anomalies": self.physical_anomalies,
            "cyber_threats": self.cyber_threats,
            "alerts_generated": self.alerts_generated,
            "errors": self.errors,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "replay_physical": self.replay_physical,
            "replay_physical_accuracy": (
                round(self.replay_physical_accuracy, 4)
                if self.replay_physical_accuracy is not None else None
            ),
            "replay_cyber": self.replay_cyber,
            "replay_cyber_accuracy": (
                round(self.replay_cyber_accuracy, 4)
                if self.replay_cyber_accuracy is not None else None
            ),
        }


class StreamingProcessor:
    """Processes streaming IoT data through the full detection pipeline.

    Pipeline:
        Raw Message
        -> Parse & Validate
        -> Feature Engineering
        -> Physical Detection (IF + AE)
        -> Cyber Detection (RF + XGB)
        -> Decision Engine
        -> Alert Manager
        -> Database Persistence
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or MODEL_ROOT
        self.stats = PipelineStats()
        self._config = load_config("models")

        self.decision_engine = DecisionEngine()
        self.alert_manager = AlertManager()
        self.device_repo = DeviceRepository()
        self.alert_repo = AlertRepository()
        self.sensor_repo = SensorReadingRepository()
        self.physical_repo = PhysicalAnomalyRepository()
        self.cyber_repo = CyberThreatRepository()
        self.metric_repo = MetricRepository()

        self._physical_if = None
        self._physical_ae = None
        self._cyber_rf = None
        self._cyber_xgb = None
        self._physical_scaler = None
        self._cyber_scaler = None

        self._phys_preprocessor = None
        self._phys_feature_names: list[str] = []
        self._cic_rf = None
        self._cic_xgb = None
        self._cic_scaler = None
        self._cic_features: list[str] = []

        self._phys_baselines: dict[str, dict[str, Any]] = {}
        self._cyber_baselines: dict[str, dict[str, Any]] = {}
        self._alert_cooldown: dict[str, float] = {}
        self._alert_cooldown_seconds = 3.0

        self._load_models()
        self._load_replay_assets()

    def _load_models(self) -> None:
        try:
            from src.models.isolation_forest import IsolationForestModel
            if_path = self.model_dir / "physical" / "isolation_forest"
            if if_path.exists() and (if_path / "metadata.json").exists():
                self._physical_if = IsolationForestModel.load(if_path)
                logger.info("Loaded Isolation Forest model")
        except Exception as e:
            logger.warning("Could not load Isolation Forest: %s", e)

        try:
            from src.models.autoencoder import AutoencoderModel
            ae_path = self.model_dir / "physical" / "autoencoder"
            if ae_path.exists() and (ae_path / "metadata.json").exists():
                self._physical_ae = AutoencoderModel.load(ae_path)
                logger.info("Loaded Autoencoder model")
        except Exception as e:
            logger.warning("Could not load Autoencoder: %s", e)

        try:
            from src.models.random_forest import RandomForestModel
            rf_path = self.model_dir / "cyber" / "random_forest"
            if rf_path.exists() and (rf_path / "metadata.json").exists():
                self._cyber_rf = RandomForestModel.load(rf_path)
                logger.info("Loaded Random Forest model")
        except Exception as e:
            logger.warning("Could not load Random Forest: %s", e)

        try:
            from src.models.xgboost import XGBoostModel
            xgb_path = self.model_dir / "cyber" / "xgboost"
            if xgb_path.exists() and (xgb_path / "metadata.json").exists():
                self._cyber_xgb = XGBoostModel.load(xgb_path)
                logger.info("Loaded XGBoost model")
        except Exception as e:
            logger.warning("Could not load XGBoost: %s", e)

    def _load_replay_assets(self) -> None:
        """Load preprocessors and CICIoT2023 models for dataset replay inference."""
        try:
            from src.preprocessing.physical import PhysicalPreprocessor
            phys_path = self.model_dir / "physical"
            if (phys_path / "scaler.pkl").exists():
                self._phys_preprocessor = PhysicalPreprocessor().load(phys_path)
                self._phys_feature_names = self._phys_preprocessor.feature_names
                logger.info(
                    "Loaded physical preprocessor (%d features)",
                    len(self._phys_feature_names),
                )
        except Exception as e:
            logger.warning("Could not load physical preprocessor: %s", e)

        try:
            import joblib
            cyber_path = self.model_dir / "cyber"
            required = ["random_forest.joblib", "xgboost.joblib",
                        "scaler.joblib", "feature_names.joblib"]
            if all((cyber_path / f).exists() for f in required):
                self._cic_rf = joblib.load(cyber_path / "random_forest.joblib")
                self._cic_xgb = joblib.load(cyber_path / "xgboost.joblib")
                self._cic_scaler = joblib.load(cyber_path / "scaler.joblib")
                self._cic_features = list(joblib.load(cyber_path / "feature_names.joblib"))
                logger.info(
                    "Loaded CICIoT2023 models (%d features, classes=%s)",
                    len(self._cic_features),
                    list(getattr(self._cic_rf, "classes_", [])),
                )
        except Exception as e:
            logger.warning("Could not load CICIoT2023 models: %s", e)

    def process_sensor_message(self, data: dict[str, Any]) -> dict[str, Any] | None:
        start_time = time.perf_counter()
        try:
            msg = PhysicalSensorMessage(**data)
            timestamp = datetime.now(timezone.utc)

            self.device_repo.upsert(msg.device_id, "iot_sensor", "online")
            self.sensor_repo.store(
                device_id=msg.device_id,
                timestamp=timestamp,
                temperature=msg.temperature,
                humidity=msg.humidity,
                pressure=msg.pressure,
                voltage=msg.voltage,
                battery=msg.battery,
            )

            physical_result = self._detect_physical(msg, timestamp)

            cyber_result = CyberThreatResult(
                device_id=msg.device_id,
                timestamp=timestamp.isoformat(),
            )

            risk_result = self.decision_engine.compute_risk(
                physical_score=physical_result.combined_score,
                cyber_score=0.0,
            )
            risk_result.device_id = msg.device_id
            risk_result.timestamp = timestamp.isoformat()

            self._handle_alerts(msg.device_id, physical_result, cyber_result, risk_result)

            latency_ms = (time.perf_counter() - start_time) * 1000
            self.stats.total_processed += 1
            self.stats.total_latency_ms += latency_ms

            return {
                "device_id": msg.device_id,
                "timestamp": timestamp.isoformat(),
                "physical": physical_result.model_dump(),
                "risk": risk_result.model_dump(),
                "latency_ms": round(latency_ms, 3),
            }

        except Exception as e:
            self.stats.errors += 1
            logger.error("Error processing sensor message: %s", e)
            return None

    def process_network_message(self, data: dict[str, Any]) -> dict[str, Any] | None:
        start_time = time.perf_counter()
        try:
            msg = NetworkMessage(**data)
            timestamp = datetime.now(timezone.utc)

            self.device_repo.upsert(msg.device_id, "iot_sensor", "online")

            cyber_result = self._detect_cyber(msg, timestamp)

            physical_result = PhysicalAnomalyResult(
                device_id=msg.device_id,
                timestamp=timestamp.isoformat(),
            )

            risk_result = self.decision_engine.compute_risk(
                physical_score=0.0,
                cyber_score=cyber_result.combined_threat_score,
            )
            risk_result.device_id = msg.device_id
            risk_result.timestamp = timestamp.isoformat()

            self._handle_alerts(msg.device_id, physical_result, cyber_result, risk_result)

            latency_ms = (time.perf_counter() - start_time) * 1000
            self.stats.total_processed += 1
            self.stats.total_latency_ms += latency_ms

            return {
                "device_id": msg.device_id,
                "timestamp": timestamp.isoformat(),
                "cyber": cyber_result.model_dump(),
                "risk": risk_result.model_dump(),
                "latency_ms": round(latency_ms, 3),
            }

        except Exception as e:
            self.stats.errors += 1
            logger.error("Error processing network message: %s", e)
            return None

    def process_message(self, data: dict[str, Any]) -> dict[str, Any] | None:
        if data.get("kind") == "dataset_replay":
            domain = data.get("domain")
            if domain == "physical":
                return self.process_replay_physical(data)
            elif domain == "cyber":
                return self.process_replay_cyber(data)
            logger.warning("Unknown replay domain: %s", domain)
            return None
        if "temperature" in data or "humidity" in data:
            return self.process_sensor_message(data)
        elif "source_ip" in data or "packet_count" in data:
            return self.process_network_message(data)
        else:
            if "payload" in data:
                return self.process_message(data["payload"])
            logger.warning("Unknown message type: %s", list(data.keys())[:5])
            return None

    def process_replay_physical(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Dataset replay path: full 178-feature row -> physical preprocessor -> IF."""
        start_time = time.perf_counter()
        try:
            import numpy as np
            import pandas as pd

            device_id = str(data.get("device_id", "replay_physical"))
            features = data.get("features", {}) or {}
            label = int(data.get("label", 0) or 0)
            timestamp = datetime.now(timezone.utc)

            if self._phys_preprocessor is None or self._physical_if is None:
                if not getattr(self, "_replay_phys_warned", False):
                    logger.warning("Replay physical: preprocessor/IF not loaded - skipping")
                    self._replay_phys_warned = True
                return None

            names = self._phys_feature_names
            row = {n: float(features.get(n, 0.0) or 0.0) for n in names}
            df = pd.DataFrame([row], columns=names)
            X = self._phys_preprocessor.transform(df)

            if_score = 0.0
            if_pred = 0
            try:
                preds, scores = self._physical_if.predict(X)
                if_pred = int(preds[0])
                if_score_raw = float(scores[0])
            except Exception as e:
                logger.warning("Replay IF prediction error: %s", e)
                return None

            # Monotonic calibration: model decision boundary (margin=0) -> 0.5,
            # normal-side scores map below the alert threshold, attacks above.
            try:
                offset = float(self._physical_if.model.offset_)
            except Exception:
                offset = -0.5
            margin = if_score_raw + offset
            if_score = float(np.clip(0.5 + margin * 3.0, 0.0, 1.0))
            confidence = if_score if if_pred == 1 else max(0.1, 1.0 - if_score)

            physical_result = PhysicalAnomalyResult(
                device_id=device_id,
                timestamp=timestamp.isoformat(),
            )
            physical_result.if_prediction = if_pred
            physical_result.if_score = if_score
            physical_result.combined_score = if_score
            physical_result.confidence = confidence

            cyber_result = CyberThreatResult(
                device_id=device_id,
                timestamp=timestamp.isoformat(),
            )

            risk_result = self.decision_engine.compute_risk(
                physical_score=if_score,
                cyber_score=0.0,
            )
            risk_result.device_id = device_id
            risk_result.timestamp = timestamp.isoformat()

            self.device_repo.upsert(device_id, "dataset_replay", "online")

            self.stats.replay_physical += 1
            if if_pred == label:
                self.stats.replay_physical_correct += 1

            if if_pred == 1:
                self.stats.physical_anomalies += 1
                self.physical_repo.store(
                    device_id=device_id,
                    timestamp=timestamp,
                    if_prediction=if_pred,
                    if_score=if_score,
                    ae_prediction=0,
                    ae_score=0.0,
                    reconstruction_error=0.0,
                    combined_score=if_score,
                    confidence=confidence,
                )

            self._handle_alerts(device_id, physical_result, cyber_result, risk_result)

            latency_ms = (time.perf_counter() - start_time) * 1000
            self.stats.total_processed += 1
            self.stats.total_latency_ms += latency_ms

            return {
                "device_id": device_id,
                "timestamp": timestamp.isoformat(),
                "kind": "dataset_replay",
                "domain": "physical",
                "physical": physical_result.model_dump(),
                "risk": risk_result.model_dump(),
                "label": label,
                "prediction": if_pred,
                "correct": if_pred == label,
                "latency_ms": round(latency_ms, 3),
            }

        except Exception as e:
            self.stats.errors += 1
            logger.error("Error processing physical replay: %s", e)
            return None

    def process_replay_cyber(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Dataset replay path: 46-feature CICIoT2023 row -> scaler -> RF + XGB."""
        start_time = time.perf_counter()
        try:
            import numpy as np

            device_id = str(data.get("device_id", "replay_cyber"))
            features = data.get("features", {}) or {}
            label = int(data.get("label", 0) or 0)
            attack_type = str(data.get("attack_type", ""))
            timestamp = datetime.now(timezone.utc)

            if self._cic_rf is None or self._cic_scaler is None:
                if not getattr(self, "_replay_cyber_warned", False):
                    logger.warning("Replay cyber: CICIoT models not loaded - skipping")
                    self._replay_cyber_warned = True
                return None

            row = np.array(
                [float(features.get(n, 0.0) or 0.0) for n in self._cic_features],
                dtype=np.float64,
            ).reshape(1, -1)
            row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
            X = self._cic_scaler.transform(row)

            rf_p, xgb_p = 0, 0
            rf_prob, xgb_prob = 0.0, 0.0
            try:
                rf_prob = float(self._cic_rf.predict_proba(X)[0][1])
                rf_p = int(rf_prob >= 0.5)
            except Exception as e:
                logger.warning("Replay RF prediction error: %s", e)
            try:
                xgb_prob = float(self._cic_xgb.predict_proba(X)[0][1])
                xgb_p = int(xgb_prob >= 0.5)
            except Exception as e:
                logger.warning("Replay XGB prediction error: %s", e)

            combined = float(np.clip((rf_prob + xgb_prob) / 2.0, 0.0, 1.0))
            pred_label = int(rf_p == 1 or xgb_p == 1)
            confidence = max(rf_prob if rf_p else 1.0 - rf_prob,
                             xgb_prob if xgb_p else 1.0 - xgb_prob)

            cyber_result = CyberThreatResult(
                device_id=device_id,
                timestamp=timestamp.isoformat(),
            )
            cyber_result.rf_prediction = "Attack" if rf_p else "Benign"
            cyber_result.rf_probability = rf_prob
            cyber_result.rf_confidence = rf_prob
            cyber_result.xgb_prediction = "Attack" if xgb_p else "Benign"
            cyber_result.xgb_probability = xgb_prob
            cyber_result.xgb_confidence = xgb_prob
            cyber_result.combined_threat_score = combined

            physical_result = PhysicalAnomalyResult(
                device_id=device_id,
                timestamp=timestamp.isoformat(),
            )

            risk_result = self.decision_engine.compute_risk(
                physical_score=0.0,
                cyber_score=combined,
            )
            risk_result.device_id = device_id
            risk_result.timestamp = timestamp.isoformat()

            self.device_repo.upsert(device_id, "dataset_replay", "online")

            self.stats.replay_cyber += 1
            if pred_label == label:
                self.stats.replay_cyber_correct += 1

            if pred_label == 1:
                self.stats.cyber_threats += 1
                self.cyber_repo.store(
                    device_id=device_id,
                    timestamp=timestamp,
                    rf_prediction=cyber_result.rf_prediction,
                    rf_probability=rf_prob,
                    xgb_prediction=cyber_result.xgb_prediction,
                    xgb_probability=xgb_prob,
                    combined_threat_score=combined,
                )

            self._handle_alerts(device_id, physical_result, cyber_result, risk_result)

            latency_ms = (time.perf_counter() - start_time) * 1000
            self.stats.total_processed += 1
            self.stats.total_latency_ms += latency_ms

            return {
                "device_id": device_id,
                "timestamp": timestamp.isoformat(),
                "kind": "dataset_replay",
                "domain": "cyber",
                "cyber": cyber_result.model_dump(),
                "risk": risk_result.model_dump(),
                "label": label,
                "prediction": pred_label,
                "correct": pred_label == label,
                "attack_type": attack_type,
                "latency_ms": round(latency_ms, 3),
            }

        except Exception as e:
            self.stats.errors += 1
            logger.error("Error processing cyber replay: %s", e)
            return None

    def _zscore_anomaly(
        self, bucket: str, fields: dict[str, float],
        warmup: int = 50, z_threshold: float = 3.5,
    ) -> tuple[float, bool]:
        """Streaming z-score over a rolling per-device baseline (Welford)."""
        import math

        baselines = self._phys_baselines if bucket.startswith("phys") else self._cyber_baselines
        b = baselines.setdefault(bucket, {})
        zs: dict[str, float] = {}
        for name, value in fields.items():
            if not math.isfinite(value):
                continue
            st = b.setdefault(name, {"n": 0, "mean": 0.0, "m2": 0.0})
            if st["n"] >= warmup and st["m2"] > 1e-9:
                std = math.sqrt(st["m2"] / (st["n"] - 1))
                if std > 1e-9:
                    zs[name] = abs(value - st["mean"]) / std
        for name, value in fields.items():
            if not math.isfinite(value):
                continue
            st = b[name]
            st["n"] += 1
            delta = value - st["mean"]
            st["mean"] += delta / st["n"]
            st["m2"] += delta * (value - st["mean"])

        if not zs:
            return 0.0, False
        max_z = max(zs.values())
        score = min(1.0, max_z / 6.0)
        return score, max_z >= z_threshold

    def _detect_physical(self, msg: PhysicalSensorMessage,
                         timestamp: datetime) -> PhysicalAnomalyResult:
        result = PhysicalAnomalyResult(
            device_id=msg.device_id,
            timestamp=timestamp.isoformat(),
        )

        features = [msg.temperature, msg.humidity, msg.pressure, msg.voltage, msg.battery]
        import numpy as np
        X = np.array(features).reshape(1, -1)

        if_score = 0.0
        if_pred = 0
        used_fallback = False
        if self._physical_if is not None:
            try:
                if_pred_arr, if_scores = self._physical_if.predict(X)
                if_pred = int(if_pred_arr[0])
                if_score = float(if_scores[0])
            except Exception as e:
                if not getattr(self, "_if_dim_warned", False):
                    logger.warning(
                        "IF unavailable for simulator input (%s) - using z-score fallback", e,
                    )
                    self._if_dim_warned = True
                z_score, z_flag = self._zscore_anomaly(
                    f"phys::{msg.device_id}",
                    {
                        "temperature": msg.temperature,
                        "humidity": msg.humidity,
                        "pressure": msg.pressure,
                        "voltage": msg.voltage,
                        "battery": msg.battery,
                    },
                )
                if_score = z_score
                if_pred = 1 if z_flag else 0
                used_fallback = True

        ae_score = 0.0
        ae_pred = 0
        recon_error = 0.0
        if self._physical_ae is not None:
            try:
                ae_pred_arr, ae_scores, reconstructed = self._physical_ae.predict(X)
                ae_pred = int(ae_pred_arr[0])
                ae_score = float(ae_scores[0])
                recon_error = float(np.mean((X - reconstructed) ** 2))
            except Exception as e:
                logger.warning("AE prediction error: %s", e)

        if used_fallback:
            combined_score = if_score
        else:
            combined_score = (if_score + ae_score) / 2.0
        confidence = 0.0
        if if_pred == 1 and ae_pred == 1:
            confidence = 0.95
        elif if_pred == 1 or ae_pred == 1:
            confidence = 0.6
        else:
            confidence = max(0.1, 1.0 - combined_score)

        result.if_prediction = if_pred
        result.if_score = if_score
        result.ae_prediction = ae_pred
        result.ae_score = ae_score
        result.reconstruction_error = recon_error
        result.combined_score = min(1.0, combined_score)
        result.confidence = confidence

        if if_pred == 1 or ae_pred == 1:
            self.stats.physical_anomalies += 1
            self.physical_repo.store(
                device_id=msg.device_id,
                timestamp=timestamp,
                if_prediction=if_pred,
                if_score=if_score,
                ae_prediction=ae_pred,
                ae_score=ae_score,
                reconstruction_error=recon_error,
                combined_score=combined_score,
                confidence=confidence,
            )

        return result

    def _detect_cyber(self, msg: NetworkMessage,
                      timestamp: datetime) -> CyberThreatResult:
        import numpy as np

        result = CyberThreatResult(
            device_id=msg.device_id,
            timestamp=timestamp.isoformat(),
        )

        protocol_map = {"TCP": 0, "UDP": 1, "ICMP": 2}
        features = [
            msg.packet_count, msg.packet_rate, msg.bytes,
            0.0,
            msg.source_port, msg.destination_port,
            protocol_map.get(msg.protocol, 0),
            float(msg.bytes) / max(msg.packet_count, 1),
        ]
        X = np.array(features).reshape(1, -1)

        rf_pred = "Benign"
        rf_conf = 0.0
        used_fallback = False
        if self._cyber_rf is not None:
            try:
                preds, probs, conf = self._cyber_rf.predict(X)
                rf_pred = str(preds[0])
                rf_conf = float(conf[0])
            except Exception as e:
                if not getattr(self, "_rf_dim_warned", False):
                    logger.warning(
                        "RF unavailable for simulator input (%s) - using z-score fallback", e,
                    )
                    self._rf_dim_warned = True
                z_score, z_flag = self._zscore_anomaly(
                    f"cyber::{msg.device_id}",
                    {
                        "packet_rate": msg.packet_rate,
                        "bytes": float(msg.bytes),
                        "packet_count": float(msg.packet_count),
                    },
                )
                rf_conf = z_score
                rf_pred = "Attack" if z_flag else "Benign"
                used_fallback = True

        xgb_pred = "Benign"
        xgb_conf = 0.0
        if self._cyber_xgb is not None and not used_fallback:
            try:
                preds, probs, conf = self._cyber_xgb.predict(X)
                xgb_pred = str(preds[0])
                xgb_conf = float(conf[0])
            except Exception as e:
                logger.warning("XGB prediction error: %s", e)

        if rf_pred == xgb_pred and rf_pred != "Benign":
            combined = min(1.0, (rf_conf + xgb_conf) / 2.0 * 1.2)
        elif rf_pred != "Benign" or xgb_pred != "Benign":
            combined = max(rf_conf, xgb_conf) * 0.7
        else:
            combined = 0.0

        result.rf_prediction = rf_pred
        result.rf_probability = rf_conf
        result.rf_confidence = rf_conf
        result.xgb_prediction = xgb_pred
        result.xgb_probability = xgb_conf
        result.xgb_confidence = xgb_conf
        result.combined_threat_score = combined

        if rf_pred != "Benign" or xgb_pred != "Benign":
            self.stats.cyber_threats += 1
            self.cyber_repo.store(
                device_id=msg.device_id,
                timestamp=timestamp,
                rf_prediction=rf_pred,
                rf_probability=rf_conf,
                xgb_prediction=xgb_pred,
                xgb_probability=xgb_conf,
                combined_threat_score=combined,
            )

        return result

    def _handle_alerts(
        self,
        device_id: str,
        physical: PhysicalAnomalyResult,
        cyber: CyberThreatResult,
        risk: UnifiedRiskResult,
    ) -> None:
        if risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            alert_type = AlertType.UNIFIED_RISK
            if physical.combined_score > 0.5:
                alert_type = AlertType.PHYSICAL_ANOMALY
            elif cyber.combined_threat_score > 0.5:
                alert_type = AlertType.CYBER_THREAT

            alert = self.alert_manager.create_alert(
                device_id=device_id,
                alert_type=alert_type,
                severity=risk.risk_level,
                physical_score=physical.combined_score,
                cyber_score=cyber.combined_threat_score,
                risk_score=risk.unified_risk_score,
                model_name="combined",
                predicted_class=cyber.rf_prediction if cyber.rf_prediction != "Benign" else cyber.xgb_prediction,
                confidence=max(physical.confidence, cyber.rf_confidence, cyber.xgb_confidence),
            )

            self.alert_repo.create(
                alert_id=alert.alert_id,
                device_id=device_id,
                alert_type=alert_type.value,
                severity=risk.risk_level.value,
                physical_score=physical.combined_score,
                cyber_score=cyber.combined_threat_score,
                risk_score=risk.unified_risk_score,
                model_name="combined",
                predicted_class=alert.predicted_class,
                confidence=alert.confidence,
            )
            self.stats.alerts_generated += 1

        elif physical.combined_score > 0.3 or cyber.combined_threat_score > 0.3:
            severity = RiskLevel.MEDIUM if risk.unified_risk_score > 0.3 else RiskLevel.LOW
            if severity != RiskLevel.MEDIUM:
                return
            now_mono = time.monotonic()
            cooldown_key = f"{device_id}:medium"
            last = self._alert_cooldown.get(cooldown_key, 0.0)
            if now_mono - last < self._alert_cooldown_seconds:
                return
            self._alert_cooldown[cooldown_key] = now_mono

            alert = self.alert_manager.create_alert(
                device_id=device_id,
                alert_type=AlertType.PHYSICAL_ANOMALY if physical.combined_score > cyber.combined_threat_score else AlertType.CYBER_THREAT,
                severity=severity,
                physical_score=physical.combined_score,
                cyber_score=cyber.combined_threat_score,
                risk_score=risk.unified_risk_score,
            )
            self.alert_repo.create(
                alert_id=alert.alert_id,
                device_id=device_id,
                alert_type=alert.alert_type.value,
                severity=severity.value,
                physical_score=physical.combined_score,
                cyber_score=cyber.combined_threat_score,
                risk_score=risk.unified_risk_score,
                model_name="combined",
                predicted_class=alert.predicted_class,
                confidence=alert.confidence,
            )
            self.stats.alerts_generated += 1

    def get_stats(self) -> dict[str, Any]:
        return self.stats.to_dict()

    def reset_stats(self) -> None:
        self.stats = PipelineStats()

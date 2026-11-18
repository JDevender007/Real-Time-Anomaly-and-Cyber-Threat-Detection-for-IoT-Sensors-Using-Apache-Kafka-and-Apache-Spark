# Reference Repository Audit: DDoS-RealTime-IDS

**Audit Date:** 2026-09-07  
**Auditor:** opencode (automated)  
**Repository:** `D:\Projects\DDoS-RealTime-IDS`

---

## 1. Structure Inventory

### Top-Level Modules

| Directory | Purpose | Size/Maturity |
|-----------|---------|---------------|
| `src/` | Core application code (13 subpackages, ~70 Python files) | **Functional** — production-grade DDoS detection system |
| `scripts/` | Training, evaluation, demo, attack generation scripts (25+ files) | **Functional** — complete train/eval/simulate pipeline |
| `tests/` | Pytest test suite (14 files, 74+ tests) + integration script (61 tests) | **Functional** — good coverage |
| `configs/` | 7 YAML config files (app, kafka, spark, model, features, risk, model_config) | **Functional** — well-structured |
| `models/` | 5 trained model directories with .joblib + metadata.json + metrics.json | **Functional** — XGBoost is production model |
| `data/` | Raw CSVs, processed train/test splits, SQLite DB, sample data | **Functional** — CICDDoS2019 dataset |
| `reports/` | 34 evaluation plots, metrics CSVs, classification reports | **Functional** — research-grade outputs |
| `dashboard/` | FastAPI + Jinja2 HTML dashboard (NOT Streamlit) | **Functional** — 2 pages |
| `logs/` | Application log files | **Functional** |
| `results/` | Alert CSVs, detection history CSVs | **Functional** |
| `.github/` | CI workflow (ci.yml) | **Functional** |

### Key Source Subpackages

| Subpackage | Files | Purpose | Maturity |
|------------|-------|---------|----------|
| `src/config/` | 3 | Settings, YAML loader, logging config | **Functional** |
| `src/data/` | 3 | Dataset loader, validator, SQLite event store | **Functional** |
| `src/detection/` | 14 | Flow features, real-time IDS, detection engine, alert manager, model manager | **Functional** |
| `src/kafka/` | 4 | Producer, consumer, topic management, serializers | **Functional** |
| `src/models/` | 8 | XGBoost, Random Forest, SVM, K-Means wrappers + model manager | **Functional** |
| `src/streaming/` | 8 | Stream processor, Kafka producer/consumer, Spark integration | **Functional** |
| `src/security/` | 3 | Risk engine, decision engine, mitigation engine | **Functional** |
| `src/mitigation/` | 5 | IP blocker, rate limiter + compat re-exports | **Functional** (simulated) |
| `src/monitoring/` | 3 | Health checks, metrics collector, performance benchmarks | **Functional** |
| `src/api/` | 4 | FastAPI routes, schemas, simulator routes, WebSocket | **Functional** |
| `src/preprocessing/` | 12 | Pipeline, scaler, encoder, data prep scripts | **Functional** |
| `src/spark/` | 5 | Session, streaming, preprocessing, prediction, training | **Mostly stubs** — graceful degradation when Spark unavailable |
| `src/training/` | ~5 | Training scripts for various models | **Functional** |
| `src/utils/` | 3 | Helpers, logger, metrics | **Functional** |

---

## 2. Stack Comparison

### Reference Repo Stack

| Component | Version/Detail |
|-----------|----------------|
| Python | 3.12+ (3.13 pyc files present) |
| Kafka | Apache Kafka 3.7.0 (KRaft mode, no ZooKeeper) |
| Spark | Apache Spark 3.5.1 |
| ML | XGBoost, scikit-learn (RF, SVM, K-Means) |
| API | FastAPI + uvicorn |
| Database | SQLite (not PostgreSQL) |
| Dashboard | FastAPI + Jinja2 HTML (NOT Streamlit) |
| Streaming | kafka-python |
| Packet capture | Scapy |
| Testing | Pytest |
| Docker | docker-compose.yml with 4 services |
| Config | YAML + .env |

### Target Project Stack

| Component | Target | Mismatch Flag |
|-----------|--------|---------------|
| Python | 3.12+ | **OK** |
| Kafka | 3.x | **OK** — same major version |
| Spark | 3.5+ | **OK** — same version |
| ML | XGBoost, RF, Isolation Forest, Autoencoder | **GAP** — no Isolation Forest, no Autoencoder in ref |
| API | FastAPI | **OK** |
| Database | PostgreSQL (SQLite fallback) | **CHANGE** — ref uses SQLite only |
| Dashboard | Streamlit | **CHANGE** — ref uses Jinja2 HTML |
| Streaming | kafka-python | **OK** |
| Extra | SHAP, Pydantic schemas, SQLAlchemy | **GAP** — not in ref |

---

## 3. Reusable Components

### Directly Reusable (with minimal adaptation)

1. **Kafka Producer/Consumer Boilerplate** (`src/kafka/producer.py`, `src/kafka/consumer.py`)
   - TrafficProducer publishes rows at configurable rate
   - EventConsumer polls with retry logic
   - Topic management via KafkaAdminClient
   - *Adaptation needed:* Topic names, message schemas for dual physical/cyber pipeline

2. **Kafka Topic Management** (`src/kafka/topics.py`)
   - Idempotent topic creation
   - *Adaptation needed:* New topic names (iot.sensor.raw, iot.network.raw, etc.)

3. **Kafka Serializers** (`src/kafka/serializers.py`)
   - JSON envelope with event_id, timestamp, source
   - *Adaptation needed:* Physical sensor message schema

4. **Spark Session Configuration** (`src/spark/session.py`)
   - Local mode config with graceful fallback
   - *Adaptation needed:* Minimal — config is already YAML-driven

5. **Config/Env Management** (`src/config/settings.py`, `src/config/loader.py`)
   - YAML loading with dot-notation access, per-config caching
   - Path definitions (PROJECT_ROOT, DATA_ROOT, MODEL_ROOT)
   - *Adaptation needed:* New paths, new config files

6. **Logging Setup** (`src/config/logging_config.py`)
   - Rotating file + console logging
   - *Adaptation needed:* Minimal — add structured logging fields

7. **Docker Compose** (`docker-compose.yml`)
   - Kafka 3.7 (KRaft), Spark 3.5.1, API service
   - Health checks present
   - *Adaptation needed:* Add PostgreSQL service, update API service

8. **Dockerfile** (`Dockerfile`)
   - Python 3.12-slim, uvicorn, healthcheck
   - *Adaptation needed:* Minimal

9. **Risk Engine** (`src/security/risk_engine.py`)
   - Risk level assignment based on confidence + thresholds
   - Configurable from risk.yaml
   - *Adaptation needed:* Extend for physical anomaly scores

10. **Decision Engine** (`src/security/decision_engine.py`)
    - Two-phase: detection → risk analysis
    - *Adaptation needed:* Add physical detection path

11. **Alert Manager** (`src/detection/alert_manager.py`)
    - Severity-based alert generation (LOW/MEDIUM/HIGH/CRITICAL)
    - CSV export
    - *Adaptation needed:* Add physical anomaly alerts, database persistence

12. **SQLite Event Store** (`src/data/event_store.py`)
    - 5 tables, CRUD operations, thread-safe
    - *Adaptation needed:* Migrate to PostgreSQL/SQLAlchemy, add more tables

13. **Model Manager** (`src/models/model_manager.py`)
    - Orchestrates training, compares by F1, saves best
    - *Adaptation needed:* Extend for 4 models (add IF, AE)

14. **Flow Feature Extraction** (`src/detection/flow_features.py`)
    - 68-feature extraction from network flows
    - *Adaptation needed:* Keep for cyber, create separate physical feature pipeline

15. **Attack Generators** (`scripts/attacks/`)
    - 4 DDoS attack types with full 68-feature generation
    - Thread-safe, rate-controlled
    - *Adaptation needed:* Extend for spoofing, brute force, port scan, false data injection

16. **IoT Device Generator** (`scripts/iot/device_generator.py`)
    - 6 device types with sensor readings + network features
    - *Adaptation needed:* Extend for physical anomaly scenarios

17. **Pytest Test Patterns** (`tests/`)
    - Good coverage of models, streaming, risk, preprocessing
    - *Adaptation needed:* Add physical model tests, dataset loader tests

18. **Model Training Scripts** (`scripts/train_models.py`)
    - CLI for training XGBoost, RF, SVM, K-Means
    - *Adaptation needed:* Add Isolation Forest, Autoencoder training

19. **Evaluation Scripts** (`scripts/evaluate_models.py`, `scripts/evaluate_comprehensive.py`)
    - Metrics, confusion matrix, ROC, PR curves, feature importance
    - *Adaptation needed:* Extend for physical anomaly metrics

20. **Preprocessing Pipeline** (`src/preprocessing/pipeline.py`)
    - Feature selection, label encoding, scaling, train/test split
    - *Adaptation needed:* Create separate physical preprocessing

---

## 4. What Must NOT Be Reused As-Is

| Component | Reason |
|-----------|--------|
| `src/detection/realtime_ids.py` | DDoS-only, captures live packets via Scapy — incompatible with streaming IoT sensor data |
| `src/detection/packet_capture.py` | Scapy live capture — not needed for virtual simulator pipeline |
| `src/detection/flow_features.py` (as sole feature pipeline) | 68 network features only — no physical sensor features |
| `src/data/loader.py` | Hard-coded for CICDDoS2019 — no multi-dataset abstraction |
| `src/models/xgboost_model.py` (as sole model) | DDoS binary classification only — needs multiclass + physical models |
| `src/models/svm_model.py` | Not needed — project uses IF, AE, RF, XGBoost only |
| `src/models/kmeans_model.py` | Not needed — project uses IF, AE, RF, XGBoost only |
| `src/streaming/stream_processor.py` | DDoS-only pipeline — no physical anomaly path |
| `src/api/routes.py` | DDoS-only endpoints — needs physical anomaly endpoints |
| `src/api/simulator_routes.py` | DDoS attack simulation only — needs physical anomaly simulation |
| `dashboard/` | Jinja2 HTML — project requires Streamlit |
| `src/data/event_store.py` (as final DB) | SQLite — project requires PostgreSQL |
| `src/security/risk_engine.py` (as-is) | DDoS-only risk scoring — needs physical + cyber unified scoring |
| `configs/features.yaml` | 68 CICDDoS2019 features — different features for physical datasets |
| `configs/model.yaml` | Includes SVM, K-Means — not needed in new project |

---

## 5. Gap List (Backlog)

### Critical Gaps (must build from scratch)

| # | Gap | Section Reference |
|---|-----|-------------------|
| 1 | **Physical anomaly detection** — Isolation Forest not present | §3, §18 |
| 2 | **Autoencoder** (neural network) not present | §3, §18 |
| 3 | **Multi-dataset abstraction** — BaseDatasetLoader, DatasetRegistry | §8 |
| 4 | **SWaT dataset loader** | §5 |
| 5 | **WADI dataset loader** | §5 |
| 6 | **CICIoT2023 dataset loader** | §6 |
| 7 | **Edge-IIoTSet dataset loader** | §7 |
| 8 | **Physical feature engineering** (rolling stats, rate of change, etc.) | §17 |
| 9 | **Physical detection pipeline** (dual-model agreement) | §18 |
| 10 | **Unified decision engine** (physical + cyber scores) | §20 |
| 11 | **Device health scoring** | §21 |
| 12 | **PostgreSQL integration** (SQLAlchemy) | §24 |
| 13 | **Streamlit dashboard** (9 pages) | §26 |
| 14 | **SHAP explainability** | §23 |
| 15 | **Cross-dataset validation** experiments | §30 |
| 16 | **Ablation study** experiments | §32 |
| 17 | **Physical anomaly simulator** (5 scenarios) | §12 |
| 18 | **Extended cyber attack simulator** (spoofing, brute force, port scan, false data injection) | §13 |

### Moderate Gaps (adapt from reference)

| # | Gap | Section Reference |
|---|-----|-------------------|
| 19 | **New Kafka topics** for physical/cyber dual pipeline | §14 |
| 20 | **Pydantic message schemas** for sensor + network messages | §15 |
| 21 | **Spark Structured Streaming** dual-path pipeline | §16 |
| 22 | **Multiclass RF/XGBoost** (ref has binary DDoS-only) | §4 |
| 23 | **Extended alert manager** with physical anomaly support | §22 |
| 24 | **FastAPI extended endpoints** for physical anomalies | §25 |
| 25 | **CLI commands** | §48 |
| 26 | **Data drift monitoring** | §45 |
| 27 | **Model versioning** with metadata | §43 |
| 28 | **Experiment tracking** (JSON metadata) | §41 |

### Minor Gaps (configuration/documentation)

| # | Gap | Section Reference |
|---|-----|-------------------|
| 29 | **Extended YAML configs** (models.yaml, simulation.yaml, database.yaml) | §35 |
| 30 | **Extended logging** (structured, performance metrics) | §36 |
| 31 | **Custom exception classes** | §37 |
| 32 | **API authentication** for admin endpoints | §40 |
| 33 | **Rate limiting** on API | §40 |
| 34 | **CORS configuration** | §40 |
| 35 | **Research experiment module** (reproducibility, metadata) | §41 |
| 36 | **Paper-ready tables** generation | §51 |
| 37 | **Research figures** generation | §52 |
| 38 | **README** with Mermaid diagrams | §60 |
| 39 | **Methodology documentation** | §60 |
| 40 | **Dataset documentation** | §60 |

---

## 6. Summary

### What DDoS-RealTime-IDS Does Well
- Complete Kafka + Spark streaming pipeline for DDoS detection
- 4 trained ML models (XGBoost production, RF, SVM, K-Means)
- 68-feature extraction from network flows
- Risk engine with configurable thresholds
- Mitigation engine (simulated)
- FastAPI REST API with WebSocket
- Docker Compose with health checks
- Comprehensive test suite (74+ tests)
- Evaluation reports with plots

### What It Lacks (vs. this project)
- No physical anomaly detection (no Isolation Forest, no Autoencoder)
- No multi-dataset support (CICDDoS2019 only)
- No PostgreSQL (SQLite only)
- No Streamlit (Jinja2 HTML only)
- No SHAP explainability
- No cross-dataset validation
- No unified cyber-physical risk scoring
- No device health scoring
- No physical anomaly simulation
- Limited cyber attack simulation (DDoS only, no spoofing/brute force/port scan)
- No Pydantic message schemas
- No SQLAlchemy ORM

### Migration Strategy
The new project (`iot-cyber-physical-security/`) will:
1. Adopt Kafka producer/consumer patterns from `src/kafka/`
2. Adapt config management from `src/config/`
3. Extend risk/decision engines from `src/security/`
4. Reuse attack generator framework from `scripts/attacks/`
5. Reuse IoT device generator concept from `scripts/iot/`
6. Reuse test patterns from `tests/`
7. Build new: physical detection, multi-dataset, PostgreSQL, Streamlit, SHAP, dual-pipeline

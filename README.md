# Real-Time Anomaly and Cyber Threat Detection for IoT Sensors

A production-grade IoT cybersecurity platform using Apache Kafka for real-time streaming and Apache Spark for distributed processing. Detects physical sensor anomalies (Isolation Forest ensemble + Autoencoder) and cyber threats (Random Forest + XGBoost) with a unified risk scoring engine.

## Architecture

```
IoT Sensors (SWaT/WADI)  -->  Apache Kafka  -->  Spark Structured Streaming
Network Traffic (CICIoT2023/Edge-IIoTSet)  -->       |
                                                     v
                                          +-------------------+
                                          |  Detection Models |
                                          |  - IF Ensemble    |
                                          |  - Autoencoder    |
                                          |  - Random Forest  |
                                          |  - XGBoost        |
                                          +-------------------+
                                                     |
                                                     v
                                          +-------------------+
                                          |  Decision Engine  |
                                          |  (Risk Scoring)   |
                                          +-------------------+
                                                     |
                                          +-------------------+
                                          | FastAPI | Streamlit|
                                          |  + Web Dashboard  |
                                          +-------------------+
```

## Quick Start (Docker - recommended)

```bash
# Start the full stack (Postgres, Kafka, Spark, API, dashboards, streaming, simulator)
docker compose up -d

# If containers start before Kafka is healthy, restart the data-plane services
docker compose restart simulator streaming replayer spark-job

# Follow streaming logs
docker compose logs -f streaming
```

### Services

| Service | URL |
|---------|-----|
| **Web Dashboard** | http://localhost:8000/dashboard |
| API (OpenAPI docs) | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |
| Spark Master UI | http://localhost:8080 |
| Kafka | localhost:9092 |
| PostgreSQL | localhost:5432 |

API requests require the header `X-API-Key: dev-api-key-12345`.

## Quick Start (local development)

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests (138 tests)
python -m pytest tests/ -v

# Train models
python scripts/train_physical_500k.py    # Physical: IF ensemble
python scripts/train_cyber_real.py       # Cyber: RF + XGBoost
python scripts/train_stacking_ensemble.py # Unified stacking

# Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Start Streamlit dashboard
streamlit run dashboard/app.py
```

CLI entry point: `iot-sec` (for example `iot-sec serve`, `iot-sec status`).

## Models

| Model | Domain | Dataset | Accuracy | F1 | ROC-AUC |
|-------|--------|---------|----------|-----|---------|
| IF Ensemble (9 IF + XGBoost) | Physical | SWaT+WADI | 99.67% | 0.983 | 0.9996 |
| Autoencoder (normal-only) | Physical | SWaT+WADI | 95.35% | 0.704 | 0.849 |
| Random Forest | Cyber | CICIoT2023 | 99.40% | 0.997 | 0.9994 |
| XGBoost | Cyber | CICIoT2023 | 99.62% | 0.998 | 0.9994 |
| Stacking Ensemble | Unified | All | 97.52% | 0.977 | 0.9895 |

## Datasets

| Dataset | Type | Rows | Features | Source |
|---------|------|------|----------|--------|
| SWaT | Physical | 1,441,719 | 51 | SUTD |
| WADI | Physical | 1,209,601 | 127 | SUTD |
| CICIoT2023 | Cyber | 7,570,414 | 46 | UNB CIC |
| Edge-IIoTSet | Cyber | 156,986 | 57 | AIMaster |

Datasets are downloaded into `data/raw/` (gitignored). See `python scripts/download_datasets.py --help`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/dashboard` | GET | Live web dashboard (dark theme) |
| `/api/v1/dashboard/health` | GET | Infrastructure health (Postgres, Kafka, Spark, models) |
| `/api/v1/dashboard/models` | GET | Trained model metrics |
| `/api/v1/devices` | GET | List all IoT devices |
| `/api/v1/alerts` | GET | Query alerts |
| `/api/v1/anomalies` | GET | Physical anomaly results |
| `/api/v1/threats` | GET | Cyber threat results |
| `/api/v1/metrics` | GET | System metrics |
| `/api/v1/models` | GET | Model training status |
| `/api/v1/statistics` | GET | Alert statistics |

Full interactive documentation at `/docs`.

## Project Structure

```
├── configs/          # YAML configs (Kafka, Spark, models, risk, simulation...)
├── dashboard/
│   ├── app.py        # Streamlit dashboard
│   └── web/          # Web dashboard (HTML/CSS/JS) served at /dashboard
├── docker/           # Per-service Dockerfiles + requirements
├── docs/             # Documentation, reference audit, research papers
├── experiments/      # Training results (JSON), figures, RESULTS.md
├── models/           # Trained model binaries (gitignored)
├── data/             # Datasets (gitignored)
├── logs/             # Runtime logs (gitignored)
├── notebooks/        # Analysis notebooks
├── scripts/          # Training, evaluation, replay, streaming entrypoints
├── src/
│   ├── api/          # FastAPI REST API with auth
│   ├── models/       # ML model wrappers (IF, AE, RF, XGBoost)
│   ├── datasets/     # Dataset loaders (SWaT, WADI, CICIoT2023, Edge-IIoTSet)
│   ├── preprocessing/# Feature scaling and encoding
│   ├── features/     # Feature engineering (rolling, rate-of-change)
│   ├── streaming/    # Spark Structured Streaming + processor
│   ├── kafka/        # Kafka producer/consumer
│   ├── database/     # SQLAlchemy models + repositories
│   ├── decision/     # Risk scoring engine
│   ├── alerts/       # Alert management
│   ├── simulation/   # IoT device + attack simulation
│   ├── monitoring/   # Performance tracking + drift detection
│   ├── explainability/# SHAP-based model explanations
│   ├── research/     # Cross-dataset validation, ablation studies
│   ├── cli/          # CLI with 9 subcommands
│   ├── config/       # Settings, YAML config loader, logging
│   └── utils/        # Helpers, exceptions, logger
├── tests/            # Unit, integration and performance tests
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Technology Stack

- **Streaming:** Apache Kafka 3.7, Apache Spark Structured Streaming
- **ML:** PyTorch (Autoencoder), scikit-learn (IF, RF), XGBoost
- **API:** FastAPI with API key auth
- **Dashboards:** Web dashboard (served by FastAPI) + Streamlit/Plotly
- **Database:** SQLAlchemy (SQLite/PostgreSQL)
- **Explainability:** SHAP TreeExplainer
- **Infrastructure:** Docker Compose, Makefile

## License

MIT

"""FastAPI application with full REST API and API key authentication."""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config.logging_config import setup_logging
from src.config.settings import API_HOST, API_PORT, create_required_directories, MODEL_ROOT
from src.database.models import init_db
from src.database.repositories import (
    DeviceRepository, AlertRepository, MetricRepository,
    PhysicalAnomalyRepository, CyberThreatRepository,
)
from src.api.auth import verify_api_key
from src.api.security import create_secure_headers

logger = logging.getLogger(__name__)

_device_repo: DeviceRepository | None = None
_alert_repo: AlertRepository | None = None
_metric_repo: MetricRepository | None = None
_physical_repo: PhysicalAnomalyRepository | None = None
_cyber_repo: CyberThreatRepository | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    create_required_directories()
    setup_logging()
    init_db()
    global _device_repo, _alert_repo, _metric_repo, _physical_repo, _cyber_repo
    _device_repo = DeviceRepository()
    _alert_repo = AlertRepository()
    _metric_repo = MetricRepository()
    _physical_repo = PhysicalAnomalyRepository()
    _cyber_repo = CyberThreatRepository()
    from src.api.auth import get_api_keys
    keys = get_api_keys()
    logger.info("API started. API key: %s... (first 8 chars)", keys[0][:8])
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="IoT Cyber-Physical Security API",
    description="Real-Time Anomaly and Cyber Threat Detection for IoT Sensors",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/v1/devices")
async def get_devices(api_key: str = Depends(verify_api_key)):
    devices = _device_repo.get_all() if _device_repo else []
    return {"devices": devices, "total": len(devices)}


@app.get("/api/v1/devices/{device_id}")
async def get_device(device_id: str, api_key: str = Depends(verify_api_key)):
    if not _device_repo:
        raise HTTPException(status_code=503, detail="Database not available")
    device = _device_repo.get_by_id(device_id)
    if not device:
        raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
    return device


@app.get("/api/v1/alerts")
async def get_alerts(limit: int = 100, api_key: str = Depends(verify_api_key)):
    alerts = _alert_repo.get_all(limit) if _alert_repo else []
    return {"alerts": alerts, "total": len(alerts)}


@app.get("/api/v1/alerts/{alert_id}")
async def get_alert(alert_id: str, api_key: str = Depends(verify_api_key)):
    if not _alert_repo:
        raise HTTPException(status_code=503, detail="Database not available")
    alerts = _alert_repo.get_all(limit=1000)
    for a in alerts:
        if a["alert_id"] == alert_id:
            return a
    raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")


@app.get("/api/v1/anomalies")
async def get_anomalies(limit: int = 100, api_key: str = Depends(verify_api_key)):
    anomalies = _physical_repo.get_all(limit) if _physical_repo else []
    return {"anomalies": anomalies, "total": len(anomalies)}


@app.get("/api/v1/threats")
async def get_threats(limit: int = 100, api_key: str = Depends(verify_api_key)):
    threats = _cyber_repo.get_all(limit) if _cyber_repo else []
    return {"threats": threats, "total": len(threats)}


@app.get("/api/v1/metrics")
async def get_metrics(api_key: str = Depends(verify_api_key)):
    metrics = {}
    if _metric_repo:
        for name in ["throughput", "latency_avg", "total_events", "anomalies_detected", "threats_detected"]:
            recent = _metric_repo.get_recent(name, limit=1)
            if recent:
                metrics[name] = recent[0]["metric_value"]
    return {"metrics": metrics}


@app.get("/api/v1/models")
async def get_models(api_key: str = Depends(verify_api_key)):
    models_info = {}
    model_dirs = {
        "isolation_forest": MODEL_ROOT / "physical" / "isolation_forest",
        "autoencoder": MODEL_ROOT / "physical" / "autoencoder",
        "random_forest": MODEL_ROOT / "cyber" / "random_forest",
        "xgboost": MODEL_ROOT / "cyber" / "xgboost",
    }
    for name, path in model_dirs.items():
        models_info[name] = {
            "status": "trained" if (path / "metadata.json").exists() else "not_trained",
            "path": str(path),
        }
    return {"models": models_info}


@app.get("/api/v1/statistics")
async def get_statistics(api_key: str = Depends(verify_api_key)):
    alert_stats = _alert_repo.get_statistics() if _alert_repo else {}
    return {
        "alerts": alert_stats,
        "devices": {"total": len(_device_repo.get_all()) if _device_repo else 0},
    }


@app.get("/api/v1/system/status")
async def system_status():
    return {
        "status": "running",
        "version": "1.0.0",
        "components": {
            "api": "online",
            "database": "online",
        },
    }


@app.post("/api/v1/simulation/start")
async def start_simulation(api_key: str = Depends(verify_api_key)):
    from src.pipeline.orchestrator import Pipeline, PipelineConfig
    global _pipeline
    if _pipeline and _pipeline._running:
        return {"status": "simulation already running", "stats": _pipeline.get_stats()}
    _pipeline = Pipeline(PipelineConfig(num_devices=5, events_per_cycle=10, attack_probability=0.2))
    _pipeline.initialize()
    result = _pipeline.run_cycle()
    return {"status": "simulation started", "cycle": result, "stats": _pipeline.get_stats()}


@app.post("/api/v1/simulation/stop")
async def stop_simulation(api_key: str = Depends(verify_api_key)):
    global _pipeline
    if _pipeline and _pipeline._running:
        _pipeline.stop()
        return {"status": "simulation stopped", "stats": _pipeline.get_stats()}
    return {"status": "simulation was not running"}


@app.post("/api/v1/simulation/anomaly")
async def inject_anomaly(api_key: str = Depends(verify_api_key)):
    from src.simulation.iot_simulator import IoTSimulator
    sim = IoTSimulator(num_devices=1)
    sensor_msgs, _ = sim.generate_once()
    if sensor_msgs:
        msg = sensor_msgs[0]
        # Inject anomalous value
        msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else msg.__dict__.copy()
        if "value" in msg_dict:
            msg_dict["value"] = msg_dict["value"] * 5.0
        return {"status": "physical anomaly injected", "device_id": msg_dict.get("device_id", "unknown")}
    return {"status": "no devices available"}


@app.post("/api/v1/simulation/attack")
async def inject_attack(api_key: str = Depends(verify_api_key)):
    from src.simulation.attack_simulator import CyberAttackSimulator
    from src.simulation.iot_simulator import IoTSimulator
    sim = IoTSimulator(num_devices=1)
    device_ids = sim.get_device_ids()
    attack_sim = CyberAttackSimulator()
    attack_type, attack_msg = attack_sim.generate_random_attack(device_id=device_ids[0] if device_ids else "unknown")
    return {"status": "cyber attack simulated", "attack_type": attack_type, "device_id": device_ids[0] if device_ids else "unknown"}


_pipeline = None

# ---------------------------------------------------------------------------
# Dashboard: serve static files + dashboard-specific JSON endpoints
# ---------------------------------------------------------------------------
_DASHBOARD_DIR = Path(__file__).resolve().parent.parent.parent / "dashboard" / "web"
_STATIC_DIR = _DASHBOARD_DIR / "static"
_TEMPLATES_DIR = _DASHBOARD_DIR / "templates"

app.mount("/dashboard/static", StaticFiles(directory=str(_STATIC_DIR)), name="dashboard-static")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_index():
    index_file = _TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/v1/dashboard/health")
async def dashboard_health():
    services: dict[str, str] = {}

    # Database
    try:
        if _device_repo:
            _device_repo.get_all()
            services["PostgreSQL"] = "ONLINE"
        else:
            services["PostgreSQL"] = "OFFLINE"
    except Exception:
        services["PostgreSQL"] = "OFFLINE"

    # Kafka
    try:
        from kafka import KafkaConsumer
        bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        consumer = KafkaConsumer(bootstrap_servers=bootstrap, request_timeout_ms=3000)
        consumer.topics()
        consumer.close()
        services["Kafka"] = "ONLINE"
    except Exception:
        services["Kafka"] = "OFFLINE"

    # Spark Master
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get("http://spark-master:8080/json/")
            services["Spark Master"] = "ONLINE" if r.status_code == 200 else "OFFLINE"
    except Exception:
        services["Spark Master"] = "OFFLINE"

    # Spark Worker
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get("http://spark-worker:8081/json/")
            services["Spark Worker"] = "ONLINE" if r.status_code == 200 else "OFFLINE"
    except Exception:
        services["Spark Worker"] = "OFFLINE"

    # ML Models
    try:
        if_model = MODEL_ROOT / "physical" / "isolation_forest" / "metadata.json"
        rf_model = MODEL_ROOT / "cyber" / "random_forest" / "random_forest.joblib"
        xgb_model = MODEL_ROOT / "cyber" / "xgboost" / "xgboost.joblib"
        loaded = sum(1 for p in [if_model, rf_model, xgb_model] if p.exists())
        services["ML Models"] = f"LOADED ({loaded}/3)"
    except Exception:
        services["ML Models"] = "MISSING"

    return {"services": services}


@app.get("/api/v1/dashboard/models")
async def dashboard_models():
    results_dir = Path(__file__).resolve().parent.parent.parent / "experiments" / "results"
    models = []

    # Physical: IF ensemble (99.67%) — best physical model
    if_file = results_dir / "if_ensemble_results.json"
    if if_file.exists():
        d = json.loads(if_file.read_text())
        models.append({
            "name": "IF Ensemble (Physical)",
            "accuracy": d.get("accuracy", 0),
            "precision": d.get("precision", 0),
            "recall": d.get("recall", 0),
            "f1": d.get("f1", 0),
            "auc": d.get("roc_auc", 0),
        })

    # Physical: Autoencoder (combined SWaT+WADI)
    ae_file = results_dir / "physical_combined_results.json"
    if ae_file.exists():
        d = json.loads(ae_file.read_text())
        m = d.get("models", {}).get("autoencoder", {}).get("metrics", {})
        models.append({
            "name": "Autoencoder (Physical)",
            "accuracy": m.get("accuracy", 0),
            "precision": m.get("precision", 0),
            "recall": m.get("recall", 0),
            "f1": m.get("f1", 0),
            "auc": m.get("roc_auc", 0),
        })

    # Cyber: RF + XGB on CICIoT2023 (binary, best real results)
    cyber_file = results_dir / "cyber_real_results.json"
    if cyber_file.exists():
        d = json.loads(cyber_file.read_text())
        rf = d.get("rf_ciciot2023", {})
        if rf:
            models.append({
                "name": "Random Forest (Cyber)",
                "accuracy": rf.get("accuracy", 0),
                "precision": rf.get("precision", 0),
                "recall": rf.get("recall", 0),
                "f1": rf.get("f1", 0),
                "auc": rf.get("roc_auc", 0),
            })
        xgb = d.get("xgb_ciciot2023", {})
        if xgb:
            models.append({
                "name": "XGBoost (Cyber)",
                "accuracy": xgb.get("accuracy", 0),
                "precision": xgb.get("precision", 0),
                "recall": xgb.get("recall", 0),
                "f1": xgb.get("f1", 0),
                "auc": xgb.get("roc_auc", 0),
            })

    # Stacking ensemble
    stack_file = results_dir / "stacking_ensemble_results.json"
    if stack_file.exists():
        d = json.loads(stack_file.read_text())
        fm = d.get("final_metrics", {})
        models.append({
            "name": "Stacking Ensemble",
            "accuracy": fm.get("accuracy", 0),
            "precision": fm.get("precision", 0),
            "recall": fm.get("recall", 0),
            "f1": fm.get("f1", 0),
            "auc": fm.get("roc_auc", 0),
        })

    return {"models": models}

"""Central settings and path definitions."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
MODEL_ROOT = PROJECT_ROOT / "models"
CONFIG_ROOT = PROJECT_ROOT / "configs"
LOG_ROOT = PROJECT_ROOT / "logs"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{PROCESSED_DATA_ROOT / 'iot_security.db'}",
)
KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
SPARK_MASTER: str = os.getenv("SPARK_MASTER", "local[*]")
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8501"))
SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production")

PHYSICAL_SENSORS = ["temperature", "humidity", "pressure", "voltage", "battery"]
CYBER_FEATURES = [
    "packet_count", "packet_rate", "bytes", "flow_duration",
    "source_port", "destination_port", "protocol",
    "connection_count", "failed_connection_count",
    "unique_destination_ports", "bytes_per_packet", "packets_per_second",
]

PHYSICAL_MODEL_PATHS = {
    "isolation_forest": MODEL_ROOT / "physical" / "isolation_forest",
    "autoencoder": MODEL_ROOT / "physical" / "autoencoder",
}
CYBER_MODEL_PATHS = {
    "random_forest": MODEL_ROOT / "cyber" / "random_forest",
    "xgboost": MODEL_ROOT / "cyber" / "xgboost",
}


def create_required_directories() -> None:
    dirs = [
        DATA_ROOT, RAW_DATA_ROOT, PROCESSED_DATA_ROOT,
        MODEL_ROOT, LOG_ROOT, EXPERIMENTS_ROOT,
        EXPERIMENTS_ROOT / "results", EXPERIMENTS_ROOT / "figures", EXPERIMENTS_ROOT / "reports",
    ]
    for model_dir in list(PHYSICAL_MODEL_PATHS.values()) + list(CYBER_MODEL_PATHS.values()):
        dirs.append(model_dir)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

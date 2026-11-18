"""Entrypoint for the spark-job Docker service.

Runs the Spark Structured Streaming SQL analytics pipelines against Kafka.
Environment:
    KAFKA_BOOTSTRAP_SERVERS  Kafka bootstrap (default kafka:29092)
    SPARK_MASTER             Spark master URL (default local[*])
    SPARK_CHECKPOINT_DIR     Checkpoint root (default /tmp/spark-checkpoints)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.logging_config import setup_logging

logger = setup_logging()


def main() -> None:
    from src.streaming.spark import run_full_pipeline

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    os.environ.setdefault("SPARK_CHECKPOINT_DIR", "/tmp/spark-checkpoints")
    logger.info("spark-job starting: kafka=%s master=%s",
                bootstrap, os.getenv("SPARK_MASTER", "local[*]"))
    run_full_pipeline(kafka_bootstrap=bootstrap)


if __name__ == "__main__":
    main()

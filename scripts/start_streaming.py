"""Start the streaming detection pipeline.

Modes (env STREAMING_MODE):
    python - ML inference pipeline: consumes sensor/network/replay topics,
             runs physical (IF) + cyber (RF/XGB) models, decision engine,
             alerts and persistence to Postgres. (default)
    spark  - Spark Structured Streaming SQL analytics on the simulator topics.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.logging_config import setup_logging
from src.config.settings import create_required_directories
from src.config.loader import load_config

logger = logging.getLogger(__name__)

_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received")
    _running = False


def _bootstrap_servers() -> str:
    kafka_config = load_config("kafka")
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS") or kafka_config.kafka.bootstrap_servers


def main():
    setup_logging()
    create_required_directories()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    mode = os.getenv("STREAMING_MODE", "python").lower()
    bootstrap = _bootstrap_servers()

    if mode == "spark":
        from src.streaming.spark import run_full_pipeline
        logger.info("Starting Spark Structured Streaming SQL analytics...")
        run_full_pipeline(kafka_bootstrap=bootstrap)
        return

    _run_python_processor(bootstrap)


def _run_python_processor(bootstrap: str):
    from src.streaming.processor import StreamingProcessor
    from src.kafka.consumer import KafkaConsumer

    kafka_config = load_config("kafka")
    topics = [
        kafka_config.kafka.topics.physical_replay,
        kafka_config.kafka.topics.cyber_replay,
        kafka_config.kafka.topics.sensor_raw,
        kafka_config.kafka.topics.network_raw,
    ]

    processor = StreamingProcessor()
    logger.info("Connecting to Kafka at %s, topics: %s", bootstrap, topics)
    consumer = KafkaConsumer(
        bootstrap_servers=bootstrap,
        topics=topics,
        group_id="iot-security-streaming",
    )
    consumer.connect()

    logger.info("Python ML streaming pipeline started. Processing messages...")
    last_stats_time = time.time()

    while _running:
        messages = consumer.poll(timeout_ms=1000)
        for msg in messages:
            processor.process_message(msg)

        if time.time() - last_stats_time > 30:
            logger.info("Pipeline stats: %s", processor.get_stats())
            last_stats_time = time.time()

    logger.info("Final stats: %s", processor.get_stats())
    consumer.close()
    logger.info("Streaming pipeline stopped")


if __name__ == "__main__":
    main()

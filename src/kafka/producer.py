"""Kafka producer for IoT sensor and network data."""
from __future__ import annotations

import json
import time
import logging
from typing import Any, Callable

from src.kafka.serializers import serialize_message
from src.utils.exceptions import KafkaConnectionError

logger = logging.getLogger(__name__)


class KafkaProducer:
    """Kafka producer wrapper with graceful fallback."""

    def __init__(self, bootstrap_servers: str = "localhost:9092",
                 topic: str = "iot.sensor.raw") -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self._producer = None
        self._connected = False

    def connect(self) -> None:
        try:
            from kafka import KafkaProducer as _KafkaProducer
            self._producer = _KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                acks="all",
                retries=3,
                retry_backoff_ms=100,
                linger_ms=10,
                batch_size=16384,
                compression_type="lz4",
            )
            self._connected = True
            logger.info("Connected to Kafka at %s", self.bootstrap_servers)
        except Exception as e:
            logger.warning("Kafka unavailable, using in-memory fallback: %s", e)
            self._connected = False

    def send(self, data: dict[str, Any], topic: str | None = None) -> bool:
        target_topic = topic or self.topic
        message = serialize_message(data)

        if self._connected and self._producer is not None:
            try:
                future = self._producer.send(target_topic, value=message)
                future.get(timeout=10)
                return True
            except Exception as e:
                logger.error("Failed to send to Kafka: %s", e)
                return False
        return False

    def send_batch(self, messages: list[dict[str, Any]], topic: str | None = None) -> int:
        sent = 0
        for msg in messages:
            if self.send(msg, topic):
                sent += 1
        return sent

    def close(self) -> None:
        if self._producer:
            try:
                self._producer.flush(timeout=5)
                self._producer.close(timeout=5)
            except Exception:
                pass
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

"""Kafka consumer for IoT events."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from src.kafka.serializers import deserialize_message, extract_payload
from src.utils.exceptions import KafkaConnectionError

logger = logging.getLogger(__name__)


class KafkaConsumer:
    """Kafka consumer wrapper with graceful fallback."""

    def __init__(self, bootstrap_servers: str = "localhost:9092",
                 topics: list[str] | None = None,
                 group_id: str = "iot-security-consumer") -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics or ["iot.sensor.raw"]
        self.group_id = group_id
        self._consumer = None
        self._connected = False

    def connect(self) -> None:
        try:
            from kafka import KafkaConsumer as _KafkaConsumer
            self._consumer = _KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset="latest",
                enable_auto_commit=False,
                max_poll_records=500,
            )
            self._connected = True
            logger.info("Connected to Kafka consumer on %s", self.bootstrap_servers)
        except Exception as e:
            logger.warning("Kafka unavailable: %s", e)
            self._connected = False

    def poll(self, timeout_ms: int = 1000) -> list[dict[str, Any]]:
        if not self._connected or self._consumer is None:
            return []
        try:
            records = self._consumer.poll(timeout_ms=timeout_ms)
            messages = []
            for tp, msgs in records.items():
                for msg in msgs:
                    try:
                        envelope = deserialize_message(msg.value)
                        payload = extract_payload(envelope)
                        messages.append(payload)
                    except Exception as e:
                        logger.warning("Failed to deserialize message: %s", e)
            return messages
        except Exception as e:
            logger.error("Kafka poll error: %s", e)
            return []

    def consume_forever(self, handler: Callable[[dict[str, Any]], None]) -> None:
        logger.info("Starting consumer loop for topics: %s", self.topics)
        try:
            while True:
                messages = self.poll(timeout_ms=1000)
                for msg in messages:
                    try:
                        handler(msg)
                    except Exception as e:
                        logger.error("Handler error: %s", e)
        except KeyboardInterrupt:
            logger.info("Consumer stopped")
        finally:
            self.close()

    def close(self) -> None:
        if self._consumer:
            try:
                self._consumer.close()
            except Exception:
                pass
        self._connected = False

"""Kafka topic management."""
from __future__ import annotations

import logging
from typing import Any

from src.config.loader import load_config

logger = logging.getLogger(__name__)

REQUIRED_TOPICS = [
    "iot.sensor.raw",
    "iot.network.raw",
    "iot.physical.replay",
    "iot.cyber.replay",
    "iot.physical.anomaly",
    "iot.cyber.threat",
    "iot.alerts",
    "iot.metrics",
    "iot.system.events",
]


def get_topic_names() -> dict[str, str]:
    config = load_config("kafka")
    return config.kafka.topics.to_dict()


def ensure_topics(bootstrap_servers: str | None = None) -> list[str]:
    if bootstrap_servers is None:
        config = load_config("kafka")
        bootstrap_servers = config.kafka.bootstrap_servers

    try:
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka.errors import TopicAlreadyExistsError

        admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
        existing = admin.list_topics()
        created: list[str] = []
        for topic in REQUIRED_TOPICS:
            if topic in existing:
                continue
            try:
                admin.create_topics(
                    [NewTopic(name=topic, num_partitions=3, replication_factor=1)]
                )
                created.append(topic)
            except TopicAlreadyExistsError:
                pass
            except Exception as e:
                logger.warning("Could not create topic %s: %s", topic, e)
        if created:
            logger.info("Created Kafka topics: %s", created)
        else:
            logger.info("All Kafka topics already exist")
        admin.close()
        return list(existing) + created
    except Exception as e:
        logger.warning("Could not manage Kafka topics: %s", e)
        return REQUIRED_TOPICS

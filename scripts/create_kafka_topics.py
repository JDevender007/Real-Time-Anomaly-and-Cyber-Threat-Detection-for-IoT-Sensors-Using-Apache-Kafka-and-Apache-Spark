"""Create Kafka topics for the IoT security platform."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config


def create_topics():
    kafka_config = load_config("kafka")
    bootstrap = kafka_config.kafka.bootstrap_servers
    topics_config = kafka_config.kafka.topics

    topic_list = [
        (topics_config.sensor_raw, 6, 3),
        (topics_config.network_raw, 6, 3),
        (topics_config.physical_anomaly, 3, 3),
        (topics_config.cyber_threat, 3, 3),
        (topics_config.alerts, 3, 3),
    ]

    try:
        from kafka import KafkaAdminClient
        from kafka.admin import NewTopic

        admin = KafkaAdminClient(bootstrap_servers=bootstrap, client_id="iot-security-admin")

        existing = set(admin.list_topics())
        to_create = []
        for name, partitions, replication in topic_list:
            if name not in existing:
                to_create.append(NewTopic(
                    name=name, num_partitions=partitions,
                    replication_factor=replication,
                ))
            else:
                print(f"  Topic '{name}' already exists ({partitions} partitions)")

        if to_create:
            result = admin.create_topics(to_create, validate_only=False)
            for topic, error in result.items():
                if error is None:
                    print(f"  Created topic '{topic}'")
                else:
                    print(f"  Failed to create '{topic}': {error}")

        admin.close()
        print(f"\nKafka topics ready at {bootstrap}")

    except ImportError:
        print("kafka-python not installed. Install with: pip install kafka-python")
        print("\nTopics that would be created:")
        for name, partitions, replication in topic_list:
            print(f"  {name} (partitions={partitions}, replication={replication})")

    except Exception as e:
        print(f"Kafka not available at {bootstrap}: {e}")
        print("\nTopics to create when Kafka is running:")
        for name, partitions, replication in topic_list:
            print(f"  {name} (partitions={partitions}, replication={replication})")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create Kafka topics")
    parser.add_argument("--list", action="store_true", help="List topics only")
    args = parser.parse_args()
    create_topics()


if __name__ == "__main__":
    main()

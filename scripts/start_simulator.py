"""Start the IoT device simulator with Kafka producer."""
from __future__ import annotations

import json
import logging
import random
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.logging_config import setup_logging
from src.config.settings import create_required_directories
from src.config.loader import load_config
from src.simulation.iot_simulator import IoTSimulator
from src.simulation.attack_simulator import CyberAttackSimulator
from src.kafka.producer import KafkaProducer

logger = logging.getLogger(__name__)

_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received")
    _running = False


def main():
    setup_logging()
    create_required_directories()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    sim_config = load_config("simulation")
    kafka_config = load_config("kafka")

    num_devices = getattr(sim_config.simulation, "num_devices", 50)
    rate = getattr(sim_config.simulation, "messages_per_second", 10)
    enable_attacks = getattr(sim_config.simulation, "enable_attack_simulation", True)

    bootstrap = kafka_config.kafka.bootstrap_servers
    sensor_topic = kafka_config.kafka.topics.sensor_raw
    network_topic = kafka_config.kafka.topics.network_raw

    logger.info("Initializing simulator with %d devices at %d msg/s", num_devices, rate)
    simulator = IoTSimulator(num_devices=num_devices, config=sim_config.simulation.__dict__
                             if hasattr(sim_config.simulation, "__dict__") else {})

    producer = KafkaProducer(bootstrap_servers=bootstrap, topic=sensor_topic)
    for attempt in range(1, 31):
        producer.connect()
        if producer.is_connected:
            break
        logger.warning("Kafka not ready (attempt %d/30), retrying in 2s...", attempt)
        time.sleep(2)
    if not producer.is_connected:
        logger.error("Kafka unavailable after 30 attempts - exiting")
        return

    sensor_count = 0
    network_count = 0

    def on_message(msg):
        nonlocal sensor_count, network_count
        topic = sensor_topic if hasattr(msg, "temperature") else network_topic
        data = msg.model_dump() if hasattr(msg, "model_dump") else msg.__dict__
        sent = producer.send(data, topic=topic)
        if sent:
            if topic == sensor_topic:
                sensor_count += 1
            else:
                network_count += 1

    simulator.register_callback(on_message)

    attack_sim = None
    if enable_attacks:
        try:
            attack_sim = CyberAttackSimulator()
            logger.info("Attack simulation enabled")
        except Exception as e:
            logger.warning("Attack simulator init failed: %s", e)

    logger.info("Simulator started. Press Ctrl+C to stop.")
    last_log = time.time()

    try:
        while _running:
            sensors, network_events = simulator.generate_once()
            for s in sensors:
                data = s.model_dump() if hasattr(s, "model_dump") else s.__dict__
                producer.send(data, topic=sensor_topic)
                sensor_count += 1
            for n in network_events:
                data = n.model_dump() if hasattr(n, "model_dump") else n.__dict__
                producer.send(data, topic=network_topic)
                network_count += 1

            if attack_sim and _running:
                try:
                    target_device = random.choice(list(simulator.devices.keys()))
                    attack_type, attack_msg = attack_sim.generate_random_attack(target_device)
                    if attack_msg:
                        data = attack_msg.model_dump() if hasattr(attack_msg, "model_dump") else attack_msg.__dict__
                        producer.send(data, topic=network_topic)
                        network_count += 1
                except Exception as e:
                    logger.warning("Attack simulation error: %s", e)

            if time.time() - last_log > 10:
                logger.info("Sent %d sensor + %d network messages", sensor_count, network_count)
                last_log = time.time()

            time.sleep(1.0 / max(rate, 1))

    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Final: %d sensor + %d network messages sent", sensor_count, network_count)
        producer.close()
        logger.info("Simulator stopped")


if __name__ == "__main__":
    main()

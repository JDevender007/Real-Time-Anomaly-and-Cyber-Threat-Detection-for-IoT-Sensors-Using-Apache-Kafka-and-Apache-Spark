"""Cyber attack simulator for testing."""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Any

from src.api.schemas import NetworkMessage
from src.utils.helpers import utc_timestamp


class CyberAttackSimulator:
    """Generates synthetic cyber attack network traffic."""

    ATTACK_TYPES = ["DDoS", "Spoofing", "BruteForce", "PortScan", "FalseDataInjection"]

    def __init__(self, local_ip_prefix: str = "10.0.0") -> None:
        self.local_ip_prefix = local_ip_prefix

    def simulate_ddoS(self, device_id: str) -> NetworkMessage:
        return NetworkMessage(
            device_id=device_id,
            timestamp=utc_timestamp(),
            source_ip=f"{self.local_ip_prefix}.{random.randint(10, 200)}",
            destination_ip=f"{self.local_ip_prefix}.1",
            source_port=random.randint(1024, 65535),
            destination_port=80,
            protocol="TCP",
            packet_count=random.randint(500, 10000),
            packet_rate=round(random.uniform(100, 1000), 2),
            bytes=random.randint(50000, 500000),
        )

    def simulate_spoofing(self, device_id: str) -> NetworkMessage:
        return NetworkMessage(
            device_id=device_id,
            timestamp=utc_timestamp(),
            source_ip=f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
            destination_ip=f"{self.local_ip_prefix}.{random.randint(10, 200)}",
            source_port=random.randint(1024, 65535),
            destination_port=random.choice([22, 80, 443]),
            protocol="TCP",
            packet_count=random.randint(10, 100),
            packet_rate=round(random.uniform(1, 10), 2),
            bytes=random.randint(200, 5000),
        )

    def simulate_brute_force(self, device_id: str) -> NetworkMessage:
        return NetworkMessage(
            device_id=device_id,
            timestamp=utc_timestamp(),
            source_ip=f"{self.local_ip_prefix}.{random.randint(10, 200)}",
            destination_ip=f"{self.local_ip_prefix}.1",
            source_port=random.randint(1024, 65535),
            destination_port=22,
            protocol="TCP",
            packet_count=random.randint(50, 500),
            packet_rate=round(random.uniform(10, 100), 2),
            bytes=random.randint(500, 5000),
        )

    def simulate_port_scan(self, device_id: str) -> NetworkMessage:
        return NetworkMessage(
            device_id=device_id,
            timestamp=utc_timestamp(),
            source_ip=f"{self.local_ip_prefix}.{random.randint(10, 200)}",
            destination_ip=f"{self.local_ip_prefix}.{random.randint(10, 200)}",
            source_port=random.randint(1024, 65535),
            destination_port=random.randint(1, 65535),
            protocol="TCP",
            packet_count=random.randint(5, 50),
            packet_rate=round(random.uniform(1, 20), 2),
            bytes=random.randint(40, 500),
        )

    def simulate_false_data_injection(self, device_id: str) -> NetworkMessage:
        return NetworkMessage(
            device_id=device_id,
            timestamp=utc_timestamp(),
            source_ip=f"{self.local_ip_prefix}.{random.randint(10, 200)}",
            destination_ip=f"{self.local_ip_prefix}.1",
            source_port=random.randint(1024, 65535),
            destination_port=random.choice([80, 443, 8080]),
            protocol="TCP",
            packet_count=random.randint(1, 20),
            packet_rate=round(random.uniform(0.5, 5), 2),
            bytes=random.randint(100, 2000),
        )

    def generate_attack(self, attack_type: str, device_id: str) -> NetworkMessage:
        generators = {
            "DDoS": self.simulate_ddoS,
            "Spoofing": self.simulate_spoofing,
            "BruteForce": self.simulate_brute_force,
            "PortScan": self.simulate_port_scan,
            "FalseDataInjection": self.simulate_false_data_injection,
        }
        gen = generators.get(attack_type, self.simulate_ddoS)
        return gen(device_id)

    def generate_random_attack(self, device_id: str) -> tuple[str, NetworkMessage]:
        attack_type = random.choice(self.ATTACK_TYPES)
        return attack_type, self.generate_attack(attack_type, device_id)

"""Virtual IoT device simulator."""
from __future__ import annotations

import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from src.api.schemas import PhysicalSensorMessage, NetworkMessage
from src.config.loader import load_config
from src.utils.helpers import utc_timestamp


class VirtualIoTDevice:
    """Simulates a single IoT device generating sensor and network data."""

    def __init__(self, device_id: str, config: dict[str, Any] | None = None) -> None:
        self.device_id = device_id
        self.config = config or {}
        self._sensor_state: dict[str, float] = {}
        self._init_state()

    def _init_state(self) -> None:
        ranges = self.config.get("sensor_ranges", {})
        self._sensor_state = {
            "temperature": random.uniform(
                ranges.get("temperature", {}).get("min", 20),
                ranges.get("temperature", {}).get("max", 35),
            ),
            "humidity": random.uniform(
                ranges.get("humidity", {}).get("min", 30),
                ranges.get("humidity", {}).get("max", 80),
            ),
            "pressure": random.uniform(
                ranges.get("pressure", {}).get("min", 950),
                ranges.get("pressure", {}).get("max", 1050),
            ),
            "voltage": random.uniform(
                ranges.get("voltage", {}).get("min", 3.0),
                ranges.get("voltage", {}).get("max", 5.0),
            ),
            "battery": random.uniform(
                ranges.get("battery", {}).get("min", 20),
                ranges.get("battery", {}).get("max", 100),
            ),
        }

    def generate_sensor_reading(self) -> PhysicalSensorMessage:
        for key in self._sensor_state:
            self._sensor_state[key] += random.gauss(0, 0.1)
            self._sensor_state[key] = max(0, self._sensor_state[key])

        return PhysicalSensorMessage(
            device_id=self.device_id,
            timestamp=utc_timestamp(),
            temperature=round(self._sensor_state["temperature"], 2),
            humidity=round(self._sensor_state["humidity"], 2),
            pressure=round(self._sensor_state["pressure"], 2),
            voltage=round(self._sensor_state["voltage"], 2),
            battery=round(self._sensor_state["battery"], 2),
        )

    def generate_network_event(self) -> NetworkMessage:
        local_prefix = self.config.get("network", {}).get("local_ip_prefix", "10.0.0")
        return NetworkMessage(
            device_id=self.device_id,
            timestamp=utc_timestamp(),
            source_ip=f"{local_prefix}.{random.randint(10, 200)}",
            destination_ip=f"{local_prefix}.{random.randint(10, 200)}",
            source_port=random.randint(1024, 65535),
            destination_port=random.choice([80, 443, 8080, 22, 53]),
            protocol=random.choice(["TCP", "UDP"]),
            packet_count=random.randint(1, 100),
            packet_rate=round(random.uniform(0.1, 50.0), 2),
            bytes=random.randint(64, 10000),
        )


class IoTSimulator:
    """Manages multiple virtual IoT devices."""

    def __init__(self, num_devices: int = 50, config: dict[str, Any] | None = None) -> None:
        self.num_devices = num_devices
        self.config = config or {}
        self.devices: dict[str, VirtualIoTDevice] = {}
        self._running = False
        self._callbacks: list[Callable] = []
        self._init_devices()

    def _init_devices(self) -> None:
        prefix = self.config.get("devices", {}).get("device_prefix", "device")
        padding = self.config.get("devices", {}).get("id_padding", 3)
        for i in range(self.num_devices):
            device_id = f"{prefix}_{str(i + 1).zfill(padding)}"
            self.devices[device_id] = VirtualIoTDevice(device_id, self.config)

    def register_callback(self, callback: Callable) -> None:
        self._callbacks.append(callback)

    def generate_once(self) -> tuple[list[PhysicalSensorMessage], list[NetworkMessage]]:
        sensors = []
        network = []
        for device in self.devices.values():
            sensors.append(device.generate_sensor_reading())
            network.append(device.generate_network_event())
        return sensors, network

    def start(self, rate: int = 10) -> None:
        self._running = True
        interval = 1.0 / max(rate, 1)
        while self._running:
            sensors, network_events = self.generate_once()
            for callback in self._callbacks:
                for s in sensors:
                    callback(s)
                for n in network_events:
                    callback(n)
            time.sleep(interval)

    def stop(self) -> None:
        self._running = False

    def get_device_ids(self) -> list[str]:
        return list(self.devices.keys())

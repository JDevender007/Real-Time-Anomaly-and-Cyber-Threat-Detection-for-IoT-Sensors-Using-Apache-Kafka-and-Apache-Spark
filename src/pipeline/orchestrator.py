"""Pipeline orchestrator: coordinates all subsystems into a unified pipeline."""
from __future__ import annotations

import logging
import time
import signal
import sys
from dataclasses import dataclass, field
from typing import Any

from src.config.settings import RANDOM_SEED
from src.streaming.processor import StreamingProcessor
from src.simulation.iot_simulator import IoTSimulator
from src.simulation.attack_simulator import CyberAttackSimulator

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    num_devices: int = 10
    events_per_cycle: int = 50
    attack_probability: float = 0.1
    cycle_interval_s: float = 1.0
    max_cycles: int = 0  # 0 = infinite


@dataclass
class PipelineStats:
    total_cycles: int = 0
    total_events: int = 0
    physical_events: int = 0
    cyber_events: int = 0
    attacks_injected: int = 0
    alerts_generated: int = 0
    errors: int = 0
    start_time: float = 0.0

    @property
    def uptime_s(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0

    @property
    def events_per_second(self) -> float:
        return self.total_events / self.uptime_s if self.uptime_s > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cycles": self.total_cycles,
            "total_events": self.total_events,
            "physical_events": self.physical_events,
            "cyber_events": self.cyber_events,
            "attacks_injected": self.attacks_injected,
            "alerts_generated": self.alerts_generated,
            "errors": self.errors,
            "uptime_s": round(self.uptime_s, 1),
            "events_per_second": round(self.events_per_second, 1),
        }


class Pipeline:
    """Main pipeline orchestrator.

    Coordinates:
    - IoT Simulator (sensor readings + network events)
    - Attack Simulator (DDoS, spoofing, brute force, etc.)
    - StreamingProcessor (model inference + decision engine + alerts)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.stats = PipelineStats()
        self._running = False
        self._processor: StreamingProcessor | None = None
        self._simulator: IoTSimulator | None = None
        self._attack_sim: CyberAttackSimulator | None = None

    def initialize(self) -> None:
        logger.info("Initializing pipeline...")
        self._processor = StreamingProcessor()
        self._simulator = IoTSimulator(num_devices=self.config.num_devices)
        self._attack_sim = CyberAttackSimulator()
        self.stats = PipelineStats(start_time=time.time())
        logger.info("Pipeline initialized with %d devices", self.config.num_devices)

    def run_cycle(self) -> dict[str, Any]:
        if not self._simulator or not self._processor:
            self.initialize()

        cycle_start = time.time()
        cycle_events = 0
        cycle_alerts = 0

        # Generate sensor readings
        sensor_msgs, network_msgs = self._simulator.generate_once()

        for msg in sensor_msgs:
            try:
                data = msg.model_dump() if hasattr(msg, "model_dump") else msg.__dict__
                result = self._processor.process_message(data)
                self.stats.physical_events += 1
                cycle_events += 1
                if result and isinstance(result, dict) and result.get("risk"):
                    cycle_alerts += 1
            except Exception as e:
                logger.debug("Error processing sensor msg: %s", e)
                self.stats.errors += 1

        # Generate network events (some may be attacks)
        for msg in network_msgs:
            try:
                data = msg.model_dump() if hasattr(msg, "model_dump") else msg.__dict__
                result = self._processor.process_message(data)
                self.stats.cyber_events += 1
                cycle_events += 1
                if result and isinstance(result, dict) and result.get("risk"):
                    cycle_alerts += 1
            except Exception as e:
                logger.debug("Error processing network msg: %s", e)
                self.stats.errors += 1

        # Inject attacks
        import random
        if random.random() < self.config.attack_probability and self._attack_sim:
            attack_type, attack_msg = self._attack_sim.generate_random_attack(
                device_id=random.choice(self._simulator.get_device_ids())
            )
            try:
                data = attack_msg.model_dump() if hasattr(attack_msg, "model_dump") else attack_msg.__dict__
                self._processor.process_message(data)
                self.stats.attacks_injected += 1
                self.stats.cyber_events += 1
                cycle_events += 1
            except Exception as e:
                logger.debug("Error processing attack: %s", e)
                self.stats.errors += 1

        self.stats.total_cycles += 1
        self.stats.total_events += cycle_events
        self.stats.alerts_generated += cycle_alerts

        cycle_time = time.time() - cycle_start
        return {
            "cycle": self.stats.total_cycles,
            "events": cycle_events,
            "alerts": cycle_alerts,
            "cycle_time_ms": round(cycle_time * 1000, 1),
        }

    def run(self, max_cycles: int = 0) -> None:
        self._running = True
        max_cycles = max_cycles or self.config.max_cycles
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info("Starting pipeline (max_cycles=%d, interval=%.1fs)",
                     max_cycles, self.config.cycle_interval_s)

        try:
            while self._running:
                result = self.run_cycle()
                if max_cycles > 0 and self.stats.total_cycles >= max_cycles:
                    break
                time.sleep(self.config.cycle_interval_s)
        finally:
            self._running = False
            logger.info("Pipeline stopped. Stats: %s", self.stats.to_dict())

    def stop(self) -> None:
        self._running = False

    def _handle_signal(self, signum: int, frame: Any) -> None:
        logger.info("Received signal %d, stopping...", signum)
        self.stop()

    def get_stats(self) -> dict[str, Any]:
        return self.stats.to_dict()

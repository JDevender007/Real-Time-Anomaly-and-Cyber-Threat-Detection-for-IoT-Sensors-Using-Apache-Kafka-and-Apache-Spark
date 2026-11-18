"""System monitoring utilities."""
from __future__ import annotations

import time
import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Tracks system performance metrics."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._throughput_counter: int = 0
        self._throughput_start: float = time.perf_counter()
        self._message_count: int = 0
        self._error_count: int = 0

    def record_latency(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)

    def record_message(self) -> None:
        self._message_count += 1
        self._throughput_counter += 1

    def record_error(self) -> None:
        self._error_count += 1

    def get_latency_stats(self) -> dict[str, float]:
        if not self._latencies:
            return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        latencies = sorted(self._latencies)
        n = len(latencies)
        return {
            "avg_ms": round(sum(latencies) / n, 3),
            "p50_ms": round(latencies[n // 2], 3),
            "p95_ms": round(latencies[int(n * 0.95)], 3) if n >= 20 else round(latencies[-1], 3),
            "p99_ms": round(latencies[int(n * 0.99)], 3) if n >= 100 else round(latencies[-1], 3),
        }

    def get_throughput(self) -> float:
        elapsed = time.perf_counter() - self._throughput_start
        if elapsed < 1.0:
            return 0.0
        rate = self._throughput_counter / elapsed
        self._throughput_counter = 0
        self._throughput_start = time.perf_counter()
        return round(rate, 2)

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_messages": self._message_count,
            "total_errors": self._error_count,
            "latency": self.get_latency_stats(),
            "current_throughput_msg_per_sec": self.get_throughput(),
        }

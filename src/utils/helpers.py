"""General utility functions."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from src.config.settings import RANDOM_SEED


def generate_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp() -> str:
    return utc_now().isoformat()


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def chunk_list(lst: list[Any], chunk_size: int) -> list[list[Any]]:
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def measure_time(func, *args, **kwargs) -> tuple[Any, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

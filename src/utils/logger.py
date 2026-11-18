"""Structured logging utility."""
from __future__ import annotations

import logging
import sys
from typing import Any


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


class LoggerMixin:
    """Mixin that adds a class-specific logger."""

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

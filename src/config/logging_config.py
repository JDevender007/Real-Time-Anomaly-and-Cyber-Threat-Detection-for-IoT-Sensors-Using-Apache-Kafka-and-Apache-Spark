"""Structured logging configuration."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from src.config.loader import load_config
from src.config.settings import LOG_ROOT, LOG_LEVEL


def setup_logging() -> logging.Logger:
    log_config = load_config("logging")
    level_str = getattr(log_config.logging, "level", LOG_LEVEL)
    level = getattr(logging, level_str, logging.INFO)

    fmt = getattr(log_config.logging, "format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    date_fmt = getattr(log_config.logging, "date_format", "%Y-%m-%d %H:%M:%S")

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_ROOT / "application.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.getLogger("kafka").setLevel(logging.WARNING)
    logging.getLogger("pyspark").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    return logging.getLogger("iot_security")

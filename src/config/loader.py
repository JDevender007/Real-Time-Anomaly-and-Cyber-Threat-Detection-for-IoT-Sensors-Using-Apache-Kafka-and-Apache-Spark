"""YAML configuration loader with caching."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.config.settings import CONFIG_ROOT

_config_cache: dict[str, dict[str, Any]] = {}


class ConfigNamespace:
    """Dot-notation access to nested config dicts."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")
        if isinstance(value, dict):
            return ConfigNamespace(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


def load_config(name: str, force_reload: bool = False) -> ConfigNamespace:
    if name in _config_cache and not force_reload:
        return ConfigNamespace(_config_cache[name])

    path = CONFIG_ROOT / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if name == "kafka":
        env_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        if env_bootstrap:
            data.setdefault("kafka", {})["bootstrap_servers"] = env_bootstrap

    _config_cache[name] = data
    return ConfigNamespace(data)


def load_all_configs(force_reload: bool = False) -> dict[str, ConfigNamespace]:
    config_files = list(CONFIG_ROOT.glob("*.yaml"))
    configs = {}
    for cf in config_files:
        name = cf.stem
        configs[name] = load_config(name, force_reload=force_reload)
    return configs

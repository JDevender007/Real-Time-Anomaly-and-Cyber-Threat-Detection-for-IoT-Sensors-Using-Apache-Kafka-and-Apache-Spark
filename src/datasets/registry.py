"""Dataset registry for managing multiple dataset loaders."""
from __future__ import annotations

from typing import Any

from src.datasets.base import BaseDatasetLoader


class DatasetRegistry:
    """Central registry for all dataset loaders."""

    def __init__(self) -> None:
        self._loaders: dict[str, BaseDatasetLoader] = {}

    def register(self, name: str, loader: BaseDatasetLoader) -> None:
        self._loaders[name] = loader

    def get(self, name: str) -> BaseDatasetLoader:
        if name not in self._loaders:
            available = ", ".join(self._loaders.keys())
            raise KeyError(f"Dataset '{name}' not registered. Available: {available}")
        return self._loaders[name]

    def list_datasets(self) -> list[str]:
        return list(self._loaders.keys())

    def list_by_type(self, dataset_type: str) -> list[str]:
        return [
            name for name, loader in self._loaders.items()
            if loader.dataset_type == dataset_type
        ]

    def describe_all(self) -> dict[str, Any]:
        return {name: loader.describe() for name, loader in self._loaders.items()}

    def load_all(self) -> dict[str, Any]:
        results = {}
        for name, loader in self._loaders.items():
            try:
                results[name] = loader.load()
            except Exception as e:
                results[name] = {"error": str(e)}
        return results


_registry: DatasetRegistry | None = None


def get_registry() -> DatasetRegistry:
    global _registry
    if _registry is None:
        _registry = DatasetRegistry()
    return _registry


def register_default_datasets() -> DatasetRegistry:
    from src.config.settings import RAW_DATA_ROOT
    from src.datasets.swat import SWaTDatasetLoader
    from src.datasets.wadi import WADIDatasetLoader
    from src.datasets.ciciot2023 import CICIoT2023DatasetLoader
    from src.datasets.edge_iiotset import EdgeIIoTSetDatasetLoader

    registry = get_registry()
    registry.register("swat", SWaTDatasetLoader(RAW_DATA_ROOT / "swat"))
    registry.register("wadi", WADIDatasetLoader(RAW_DATA_ROOT / "wadi"))
    registry.register("ciciot2023", CICIoT2023DatasetLoader(RAW_DATA_ROOT / "ciciot2023"))
    registry.register("edge_iiotset", EdgeIIoTSetDatasetLoader(RAW_DATA_ROOT / "edge_iiotset"))
    return registry

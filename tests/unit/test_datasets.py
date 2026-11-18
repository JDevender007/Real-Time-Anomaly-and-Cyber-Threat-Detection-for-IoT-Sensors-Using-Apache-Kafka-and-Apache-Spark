"""Unit tests for dataset modules."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.datasets.base import BaseDatasetLoader
from src.datasets.registry import DatasetRegistry, get_registry
from src.datasets.swat import SWaTDatasetLoader
from src.datasets.wadi import WADIDatasetLoader
from src.datasets.ciciot2023 import CICIoT2023DatasetLoader
from src.datasets.edge_iiotset import EdgeIIoTSetDatasetLoader
from src.config.settings import RAW_DATA_ROOT


class TestBaseDatasetLoader:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseDatasetLoader(Path("."), "test", "physical")


class TestSWaTLoader:
    def test_init(self):
        loader = SWaTDatasetLoader(RAW_DATA_ROOT / "swat")
        assert loader.name == "SWaT"
        assert loader.dataset_type == "physical"

    def test_describe(self):
        loader = SWaTDatasetLoader(RAW_DATA_ROOT / "swat")
        desc = loader.describe()
        assert desc["name"] == "SWaT"
        assert desc["type"] == "physical"

    def test_get_labels(self):
        loader = SWaTDatasetLoader(RAW_DATA_ROOT / "swat")
        assert loader.get_labels() == ["attack_label"]

    def test_load_no_data(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SWaTDatasetLoader(Path(tmpdir))
            with pytest.raises(Exception):
                loader.load()


class TestWADILoader:
    def test_init(self):
        loader = WADIDatasetLoader(RAW_DATA_ROOT / "wadi")
        assert loader.name == "WADI"
        assert loader.dataset_type == "physical"

    def test_get_labels(self):
        loader = WADIDatasetLoader(RAW_DATA_ROOT / "wadi")
        assert loader.get_labels() == ["attack_label"]


class TestCICIoT2023Loader:
    def test_init(self):
        loader = CICIoT2023DatasetLoader(RAW_DATA_ROOT / "ciciot2023")
        assert loader.name == "CICIoT2023"
        assert loader.dataset_type == "cyber"

    def test_get_labels(self):
        loader = CICIoT2023DatasetLoader(RAW_DATA_ROOT / "ciciot2023")
        assert loader.get_labels() == ["attack_label"]


class TestEdgeIIoTSetLoader:
    def test_init(self):
        loader = EdgeIIoTSetDatasetLoader(RAW_DATA_ROOT / "edge_iiotset")
        assert loader.name == "Edge-IIoTSet"
        assert loader.dataset_type == "cyber"

    def test_get_labels(self):
        loader = EdgeIIoTSetDatasetLoader(RAW_DATA_ROOT / "edge_iiotset")
        assert loader.get_labels() == ["attack_label"]


class TestDatasetRegistry:
    def test_empty_registry(self):
        registry = DatasetRegistry()
        assert registry.list_datasets() == []

    def test_register_and_get(self):
        registry = DatasetRegistry()
        loader = SWaTDatasetLoader(RAW_DATA_ROOT / "swat")
        registry.register("swat", loader)
        assert registry.get("swat") is loader

    def test_get_missing(self):
        registry = DatasetRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_by_type(self):
        registry = DatasetRegistry()
        registry.register("swat", SWaTDatasetLoader(RAW_DATA_ROOT / "swat"))
        registry.register("ciciot2023", CICIoT2023DatasetLoader(RAW_DATA_ROOT / "ciciot2023"))
        physical = registry.list_by_type("physical")
        cyber = registry.list_by_type("cyber")
        assert "swat" in physical
        assert "ciciot2023" in cyber

    def test_describe_all(self):
        registry = DatasetRegistry()
        registry.register("swat", SWaTDatasetLoader(RAW_DATA_ROOT / "swat"))
        desc = registry.describe_all()
        assert "swat" in desc

"""Unit tests for cyber ML models."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.random_forest import RandomForestModel
from src.models.xgboost import XGBoostModel
from src.preprocessing.cyber import CyberPreprocessor


@pytest.fixture
def synthetic_cyber_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 8)
    y = np.array(["Benign"] * 100 + ["DDoS"] * 50 + ["BruteForce"] * 30 + ["Scan"] * 20)
    return X, y


@pytest.fixture
def cyber_feature_names():
    return ["packet_count", "packet_rate", "bytes", "flow_duration",
            "source_port", "destination_port", "protocol_encoded", "bytes_per_packet"]


class TestRandomForestModel:
    def test_init(self):
        model = RandomForestModel(n_estimators=50)
        assert model.model.n_estimators == 50

    def test_fit_predict(self, synthetic_cyber_data, cyber_feature_names):
        X, y = synthetic_cyber_data
        model = RandomForestModel(n_estimators=50)
        model.fit(X, y, feature_names=cyber_feature_names, label_classes=sorted(set(y)))
        predictions, probabilities, confidence = model.predict(X)
        assert predictions.shape == (200,)
        assert probabilities.shape == (200, 4)
        assert confidence.shape == (200,)

    def test_evaluate(self, synthetic_cyber_data, cyber_feature_names):
        X, y = synthetic_cyber_data
        model = RandomForestModel(n_estimators=50)
        model.fit(X, y, feature_names=cyber_feature_names, label_classes=sorted(set(y)))
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "confusion_matrix" in metrics
        assert "per_class_metrics" in metrics

    def test_feature_importance(self, synthetic_cyber_data, cyber_feature_names):
        X, y = synthetic_cyber_data
        model = RandomForestModel(n_estimators=50)
        model.fit(X, y, feature_names=cyber_feature_names)
        importance = model.get_feature_importance()
        assert len(importance) == 8
        assert all(k in importance for k in cyber_feature_names)

    def test_save_load(self, synthetic_cyber_data, cyber_feature_names, tmp_path):
        X, y = synthetic_cyber_data
        model = RandomForestModel(n_estimators=50)
        model.fit(X, y, feature_names=cyber_feature_names, label_classes=sorted(set(y)))
        model.save(tmp_path / "rf_model")

        loaded = RandomForestModel.load(tmp_path / "rf_model")
        predictions_orig, _, _ = model.predict(X)
        predictions_loaded, _, _ = loaded.predict(X)
        np.testing.assert_array_equal(predictions_orig, predictions_loaded)


class TestXGBoostModel:
    def test_init(self):
        model = XGBoostModel(n_estimators=50, max_depth=4)
        assert model.model.n_estimators == 50

    def test_fit_predict(self, synthetic_cyber_data, cyber_feature_names):
        X, y = synthetic_cyber_data
        model = XGBoostModel(n_estimators=50, max_depth=4)
        model.fit(X, y, feature_names=cyber_feature_names, label_classes=sorted(set(y)))
        predictions, probabilities, confidence = model.predict(X)
        assert predictions.shape == (200,)
        assert probabilities.shape == (200, 4)
        assert confidence.shape == (200,)

    def test_evaluate(self, synthetic_cyber_data, cyber_feature_names):
        X, y = synthetic_cyber_data
        model = XGBoostModel(n_estimators=50, max_depth=4)
        model.fit(X, y, feature_names=cyber_feature_names, label_classes=sorted(set(y)))
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_feature_importance(self, synthetic_cyber_data, cyber_feature_names):
        X, y = synthetic_cyber_data
        model = XGBoostModel(n_estimators=50, max_depth=4)
        model.fit(X, y, feature_names=cyber_feature_names)
        importance = model.get_feature_importance()
        assert len(importance) == 8

    def test_save_load(self, synthetic_cyber_data, cyber_feature_names, tmp_path):
        X, y = synthetic_cyber_data
        model = XGBoostModel(n_estimators=50, max_depth=4)
        model.fit(X, y, feature_names=cyber_feature_names, label_classes=sorted(set(y)))
        model.save(tmp_path / "xgb_model")

        loaded = XGBoostModel.load(tmp_path / "xgb_model")
        predictions_orig, _, _ = model.predict(X)
        predictions_loaded, _, _ = loaded.predict(X)
        np.testing.assert_array_equal(predictions_orig, predictions_loaded)


class TestCyberPreprocessor:
    def test_fit_transform(self):
        import pandas as pd
        df = pd.DataFrame({
            "packet_count": [100, 200, 300],
            "packet_rate": [10.0, 20.0, 30.0],
            "label": ["Benign", "DDoS", "Benign"],
        })
        pp = CyberPreprocessor()
        X, y = pp.fit_transform(df, ["packet_count", "packet_rate"], "label")
        assert X.shape == (3, 2)
        assert y.shape == (3,)

    def test_save_load(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({
            "packet_count": [100, 200, 300],
            "packet_rate": [10.0, 20.0, 30.0],
            "label": ["Benign", "DDoS", "Benign"],
        })
        pp = CyberPreprocessor()
        pp.fit(df, ["packet_count", "packet_rate"], "label")
        pp.save(tmp_path / "preprocessor")

        loaded = CyberPreprocessor()
        loaded.load(tmp_path / "preprocessor")
        assert loaded.feature_names == ["packet_count", "packet_rate"]

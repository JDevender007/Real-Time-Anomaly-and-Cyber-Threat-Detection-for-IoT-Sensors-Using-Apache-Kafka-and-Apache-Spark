"""Unit tests for physical ML models."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderModel, AutoencoderNetwork
from src.preprocessing.physical import PhysicalPreprocessor


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    X_normal = rng.randn(200, 5)
    X_anomaly = rng.randn(20, 5) + 5.0
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * 200 + [1] * 20)
    return X, y


@pytest.fixture
def feature_names():
    return ["temperature", "humidity", "pressure", "voltage", "battery"]


class TestIsolationForestModel:
    def test_init(self):
        model = IsolationForestModel(n_estimators=50, contamination=0.1)
        assert model.model.n_estimators == 50

    def test_fit_predict(self, synthetic_data, feature_names):
        X, y = synthetic_data
        model = IsolationForestModel(n_estimators=50, contamination=0.1)
        model.fit(X, feature_names)
        predictions, scores = model.predict(X)
        assert predictions.shape == (220,)
        assert scores.shape == (220,)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_evaluate(self, synthetic_data, feature_names):
        X, y = synthetic_data
        model = IsolationForestModel(n_estimators=50, contamination=0.1)
        model.fit(X, feature_names)
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "confusion_matrix" in metrics

    def test_save_load(self, synthetic_data, feature_names, tmp_path):
        X, y = synthetic_data
        model = IsolationForestModel(n_estimators=50, contamination=0.1)
        model.fit(X, feature_names)
        model.save(tmp_path / "if_model")

        loaded = IsolationForestModel.load(tmp_path / "if_model")
        predictions_orig, _ = model.predict(X)
        predictions_loaded, _ = loaded.predict(X)
        np.testing.assert_array_equal(predictions_orig, predictions_loaded)

    def test_predict_before_fit(self):
        model = IsolationForestModel()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((10, 5)))


class TestAutoencoderModel:
    def test_init(self):
        model = AutoencoderModel(input_dim=5, encoding_dim=8, hidden_dims=[16, 8])
        assert model.input_dim == 5
        assert model.encoding_dim == 8

    def test_fit_predict(self, synthetic_data, feature_names):
        X, y = synthetic_data
        model = AutoencoderModel(
            input_dim=5, encoding_dim=8, hidden_dims=[16, 8],
            epochs=10, batch_size=32, early_stopping_patience=5,
        )
        model.fit(X, feature_names)
        predictions, scores, reconstructed = model.predict(X)
        assert predictions.shape == (220,)
        assert scores.shape == (220,)
        assert reconstructed.shape == (220, 5)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_evaluate(self, synthetic_data, feature_names):
        X, y = synthetic_data
        model = AutoencoderModel(
            input_dim=5, encoding_dim=8, hidden_dims=[16, 8],
            epochs=10, batch_size=32,
        )
        model.fit(X, feature_names)
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_feature_reconstruction_errors(self, synthetic_data, feature_names):
        X, y = synthetic_data
        model = AutoencoderModel(
            input_dim=5, encoding_dim=8, hidden_dims=[16, 8],
            epochs=10, batch_size=32,
        )
        model.fit(X, feature_names)
        errors = model.get_feature_reconstruction_errors(X)
        assert errors.shape == (220, 5)
        assert np.all(errors >= 0)

    def test_save_load(self, synthetic_data, feature_names, tmp_path):
        X, y = synthetic_data
        model = AutoencoderModel(
            input_dim=5, encoding_dim=8, hidden_dims=[16, 8],
            epochs=10, batch_size=32,
        )
        model.fit(X, feature_names)
        model.save(tmp_path / "ae_model")

        loaded = AutoencoderModel.load(tmp_path / "ae_model")
        predictions_orig, _, _ = model.predict(X)
        predictions_loaded, _, _ = loaded.predict(X)
        np.testing.assert_array_equal(predictions_orig, predictions_loaded)


class TestPhysicalPreprocessor:
    def test_fit_transform(self):
        import pandas as pd
        df = pd.DataFrame({
            "temperature": [20.0, 25.0, 30.0, 35.0],
            "humidity": [40.0, 50.0, 60.0, 70.0],
        })
        pp = PhysicalPreprocessor()
        X = pp.fit_transform(df, ["temperature", "humidity"])
        assert X.shape == (4, 2)

    def test_save_load(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({
            "temperature": [20.0, 25.0, 30.0, 35.0],
            "humidity": [40.0, 50.0, 60.0, 70.0],
        })
        pp = PhysicalPreprocessor()
        pp.fit(df, ["temperature", "humidity"])
        pp.save(tmp_path / "preprocessor")

        loaded = PhysicalPreprocessor()
        loaded.load(tmp_path / "preprocessor")
        assert loaded.feature_names == ["temperature", "humidity"]

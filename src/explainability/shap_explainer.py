"""SHAP explainability module."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Explainer:
    """SHAP-based model explainability."""

    def __init__(self) -> None:
        self._shap_available = False
        try:
            import shap
            self._shap_available = True
        except ImportError:
            logger.warning("SHAP not available")

    def explain_tree_model(self, model, X: np.ndarray,
                           feature_names: list[str] | None = None) -> dict[str, Any]:
        if not self._shap_available:
            return {"error": "SHAP not installed"}

        import shap
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:100])
            importance = np.abs(shap_values).mean(axis=0) if isinstance(shap_values, np.ndarray) else [
                np.abs(sv).mean(axis=0) for sv in shap_values
            ]

            if isinstance(importance, list):
                importance = np.mean(importance, axis=0)

            feature_importance = {}
            if feature_names:
                for i, name in enumerate(feature_names):
                    if i < len(importance):
                        feature_importance[name] = float(importance[i])

            return {
                "feature_importance": feature_importance,
                "shap_values_shape": str(shap_values.shape) if hasattr(shap_values, "shape") else "unknown",
            }
        except Exception as e:
            logger.error("SHAP explanation failed: %s", e)
            return {"error": str(e)}

    def explain_autoencoder(self, model, X: np.ndarray,
                            feature_names: list[str] | None = None) -> dict[str, Any]:
        try:
            feature_errors = model.get_feature_reconstruction_errors(X[:100])
            mean_errors = feature_errors.mean(axis=0)
            ranking = {}
            if feature_names:
                sorted_indices = np.argsort(mean_errors)[::-1]
                for rank, idx in enumerate(sorted_indices):
                    if idx < len(feature_names):
                        ranking[feature_names[idx]] = {
                            "reconstruction_error": float(mean_errors[idx]),
                            "rank": rank + 1,
                        }
            return {
                "feature_reconstruction_errors": ranking,
                "mean_reconstruction_error": float(mean_errors.mean()),
            }
        except Exception as e:
            logger.error("Autoencoder explanation failed: %s", e)
            return {"error": str(e)}

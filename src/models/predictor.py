"""
src/models/predictor.py
Model loading and candidate scoring.
"""

from pathlib import Path
from typing import Optional, Any
import numpy as np
import joblib


class PathSnapPredictor:
    """Wrapper around trained classifier and feature scaler."""

    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        self.model = None
        self.scaler = None
        self.model_name = "unloaded"

        if model_path and Path(model_path).exists():
            self.model = joblib.load(model_path)
            self.model_name = Path(model_path).stem

        if scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_candidate_scores(self, X_features: np.ndarray) -> np.ndarray:
        """
        Predicts match probability P(label=1 | features) for each candidate.
        If no model is loaded, falls back to inverse perpendicular distance heuristic.
        """
        if len(X_features) == 0:
            return np.array([])

        if self.model is not None:
            X_eval = X_features
            # LightGBM tree models work well on raw features, but if scaler is provided and required:
            # check if model has predict_proba
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_eval)
                if probs.shape[1] > 1:
                    return probs[:, 1]
                return probs[:, 0]
            elif hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(X_eval)
                # sigmoid
                return 1.0 / (1.0 + np.exp(-scores))
            else:
                preds = self.model.predict(X_eval)
                return preds.astype(float)

        # Baseline stub: inverse perpendicular distance (Feature 0)
        # d_perp is column 0
        d_perp = X_features[:, 0]
        scores = np.exp(-d_perp / 15.0)  # gaussian-like decay over 15 meters
        return scores

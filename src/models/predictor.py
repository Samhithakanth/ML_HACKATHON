"""
src/models/predictor.py
Loads the saved candidate classifier (models/lightgbm_tuned.joblib by default) and scores candidates.
Feature order = src.features.build.FEATURE_COLUMNS (checked at load time).
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.features.build import FEATURE_COLUMNS


class PathSnapPredictor:
    """Wrapper around the trained classifier. Models that need scaling carry their scaler in a Pipeline."""

    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        self.model = None
        self.model_name = "unloaded"
        if model_path and Path(model_path).exists():
            self.model = joblib.load(model_path)
            self.model_name = Path(model_path).stem
            n = getattr(self.model, "n_features_in_", len(FEATURE_COLUMNS))
            if n != len(FEATURE_COLUMNS):
                raise ValueError(f"{model_path} expects {n} features, pipeline builds {len(FEATURE_COLUMNS)}")
        # scaler_path kept for backwards compatibility; the scaler (if any) lives inside the model pipeline

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_candidate_scores(self, X_features: np.ndarray) -> np.ndarray:
        """P(candidate is the true arc | features). Without a model: exp(-distance / 15 m) baseline."""
        if len(X_features) == 0:
            return np.array([])
        if self.model is not None:
            return self.model.predict_proba(X_features)[:, 1]
        return np.exp(-X_features[:, 0] / 15.0)   # column 0 = dist_m

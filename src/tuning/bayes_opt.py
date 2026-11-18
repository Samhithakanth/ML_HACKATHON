"""
src/tuning/bayes_opt.py
Bayesian Hyperparameter Optimization for PathSnap Classifiers using Optuna (TPE Sampler).
Optimizes LightGBM across candidate hyperparameters using GroupKFold on training trips.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score


def run_bayesian_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_trials: int = 20,
    timeout_s: int = 120,
    random_state: int = 42
) -> Tuple[Dict[str, Any], float]:
    """
    Executes Tree-structured Parzen Estimator (TPE) Bayesian search
    maximizing cross-validated Macro F1-score across trip groups.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    gkf = GroupKFold(n_splits=5)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 250),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": -1
        }

        clf = lgb.LGBMClassifier(**params)
        f1_scores = []
        for trn_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
            clf.fit(X_train[trn_idx], y_train[trn_idx])
            preds = clf.predict(X_train[val_idx])
            f1_scores.append(f1_score(y_train[val_idx], preds, average="macro"))

        return float(np.mean(f1_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)

    best_params = study.best_params
    best_params["random_state"] = random_state
    best_params["n_jobs"] = -1
    return best_params, float(study.best_value)

"""
src/models/zoo.py  (Person 2)

The candidate classifiers compared in Section 3. Six models from six different families.
Every model is an sklearn Pipeline, so the StandardScaler (where used) is fit on the training
fold only and is saved inside the model artifact.
"""
from __future__ import annotations

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

# name -> (family, short config string, factory, max training rows (None = all sampled rows))
MODEL_SPECS = {
    "logistic_regression": (
        "Linear", "StandardScaler + L2 logistic regression, C=1.0, lbfgs",
        lambda: Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))]),
        None),
    "gaussian_nb": (
        "Probabilistic generative", "StandardScaler + Gaussian naive Bayes, var_smoothing=1e-9",
        lambda: Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
        None),
    "knn": (
        "Instance-based", "StandardScaler + KNN, k=25, distance-weighted, 120k-row training subsample",
        lambda: Pipeline([("scaler", StandardScaler()),
                          ("clf", KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))]),
        120_000),
    "random_forest": (
        "Bagging ensemble", "300 trees, min_samples_leaf=5, max_features=sqrt",
        lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                       n_jobs=-1, random_state=SEED),
        None),
    "lightgbm": (
        "Gradient boosting", "LightGBM defaults: 100 trees, lr=0.1, num_leaves=31",
        lambda: lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
        None),
    "mlp": (
        "Neural network", "StandardScaler + MLP (128, 64), ReLU, Adam, early stopping, <=30 epochs, 300k-row subsample",
        lambda: Pipeline([("scaler", StandardScaler()),
                          ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), batch_size=1024,
                                                learning_rate_init=1e-3, max_iter=30, early_stopping=True,
                                                n_iter_no_change=4, random_state=SEED))]),
        300_000),
}

"""
src/models/common.py  (Person 2)
Shared helpers for the experiments: data split rules, per-trip sampling and observation-level top-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import SUSPECT_TRIPS
from src.features.build import FEATURE_COLUMNS

SEED = 42
N_FOLDS = 5
TRAIN_POINTS_PER_TRIP = 800    # per-trip cap for training rows: balances long trips (trip 54 = 9% of train) and cost
EVAL_POINTS_PER_TRIP = 1000    # per-trip cap for CV evaluation (identical subset for every model)


def clean_train(F: pd.DataFrame, P: pd.DataFrame):
    """Drop the two label-suspect trips (33, 54) from training/CV; they are reported separately."""
    return F[~F.trip_id.isin(SUSPECT_TRIPS)].reset_index(drop=True), P[~P.trip_id.isin(SUSPECT_TRIPS)].reset_index(drop=True)


def sample_points(P: pd.DataFrame, cap: int, seed: int = SEED) -> pd.DataFrame:
    """Random subset of at most `cap` points per trip (all candidates of a sampled point are kept)."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in P.groupby("trip_id", sort=True):
        idx = g.index.values
        keep.append(idx if len(idx) <= cap else np.sort(rng.choice(idx, cap, replace=False)))
    return P.loc[np.concatenate(keep)]


def rows_for_points(F: pd.DataFrame, Psub: pd.DataFrame) -> np.ndarray:
    """Boolean mask over F for the candidate rows of the points in Psub."""
    return np.isin(point_keys(F), point_keys(Psub))


def point_keys(df: pd.DataFrame) -> np.ndarray:
    return df.trip_id.astype(np.int64).values * 1_000_000 + df.point_idx.values.astype(np.int64)


def top1(F: pd.DataFrame, prob: np.ndarray, P: pd.DataFrame) -> dict:
    """Observation-level top-1 over the points in P (points with no candidates count as wrong)."""
    k = point_keys(F)
    order = np.lexsort((-prob, k))
    ks = k[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    best_lab = pd.Series(F.label.values[order][first], index=ks[first])
    hit = best_lab.reindex(point_keys(P)).fillna(0).values.astype(bool)
    inc = P.label_in_candidates.values
    return {"top1": float(hit.mean()),
            "top1_given_in_cands": float(hit[inc].mean()) if inc.any() else float("nan"),
            "n_points": int(len(P))}


def per_trip_top1(F: pd.DataFrame, prob: np.ndarray, P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tid, g in P.groupby("trip_id"):
        m = (F.trip_id == tid).values
        rows.append({"trip_id": tid, **top1(F[m], prob[m], g)})
    return pd.DataFrame(rows)


def X_of(F: pd.DataFrame) -> np.ndarray:
    return F[FEATURE_COLUMNS].values.astype(np.float32)

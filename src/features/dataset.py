"""
src/features/dataset.py  (Person 2)

Builds the training / evaluation tables from the raw files:
    data/processed/{partition}_features.parquet   one row per (point, candidate arc) + label
    data/processed/{partition}_points.parquet     one row per GPS point (reference arc, flags)

    python -m src.features.dataset            # both partitions (train, student_test)

Labels come from the P1 monotone-DP alignment (src.data.labels.align_trip_to_route).
Candidates come from src.data.candidates.candidate_table (R = 100 m, K = 30).
The evaluator partition is never read.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data.loader import load_trip, list_trips, REPO_ROOT, SUSPECT_TRIPS
from src.data.labels import align_trip_to_route
from src.data.candidates import candidate_table
from src.features.build import build_features, FEATURE_COLUMNS

PROC_DIR = REPO_ROOT / "data" / "processed"
PARTITIONS = ("train", "student_test")


def build_trip(trip_id: str, partition: str):
    trip = load_trip(trip_id, partition, load_route=True)
    _, ref_arc, ref_dist = align_trip_to_route(trip)
    cands = candidate_table(trip)
    feats = build_features(trip, cands)
    feats["label"] = (feats.arc_id.values == ref_arc[feats.point_idx.values]).astype(np.int8)
    feats.insert(0, "trip_id", trip_id)
    n = len(trip.track)
    hit = feats.groupby("point_idx").label.max().reindex(np.arange(n), fill_value=0).values
    ncand = feats.groupby("point_idx").size().reindex(np.arange(n), fill_value=0).values
    pts = pd.DataFrame({
        "trip_id": trip_id, "partition": partition, "point_idx": np.arange(n, dtype=np.int32),
        "t": trip.track.t.values, "dt_prev": np.r_[np.nan, np.diff(trip.track.t.values)],
        "label_arc": ref_arc.astype(np.int32), "label_dist_m": ref_dist.astype(np.float32),
        "label_in_candidates": hit.astype(bool), "n_candidates": ncand.astype(np.int16),
        "trip_label_suspect": trip_id in SUSPECT_TRIPS,
    })
    return feats, pts


def build_partition(partition: str, n_jobs: int = -1):
    t0 = time.time()
    trips = list_trips(partition)
    out = Parallel(n_jobs=n_jobs, verbose=0)(delayed(build_trip)(tid, partition) for tid in trips)
    F = pd.concat([o[0] for o in out], ignore_index=True)
    P = pd.concat([o[1] for o in out], ignore_index=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    F.to_parquet(PROC_DIR / f"{partition}_features.parquet", index=False)
    P.to_parquet(PROC_DIR / f"{partition}_points.parquet", index=False)
    clean = P[~P.trip_label_suspect]
    print(f"[{partition}] {len(trips)} trips, {len(P):,} points, {len(F):,} candidate rows, "
          f"{len(FEATURE_COLUMNS)} features, miss rate all={1 - P.label_in_candidates.mean():.4f} "
          f"clean={1 - clean.label_in_candidates.mean():.4f}  ({time.time() - t0:.0f} s)", flush=True)


def load_partition(partition: str):
    F = pd.read_parquet(PROC_DIR / f"{partition}_features.parquet")
    P = pd.read_parquet(PROC_DIR / f"{partition}_points.parquet")
    return F, P


if __name__ == "__main__":
    parts = sys.argv[1:] or PARTITIONS
    for part in parts:
        build_partition(part)

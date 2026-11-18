"""
api/pipeline.py
Core model inference pipeline used by the API (identical code path to the experiments).

raw .track/.nodes/.arcs -> candidates (R=100 m, K=30) -> 31 features -> tuned LightGBM probability
-> HMM / Viterbi (configs/hmm.yaml).  NEVER reads .route (load_trip(..., load_route=False)).
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from src.data.loader import load_trip
from src.data.candidates import candidate_table
from src.features.build import build_features, FEATURE_COLUMNS
from src.models.predictor import PathSnapPredictor
from src.hmm.decoder import build_lattice, decode, DEFAULT_PARAMS
from src.eval.metrics import top1_from_scores


def predict_trip(
    trip_id: str,
    data_dir: Path,
    partition: str = "student_test",
    predictor: Optional[PathSnapPredictor] = None,
    hmm_config: Optional[Dict[str, Any]] = None,
    use_stub: bool = False,
    **_ignored,
) -> Dict[str, Any]:
    """
    Returns {trip_id, predicted_arcs (HMM, -1 = no candidate), classifier_arcs (argmax per point),
             track [lon, lat, t], nodes [lon, lat], arcs [from, to], timings_ms, model}.
    use_stub (or no model): nearest-candidate emission exp(-d/15 m) instead of the classifier.
    """
    t_start = time.perf_counter()
    trip = load_trip(trip_id, partition, raw_dir=data_dir, load_route=False)
    params = dict(DEFAULT_PARAMS, **{k: v for k, v in (hmm_config or {}).items() if k in DEFAULT_PARAMS})

    t0 = time.perf_counter()
    cands = candidate_table(trip)
    t_cand = time.perf_counter() - t0

    t0 = time.perf_counter()
    if use_stub or predictor is None or not predictor.is_loaded:
        prob = np.exp(-cands.dist_m.values.astype(float) / 15.0)
        model_name = "nearest-arc-stub"
    else:
        feats = build_features(trip, cands)
        prob = predictor.predict_candidate_scores(feats[FEATURE_COLUMNS].values.astype(np.float32))
        model_name = predictor.model_name
    clf_arcs = top1_from_scores(cands, prob, len(trip.track))
    t_clf = time.perf_counter() - t0

    t0 = time.perf_counter()
    lattice = build_lattice(trip, cands, gap_s=params["gap_s"])
    hmm_arcs = decode(lattice, prob, params)
    t_hmm = time.perf_counter() - t0

    ms = lambda s: int(round(s * 1000.0))
    return {
        "trip_id": trip_id,
        "model": model_name,
        "predicted_arcs": hmm_arcs.tolist(),
        "classifier_arcs": clf_arcs.tolist(),
        "track": trip.track[["lon", "lat", "t"]].values,
        "nodes": trip.nodes[["lon", "lat"]].values,
        "arcs": trip.arcs[["from_node", "to_node"]].values,
        "timings_ms": {"candidates": ms(t_cand), "classifier": ms(t_clf), "hmm": ms(t_hmm),
                       "total": ms(time.perf_counter() - t_start)},
    }

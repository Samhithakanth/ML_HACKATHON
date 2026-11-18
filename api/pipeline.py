"""
api/pipeline.py
Core model inference pipeline.
Strict data contract compliance: Uses ONLY raw .track / .nodes / .arcs.
NEVER reads .route or references ground truth.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from src.data.loader import load_trip_raw, get_metric_scale
from src.data.candidates import generate_candidates
from src.features.extract import extract_candidate_features
from src.models.predictor import PathSnapPredictor
from src.hmm.viterbi import viterbi_decode


def predict_trip(
    trip_id: str,
    data_dir: Path,
    partition: str = "student_test",
    predictor: Optional[PathSnapPredictor] = None,
    hmm_config: Optional[Dict[str, Any]] = None,
    spatial_k: int = 35,
    max_candidates: int = 12,
    use_stub: bool = False
) -> Dict[str, Any]:
    """
    Executes the full inference pipeline for a given trip:
    raw GPS + road network -> candidates -> feature extraction -> model inference -> HMM/Viterbi.
    NEVER reads or references .route.

    Returns:
        {
            "trip_id": str,
            "predicted_arcs": List[int],    # HMM Viterbi decoded arc_id per point
            "classifier_arcs": List[int],   # Argmax classifier-only arc_id per point
            "track": np.ndarray,
            "nodes": np.ndarray,
            "arcs": np.ndarray,
            "timings_ms": {
                "candidates": int,
                "classifier": int,
                "hmm": int,
                "total": int
            }
        }
    """
    t_start = time.perf_counter()

    # 1. Load raw inputs only (load_route=False)
    track, nodes, arcs, _ = load_trip_raw(trip_id, data_dir, partition=partition, load_route=False)

    mean_lat = float(np.mean(track[:, 1]))
    scale = get_metric_scale(mean_lat)

    nodes_m = nodes * scale
    track_m = track[:, :2] * scale
    ts = track[:, 2]

    # 2. Candidate Generation
    t_cand_0 = time.perf_counter()
    candidates_per_point = generate_candidates(
        track_m=track_m,
        nodes_m=nodes_m,
        arcs=arcs,
        spatial_k=spatial_k,
        max_candidates=max_candidates
    )
    t_cand_ms = int(round((time.perf_counter() - t_cand_0) * 1000.0))

    # Stub mode: Nearest-arc baseline
    if use_stub or predictor is None or not predictor.is_loaded:
        # Nearest candidate is rank 0 from candidates_per_point
        t_clf_0 = time.perf_counter()
        nearest_arcs = []
        last_valid = 0
        for cands in candidates_per_point:
            if cands:
                last_valid = cands[0]
            nearest_arcs.append(last_valid)
        t_clf_ms = int(round((time.perf_counter() - t_clf_0) * 1000.0))

        # Under stub, classifier and HMM are both nearest-arc baseline
        t_total_ms = int(round((time.perf_counter() - t_start) * 1000.0))
        return {
            "trip_id": trip_id,
            "predicted_arcs": nearest_arcs,
            "classifier_arcs": nearest_arcs,
            "track": track,
            "nodes": nodes,
            "arcs": arcs,
            "timings_ms": {
                "candidates": t_cand_ms,
                "classifier": t_clf_ms,
                "hmm": 0,
                "total": t_total_ms
            }
        }

    # 3. Feature Extraction & Classifier Scoring
    t_clf_0 = time.perf_counter()
    X_features = extract_candidate_features(
        track_m=track_m,
        ts=ts,
        nodes_m=nodes_m,
        arcs=arcs,
        candidates_per_point=candidates_per_point
    )
    candidate_scores = predictor.predict_candidate_scores(X_features)
    t_clf_ms = int(round((time.perf_counter() - t_clf_0) * 1000.0))

    # 4. HMM & Viterbi Sequential Decoding
    t_hmm_0 = time.perf_counter()
    hmm_selected_arcs, classifier_only_arcs = viterbi_decode(
        candidates_per_point=candidates_per_point,
        candidate_scores=candidate_scores,
        arcs=arcs,
        timestamps=ts,
        hmm_config=hmm_config
    )
    t_hmm_ms = int(round((time.perf_counter() - t_hmm_0) * 1000.0))

    t_total_ms = int(round((time.perf_counter() - t_start) * 1000.0))

    return {
        "trip_id": trip_id,
        "predicted_arcs": hmm_selected_arcs,
        "classifier_arcs": classifier_only_arcs,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "timings_ms": {
            "candidates": t_cand_ms,
            "classifier": t_clf_ms,
            "hmm": t_hmm_ms,
            "total": t_total_ms
        }
    }

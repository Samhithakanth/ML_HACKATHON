"""
src/features/extract.py
Feature extraction for (GPS observation, candidate road arc) pairs.
Extracts 10 engineered geometric, kinematic, and topological features.
"""

from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
from src.data.labels import pt_to_segment


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes angle difference in radians [0, pi] between two 2D vectors."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    dot = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(dot))


def compute_trip_kinematics(track_m: np.ndarray, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes normalized GPS headings and instantaneous speeds for each point."""
    n_pts = len(track_m)
    gps_headings = np.zeros((n_pts, 2))
    speeds = np.zeros(n_pts)

    for i in range(n_pts):
        prev_idx = max(0, i - 1)
        next_idx = min(n_pts - 1, i + 1)
        dt = ts[next_idx] - ts[prev_idx]
        disp = track_m[next_idx] - track_m[prev_idx]
        norm = np.linalg.norm(disp)
        if norm > 0:
            gps_headings[i] = disp / norm
        if dt > 0:
            speeds[i] = norm / dt

    return gps_headings, speeds


def extract_candidate_features(
    track_m: np.ndarray,
    ts: np.ndarray,
    nodes_m: np.ndarray,
    arcs: np.ndarray,
    candidates_per_point: List[List[int]]
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Extracts 10 features for all candidate arcs across all GPS points in a trip.

    Returns:
        (X_features: np.ndarray of shape (total_candidates, 10),
         cand_map: list of arc_ids for each candidate row)
    """
    arc_u = nodes_m[arcs[:, 0]]
    arc_v = nodes_m[arcs[:, 1]]
    arc_lengths = np.linalg.norm(arc_v - arc_u, axis=1)

    node_degrees = Counter(arcs[:, 0]) + Counter(arcs[:, 1])
    gps_headings, speeds = compute_trip_kinematics(track_m, ts)

    rows = []
    n_pts = len(track_m)

    for i in range(n_pts):
        pt = track_m[i]
        cands = candidates_per_point[i]
        for arc_id in cands:
            u, v = arcs[arc_id]
            u_pos = nodes_m[u]
            v_pos = nodes_m[v]

            d_perp, t_proj, d_u, d_v = pt_to_segment(pt, u_pos, v_pos)
            arc_vec = v_pos - u_pos
            angle_diff = angle_between_vectors(gps_headings[i], arc_vec)

            feat = [
                d_perp,
                angle_diff,
                t_proj,
                arc_lengths[arc_id],
                d_u,
                d_v,
                speeds[i],
                float(node_degrees[u]),
                float(node_degrees[v]),
                abs(d_u - d_v),
            ]
            rows.append(feat)

    X_features = np.array(rows, dtype=np.float32) if rows else np.empty((0, 10), dtype=np.float32)
    return X_features

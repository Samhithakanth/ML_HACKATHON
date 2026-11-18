"""
src/data/labels.py
Server-side ground truth route alignment.
Maps chronological ground truth .route arcs to observation-level reference labels.
STRICTLY SERVER-SIDE: Kept in memory on Laptop 1; NEVER sent to client or UI.
"""

import numpy as np
from typing import List


def pt_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Computes perpendicular distance and projection factor t for point p to segment ab."""
    ab = b - a
    l2 = np.dot(ab, ab)
    if l2 == 0:
        return float(np.linalg.norm(p - a)), 0.0, float(np.linalg.norm(p - a)), float(np.linalg.norm(p - b))
    t = float(np.clip(np.dot(p - a, ab) / l2, 0.0, 1.0))
    proj = a + t * ab
    dist = float(np.linalg.norm(p - proj))
    d_u = float(np.linalg.norm(p - a))
    d_v = float(np.linalg.norm(p - b))
    return dist, t, d_u, d_v


def align_to_route(
    track_m: np.ndarray,
    nodes_m: np.ndarray,
    arcs: np.ndarray,
    route: np.ndarray
) -> List[int]:
    """
    Aligns each GPS observation point to the corresponding ground truth arc in route.
    Uses chronological window projection along the route.

    Returns:
        List of reference arc_ids, one per GPS point.
    """
    if route is None or len(route) == 0:
        return []

    r_arcs = arcs[route]
    curr_r_idx = 0
    ref_arcs = []
    n_route = len(route)

    for pt in track_m:
        w_start = max(0, curr_r_idx - 2)
        w_end = min(n_route, curr_r_idx + 12)
        best_dist = 1e9
        best_arc = int(route[curr_r_idx])
        best_k = curr_r_idx

        for k in range(w_start, w_end):
            u, v = r_arcs[k]
            d, _, _, _ = pt_to_segment(pt, nodes_m[u], nodes_m[v])
            if d < best_dist:
                best_dist = d
                best_arc = int(route[k])
                best_k = k

        ref_arcs.append(best_arc)
        curr_r_idx = best_k

    return ref_arcs

"""
src/data/candidates.py
Spatial road segment candidate generation using KD-Tree.
"""

from typing import List, Dict, Tuple
import numpy as np
import scipy.spatial


def build_arc_spatial_tree(nodes_m: np.ndarray, arcs: np.ndarray) -> Tuple[scipy.spatial.cKDTree, np.ndarray]:
    """
    Samples 4 points per arc (start, 25%, 75%, end) to index in a KD-Tree.
    Returns:
        (tree, arc_ids_map)
    """
    arc_u = nodes_m[arcs[:, 0]]
    arc_v = nodes_m[arcs[:, 1]]
    arc_p1 = arc_u * 0.75 + arc_v * 0.25
    arc_p2 = arc_u * 0.25 + arc_v * 0.75

    sampled_pts = np.vstack([arc_u, arc_v, arc_p1, arc_p2])
    arc_ids_map = np.tile(np.arange(len(arcs)), 4)
    tree = scipy.spatial.cKDTree(sampled_pts)
    return tree, arc_ids_map


def generate_candidates(
    track_m: np.ndarray,
    nodes_m: np.ndarray,
    arcs: np.ndarray,
    spatial_k: int = 35,
    max_candidates: int = 12
) -> List[List[int]]:
    """
    Finds top candidate road arcs for each GPS observation point.
    Returns:
        List of candidate arc lists, one list per GPS point.
    """
    tree, arc_ids_map = build_arc_spatial_tree(nodes_m, arcs)
    k_query = min(len(arc_ids_map), spatial_k)
    _, indices = tree.query(track_m, k=k_query)

    n_pts = len(track_m)
    candidates_per_point = []

    for i in range(n_pts):
        cand_indices = indices[i]
        if np.ndim(cand_indices) == 0:
            cand_indices = [cand_indices]
        cands = list(dict.fromkeys(arc_ids_map[cand_indices]))[:max_candidates]
        candidates_per_point.append([int(c) for c in cands])

    return candidates_per_point


# ---------------------------------------------------------------------------
# Person 1 candidate generation (used for training, evaluation and the API):
# all arcs within R = 100 m of each GPS point, capped at the K = 30 nearest.
# Chosen from a train-only miss-rate sweep (docs/P1_DATA_NOTES.md).
# ---------------------------------------------------------------------------
import pandas as pd
import shapely
from shapely import STRtree

CAND_RADIUS_M = 100.0
CAND_MAX_K = 30


def candidate_table(trip, radius: float = CAND_RADIUS_M, max_k: int = CAND_MAX_K) -> pd.DataFrame:
    """DataFrame(point_idx, arc_id, dist_m, frac, rank); frac = projection position along the arc in [0, 1]."""
    xy = trip.nodes[["x", "y"]].values
    a = xy[trip.arcs.from_node.values]
    b = xy[trip.arcs.to_node.values]
    tree = STRtree(shapely.linestrings(np.stack([a, b], axis=1)))
    pts = shapely.points(trip.track[["x", "y"]].values)
    pi, ai = tree.query(pts, predicate="dwithin", distance=radius)
    px = trip.track.x.values[pi]; py = trip.track.y.values[pi]
    ax, ay, bx, by = a[ai, 0], a[ai, 1], b[ai, 0], b[ai, 1]
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    frac = np.clip(np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1), 0.0), 0, 1)
    dist = np.hypot(px - (ax + frac * dx), py - (ay + frac * dy))
    df = pd.DataFrame({"point_idx": pi.astype(np.int32), "arc_id": ai.astype(np.int32),
                       "dist_m": dist.astype(np.float32), "frac": frac.astype(np.float32)})
    df = df.sort_values(["point_idx", "dist_m"], kind="stable").reset_index(drop=True)
    df["rank"] = df.groupby("point_idx").cumcount().astype(np.int16)
    if max_k is not None:
        df = df[df["rank"] < max_k].reset_index(drop=True)
    return df

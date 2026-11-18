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

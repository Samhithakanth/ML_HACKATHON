"""
src/hmm/viterbi.py
Route polyline construction for the API / maps. The HMM decoder itself lives in src/hmm/decoder.py.
"""

from typing import List, Tuple, Optional
import numpy as np


def build_route_polylines(
    arcs_seq: List[int],
    arcs: np.ndarray,
    nodes: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    gap_threshold: float = 5.0
) -> Tuple[List[List[List[float]]], int]:
    """
    Constructs polylines from an arc sequence, converting to [lat, lon] coordinates.
    Splits into separate paths whenever there is a topological break or GPS gap.
    Returns:
        (paths: list of [[lat, lon], ...], unique_arcs_count: int)
    """
    if len(arcs_seq) == 0:
        return [], 0

    node_lats = nodes[:, 1]
    node_lons = nodes[:, 0]

    paths = []
    curr_path = []
    last_node = None
    unique_arcs = set()

    prev_arc = None
    for i, arc_id in enumerate(arcs_seq):
        if arc_id < 0 or arc_id >= len(arcs):   # -1 = point without candidates: end the current piece
            if len(curr_path) >= 2:
                paths.append(curr_path)
            curr_path, last_node, prev_arc = [], None, None
            continue
        unique_arcs.add(arc_id)
        u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])

        # Check gap
        has_gap = False
        if timestamps is not None and i > 0 and i < len(timestamps):
            if timestamps[i] - timestamps[i - 1] > gap_threshold:
                has_gap = True

        # Several consecutive GPS points on the same arc: the arc is already drawn
        if arc_id == prev_arc and not has_gap:
            continue
        prev_arc = arc_id

        # Check topological break
        is_break = (last_node is not None and last_node != u) or has_gap

        if is_break:
            if len(curr_path) >= 2:
                paths.append(curr_path)
            curr_path = [
                [float(node_lats[u]), float(node_lons[u])],
                [float(node_lats[v]), float(node_lons[v])]
            ]
        else:
            if not curr_path:
                curr_path.append([float(node_lats[u]), float(node_lons[u])])
            curr_path.append([float(node_lats[v]), float(node_lons[v])])

        last_node = v

    if len(curr_path) >= 2:
        paths.append(curr_path)

    return paths, len(unique_arcs)

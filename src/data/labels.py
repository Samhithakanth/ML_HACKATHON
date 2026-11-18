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


# ---------------------------------------------------------------------------
# Person 1 reference labels: monotone DP alignment of the GPS sequence to .route
#   minimise sum_t dist(point_t, route_arc[j_t])  s.t.  j_1 <= j_2 <= ... <= j_T
# This is the label definition used for training AND for server-side scoring.
# ---------------------------------------------------------------------------
def _point_seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1), 0.0)
    t = np.clip(t, 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def align_trip_to_route(trip):
    """Return (route_idx per point, reference arc_id per point, distance to that arc in metres)."""
    xy = trip.nodes[["x", "y"]].values
    arcs = trip.arcs.iloc[trip.route]
    a = xy[arcs.from_node.values]
    b = xy[arcs.to_node.values]
    ax, ay, bx, by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    T, R = len(trip.track), len(trip.route)
    px, py = trip.track.x.values, trip.track.y.values
    bp = np.zeros((T, R), dtype=np.uint16 if R < 65535 else np.uint32)
    ar = np.arange(R)
    prev = _point_seg_dist(px[0], py[0], ax, ay, bx, by)
    for t in range(1, T):
        m = np.minimum.accumulate(prev)
        rec = np.where(prev == m, ar, 0)
        bp[t] = np.maximum.accumulate(rec)
        prev = m + _point_seg_dist(px[t], py[t], ax, ay, bx, by)
    j = np.empty(T, dtype=np.int64)
    j[-1] = int(np.argmin(prev))
    for t in range(T - 1, 0, -1):
        j[t - 1] = bp[t, j[t]]
    arc_id = trip.route[j]
    d = _point_seg_dist(px, py, ax[j], ay[j], bx[j], by[j])
    return j, arc_id, d

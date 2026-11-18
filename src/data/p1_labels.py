"""Derive a per-GPS-point reference arc from the ordered .route file.

.route only lists the arcs traversed (no per-point labels), so we align the GPS
sequence to the route with a monotone dynamic programme:
    minimise  sum_t dist(point_t, route_arc[j_t])   subject to  j_1 <= j_2 <= ... <= j_T
(prefix-min recurrence, O(T*R)). The vehicle can only move forward along its route,
so the assignment is non-decreasing in route index.
"""
import numpy as np


def point_seg_dist(px, py, ax, ay, bx, by):
    """Distance from points (n,) to segments (m,) -> (n, m) not formed; here p is scalar-broadcast."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1), 0.0)
    t = np.clip(t, 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def align_to_route(trip):
    """Return (route_idx per point, arc_id per point, dist_m per point)."""
    xy = trip.nodes[["x", "y"]].values
    arcs = trip.arcs.iloc[trip.route]
    a = xy[arcs.from_node.values]
    b = xy[arcs.to_node.values]
    ax, ay, bx, by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    T, R = len(trip.track), len(trip.route)
    px, py = trip.track.x.values, trip.track.y.values
    bp = np.zeros((T, R), dtype=np.uint16 if R < 65535 else np.uint32)
    ar = np.arange(R)
    prev = point_seg_dist(px[0], py[0], ax, ay, bx, by)
    for t in range(1, T):
        m = np.minimum.accumulate(prev)
        rec = np.where(prev == m, ar, 0)
        bp[t] = np.maximum.accumulate(rec)
        prev = m + point_seg_dist(px[t], py[t], ax, ay, bx, by)
    j = np.empty(T, dtype=np.int64)
    j[-1] = int(np.argmin(prev))
    for t in range(T - 1, 0, -1):
        j[t - 1] = bp[t, j[t]]
    arc_id = trip.route[j]
    d = point_seg_dist(px, py, ax[j], ay[j], bx[j], by[j])
    return j, arc_id, d

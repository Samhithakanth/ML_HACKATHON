"""Candidate road-segment generation: all arcs within R metres of each GPS point."""
import numpy as np, pandas as pd, shapely
from shapely import STRtree


def arc_geometries(trip):
    xy = trip.nodes[["x", "y"]].values
    a = xy[trip.arcs.from_node.values]
    b = xy[trip.arcs.to_node.values]
    coords = np.stack([a, b], axis=1)           # (n_arcs, 2, 2)
    return shapely.linestrings(coords), a, b


def generate_candidates(trip, radius=100.0, max_k=None):
    """Return DataFrame(point_idx, arc_id, dist_m, frac, rank) for arcs within `radius`.
    frac = projection of the point onto the arc, in [0,1] (position along the arc)."""
    geoms, a, b = arc_geometries(trip)
    tree = STRtree(geoms)
    pts = shapely.points(trip.track[["x", "y"]].values)
    pi, ai = tree.query(pts, predicate="dwithin", distance=radius)
    px = trip.track.x.values[pi]; py = trip.track.y.values[pi]
    ax, ay, bx, by = a[ai, 0], a[ai, 1], b[ai, 0], b[ai, 1]
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    frac = np.clip(np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1), 0.0), 0, 1)
    dist = np.hypot(px - (ax + frac * dx), py - (ay + frac * dy))
    df = pd.DataFrame({"point_idx": pi, "arc_id": ai, "dist_m": dist, "frac": frac})
    df = df.sort_values(["point_idx", "dist_m"], kind="stable").reset_index(drop=True)
    df["rank"] = df.groupby("point_idx").cumcount()
    if max_k is not None:
        df = df[df["rank"] < max_k].reset_index(drop=True)
    return df

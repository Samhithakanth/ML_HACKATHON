"""
src/features/build.py  (Person 2)

Feature engineering for (GPS point, candidate arc) pairs.

`build_features(trip, cands)` works from the raw .track/.nodes/.arcs only (never .route), so the
API uses exactly the same code as training. All operations are vectorised (the longest trip has
19,313 points and ~300k candidate rows).

Feature groups
--------------
geometry   : dist_m, frac, rank, arc length, distance to both arc end nodes, zero-length guard
relative   : distance gap / ratio to the nearest candidate, number of candidates at the point
heading    : cosine between GPS motion heading and arc bearing (sign separates the two directions
             of a two-way street), computed with a backward, a forward and a centred difference
kinematics : speed, dt to previous/next fix, acceleration
topology   : in/out degree of the arc end nodes, whether the arc has a reverse twin
context    : does the same arc (or a graph-connected arc) appear as a candidate at the previous /
             next points, and how close is it; signed progress along the arc between t-1 and t+1;
             fraction of the +-W window in which the arc is a candidate and its mean distance there
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_DIST = 150.0   # used when an arc is not a candidate at a neighbouring point (> candidate radius)
WINDOW = 5             # +- points for the windowed context features

FEATURE_COLUMNS = [
    # geometry
    "dist_m", "frac", "rank", "log_arc_len", "zero_len", "d_from_node", "d_to_node",
    # relative to the other candidates at the same point
    "dist_gap", "dist_ratio", "n_cands", "gap_2nd",
    # heading
    "cos_c", "cos_b", "cos_f", "abs_cos_c", "move_c",
    # kinematics
    "speed", "dt_prev", "dt_next", "accel",
    # topology
    "out_deg_to", "in_deg_from", "has_twin",
    # context
    "prev_same_dist", "next_same_dist", "prev_conn_dist", "next_conn_dist", "progress_along",
    "progress_vs_move", "win_frac", "win_mean_dist",
]


def _unit(dx, dy):
    n = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = np.where(n > 0, dx / n, 0.0), np.where(n > 0, dy / n, 0.0)
    return ux, uy, n


def _lookup(sorted_keys, values, query, fill):
    """Vectorised dict lookup: value for each query key, `fill` if absent."""
    pos = np.searchsorted(sorted_keys, query)
    pos = np.clip(pos, 0, len(sorted_keys) - 1)
    hit = sorted_keys[pos] == query
    out = np.full(len(query), fill, dtype=np.float64)
    out[hit] = values[pos[hit]]
    return out


def _min_by_key(keys, vals):
    """Group-min of vals by integer key -> (unique sorted keys, min values)."""
    order = np.lexsort((vals, keys))
    k, v = keys[order], vals[order]
    first = np.r_[True, k[1:] != k[:-1]]
    return k[first], v[first]


def build_features(trip, cands: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `cands` (point_idx, arc_id, dist_m, frac, rank) with all FEATURE_COLUMNS added.
    Row order is preserved."""
    df = cands[["point_idx", "arc_id", "dist_m", "frac", "rank"]].copy()
    if len(df) == 0:
        for c in FEATURE_COLUMNS:
            if c not in df:
                df[c] = np.zeros(0, dtype=np.float32)
        return df

    T = len(trip.track)
    x, y, t = trip.track.x.values, trip.track.y.values, trip.track.t.values.astype(float)
    nxy = trip.nodes[["x", "y"]].values
    fr, to = trip.arcs.from_node.values, trip.arcs.to_node.values
    n_nodes, n_arcs = len(nxy), len(fr)

    # ---------------- per-point kinematics ----------------
    dt_prev = np.r_[np.nan, np.diff(t)]
    dt_next = np.r_[np.diff(t), np.nan]
    bx, by, bn = _unit(x - np.r_[x[0], x[:-1]], y - np.r_[y[0], y[:-1]])          # backward displacement
    fx, fy, fn = _unit(np.r_[x[1:], x[-1]] - x, np.r_[y[1:], y[-1]] - y)          # forward displacement
    ip, inx = np.maximum(np.arange(T) - 1, 0), np.minimum(np.arange(T) + 1, T - 1)
    cx, cy, cn = _unit(x[inx] - x[ip], y[inx] - y[ip])                             # centred displacement
    dtc = t[inx] - t[ip]
    speed = np.where(dtc > 0, cn / np.where(dtc > 0, dtc, 1), 0.0)
    sp_b = np.where(dt_prev > 0, bn / np.where(dt_prev > 0, dt_prev, 1), 0.0)
    sp_f = np.where(dt_next > 0, fn / np.where(dt_next > 0, dt_next, 1), 0.0)
    accel = np.where(dtc > 0, (sp_f - sp_b) / np.where(dtc > 0, dtc, 1), 0.0)
    dt_prev_f = np.nan_to_num(dt_prev, nan=1.0)
    dt_next_f = np.nan_to_num(dt_next, nan=1.0)

    # ---------------- per-arc geometry / topology ----------------
    ax_, ay_ = nxy[fr, 0], nxy[fr, 1]
    vx, vy, alen = _unit(nxy[to, 0] - ax_, nxy[to, 1] - ay_)
    out_deg = np.bincount(fr, minlength=n_nodes)
    in_deg = np.bincount(to, minlength=n_nodes)
    pair = fr.astype(np.int64) * n_nodes + to
    rev = to.astype(np.int64) * n_nodes + fr
    has_twin = np.isin(rev, pair)

    p = df.point_idx.values.astype(np.int64)
    a = df.arc_id.values.astype(np.int64)
    dist = df.dist_m.values.astype(np.float64)
    frac = df.frac.values.astype(np.float64)

    L = alen[a]
    df["log_arc_len"] = np.log1p(L)
    df["zero_len"] = (L < 1e-6).astype(np.float32)
    df["d_from_node"] = np.hypot(x[p] - ax_[a], y[p] - ay_[a])
    df["d_to_node"] = np.hypot(x[p] - nxy[to[a], 0], y[p] - nxy[to[a], 1])

    # ---------------- relative to other candidates ----------------
    g = df.groupby("point_idx").dist_m
    dmin = g.transform("min").values.astype(np.float64)
    n_c = g.transform("size").values
    # distance of the 2nd nearest candidate (or MISSING_DIST if alone) -> ambiguity of the point
    second = df[df["rank"] == 1].set_index("point_idx").dist_m
    d2 = second.reindex(p).fillna(MISSING_DIST).values.astype(np.float64)
    df["dist_gap"] = dist - dmin
    df["dist_ratio"] = dist / (dmin + 1.0)
    df["n_cands"] = n_c
    df["gap_2nd"] = d2 - dmin

    # ---------------- heading ----------------
    ux, uy = vx[a], vy[a]
    df["cos_c"] = cx[p] * ux + cy[p] * uy
    df["cos_b"] = bx[p] * ux + by[p] * uy
    df["cos_f"] = fx[p] * ux + fy[p] * uy
    df["abs_cos_c"] = np.abs(df["cos_c"].values)
    df["move_c"] = cn[p]

    # ---------------- kinematics ----------------
    df["speed"] = speed[p]
    df["dt_prev"] = dt_prev_f[p]
    df["dt_next"] = dt_next_f[p]
    df["accel"] = accel[p]

    # ---------------- topology ----------------
    df["out_deg_to"] = out_deg[to[a]]
    df["in_deg_from"] = in_deg[fr[a]]
    df["has_twin"] = has_twin[a].astype(np.float32)

    # ---------------- context: same arc at neighbouring points ----------------
    key = p * n_arcs + a
    order = np.argsort(key)
    sk, sdist, sfrac = key[order], dist[order], frac[order]
    k_prev, k_next = (p - 1) * n_arcs + a, (p + 1) * n_arcs + a
    prev_same = _lookup(sk, sdist, k_prev, MISSING_DIST)
    next_same = _lookup(sk, sdist, k_next, MISSING_DIST)
    df["prev_same_dist"] = prev_same
    df["next_same_dist"] = next_same

    # graph-connected arc at t+1 (successor: from_node == to_node(a)) / t-1 (predecessor: to_node == from_node(a))
    nk_next, nv_next = _min_by_key((p - 1) * n_nodes + fr[a], dist)   # candidates at t, keyed for point t-1, by from_node
    nk_prev, nv_prev = _min_by_key((p + 1) * n_nodes + to[a], dist)   # candidates at t, keyed for point t+1, by to_node
    succ = _lookup(nk_next, nv_next, p * n_nodes + to[a], MISSING_DIST)
    pred = _lookup(nk_prev, nv_prev, p * n_nodes + fr[a], MISSING_DIST)
    df["next_conn_dist"] = np.minimum(next_same, succ)
    df["prev_conn_dist"] = np.minimum(prev_same, pred)

    # signed progress along the arc between t-1 and t+1 (positive = moving in the arc's direction)
    f_prev = _lookup(sk, sfrac, k_prev, np.nan)
    f_next = _lookup(sk, sfrac, k_next, np.nan)
    f_prev = np.where(np.isnan(f_prev), frac, f_prev)
    f_next = np.where(np.isnan(f_next), frac, f_next)
    prog = (f_next - f_prev) * L
    df["progress_along"] = prog
    df["progress_vs_move"] = prog / (cn[p] + 1.0)

    # ---------------- windowed context ----------------
    cnt = np.zeros(len(df)); dsum = np.zeros(len(df)); valid = np.zeros(len(df))
    for off in range(-WINDOW, WINDOW + 1):
        if off == 0:
            continue
        q = p + off
        ok = (q >= 0) & (q < T)
        d_off = _lookup(sk, sdist, q * n_arcs + a, np.nan)
        hit = ok & ~np.isnan(d_off)
        valid += ok
        cnt += hit
        dsum += np.where(hit, d_off, MISSING_DIST * ok)
    df["win_frac"] = cnt / np.maximum(valid, 1)
    df["win_mean_dist"] = dsum / np.maximum(valid, 1)

    for c in FEATURE_COLUMNS:
        df[c] = df[c].astype(np.float32)
    return df

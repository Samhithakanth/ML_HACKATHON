"""
src/hmm/decoder.py  (Person 3)

HMM map matching decoded with log-space Viterbi.

Hidden state at time t : one of the candidate arcs of GPS point t (src.data.candidates.candidate_table).
Emission               : w_emis * log P_clf(candidate is the true arc)  (classifier probability, clipped).
Transition a -> b      : Newson & Krumm (2009) style. With d_gc the straight-line distance between the two
                         GPS fixes and d_route the network distance from the projection of fix t-1 on arc a
                         to the projection of fix t on arc b,
                             log p = -|d_route - d_gc| / beta
                         d_route:  same arc, moving forward      (f_b - f_a) * L_a
                                   same arc, moving backward     back_mult * (f_a - f_b) * L_a
                                   different arcs                (1 - f_a) L_a + SP(to_a, from_b) + f_b L_b
                         SP = shortest path on the directed road graph (bounded Dijkstra).
Edge cases
* no path within the search limit (disconnected components, 89/90 trips have several): floor log-prob
  `log_floor` instead of -inf, so the decoder can bridge between components when the emissions insist;
* GPS gap dt > gap_s, or a point with no candidates: the chain is cut and Viterbi restarts (independent
  segments); points without candidates get arc -1 (no prediction);
* zero-length arcs get a tiny positive weight so the sparse graph keeps them.

Everything except the Viterbi recursion itself is vectorised over the whole trip; the recursion is a
loop over time steps with (K_prev x K_cur) numpy ops (longest trip: 19,313 points).
The expensive part (route distances) is computed once per trip in `build_lattice`, so decoder parameters
can be tuned cheaply with `decode`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

DEFAULT_PARAMS = {
    "beta": 10.0,          # metres; scale of the |d_route - d_gc| transition penalty
    "w_emis": 1.0,         # weight of the classifier log-probability
    "log_floor": -20.0,    # transition log-prob when no route exists within the search limit
    "back_mult": 4.0,      # cost multiplier for moving backwards along the same arc
    "min_prob": 1e-6,      # emission probability clip
    "gap_s": 60.0,         # cut the chain when the time gap exceeds this (seconds)
}
LIMIT_CAP_M = 3000.0       # max Dijkstra search radius
CORRIDOR_M = 800.0         # graph restricted to nodes within this distance of the GPS track


@dataclass
class Lattice:
    n_points: int
    starts: np.ndarray       # (T+1,) offsets of each point's candidate rows
    arc: np.ndarray          # (rows,) candidate arc ids
    linked: np.ndarray       # (T,) step t-1 -> t is part of a chain
    pair_off: np.ndarray     # (T+1,) offsets of each step's (K_prev*K_cur) transition block
    d_route: np.ndarray      # (pairs,) route distance without the backward term (inf = unreachable)
    back_amt: np.ndarray     # (pairs,) backward movement along the same arc (metres)
    d_gc: np.ndarray         # (pairs,) straight-line distance between the two GPS fixes
    dt: np.ndarray           # (T,) time since previous fix
    gap_s: float


def _graph(trip, keep_nodes):
    """Directed CSR graph on the corridor nodes; weight = arc length (parallel arcs: min)."""
    nxy = trip.nodes[["x", "y"]].values
    fr, to = trip.arcs.from_node.values, trip.arcs.to_node.values
    L = np.hypot(*(nxy[to] - nxy[fr]).T)
    remap = -np.ones(len(nxy), dtype=np.int64)
    remap[keep_nodes] = np.arange(len(keep_nodes))
    u, v = remap[fr], remap[to]
    ok = (u >= 0) & (v >= 0) & (u != v)
    u, v, w = u[ok], v[ok], np.maximum(L[ok], 1e-3)
    order = np.lexsort((w, v, u))
    u, v, w = u[order], v[order], w[order]
    first = np.r_[True, (u[1:] != u[:-1]) | (v[1:] != v[:-1])]
    n = len(keep_nodes)
    return csr_matrix((w[first], (u[first], v[first])), shape=(n, n)), remap


def build_lattice(trip, cands, gap_s: float = DEFAULT_PARAMS["gap_s"], limit_cap: float = LIMIT_CAP_M,
                  corridor: float = CORRIDOR_M) -> Lattice:
    """cands: candidate rows of one trip sorted by point_idx (point_idx, arc_id, frac)."""
    T = len(trip.track)
    p = cands.point_idx.values.astype(np.int64)
    arc = cands.arc_id.values.astype(np.int64)
    frac = cands.frac.values.astype(np.float64)
    assert np.all(np.diff(p) >= 0), "candidates must be sorted by point_idx"
    starts = np.searchsorted(p, np.arange(T + 1))
    cnt = np.diff(starts)

    x, y, t = trip.track.x.values, trip.track.y.values, trip.track.t.values.astype(float)
    dt = np.r_[np.inf, np.diff(t)]
    step_gc = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
    linked = np.zeros(T, dtype=bool)
    linked[1:] = (cnt[1:] > 0) & (cnt[:-1] > 0) & (dt[1:] <= gap_s)

    npairs = np.zeros(T, dtype=np.int64)
    npairs[1:] = np.where(linked[1:], cnt[:-1] * cnt[1:], 0)
    pair_off = np.r_[0, np.cumsum(npairs)]
    total = int(pair_off[-1])
    step = np.repeat(np.arange(T), npairs)
    q = np.arange(total) - pair_off[step]
    k2 = cnt[step]
    i = starts[step - 1] + q // np.maximum(k2, 1)
    j = starts[step] + q % np.maximum(k2, 1)

    nxy = trip.nodes[["x", "y"]].values
    fr, to = trip.arcs.from_node.values, trip.arcs.to_node.values
    alen = np.hypot(*(nxy[to] - nxy[fr]).T)
    a, b = arc[i], arc[j]
    fa, fb = frac[i], frac[j]
    La, Lb = alen[a], alen[b]
    same = a == b
    d_route = np.empty(total)
    back = np.zeros(total)
    fwd = fb >= fa
    d_route[same] = np.where(fwd[same], (fb - fa)[same] * La[same], 0.0)
    back[same] = np.where(fwd[same], 0.0, (fa - fb)[same] * La[same])

    diff = ~same
    src, tgt = to[a[diff]], fr[b[diff]]
    sp = np.full(diff.sum(), np.inf)
    sp[src == tgt] = 0.0
    need = src != tgt
    if need.any():
        tree = cKDTree(np.c_[x, y])
        dn, _ = tree.query(nxy, distance_upper_bound=corridor)
        keep = np.flatnonzero(np.isfinite(dn))
        G, remap = _graph(trip, keep)
        s_loc, t_loc = remap[src[need]], remap[tgt[need]]
        ok = (s_loc >= 0) & (t_loc >= 0)
        vals = np.full(need.sum(), np.inf)
        if ok.any():
            gmax = step_gc[1:][linked[1:]].max() if linked[1:].any() else 0.0
            limit = float(min(limit_cap, 3.0 * gmax + 300.0))
            key = s_loc[ok] * G.shape[0] + t_loc[ok]
            ukey, inv = np.unique(key, return_inverse=True)
            us, ut = ukey // G.shape[0], ukey % G.shape[0]
            usrc, sinv = np.unique(us, return_inverse=True)
            out = np.full(len(ukey), np.inf)
            batch = max(1, int(2e7 // max(G.shape[0], 1)))
            for b0 in range(0, len(usrc), batch):
                bs = usrc[b0:b0 + batch]
                D = dijkstra(G, directed=True, indices=bs, limit=limit)
                m = (sinv >= b0) & (sinv < b0 + batch)
                out[m] = D[sinv[m] - b0, ut[m]]
            vals[ok] = out[inv]
        sp[need] = vals
    d_route[diff] = (1 - fa[diff]) * La[diff] + sp + fb[diff] * Lb[diff]
    return Lattice(T, starts, arc, linked, pair_off, d_route.astype(np.float32), back.astype(np.float32),
                   step_gc[step].astype(np.float32), dt, gap_s)


def transition_logp(lat: Lattice, params: dict) -> np.ndarray:
    d = lat.d_route + params["back_mult"] * lat.back_amt
    lp = -np.abs(d - lat.d_gc) / params["beta"]
    return np.where(np.isfinite(lp), np.maximum(lp, params["log_floor"]), params["log_floor"]).astype(np.float32)


def decode(lat: Lattice, prob: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Most likely arc per GPS point (-1 for points without candidates). prob: per candidate row."""
    P = dict(DEFAULT_PARAMS, **(params or {}))
    linked = lat.linked
    if P["gap_s"] < lat.gap_s:   # stricter gap than the lattice was built with
        linked = linked & (lat.dt <= P["gap_s"])
    emis = P["w_emis"] * np.log(np.clip(np.asarray(prob, dtype=np.float64), P["min_prob"], 1.0))
    trans = transition_logp(lat, P)
    T, st, po = lat.n_points, lat.starts, lat.pair_off
    out = np.full(T, -1, dtype=np.int64)
    bp = [None] * T
    delta, seg0 = None, None

    def finish(t_end):
        jj = int(np.argmax(delta))
        for tt in range(t_end, seg0 - 1, -1):
            out[tt] = lat.arc[st[tt] + jj]
            if tt > seg0:
                jj = int(bp[tt][jj])

    for t in range(T):
        k = st[t + 1] - st[t]
        if k == 0:
            if delta is not None:
                finish(t - 1); delta = None
            continue
        em = emis[st[t]:st[t + 1]]
        if delta is None or not linked[t]:
            if delta is not None:
                finish(t - 1)
            delta, seg0 = em.copy(), t
            continue
        k1 = len(delta)
        M = delta[:, None] + trans[po[t]:po[t + 1]].reshape(k1, k)
        arg = M.argmax(axis=0)
        delta = M[arg, np.arange(k)] + em
        delta -= delta.max()
        bp[t] = arg
    if delta is not None:
        finish(T - 1)
    return out

"""
src/hmm/break_analysis.py  (Person 3)
Classifies the remaining continuity "breaks" of the tuned HMM on student-test: a break is a pair of
consecutive points whose arcs are neither identical nor directly connected. It can be
  * multi-hop  : a directed path of <= 300 m joins them (the vehicle crossed >= 1 short arc within one
                 GPS interval; not a real discontinuity, the continuity metric is strict),
  * gap/cut    : the chain was cut there (dt > gap_s or a point without candidates),
  * jump       : no directed path of <= 300 m (a true discontinuity, e.g. across network components).
    python -m src.hmm.break_analysis   ->  results/hmm/break_analysis.csv
"""
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from src.data.loader import load_trip, REPO_ROOT
from src.features.dataset import load_partition
from src.hmm.decoder import build_lattice, decode


def main():
    F, P = load_partition("student_test")
    prob = np.load(REPO_ROOT / "results" / "test_prob_lightgbm_tuned.npy")
    prm = yaml.safe_load(open(REPO_ROOT / "configs" / "hmm.yaml"))["hmm"]
    rows = []
    for tid, g in P.groupby("trip_id"):
        m = (F.trip_id == tid).values
        trip = load_trip(tid, "student_test", load_route=False)
        lat = build_lattice(trip, F[m], gap_s=prm["gap_s"])
        pred = decode(lat, prob[m], prm)
        a = trip.arcs.values
        xy = trip.nodes[["x", "y"]].values
        w = np.maximum(np.hypot(*(xy[a[:, 1]] - xy[a[:, 0]]).T), 1e-3)
        G = csr_matrix((w, (a[:, 0], a[:, 1])), shape=(len(xy), len(xy)))
        u, v = pred[:-1], pred[1:]
        ok = (u >= 0) & (v >= 0)
        brk = np.flatnonzero(~ok | ((u != v) & (a[np.maximum(u, 0), 1] != a[np.maximum(v, 0), 0])))
        cut = ~lat.linked[1:] if prm["gap_s"] >= lat.gap_s else ~(lat.linked[1:] & (lat.dt[1:] <= prm["gap_s"]))
        kinds = []
        src = np.unique(a[u[brk][ok[brk]], 1]) if len(brk) else []
        D = dict(zip(src, dijkstra(G, indices=src, limit=300.0))) if len(src) else {}
        for i in brk:
            if cut[i] or not ok[i]:
                kinds.append("gap/cut")
            elif np.isfinite(D[a[u[i], 1]][a[v[i], 0]]):
                kinds.append("multi-hop (path <= 300 m)")
            else:
                kinds.append("jump (no path <= 300 m)")
        k = pd.Series(kinds, dtype=object).value_counts()
        rows.append({"trip_id": tid, "n_points": len(g), "breaks": len(brk), **k.to_dict()})
    d = pd.DataFrame(rows).fillna(0)
    d.loc["total"] = d.sum(numeric_only=True); d.loc["total", "trip_id"] = "total"
    d.to_csv(REPO_ROOT / "results" / "hmm" / "break_analysis.csv", index=False)
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()

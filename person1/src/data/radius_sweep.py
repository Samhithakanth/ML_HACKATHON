"""Choose candidate radius R / cap K from TRAIN trips only (candidate miss rate vs cost)."""
import numpy as np, pandas as pd
from src.data.loader import list_trips, load_trip, RAW_DIR
from src.data.labels import align_to_route
from src.data.candidates import generate_candidates

RS = [20, 30, 40, 50, 75, 100, 150, 200]
KS = [5, 10, 20, 30, 50, 100]
if __name__ == "__main__":
    rec = []
    for tid in list_trips("train"):
        t = load_trip(tid, "train")
        _, arc, dl = align_to_route(t)
        c = generate_candidates(t, max(RS))
        lab = pd.Series(arc, index=np.arange(len(arc)))
        c["label"] = (c.arc_id.values == lab.loc[c.point_idx].values)
        # the labelled arc may have a bidirectional twin with identical geometry: record rank of true arc
        tr = c[c.label].set_index("point_idx")
        rank = pd.Series(np.nan, index=np.arange(len(arc))); rank.loc[tr.index] = tr["rank"].values
        for R in RS:
            nc = c[c.dist_m <= R].groupby("point_idx").size().reindex(np.arange(len(arc)), fill_value=0)
            rec.append(dict(trip_id=tid, R=R, n=len(arc), miss=int((dl > R).sum()), cands=int(nc.sum()), max_c=int(nc.max())))
        for K in KS:  # at R=max: label rank (by distance) < K ?
            rec.append(dict(trip_id=tid, R=-K, n=len(arc), miss=int((~(rank < K)).sum()), cands=0, max_c=0))
        print(tid, flush=True)
    d = pd.DataFrame(rec)
    d.to_csv(RAW_DIR.parents[1] / "results" / "eda" / "radius_sweep_by_trip.csv", index=False)
    bad = {"00000033", "00000054"}
    for name, sub in [("all train", d), ("excl. 33,54", d[~d.trip_id.isin(bad)])]:
        g = sub.groupby("R").agg(n=("n", "sum"), miss=("miss", "sum"), cands=("cands", "sum"))
        g["miss_rate"] = g.miss / g.n; g["cands_per_pt"] = g.cands / g.n
        print(name); print(g.sort_index(ascending=False).round(4).to_string())

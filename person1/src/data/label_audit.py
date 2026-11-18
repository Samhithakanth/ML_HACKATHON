"""Align every trip to its reference route and record label-quality stats."""
import numpy as np, pandas as pd
from src.data.loader import PARTITIONS, list_trips, load_trip, RAW_DIR
from src.data.labels import align_to_route

if __name__ == "__main__":
    rows, store = [], {}
    for p in PARTITIONS:
        for tid in list_trips(p):
            t = load_trip(tid, p)
            j, arc, d = align_to_route(t)
            rows.append(dict(trip_id=tid, partition=p, n_gps=len(d), n_route=len(t.route),
                             d_med=np.median(d), d_p90=np.percentile(d, 90), d_p99=np.percentile(d, 99), d_max=d.max(),
                             frac_gt50=(d > 50).mean(), frac_gt100=(d > 100).mean(), frac_gt200=(d > 200).mean(),
                             route_cov=len(np.unique(j)) / len(t.route)))
            print(p, tid, round(rows[-1]["d_med"], 1), round(rows[-1]["frac_gt100"], 3), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR.parents[1] / "results" / "eda" / "label_audit.csv", index=False)

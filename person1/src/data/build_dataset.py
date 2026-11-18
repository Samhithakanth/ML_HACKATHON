"""Build points + candidates tables (parquet) for train and student_test.

points.parquet      : one row per GPS observation, with derived reference arc (label_arc)
candidates.parquet  : one row per (GPS point, candidate arc) with label = (arc_id == label_arc)
Settings (chosen from TRAIN-trip miss-rate sweep, see results/eda/radius_sweep_by_trip.csv):
    RADIUS_M = 100, MAX_K = 30
"""
import json, numpy as np, pandas as pd
from src.data.loader import PARTITIONS, list_trips, load_trip, RAW_DIR
from src.data.labels import align_to_route
from src.data.candidates import generate_candidates

RADIUS_M, MAX_K = 100.0, 30
BAD_LABEL_FRAC = 0.10   # trip flagged if >10% of points are >100 m from their aligned reference arc
OUT = RAW_DIR.parents[1] / "data" / "processed"


def build_trip(t):
    j, arc, dl = align_to_route(t)
    n = len(t.track)
    pts = pd.DataFrame({"trip_id": t.trip_id, "partition": t.partition, "point_idx": np.arange(n),
                        "lon": t.track.lon.values, "lat": t.track.lat.values, "t": t.track.t.values,
                        "x": t.track.x.values, "y": t.track.y.values,
                        "dt_prev": np.r_[np.nan, np.diff(t.track.t.values)],
                        "label_route_pos": j, "label_arc": arc, "label_dist_m": dl})
    c = generate_candidates(t, RADIUS_M, MAX_K)
    c["label"] = (c.arc_id.values == arc[c.point_idx.values]).astype(np.int8)
    hit = c.groupby("point_idx").label.max().reindex(np.arange(n), fill_value=0)
    pts["label_in_candidates"] = hit.values.astype(bool)
    pts["n_candidates"] = c.groupby("point_idx").size().reindex(np.arange(n), fill_value=0).values
    c.insert(0, "trip_id", t.trip_id)
    c = c.astype({"point_idx": "int32", "arc_id": "int32", "rank": "int16", "dist_m": "float32", "frac": "float32"})
    return pts, c


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for p in PARTITIONS:
        P, C, Q = [], [], []
        for tid in list_trips(p):
            t = load_trip(tid, p)
            pts, c = build_trip(t)
            bad = (pts.label_dist_m > RADIUS_M).mean() > BAD_LABEL_FRAC
            pts["trip_label_suspect"] = bad
            P.append(pts); C.append(c)
            Q.append(dict(trip_id=tid, n=len(pts), miss_rate=1 - pts.label_in_candidates.mean(), suspect=bool(bad)))
            print(p, tid, len(pts), len(c), round(Q[-1]["miss_rate"], 4), "SUSPECT" if bad else "", flush=True)
        P = pd.concat(P, ignore_index=True); C = pd.concat(C, ignore_index=True)
        P.to_parquet(OUT / f"{p}_points.parquet", index=False)
        C.to_parquet(OUT / f"{p}_candidates.parquet", index=False)
        pd.DataFrame(Q).to_csv(OUT / f"{p}_quality.csv", index=False)
        ok = P[~P.trip_label_suspect]
        print(f"== {p}: {len(P)} points, {len(C)} candidate rows, miss rate all={1-P.label_in_candidates.mean():.4f} "
              f"clean-trips={1-ok.label_in_candidates.mean():.4f}, cands/pt={len(C)/len(P):.1f}")
    json.dump({"radius_m": RADIUS_M, "max_k": MAX_K, "projection": "per-trip aeqd centred on GPS centroid",
               "bad_label_frac": BAD_LABEL_FRAC}, open(OUT / "build_config.json", "w"), indent=2)

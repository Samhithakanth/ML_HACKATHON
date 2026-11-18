"""Dataset inventory + data-quality scan for train and student_test partitions."""
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from src.data.p1_loader import RAW_DIR, PARTITIONS, list_trips, load_trip

OUT = RAW_DIR.parents[1] / "results" / "eda"


def trip_stats(t):
    tr, n, a, r = t.track, t.nodes, t.arcs, t.route
    dt = np.diff(tr.t.values)
    step = np.hypot(np.diff(tr.x), np.diff(tr.y))
    N = len(n)
    g = coo_matrix((np.ones(len(a)), (a.from_node, a.to_node)), shape=(N, N))
    ncomp_w, lab_w = connected_components(g, directed=True, connection="weak")
    ncomp_s, _ = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(a.from_node, minlength=N) + np.bincount(a.to_node, minlength=N)
    ra = a.iloc[r]
    xy = n[["x", "y"]].values
    alen = np.hypot(*(xy[a.to_node.values] - xy[a.from_node.values]).T)
    inbox = ((tr.lon >= n.lon.min()) & (tr.lon <= n.lon.max()) & (tr.lat >= n.lat.min()) & (tr.lat <= n.lat.max())).mean()
    return dict(
        trip_id=t.trip_id, partition=t.partition,
        n_gps=len(tr), n_nodes=N, n_arcs=len(a), n_route_arcs=len(r),
        files_ok=True,
        gps_nan=int(tr.isna().sum().sum()),
        gps_dup_rows=int(tr.duplicated().sum()),
        lonlat_swapped_suspect=bool((tr.lon > 30).any()),
        t_start=tr.t.iloc[0], duration_s=tr.t.iloc[-1] - tr.t.iloc[0],
        t_monotonic=bool((dt > 0).all()), dt_eq1_frac=float((dt == 1).mean()),
        dt_max=dt.max(), n_gaps_gt5s=int((dt > 5).sum()),
        step_median_m=np.median(step), step_max_m=step.max(), n_jumps_gt100m=int((step > 100).sum()),
        gps_extent_km_x=(tr.x.max() - tr.x.min()) / 1e3, gps_extent_km_y=(tr.y.max() - tr.y.min()) / 1e3,
        gps_in_net_bbox=inbox,
        weak_components=ncomp_w, strong_components=ncomp_s,
        isolated_nodes=int((deg == 0).sum()),
        arc_len_median_m=np.median(alen), arc_len_max_m=alen.max(), zero_len_arcs=int((alen == 0).sum()),
        route_in_range=bool(r.max() < len(a) and r.min() >= 0),
        route_connected_frac=float((ra.to_node.values[:-1] == ra.from_node.values[1:]).mean()) if len(r) > 1 else 1.0,
        route_len_km=alen[r].sum() / 1e3, track_len_km=step.sum() / 1e3,
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in PARTITIONS:
        for tid in list_trips(p):
            rows.append(trip_stats(load_trip(tid, p)))
            print(p, tid, rows[-1]["n_gps"], rows[-1]["n_nodes"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trip_inventory.csv", index=False)
    print(df.groupby("partition").describe().T.to_string())

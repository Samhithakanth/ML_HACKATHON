"""EDA figures (partition named in each title)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; E = ROOT / "results" / "eda"; O = E / "plots"; O.mkdir(exist_ok=True, parents=True)
inv = pd.read_csv(E / "trip_inventory.csv", dtype={"trip_id": str}); sw = pd.read_csv(E / "radius_sweep_by_trip.csv", dtype={"trip_id": str})
tr = inv[inv.partition == "train"]; st = inv[inv.partition == "student_test"]
pts = {p: pd.read_parquet(ROOT / "data/processed" / f"{p}_points.parquet", columns=["trip_id", "dt_prev", "label_dist_m", "label_in_candidates", "n_candidates"]) for p in ["train", "student_test"]}

def save(name): plt.tight_layout(); plt.savefig(O / name, dpi=130); plt.close()

# 1 GPS points per trip
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for a, (n, d) in zip(ax, [("train (80 trips)", tr), ("student_test (10 trips)", st)]):
    a.hist(d.n_gps, bins=np.logspace(np.log10(d.n_gps.min()), np.log10(d.n_gps.max()), 20), color="#2a6fbb"); a.set_xscale("log"); a.set_title(f"GPS points per trip - {n}"); a.set_xlabel("points (log)")
save("01_points_per_trip.png")
# 2 time step distribution
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for a, p in zip(ax, ["train", "student_test"]):
    dt = pts[p].dt_prev.dropna(); vc = dt.clip(upper=11).round().value_counts().sort_index()
    a.bar(vc.index, vc.values, color="#2a6fbb"); a.set_yscale("log"); a.set_title(f"Δt between fixes (s, clipped at 11) - {p}")
    a.set_xlabel(f"{(dt==1).mean():.1%} exactly 1 s; max {dt.max():.0f} s")
save("02_dt_distribution.png")
# 3 network size & connectivity
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
ax[0].scatter(tr.n_nodes, tr.n_arcs, s=12, label="train"); ax[0].scatter(st.n_nodes, st.n_arcs, s=18, c="r", label="student_test")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs"); ax[0].legend(); ax[0].set_title("Road-network size per trip")
ax[1].hist(inv.weak_components, bins=30, color="#2a6fbb"); ax[1].set_title("Weakly connected components (train+test)"); ax[1].set_xlabel("components")
ax[2].hist(inv.arc_len_median_m, bins=30, color="#2a6fbb"); ax[2].set_title("Median arc length (m)")
save("03_network_stats.png")
# 4 label-to-GPS distance (derived reference labels)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for a, p in zip(ax, ["train", "student_test"]):
    d = pts[p].label_dist_m; a.hist(d.clip(upper=250), bins=50, color="#2a6fbb"); a.set_yscale("log")
    a.set_title(f"GPS->reference-arc distance (m) - {p}"); a.set_xlabel(f"median {d.median():.1f} m, p99 {d.quantile(.99):.0f} m")
save("04_label_distance.png")
# 5 candidate miss rate vs radius & K (train, clean vs all)
bad = {"00000033", "00000054"}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
for nm, sub in [("all train", sw), ("excl. suspect trips 33,54", sw[~sw.trip_id.isin(bad)])]:
    g = sub.groupby("R").agg(n=("n", "sum"), miss=("miss", "sum"), c=("cands", "sum")); g["mr"] = g.miss / g.n; g["cpp"] = g.c / g.n
    r = g[g.index > 0]; k = g[g.index < 0].sort_index(ascending=False)
    ax[0].plot(r.index, r.mr * 100, "o-", label=nm); ax[1].plot(-k.index, k.mr * 100, "o-", label=nm)
ax[0].set_xlabel("radius R (m)"); ax[0].set_ylabel("candidate miss rate (%)"); ax[0].set_title("Miss rate vs radius (train)"); ax[0].axvline(100, ls="--", c="k"); ax[0].legend()
ax[1].set_xlabel("K nearest arcs (R=200)"); ax[1].set_title("Miss rate vs top-K cap (train)"); ax[1].axvline(30, ls="--", c="k")
save("05_candidate_miss_rate.png")
# 6 candidates per point
fig, ax = plt.subplots(figsize=(5, 3.5)); ax.hist(pts["train"].n_candidates, bins=np.arange(-0.5, 31.5, 1), color="#2a6fbb"); ax.set_title("Candidates per GPS point (R=100 m, K<=30) - train"); save("06_candidates_per_point.png")
print("ok")

"""Visual + numeric sanity check of GPS->route alignment on selected trips."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.loader import load_trip, RAW_DIR
from src.data.labels import align_to_route

CASES = [("train", "00000090"), ("train", "00000078"), ("train", "00000021"), ("train", "00000042"),
         ("student_test", "00000007"), ("student_test", "00000091")]

def diag(t, j, d):
    dt = np.diff(t.track.t.values); dj = np.diff(j)
    big = dt > 60
    return dict(n=len(j), d_med=np.median(d), d_p99=np.percentile(d, 99), d_max=d.max(),
                backward=int((dj < 0).sum()), jump_gt20_no_gap=int(((dj > 20) & ~big).sum()),
                jump_gt20_at_gap=int(((dj > 20) & big).sum()), max_jump_no_gap=int(dj[~big].max()) if (~big).any() else 0,
                route_cov=len(np.unique(j)) / len(t.route), n_big_gaps=int(big.sum()))

if __name__ == "__main__":
    out = RAW_DIR.parents[1] / "results" / "eda" / "plots"
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (p, tid) in zip(axes.ravel(), CASES):
        t = load_trip(tid, p); j, arc, d = align_to_route(t)
        print(p, tid, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in diag(t, j, d).items()})
        xy = t.nodes[["x", "y"]].values; ra = t.arcs.iloc[t.route]
        for f, to in zip(ra.from_node, ra.to_node): ax.plot(*zip(xy[f], xy[to]), c="#1a9850", lw=2.5, zorder=1)
        ax.scatter(t.track.x, t.track.y, s=3, c="#2166ac", zorder=2)
        bad = d > 50; ax.scatter(t.track.x[bad], t.track.y[bad], s=8, c="red", zorder=3)
        ax.set_title(f"{p} {tid}: n={len(d)}, med d={np.median(d):.1f} m, red=>50 m"); ax.set_aspect("equal"); ax.tick_params(labelsize=6)
    plt.tight_layout(); plt.savefig(out / "07_label_alignment_check.png", dpi=110)

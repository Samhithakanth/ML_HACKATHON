"""
src/viz/static_route_plot.py  (Persons 2/3, for the PDF report)
Static (matplotlib) rendering of the same layers as the Folium maps: GPS fixes, nearby road network,
reference route, classifier-only route and HMM route, for one student-test trip, in local metres.
Used in the PDF where an interactive map cannot be embedded; the interactive maps are in results/maps/.

    python -m src.viz.static_route_plot 00000077 [x0 x1 y0 y1 | full]   ->  results/figures/route_trip_<id>.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.collections import LineCollection

from src.data.labels import align_trip_to_route
from src.data.loader import load_trip, REPO_ROOT
from src.eval.metrics import top1_from_scores, compute_top1_accuracy
from src.features.dataset import load_partition
from src.hmm.decoder import build_lattice, decode


def segs(trip, arc_ids):
    xy = trip.nodes[["x", "y"]].values
    a = trip.arcs.values[np.unique(arc_ids[arc_ids >= 0])]
    return np.stack([xy[a[:, 0]], xy[a[:, 1]]], axis=1)


def main(tid="00000077", window=None):
    F, P = load_partition("student_test")
    m = (F.trip_id == tid).values
    Fi = F[m]
    prob = np.load(REPO_ROOT / "results" / "test_prob_lightgbm_tuned.npy")[m]
    prm = yaml.safe_load(open(REPO_ROOT / "configs" / "hmm.yaml"))["hmm"]
    trip = load_trip(tid, "student_test", load_route=True)
    ref = align_trip_to_route(trip)[1]
    clf = top1_from_scores(Fi, prob, len(trip.track))
    hmm = decode(build_lattice(trip, Fi, gap_s=prm["gap_s"]), prob, prm)
    x, y = trip.track.x.values, trip.track.y.values
    if window == "full":  # whole trip, square-ish padding
        pad = 150
        window = (x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad)
    if window is None:   # the 1.2 km window around the GPS stretch with the most classifier errors
        err = (clf != ref).astype(float)
        k = int(np.argmax(np.convolve(err, np.ones(200), "same")))
        window = (x[k] - 600, x[k] + 600, y[k] - 450, y[k] + 450)
    x0, x1, y0, y1 = window
    nxy = trip.nodes[["x", "y"]].values
    a = trip.arcs.values
    inwin = lambda p: (p[:, 0] > x0 - 200) & (p[:, 0] < x1 + 200) & (p[:, 1] > y0 - 200) & (p[:, 1] < y1 + 200)
    keep = inwin(nxy[a[:, 0]]) | inwin(nxy[a[:, 1]])
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.add_collection(LineCollection(np.stack([nxy[a[keep, 0]], nxy[a[keep, 1]]], 1), colors="#c9ced6", linewidths=0.8,
                                     label="road network"))
    ax.add_collection(LineCollection(segs(trip, ref), colors="#16a34a", linewidths=6, alpha=0.45, label="reference route"))
    ax.add_collection(LineCollection(segs(trip, clf), colors="#dc2626", linewidths=1.6, linestyles=(0, (3, 2)),
                                     label="classifier only (argmax)"))
    ax.add_collection(LineCollection(segs(trip, hmm), colors="#f97316", linewidths=2.6, label="classifier + HMM"))
    ax.scatter(x, y, s=5, color="#2563eb", zorder=5, label="GPS fixes (1 Hz)")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_xlabel("x (m, local azimuthal-equidistant)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Student-test trip {tid}: top-1 classifier {compute_top1_accuracy(clf, ref):.3f} → HMM "
                 f"{compute_top1_accuracy(hmm, ref):.3f} (whole trip)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = REPO_ROOT / "results" / "figures" / f"route_trip_{tid}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out, "window", [round(v) for v in window])


if __name__ == "__main__":
    tid = sys.argv[1] if len(sys.argv) > 1 else "00000077"
    w = "full" if sys.argv[2:3] == ["full"] else (tuple(map(float, sys.argv[2:6])) if len(sys.argv) >= 6 else None)
    main(tid, w)

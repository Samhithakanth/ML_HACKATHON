"""
src/models/error_analysis.py  (Person 2/3)
Where does the classifier (argmax per point) go wrong on student-test, and does the HMM fix it?

    python -m src.models.error_analysis      # needs results/test_prob_lightgbm_tuned.npy and configs/hmm.yaml

Error types: reverse twin (same street, opposite direction), adjacent arc (shares a node: intersection),
no candidate, other. Also split by speed (< 2 m/s: heading is unreliable when the vehicle is ~stopped).
Output: results/error_analysis.csv
"""
import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_trip, REPO_ROOT
from src.eval.metrics import top1_from_scores
from src.features.dataset import load_partition
from src.hmm.decoder import build_lattice, decode


def error_types(pred, ref, arcs):
    q = np.maximum(pred, 0)
    twin = (pred >= 0) & (arcs[q, 0] == arcs[ref, 1]) & (arcs[q, 1] == arcs[ref, 0])
    adj = (pred >= 0) & ~twin & ((arcs[q, 1] == arcs[ref, 0]) | (arcs[q, 0] == arcs[ref, 1]))
    return np.select([pred == ref, pred < 0, twin, adj], ["correct", "no candidate", "reverse twin", "adjacent arc"],
                     "other")


def main():
    F, P = load_partition("student_test")
    prob = np.load(REPO_ROOT / "results" / "test_prob_lightgbm_tuned.npy")
    params = yaml.safe_load(open(REPO_ROOT / "configs" / "hmm.yaml"))["hmm"]
    rows = []
    for tid, g in P.groupby("trip_id"):
        m = (F.trip_id == tid).values
        Fi, ref = F[m], g.label_arc.values
        trip = load_trip(tid, "student_test", load_route=False)
        arcs = trip.arcs.values
        speed = Fi.groupby("point_idx").speed.first().reindex(range(len(g))).fillna(0).values
        clf = top1_from_scores(Fi, prob[m], len(g))
        hmm = decode(build_lattice(trip, Fi, gap_s=params["gap_s"]), prob[m], params)
        for name, pred in (("classifier only", clf), ("classifier + HMM", hmm)):
            rows.append(pd.DataFrame({"method": name, "trip_id": tid, "type": error_types(pred, ref, arcs),
                                      "slow": speed < 2.0}))
    d = pd.concat(rows, ignore_index=True)
    out = []
    for name, g in d.groupby("method", sort=False):
        n = len(g)
        r = {"method": name, "n_points": n, "error_rate": float((g.type != "correct").mean()),
             "error_rate_slow(<2m/s)": float((g[g.slow].type != "correct").mean()),
             "error_rate_moving": float((g[~g.slow].type != "correct").mean())}
        for t in ("reverse twin", "adjacent arc", "no candidate", "other"):
            r[f"share_of_points_{t}"] = float((g.type == t).mean())
        out.append(r)
    out = pd.DataFrame(out)
    out.to_csv(REPO_ROOT / "results" / "error_analysis.csv", index=False)
    print(out.round(4).T.to_string())


if __name__ == "__main__":
    main()

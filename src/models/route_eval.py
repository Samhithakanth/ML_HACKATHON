"""
src/models/route_eval.py  (Person 2/3)
Sequence-level numbers for every model of the comparison (student-test, all 10 trips):
classifier-only route continuity / edit distance, and top-1 after HMM decoding with the tuned decoder
(configs/hmm.yaml). Then writes the formatted Section 3.1 table.

    python -m src.models.route_eval

Inputs: results/test_prob_<model>.npy (written by train_compare / bayes_opt), configs/hmm.yaml
Inputs (optional): results/inference_timing.csv (python -m src.models.timing)
Outputs: results/model_route_metrics.csv, results/model_comparison_table.csv (report Section 3.1)
"""
import os

import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import yaml

from src.data.loader import load_trip, REPO_ROOT
from src.eval.metrics import sequence_metrics, top1_from_scores
from src.features.dataset import load_partition
from src.hmm.decoder import build_lattice, decode

RES = REPO_ROOT / "results"
NAMES = {"logistic_regression": "Logistic Regression", "gaussian_nb": "Gaussian Naive Bayes", "knn": "K-Nearest Neighbours",
         "random_forest": "Random Forest", "lightgbm": "LightGBM (untuned)", "mlp": "MLP",
         "lightgbm_tuned": "LightGBM (Bayesian-tuned)"}


def _trip_rows(tid, ref, Fi, probs, params):
    """All models on one trip (one lattice shared by every model). Runs in a worker process."""
    trip = load_trip(tid, "student_test", load_route=False)
    arcs = trip.arcs.values
    lat = build_lattice(trip, Fi, gap_s=params["gap_s"])
    rows = []
    for m, p in probs.items():
        c = sequence_metrics(top1_from_scores(Fi, p, len(ref)), ref, arcs)
        h = sequence_metrics(decode(lat, p, params), ref, arcs)
        rows.append({"model": m, "trip_id": tid, "n": len(ref), "clf_top1": c["top1"], "clf_continuity": c["continuity"],
                     "clf_edit_distance": c["edit_distance"], "hmm_top1": h["top1"], "hmm_continuity": h["continuity"],
                     "hmm_edit_distance": h["edit_distance"]})
    return rows


def main():
    F, P = load_partition("student_test")
    params = yaml.safe_load(open(REPO_ROOT / "configs" / "hmm.yaml"))["hmm"]
    models = [m for m in NAMES if (RES / f"test_prob_{m}.npy").exists()]
    probs = {m: np.load(RES / f"test_prob_{m}.npy") for m in models}
    groups = [(tid, g.label_arc.values, (F.trip_id == tid).values) for tid, g in P.groupby("trip_id")]
    out = Parallel(n_jobs=min(len(groups), os.cpu_count() or 1))(
        delayed(_trip_rows)(tid, ref, F[msk][["point_idx", "arc_id", "frac"]], {m: probs[m][msk] for m in models}, params)
        for tid, ref, msk in groups)
    rows = [r for o in out for r in o]
    D = pd.DataFrame(rows)
    S = D.groupby("model").apply(lambda g: pd.Series({
        c: np.average(g[c], weights=g.n) for c in ["clf_top1", "clf_continuity", "hmm_top1", "hmm_continuity"]
    } | {"clf_edit_distance": g.clf_edit_distance.mean(), "hmm_edit_distance": g.hmm_edit_distance.mean()}),
        include_groups=False).reset_index()
    S.to_csv(RES / "model_route_metrics.csv", index=False)

    C = pd.read_csv(RES / "model_comparison.csv").set_index("model")
    # inference time: idle-machine benchmark (src/models/timing.py) if available, else the time recorded during training
    timing = (pd.read_csv(RES / "inference_timing.csv").set_index("model").ms_per_1k_rows
              if (RES / "inference_timing.csv").exists() else None)
    T = []
    tuned = pd.read_csv(RES / "tuning" / "untuned_vs_tuned.csv") if (RES / "tuning" / "untuned_vs_tuned.csv").exists() else None
    for m in models:
        s = S.set_index("model").loc[m]
        if m in C.index:
            r = C.loc[m]
            cv_f1 = f"{r.cv_macro_f1_mean:.4f} ± {r.cv_macro_f1_std:.4f}"
            fam, cfg, f1, ll, inf = r.family, r.config, r.test_macro_f1, r.test_log_loss, r.inference_ms_per_1k_rows
        elif m == "lightgbm_tuned" and tuned is not None:
            r = tuned.iloc[1]
            cv_f1, fam, cfg = f"{r.cv_macro_f1:.4f}", "Gradient boosting", "Optuna TPE best (see Section 4)"
            f1, ll, inf = r.test_macro_f1, r.test_log_loss, r.inference_ms_per_1k_rows
        else:
            continue
        T.append({"Model": NAMES[m], "Algorithm family": fam, "Key configuration": cfg, "CV macro F1 (5-fold, train)": cv_f1,
                  "Student-test top-1 (classifier only)": round(s.clf_top1, 4),
                  "Student-test top-1 (+HMM)": round(s.hmm_top1, 4),
                  "Route continuity (classifier only)": round(s.clf_continuity, 4),
                  "Route continuity (+HMM)": round(s.hmm_continuity, 4),
                  "Student-test macro F1": round(f1, 4), "Student-test log loss": round(ll, 4),
                  "Inference time (ms / 1k candidates)": round(float(timing[m]) if timing is not None and m in timing.index else inf, 2)})
    T = pd.DataFrame(T)
    T.to_csv(RES / "model_comparison_table.csv", index=False)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()

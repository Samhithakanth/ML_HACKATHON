"""
src/hmm/evaluate.py  (Person 3)

1. Tune the HMM decoder parameters (Optuna TPE) on the 78 clean TRAIN trips using the tuned model's
   5-fold out-of-fold probabilities (so the emissions have test-like quality, no leakage).
   Objective: observation-level top-1 over all clean train points.
2. Compare on the 10 student-test trips (never used for any choice):
       classifier-only (argmax per point)  vs  classifier + HMM (tuned decoder)
   plus two ablations: HMM with the default decoder parameters, and HMM with a distance-only
   emission (no classifier: classical Newson-Krumm Gaussian emission, sigma = 10 m).
3. Report the label-suspect trips 33 and 54 separately.

    python -m src.hmm.evaluate [n_trials]      # default 30 decoder trials

Outputs: configs/hmm.yaml (tuned decoder), results/hmm/decoder_trials.csv, results/hmm/hmm_vs_classifier_per_trip.csv,
results/hmm/hmm_vs_classifier_summary.csv, results/figures/hmm_*.png
"""
from __future__ import annotations

import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yaml
from joblib import Parallel, delayed

from src.data.loader import REPO_ROOT, SUSPECT_TRIPS, load_trip
from src.eval.metrics import sequence_metrics, top1_from_scores
from src.features.dataset import load_partition, PROC_DIR
from src.hmm.decoder import build_lattice, decode, DEFAULT_PARAMS
from src.models.common import SEED

OUT = REPO_ROOT / "results" / "hmm"
FIG = REPO_ROOT / "results" / "figures"
LATTICE_GAP_S = 120.0        # lattices are built with the loosest gap; decode() can cut stricter
DIST_SIGMA = 10.0


def _lattice(tid, partition, Fi):
    trip = load_trip(tid, partition, load_route=False)
    return tid, build_lattice(trip, Fi, gap_s=LATTICE_GAP_S), trip.arcs.values


def lattices(F, partition, trip_ids):
    groups = {tid: g[["point_idx", "arc_id", "frac"]] for tid, g in F.groupby("trip_id") if tid in trip_ids}
    out = Parallel(n_jobs=-1)(delayed(_lattice)(tid, partition, groups[tid]) for tid in sorted(groups))
    return {tid: (lat, arcs) for tid, lat, arcs in out}


def total_top1(L, F, P, prob, params):
    correct = 0
    for tid, (lat, _) in L.items():
        m = (F.trip_id == tid).values
        pred = decode(lat, prob[m], params)
        correct += int((pred == P.loc[P.trip_id == tid, "label_arc"].values).sum())
    return correct / len(P[P.trip_id.isin(L)])


def compare(L, F, P, variants, partition):
    rows = []
    for tid, (lat, arcs) in L.items():
        m = (F.trip_id == tid).values
        Fi, ref = F[m], P.loc[P.trip_id == tid, "label_arc"].values
        for name, (prob, params) in variants.items():
            t0 = time.perf_counter()
            pred = top1_from_scores(Fi, prob[m], len(ref)) if params is None else decode(lat, prob[m], params)
            rows.append({"partition": partition, "trip_id": tid, "method": name, "n_points": len(ref),
                         "decode_s": time.perf_counter() - t0, **sequence_metrics(pred, ref, arcs)})
    return pd.DataFrame(rows)


def summarise(D):
    w = lambda g: pd.Series({
        "n_trips": g.trip_id.nunique(), "n_points": g.n_points.sum(),
        "top1 (point-weighted)": np.average(g.top1, weights=g.n_points),
        "top1 (trip mean)": g.top1.mean(), "continuity": np.average(g.continuity, weights=g.n_points),
        "breaks / 1k points": 1000 * g.n_breaks.sum() / g.n_points.sum(),
        "edit_distance (trip mean)": g.edit_distance.mean(), "route_precision": g.route_precision.mean(),
        "route_recall": g.route_recall.mean(), "route_f1": g.route_f1.mean(),
        "decode_s_per_1k_points": 1000 * g.decode_s.sum() / g.n_points.sum()})
    return D.groupby(["partition", "method"], sort=False).apply(w, include_groups=False).reset_index()


def main(n_trials=30):
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    F_all, P_all = load_partition("train")
    clean = ~F_all.trip_id.isin(SUSPECT_TRIPS).values
    F, P = F_all[clean].reset_index(drop=True), P_all[~P_all.trip_id.isin(SUSPECT_TRIPS)].reset_index(drop=True)
    oof = np.load(PROC_DIR / "train_oof_prob_tuned.npy")
    assert len(oof) == len(F) and not np.isnan(oof).any()

    t0 = time.time()
    L_train = lattices(F, "train", set(P.trip_id))
    print(f"train lattices: {len(L_train)} trips in {time.time() - t0:.0f} s", flush=True)

    # ---------------- 1. decoder tuning on train (OOF emissions) ----------------
    def objective(trial):
        params = {"beta": trial.suggest_float("beta", 1.0, 100.0, log=True),
                  "w_emis": trial.suggest_float("w_emis", 0.2, 3.0, log=True),
                  "log_floor": trial.suggest_float("log_floor", -40.0, -5.0),
                  "back_mult": trial.suggest_float("back_mult", 1.0, 20.0, log=True),
                  "gap_s": trial.suggest_float("gap_s", 10.0, LATTICE_GAP_S)}
        s = total_top1(L_train, F, P, oof, params)
        print(f"decoder trial {trial.number:2d}: top1={s:.4f} {json.dumps({k: round(v, 3) for k, v in params.items()})}",
              flush=True)
        return s

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8))
    study.enqueue_trial({k: DEFAULT_PARAMS[k] for k in ("beta", "w_emis", "log_floor", "back_mult", "gap_s")})
    study.optimize(objective, n_trials=n_trials)
    study.trials_dataframe().to_csv(OUT / "decoder_trials.csv", index=False)
    best = dict(DEFAULT_PARAMS, **study.best_params)
    clf_train = total_top1_clf(F, P, oof)
    print(f"train OOF: classifier-only top1={clf_train:.4f}  default HMM={study.trials[0].value:.4f}  "
          f"tuned HMM={study.best_value:.4f}")
    yaml.safe_dump({"hmm": {k: float(v) for k, v in best.items()},
                    "tuned_on": "78 clean train trips, 5-fold out-of-fold LightGBM-tuned probabilities",
                    "objective": "observation-level top-1", "train_oof_top1_classifier_only": round(clf_train, 4),
                    "train_oof_top1_hmm_default": round(study.trials[0].value, 4),
                    "train_oof_top1_hmm_tuned": round(study.best_value, 4)},
                   open(REPO_ROOT / "configs" / "hmm.yaml", "w"), sort_keys=False)

    # ---------------- 2. student-test comparison ----------------
    Ft, Pt = load_partition("student_test")
    pt = np.load(REPO_ROOT / "results" / "test_prob_lightgbm_tuned.npy")
    dist_prob = np.exp(-0.5 * (Ft.dist_m.values / DIST_SIGMA) ** 2)
    L_test = lattices(Ft, "student_test", set(Pt.trip_id))
    variants = {"classifier only": (pt, None), "classifier + HMM (default decoder)": (pt, DEFAULT_PARAMS),
                "classifier + HMM (tuned decoder)": (pt, best),
                "distance-only HMM (no classifier)": (dist_prob, best)}
    D = compare(L_test, Ft, Pt, variants, "student_test")

    # train OOF and suspect trips, for the over/under-fitting and limitation discussion
    D_tr = compare(L_train, F, P, {"classifier only": (oof, None), "classifier + HMM (tuned decoder)": (oof, best)},
                   "train (OOF)")
    import joblib
    model = joblib.load(REPO_ROOT / "models" / "lightgbm_tuned.joblib")
    Fs = F_all[~clean].reset_index(drop=True); Ps = P_all[P_all.trip_id.isin(SUSPECT_TRIPS)].reset_index(drop=True)
    from src.models.common import X_of
    ps = model.predict_proba(X_of(Fs))[:, 1]
    L_s = lattices(Fs, "train", SUSPECT_TRIPS)
    D_s = compare(L_s, Fs, Ps, {"classifier only": (ps, None), "classifier + HMM (tuned decoder)": (ps, best)},
                  "train suspect (33, 54)")
    D = pd.concat([D, D_tr, D_s], ignore_index=True)
    D.to_csv(OUT / "hmm_vs_classifier_per_trip.csv", index=False)
    S = summarise(D)
    S.to_csv(OUT / "hmm_vs_classifier_summary.csv", index=False)
    print(S.round(4).to_string(index=False))
    plots(D[D.partition == "student_test"], study)


def total_top1_clf(F, P, prob):
    correct = 0
    for tid, g in P.groupby("trip_id"):
        m = (F.trip_id == tid).values
        correct += int((top1_from_scores(F[m], prob[m], len(g)) == g.label_arc.values).sum())
    return correct / len(P)


def plots(D, study):
    piv = D.pivot(index="trip_id", columns="method", values="top1")
    cols = ["classifier only", "classifier + HMM (tuned decoder)", "distance-only HMM (no classifier)"]
    ax = piv[cols].plot.bar(figsize=(10, 4), width=0.8)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("top-1 accuracy"); ax.set_title("Student-test: classifier-only vs HMM per trip")
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout(); plt.savefig(FIG / "hmm_vs_classifier_top1.png", dpi=150); plt.close()
    piv = D.pivot(index="trip_id", columns="method", values="continuity")
    ax = piv[cols].plot.bar(figsize=(10, 4), width=0.8)
    ax.set_ylim(0.8, 1.0); ax.set_ylabel("route continuity"); ax.set_title("Student-test: route continuity per trip")
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout(); plt.savefig(FIG / "hmm_vs_classifier_continuity.png", dpi=150); plt.close()
    t = study.trials_dataframe()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(t.number, t.value, s=18); ax.plot(t.number, t.value.cummax(), color="C3")
    ax.set_xlabel("trial"); ax.set_ylabel("train OOF top-1"); ax.set_title("HMM decoder tuning (Optuna TPE)")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIG / "hmm_decoder_tuning.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)

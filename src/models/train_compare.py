"""
src/models/train_compare.py  (Person 2)

Trains and compares the candidate classifiers of src/models/zoo.py.

    python -m src.models.train_compare            # all models
    python -m src.models.train_compare lightgbm   # a subset

Protocol
* Train partition only for model fitting and selection: 78 clean trips (trips 33 and 54 excluded,
  label-suspect, reported separately), GroupKFold(5) by trip_id.
* Training rows: at most 800 random points per trip (all their candidates), so that no single long
  trip dominates; CV evaluation: at most 1000 points per held-out trip, the same subset for every model.
* Student-test (10 trips): every point, used only to report the selected configurations, never to choose them.
* Primary selection metric: CV observation-level top-1 (what the decoder needs); log loss is reported as
  the calibration check because the HMM consumes the probabilities.

Outputs: results/model_comparison.csv, results/model_cv_folds.csv, results/model_per_trip_test.csv,
results/figures/models_*.png, models/<name>.joblib, models/scaler.joblib, results/logs/train_compare_log.txt
"""
from __future__ import annotations

import json
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.data.loader import REPO_ROOT, SUSPECT_TRIPS
from src.eval.metrics import candidate_metrics
from src.features.build import FEATURE_COLUMNS
from src.features.dataset import load_partition
from src.models.common import (SEED, N_FOLDS, TRAIN_POINTS_PER_TRIP, EVAL_POINTS_PER_TRIP, clean_train,
                               sample_points, rows_for_points, top1, per_trip_top1, X_of)
from src.models.zoo import MODEL_SPECS

RES = REPO_ROOT / "results"
FIG = RES / "figures"
MOD = REPO_ROOT / "models"


def subsample_rows(n_rows: int, cap, seed=SEED):
    if cap is None or n_rows <= cap:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, cap, replace=False))


def fit(name, X, y):
    _, _, factory, cap = MODEL_SPECS[name]
    idx = subsample_rows(len(X), cap)
    m = factory()
    t0 = time.perf_counter()
    m.fit(X[idx], y[idx])
    return m, time.perf_counter() - t0


def predict(m, X):
    t0 = time.perf_counter()
    p = m.predict_proba(X)[:, 1]
    return p, time.perf_counter() - t0


def merge(path, new):
    """Write `new`, replacing earlier rows of the same models (so a subset of models can be re-run)."""
    if path.exists():
        old = pd.read_csv(path, dtype={"trip_id": str})
        old = old[~old.model.isin(new.model.unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def main(names):
    FIG.mkdir(parents=True, exist_ok=True); MOD.mkdir(exist_ok=True); (RES / "logs").mkdir(exist_ok=True)
    log = open(RES / "logs" / "train_compare_log.txt", "a", encoding="utf-8")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); log.write(s + "\n"); log.flush()

    F_all, P_all = load_partition("train")
    F, P = clean_train(F_all, P_all)
    Ft, Pt = load_partition("student_test")
    Xt, yt = X_of(Ft), Ft.label.values
    say(f"\n=== run {time.strftime('%Y-%m-%d %H:%M:%S')}  seed={SEED} folds={N_FOLDS} "
        f"train_pts/trip={TRAIN_POINTS_PER_TRIP} eval_pts/trip={EVAL_POINTS_PER_TRIP}")
    say(f"clean train: {P.trip_id.nunique()} trips, {len(P):,} points, {len(F):,} rows; "
        f"student_test: {len(Pt):,} points, {len(Ft):,} rows; features={len(FEATURE_COLUMNS)}")

    P_tr = sample_points(P, TRAIN_POINTS_PER_TRIP, SEED)
    P_ev = sample_points(P, EVAL_POINTS_PER_TRIP, SEED + 1)
    m_tr, m_ev = rows_for_points(F, P_tr), rows_for_points(F, P_ev)
    trips = np.array(sorted(P.trip_id.unique()))
    folds = list(GroupKFold(n_splits=N_FOLDS).split(trips, groups=trips))
    json.dump({"seed": SEED, "folds": [[trips[v].tolist()] for _, v in folds]},
              open(RES / "cv_folds.json", "w"), indent=1)

    # train-only scaler, saved for reference / the report (models that need scaling carry their own)
    joblib.dump(StandardScaler().fit(X_of(F[m_tr])), MOD / "scaler.joblib")

    fold_rows, summary, per_trip = [], [], []
    for name in names:
        family, cfg, _, _ = MODEL_SPECS[name]
        say(f"\n--- {name} ({family}): {cfg}")
        fr = []
        for k, (tr_i, va_i) in enumerate(folds):
            tr_trips, va_trips = set(trips[tr_i]), set(trips[va_i])
            mtr = m_tr & F.trip_id.isin(tr_trips).values
            mva = m_ev & F.trip_id.isin(va_trips).values
            model, t_fit = fit(name, X_of(F[mtr]), F.label.values[mtr])
            p_va, t_pred = predict(model, X_of(F[mva]))
            # training-fit check on (at most) 2000 sampled training points
            P_fit = sample_points(P_tr[P_tr.trip_id.isin(tr_trips)], 40, SEED + 2)
            mfit = rows_for_points(F, P_fit)
            p_tr, _ = predict(model, X_of(F[mfit]))
            r = {"model": name, "fold": k, "fit_s": t_fit, "pred_s": t_pred, "n_val_rows": int(mva.sum()),
                 **candidate_metrics(F.label.values[mva], p_va),
                 **top1(F[mva], p_va, P_ev[P_ev.trip_id.isin(va_trips)]),
                 "train_top1": top1(F[mfit], p_tr, P_fit)["top1"],
                 "train_macro_f1": candidate_metrics(F.label.values[mfit], p_tr)["macro_f1"]}
            fr.append(r)
            say(f"  fold {k}: top1={r['top1']:.4f} macroF1={r['macro_f1']:.4f} logloss={r['log_loss']:.4f} "
                f"(train top1={r['train_top1']:.4f}) fit={t_fit:.1f}s")
        fold_rows += fr
        cv = pd.DataFrame(fr)

        # final model: all clean train trips (sampled rows), evaluated on every student-test point
        model, t_fit = fit(name, X_of(F[m_tr]), F.label.values[m_tr])
        p_t, t_pred = predict(model, Xt)
        joblib.dump(model, MOD / f"{name}.joblib", compress=3)
        np.save(RES / f"test_prob_{name}.npy", p_t.astype(np.float32))
        tm = candidate_metrics(yt, p_t)
        t1 = top1(Ft, p_t, Pt)
        # label-suspect trips, reported separately (never used for fitting or selection)
        ms = F_all.trip_id.isin(SUSPECT_TRIPS).values
        p_s, _ = predict(model, X_of(F_all[ms]))
        sus = top1(F_all[ms], p_s, P_all[P_all.trip_id.isin(SUSPECT_TRIPS)])
        cm = confusion_matrix(yt, (p_t >= 0.5).astype(int), labels=[0, 1])
        summary.append({
            "model": name, "family": family, "config": cfg,
            "cv_top1_mean": cv.top1.mean(), "cv_top1_std": cv.top1.std(),
            "cv_macro_f1_mean": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
            "cv_log_loss_mean": cv.log_loss.mean(), "cv_roc_auc_mean": cv.roc_auc.mean(),
            "cv_train_top1_mean": cv.train_top1.mean(), "cv_train_macro_f1_mean": cv.train_macro_f1.mean(),
            "test_top1": t1["top1"], "test_top1_given_in_cands": t1["top1_given_in_cands"],
            "test_macro_f1": tm["macro_f1"], "test_precision": tm["precision"], "test_recall": tm["recall"],
            "test_log_loss": tm["log_loss"], "test_roc_auc": tm["roc_auc"],
            "suspect_trips_top1": sus["top1"],
            "train_time_s": t_fit, "test_inference_s": t_pred,
            "inference_ms_per_1k_rows": 1000 * t_pred / len(Xt) * 1000,
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        })
        pt = per_trip_top1(Ft, p_t, Pt); pt.insert(0, "model", name); per_trip.append(pt)
        say(f"  CV top1={cv.top1.mean():.4f}+-{cv.top1.std():.4f} macroF1={cv.macro_f1.mean():.4f} "
            f"logloss={cv.log_loss.mean():.4f} | TEST top1={t1['top1']:.4f} macroF1={tm['macro_f1']:.4f} "
            f"logloss={tm['log_loss']:.4f} | suspect top1={sus['top1']:.4f} | fit={t_fit:.1f}s pred={t_pred:.2f}s")
        # save this model's rows now (a crash / stop later does not lose finished models)
        merge(RES / "model_comparison.csv", pd.DataFrame(summary[-1:]))
        merge(RES / "model_cv_folds.csv", cv)
        merge(RES / "model_per_trip_test.csv", pt)
        if hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf"), "loss_curve_"):
            clf = model.named_steps["clf"]
            pd.DataFrame({"epoch": np.arange(1, len(clf.loss_curve_) + 1), "train_loss": clf.loss_curve_,
                          "val_score": clf.validation_scores_}).to_csv(RES / f"loss_curve_{name}.csv", index=False)

    S = pd.read_csv(RES / "model_comparison.csv")
    S = S.sort_values("cv_top1_mean", ascending=False)
    say("\n" + S[["model", "cv_top1_mean", "cv_macro_f1_mean", "cv_log_loss_mean", "test_top1", "test_macro_f1",
                  "test_log_loss", "train_time_s", "inference_ms_per_1k_rows"]].round(4).to_string(index=False))
    plots(S)


def plots(S):
    S = S.sort_values("cv_top1_mean", ascending=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    y = np.arange(len(S))
    ax[0].barh(y - 0.2, S.cv_top1_mean, 0.4, xerr=S.cv_top1_std, label="CV (train, 5 folds)", color="#4C72B0")
    ax[0].barh(y + 0.2, S.test_top1, 0.4, label="student-test", color="#DD8452")
    ax[0].set_title("Observation-level top-1 accuracy"); ax[0].set_xlim(0.5, 1)
    ax[1].barh(y - 0.2, S.cv_macro_f1_mean, 0.4, xerr=S.cv_macro_f1_std, color="#4C72B0")
    ax[1].barh(y + 0.2, S.test_macro_f1, 0.4, color="#DD8452")
    ax[1].set_title("Candidate macro F1 (threshold 0.5)"); ax[1].set_xlim(0.5, 1)
    ax[2].barh(y - 0.2, S.cv_log_loss_mean, 0.4, color="#4C72B0")
    ax[2].barh(y + 0.2, S.test_log_loss, 0.4, color="#DD8452")
    ax[2].set_title("Log loss (lower is better)"); ax[2].set_xscale("log")
    for a in ax:
        a.set_yticks(y); a.set_yticklabels(S.model); a.grid(axis="x", alpha=.3)
    ax[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "models_comparison.png", dpi=150); plt.close(fig)

    # confusion matrices (student-test, candidate level, threshold 0.5)
    n = len(S)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow))
    for a in np.atleast_1d(ax).ravel()[n:]:
        a.axis("off")
    for a, (_, r) in zip(np.atleast_1d(ax).ravel(), S.iterrows()):
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        a.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8)
        a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["pred 0", "pred 1"])
        a.set_yticklabels(["true 0", "true 1"]); a.set_title(r.model, fontsize=9)
    fig.suptitle("Student-test confusion matrices (row-normalised colour, counts shown)")
    fig.tight_layout(); fig.savefig(FIG / "models_confusion_matrices.png", dpi=150); plt.close(fig)

    # train vs CV vs test (over/underfitting check)
    fig, a = plt.subplots(figsize=(8, 4))
    x = np.arange(len(S))
    a.bar(x - 0.27, S.cv_train_top1_mean, 0.27, label="train (fit rows)")
    a.bar(x, S.cv_top1_mean, 0.27, label="CV held-out trips")
    a.bar(x + 0.27, S.test_top1, 0.27, label="student-test")
    a.set_xticks(x); a.set_xticklabels(S.model, rotation=20); a.set_ylim(0.5, 1.0); a.legend(fontsize=8)
    a.set_title("Top-1 accuracy: train vs CV vs student-test"); a.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "models_train_cv_test.png", dpi=150); plt.close(fig)

    lc = RES / "loss_curve_mlp.csv"
    if lc.exists():
        d = pd.read_csv(lc)
        fig, a = plt.subplots(figsize=(6, 4))
        a.plot(d.epoch, d.train_loss, label="training log loss"); a.set_xlabel("epoch"); a.set_ylabel("log loss")
        b = a.twinx(); b.plot(d.epoch, d.val_score, color="C1", label="validation accuracy (10% hold-out)")
        b.set_ylabel("validation accuracy"); a.set_title("MLP training history")
        fig.legend(loc="center right", bbox_to_anchor=(0.86, 0.55), fontsize=8); fig.tight_layout(); fig.savefig(FIG / "models_mlp_loss_curve.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODEL_SPECS))

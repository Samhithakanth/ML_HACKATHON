"""
src/tuning/bayes_opt.py  (Person 3)

Bayesian hyperparameter optimisation (Optuna, TPE sampler) of the model selected in Section 3.

    python -m src.tuning.bayes_opt [n_trials]        # default 30 trials

* Objective: mean group-CV log loss on 3 trip folds (GroupKFold(3) over the same 78 clean train trips,
  same per-trip training / evaluation samples as src/models/train_compare.py; 3 folds keep the search
  affordable). The final untuned vs tuned comparison re-evaluates both on the SAME 5 folds as Section 3. Log loss is used
  because the HMM consumes the probabilities, so calibration matters, and it is a smooth proper
  scoring rule; top-1 and macro F1 are logged for every trial as well.
* A median pruner stops clearly bad trials after their first folds.
* Student-test is only used to REPORT untuned vs tuned, never inside the objective.
* After the search the best configuration is refit on all clean train trips -> models/<name>_tuned.joblib,
  and 5-fold out-of-fold probabilities for EVERY clean train candidate row are saved
  (data/processed/train_oof_prob_tuned.npy) so the HMM decoder can be tuned on train without leakage.

Outputs: results/tuning/optuna_trials.csv (full trial log), results/tuning/best_params.json,
results/tuning/untuned_vs_tuned.csv, results/figures/tuning_history.png, results/figures/tuning_importance.png
"""
from __future__ import annotations

import json
import sys
import time

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data.loader import REPO_ROOT
from src.eval.metrics import candidate_metrics
from src.features.dataset import load_partition, PROC_DIR
from src.models.common import (SEED, N_FOLDS, TRAIN_POINTS_PER_TRIP, EVAL_POINTS_PER_TRIP, clean_train,
                               sample_points, rows_for_points, top1, X_of)

MODEL_NAME = "lightgbm"
SEARCH_FOLDS = 3
OUT = REPO_ROOT / "results" / "tuning"
FIG = REPO_ROOT / "results" / "figures"
MOD = REPO_ROOT / "models"


def search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10, 12]),   # -1 = unlimited
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }


def make(params):
    return lgb.LGBMClassifier(**params, random_state=SEED, n_jobs=-1, verbose=-1)


def main(n_trials=30):
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    F, P = clean_train(*load_partition("train"))
    Ft, Pt = load_partition("student_test")
    P_tr = sample_points(P, TRAIN_POINTS_PER_TRIP, SEED)
    P_ev = sample_points(P, EVAL_POINTS_PER_TRIP, SEED + 1)
    m_tr, m_ev = rows_for_points(F, P_tr), rows_for_points(F, P_ev)
    trips = np.array(sorted(P.trip_id.unique()))
    folds = list(GroupKFold(n_splits=N_FOLDS).split(trips, groups=trips))
    X, y = X_of(F), F.label.values

    def masks(fs):
        out = []
        for tr_i, va_i in fs:
            tr_t, va_t = set(trips[tr_i]), set(trips[va_i])
            out.append((m_tr & F.trip_id.isin(tr_t).values, m_ev & F.trip_id.isin(va_t).values, va_t))
        return out
    fold_masks = masks(folds)                                                      # 5 folds: final comparison
    search_masks = masks(GroupKFold(n_splits=SEARCH_FOLDS).split(trips, groups=trips))  # 3 folds: search

    def cv_eval(params, trial=None, fm=None):
        rows = []
        for k, (mtr, mva, va_t) in enumerate(fm or fold_masks):
            m = make(params).fit(X[mtr], y[mtr])
            p = m.predict_proba(X[mva])[:, 1]
            rows.append({**candidate_metrics(y[mva], p), **top1(F[mva], p, P_ev[P_ev.trip_id.isin(va_t)])})
            if trial is not None:
                trial.report(float(np.mean([r["log_loss"] for r in rows])), k)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return pd.DataFrame(rows).mean(numeric_only=True)

    def objective(trial):
        t0 = time.perf_counter()
        r = cv_eval(search_space(trial), trial, search_masks)
        for c in ("top1", "macro_f1", "roc_auc"):
            trial.set_user_attr(c, float(r[c]))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        print(f"trial {trial.number:3d}: logloss={r.log_loss:.5f} top1={r.top1:.4f} macroF1={r.macro_f1:.4f}", flush=True)
        return float(r.log_loss)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", study_name="lightgbm_candidate_classifier",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    # trial 0 = the untuned LightGBM defaults (reg 1e-4 ~ 0), so the search starts from the baseline
    study.enqueue_trial({"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "max_depth": -1,
                         "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                         "reg_alpha": 1e-4, "reg_lambda": 1e-4})
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    print(f"search finished in {time.time() - t0:.0f} s")
    trials = study.trials_dataframe()
    trials.to_csv(OUT / "optuna_trials.csv", index=False)
    best = dict(study.best_params, subsample_freq=1)
    json.dump({"best_params": best, "best_cv_log_loss": study.best_value, "n_trials": len(study.trials),
               "seed": SEED, "objective": "mean 3-fold group-CV log loss (train, clean trips)"}, open(OUT / "best_params.json", "w"), indent=2)

    # ---------------- untuned vs tuned ----------------
    rows = []
    for label, params in [("untuned (LightGBM defaults)", {}), ("tuned (Optuna TPE)", best)]:
        cv = cv_eval(params)
        t_fit0 = time.perf_counter(); m = make(params).fit(X[m_tr], y[m_tr]); t_fit = time.perf_counter() - t_fit0
        t_p0 = time.perf_counter(); pt = m.predict_proba(X_of(Ft))[:, 1]; t_pred = time.perf_counter() - t_p0
        tm, t1 = candidate_metrics(Ft.label.values, pt), top1(Ft, pt, Pt)
        rows.append({"configuration": label, "cv_log_loss": cv.log_loss, "cv_top1": cv.top1, "cv_macro_f1": cv.macro_f1,
                     "test_top1": t1["top1"], "test_macro_f1": tm["macro_f1"], "test_log_loss": tm["log_loss"],
                     "test_precision": tm["precision"], "test_recall": tm["recall"],
                     "train_time_s": t_fit, "test_inference_s": t_pred,
                     "inference_ms_per_1k_rows": 1e6 * t_pred / len(Ft)})
        if params:
            joblib.dump(m, MOD / f"{MODEL_NAME}_tuned.joblib", compress=3)
            np.save(REPO_ROOT / "results" / f"test_prob_{MODEL_NAME}_tuned.npy", pt.astype(np.float32))
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(OUT / "untuned_vs_tuned.csv", index=False)
    print(cmp_.round(4).to_string(index=False))

    # ---------------- out-of-fold probabilities on every clean train row (for HMM tuning) ----------------
    oof = np.full(len(F), np.nan, dtype=np.float32)
    for tr_i, va_i in folds:
        mtr = m_tr & F.trip_id.isin(set(trips[tr_i])).values
        mva = F.trip_id.isin(set(trips[va_i])).values
        oof[mva] = make(best).fit(X[mtr], y[mtr]).predict_proba(X[mva])[:, 1]
    np.save(PROC_DIR / "train_oof_prob_tuned.npy", oof)
    print("OOF top1 (all clean train points):", round(top1(F, oof, P)["top1"], 4))
    plots(study)


def plots(study):
    t = study.trials_dataframe()
    t = t[t.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(t.number, t.value, s=18, label="trial (CV log loss)")
    ax.plot(t.number, t.value.cummin(), color="C3", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("5-fold group-CV log loss"); ax.set_title("Optuna TPE optimisation history")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIG / "tuning_history.png", dpi=150); plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(6, 4))
        k = list(imp)[::-1]
        ax.barh(k, [imp[i] for i in k]); ax.set_title("Hyperparameter importance (fANOVA)")
        fig.tight_layout(); fig.savefig(FIG / "tuning_importance.png", dpi=150); plt.close(fig)
        pd.Series(imp, name="importance").to_csv(OUT / "param_importance.csv")
    except Exception as e:  # importance needs >1 completed trial with variation
        print("importance plot skipped:", e)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)

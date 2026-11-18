"""
src/eval/final_eval.py

ONE-SHOT final evaluation on the evaluator_test partition with the FROZEN pipeline
(models/lightgbm_tuned.joblib + configs/hmm.yaml). Nothing is tuned or selected here: the model, the features, the
candidate settings and the decoder parameters were all fixed on train / student_test before this partition was read.

    python -m src.eval.final_eval [path/to/folder_containing_evaluator_test]

Uses exactly the API's prediction code (api.pipeline.predict_trip). Reference labels come from the same monotone-DP
alignment of the GPS track to .route that was used for train and student_test (src.data.labels).

Outputs: results/final/evaluator_test_per_trip.csv, results/final/evaluator_test_summary.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from api.pipeline import predict_trip
from src.data.candidates import candidate_table
from src.data.labels import align_trip_to_route
from src.data.loader import REPO_ROOT, list_trips, load_trip
from src.eval.metrics import candidate_metrics, compute_macro_f1, sequence_metrics
from src.features.build import FEATURE_COLUMNS, build_features
from src.models.predictor import PathSnapPredictor

PARTITION = "evaluator_test"
OUT = REPO_ROOT / "results" / "final"


def main(data_dir: Path):
    OUT.mkdir(parents=True, exist_ok=True)
    predictor = PathSnapPredictor(model_path=REPO_ROOT / "models" / "lightgbm_tuned.joblib")
    assert predictor.is_loaded, "models/lightgbm_tuned.joblib not found"
    hmm_cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "hmm.yaml"))["hmm"]

    rows, Y, PR = [], [], []
    for tid in list_trips(PARTITION, data_dir):
        trip = load_trip(tid, PARTITION, raw_dir=data_dir, load_route=True)
        _, ref, ref_dist = align_trip_to_route(trip)
        cands = candidate_table(trip)
        hit = cands.assign(ok=(cands.arc_id.values == ref[cands.point_idx.values])).groupby("point_idx").ok.max()
        miss = 1.0 - hit.reindex(range(len(ref)), fill_value=False).mean()

        feats = build_features(trip, cands)       # candidate-level scores for macro F1 / log loss (same definition as Section 3)
        PR.append(predictor.predict_candidate_scores(feats[FEATURE_COLUMNS].values.astype(np.float32)))
        Y.append((cands.arc_id.values == ref[cands.point_idx.values]).astype(int))
        t0 = time.perf_counter()
        res = predict_trip(tid, data_dir, partition=PARTITION, predictor=predictor, hmm_config=hmm_cfg)
        wall = time.perf_counter() - t0
        arcs = trip.arcs[["from_node", "to_node"]].values
        for method, pred in (("classifier only", np.asarray(res["classifier_arcs"])),
                             ("classifier + HMM", np.asarray(res["predicted_arcs"]))):
            rows.append({"trip_id": tid, "method": method, "n_points": len(ref),
                         "macro_f1": compute_macro_f1(pred, ref), "candidate_miss_rate": miss,
                         "label_median_dist_m": float(np.median(ref_dist)), "total_ms": res["timings_ms"]["total"],
                         "wall_s": wall, **sequence_metrics(pred, ref, arcs)})
        print(f"{tid}: n={len(ref):5d}  clf={rows[-2]['top1']:.4f}  hmm={rows[-1]['top1']:.4f}  "
              f"cont={rows[-1]['continuity']:.4f}  miss={miss:.4f}  {res['timings_ms']['total']} ms", flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(OUT / "evaluator_test_per_trip.csv", index=False)

    def agg(g):
        w = g.n_points
        return pd.Series({
            "n_trips": g.trip_id.nunique(), "n_points": int(w.sum()),
            "top1 (point-weighted)": np.average(g.top1, weights=w), "top1 (trip mean)": g.top1.mean(),
            "top1 (worst trip)": g.top1.min(), "macro_f1 (point-weighted)": np.average(g.macro_f1, weights=w),
            "continuity": np.average(g.continuity, weights=w), "breaks / 1k points": 1000 * g.n_breaks.sum() / w.sum(),
            "edit_distance (trip mean)": g.edit_distance.mean(), "route_precision": g.route_precision.mean(),
            "route_recall": g.route_recall.mean(), "route_f1": g.route_f1.mean(),
            "candidate_miss_rate (point-weighted)": np.average(g.candidate_miss_rate, weights=w),
            "inference_ms / trip (mean)": g.total_ms.mean(), "inference_ms / 1k points": 1000 * g.total_ms.sum() / w.sum()})

    S = D.groupby("method", sort=False).apply(agg, include_groups=False).reset_index()
    S.to_csv(OUT / "evaluator_test_summary.csv", index=False)
    print("\n" + S.round(4).T.to_string(header=False))

    # candidate-level metrics (same definition as the model-comparison table: binary "is this candidate the reference arc")
    C = candidate_metrics(np.concatenate(Y), np.concatenate(PR))
    C.update({"n_candidate_rows": int(sum(len(y) for y in Y)), "positive_rate": float(np.concatenate(Y).mean())})
    pd.DataFrame([C]).to_csv(OUT / "evaluator_test_candidate_level.csv", index=False)
    print("\ncandidate-level (tuned LightGBM, threshold 0.5):", {k: round(v, 4) for k, v in C.items()})


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT.parent / "eval_data")

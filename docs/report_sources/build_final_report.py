"""
docs/report_sources/build_final_report.py

Assembles the single consolidated technical report (Markdown -> HTML -> PDF) in the structure required by
Deliverables.pdf (sections 2-7), plus a final held-out evaluation (8) and the integrity statement (9).
Every table is read from the CSV files in results/; prose is taken from the team's section files, with the
corrections listed in the CORRECTIONS block below applied (each replacement is asserted so nothing fails silently).

    python docs/report_sources/build_final_report.py   # -> docs/report_sources/FINAL_REPORT.md and report/FINAL_REPORT.pdf (PDF needs Microsoft Edge)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
REP = Path(__file__).resolve().parent        # sources: docs/report_sources (section drafts, screenshots, FINAL_REPORT.md)
OUT_PDF = ROOT / "report" / "FINAL_REPORT.pdf"   # the only file kept in report/


def rd(p, **kw):
    return pd.read_csv(R / p, **kw)


def fmt(x, n=3):
    return "N/A" if pd.isna(x) else f"{x:.{n}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def sub(text, old, new, count=1):
    assert old in text, f"CORRECTION TARGET NOT FOUND: {old[:70]!r}"
    return text.replace(old, new, count)


# ------------------------------------------------------------------ tables from CSVs
def table_3_1():
    """Deliverables 3.1 columns: Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy |
    Route continuity | Student-Test Macro F1-score | Log Loss | Inference Time  (classifier only); HMM table follows."""
    t = rd("model_comparison_table.csv")
    rf = rd("model_route_metrics.csv").set_index("model").loc["random_forest"]
    rows, hrows = [], []
    for _, r in t.iterrows():
        rows.append([r["Model"], r["Algorithm family"], r["Key configuration"], r["CV macro F1 (5-fold, train)"],
                     fmt(r["Student-test top-1 (classifier only)"]), fmt(r["Route continuity (classifier only)"]),
                     fmt(r["Student-test macro F1"]), fmt(r["Student-test log loss"]),
                     fmt(r["Inference time (ms / 1k candidates)"], 2) + " ms / 1k candidates"])
        hrows.append([r["Model"], fmt(r["Student-test top-1 (classifier only)"]), fmt(r["Student-test top-1 (+HMM)"]),
                      fmt(r["Route continuity (classifier only)"]), fmt(r["Route continuity (+HMM)"])])
        if r["Model"].startswith("K-Nearest"):          # Random Forest row: only what is verifiable from committed results
            rows.append(["Random Forest", "Bagging ensemble", "300 trees, min_samples_leaf=5, max_features=sqrt", "N/A (1)",
                         fmt(rf.clf_top1), fmt(rf.clf_continuity), "N/A (1)", "N/A (1)", "N/A (1)"])
            hrows.append(["Random Forest", fmt(rf.clf_top1), fmt(rf.hmm_top1), fmt(rf.clf_continuity), fmt(rf.hmm_continuity)])
    h = ["Model", "Algorithm Family", "Key Configuration", "CV Macro F1-score", "Top-1 Accuracy", "Route continuity",
         "Student-Test Macro F1-score", "Log Loss", "Inference Time"]
    return (md_table(h, rows) + "\n\n**Table 3.1b. The same models followed by HMM / Viterbi decoding (student-test).**\n\n" +
            md_table(["Model", "Top-1 (classifier only)", "Top-1 (+ HMM)", "Route continuity (classifier only)", "Route continuity (+ HMM)"], hrows) +
            "\n\nTop-1 Accuracy = observation-level top-1 on the 10 student-test trips (19,160 points); Route continuity = share of consecutive predicted arcs that are the same or directly connected; "
            "CV = 5-fold GroupKFold by trip on the 78 clean train trips (mean ± std over folds). All figures are measured; none is estimated.\n\n"
            "(1) N/A: the cross-validation, macro-F1, log-loss and timing rows of the Random Forest are missing from the committed "
            "`results/model_comparison.csv` (the file was rewritten by a later partial run). Only its route metrics survive "
            "(`results/model_route_metrics.csv`). They are regenerated with `python -m src.models.train_compare random_forest`; "
            "no unverified figure is quoted here.")


def table_tuning():
    t = rd("tuning/untuned_vs_tuned.csv").set_index("configuration")
    cols = [("cv_top1", "CV top-1"), ("cv_macro_f1", "CV macro F1"), ("cv_log_loss", "CV log loss"), ("test_top1", "Student-test top-1"),
            ("test_macro_f1", "Student-test macro F1"), ("test_log_loss", "Student-test log loss"),
            ("inference_ms_per_1k_rows", "Inference (ms / 1k rows)"), ("train_time_s", "Fit time (s)")]
    rows = [[k] + [fmt(v[c], 3 if c not in ("inference_ms_per_1k_rows", "train_time_s") else 1) for c, _ in cols] for k, v in t.iterrows()]
    return md_table(["Configuration"] + [n for _, n in cols], rows)


def importance_sentence():
    p = rd("tuning/param_importance.csv").rename(columns={"Unnamed: 0": "param"}).sort_values("importance", ascending=False)
    return ", ".join(f"{a} {b:.2f}" for a, b in zip(p.param, p.importance)) + "."


# ------------------------------------------------------------------ final evaluation (evaluator_test)
def final_eval_section():
    per = rd("final/evaluator_test_per_trip.csv", dtype={"trip_id": str})
    S = rd("final/evaluator_test_summary.csv").set_index("method")
    C = rd("final/evaluator_test_candidate_level.csv").iloc[0]
    chk = rd("final/evaluator_test_api_check.csv", dtype={"trip": str})
    st = rd("hmm/hmm_vs_classifier_summary.csv")
    st = st[st.partition == "student_test"].set_index("method")
    tun = rd("tuning/untuned_vs_tuned.csv").set_index("configuration").loc["tuned (Optuna TPE)"]
    clf, hmm = S.loc["classifier only"], S.loc["classifier + HMM"]
    sc, sh = st.loc["classifier only"], st.loc["classifier + HMM (tuned decoder)"]

    # figure: per-trip top-1
    piv = per.pivot(index="trip_id", columns="method", values="top1")[["classifier only", "classifier + HMM"]]
    ax = piv.plot.bar(figsize=(9, 3.8), width=0.8, color=["#d95f02", "#1b9e77"])
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("observation-level top-1"); ax.set_xlabel("evaluator_test trip")
    ax.set_title("Final evaluation: classifier-only vs classifier + HMM (evaluator_test, 10 trips)")
    ax.grid(axis="y", alpha=.3); ax.legend(loc="lower right", fontsize=8); plt.tight_layout()
    plt.savefig(R / "figures" / "final_evaluator_test_top1.png", dpi=150); plt.close()

    p_c = per[per.method == "classifier only"].set_index("trip_id")
    p_h = per[per.method == "classifier + HMM"].set_index("trip_id")
    rows = [[t, int(p_h.n_points[t]), fmt(p_c.top1[t], 4), fmt(p_h.top1[t], 4), f"{100*(p_h.top1[t]-p_c.top1[t]):+.1f}",
             fmt(p_h.continuity[t], 4), int(p_h.n_breaks[t]), fmt(p_h.route_f1[t], 3), fmt(p_h.total_ms[t], 0)] for t in p_h.index]
    trip_tbl = md_table(["Trip", "GPS points", "Top-1 classifier", "Top-1 + HMM", "Gain (pp)", "Continuity + HMM",
                         "Breaks + HMM", "Route F1 + HMM", "API time (ms)"], rows)

    n = p_c.n_points; w15 = "00000015"
    rest = [t for t in p_c.index if t != w15]
    clf_rest = np.average(p_c.top1[rest], weights=n[rest]); hmm_rest = np.average(p_h.top1[rest], weights=n[rest])

    cmp_tbl = md_table(["Metric", "student_test (development)", "evaluator_test (final)"], [
        ["Classifier-only top-1 (point-weighted)", fmt(sc["top1 (point-weighted)"], 4), fmt(clf["top1 (point-weighted)"], 4)],
        ["Classifier + HMM top-1 (point-weighted)", fmt(sh["top1 (point-weighted)"], 4), fmt(hmm["top1 (point-weighted)"], 4)],
        ["Classifier + HMM top-1 (trip mean)", fmt(sh["top1 (trip mean)"], 4), fmt(hmm["top1 (trip mean)"], 4)],
        ["Route continuity: classifier only / + HMM", f"{sc.continuity:.4f} / {sh.continuity:.4f}", f"{clf.continuity:.4f} / {hmm.continuity:.4f}"],
        ["Breaks per 1,000 points: classifier only / + HMM", f"{sc['breaks / 1k points']:.1f} / {sh['breaks / 1k points']:.1f}", f"{clf['breaks / 1k points']:.1f} / {hmm['breaks / 1k points']:.1f}"],
        ["Normalised edit distance (trip mean): classifier only / + HMM", f"{sc['edit_distance (trip mean)']:.3f} / {sh['edit_distance (trip mean)']:.3f}", f"{clf['edit_distance (trip mean)']:.3f} / {hmm['edit_distance (trip mean)']:.3f}"],
        ["Route precision / recall / F1 (+ HMM)", f"{sh.route_precision:.3f} / {sh.route_recall:.3f} / {sh.route_f1:.3f}", f"{hmm.route_precision:.3f} / {hmm.route_recall:.3f} / {hmm.route_f1:.3f}"],
        ["Candidate-level macro F1 (tuned LightGBM, threshold 0.5)", fmt(tun.test_macro_f1, 3), fmt(C.macro_f1, 3)],
        ["Candidate-level log loss", fmt(tun.test_log_loss, 3), fmt(C.log_loss, 3)],
        ["Candidate miss rate", "0.0009", fmt(hmm["candidate_miss_rate (point-weighted)"], 4)],
    ])
    maxdiff = max((chk.api_hmm - chk.off_hmm).abs().max(), (chk.api_clf - chk.off_clf).abs().max())

    return f"""## Appendix A. Final Held-Out Evaluation (evaluator_test)

**Protocol.** The evaluator_test partition (10 trips, {int(hmm.n_points):,} GPS points, {int(C.n_candidate_rows):,} candidate rows) was never read by any
training, feature, tuning or decoder script. The pipeline was **frozen** before this run: tuned LightGBM (`models/lightgbm_tuned.joblib`),
31 features, candidates R = 100 m / K = 30, HMM decoder parameters from `configs/hmm.yaml` (tuned on train out-of-fold predictions only).
Nothing was changed after seeing these results. The evaluation script (`python -m src.eval.final_eval`) was executed three times, only to add
output columns and the candidate-level metrics, and a fourth time through the live API (below); all runs gave identical per-trip scores.
Reference labels come from the same monotone-DP alignment of GPS to `.route` as for train and student_test; their quality is the same
(median GPS-to-reference distance 4.1-5.7 m per trip, maximum 61 m; the reference arc is among the candidates for 100 % of points).
Source: `results/final/*.csv`; code: `src/eval/final_eval.py`.

### A.1 Headline result

| | Top-1 accuracy (point-weighted) | Route continuity | Route F1 |
|---|---|---|---|
| Classifier only | {clf["top1 (point-weighted)"]:.4f} | {clf.continuity:.4f} | {clf.route_f1:.3f} |
| **Classifier + HMM (final system)** | **{hmm["top1 (point-weighted)"]:.4f}** | **{hmm.continuity:.4f}** | **{hmm.route_f1:.3f}** |

The final system assigns the correct reference road segment to **{100*hmm["top1 (point-weighted)"]:.1f} % of the {int(hmm.n_points):,} GPS observations**
(trip mean {100*hmm["top1 (trip mean)"]:.1f} %, worst trip {100*hmm["top1 (worst trip)"]:.1f} %, best trip 100 %). Decoding raised top-1 by
{100*(hmm["top1 (point-weighted)"]-clf["top1 (point-weighted)"]):.1f} percentage points and cut route breaks from {clf["breaks / 1k points"]:.1f} to
{hmm["breaks / 1k points"]:.1f} per 1,000 points.

![Per-trip top-1, evaluator_test](../results/figures/final_evaluator_test_top1.png)

### A.2 Per-trip results (classifier + HMM)

{trip_tbl}

### A.3 Comparison with the development set

{cmp_tbl}

### A.4 Interpretation

* **The final system generalised.** HMM top-1 is {100*hmm["top1 (point-weighted)"]:.1f} % on the unseen evaluator set against {100*sh["top1 (point-weighted)"]:.1f} % on student_test,
  a difference of {100*(sh["top1 (point-weighted)"]-hmm["top1 (point-weighted)"]):.1f} percentage points, in the same range as the fold-to-fold spread of the models (about 3.5 pp in cross-validation). The decoder
  parameters were tuned on train only, so there is no tuning on this set to inflate the number.
* **The classifier alone is weaker here, because of one trip.** Classifier-only top-1 is {100*clf["top1 (point-weighted)"]:.1f} % (student_test: {100*sc["top1 (point-weighted)"]:.1f} %).
  Trip {w15} has 6,854 points (35 % of the set) and the classifier gets only {100*p_c.top1[w15]:.1f} % of them right. Without it the classifier-only top-1 is
  {100*clf_rest:.1f} % and the HMM top-1 {100*hmm_rest:.1f} % on the other nine trips. For this trip, **32 % of all points are the reverse twin** of the correct street (same geometry,
  opposite direction), 6 % an adjacent arc and 1.6 % another arc; the HMM brings the reverse-twin share down to 3 %, and top-1 from {100*p_c.top1[w15]:.1f} % to {100*p_h.top1[w15]:.1f} %.
  This is the failure mode identified in Section 5.6 (single-point classification cannot separate a street from its opposite direction), appearing here in extreme form,
  and it is the clearest demonstration of why the HMM stage is needed.
* **The weakest trip is 00000030** ({100*p_h.top1["00000030"]:.1f} % with the HMM, {100*p_c.top1["00000030"]:.1f} % without; 654 points, median speed 16 m/s). After decoding, 6.7 % of its points are still the reverse twin,
  3.2 % an adjacent arc and 4.1 % another arc. We did not investigate this trip further, so we do not claim a cause.
* **Candidate-level quality is lower than on student_test** (macro F1 {C.macro_f1:.3f} vs {tun.test_macro_f1:.3f}; log loss {C.log_loss:.3f} vs {tun.test_log_loss:.3f}), consistent with trip 00000015 being a hard trip for
  the classifier. Candidate generation was not a limiting factor: the reference arc was among the candidates for every one of the {int(hmm.n_points):,} points.
* **Route continuity** is {hmm.continuity:.3f} (about {hmm["breaks / 1k points"]:.0f} breaks per 1,000 points). As in Section 5.6, remaining breaks are expected to be pairs of arcs joined by a short directed path (the vehicle crossing short arcs within one GPS interval); we
  did not repeat the break analysis on this set. Trips 00000017, 00000030 and 00000059 have the lowest continuity (0.88, 0.91, 0.93).
* **Limitations.** Reference labels are derived from the reference route, which is not independent ground truth (Section 2); ten trips are a small sample; the per-trip numbers vary from 86 % to 100 %.

### A.5 End-to-end check through the API

All 10 evaluator trips were also run through the deployed pipeline: the API was started with `DATA_DIR`/`EVAL_PARTITION` pointing at the evaluator data and every trip
was sent through `POST /predict`. All 10 requests returned HTTP 200 (1-4 s each; 0.5 s for the 458-point trip, 4.2 s for the 6,854-point trip), the API's server-side top-1 equals the offline
evaluation to the fourth decimal on every trip (maximum difference {maxdiff:.1f}), and no response contained reference labels or the reference route.
Evidence: `results/final/evaluator_test_api_check.csv` and the request log lines in Section 7.4.

### A.6 Maps of selected evaluator trips

Interactive maps: `results/maps/evaluator_test_<trip>.html` (code: `src/viz/make_final_maps.py`). Layers: road network, raw GPS, reference route, classifier-only route,
HMM route, per-point correctness.

![Evaluator trip 00000015: largest classifier to HMM gain](../results/maps/screenshots/evaluator_test_00000015.png)
![Evaluator trip 00000030: weakest trip](../results/maps/screenshots/evaluator_test_00000030.png)
![Evaluator trip 00000049: 100 % correct](../results/maps/screenshots/evaluator_test_00000049.png)
"""


# ------------------------------------------------------------------ corrected section 7
OVERVIEW_2_1 = """## 2. Problem Context, Dataset and Exploratory Data Analysis

### 2.1 Problem and approach in brief
A vehicle's GPS track is noisy (parallel-street drift, jumps, dropouts), so the road segments actually driven must be recovered by matching the track to a road network. PathSnap does this in three stages:
(i) **candidate generation**: for every GPS point, all arcs of the road network within 100 m (at most the 30 nearest); (ii) a **classifier** that scores each (point, candidate) pair; (iii) **HMM / Viterbi decoding** over the directed road graph so that the chosen arcs form a connected route.
A FastAPI backend (Laptop 1) runs the pipeline and an evaluator UI (Laptop 2) displays the result. The data come as 80 training trips, 10 student-test trips and 10 evaluator trips (Appendix A); the analysis below uses the training and student-test partitions only.

"""

SECTION_7 = """## 7. API, UI and Deployment

### 7.1 Two-laptop architecture
PathSnap runs on two machines over TCP/IP port 8000:

1. **Laptop 1: model host and backend.** FastAPI 0.162.0 under Uvicorn 0.54.0, bound to `0.0.0.0:8000` (`api/main.py`).
   At start-up it loads the tuned LightGBM (`models/lightgbm_tuned.joblib`), the HMM parameters (`configs/hmm.yaml`), the road networks of the evaluation trips, and builds the **private
   server-side reference labels** (monotone-DP alignment of the GPS track to the trip's `.route`, kept in server memory only).
   The prediction function (`api/pipeline.py`, `predict_trip`) loads each trip with `load_route=False`: it reads only `.track`, `.nodes` and `.arcs`.
   The pipeline is: local metric projection per trip, candidate generation (all arcs within 100 m, at most the 30 nearest), 31 features per
   (point, candidate), tuned-LightGBM probabilities, then HMM / Viterbi decoding. The reference labels are used only inside the scoring step of `POST /predict`;
   reference geometry and labels are never returned. Every request is logged (client IP, endpoint, status, latency) to stdout and `logs/api.log`.
2. **Laptop 2: evaluator UI.** A pure client with no model, no data and no labels: either the Streamlit app (`ui/app.py`, Folium map) or the web page served by the API at `/web/` (`ui/index.html`, Leaflet).
   Flow: `GET /health`, `GET /trips`, then the evaluator selects one of the 10 trips; the UI optionally calls `GET /trips/{id}/preview` and `/network`; pressing the run button sends **only the trip ID** in
   `POST /predict {"trip_id": "..."}`; the response contains the predicted (and classifier-only) route polylines, per-point correctness flags, metrics and warnings, which the UI draws.

### 7.2 Endpoints

| Method | Endpoint | Request | Response |
|---|---|---|---|
| GET | `/health` | none | `{"status":"ok","model":"lightgbm_tuned","model_loaded":true,"n_trips":10}` |
| GET | `/trips` | none | `{"trips":[{"trip_id","n_gps_points","duration_s"}, ...]}` (IDs and sizes only) |
| GET | `/trips/{id}/preview` | none | raw GPS track `[lat, lon]`, timestamps, bounding box |
| GET | `/trips/{id}/network` | `?buffer_m=150` | road segments near the trajectory |
| POST | `/predict` | `{"trip_id": "00000015"}` | `pred_route`, `classifier_only_route`, `point_correct`, `metrics`, `warnings` |

Errors are structured JSON: unknown trip -> HTTP 404 `TRIP_NOT_FOUND`; malformed body -> 422 `BAD_REQUEST`; internal failure -> 500 `PIPELINE_ERROR`.
If a trip has no `.route` on the server, `/predict` still returns the route and returns `null` accuracy with a warning (never an invented value).
The response `metrics` are: HMM and classifier-only pointwise accuracy, route continuity (both), macro F1 (both), number of disconnected breaks, inference time (ms).

### 7.3 Screenshots of the working UI

The screenshots were taken from the real evaluator UI (`http://<laptop-1-ip>:8000/web/`, served by the API) running against the 10 evaluator trips. Each result is produced by a live `POST /predict`; nothing is mocked
(`docs/report_sources/capture_ui_screenshots.py` drives the page and takes the screenshots).

![UI connected to the API (badge: Connected, lightgbm_tuned); trip list loaded, raw GPS of trip 00000015 shown](screenshots/ui_1_connected.png)

*Figure 7.1. Evaluator UI after connecting to the API at `http://10.8.120.216:8000` (Laptop 1). The badge shows the loaded model; the dropdown lists the 10 evaluation trips.*

![UI result for trip 00000015: HMM 93.5 %, classifier-only 60.1 %](screenshots/ui_3_result_00000015.png)

*Figure 7.2. Result for trip 00000015 (6,854 GPS points): observation top-1 93.5 % with HMM against 60.1 % classifier-only, route continuity 99.8 % against 96.4 %, 11 disconnected breaks, latency 5,305 ms.
Green points are correctly matched observations, red points are mismatches (scored server-side; labels are never sent to the UI).*

![UI result for trip 00000049: HMM 100.0 %, classifier-only 98.6 %](screenshots/ui_3_result_00000049.png)

*Figure 7.3. Result for trip 00000049 (2,642 GPS points): 100.0 % with HMM against 98.6 % classifier-only.*

### 7.4 Screenshots of successful API responses and the API request log

Browser views of two API responses (the same endpoints the UI calls):

![GET /health](screenshots/api_health.png)

![GET /trips](screenshots/api_trips.png)

*Figure 7.4. `GET /health` returns the loaded model and trip count; `GET /trips` returns only trip IDs and sizes (no labels).*

API log of the evaluator_test run (client IP, endpoint, status, latency):

```text
2026-11-18 16:26:00 [INFO] [Client: SYSTEM] STARTUP - Successfully loaded model: lightgbm_tuned
2026-11-18 16:26:00 [INFO] [Client: SYSTEM] STARTUP - Discovered 10 evaluation trips in 'evaluator_test' partition
2026-11-18 16:26:03 [INFO] [Client: SYSTEM] STARTUP - PathSnap API Server initialization complete.
2026-11-18 16:26:45 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [4231 ms]
2026-11-18 16:26:48 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [2796 ms]
2026-11-18 16:26:49 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [477 ms]
2026-11-18 16:26:50 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [1377 ms]
2026-11-18 16:26:51 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [1238 ms]
2026-11-18 16:26:53 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [1286 ms]
2026-11-18 16:26:53 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [578 ms]
2026-11-18 16:26:54 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [1260 ms]
2026-11-18 16:26:56 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [1392 ms]
2026-11-18 16:26:58 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [2346 ms]
```

Cross-machine requests: on 2026-11-18 Laptop 2 (10.8.120.202) reached Laptop 1 (10.8.120.216) over the LAN; the log (`logs/api.log`) records its requests, e.g.
`[Client: 10.8.120.202] /web/ - HTTP 200 [13 ms]`, `/web/app.js - HTTP 200`.

> **To add before submission:** Deliverables section 7 asks for screenshots of the API responses *on the CMD of the receiving laptop*. Take this screenshot of the Laptop 1 terminal during the live two-laptop run
> (the log lines above show what it will contain); it cannot be produced from code and is deliberately not simulated here.

### 7.5 Deployment notes
Python 3.12 is required (`numpy==2.5.5` has no build for 3.11). `uvicorn api.main:app --host 0.0.0.0 --port 8000` on Laptop 1; on Laptop 2 set `UI_API_URL` (Streamlit) or open `http://<laptop-1-ip>:8000/web/`.
Port 8000 must be allowed through the Windows firewall on Laptop 1. The data folder, partition and model can be overridden with the `DATA_DIR`, `EVAL_PARTITION` and `MODEL_PATH` environment variables (used for the evaluator_test run in Appendix A.5).
Full instructions: `docs/HOW_TO_RUN.md` and `docs/DEPLOYMENT_GUIDE.md`.
"""

INTEGRITY = """## 8. Academic Integrity, Plagiarism and AI-Use Disclosure

* **Independent work.** The data processing, label derivation, candidate generation, features, models, Bayesian optimisation, HMM decoder, API, UI and maps are the team's own implementation. No model files, weights, code or results
  were shared with other teams.
* **Evaluation integrity.** Models, features, candidate settings and decoder parameters were selected using the 78 clean training trips (cross-validation grouped by trip) and, for reporting only, student_test.
  The evaluator_test partition was read only for the final evaluation in Appendix A, after the pipeline was frozen. Ground-truth routes stay on the server and are never returned by the API. No prediction is hard-coded.
* **Reference labels.** Per-point labels are derived from the supplied reference route and are not independently verified ground truth; train trips 33 and 54 show a reference that is up to about 130 m from the GPS and were excluded from fitting and reported separately.
* **AI assistance (draft for the team to confirm and edit).** An AI coding assistant (Claude Code) was used during the project to explore the dataset, write and run code (data processing, evaluation scripts, parts of the pipeline), run the final evaluation, and draft documentation and this report from the experiment outputs.
  Every number in this report is read from the CSV files in `results/` produced by scripts in the repository. Where a result could not be reproduced from committed files (Random Forest cross-validation figures, note (1) in Section 3.1) this is stated and
  no figure is quoted. Each team member must be able to explain the code, experiments and conclusions; the report text must be read and checked by the team before submission.
"""


def build() -> Path:
    s345 = (REP / "REPORT_SECTIONS_3_4_5.md").read_text(encoding="utf-8")
    s2 = (REP / "REPORT_SECTION_2_2_EDA.md").read_text(encoding="utf-8")
    s6 = (REP / "REPORT_SECTION_6_AND_7.md").read_text(encoding="utf-8")

    # ---- section 2 (from the EDA write-up) ----
    s2 = re.sub(r"<!--.*?-->\s*", "", s2, flags=re.S)
    s2 = sub(s2, "# 2.2 Dataset Understanding and Exploratory Data Analysis", OVERVIEW_2_1 + "### 2.2 Dataset Understanding and Exploratory Data Analysis")
    s2 = s2.replace("\n## ", "\n#### ").replace("\n### Folium maps", "\n#### Folium maps")
    s2 = sub(s2, "#### Visualisations", "#### Visualisations\n\n![Candidate miss rate vs radius and cap (train)](../results/eda/plots/05_candidate_miss_rate.png)\n"
             "![GPS over the reference route on six hard trips (blue GPS, green route, red = more than 50 m away)](../results/eda/plots/07_label_alignment_check.png)\n"
             "![Road-network size and connectivity](../results/eda/plots/03_network_stats.png)\n"
             "![Time step between fixes](../results/eda/plots/02_dt_distribution.png)\n")

    # ---- sections 3-5: fill placeholders and apply corrections ----
    t = s345
    t = sub(t, "<!-- TABLE_3_1 -->", table_3_1())
    t = sub(t, "<!-- TUNING_TABLE -->", table_tuning())
    t = sub(t, "<!-- IMPORTANCE -->", importance_sentence())
    # Random Forest: keep only what is verifiable
    a, b = t.index("#### 3.2.4 Random Forest"), t.index("#### 3.2.5 LightGBM")
    rf = rd("model_route_metrics.csv").set_index("model").loc["random_forest"]
    t = t[:a] + f"""#### 3.2.4 Random Forest (bagging ensemble)
* **A1. Why was this algorithm considered suitable?** Averaged deep trees can fit the feature interactions logistic regression misses (heading x speed, distance x number of candidates, context x distance), need no scaling and tolerate the 150 m fill value.
* **B5-6. Important hyperparameters (initial values) and loss / objective.** 300 trees, min_samples_leaf 5, max_features sqrt, bootstrap, random_state 42; each tree minimises Gini impurity.
* **C7-8. Student-test performance (only the part reproducible from the committed files)** (`results/model_route_metrics.csv`): student-test observation-level top-1 {rf.clf_top1:.4f}; after HMM decoding {rf.hmm_top1:.4f};
  route continuity {rf.clf_continuity:.4f} -> {rf.hmm_continuity:.4f}.
* **A2 and C9. Not available.** Its cross-validation scores, macro F1, log loss, train-fit accuracy and timings are not in the committed results (the row is missing from `results/model_comparison.csv`),
  so no strengths/limitations, over/underfitting conclusion or cost figure is drawn for this model here. Regenerate with `python -m src.models.train_compare random_forest`.

""" + t[b:]
    t = sub(t, "Fold-to-fold standard deviation of CV top-1 is about 0.035–0.038 for every model, larger than the gap between\nthe four best models (0.860–0.869).",
            "Fold-to-fold standard deviation of CV top-1 is about 0.03-0.04 for every model, larger than the gap between\nthe best models with a verified CV score (LightGBM 0.860 and MLP 0.860).")
    t = sub(t, "its log loss (0.043) is worse than the forest's: defaults", "its log loss (0.043) is worse than the MLP's (0.038): defaults")
    a, b = t.index("### 3.4 Model selection"), t.index("## 4. Bayesian")
    cvm = rd("model_comparison.csv").set_index("model")
    order = ", ".join(f"{n} {cvm.cv_top1_mean[m]:.3f}" for m, n in [("lightgbm", "LightGBM"), ("mlp", "MLP"), ("logistic_regression", "Logistic Regression"), ("knn", "KNN"), ("gaussian_nb", "Gaussian NB")])
    t = t[:a] + f"""### 3.4 Model selection

Ranked by the primary metric (CV observation-level top-1; untuned models with a verified CV score): {order}. (Random Forest: CV score not available, see 3.1; its student-test top-1 of {rf.clf_top1:.3f} is level with the untuned LightGBM's {cvm.test_top1['lightgbm']:.3f}.)
The best three are within one fold standard deviation (about 0.04) of each other, so CV alone does not separate them. We chose **LightGBM for Bayesian optimisation** because:
1. **It is among the best before tuning** (student-test top-1 {cvm.test_top1['lightgbm']:.3f}, tied with the MLP's {cvm.test_top1['mlp']:.3f} and the Random Forest's {rf.clf_top1:.3f}).
2. **Cost makes tuning feasible.** One LightGBM fit takes about {cvm.train_time_s['lightgbm']:.1f} s against {cvm.train_time_s['mlp']:.0f} s for the MLP; thirty trials of group cross-validation cost minutes.
3. **It has the richest search space** (tree count, learning rate, leaves, depth, sampling, L1/L2), so the most to gain from tuning. Logistic regression and naive Bayes underfit, so tuning cannot remove their bias.
4. **Deployment.** The model is small and fast enough for the API.

The choice was confirmed after tuning: the tuned LightGBM is the best model on every quality metric in cross-validation and on student-test (top-1 0.882, macro F1 0.933, log loss 0.032), and also after HMM decoding (0.963, table 3.1).
A better classifier gives a better decoded route, so the classifier choice still matters even with the HMM.

---

""" + t[b:]
    t = sub(t, "Its CV top-1 (0.873) is also above the untuned random forest (0.869), so after tuning,\nLightGBM is the best model overall.", "Its CV top-1 (0.873) is above every untuned model with a verified CV score, so after tuning,\nLightGBM is the best model overall.")
    # figures
    t = sub(t, "### 3.2 Per-model analysis", "### 3.2 Compulsory Questions for Every Model\n\nFor each model the questions of Deliverables 3.2 are answered under the labels **A** (model understanding, Q1-2), **B** (training and configuration, Q5-6) and **C** (experimental results, Q7-9).")
    for old, new in (("* **Why suitable.**", "* **A1. Why was this algorithm considered suitable?**"), ("* **Strengths / limitations.**", "* **A2. Major strengths and limitations.**"),
                     ("* **Hyperparameters / objective.**", "* **B5-6. Important hyperparameters (initial values) and loss / objective.**"),
                     ("* **Results (untuned).**", "* **C7-8. Student-test performance: top-1, macro F1, log loss (untuned).**"),
                     ("* **Results.**", "* **C7-8. Student-test performance: top-1, macro F1, log loss.**"), ("* **Fit.**", "* **C9. Over- or underfitting and the evidence.**")):
        assert old in t, old
        t = t.replace(old, new)
    t = sub(t, "### 3.3 Visualisations\n", "### 3.3 Model Visualisations\n\n![Model comparison](../results/figures/models_comparison.png)\n![Train / CV / test top-1](../results/figures/models_train_cv_test.png)\n"
            "![Confusion matrices](../results/figures/models_confusion_matrices.png)\n![MLP loss curve](../results/figures/models_mlp_loss_curve.png)\n")
    t = sub(t, "## 5. HMM and Viterbi Route Reconstruction\n", "## 5. HMM and Viterbi Route Reconstruction\n")
    t = t.rstrip() + "\n\n![HMM vs classifier, top-1 per trip](../results/figures/hmm_vs_classifier_top1.png)\n![HMM decoder tuning](../results/figures/hmm_decoder_tuning.png)\n"
    t = sub(t, "Parameter importance (fANOVA, `tuning_importance.png`):", "![Tuning history](../results/figures/tuning_history.png)\n![Parameter importance](../results/figures/tuning_importance.png)\n\nParameter importance (fANOVA, `tuning_importance.png`):")

    # ---- section 6 (Folium) from person 4, plus EDA maps ----
    a, b = s6.index("## 6. Geographical"), s6.index("## 7. API")
    sec6 = s6[a:b].rstrip().rstrip("-").rstrip()
    sec6 += """

#### Screenshots of the interactive maps

![Train trip 00000001: GPS, nearby network, reference route and the candidate arcs of one point](../results/maps/screenshots/eda_train_00000001.png)
![Train trip 00000033: stationary GPS cluster about 130 m from the reference route (label-suspect trip)](../results/maps/screenshots/eda_train_00000033_drift.png)

The maps of the held-out evaluator trips are in Appendix A.6. Every map is generated from the team's own pipeline (`src/viz/folium_map.py`, `src/viz/make_report_maps.py`, `src/viz/make_final_maps.py`, `src/viz/p1_folium_map.py`).
"""

    front = """# PathSnap: Distributed Map-Matching and Route Reconstruction

**Technical Report** · PES University, Department of Computer Science and Engineering (RR Campus) · UE24CS352A: Machine Learning · PathSnap Hackathon

| SRN | Name |
|---|---|
| PES1UG24AM244 | Samarth Rajesh Rao |
| PES1UG24AM245 | Samhitha Kanth |
| PES1UG24AM243 | Samarth Ambedkar |
| PES1UG24AM246 | Sampreeth Jaladi |

Repository: https://github.com/Samhithakanth/ML_HACKATHON

## Abstract

GPS fixes from vehicles are noisy, so a track must be matched to the road network to recover the route actually driven. PathSnap generates candidate road segments for every GPS point (all arcs within 100 m, at most 30),
scores each candidate with a gradient-boosted classifier trained on 31 geometric, heading, kinematic, topological and temporal-context features, and decodes a connected route with an HMM and Viterbi algorithm over the directed road graph.
Six models from different families were compared by cross-validation grouped by trip; LightGBM was tuned with Bayesian optimisation (Optuna TPE, 30 trials). The decoder was tuned on out-of-fold training predictions only.
On the held-out evaluator_test partition (10 trips, 19,674 points) the frozen system assigns the correct reference road segment to **95.8 %** of observations (classifier alone: 81.5 %) with route continuity 0.978.
The system is served by a FastAPI backend and an evaluator UI on two laptops; reference labels stay server-side.

**Data partitions.** `train`: 80 trips (78 clean + 2 label-suspect trips reported separately) for fitting, cross-validation, tuning and decoder tuning. `student_test`: 10 trips for development-stage comparison. `evaluator_test`: 10 trips, final evaluation only (Appendix A).

[TOC]

---

"""
    out = "\n\n".join([front, s2.strip(), "---", t.split("## 3. Model Experimentation")[0].strip() if False else "", ""])
    # sections 3-5 text starts with header lines; extract from "## 3." onwards
    body345 = t[t.index("## 3. Model Experimentation"):]
    out = "\n\n".join([front, s2.strip(), "---", body345.strip(), "---", sec6.strip(), "---", SECTION_7.strip(), "---", INTEGRITY.strip(), "---", final_eval_section().strip()]) + "\n"
    out = re.sub(r"\n---\n\s*\n---\n", "\n---\n", out)
    out = out.replace("](../results/", "](../../results/")      # image links are relative to docs/report_sources/
    md_path = REP / "FINAL_REPORT.md"
    md_path.write_text(out, encoding="utf-8")
    return md_path


CSS = """
body{font-family:Segoe UI,Arial,sans-serif;font-size:10.5pt;line-height:1.42;color:#111;max-width:none;margin:0}
h1{font-size:21pt;margin:.2em 0}h2{font-size:15pt;border-bottom:1.5px solid #333;padding-bottom:2px;margin-top:1.4em;page-break-after:avoid}
h3{font-size:12.5pt;margin-top:1.1em;page-break-after:avoid}h4{font-size:11pt;page-break-after:avoid}
table{border-collapse:collapse;width:100%;font-size:8.6pt;margin:.6em 0;page-break-inside:auto}tr{page-break-inside:avoid}
th,td{border:1px solid #888;padding:2.5px 5px;vertical-align:top}th{background:#e8edf3}
code{font-family:Consolas,monospace;font-size:8.8pt;background:#f2f2f2;padding:0 2px}pre{background:#f5f5f5;padding:7px;font-size:7.8pt;overflow-wrap:anywhere;white-space:pre-wrap;border:1px solid #ddd}
pre code{background:none;padding:0}img{max-width:100%;display:block;margin:.6em auto;page-break-inside:avoid}blockquote{border-left:4px solid #c0392b;background:#fdf2f0;margin:.8em 0;padding:4px 10px}
@page{size:A4;margin:16mm 14mm}
"""


def to_pdf(md_path: Path) -> Path:
    import markdown
    html_body = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "toc", "sane_lists"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>PathSnap Technical Report</title><style>{CSS}</style></head><body>{html_body}</body></html>"
    html_path = md_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    pdf = OUT_PDF
    pdf.parent.mkdir(exist_ok=True)
    edge = next((p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe") if Path(p).exists()), None)
    assert edge, "Microsoft Edge not found; open FINAL_REPORT.html in a browser and print to PDF"
    subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", html_path.as_uri()],
                   check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    html_path.unlink()                       # generated intermediate; the PDF is the deliverable
    return pdf


if __name__ == "__main__":
    md = build()
    print("wrote", md)
    if "--no-pdf" not in sys.argv:
        print("wrote", to_pdf(md))

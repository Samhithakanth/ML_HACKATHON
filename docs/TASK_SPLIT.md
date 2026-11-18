# PathSnap: 4-person task split

Problem: reconstruct a vehicle's road-segment sequence from noisy 1 Hz GPS. Pipeline:
**candidate arcs per GPS point -> classifier scores them -> HMM/Viterbi over the road graph -> FastAPI (laptop 1) + UI (laptop 2).**

Data (verified): 80 train trips, 10 student-test trips, 10 hidden evaluator trips. Every trip has its own road network, and trips are
located all over the world. See [P1_DATA_NOTES.md](P1_DATA_NOTES.md).

## Hard requirements (from the organiser PDFs)
- Train and compare **at least 5 models** from different families; group-aware CV by trip.
- **Bayesian hyperparameter optimisation** on the chosen model; keep the trial log; compare untuned vs tuned.
- **HMM + Viterbi**; compare classifier-only vs classifier+HMM.
- **Folium maps** (GPS, network, reference route, predicted route, legend, layer control).
- **FastAPI** on laptop 1 and **UI** on laptop 2; must work for **all 10** hidden trips. Ground truth stays server-side. No hardcoded outputs.
- **Public GitHub repo** with README, no secrets. **One PDF report**; every number must be reproducible from the repo.
- Never touch evaluator data. Fit preprocessing on train only. Do not share code/models/results with other teams.
- Every member must be able to explain the whole pipeline in the viva. AI use must be disclosed; do not fabricate results.

## Dependency order
`P1 (data, labels, candidates)  ->  P2 (features, 5+ models)  ->  P3 (Bayesian opt, HMM)  ->  P4 plugs the real model into the API`.
P4 starts in parallel with a stub predictor. Freeze the file formats early.

---

## Person 1: Data, EDA, labels, candidates (STATUS: DONE)
**Done and in this repo** (`src/data/p1_*.py` plus the EDA scripts, `src/viz/`, `notebooks/01_eda.ipynb`, `results/eda/`, `docs/P1_DATA_NOTES.md`):
1. Loader for `.track/.nodes/.arcs/.route`; **per-trip local azimuthal-equidistant projection** (trips are worldwide, so no single UTM zone).
2. Inventory and quality scan of all 90 trips (counts, gaps, jumps, connectivity, isolated nodes, zero-length arcs, broken routes).
3. **Per-point reference labels** via a monotone DP alignment of GPS to `.route`; validated numerically and visually on hard trips.
4. Flagged **trips 33 and 54** (GPS drifts ~130 m from reference route) -> `trip_label_suspect`.
5. **Candidate generation**: all arcs within R = 100 m, capped at the K = 30 nearest; chosen by a train-only miss-rate sweep
   (clean-trip miss rate 0.65% train / 0.09% student-test).
6. Handoff parquet tables (regenerate with `python -m src.data.build_dataset`), EDA plots, Folium maps, notebook.

**Remaining for P1:** write report **Section 2.2** (preprocessing, patterns, features, missing-value analysis, plots, Folium map, partition named for
each analysis); screenshot the Folium map for the PDF; optionally add unit tests for labels/candidates. Then support P2/P3 with edge cases.

**Viva topics:** projection choice, label derivation, candidate radius/K, suspect trips, leakage handling.

---

## Person 2: Features and model experiments
**Input:** `data/processed/*_points.parquet`, `*_candidates.parquet` (schemas in docs/P1_DATA_NOTES.md).

1. **Feature engineering** per (point, candidate arc). Candidates already carry `dist_m`, `frac`, `rank`. Add:
   - distance to arc endpoints, arc length, arc bearing vs GPS motion heading (separates the two directions of a street),
   - speed and speed consistency with arc length, gap since previous fix (`dt_prev`),
   - distance gap to the nearest candidate, number of candidates, and context from previous/next points,
   - guard zero-length arcs. About 26% of train points hit the K = 30 cap, so `rank` and distance-gap features matter.
2. **Preprocessing:** fit any scaler on train only; save it.
3. **Evaluation harness:** `GroupKFold` by `trip_id` on the 80 train trips; decide how to treat the two suspect trips (recommend exclude or report separately).
   Metrics: macro F1, precision, recall, log loss, **observation-level top-1** (best-scoring candidate per point vs reference arc),
   **candidate miss rate**, route continuity, training and inference time. Log seeds and configs.
4. **Train at least 5 models from different families**, e.g. Logistic Regression, KNN, Random Forest / Extra Trees, HistGradientBoosting / LightGBM / XGBoost,
   MLP; optionally SVM, Gaussian NB, decision tree.
5. Evaluate on the 10 student-test trips. Produce the comparison table (Deliverables 3.1), per-model write-ups (3.2) and plots (3.3): CV F1 bar chart,
   student-test top-1 / F1, log loss, confusion matrices, loss curves only for models with epoch history.
6. Check over/underfitting with train vs CV vs test numbers. Justify the **primary selection metric** (top-1 or macro F1, with log loss because the HMM needs probabilities).
7. Export `predict_proba` per candidate, the scaler and the model, plus a `build_features(trip, candidates)` function that works from raw files only (P4 needs it).

**Output:** `src/features/`, `src/models/`, `notebooks/02_models.ipynb`, `results/model_comparison.csv`, saved models, report **Section 3**.

---

## Person 3: Bayesian optimisation and HMM / Viterbi
**Input:** the selected model from P2 and candidate probabilities.

1. **Bayesian optimisation** (Optuna or scikit-optimize) on the selected model: define the search space, the objective (group-CV macro F1 or log loss),
   30+ trials, **keep the trial log**, refit with the best parameters. Compare untuned vs tuned on CV F1, student-test top-1, macro F1, log loss and inference time.
   Plot optimisation history and parameter importance.
2. **HMM + Viterbi** (log-space, vectorised; the longest trip has 19,313 points):
   - hidden states = candidate arcs at each time step; keep the candidate-to-arc mapping,
   - emission = classifier probability (optionally combined with a distance term),
   - transition = road-graph connectivity (successor arc, same-arc stay), penalising shortest-path vs straight-line distance mismatch,
   - **handle:** no candidates (0.68% of train points), **large time gaps** (break and restart the sequence; 22 train trips have a gap over 60 s),
     **disconnected networks** (89 of 90 trips have several components; use a floor probability or a bridging step),
   - tune decoder parameters on train and student-test only.
3. **Comparison:** classifier-only vs classifier+HMM: pointwise accuracy, sequence-level metrics (e.g. route precision/recall or edit distance), route continuity;
   discuss cases where HMM does not help.
4. Own `src/eval/metrics.py` so the API and the report use identical metric definitions.

**Output:** `src/tuning/`, `src/hmm/`, `src/eval/metrics.py`, the trial log, `configs/hmm.yaml`, final tuned model; report **Sections 4 and 5**.

---

## Person 4: API, UI, Folium, deployment, repo and report assembly
Full step-by-step in [PERSON4_WORKFLOW.md](../PERSON4_WORKFLOW.md). Summary:
1. Repo hygiene: public repo, `.gitignore`, pinned `requirements.txt`, README (install / run / deploy / reproduce), secrets check.
2. **FastAPI** (laptop 1): `GET /health`, `GET /trips`, `POST /predict {trip_id}`; one `predict_trip` function (raw files only, never `.route`);
   server-side scoring against reference labels (`src.data.p1_labels.align_to_route`) that are never returned; request logging to stdout and file.
   Start with a stub (nearest-candidate baseline), swap in the real model when P2/P3 deliver.
3. **UI** (laptop 2, e.g. Streamlit): list the 10 trips, send the trip ID only, show the predicted route on a map and the metrics; configurable API URL.
4. **Folium**: extend `src/viz/p1_folium_map.py` (GPS, network near the GPS, reference route, predicted route, legend, layer control); report maps for representative trips.
5. **Two-laptop test early** (LAN IP, firewall port 8000, hotspot as plan B); run all 10 trips; keep the API log visible.
6. Assemble the single PDF in the Deliverables order; write **Section 7** (UI screenshots, API responses on the receiving laptop's CMD); verify every number against `results/`;
   AI-use disclosure.

**Output:** `api/`, `ui/`, README, final PDF.

---

## Timeline and handoffs

| Phase | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| 1 | DONE: data, labels, candidates | Metrics harness, CV, feature design on a few trips | Prototype Viterbi on reference-route emissions | Repo, stub API and UI, Folium function |
| 2 | Report 2.2, map screenshots, tests | Features, train 5+ models | Build the HMM on P2's early outputs | Wire the pipeline into the API |
| 3 | Edge cases, help P2/P3 | Pick winner (**handoff to P3**) | Bayesian optimisation, HMM tuning | Two-laptop test |
| 4 | Review the report | Report Section 3 | Report Sections 4 and 5 | Assemble PDF, final dry run on all 10 trips |

## Shared rules
- Agree formats up front: the two parquet tables, `(point_idx, arc_id, prob)` for classifier output, model artifact paths, metric definitions.
- Leakage: nobody touches evaluator data; scalers fit on train only; no model or hyperparameter choice from the final evaluation.
- Each person should understand their neighbours' parts; evaluators ask anyone.
- Originality: no copying other teams, no fabricated numbers; report only what the repo reproduces.

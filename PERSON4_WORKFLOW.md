# Person 4 workflow: API, UI, Folium, repo, report assembly

You own everything the evaluator sees. You can start now: the data contract is frozen, and until the real model arrives you
plug in a **stub predictor** (nearest-candidate baseline). When P2/P3 deliver, you swap one function.

Read first (30 min): `docs/DATA_NOTES.md`, then the PDFs *Main Guidelines* (section 6 "Final Evaluation"),
*Deliverables* (sections 1, 6, 7) and *Assessment and Plagiarism Policy* (section 7, leakage rules).

## 0. What the evaluator will do (this is your acceptance test)
1. API running on Laptop 1, UI on Laptop 2. Evaluator opens the UI.
2. UI lists the 10 hidden trips. Evaluator picks one.
3. UI sends **only the trip ID** to the API.
4. API runs the full pipeline (candidates -> classifier -> HMM/Viterbi) on that trip's raw GPS + road network.
5. API scores against its **own server-side reference labels** and returns predicted route + metrics.
6. UI shows the route on a map and the scores. **Works for all 10 trips.** The evaluator will also look at the API logs on Laptop 1.

Hard rules: ground truth never reaches the model or the UI. No hardcoded predictions. No bypassing the API.
No secrets in the repo.

## 1. Setup (day 0)
- [ ] Create the **public** GitHub repo; add all 4 teammates as collaborators. Commit early and often (history is reviewed).
- [ ] Folder layout (the existing code already follows it):
  ```
  src/data/    P1   loader, labels, candidates, build_dataset
  src/features/ P2
  src/models/   P2
  src/tuning/   P3
  src/hmm/      P3
  src/viz/      P4   folium_map.py
  api/          P4   main.py, pipeline.py, schemas.py, config.py
  ui/           P4   app.py
  configs/      shared (hmm.yaml, api.yaml)
  models/       trained artifacts (model, scaler, hmm config)
  data/raw|processed   NOT committed if large (.gitignore), document how to obtain
  results/ docs/ notebooks/ tests/
  ```
- [ ] `.gitignore`: `data/raw/`, `*.pyc`, `.venv/`, `.env`, large parquet. Check size limits for model files.
- [ ] `requirements.txt` with pinned versions: fastapi, uvicorn, pydantic, requests, streamlit (or your UI choice), folium, streamlit-folium,
  numpy, pandas, scipy, shapely, pyproj, pyarrow, scikit-learn (+ lightgbm/xgboost/optuna once P2/P3 decide). Test in a **clean venv**.
- [ ] README skeleton: Install / Run API / Run UI / Reproduce training / Two-laptop deployment / Repo map / AI-use disclosure.

## 2. The API contract (agree with P2/P3 now, then don't change it)
Everything the model side must expose is **one function**:
```python
# api/pipeline.py
def predict_trip(trip_id: str, data_dir: Path) -> dict:
    """Uses ONLY raw .track/.nodes/.arcs. Never reads .route."""
    return {"trip_id": ..., "predicted_arcs": [int per GPS point],    # arc_id for each point_idx
            "classifier_arcs": [int per point],                       # before HMM (for comparison)
            "timings_ms": {"candidates":..,"classifier":..,"hmm":..,"total":..}}
```
Endpoints (`api/main.py`):
| Endpoint | Behaviour |
|---|---|
| `GET /health` | `{"status":"ok","model_loaded":true}` |
| `GET /trips` | list of the 10 evaluation trip IDs (IDs only) |
| `POST /predict` body `{"trip_id": "00000007"}` | runs `predict_trip`, then scores internally, returns below |

`/predict` response (shared with the UI):
```json
{"trip_id":"00000007","n_points":1420,
 "points":[{"point_idx":0,"lat":..,"lon":..,"pred_arc":123}, ...],
 "route_geometry":[[[lat,lon],[lat,lon]], ...],       // predicted arcs as polylines, de-duplicated in order
 "metrics":{"top1_accuracy":..,"macro_f1":..,"classifier_only_top1":..,"route_continuity":..,"inference_ms":..}}
```
**Never return** `label_arc`, `.route` contents, or reference geometry. The UI may only draw GPS, network, and the *predicted* route.
(For the report's reference-vs-predicted map you make a separate offline map script using student-test trips.)

Server-side scoring: at startup, build reference labels for the evaluation trips with `src.data.labels.align_to_route`
(needs `.route`, which lives only on Laptop 1) and keep them in a private dict used only by the scoring function.
Route continuity: fraction of consecutive predicted arcs (that differ) that are connected (`to_node == next from_node`).
Agree the exact metric definitions with P3 so API numbers match the report.

Logging: use Python `logging` to stdout **and** a file `logs/api.log`; log timestamp, client IP, endpoint, trip_id, status, latency.
Evaluators check this on screen, so keep lines readable.

## 3. Build order
**Step A, stub pipeline (day 1).** Implement `predict_trip` as: load trip -> `generate_candidates(trip, 100, 30)` -> pick the nearest
candidate per point (rank 0). It's a real baseline (call it "nearest-arc baseline" in docs). Points with no candidate: carry the previous arc.
- Done when: `POST /predict` works for all 10 `student_test` trips and returns metrics.

**Step B, API hardening (day 1-2).**
- [ ] Load model/data once at startup (FastAPI `lifespan`), not per request. Cache road networks and candidates per trip.
- [ ] Config via `configs/api.yaml` / env vars: `DATA_DIR`, `EVAL_TRIPS`, model path. No absolute paths in code.
- [ ] Pydantic schemas, 404 for unknown trip_id, 500 handler that logs the traceback but returns a clean message.
- [ ] Test the biggest trips for latency (student_test max is 3,958 points; hidden trips could be similar). Measure and report ms.
- [ ] `pytest` smoke tests in `tests/`: health, trips, predict on every trip, no label fields in the response.

**Step C, Folium (day 1-3).** Extend `src/viz/folium_map.py` (exists). Required layers: GPS observations, road network (only near the
GPS, full networks have 100k+ arcs), reference route, predicted route, with legend, layer control and fitted zoom.
- [ ] `make_map(trip, predicted_arcs=...)` already supports a predicted layer; test with the stub output.
- [ ] Add classifier-only vs HMM layers once P3 delivers.
- [ ] Script `python -m src.viz.make_report_maps` that writes HTML for 2-3 representative student-test trips (incl. one failure case) to `results/maps/`.
- [ ] Take screenshots of at least one map for the PDF.

**Step D, UI (day 2-3).** Streamlit is the quickest (Laptop 2 only needs Python + requests).
- [ ] API base URL in a sidebar field / `UI_API_URL` env var (default `http://<laptop1-ip>:8000`).
- [ ] On load: `GET /trips`, show 10 buttons or a dropdown.
- [ ] On select: `POST /predict` with the trip ID **only**, show a spinner, then the map (GPS + predicted route + network from response data) and a metrics table.
- [ ] Show a clear error if the API is unreachable.
- [ ] UI must not import anything from `src/data` labels or read `.route`.

**Step E, two-laptop test (day 3, do it early, not the night before).**
- [ ] `uvicorn api.main:app --host 0.0.0.0 --port 8000` on Laptop 1; find its LAN IP (`ipconfig`).
- [ ] Windows firewall: allow inbound TCP 8000 on Laptop 1 (private network profile).
- [ ] Same Wi-Fi/hotspot; `curl http://<ip>:8000/health` from Laptop 2 first, then the UI.
- [ ] Run all 10 trips end to end. Keep the API terminal visible: those logs are your evidence.
- [ ] Plan B: a mobile hotspot or ethernet cable in case venue Wi-Fi isolates clients.

**Step F, integrate the real model (when P2/P3 deliver).**
- [ ] Inputs you need from them: model file(s) + scaler in `models/`, `configs/hmm.yaml`, and `src/features` + `src/hmm` importable functions.
- [ ] Replace the stub body of `predict_trip` with: candidates -> `build_features` -> `model.predict_proba` -> `viterbi_decode`. Keep the same signature.
- [ ] Verify the API output equals P3's offline result on a student-test trip (same arcs). Mismatch means a preprocessing difference; fix it with P2.
- [ ] Re-run Step E.

**Step G, repo and report (final stretch).**
- [ ] README: exact commands for install, run API, run UI, retrain, regenerate maps. Test it on a **fresh clone in a new venv**.
- [ ] Confirm all required components exist in the repo (Deliverables table: data processing, features, training for every model,
  evaluation, Bayesian optimisation + trial logs, HMM, API, UI, Folium, CSVs/logs/configs).
- [ ] Secrets scan (`git grep -i -E "api[_-]?key|password|token"`).
- [ ] Assemble the single PDF from the 4 people's sections, in the order of *Deliverables* section 2.
- [ ] Write report **section 7**: UI screenshots, and API-response screenshots **taken in the CMD of the receiving laptop**.
- [ ] Check every number in the report against files in `results/` (verifiable from the repo).
- [ ] AI-use disclosure statement per the event policy.

## 4. Definition of done for your part
- [ ] 10/10 trips work through UI -> API on two machines.
- [ ] Response contains no ground truth; the UI never sees it.
- [ ] Logs show each request.
- [ ] Fresh-clone install works from the README.
- [ ] Map shows all four layers with legend and layer control.

## 5. Things you will be asked in the viva
Request flow end to end; where model loading happens; why ground truth is server-side; how route continuity is computed;
what happens for a trip with a huge time gap; how the API handles an unknown trip ID; how you'd change the port/IP; what each
Folium layer is. Be able to explain the stub too, if you describe it as a baseline.

## 6. Open questions to settle with teammates today
1. Final metric definitions (top-1, macro F1, continuity): same code for API and report (put it in `src/eval/metrics.py`, owned by P3).
2. Whether the hidden evaluation trips are supplied as the same four-file folders; assume yes and use the `student_test` folders as stand-ins.
3. Which classifier library the final model uses (affects `requirements.txt` and model loading).

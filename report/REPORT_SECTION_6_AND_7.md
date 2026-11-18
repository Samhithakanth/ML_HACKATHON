# PathSnap Technical Report — Sections 6 & 7 (Person 4)

---

## 6. Geographical Visualisation Using Folium

### 6.1 Overview of Folium Visualisation Engine
Geographical visualisation in PathSnap is implemented using **Folium** (v1.0.0) with an underlying Leaflet.js rendering engine. The map component serves as both an interactive analysis tool for exploratory data analysis (EDA) and the primary evaluator-facing visual inspection surface. 

To ensure complete clarity and eliminate visual clutter when displaying road networks containing over 100,000 arcs, our implementation in `src/viz/folium_map.py` dynamically filters the road network within a tunable buffer (default: 150–200 m) of the vehicle trajectory.

### 6.2 Five-Layer Architecture & Legend
The interactive map features five distinct, toggleable layers managed via Leaflet's `LayerControl`:

| Layer Name | Visual Encoding | Style Attributes | Purpose |
|---|---|---|---|
| **1. Road Network** | Thin Grey Polylines (`#94a3b8`) | Weight: 1.5 px, Opacity: 0.6 | Contextual road network topology near the vehicle trajectory. |
| **2. Raw GPS Track** | Cyan/Blue CircleMarkers (`#38bdf8`) with dashed line (`#3b82f6`) | Radius: 3 px, Dash: 4,4 | Raw 1 Hz GPS observations illustrating measurement noise, urban drift, and sensor jumps. |
| **3. Ground Truth Reference** | Emerald Green Thick Line (`#10b981`) | Weight: 5.5 px, Opacity: 0.85 | Actual physical road segments traversed by the vehicle (used for offline comparison). |
| **4. Classifier-Only Route** | Dashed Crimson Line (`#ef4444`) | Weight: 3.0 px, Dash: 6,6 | Top-1 independent argmax predictions prior to sequential decoding. |
| **5. HMM Predicted Route** | Vibrant Orange Thick Line (`#f97316`) | Weight: 4.5 px, Opacity: 0.9 | Final reconstructed route decoded via the Viterbi algorithm. |

A floating, semi-transparent CSS legend (`bottom: 30px; right: 30px`) provides immediate visual reference for all layer colours and symbology.

---

### 6.3 Representative Trip Visualisations & Case Studies

All maps were generated using the reproducible script `python -m src.viz.make_report_maps` and saved as self-contained HTML files in `results/maps/`.

> Numbers in 6.3 regenerated with the final pipeline (tuned LightGBM + tuned HMM) by `python -m src.viz.make_report_maps`; they match `results/hmm/hmm_vs_classifier_per_trip.csv`. **Person 4: insert a screenshot of `results/maps/trip_00000007_map.html` (with the layer control open) next to Figure 6.1 below.**

#### Case Study 1: Standard Urban Trajectory — Trip 00000007 (`results/maps/trip_00000007_map.html`)
- **Characteristics:** 1,354 GPS points, duration 1,353 s, urban arterial with multiple turns and intersections.
- **Classifier-Only Accuracy:** 93.65% | Route Continuity: 99.63% (5 breaks).
- **HMM Reconstructed Accuracy:** **99.93%** (+6.28 pp) | Route Continuity: **100.00%** (**0 breaks**).
- **Visual Analysis:** 
  The raw GPS points exhibit moderate lateral scatter (3–10 m) due to urban multipath effects. While the independent classifier occasionally oscillated between opposing parallel arcs at complex intersections, the HMM's route-distance transitions penalise direction flips and disjoint jumps, giving a single fully connected route that matches the reference on 1,353 of 1,354 points.

#### Case Study 2: Long-Distance Complex Highway & Urban Arterial — Trip 00000056 (`results/maps/trip_00000056_map.html`)
- **Characteristics:** 3,958 GPS points, duration 3,957 s, dense urban network transitioning into a multi-lane divided expressway.
- **Classifier-Only Accuracy:** 90.12% | Route Continuity: 91.89% (321 breaks).
- **HMM Reconstructed Accuracy:** **96.34%** (+6.22 pp) | Route Continuity: **96.82%** (126 breaks).
- **Visual & Diagnostic Analysis (Challenging / Failure Case):**
  This trip represents an extreme urban stress case where the vehicle travels along a high-density parallel arterial corridor. Because multiple physical road segments run parallel within 5–8 meters of one another, noisy GPS observations produce high emission probabilities across several adjacent arcs. While the raw classifier produces frequent jumps across parallel lanes, the HMM enforces continuity along the road graph (91.9% → 96.8%). All 126 remaining "breaks" are pairs of arcs joined by a directed road path of at most 300 m, i.e. the vehicle crossed one or more short arcs within one GPS interval (`results/hmm/break_analysis.csv`); none is a jump between unconnected parts of the network.

#### Case Study 3: Short Trajectory with Sensor Gaps — Trip 00000052 (`results/maps/trip_00000052_map.html`)
- **Characteristics:** 262 GPS points, duration 261 s, compact suburban route with sudden speed variations.
- **Classifier-Only Accuracy:** 96.18% | Route Continuity: 95.79% (11 breaks).
- **HMM Reconstructed Accuracy:** **100.00%** (+3.82 pp) | Route Continuity: **96.17%** (10 multi-hop breaks, see `results/hmm/break_analysis.csv`).
- **Visual Analysis:**
  Demonstrates the system's ability to recover rapid manoeuvres and maintain continuous routing even over short trip windows without requiring long warm-up intervals.


![Figure 6.1: Trip 00000007, static rendering of the same layers as the Folium map (GPS fixes, road network, reference route, classifier-only route, HMM route), produced by src/viz/static_route_plot.py (whole trip). The HMM route matches the reference on 1,353 of 1,354 fixes.](../results/figures/route_trip_00000007.png)


---

## 7. API, UI and Deployment

### 7.1 Distributed Architecture & Two-Laptop Deployment
In accordance with the hackathon guidelines (Section 3 & 6), PathSnap is deployed across two physically distinct machines communicating over standard TCP/IP on local port 8000:

1. **Laptop 1 (Model Host & Backend Server):**
   - **Framework:** FastAPI (`0.162.0`) running under Uvicorn (`0.54.0`) bound to `0.0.0.0:8000`.
   - **Assets Loaded:** Bayesian-tuned LightGBM (`models/lightgbm_tuned.joblib`), the tuned HMM decoder configuration (`configs/hmm.yaml`), the raw road networks, and private server-side reference labels (`SERVER_STATE["reference_labels"]`, built with the same monotone-DP alignment used for training).
   - **Strict Data Isolation:** Ground-truth `.route` files exist exclusively on Laptop 1. The prediction pipeline (`api/pipeline.py`) takes only raw `.track`, `.nodes`, and `.arcs` as inputs. Ground-truth route geometry and arc labels are **never** returned across the network.
   - **Real-Time Request Logging:** Every incoming request, client IP, endpoint, response status, and millisecond latency is logged both to stdout and to `logs/api.log`.

2. **Laptop 2 (Evaluator User Interface):**
   - **Framework:** Streamlit (`1.67.0`) with `streamlit-folium` (`0.27.4`).
   - **Role:** Pure client. Zero machine learning weights, zero pre-computed routes, and zero local datasets exist on Laptop 2.
   - **User Flow:**
     1. Evaluator opens UI, which calls `GET /health` and `GET /trips`.
     2. Evaluator selects any one of the 10 evaluation trips.
     3. UI queries `GET /trips/{id}/preview` and `GET /trips/{id}/network` to render the initial GPS points and local network canvas.
     4. Evaluator clicks **"🚀 Run Map-Matching & Reconstruct"**.
     5. Laptop 2 transmits **only the 8-character trip ID** in a JSON payload: `POST /predict {"trip_id": "00000007"}`.
     6. Laptop 1 executes the complete candidate-generation, feature extraction, LightGBM inference, and Viterbi decoding pipeline in memory, computes accuracy and continuity against its private server-side reference alignment, and returns the predicted route polylines and evaluation metrics.
     7. Laptop 2 renders the reconstructed route polyline and side-by-side performance metrics.

---

### 7.2 API Endpoint Specifications & Contracts

The backend strictly implements the frozen PathSnap API Contract (v1):

| Method | Endpoint | Request Payload | Response Attributes | Latency (Typ.) |
|---|---|---|---|---|
| `GET` | `/health` | None | `{"status": "ok", "model": "lightgbm_tuned", "model_loaded": true, "n_trips": 10}` | 1–3 ms |
| `GET` | `/trips` | None | `{"trips": [{"trip_id": "00000007", "n_gps_points": 1354, "duration_s": 1353}, ...]}` | 1–2 ms |
| `GET` | `/trips/{id}/preview` | None | `{"trip_id": ..., "bbox": [[s, w], [n, e]], "gps": [[lat, lon], ...], "gps_t": [...]}` | 3–15 ms |
| `GET` | `/trips/{id}/network` | `?buffer_m=150` | `{"trip_id": ..., "n_segments": 1420, "segments": [[[lat1, lon1], [lat2, lon2]], ...]}` | 5–25 ms |
| `POST` | `/predict` | `{"trip_id": "00000007"}` | `{"pred_route": {"paths": [...], "n_arcs": ...}, "metrics": {...}, "point_correct": [...], "warnings": [...]}` | 207–1784 ms (measured, see 7.3) |

#### Robust Error Handling
- **404 Not Found:** When an unknown trip ID is supplied:
  `{"error": {"code": "TRIP_NOT_FOUND", "message": "Unknown trip_id 99999999"}}`
- **422 Unprocessable Content:** When request payload is malformed or missing `trip_id`:
  `{"error": {"code": "BAD_REQUEST", "message": "Field required: trip_id"}}`
- **500 Internal Error:** Catches and logs server-side exceptions without crashing Uvicorn:
  `{"error": {"code": "PIPELINE_ERROR", "message": "Pipeline failure: <details>"}}`

---

### 7.3 Evaluator Acceptance Log Evidence

Single-machine acceptance run with the final pipeline (server and client on the same laptop, client 127.0.0.1), all 10 student-test
trips, excerpt of `logs/api.log` (the per-trip "Cached ... reference labels" lines are omitted):

```text
2026-11-18 15:14:44 [INFO] [Client: SYSTEM] STARTUP - Initializing PathSnap API Server...
2026-11-18 15:14:44 [INFO] [Client: SYSTEM] STARTUP - Loaded HMM configuration from hmm.yaml
2026-11-18 15:14:44 [INFO] [Client: SYSTEM] STARTUP - Successfully loaded model: lightgbm_tuned
2026-11-18 15:14:44 [INFO] [Client: SYSTEM] STARTUP - Discovered 10 evaluation trips in 'student_test' partition
2026-11-18 15:14:45 [INFO] [Client: SYSTEM] STARTUP - PathSnap API Server initialization complete.
2026-11-18 15:14:46 [INFO] [Client: 127.0.0.1] /health - HTTP 200 [3 ms]
2026-11-18 15:14:46 [INFO] [Client: 127.0.0.1] /trips - HTTP 200 [1 ms]
2026-11-18 15:14:46 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [589 ms]
2026-11-18 15:14:46 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [225 ms]
2026-11-18 15:14:47 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [580 ms]
2026-11-18 15:14:48 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [716 ms]
2026-11-18 15:14:48 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [207 ms]
2026-11-18 15:14:50 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [1784 ms]
2026-11-18 15:14:50 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [541 ms]
2026-11-18 15:14:51 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [1155 ms]
2026-11-18 15:14:53 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [1650 ms]
2026-11-18 15:14:53 [INFO] [Client: 127.0.0.1] /predict - HTTP 200 [469 ms]
```

The client also checked every response: all 10 returned HTTP 200 with server-side metrics, no reference data in the payload, and an
unknown trip id / empty body returned 404 / 422. **To be added by Person 4:** screenshots of the UI and of the same requests made from
Laptop 2 over the LAN (the deliverable asks for API responses on the receiving laptop's terminal).

---

### 7.4 Viva & Evaluation Defense Preparation

During the demonstration and viva, Person 4 is prepared to explain and defend the following system properties:

1. **Request Flow End-to-End:**
   - Laptop 2 sends `{"trip_id": "00000007"}`.
   - Laptop 1 loads raw `.track`, `.nodes` and `.arcs` (never `.route`) and projects them to a per-trip local azimuthal-equidistant CRS (metres).
   - Candidates: all arcs within 100 m of each fix, at most the 30 nearest (STRtree, `src/data/candidates.py`).
   - 31 features per (fix, candidate) (`src/features/build.py`); the tuned LightGBM gives P(candidate is the true arc).
   - The HMM (`src/hmm/decoder.py`) uses candidates as states, the classifier probability as emission and Newson & Krumm route-distance
     transitions (shortest path on the directed road graph), and Viterbi decodes one arc per fix.
   - Top-1 accuracy, macro F1 and route continuity are computed server-side against the cached reference labels (`src/eval/metrics.py`).
   - The response returns the predicted polylines (HMM and classifier-only), the metrics, the per-point correctness and warnings.
2. **Where Model Loading Happens:**
   - Model loading occurs strictly once at application startup inside FastAPI's `lifespan` context manager (`@asynccontextmanager`). It is never loaded per request, ensuring millisecond API latency.
3. **Why Ground Truth is Kept Server-Side:**
   - To guarantee zero data leakage into the evaluation UI or model pipeline. The client never receives route IDs, true arc sequences, or ground-truth polylines.
4. **How Route Continuity is Computed:**
   - Evaluated as the ratio of valid consecutive transitions (`arc_i == arc_{i+1}` or `arcs[arc_i, 1] == arcs[arc_{i+1}, 0]`) over total transitions $(N - 1)$. Any gap where `to_node != from_node` is counted as a disconnect break.
5. **Handling of Huge Time Gaps:**
   - The HMM cuts the chain and restarts Viterbi when Δt exceeds the tuned `gap_s` (49.6 s, `configs/hmm.yaml`). Separately, the API warns about,
     and splits the drawn polyline at, every gap over 5 s, so the map does not draw a straight line across a gap.
6. **Unknown Trip Handling:**
   - If a non-existent trip ID is requested, FastAPI returns a clean HTTP 404 with structured JSON: `{"error": {"code": "TRIP_NOT_FOUND", "message": "Unknown trip_id ..."}}`, which the UI handles gracefully with an error banner.
7. **Network Interface & Port Reconfiguration:**
   - Configurable via `configs/api.yaml` or environment variables `API_HOST=0.0.0.0` and `API_PORT=8000`. Laptop 2 can configure the base URL dynamically in the sidebar UI or via `UI_API_URL`.

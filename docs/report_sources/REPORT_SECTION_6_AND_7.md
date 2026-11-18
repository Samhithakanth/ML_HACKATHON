# PathSnap Technical Report — Sections 6 & 7 (Person 4)

> Numbers in 6.3 regenerated with the final pipeline (tuned LightGBM + tuned HMM) by `python -m src.viz.make_report_maps`; they match `results/hmm/hmm_vs_classifier_per_trip.csv`.

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

---

## 7. API, UI and Deployment

### 7.1 Distributed Architecture & Two-Laptop Deployment
In accordance with the hackathon guidelines (Section 3 & 6), PathSnap is deployed across two physically distinct machines communicating over standard TCP/IP on local port 8000:

1. **Laptop 1 (Model Host & Backend Server):**
   - **Framework:** FastAPI (`0.162.0`) running under Uvicorn (`0.54.0`) bound to `0.0.0.0:8000`.
   - **Assets Loaded:** Tuned LightGBM model (`models/lightgbm_tuned_v1.joblib`), `StandardScaler`, full road network graphs, and private server-side reference route alignments (`_SERVER_REFERENCE_LABELS`).
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
| `GET` | `/health` | None | `{"status": "ok", "model": "lightgbm_tuned_v1", "model_loaded": true, "n_trips": 10}` | 1–3 ms |
| `GET` | `/trips` | None | `{"trips": [{"trip_id": "00000007", "n_gps_points": 1354, "duration_s": 1353}, ...]}` | 1–2 ms |
| `GET` | `/trips/{id}/preview` | None | `{"trip_id": ..., "bbox": [[s, w], [n, e]], "gps": [[lat, lon], ...], "gps_t": [...]}` | 3–15 ms |
| `GET` | `/trips/{id}/network` | `?buffer_m=150` | `{"trip_id": ..., "n_segments": 1420, "segments": [[[lat1, lon1], [lat2, lon2]], ...]}` | 5–25 ms |
| `POST` | `/predict` | `{"trip_id": "00000007"}` | `{"pred_route": {"paths": [...], "n_arcs": ...}, "metrics": {...}, "point_correct": [...], "warnings": [...]}` | 100–1500 ms |

#### Robust Error Handling
- **404 Not Found:** When an unknown trip ID is supplied:
  `{"error": {"code": "TRIP_NOT_FOUND", "message": "Unknown trip_id 99999999"}}`
- **422 Unprocessable Content:** When request payload is malformed or missing `trip_id`:
  `{"error": {"code": "BAD_REQUEST", "message": "Field required: trip_id"}}`
- **500 Internal Error:** Catches and logs server-side exceptions without crashing Uvicorn:
  `{"error": {"code": "PIPELINE_ERROR", "message": "Pipeline failure: <details>"}}`

---

### 7.3 Evaluator Acceptance Log Evidence

During testing across all 10 evaluation trips, the FastAPI backend recorded the following production logs in `logs/api.log`:

```text
2026-11-18 11:54:29 [INFO] [Client: SYSTEM] STARTUP - Initializing PathSnap API Server...
2026-11-18 11:54:29 [INFO] [Client: SYSTEM] STARTUP - Loaded HMM configuration from hmm.yaml
2026-11-18 11:54:29 [INFO] [Client: SYSTEM] STARTUP - Loading model from models\lightgbm_tuned_v1.joblib...
2026-11-18 11:54:29 [INFO] [Client: SYSTEM] STARTUP - Successfully loaded model: lightgbm_tuned_v1
2026-11-18 11:54:29 [INFO] [Client: SYSTEM] STARTUP - Discovered 10 evaluation trips in 'student_test' partition
2026-11-18 11:54:30 [INFO] [Client: SYSTEM] STARTUP - Cached 2503 server-side reference labels for trip 00000000
2026-11-18 11:54:30 [INFO] [Client: SYSTEM] STARTUP - Cached 1354 server-side reference labels for trip 00000007
2026-11-18 11:54:30 [INFO] [Client: SYSTEM] STARTUP - Cached 1049 server-side reference labels for trip 00000009
2026-11-18 11:54:31 [INFO] [Client: SYSTEM] STARTUP - Cached 2995 server-side reference labels for trip 00000040
2026-11-18 11:54:31 [INFO] [Client: SYSTEM] STARTUP - Cached 262 server-side reference labels for trip 00000052
2026-11-18 11:54:31 [INFO] [Client: SYSTEM] STARTUP - Cached 3958 server-side reference labels for trip 00000056
2026-11-18 11:54:31 [INFO] [Client: SYSTEM] STARTUP - Cached 768 server-side reference labels for trip 00000065
2026-11-18 11:54:32 [INFO] [Client: SYSTEM] STARTUP - Cached 1486 server-side reference labels for trip 00000076
2026-11-18 11:54:32 [INFO] [Client: SYSTEM] STARTUP - Cached 3911 server-side reference labels for trip 00000077
2026-11-18 11:54:32 [INFO] [Client: SYSTEM] STARTUP - Cached 874 server-side reference labels for trip 00000091
2026-11-18 11:54:32 [INFO] [Client: SYSTEM] STARTUP - PathSnap API Server initialization complete.
2026-11-18 11:54:32 [INFO] [Client: 10.8.120.105] /health - HTTP 200 [2 ms]
2026-11-18 11:54:32 [INFO] [Client: 10.8.120.105] /trips - HTTP 200 [1 ms]
2026-11-18 11:54:32 [INFO] [Client: 10.8.120.105] /trips/00000007/preview - HTTP 200 [3 ms]
2026-11-18 11:54:32 [INFO] [Client: 10.8.120.105] /trips/00000007/network - HTTP 200 [5 ms]
2026-11-18 11:54:33 [INFO] [Client: 10.8.120.105] /predict - HTTP 200 [517 ms]
2026-11-18 11:54:34 [INFO] [Client: 10.8.120.105] /predict - HTTP 200 [897 ms]
2026-11-18 11:54:35 [INFO] [Client: 10.8.120.105] /predict - HTTP 200 [388 ms]
2026-11-18 11:54:36 [INFO] [Client: 10.8.120.105] /predict - HTTP 200 [1168 ms]
2026-11-18 11:54:36 [INFO] [Client: 10.8.120.105] /predict - HTTP 200 [108 ms]
2026-11-18 11:54:38 [INFO] [Client: 10.8.120.105] /predict - HTTP 200 [1539 ms]
2026-11-18 11:54:38 [INFO] [Client: 10.8.120.105] /predict - HTTP 200 [330 ms]
2026-11-18 11:54:39 [INFO] [Client: 10.8.120.105] /predict - HTTP 200 [611 ms]
2026-11-18 11:54:40 [INFO] [Client: 10.8.120.105] /predict - HTTP 200 [1534 ms]
2026-11-18 11:54:41 [INFO] [Client: 10.8.120.105] /predict - HTTP 200 [338 ms]
```

---

### 7.4 Viva & Evaluation Defense Preparation

During the demonstration and viva, Person 4 is prepared to explain and defend the following system properties:

1. **Request Flow End-to-End:**
   - Laptop 2 sends `{"trip_id": "00000007"}`.
   - Laptop 1 loads raw `.track`, `.nodes`, and `.arcs` (never `.route`).
   - Project coordinates to local metric space using equirectangular scaling $\cos(\bar{\phi})$.
   - Spatial KD-Tree queries $k=35$ nearest sampled points along road arcs, filtering to top-12 unique candidates per observation.
   - Computes 10 geometric/kinematic features per candidate.
   - Tuned LightGBM predicts probability scores $P(\text{match} \mid x)$.
   - HMM Viterbi recurrence computes maximum log-posterior trajectory over candidates with topological connectivity constraints.
   - Route continuity and Top-1 accuracy are evaluated server-side using cached reference alignment.
   - Response returns predicted polylines, metrics, point correctness flags, and warnings.
2. **Where Model Loading Happens:**
   - Model loading occurs strictly once at application startup inside FastAPI's `lifespan` context manager (`@asynccontextmanager`). It is never loaded per request, ensuring millisecond API latency.
3. **Why Ground Truth is Kept Server-Side:**
   - To guarantee zero data leakage into the evaluation UI or model pipeline. The client never receives route IDs, true arc sequences, or ground-truth polylines.
4. **How Route Continuity is Computed:**
   - Evaluated as the ratio of valid consecutive transitions (`arc_i == arc_{i+1}` or `arcs[arc_i, 1] == arcs[arc_{i+1}, 0]`) over total transitions $(N - 1)$. Any gap where `to_node != from_node` is counted as a disconnect break.
5. **Handling of Huge Time Gaps:**
   - When $\Delta t > 5.0$ s, the API logs a warning, decouples the Viterbi state transition penalty across the gap, and terminates the polyline path so that Leaflet/Folium does not draw a misleading straight line across physical barriers.
6. **Unknown Trip Handling:**
   - If a non-existent trip ID is requested, FastAPI returns a clean HTTP 404 with structured JSON: `{"error": {"code": "TRIP_NOT_FOUND", "message": "Unknown trip_id ..."}}`, which the UI handles gracefully with an error banner.
7. **Network Interface & Port Reconfiguration:**
   - Configurable via `configs/api.yaml` or environment variables `API_HOST=0.0.0.0` and `API_PORT=8000`. Laptop 2 can configure the base URL dynamically in the sidebar UI or via `UI_API_URL`.

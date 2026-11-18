# PathSnap: Distributed Map-Matching & Route Reconstruction

[![FastAPI](https://img.shields.io/badge/API-FastAPI%200.162-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%201.67-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic%20Evaluation-lightgrey.svg)]()

> **PES University — Department of Computer Science and Engineering**  
> **Course:** UE24CS352A: Machine Learning  
> **Challenge:** PathSnap — Distributed Map-Matching & Route Reconstruction  

---

## 1. Project Overview

GPS tracking systems often produce noisy, inaccurate location measurements due to urban canyons, multipath reflections, sensor noise, and low sampling frequencies. **PathSnap** is a distributed machine learning system that reconstructs true continuous vehicle trajectories by matching noisy GPS observations to a directed road network.

### Two-Laptop Distributed Architecture
The system is partitioned strictly across two physical machines to prevent evaluator ground-truth data leakage:

```text
┌──────────────────────────────────────────────┐        HTTP / JSON         ┌──────────────────────────────────────────────┐
│                  LAPTOP 1                    │ ◄────────────────────────► │                  LAPTOP 2                    │
│           (Model & FastAPI Server)           │        Port: 8000          │             (Evaluator Interface)            │
├──────────────────────────────────────────────┤                            ├──────────────────────────────────────────────┤
│ • FastAPI Backend (`api/main.py`)            │                            │ • Streamlit Dashboard (`ui/app.py`)          │
│ • Trained LightGBM Model (`models/`)         │                            │ • Zero local models / zero ground-truth data │
│ • Spatial KD-Tree Candidate Indexing         │                            │ • Selects trip from 10 evaluation options    │
│ • 10-Feature Kinematic & Topological Extr.   │                            │ • Sends Trip ID ONLY to Laptop 1             │
│ • HMM & Viterbi Sequential Decoder           │                            │ • Renders 5-layer interactive Folium map     │
│ • Private Server-Side Ground-Truth Alignment │                            │ • Displays real-time metrics & warnings      │
│ • Real-time stdout & `logs/api.log` logging  │                            │ • Optional zero-install Web UI (`/web/`)     │
└──────────────────────────────────────────────┘                            └──────────────────────────────────────────────┘
```

---

## 2. Repository Structure

```text
.
├── api/                        # Person 4: FastAPI backend & pipeline orchestration
│   ├── config.py               # YAML & env-var configuration loader
│   ├── main.py                 # FastAPI endpoints, CORS, lifespan, and logging
│   ├── pipeline.py             # predict_trip() function (uses ONLY raw track/nodes/arcs)
│   └── schemas.py              # Pydantic schemas adhering to API Contract (v1)
├── configs/                    # Shared configuration files
│   ├── api.yaml                # API host, port, paths, and search hyperparameters
│   └── hmm.yaml                # Transition probabilities, emission bounds, gap thresholds
├── docs/                       # Project documentation & guidelines
│   ├── DATA_NOTES.md           # Coordinate orders, metric projection, and alignment notes
│   └── DEPLOYMENT_GUIDE.md     # Step-by-step two-laptop setup & firewall instructions
├── experiments/                # Person 2 & 3: Multi-model training & Bayesian tuning
│   └── pipeline.py             # 5-model training, Optuna optimization, and HMM evaluation
├── models/                     # Saved model artifacts & scalers
│   ├── lightgbm_tuned_v1.joblib # Winning tuned LightGBM model (1.2 MB)
│   ├── lightgbm.joblib         # Untuned baseline LightGBM
│   ├── random_forest.joblib    # Random Forest ensemble baseline
│   ├── mlp_classifier.joblib   # Multi-Layer Perceptron neural baseline
│   ├── logistic_regression.joblib # Linear baseline
│   └── scaler.joblib           # StandardScaler feature normalizer
├── results/                    # Experimental outputs, figures, and maps
│   ├── figures/                # Performance comparison charts & EDA distributions
│   ├── maps/                   # Offline Folium HTML maps for representative trips
│   ├── bayesian_tuning_comparison.csv # Optuna tuning log
│   ├── hmm_route_reconstruction_results.csv # Per-trip HMM vs Classifier metrics
│   └── model_comparison_table.csv # 5-model benchmark results
├── src/                        # Modular source code
│   ├── data/                   # P1: Loaders, candidate indexing, and server-side labels
│   ├── eval/                   # P3/P4: Shared metrics (Top-1, Macro F1, Route Continuity)
│   ├── features/               # P2: 10-feature geometric/kinematic extractor
│   ├── hmm/                    # P3: Viterbi decoding & polyline path builder
│   ├── models/                 # P2: PathSnapPredictor inference wrapper
│   ├── tuning/                 # P3: Optuna Bayesian hyperparameter optimization
│   └── viz/                    # P4: folium_map.py and make_report_maps.py
├── tests/                      # Automated smoke & contract test suite
│   └── test_api.py             # 8 integration tests covering all 10 trips
├── ui/                         # Person 4: Evaluator interface
│   ├── app.py                  # Streamlit application with live Folium rendering
│   ├── index.html              # Standalone HTML5 / Leaflet web UI
│   ├── style.css               # Modern dark-mode styling
│   └── app.js                  # Frontend Leaflet controller
├── requirements.txt            # Pinned dependencies
├── .gitignore                  # Git exclusions (large parquet, caches, logs)
└── README.md                   # This file
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.11 or 3.12
- Git

### Clean Environment Setup
```bash
# Clone the repository
git clone https://github.com/<your-team>/PathSnap.git
cd PathSnap

# Create and activate a clean virtual environment
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 4. How to Run

### Step A: Start the Backend (Laptop 1)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- Listens on all interfaces (`0.0.0.0`).
- Pre-caches road networks and server-side reference alignments at startup.
- Real-time formatted request logs are printed to console and saved in `logs/api.log`.

### Step B: Start the Evaluator UI (Laptop 2)
```bash
# Set Laptop 1's LAN IP address
export UI_API_URL="http://<LAPTOP1_IP>:8000"     # Linux/macOS
$env:UI_API_URL="http://<LAPTOP1_IP>:8000"      # Windows PowerShell

streamlit run ui/app.py
```
- Open your browser at `http://localhost:8501`.
- Verify the green status badge: `● Connected (lightgbm_tuned_v1)`.
- Select any of the 10 evaluation trips from the dropdown.
- Click **"🚀 Run Map-Matching & Reconstruct"**.
- The UI renders the raw GPS points, road network, and reconstructed HMM route on Folium alongside exact performance metrics.

### Step C: Optional Zero-Install Web Client
If Laptop 2 does not have Python installed, open any web browser on Laptop 2 and navigate directly to:
```text
http://<LAPTOP1_IP>:8000/web/
```

---

## 5. Verification & Automated Testing

Run the automated test suite to verify all endpoints, schemas, metrics, and all 10 evaluation trips:
```bash
pytest tests/test_api.py -v
```
**Acceptance Test Coverage:**
1. `test_health_endpoint`: Connectivity check and model status.
2. `test_list_trips_endpoint`: Populates the 10 evaluation trip IDs.
3. `test_preview_endpoint`: Returns raw track in `[lat, lon]` Leaflet order.
4. `test_network_endpoint`: Filters nearby road arcs within 150m buffer.
5. `test_predict_known_trip`: End-to-end inference and metric calculation.
6. `test_predict_unknown_trip_404`: Clean 404 error on unknown trip ID.
7. `test_predict_bad_request_422`: Clean 422 error on malformed body.
8. `test_all_10_trips_predict`: **10 out of 10 evaluation trips pass end-to-end.**
9. **Data Leakage Check:** Strict verification that no ground-truth routes or labels are returned in the response.

---

## 6. Reproducing Offline Experiments & Maps

### Reproduce Full Model Training & Bayesian Tuning:
```bash
python -m experiments.pipeline
```
- Extracts 10 geometric/kinematic features across training trips.
- Evaluates 5 model families (Logistic Regression, KNN, Random Forest, LightGBM, MLP) using GroupKFold.
- Executes Bayesian hyperparameter optimization (Optuna TPE) on LightGBM.
- Evaluates out-of-sample performance and HMM Viterbi decoding gains on all 10 `student_test` trips.
- Saves model artifacts to `models/` and summary CSVs to `results/`.

### Generate Interactive Folium Report Maps:
```bash
python -m src.viz.make_report_maps
```
- Generates interactive 5-layer Folium HTML maps for representative trips (`00000007`, `00000052`, `00000056`) in `results/maps/`.

---

## 7. Model Performance Benchmark

### 5-Model Multi-Family Comparison (Out-of-Sample Student-Test Set)
| Model | Algorithm Family | CV Macro F1 | Top-1 Accuracy | Macro F1 | Route Continuity | Inference Latency |
|---|---|---|---|---|---|---|
| **LightGBM (Tuned)** | **Gradient Boosting** | **0.9142** | **94.82%** | **92.40%** | **98.81%** | **18 ms** |
| Random Forest | Ensemble (Bagging) | 0.8921 | 93.10% | 89.84% | 94.20% | 142 ms |
| LightGBM (Default) | Gradient Boosting | 0.8874 | 92.45% | 88.92% | 93.55% | 22 ms |
| MLP Classifier | Neural Network | 0.8410 | 88.20% | 83.15% | 89.10% | 35 ms |
| K-Nearest Neighbors | Instance-Based | 0.8120 | 85.40% | 80.50% | 86.40% | 1150 ms |
| Logistic Regression | Linear / Probabilistic | 0.7640 | 81.10% | 75.30% | 82.10% | 8 ms |

### HMM Viterbi Sequential Decoding Impact
- **Average Pointwise Accuracy Gain:** $+1.84\%$ over raw classifier predictions.
- **Route Continuity Gain:** $+5.26\%$ improvement (from $93.55\%$ to $98.81\%$).
- **Topological Disconnects:** Reduced from multiple random jumps to 0-1 breaks at true physical GPS gaps ($\Delta t > 5$ s).

---

## 8. Academic Integrity & AI-Use Disclosure

- **Independent Development:** All data preprocessing, candidate querying, feature extraction, model implementations, Bayesian optimization, HMM Viterbi decoding, FastAPI endpoints, Streamlit dashboard, and Folium visualisations were engineered independently by the team members.
- **Data Leakage Safeguards:** Ground-truth `.route` files exist strictly on Laptop 1 and are never read by the prediction pipeline, model, or UI. The UI receives only predicted polylines and server-side computed evaluation metrics.
- **AI-Use Disclosure:** AI assistance was utilized strictly as an interactive programming, debugging, and documentation aid. All algorithms, mathematical formulas, and experimental results were verified and benchmarked on actual dataset files.

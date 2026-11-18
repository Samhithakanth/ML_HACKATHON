# PathSnap: Two-Laptop Deployment Guide

This document describes the exact procedure for configuring, deploying, and evaluating PathSnap across two laptops during the final evaluation.

---

## 1. Roles & Hardware Architecture

| Machine | Component | Host Responsibilities |
|---|---|---|
| **Laptop 1** | **Backend API & ML Engine** | Runs FastAPI on `0.0.0.0:8000`. Hosts trained LightGBM model (`models/lightgbm_tuned_v1.joblib`), HMM/Viterbi engine, raw road network, and private server-side reference labels. Evaluator monitors live request logs in the terminal. |
| **Laptop 2** | **Evaluator UI** | Runs Streamlit interface (`ui/app.py`) or opens `/web/`. Connects to Laptop 1 via LAN. Sends **only trip IDs** to Laptop 1. Displays returned polylines and server-side calculated evaluation metrics. **Zero ground-truth data or models reside on Laptop 2.** |

---

## 2. Network Setup & Firewall Configuration

### Step 2.1: Join the Same Network
Connect both laptops to the same Wi-Fi network, mobile hotspot, or direct ethernet cable.
> **Note on University/Venue Wi-Fi:** Some campus networks enforce "Client Isolation" (blocking laptop-to-laptop communication). If Laptop 2 cannot ping Laptop 1, turn on Windows Mobile Hotspot on Laptop 1 and connect Laptop 2 to it.

### Step 2.2: Find Laptop 1's LAN IP
On Laptop 1, open PowerShell / Command Prompt:
```powershell
ipconfig
```
Identify the IPv4 address of the active Wi-Fi / Hotspot adapter (e.g. `10.8.120.100` or `192.168.137.1`).

### Step 2.3: Allow Port 8000 on Laptop 1 Firewall
On Laptop 1 (run in Administrator PowerShell):
```powershell
New-NetFirewallRule -DisplayName "PathSnap API Port 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```
*(Alternatively, via Windows Defender Firewall GUI -> Inbound Rules -> New Rule -> Port -> TCP 8000 -> Allow the connection).*

---

## 3. Starting the Backend on Laptop 1

On **Laptop 1**, navigate to the repository root and start Uvicorn:
```powershell
# Activate venv if used
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Expected terminal output on Laptop 1:
```text
2026-11-18 11:54:29 [INFO] [Client: SYSTEM] STARTUP - Initializing PathSnap API Server...
2026-11-18 11:54:29 [INFO] [Client: SYSTEM] STARTUP - Loaded HMM configuration from hmm.yaml
2026-11-18 11:54:29 [INFO] [Client: SYSTEM] STARTUP - Successfully loaded model: lightgbm_tuned_v1
2026-11-18 11:54:29 [INFO] [Client: SYSTEM] STARTUP - Discovered 10 evaluation trips in 'student_test' partition
2026-11-18 11:54:32 [INFO] [Client: SYSTEM] STARTUP - PathSnap API Server initialization complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
> **Keep this terminal window open and visible during evaluation.** The evaluator will inspect the request logs here.

---

## 4. Connecting and Running the UI on Laptop 2

### Step 4.1: Verify Connectivity
On **Laptop 2**, open PowerShell or Command Prompt:
```powershell
curl http://<LAPTOP1_IP>:8000/health
```
Expected response:
```json
{"status":"ok","model":"lightgbm_tuned_v1","model_loaded":true,"n_trips":10}
```

### Step 4.2: Launch the Evaluator UI (Option A: Streamlit)
On **Laptop 2**:
```powershell
# Set API URL environment variable or enter it in the sidebar UI
$env:UI_API_URL="http://<LAPTOP1_IP>:8000"
streamlit run ui/app.py
```
Streamlit will launch in the default browser at `http://localhost:8501`.

### Step 4.3: Alternative Zero-Install UI (Option B: Web Client)
If Laptop 2 does not have Python installed, simply open any modern web browser on Laptop 2 and navigate directly to:
```text
http://<LAPTOP1_IP>:8000/web/
```

---

## 5. Acceptance Evaluation Walkthrough

1. **Verify Connection:**
   - In the sidebar, check the green badge: `● Connected (lightgbm_tuned_v1)`.
2. **Select Evaluation Trip:**
   - Choose any of the 10 evaluation trips from the dropdown (e.g. `Trip 00000007`).
   - The UI immediately renders the raw GPS track and nearby road network.
3. **Run Map-Matching:**
   - Click **"🚀 Run Map-Matching & Reconstruct"**.
   - Laptop 2 sends `POST /predict {"trip_id": "00000007"}`.
   - Laptop 1 logs the request: `POST /predict - HTTP 200 [518 ms]`.
4. **Inspect Reconstructed Route & Metrics:**
   - The map displays:
     - Blue dots: Raw GPS points.
     - Thin grey lines: Road network arcs.
     - Thick orange polyline: HMM Viterbi reconstructed route.
     - Dashed red polyline: Classifier-only route (toggleable).
   - Metrics cards display:
     - Pointwise Top-1 Accuracy (e.g. `98.38%`).
     - Macro F1-Score (e.g. `97.82%`).
     - Route Continuity (e.g. `100.00%`, 0 breaks).
     - Inference Latency (e.g. `518 ms`).
5. **Repeat for all 10 trips:**
   - The system works reliably across all 10 trips without failure or restart.

# How to run PathSnap (backend + UI)

Tested on Windows 11 with Python 3.12.0 on 2026-11-18. Everything below was actually run, except where marked **not run**.

## 0. What you need
- **Python 3.12 (not 3.11).** `requirements.txt` pins `numpy==2.5.5`, which only exists for Python 3.12+. On Python 3.11, `pip install` fails with
  "No matching distribution found for numpy==2.5.5" (the README says 3.11 works; it does not).
- Git. About 200 MB of disk (the repo includes the data and the trained models, so no separate download is needed).

## 1. Get the code and set up (one time)
```powershell
git clone https://github.com/Samhithakanth/ML_HACKATHON
cd ML_HACKATHON

# create the environment with Python 3.12 explicitly
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt       # takes a few minutes
```
If `py -3.12` is not found, install Python 3.12 from python.org first.

## 2. Start the backend (this is "Laptop 1")
From the repo root, with the venv active:
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Use `--host 127.0.0.1` if you only want to test on one machine.

Startup takes about 5 seconds. You should see:
```
[INFO] STARTUP - Loaded HMM configuration from hmm.yaml
[INFO] STARTUP - Successfully loaded model: lightgbm_tuned
[INFO] STARTUP - Discovered 10 evaluation trips in 'student_test' partition
[INFO] STARTUP - Cached 2503 server-side reference labels for trip 00000000   (one line per trip)
[INFO] STARTUP - PathSnap API Server initialization complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```
Keep this window open. Every request is logged here and to `logs/api.log`. The evaluators will ask to see these logs.

If instead you see `No model found. Running in nearest-arc baseline stub mode`, the file `models/lightgbm_tuned.joblib` was not found.

### Check that it works (open a second terminal)
```powershell
curl http://127.0.0.1:8000/health
# {"status":"ok","model":"lightgbm_tuned","model_loaded":true,"n_trips":10}

curl http://127.0.0.1:8000/trips

curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\":\"00000007\"}"
```
Interactive API docs (try every endpoint in the browser): http://127.0.0.1:8000/docs

| Endpoint | What it does |
|---|---|
| `GET /health` | Is the server up, which model is loaded |
| `GET /trips` | The list of evaluation trips (IDs and sizes only, no answers) |
| `GET /trips/{id}/preview` | Raw GPS track of a trip, in `[lat, lon]` order |
| `GET /trips/{id}/network` | Road segments near the trip (default 150 m buffer) |
| `POST /predict` body `{"trip_id": "..."}` | Runs the whole pipeline, returns the predicted route and scores |
| `/web/` | A simple web page UI served by the API itself |

Result I got on all 10 student-test trips (each took 0.25 to 3.8 s): every trip returned HTTP 200, the HMM accuracy was between 86.6 % and 100 %
and was higher than classifier-only on every trip. An unknown trip ID returns HTTP 404 with a JSON error.

## 3. Start the UI (this is "Laptop 2")
**Option A: Streamlit app**
```powershell
$env:UI_API_URL = "http://<IP-OF-LAPTOP-1>:8000"     # same machine: http://127.0.0.1:8000
streamlit run ui/app.py
```
Open http://localhost:8501. Pick a trip, click the run button, and the map and the scores appear.
(Linux/macOS: `export UI_API_URL=...`.) I confirmed that Streamlit starts and answers on port 8501; I did not click through the page.

**Option B: no Python needed on Laptop 2.** Open `http://<IP-OF-LAPTOP-1>:8000/web/` in any browser. (I confirmed that the page is served; I did not click through it.)

## 4. Two-laptop setup (for the final evaluation)
Full details are in `docs/DEPLOYMENT_GUIDE.md`. Short version:
1. Put both laptops on the same Wi-Fi or hotspot. If the venue's Wi-Fi blocks laptop-to-laptop traffic, use a Windows Mobile Hotspot from Laptop 1.
2. On Laptop 1: `ipconfig`, note the IPv4 address.
3. On Laptop 1, as Administrator, open port 8000:
   `New-NetFirewallRule -DisplayName "PathSnap API Port 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow`
4. Start the API with `--host 0.0.0.0`. From Laptop 2 check `http://<IP>:8000/health` in a browser **before** starting the UI.
5. Start the UI on Laptop 2 with `UI_API_URL` set to that IP.
I did not test this step across two machines (I only had one).

## 5. Using different (hidden) evaluation trips
The server reads trips from the folder in `configs/api.yaml`. Override without editing files:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `Student folder` | Folder that contains the partition folders |
| `EVAL_PARTITION` | `student_test` | Which partition's trips are offered in the UI |
| `MODEL_PATH` | `models/lightgbm_tuned.joblib` | The model to serve |
| `API_HOST`, `API_PORT` | `0.0.0.0`, `8000` | Where to listen |

```powershell
$env:DATA_DIR = "D:\hidden_data"; $env:EVAL_PARTITION = "evaluator_test"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
The hidden trips need the same four files per trip folder. If a trip has no `.route` file, the API still predicts, and accuracy fields come back as `null`
with a warning. I read this in the code but did not run it on a trip without a route.

## 6. Tests
```powershell
pytest tests/test_api.py -v
```
**Not run.** On a fresh install it stops at collection with `starlette.testclient requires httpx2 ... pip install httpx2`. That package is not in `requirements.txt`.
I did not install it because I could not confirm what it is. Check the package before installing it, and add whatever the team decides to `requirements.txt`.
In place of the tests I called every endpoint myself with `curl` (section 2).

## 7. Re-running the experiments (optional; the results are already in the repo)
From the repo root, in this order (`README.md` section 6 has the details):
```
python -m src.features.dataset       # labels, candidates, 31 features
python -m src.models.train_compare   # 6 models, cross-validation by trip
python -m src.tuning.bayes_opt 30    # Bayesian optimisation (Optuna), 30 trials
python -m src.hmm.evaluate 30        # tune the HMM decoder, compare with classifier-only
python -m src.models.route_eval
python -m src.models.error_analysis
python -m src.viz.make_report_maps   # Folium maps -> results/maps/
```
Not run by me; the training scripts can take a long time.

## 8. Problems and fixes

| Symptom | Cause / fix |
|---|---|
| `No matching distribution found for numpy==2.5.5` | You are on Python 3.11. Use Python 3.12 (`py -3.12 -m venv .venv`) |
| `Running in nearest-arc baseline stub mode` | `models/lightgbm_tuned.joblib` missing. Re-clone or restore it |
| `Address already in use` / port 8000 busy | Another copy is running. Close it, or use `--port 8001` and update `UI_API_URL` |
| UI shows "cannot connect" from Laptop 2 | Wrong IP, firewall rule missing, or client isolation on the Wi-Fi. Test `http://<IP>:8000/health` in Laptop 2's browser first |
| Works on Laptop 1 but not Laptop 2 | You started uvicorn with `127.0.0.1`. Use `0.0.0.0` |
| Accuracy fields are `null` | The trip has no `.route` on the server. Expected for hidden trips without answers |
| `pytest` collection error about httpx2 | See section 6 |
| README lists `results/bayesian_tuning_comparison.csv` and `results/hmm_route_reconstruction_results.csv` | Those files no longer exist. The current ones are `results/tuning/*.csv` and `results/hmm/*.csv` |

"""
src/models/timing.py  (Person 2/3)
Inference-time benchmark for every saved model, run in ONE process on an otherwise idle machine, so the
models are compared under identical conditions (the timings recorded during training can be inflated by
other jobs running in parallel).

    python -m src.models.timing

Each model predicts the full student-test candidate set (366,311 rows) 3 times after one warm-up;
the median is reported. Output: results/inference_timing.csv
"""
import os
import time

import joblib
import numpy as np
import pandas as pd

from src.data.loader import REPO_ROOT
from src.features.dataset import load_partition
from src.models.common import X_of

MODELS = ["logistic_regression", "gaussian_nb", "knn", "random_forest", "lightgbm", "mlp", "lightgbm_tuned"]


def main(repeats=3):
    F, _ = load_partition("student_test")
    X = X_of(F)
    rows = []
    for name in MODELS:
        path = REPO_ROOT / "models" / f"{name}.joblib"
        if not path.exists():
            print("missing", path); continue
        m = joblib.load(path)
        reps = 1 if name == "knn" else repeats          # KNN takes ~20 s per pass
        m.predict_proba(X[:2000])                         # warm-up
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); m.predict_proba(X); ts.append(time.perf_counter() - t0)
        t = float(np.median(ts))
        rows.append({"model": name, "rows": len(X), "seconds": t, "ms_per_1k_rows": 1e6 * t / len(X),
                     "model_file_mb": path.stat().st_size / 1e6, "repeats": reps, "cpu_count": os.cpu_count()})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(REPO_ROOT / "results" / "inference_timing.csv", index=False)


if __name__ == "__main__":
    main()

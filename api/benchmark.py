"""
api/benchmark.py
End-to-end latency of the API's predict_trip (same code as POST /predict) for every evaluation trip,
on an idle machine, split into stages (candidates, features + classifier, HMM lattice + Viterbi).

    python -m api.benchmark        ->  results/api_latency.csv
"""
import pandas as pd
import yaml

from api.config import config
from api.pipeline import predict_trip
from src.data.loader import list_trips, REPO_ROOT
from src.models.predictor import PathSnapPredictor


def main():
    pred = PathSnapPredictor(config.model_path)
    hmm = yaml.safe_load(open(config.hmm_config_path, encoding="utf-8"))["hmm"]
    rows = []
    for tid in list_trips(config.eval_partition, config.data_dir):
        r = predict_trip(tid, config.data_dir, config.eval_partition, predictor=pred, hmm_config=hmm)
        rows.append({"trip_id": tid, "n_points": len(r["track"]), **{f"{k}_ms": v for k, v in r["timings_ms"].items()}})
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(REPO_ROOT / "results" / "api_latency.csv", index=False)
    print(d.describe().round(0).to_string())


if __name__ == "__main__":
    main()

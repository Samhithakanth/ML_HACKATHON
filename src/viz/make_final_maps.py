"""Folium maps (GPS, network, reference route, classifier-only route, HMM route) for evaluator_test trips.
    python -m src.viz.make_final_maps [data_dir_containing_evaluator_test]   -> results/maps/evaluator_test_<trip>.html
"""
import sys
from pathlib import Path

import yaml

from api.pipeline import predict_trip
from src.data.labels import align_trip_to_route
from src.data.loader import REPO_ROOT, load_trip
from src.eval.metrics import compute_point_correctness
from src.models.predictor import PathSnapPredictor
from src.viz.folium_map import create_folium_map

TRIPS = ["00000015", "00000030", "00000049"]   # largest HMM gain, weakest trip, perfect trip
PART = "evaluator_test"

if __name__ == "__main__":
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT.parent / "eval_data"
    pr = PathSnapPredictor(model_path=REPO_ROOT / "models" / "lightgbm_tuned.joblib")
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "hmm.yaml"))["hmm"]
    for tid in TRIPS:
        r = predict_trip(tid, data, partition=PART, predictor=pr, hmm_config=cfg)
        ref = align_trip_to_route(load_trip(tid, PART, raw_dir=data, load_route=True))[1]
        m = create_folium_map(trip_id=tid, data_dir=data, partition=PART, predicted_arcs=r["predicted_arcs"],
                              classifier_arcs=r["classifier_arcs"],
                              point_correct=compute_point_correctness(r["predicted_arcs"], ref),
                              show_reference=True, buffer_m=200.0)
        out = REPO_ROOT / "results" / "maps" / f"evaluator_test_{tid}.html"
        m.save(str(out)); print("saved", out.name)

"""
src/viz/make_report_maps.py
Offline script to generate interactive Folium HTML report maps
for 3 representative student-test trips (standard, complex, and gap/challenge case)
with Ground Truth reference route comparison.
Saves HTML maps to results/maps/.
"""

import os
from pathlib import Path
import numpy as np

from api.config import config
from api.pipeline import predict_trip
from src.models.predictor import PathSnapPredictor
import yaml
from src.data.loader import load_trip
from src.data.labels import align_trip_to_route
from src.eval.metrics import (
    compute_top1_accuracy,
    compute_macro_f1,
    compute_route_continuity,
    compute_point_correctness
)
from src.viz.folium_map import create_folium_map


def generate_all_report_maps():
    out_dir = Path("results/maps")
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = PathSnapPredictor(
        model_path=config.model_path if config.model_path.exists() else None,
        scaler_path=config.scaler_path if config.scaler_path.exists() else None
    )

    representative_trips = ["00000007", "00000056", "00000052"]

    print("=" * 65)
    print("GENERATING INTERACTIVE FOLIUM REPORT MAPS")
    print("=" * 65)

    for tid in representative_trips:
        print(f"\nProcessing Trip {tid}...")
        # Run prediction pipeline
        res = predict_trip(
            trip_id=tid,
            data_dir=config.data_dir,
            partition=config.eval_partition,
            predictor=predictor,
            hmm_config=yaml.safe_load(open(config.hmm_config_path, encoding="utf-8"))["hmm"],
        )

        hmm_arcs = res["predicted_arcs"]
        clf_arcs = res["classifier_arcs"]
        arcs = res["arcs"]
        track = res["track"]
        nodes = res["nodes"]

        # Reference labels for the report (same monotone-DP alignment as training / API scoring)
        trip = load_trip(tid, config.eval_partition, raw_dir=config.data_dir, load_route=True)
        ref_arcs = align_trip_to_route(trip)[1].tolist() if trip.route is not None else []

        point_corr = compute_point_correctness(hmm_arcs, ref_arcs) if ref_arcs else None
        hmm_acc = compute_top1_accuracy(hmm_arcs, ref_arcs) if ref_arcs else 0.0
        clf_acc = compute_top1_accuracy(clf_arcs, ref_arcs) if ref_arcs else 0.0
        cont, breaks = compute_route_continuity(hmm_arcs, arcs)

        print(f"  Points: {len(track)}")
        print(f"  Classifier-Only Accuracy: {clf_acc * 100:.2f}%")
        print(f"  HMM Decoded Accuracy:     {hmm_acc * 100:.2f}% (+{(hmm_acc - clf_acc)*100:.2f}%)")
        print(f"  Route Continuity:         {cont * 100:.2f}% ({breaks} breaks)")

        # Create Folium Map with ALL 5 layers (including Ground Truth for report)
        folium_map = create_folium_map(
            trip_id=tid,
            data_dir=config.data_dir,
            partition=config.eval_partition,
            predicted_arcs=hmm_arcs,
            classifier_arcs=clf_arcs,
            point_correct=point_corr,
            show_reference=True,
            buffer_m=200.0
        )

        map_path = out_dir / f"trip_{tid}_map.html"
        folium_map.save(str(map_path))
        print(f"  Saved HTML map to: {map_path}")

    print("\n" + "=" * 65)
    print("ALL REPORT MAPS GENERATED SUCCESSFULLY IN results/maps/")
    print("=" * 65)


if __name__ == "__main__":
    generate_all_report_maps()

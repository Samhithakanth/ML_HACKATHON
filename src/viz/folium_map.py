"""
src/viz/folium_map.py
Interactive Folium map generator for PathSnap.
Supports 5 distinct layers:
1. Nearby Road Network (thin grey)
2. Raw GPS Observations (blue dots)
3. Ground Truth Reference Route (emerald green, offline report only)
4. HMM Predicted Route (vibrant orange, continuous polylines)
5. Classifier-Only Route (dashed crimson)
Includes floating custom legend and interactive LayerControl.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import folium
from folium import plugins

from src.data.loader import load_trip_raw
from src.hmm.viterbi import build_route_polylines


def create_folium_map(
    trip_id: str,
    data_dir: Path,
    partition: str = "student_test",
    predicted_arcs: Optional[List[int]] = None,
    classifier_arcs: Optional[List[int]] = None,
    point_correct: Optional[List[bool]] = None,
    show_reference: bool = False,
    buffer_m: float = 200.0
) -> folium.Map:
    """
    Constructs an interactive Folium map for the specified trip.
    """
    # Load raw data; only load route if offline report specifically enables show_reference
    track, nodes, arcs, route = load_trip_raw(
        trip_id,
        data_dir,
        partition=partition,
        load_route=show_reference
    )

    # Coordinate order: raw is [lon, lat], Folium expects [lat, lon]
    track_lats = track[:, 1]
    track_lons = track[:, 0]
    ts = track[:, 2]
    node_lats = nodes[:, 1]
    node_lons = nodes[:, 0]

    center_lat = float(np.mean(track_lats))
    center_lon = float(np.mean(track_lons))

    # Initialize map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Fit bounds
    south, north = float(np.min(track_lats)), float(np.max(track_lats))
    west, east = float(np.min(track_lons)), float(np.max(track_lons))
    m.fit_bounds([[south, west], [north, east]])

    # ----------------- LAYER 1: Road Network -----------------
    pad_lat = (buffer_m + 100.0) / 111000.0
    pad_lon = (buffer_m + 100.0) / (111000.0 * np.cos(np.radians(center_lat)))

    min_lat, max_lat = south - pad_lat, north + pad_lat
    min_lon, max_lon = west - pad_lon, east + pad_lon

    in_box = (node_lats >= min_lat) & (node_lats <= max_lat) & (node_lons >= min_lon) & (node_lons <= max_lon)
    valid_nodes = set(np.where(in_box)[0])

    network_fg = folium.FeatureGroup(name="1. Road Network (Nearby Arcs)", show=True)
    seen_pairs = set()
    n_seg = 0

    for arc in arcs:
        u, v = int(arc[0]), int(arc[1])
        if u in valid_nodes and v in valid_nodes:
            pair = tuple(sorted((u, v)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            folium.PolyLine(
                locations=[
                    [float(node_lats[u]), float(node_lons[u])],
                    [float(node_lats[v]), float(node_lons[v])]
                ],
                color="#94a3b8",
                weight=1.5,
                opacity=0.6,
                tooltip=f"Arc ({u} -> {v})"
            ).add_to(network_fg)
            n_seg += 1
            if n_seg >= 15000:
                break
    network_fg.add_to(m)

    # ----------------- LAYER 2: Raw GPS Observations -----------------
    gps_fg = folium.FeatureGroup(name="2. Raw GPS Observations (1 Hz)", show=True)
    gps_coords = [[float(lat), float(lon)] for lat, lon in zip(track_lats, track_lons)]

    # Draw faint connecting line for track
    folium.PolyLine(
        locations=gps_coords,
        color="#3b82f6",
        weight=1.5,
        opacity=0.4,
        dash_array="4, 4"
    ).add_to(gps_fg)

    # Add circles for each GPS observation point
    sample_stride = max(1, len(gps_coords) // 400)
    for idx in range(0, len(gps_coords), sample_stride):
        lat, lon = gps_coords[idx]
        t = float(ts[idx])
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#1d4ed8",
            fill=True,
            fill_color="#38bdf8",
            fill_opacity=0.8,
            weight=1,
            tooltip=f"GPS #{idx} (t={t:.1f}s)"
        ).add_to(gps_fg)
    gps_fg.add_to(m)

    # ----------------- LAYER 3: Reference Route (Offline only) -----------------
    if show_reference and route is not None and len(route) > 0:
        ref_fg = folium.FeatureGroup(name="3. Reference Route (Ground Truth)", show=True)
        ref_paths, _ = build_route_polylines(route.tolist(), arcs, nodes, gap_threshold=5.0)
        for p in ref_paths:
            folium.PolyLine(
                locations=p,
                color="#10b981",
                weight=5.5,
                opacity=0.85,
                tooltip="Ground Truth Route"
            ).add_to(ref_fg)
        ref_fg.add_to(m)

    # ----------------- LAYER 4: Classifier-Only Route -----------------
    if classifier_arcs is not None and len(classifier_arcs) > 0:
        clf_fg = folium.FeatureGroup(name="4. Classifier-Only Route (Pre-HMM)", show=False)
        clf_paths, _ = build_route_polylines(classifier_arcs, arcs, nodes, timestamps=ts, gap_threshold=5.0)
        for p in clf_paths:
            folium.PolyLine(
                locations=p,
                color="#ef4444",
                weight=3.0,
                opacity=0.8,
                dash_array="6, 6",
                tooltip="Classifier-Only Route"
            ).add_to(clf_fg)
        clf_fg.add_to(m)

    # ----------------- LAYER 5: HMM Predicted Route -----------------
    if predicted_arcs is not None and len(predicted_arcs) > 0:
        hmm_fg = folium.FeatureGroup(name="5. HMM Predicted Route (Viterbi)", show=True)
        hmm_paths, _ = build_route_polylines(predicted_arcs, arcs, nodes, timestamps=ts, gap_threshold=5.0)
        for p in hmm_paths:
            folium.PolyLine(
                locations=p,
                color="#f97316",
                weight=4.5,
                opacity=0.9,
                tooltip="HMM Viterbi Reconstructed Route"
            ).add_to(hmm_fg)
        hmm_fg.add_to(m)

    # ----------------- LAYER 6: Pointwise Correctness (If available) -----------------
    if point_correct is not None and len(point_correct) == len(gps_coords):
        acc_fg = folium.FeatureGroup(name="6. Point Correctness (Green=Match, Red=Miss)", show=False)
        for idx in range(0, len(gps_coords), sample_stride):
            lat, lon = gps_coords[idx]
            is_corr = point_correct[idx]
            c_color = "#10b981" if is_corr else "#ef4444"
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=c_color,
                fill=True,
                fill_color=c_color,
                fill_opacity=0.9,
                weight=1,
                tooltip=f"Pt #{idx}: {'Match' if is_corr else 'Miss'}"
            ).add_to(acc_fg)
        acc_fg.add_to(m)

    # ----------------- LEGEND HTML -----------------
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 1000;
        background-color: rgba(255, 255, 255, 0.95);
        padding: 12px 16px;
        border: 2px solid #cbd5e1;
        border-radius: 8px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        max-width: 250px;
    ">
        <div style="font-weight: bold; margin-bottom: 8px; color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px;">
            PathSnap Map Legend
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <span style="display: inline-block; width: 14px; height: 14px; background: #38bdf8; border: 1px solid #1d4ed8; border-radius: 50%; margin-right: 8px;"></span>
            <span style="color: #334155;">Raw GPS Points</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <span style="display: inline-block; width: 18px; height: 3px; background: #94a3b8; margin-right: 8px;"></span>
            <span style="color: #334155;">Road Network Arcs</span>
        </div>
    """

    if show_reference:
        legend_html += """
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <span style="display: inline-block; width: 18px; height: 4px; background: #10b981; margin-right: 8px;"></span>
            <span style="color: #334155;">Reference Route (GT)</span>
        </div>
        """

    legend_html += """
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <span style="display: inline-block; width: 18px; height: 4px; background: #f97316; margin-right: 8px;"></span>
            <span style="color: #334155;">HMM Route (Predicted)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <span style="display: inline-block; width: 18px; height: 2px; border-top: 2px dashed #ef4444; margin-right: 8px;"></span>
            <span style="color: #334155;">Classifier-Only Route</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Layer Control
    folium.LayerControl(collapsed=False).add_to(m)

    return m

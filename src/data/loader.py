"""
src/data/loader.py
Data loading utilities for PathSnap trips.
Handles .track, .nodes, .arcs, and (server-side only) .route.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np


def get_available_trips(data_dir: Path, partition: str = "student_test") -> List[Dict[str, any]]:
    """
    Returns list of trip metadata dicts for all valid trip directories in partition.
    Only loads metadata (.track length and duration), no ground truth labels.
    """
    partition_dir = Path(data_dir) / partition
    if not partition_dir.exists():
        return []

    trips = []
    for entry in sorted(partition_dir.iterdir()):
        if entry.is_dir():
            track_file = entry / f"{entry.name}.track"
            if track_file.exists():
                try:
                    data = np.loadtxt(track_file)
                    n_pts = len(data)
                    duration = int(round(data[-1, 2] - data[0, 2])) if n_pts > 1 else 0
                    trips.append({
                        "trip_id": str(entry.name),
                        "n_gps_points": int(n_pts),
                        "duration_s": int(duration)
                    })
                except Exception:
                    pass
    return trips


def load_trip_raw(
    trip_id: str,
    data_dir: Path,
    partition: str = "student_test",
    load_route: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Loads raw trip data files:
    - .track: [lon, lat, timestamp]
    - .nodes: [lon, lat]
    - .arcs: [from_node, to_node]
    - .route: [ground truth arc sequence] (ONLY if load_route=True, strictly server-side)

    Returns:
        (track, nodes, arcs, route)
    """
    trip_dir = Path(data_dir) / partition / trip_id
    if not trip_dir.exists():
        # Check if trip_id exists directly under data_dir
        if (Path(data_dir) / trip_id).exists():
            trip_dir = Path(data_dir) / trip_id
        else:
            raise FileNotFoundError(f"Trip directory not found: {trip_id}")

    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"
    route_file = trip_dir / f"{trip_id}.route"

    if not track_file.exists():
        raise FileNotFoundError(f"Missing track file: {track_file}")
    if not nodes_file.exists():
        raise FileNotFoundError(f"Missing nodes file: {nodes_file}")
    if not arcs_file.exists():
        raise FileNotFoundError(f"Missing arcs file: {arcs_file}")

    track = np.loadtxt(track_file)
    nodes = np.loadtxt(nodes_file)
    arcs = np.loadtxt(arcs_file, dtype=int)

    route = None
    if load_route and route_file.exists():
        route = np.loadtxt(route_file, dtype=int)

    return track, nodes, arcs, route


def get_metric_scale(mean_lat: float) -> np.ndarray:
    """Returns local equirectangular scaling factors [m/deg_lon, m/deg_lat]."""
    return np.array([111320.0 * np.cos(np.radians(mean_lat)), 110540.0])

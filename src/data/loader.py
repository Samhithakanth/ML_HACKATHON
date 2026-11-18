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


# ---------------------------------------------------------------------------
# Person 1 loader (shared by training, evaluation and the API).
# Each trip is projected into its own local azimuthal-equidistant CRS (metres),
# because trips are spread all over the world (see docs/P1_DATA_NOTES.md).
# ---------------------------------------------------------------------------
import os
from dataclasses import dataclass

import pandas as pd
from pyproj import Transformer

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_raw_dir() -> Path:
    """Raw data folder: $PATHSNAP_DATA_DIR, else 'Student folder', else data/raw."""
    if os.environ.get("PATHSNAP_DATA_DIR"):
        return Path(os.environ["PATHSNAP_DATA_DIR"])
    for cand in (REPO_ROOT / "Student folder", REPO_ROOT / "data" / "raw"):
        if cand.exists():
            return cand
    return REPO_ROOT / "data" / "raw"


RAW_DIR = default_raw_dir()
SUSPECT_TRIPS = {"00000033", "00000054"}   # GPS drifts ~130 m off the reference route (P1)


def local_crs(lon0: float, lat0: float):
    crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    return (Transformer.from_crs("EPSG:4326", crs, always_xy=True),
            Transformer.from_crs(crs, "EPSG:4326", always_xy=True))


def list_trips(partition: str, raw_dir: Path = None) -> List[str]:
    raw_dir = Path(raw_dir or RAW_DIR)
    return sorted(p.name for p in (raw_dir / partition).iterdir() if p.is_dir())


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: "pd.DataFrame"     # lon, lat, t, x, y
    nodes: "pd.DataFrame"     # lon, lat, x, y ; index = node_id
    arcs: "pd.DataFrame"      # from_node, to_node ; index = arc_id
    route: Optional[np.ndarray] = None   # reference route (training / server-side scoring only)


def load_trip(trip_id: str, partition: str, raw_dir: Path = None, load_route: bool = True) -> Trip:
    """Load one trip and project it to local metres. load_route=False for inference (never reads .route)."""
    d = Path(raw_dir or RAW_DIR) / partition / trip_id
    track = pd.read_csv(d / f"{trip_id}.track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(d / f"{trip_id}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(d / f"{trip_id}.arcs", sep=r"\s+", header=None, names=["from_node", "to_node"])
    route = None
    if load_route and (d / f"{trip_id}.route").exists():
        route = pd.read_csv(d / f"{trip_id}.route", sep=r"\s+", header=None)[0].to_numpy()
    fwd, _ = local_crs(track.lon.mean(), track.lat.mean())
    track["x"], track["y"] = fwd.transform(track.lon.values, track.lat.values)
    nodes["x"], nodes["y"] = fwd.transform(nodes.lon.values, nodes.lat.values)
    return Trip(trip_id, partition, track, nodes, arcs, route)

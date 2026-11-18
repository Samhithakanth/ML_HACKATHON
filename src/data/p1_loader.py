"""Raw-data loading for PathSnap trips (.track/.nodes/.arcs/.route)."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PARTITIONS = ("train", "student_test")
# Trips are located all over the world (one city per trip), so no single UTM zone works.
# Each trip gets its own local azimuthal-equidistant CRS (metres) centred on the GPS centroid.
def local_crs(lon0: float, lat0: float):
    crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    return (Transformer.from_crs("EPSG:4326", crs, always_xy=True),
            Transformer.from_crs(crs, "EPSG:4326", always_xy=True))


def list_trips(partition: str, raw_dir: Path = RAW_DIR):
    return sorted(p.name for p in (Path(raw_dir) / partition).iterdir() if p.is_dir())


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame      # lon, lat, t (+ x, y projected, metres)
    nodes: pd.DataFrame      # lon, lat (+ x, y); index = node_id
    arcs: pd.DataFrame       # from_node, to_node; index = arc_id
    route: np.ndarray        # ordered arc_ids (reference route)
    to_xy: object = None     # lon/lat -> local metres
    to_lonlat: object = None # local metres -> lon/lat


def load_trip(trip_id: str, partition: str, raw_dir: Path = RAW_DIR, project=True) -> Trip:
    d = Path(raw_dir) / partition / trip_id
    track = pd.read_csv(d / f"{trip_id}.track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(d / f"{trip_id}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(d / f"{trip_id}.arcs", sep=r"\s+", header=None, names=["from_node", "to_node"])
    route = pd.read_csv(d / f"{trip_id}.route", sep=r"\s+", header=None)[0].to_numpy()
    fwd = inv = None
    if project:
        fwd, inv = local_crs(track.lon.mean(), track.lat.mean())
        track["x"], track["y"] = fwd.transform(track.lon.values, track.lat.values)
        nodes["x"], nodes["y"] = fwd.transform(nodes.lon.values, nodes.lat.values)
    return Trip(trip_id, partition, track, nodes, arcs, route, fwd.transform if fwd else None, inv.transform if inv else None)

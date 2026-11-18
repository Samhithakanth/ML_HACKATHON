"""Interactive Folium map: GPS fixes, road network, reference route, (optional) predicted route / candidates."""
import numpy as np, pandas as pd, folium
from pathlib import Path
from src.data.p1_loader import load_trip
from src.data.p1_candidates import generate_candidates

ROOT = Path(__file__).resolve().parents[2]


def _line(trip, arc_ids):
    n = trip.nodes; a = trip.arcs.iloc[arc_ids]
    return [[(n.lat.values[f], n.lon.values[f]), (n.lat.values[t], n.lon.values[t])] for f, t in zip(a.from_node, a.to_node)]


def make_map(trip, predicted_arcs=None, point_slice=None, network_radius_m=150, show_candidates_for=None, out=None):
    """predicted_arcs: ordered arc_ids (None until the model exists). point_slice limits the extent for readability."""
    tr = trip.track if point_slice is None else trip.track.iloc[point_slice]
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=15, tiles="OpenStreetMap")
    # network near the plotted GPS only (full networks have 100k+ arcs)
    from shapely import STRtree, points
    from src.data.p1_candidates import arc_geometries
    g, _, _ = arc_geometries(trip)
    pi, ai = STRtree(g).query(points(tr[["x", "y"]].values), predicate="dwithin", distance=network_radius_m)
    fg = folium.FeatureGroup("Road network (within %d m of GPS)" % network_radius_m, show=True)
    for seg in _line(trip, np.unique(ai)):
        folium.PolyLine(seg, color="#888", weight=2, opacity=0.7).add_to(fg)
    fg.add_to(m)
    ref = folium.FeatureGroup("Reference route (derived ground truth)", show=True)
    from src.data.p1_labels import align_to_route
    j, arc, _ = align_to_route(trip)
    ids = pd.unique(arc[tr.index.values]) if point_slice is not None else trip.route
    for seg in _line(trip, ids): folium.PolyLine(seg, color="#1a9850", weight=5, opacity=0.9).add_to(ref)
    ref.add_to(m)
    gp = folium.FeatureGroup("GPS observations", show=True)
    for i, r in tr.iterrows():
        folium.CircleMarker((r.lat, r.lon), radius=3, color="#2166ac", fill=True, fill_opacity=0.8, weight=1,
                            tooltip=f"idx {i}  t={r.t:.0f}s").add_to(gp)
    gp.add_to(m)
    if predicted_arcs is not None:
        pr = folium.FeatureGroup("Predicted route", show=True)
        for seg in _line(trip, np.asarray(predicted_arcs)): folium.PolyLine(seg, color="#d73027", weight=4, dash_array="8").add_to(pr)
        pr.add_to(m)
    if show_candidates_for is not None:
        c = generate_candidates(trip, 100, 30); c = c[c.point_idx == show_candidates_for]
        cg = folium.FeatureGroup(f"Candidates of point {show_candidates_for}", show=True)
        for seg, d in zip(_line(trip, c.arc_id.values), c.dist_m): folium.PolyLine(seg, color="#fc8d59", weight=4, tooltip=f"{d:.0f} m").add_to(cg)
        cg.add_to(m)
    legend = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 12px;border:1px solid #999;font-size:13px">
    <b>Trip %s</b><br><span style="color:#2166ac">&#9679;</span> GPS fixes<br><span style="color:#1a9850">&#9644;</span> Reference route<br>
    <span style="color:#d73027">&#9644;</span> Predicted route<br><span style="color:#fc8d59">&#9644;</span> Candidate arcs<br><span style="color:#888">&#9644;</span> Road network</div>""" % trip.trip_id
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    lo, hi = tr.lat.min(), tr.lat.max(); m.fit_bounds([[lo, tr.lon.min()], [hi, tr.lon.max()]])
    if out: Path(out).parent.mkdir(parents=True, exist_ok=True); m.save(out)
    return m


if __name__ == "__main__":
    t = load_trip("00000001", "train")
    make_map(t, point_slice=slice(0, 300), show_candidates_for=150, out=ROOT / "results/maps/eda_train_00000001.html")
    s = load_trip("00000033", "train")
    make_map(s, point_slice=slice(2500, 2800), out=ROOT / "results/maps/eda_train_00000033_drift.html")
    print("maps written")

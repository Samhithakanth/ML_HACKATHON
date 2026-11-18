"""
tests/test_models_hmm.py  (Persons 2 and 3)
Unit tests for the shared metrics, the feature builder and the HMM decoder, on small synthetic road networks
where the correct answer is known. No data files and no API needed:  pytest tests/test_models_hmm.py
"""
import numpy as np
import pandas as pd
import pytest

from src.data.candidates import candidate_table
from src.data.loader import Trip
from src.eval.metrics import (collapse, compute_route_continuity, normalized_edit_distance, route_set_prf,
                              top1_from_scores, candidate_metrics)
from src.features.build import FEATURE_COLUMNS, build_features
from src.hmm.decoder import DEFAULT_PARAMS, build_lattice, decode


# ------------------------------------------------------------------ synthetic trips
def straight_road(n_nodes=11, spacing=50.0, two_way=True, gps=None, t=None, extra_arcs=()):
    """East-west road of n_nodes nodes; arcs 0..n-2 go east, (if two_way) arcs n-1.. go west (reverse twins)."""
    nodes = pd.DataFrame({"x": np.arange(n_nodes) * spacing, "y": np.zeros(n_nodes)})
    arcs = [(i, i + 1) for i in range(n_nodes - 1)]
    if two_way:
        arcs += [(i + 1, i) for i in range(n_nodes - 1)]
    arcs += list(extra_arcs)
    nodes["lon"], nodes["lat"] = nodes.x, nodes.y
    arcs = pd.DataFrame(arcs, columns=["from_node", "to_node"])
    if gps is None:                                    # vehicle drives east at 10 m/s, 1 Hz, 3 m north of the road
        xs = np.arange(5.0, (n_nodes - 1) * spacing - 5, 10.0)
        gps = np.c_[xs, np.full(len(xs), 3.0)]
    track = pd.DataFrame({"x": gps[:, 0], "y": gps[:, 1]})
    track["t"] = np.arange(len(track), dtype=float) if t is None else t
    track["lon"], track["lat"] = track.x, track.y
    return Trip("test", "test", track, nodes, arcs, None)


def east_arc(x, spacing=50.0):
    return int(x // spacing)


# ------------------------------------------------------------------ metrics
def brute_edit(a, b):
    D = np.zeros((len(a) + 1, len(b) + 1), int)
    D[:, 0], D[0, :] = range(len(a) + 1), range(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return D[-1, -1] / max(len(a), len(b), 1)


def test_edit_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(300):
        a, b = rng.integers(0, 5, rng.integers(1, 12)), rng.integers(0, 5, rng.integers(1, 12))
        assert normalized_edit_distance(a, b) == pytest.approx(brute_edit(collapse(a), collapse(b)))


def test_route_continuity_and_set_prf():
    arcs = np.array([[0, 1], [1, 2], [2, 3], [5, 6]])
    assert compute_route_continuity([0, 0, 1, 2], arcs) == (1.0, 0)
    cont, breaks = compute_route_continuity([0, 1, 3], arcs)          # 1 -> 3 is not connected
    assert breaks == 1 and cont == pytest.approx(0.5)
    assert compute_route_continuity([0, -1, 1], arcs)[1] == 2          # no prediction counts as a break
    prf = route_set_prf([0, 0, 1, 3], [0, 1, 2])
    assert prf["route_precision"] == pytest.approx(2 / 3) and prf["route_recall"] == pytest.approx(2 / 3)


def test_top1_from_scores_picks_best_and_marks_missing():
    c = pd.DataFrame({"point_idx": [0, 0, 0, 2, 2], "arc_id": [5, 6, 7, 1, 2]})
    out = top1_from_scores(c, np.array([0.1, 0.8, 0.3, 0.4, 0.6]), n_points=3)
    assert out.tolist() == [6, -1, 2]


def test_candidate_metrics_perfect_classifier():
    m = candidate_metrics([0, 1, 0, 1], [0.01, 0.99, 0.02, 0.98])
    assert m["macro_f1"] == 1.0 and m["roc_auc"] == 1.0 and m["log_loss"] < 0.05


# ------------------------------------------------------------------ features
def test_features_heading_sign_separates_reverse_twin():
    trip = straight_road()
    c = candidate_table(trip, radius=20.0, max_k=30)
    f = build_features(trip, c)
    assert all(col in f.columns for col in FEATURE_COLUMNS)
    assert not f[FEATURE_COLUMNS].isna().any().any()
    east = f.arc_id < 10
    assert (f.loc[east, "cos_c"] > 0.99).all() and (f.loc[~east, "cos_c"] < -0.99).all()


def test_features_zero_length_arc_is_guarded():
    trip = straight_road(extra_arcs=[(3, 3)])           # zero-length self-loop at node 3
    f = build_features(trip, candidate_table(trip, radius=30.0, max_k=30))
    z = f[f.arc_id == len(trip.arcs) - 1]
    assert len(z) and (z.zero_len == 1).all() and np.isfinite(z[FEATURE_COLUMNS].values).all()


# ------------------------------------------------------------------ HMM decoder
def test_hmm_resolves_reverse_twin_from_uninformative_emissions():
    """Both directions get the SAME probability; only the transition model knows the vehicle moves east."""
    trip = straight_road()
    c = candidate_table(trip, radius=20.0, max_k=30)
    prob = np.where(c.dist_m.values < 10, 0.5, 0.01)    # twins tie exactly
    pred = decode(build_lattice(trip, c), prob, DEFAULT_PARAMS)
    expected = np.array([east_arc(x) for x in trip.track.x])
    assert (pred == expected).mean() > 0.95


def test_hmm_overrides_isolated_wrong_emission():
    trip = straight_road()
    c = candidate_table(trip, radius=20.0, max_k=30)
    truth = np.array([east_arc(x) for x in trip.track.x])
    prob = np.where(c.arc_id.values == truth[c.point_idx.values], 0.9, 0.05)
    bad = 22                                             # mid-arc point where the classifier prefers the westbound twin
    twin = truth[bad] + 10                               # westbound twin of east arc k is arc 10 + k
    m = (c.point_idx.values == bad)
    prob[m] = np.select([c.arc_id.values[m] == twin, c.arc_id.values[m] == truth[bad]], [0.9, 0.05], 0.01)
    clf = top1_from_scores(c, prob, len(trip.track))
    hmm = decode(build_lattice(trip, c), prob, DEFAULT_PARAMS)
    assert clf[bad] != truth[bad] and hmm[bad] == truth[bad]


def test_hmm_cuts_at_gaps_and_missing_candidates():
    n = len(straight_road().track)                       # 49 fixes
    t = np.arange(n, dtype=float); t[30:] += 500.0        # 500 s gap after point 29
    trip = straight_road(t=t)
    c = candidate_table(trip, radius=20.0, max_k=30)
    c = c[c.point_idx != 10].reset_index(drop=True)      # point 10 has no candidates
    lat = build_lattice(trip, c, gap_s=60.0)
    assert not lat.linked[10] and not lat.linked[11] and not lat.linked[30] and lat.linked[29]
    pred = decode(lat, np.where(c.dist_m.values < 10, 0.6, 0.01), DEFAULT_PARAMS)
    assert pred[10] == -1 and (pred[np.r_[0:10, 11:n]] >= 0).all()


def test_hmm_disconnected_components_use_floor_not_failure():
    """Two disconnected one-way roads; the vehicle 'teleports' between them. Decoding must not fail and must follow the GPS."""
    a = straight_road(n_nodes=6, two_way=False)
    nodes = pd.concat([a.nodes, a.nodes.assign(y=a.nodes.y + 500, lat=a.nodes.y + 500)], ignore_index=True)
    arcs = pd.concat([a.arcs, a.arcs + 6], ignore_index=True)
    xs = np.arange(5.0, 245, 10.0)
    gps = np.c_[xs, np.where(np.arange(len(xs)) < 12, 3.0, 503.0)]
    trip = Trip("t", "t", pd.DataFrame({"x": gps[:, 0], "y": gps[:, 1], "t": np.arange(len(xs), dtype=float),
                                         "lon": gps[:, 0], "lat": gps[:, 1]}), nodes, arcs, None)
    c = candidate_table(trip, radius=20.0, max_k=30)
    pred = decode(build_lattice(trip, c), np.where(c.dist_m.values < 10, 0.9, 0.01), DEFAULT_PARAMS)
    assert (pred[:12] < 5).all() and (pred[12:] >= 5).all()

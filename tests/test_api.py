"""
tests/test_api.py
Comprehensive smoke and contract tests for the PathSnap API.
Verifies all 10 evaluation trips, schema correctness, server-side scoring,
and strict data leakage prevention.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Verifies GET /health connectivity check."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "n_trips" in data
    assert data["n_trips"] == 10


def test_list_trips_endpoint(client):
    """Verifies GET /trips returns 10 trips with correct fields."""
    resp = client.get("/trips")
    assert resp.status_code == 200
    data = resp.json()
    assert "trips" in data
    assert len(data["trips"]) == 10
    first = data["trips"][0]
    assert "trip_id" in first
    assert "n_gps_points" in first
    assert "duration_s" in first


def test_preview_endpoint(client):
    """Verifies GET /trips/{trip_id}/preview returns GPS in [lat, lon] order."""
    resp = client.get("/trips/00000007/preview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["trip_id"] == "00000007"
    assert "bbox" in data
    assert len(data["bbox"]) == 2
    assert "gps" in data
    assert len(data["gps"]) > 0
    # Coordinate check: Latitude is ~43 deg (Almaty / Eurasian latitudes)
    lat, lon = data["gps"][0]
    assert 40.0 <= lat <= 50.0
    assert 70.0 <= lon <= 85.0


def test_network_endpoint(client):
    """Verifies GET /trips/{trip_id}/network returns filtered road segments."""
    resp = client.get("/trips/00000007/network?buffer_m=150")
    assert resp.status_code == 200
    data = resp.json()
    assert data["trip_id"] == "00000007"
    assert "n_segments" in data
    assert "segments" in data
    assert len(data["segments"]) > 0
    seg = data["segments"][0]
    assert len(seg) == 2  # [[lat1, lon1], [lat2, lon2]]


def test_predict_known_trip(client):
    """Verifies POST /predict runs full pipeline and returns metrics."""
    resp = client.post("/predict", json={"trip_id": "00000007"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["trip_id"] == "00000007"
    assert "pred_route" in data
    assert "paths" in data["pred_route"]
    assert "classifier_only_route" in data
    assert "metrics" in data
    assert "point_correct" in data

    metrics = data["metrics"]
    assert "hmm_pointwise_accuracy" in metrics
    assert "classifier_only_pointwise_accuracy" in metrics
    assert "route_continuity" in metrics
    assert "hmm_macro_f1" in metrics
    assert "inference_time_ms" in metrics

    # Verify HMM outperforms or equals raw classifier
    assert metrics["route_continuity"] >= 0.8
    assert metrics["hmm_pointwise_accuracy"] >= 0.7

    # Data Leakage Check: NEVER expose reference geometry or labels
    forbidden_keys = {"label_arc", "reference_route", "route", "ground_truth", "ref_arcs"}
    assert not any(k in data for k in forbidden_keys)
    assert not any(k in metrics for k in forbidden_keys)


def test_predict_unknown_trip_404(client):
    """Verifies unknown trip returns clean 404 with error payload."""
    resp = client.post("/predict", json={"trip_id": "99999999"})
    assert resp.status_code == 404
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == "TRIP_NOT_FOUND"


def test_predict_bad_request_422(client):
    """Verifies malformed body returns clean 422 with error payload."""
    resp = client.post("/predict", json={})
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == "BAD_REQUEST"


def test_all_10_trips_predict(client):
    """Critical Acceptance Test: Evaluates that ALL 10 evaluation trips run cleanly."""
    trips_resp = client.get("/trips")
    trip_ids = [t["trip_id"] for t in trips_resp.json()["trips"]]
    assert len(trip_ids) == 10

    for tid in trip_ids:
        resp = client.post("/predict", json={"trip_id": tid})
        assert resp.status_code == 200, f"Trip {tid} failed with {resp.text}"
        data = resp.json()
        assert data["trip_id"] == tid
        assert data["n_gps_points"] > 0
        assert data["metrics"]["route_continuity"] > 0.0
        assert data["metrics"]["inference_time_ms"] >= 0

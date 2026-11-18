"""
api/schemas.py
Pydantic schemas matching PathSnap API Contract (v1).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    model_loaded: bool
    n_trips: int


class TripSummary(BaseModel):
    trip_id: str
    n_gps_points: int
    duration_s: int


class TripsListResponse(BaseModel):
    trips: List[TripSummary]


class PreviewResponse(BaseModel):
    trip_id: str
    bbox: List[List[float]]  # [[south, west], [north, east]]
    gps: List[List[float]]   # [[lat, lon], ...]
    gps_t: List[float]       # [t0, t1, ...]


class NetworkResponse(BaseModel):
    trip_id: str
    n_segments: int
    segments: List[List[List[float]]]  # [[[lat1, lon1], [lat2, lon2]], ...]


class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="8-character trip identifier, e.g. 00000007")


class RoutePath(BaseModel):
    paths: List[List[List[float]]]
    n_arcs: int


class Metrics(BaseModel):
    hmm_pointwise_accuracy: float
    classifier_only_pointwise_accuracy: float
    route_continuity: float
    hmm_macro_f1: float
    classifier_only_macro_f1: float
    n_disconnected_breaks: int
    inference_time_ms: int


class PredictResponse(BaseModel):
    trip_id: str
    n_gps_points: int
    pred_route: RoutePath
    classifier_only_route: RoutePath
    point_correct: List[bool]
    metrics: Metrics
    warnings: List[str] = []


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail

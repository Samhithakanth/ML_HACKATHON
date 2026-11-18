# PathSnap API Contract (v1)

Backend (Laptop 1, FastAPI) <-> UI (Laptop 2). The backend is built to this spec, so build the UI against it (use a mock server until the real API is up).

## Ground rules
- **Base URL:** `http://<LAPTOP1_IP>:8000`. Make this a single editable config value in the UI, with no hardcoded IP.
- **CORS:** enabled for all origins on the backend.
- **Format:** JSON everywhere. `Content-Type: application/json`.
- **Coordinate order: ALWAYS `[lat, lon]`** (the Leaflet/Folium order). The raw dataset files are `lon, lat`, but the backend converts them. The API never returns `lon, lat`.
- **Trip IDs** are 8-character strings, e.g. `"00000007"`. Never convert them to integers, because the leading zeros matter.
- **Ground truth never reaches the UI.** There is no reference-route geometry in any response. The server scores internally and returns only the metrics and per-point correct/incorrect flags.
- **Every UI request must go through the API.** No local files, and no bypassing the backend.

---

## 1. `GET /health`
Connectivity check. Call it on page load and show a green/red "API connected" badge.

```json
{ "status": "ok", "model": "lightgbm-tuned-v1", "n_trips": 10 }
```

## 2. `GET /trips`
The 10 evaluation trips, to populate the selector. **No labels.**

```json
{
  "trips": [
    { "trip_id": "00000007", "n_gps_points": 1354, "duration_s": 1353 },
    { "trip_id": "00000009", "n_gps_points": 1049, "duration_s": 1049 }
  ]
}
```

## 3. `GET /trips/{trip_id}/preview`
Raw GPS only, with no prediction. Use it to draw the track as soon as a trip is selected, before the user clicks Run.

```json
{
  "trip_id": "00000007",
  "bbox": [[43.388, 77.122], [43.496, 77.271]],
  "gps": [[43.4001, 77.1500], [43.4002, 77.1501]],
  "gps_t": [1.0, 2.0]
}
```
- `bbox` is `[[south, west], [north, east]]`. Pass it to `map.fitBounds`.
- `gps` and `gps_t` have the same length and the same order.

## 4. `GET /trips/{trip_id}/network`
Road network near the track, for the "road network" map layer. It is a separate endpoint because the full network can have 150k+ nodes. The backend only returns arcs within about 150 m of the track and merges two-way pairs.

Optional query: `?buffer_m=150`.

```json
{
  "trip_id": "00000007",
  "n_segments": 2,
  "segments": [
    [[43.4001, 77.1500], [43.4010, 77.1512]],
    [[43.4010, 77.1512], [43.4020, 77.1530]]
  ]
}
```
Each segment is a 2-point polyline. Render them all as one thin grey layer. Use canvas rendering (`L.canvas()`), since there can be up to about 50k segments.

## 5. `POST /predict`  <- main endpoint
Runs the full pipeline: candidates, classifier, HMM/Viterbi.

**Request**
```json
{ "trip_id": "00000007" }
```

**Response 200**
```json
{
  "trip_id": "00000007",
  "n_gps_points": 1354,

  "pred_route": {
    "paths": [
      [[43.4001, 77.1500], [43.4010, 77.1512], [43.4020, 77.1530]],
      [[43.4100, 77.1600], [43.4110, 77.1610]]
    ],
    "n_arcs": 123
  },

  "classifier_only_route": {
    "paths": [ [[43.4001, 77.1500], [43.4012, 77.1520]] ],
    "n_arcs": 130
  },

  "point_correct": [true, true, false, true],

  "metrics": {
    "hmm_pointwise_accuracy": 0.94,
    "classifier_only_pointwise_accuracy": 0.81,
    "route_continuity": 0.99,
    "hmm_macro_f1": 0.92,
    "classifier_only_macro_f1": 0.78,
    "n_disconnected_breaks": 1,
    "inference_time_ms": 2310
  },

  "warnings": ["GPS gap of 210 s at t=530; route split here"]
}
```

Field notes:
- `pred_route.paths` is a **list of polylines, not one line**. The backend breaks the path wherever the route is disconnected, for example at large GPS gaps. **Draw each path as a separate `L.polyline`, and never connect the end of one path to the start of the next.**
- `classifier_only_route` has the same shape. Offer it as a toggleable layer so evaluators can see the HMM improvement.
- `point_correct` has one boolean per GPS point, in the same order as `/preview` `gps`. Show it as green (correct) and red (wrong) dots on the raw GPS layer, for the HMM prediction.
- All `metrics` values are floats in `[0,1]` (shown as %), except `n_disconnected_breaks` (int) and `inference_time_ms` (int).
- `warnings` may be empty. Show the warnings in an info box.
- Latency: expect **2-30 s** for large trips. Use a **120 s timeout**, show a spinner, and disable the Run button while a request is in flight.

**Errors**
```json
{ "error": { "code": "TRIP_NOT_FOUND", "message": "Unknown trip_id 00000099" } }
```
| HTTP | code | When |
|---|---|---|
| 404 | `TRIP_NOT_FOUND` | trip_id isn't in the 10 evaluation trips |
| 422 | `BAD_REQUEST` | body is missing or malformed |
| 500 | `PIPELINE_ERROR` | anything failed server-side |

Show `error.message` to the user and let them retry.

---

## UI requirements (checklist)
- [ ] API-URL config field, plus a connection badge from `/health`
- [ ] Trip selector built from `/trips` (all 10 must work)
- [ ] On trip select: call `/preview`, draw the raw GPS (blue dots) and fit the bounds
- [ ] Run button calls `/predict` with a loading state and a timeout
- [ ] Map layers with a **layer-control toggle and a legend**:
  - raw GPS (blue)
  - road network (thin grey, from `/network`, loaded lazily)
  - HMM predicted route (green or orange, thick)
  - classifier-only route (dashed red)
  - correct/incorrect points (green/red)
- [ ] Metrics panel (HMM vs classifier-only side by side)
- [ ] Warnings and error display
- [ ] Works on all 10 trips and with the empty `warnings` case

## Mock server
Until the real API is up, mock all five endpoints with the example payloads above.

## Changing this contract
Any change to field names or shapes should go through the backend person first. The backend will not rename fields unilaterally. New fields may be added, so the UI should ignore unknown fields.

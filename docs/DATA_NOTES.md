# PathSnap Data & Coordinate Reference Notes

## 1. Dataset Overview & File Conventions
The dataset is structured into trips, each represented by a directory named with an 8-character string ID (e.g., `00000007`). Each trip directory contains 4 primary files:

| File Extension | Content & Format | Coordinate Order in File | Coordinate Order in API/UI |
|---|---|---|---|
| `.track` | Raw GPS observations: `[longitude, latitude, timestamp_s]` (space-separated, no headers). Nominally 1 Hz (~97% of steps $\Delta t = 1$ s). | `lon, lat` | **`lat, lon`** |
| `.nodes` | Road network intersection nodes: `[longitude, latitude]` (space-separated, 0-based row index = `node_id`). | `lon, lat` | **`lat, lon`** |
| `.arcs` | Directed road segments: `[from_node_id, to_node_id]` (space-separated integers, 0-based row index = `arc_id`). | N/A | N/A |
| `.route` | Ground-truth chronological sequence of traversed `arc_id`s (1 integer per line). **Kept strictly server-side on Laptop 1.** | N/A | **NEVER EXPOSED** |

## 2. Coordinate Order & Projections
- **Crucial Rule:** Raw files store coordinates as `[lon, lat]` (GIS convention $x, y$). However, Leaflet, Folium, and the PathSnap REST API use `[lat, lon]` ($y, x$). All data loaders must explicitly flip coordinates to `[lat, lon]` when returning API responses or Folium geometry.
- **Metric Distance Projection:** Ordinary Euclidean distance on raw angular degrees is invalid. For candidate generation and feature extraction, coordinates are projected to local meters using an equirectangular metric projection scaled by the mean latitude $\phi_0$ of the trajectory:
  $$\Delta x = 111320.0 \cdot \cos(\phi_0) \cdot \Delta \text{lon}$$
  $$\Delta y = 110540.0 \cdot \Delta \text{lat}$$
  This provides sub-meter spatial accuracy without heavy GIS projection dependencies.

## 3. Data Partitions
- **`train` (80 trips):** Used exclusively for spatial candidate generation, feature extraction, cross-validation, and model training.
- **`student_test` (10 trips):** Used for out-of-sample model evaluation, HMM tuning, and development validation.
- **`evaluator_test` (10 hidden trips):** Reserved for live evaluation on Laptop 1.

## 4. Ground-Truth Route Alignment (`align_to_route`)
The ground truth sequence in `.route` defines which physical arcs were traversed by the vehicle. Because GPS points are recorded at 1 Hz, multiple consecutive GPS points typically lie along the same road arc.
At server startup on Laptop 1, reference labels for evaluation trips are built:
For each GPS point $p_i$, the reference road segment is found by sliding a forward window over the `.route` sequence and matching $p_i$ to the closest arc along the vehicle's chronological path.
These labels are cached **server-side only** in a private dictionary and used solely inside `POST /predict` to evaluate Top-1 accuracy, Macro F1, and point correctness flags.

## 5. Temporal Gaps & Anomalies
While ~97% of GPS samples occur at 1 s intervals, occasional sensor dropouts produce time gaps $\Delta t > 5$ s.
When $\Delta t > 5$ s:
1. The API emits a descriptive entry in `warnings` (e.g. `GPS gap of 42.0 s at t=125.0; route split here`).
2. The route polyline is split into discrete sub-paths so that map rendering does not draw an artificial straight line across disconnected space.
3. The Viterbi decoder handles the gap by lowering transition expectations across the discontinuity.

## 6. Route Continuity Metric
Route continuity is defined as:
$$\text{Continuity} = \frac{\sum_{i=1}^{N-1} \mathbb{I}(\text{arc}_i == \text{arc}_{i+1} \lor \text{arcs}[\text{arc}_i, 1] == \text{arcs}[\text{arc}_{i+1}, 0])}{N - 1}$$
Where:
- $\text{arc}_i == \text{arc}_{i+1}$: Vehicle remains on the same segment.
- $\text{arcs}[\text{arc}_i, \text{to\_node}] == \text{arcs}[\text{arc}_{i+1}, \text{from\_node}]$: Vehicle transitions along connected road topology.
Any jump where `to_node != from_node` is considered a topological disconnect.

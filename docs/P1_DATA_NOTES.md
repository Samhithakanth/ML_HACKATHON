# PathSnap – data notes and handoff contract (Person 1)

Reproduce everything from the repo root (raw data in `data/raw/{train,student_test}/<trip>/`):

```
python -m src.data.eda_inventory   # results/eda/trip_inventory.csv
python -m src.data.label_audit     # results/eda/label_audit.csv
python -m src.data.radius_sweep    # results/eda/radius_sweep_by_trip.csv (TRAIN only)
python -m src.data.build_dataset   # data/processed/*.parquet
python -m src.viz.eda_plots        # results/eda/plots/*.png
python -m src.viz.p1_folium_map       # results/maps/*.html
```

## Dataset structure (verified, not assumed)
| | train | student_test |
|---|---|---|
| trips | 80 | 10 |
| GPS points | 208,417 (361 – 19,313 per trip) | 19,160 (262 – 3,958) |
| nodes / trip (median) | 9,608 (299 – 155,003) | 6,596 (1,770 – 26,635) |
| arcs / trip (median) | 19,526 | 13,145 |

* Every trip has its **own road network** (nodes/arcs are per-trip files, ids are local row indices).
* **Trips are located all over the world** (Ecuador, USA, Europe, Asia ...), one city per trip. A single UTM zone is
  therefore wrong. We project each trip into a **local azimuthal-equidistant CRS (metres) centred on its GPS centroid**
  (`src/data/p1_loader.py`). Distortion is negligible at trip scale (<= ~127 km extent).
* Arcs are **directed**; ~82% have a reverse twin (identical geometry). Arcs are straight node-to-node segments.
* `.route` = ordered arc ids. It is a fully connected path in 70/80 train and 9/10 test trips (see anomalies). **There are no per-point labels.**

## Data-quality findings
* No NaN / duplicate GPS rows, no lon/lat swaps, timestamps strictly increasing in all 90 trips, all GPS inside network bbox, no isolated nodes.
* Sampling: mean 93.8% (train) / 98.2% (test) of steps are exactly 1 s. Large gaps: 22 train trips have a gap > 60 s
  (trip 90: 153,064 s; trip 78: 52,810 s). 1 test trip has a gap of 210 s. **HMM must break/restart sequences at big gaps.**
* Position jumps > 100 m between fixes: 23 train trips (up to 888 m), 0 test trips.
* Road network is **not connected**: 79/80 train and 10/10 test trips have >1 weakly connected component (and many strong ones,
  because of one-way arcs). Transitions between disconnected components must have a fallback in the HMM.
* 29 train / 3 test trips contain zero-length arcs (identical endpoints) – distances/heading must guard against this.
* Reference route has breaks (consecutive arcs not sharing a node) in 10 train + 1 test trip (worst 1.9% of steps).
* Trips are long and uneven: trip 54 alone has 19,313 points (9% of train) – use group-aware CV and consider per-trip weighting.

## Reference labels (derived)
`.route` gives arcs, not per-point labels. `src/data/p1_labels.py` aligns the GPS sequence to the route with a **monotone
dynamic programme** (non-decreasing route index, min sum of point-to-arc distance, O(T·R)).
* 88/90 trips: median GPS→label distance ≈ 4.6 m (train) / 4.7 m (test), p99 ≈ 21 / 25 m → labels are reliable.
* **Suspect trips: train 00000033 and 00000054.** GPS drifts 120–140 m away from the reference route for long
  stretches (offset ≈ 0 at start, ≈ 130 m mid-trip – not a constant shift). 54% / 87% of their points have no true arc within
  the 100 m candidate radius. Flagged with `trip_label_suspect=True`. Recommendation: **exclude (or at least report
  separately) these two trips when training/evaluating; document as a reference-route limitation.** All 10 student_test trips are clean.
* Labels are reference information, not verified truth – state this in the report.

## Candidate generation
All arcs within **R = 100 m** of the GPS point, **capped at K = 30 nearest** (sorted by distance). Chosen from the TRAIN sweep
(`results/eda/plots/05_candidate_miss_rate.png`):

| | train (all) | train (excl. 33, 54) | student_test |
|---|---|---|---|
| candidate miss rate | 10.2% | 0.65% | 0.09% |
| candidates / point | 15.6 | – | 19.1 |

R = 50 m doubles the miss rate (1.4%); R = 150 m gives only −0.2 pp for ~2x the candidates.
Because arcs are directed with twins, both directions of a street are candidates (disambiguate by heading – a P2 feature).

## Handoff files (`data/processed/`)
**`{train,student_test}_points.parquet`** – one row per GPS fix
`trip_id, partition, point_idx, lon, lat, t, x, y (metres, per-trip CRS), dt_prev, label_route_pos, label_arc, label_dist_m,
label_in_candidates, n_candidates, trip_label_suspect`

**`{train,student_test}_candidates.parquet`** – one row per (point, candidate arc)
`trip_id, point_idx, arc_id, dist_m, frac (position along arc 0-1), rank (0 = nearest), label (1 if arc_id == label_arc)`

Join key: `(trip_id, point_idx)` ↔ points; `(trip_id, arc_id)` ↔ raw `.arcs` row index. `trip_id` is an 8-char zero-padded string.
Load raw geometry with `src.data.p1_loader.load_trip(trip_id, partition)` (returns track/nodes/arcs/route + `to_xy` projector).
Use `GroupKFold` on `trip_id`. `label_in_candidates == False` points cannot be scored correctly by any classifier
(that is the candidate miss rate and is reported separately).

For the API (Person 4): `generate_candidates(trip, 100, 30)` in `src/data/p1_candidates.py` must run at inference time from the
raw `.track/.nodes/.arcs` only – never from `.route`.

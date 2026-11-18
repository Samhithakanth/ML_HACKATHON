import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))
md("""# 01 - Dataset exploration (Person 1)
Partitions used: **train (80 trips)** for all fitting decisions (label audit, candidate radius/K); **student_test (10 trips)** only to report the same statistics.
The evaluator-test partition is never touched. Run from the repo root: `python -m src.data.eda_inventory` etc. (see `docs/DATA_NOTES.md`); this notebook only reads their outputs.""")
code("""import sys, os; os.chdir(os.path.abspath('..')) if os.path.basename(os.getcwd())=='notebooks' else None; sys.path.insert(0, '.')
import pandas as pd, numpy as np
from IPython.display import Image, display
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
inv = pd.read_csv('results/eda/trip_inventory.csv', dtype={'trip_id': str})""")
md("## 1. Inventory (partition = train / student_test)")
code("""inv.groupby('partition').agg(trips=('trip_id','count'), gps=('n_gps','sum'), gps_min=('n_gps','min'), gps_med=('n_gps','median'), gps_max=('n_gps','max'),
    nodes_med=('n_nodes','median'), arcs_med=('n_arcs','median'), route_arcs_med=('n_route_arcs','median'))""")
code("""display(Image('results/eda/plots/01_points_per_trip.png'))""")
md("""## 2. Data quality
* No NaN / duplicate rows; timestamps strictly increasing; every GPS fix lies inside its network's bounding box; no isolated nodes.
* **Trips are worldwide** (one city per trip) -> a per-trip local azimuthal-equidistant projection is used instead of a global UTM zone.""")
code("""q = inv.groupby('partition').agg(nan=('gps_nan','sum'), dups=('gps_dup_rows','sum'), nonmono=('t_monotonic', lambda s: (~s).sum()),
    dt1_frac=('dt_eq1_frac','mean'), trips_gap_gt60s=('dt_max', lambda s: (s>60).sum()), trips_jump_gt100m=('n_jumps_gt100m', lambda s:(s>0).sum()),
    trips_zero_len_arcs=('zero_len_arcs', lambda s:(s>0).sum()), trips_multi_component=('weak_components', lambda s:(s>1).sum()),
    trips_broken_route=('route_connected_frac', lambda s:(s<1).sum()))
q""")
code("""display(Image('results/eda/plots/02_dt_distribution.png')); display(Image('results/eda/plots/03_network_stats.png'))
inv.sort_values('dt_max', ascending=False)[['trip_id','partition','n_gps','dt_max','n_gaps_gt5s','n_jumps_gt100m','step_max_m']].head(6)""")
md("""## 3. Reference route -> per-point labels
`.route` lists arcs only. Labels are derived with a monotone DP alignment (`src/data/labels.py`). Validation: GPS-to-label distance and a visual check on hard trips.""")
code("""la = pd.read_csv('results/eda/label_audit.csv', dtype={'trip_id': str})
display(la.groupby('partition')[['d_med','d_p90','d_p99','frac_gt100']].describe().T.round(2))
la[la.frac_gt100 > 0.1][['trip_id','partition','d_med','d_p90','frac_gt100']]""")
code("""display(Image('results/eda/plots/04_label_distance.png')); display(Image('results/eda/plots/07_label_alignment_check.png'))""")
md("""Trips **33 and 54** (train) are flagged: the GPS drifts ~130 m from the reference route for long stretches (offset ~0 at the start, ~130 m mid-trip - not a constant shift), so the true arc is outside any sensible candidate radius.
The reference route is therefore *reference* information, not verified truth. Recommendation: exclude/report separately.""")
md("## 4. Candidate generation (radius R, cap K chosen on train only)")
code("""display(Image('results/eda/plots/05_candidate_miss_rate.png')); display(Image('results/eda/plots/06_candidates_per_point.png'))
pts = {p: pd.read_parquet(f'data/processed/{p}_points.parquet') for p in ['train','student_test']}
for p, d in pts.items():
    ok = d[~d.trip_label_suspect]
    print(f'{p}: miss all={1-d.label_in_candidates.mean():.4f}  clean trips={1-ok.label_in_candidates.mean():.4f}  cands/pt={d.n_candidates.mean():.1f}  at K cap={(d.n_candidates==30).mean():.1%}  no candidates={(d.n_candidates==0).mean():.2%}')""")
md("""Chosen **R = 100 m, K = 30**: clean-trip miss rate ~0.65% (train) / 0.09% (student_test). R = 50 m doubles the miss rate; R = 150 m adds ~2x candidates for ~0.2 pp. ~26% of train points hit the K cap (dense urban networks), so P2 should include `rank` / distance-gap features.""")
md("## 5. Interactive map\nOpen `results/maps/eda_train_00000001.html` (GPS fixes, nearby network, reference route, candidates of one point) and `results/maps/eda_train_00000033_drift.html` (GPS drift).")
code("""import folium
from src.data.loader import load_trip
from src.viz.folium_map import make_map
make_map(load_trip('00000001','train'), point_slice=slice(0,150), show_candidates_for=75)""")
md("## 6. Hand-off\nSee `docs/DATA_NOTES.md` for the table schemas: `data/processed/{train,student_test}_{points,candidates}.parquet`.")
nb.cells = C; nbf.write(nb, 'notebooks/01_eda.ipynb')

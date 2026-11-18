# PathSnap Technical Report — Sections 3, 4 and 5 (Persons 2 and 3)

Every number in this file is read from a CSV in `results/` produced by the scripts listed in the README
(section 6, "Reproducing the Experiments"). Seeds are fixed (42). Nothing here was typed in by hand from memory.

**Data partitions used**

| Partition | Use |
|---|---|
| train, 78 clean trips | model fitting, 5-fold GroupKFold cross-validation by `trip_id`, Bayesian optimisation, HMM decoder tuning |
| train, trips 33 and 54 | excluded from fitting and selection (reference route drifts ~130 m from the GPS, see `person1/docs/DATA_NOTES.md`); reported separately |
| student_test, 10 trips | final reporting only; never used for any model, hyperparameter or decoder choice |
| evaluator (hidden) | never read by any script |

**Problem framing.** For every GPS point we generate candidate arcs (all arcs within 100 m, at most the 30 nearest;
P1's train-only sweep). Each (point, candidate) pair is a binary example: is this candidate the reference arc of the point?
The reference arc comes from P1's monotone dynamic-programming alignment of the GPS sequence to `.route`.
Clean train has 183,260 points and 2,972,785 candidate rows (6.2 % positive); student_test has 19,160 points and 366,311 rows.
A classifier scores the candidates; "classifier-only" picks the best-scoring candidate per point; the HMM (Section 5)
decodes a connected sequence from the same scores.

**Features (31, `src/features/build.py`).** All computed from the raw `.track/.nodes/.arcs` only, vectorised per trip:

| Group | Features | Why |
|---|---|---|
| geometry | `dist_m`, `frac`, `rank`, `log_arc_len`, `zero_len`, `d_from_node`, `d_to_node` | the true arc is usually the closest; zero-length arcs guarded |
| relative | `dist_gap`, `dist_ratio`, `n_cands`, `gap_2nd` | 26 % of train points hit the K = 30 cap, so absolute distance alone is ambiguous |
| heading | `cos_c`, `cos_b`, `cos_f`, `abs_cos_c`, `move_c` | cosine between GPS motion and arc bearing; its **sign separates the two directions of a street** (82 % of arcs have a reverse twin with identical geometry) |
| kinematics | `speed`, `dt_prev`, `dt_next`, `accel` | heading is unreliable when the vehicle is almost stopped; gaps |
| topology | `out_deg_to`, `in_deg_from`, `has_twin` | intersections vs plain road |
| context | `prev_same_dist`, `next_same_dist`, `prev_conn_dist`, `next_conn_dist`, `progress_along`, `progress_vs_move`, `win_frac`, `win_mean_dist` | is the same / a graph-connected arc also close at t−1 and t+1, does the projection move forward along the arc, how often is the arc a candidate in a ±5-point window |

On one test trip the true arcs average 7 m from the fix with heading cosine 0.84, against 49 m and −0.13 for the other candidates.

**Protocol (`src/models/train_compare.py`).** GroupKFold(5) by trip on the 78 clean train trips (fold membership in
`results/cv_folds.json`). Training rows: at most 800 random points per trip with all their candidates (so the longest trips
do not dominate); CV evaluation: at most 1,000 points per held-out trip, the same subset for every model. The final model of each
family is refit on the sampled rows of all 78 trips and evaluated on every student-test point. Scaling (StandardScaler) is
inside each model's Pipeline, so it is fit on the training fold only.

**Metrics (`src/eval/metrics.py`, shared with the API).**
*Observation-level top-1* = fraction of all points whose best-scoring candidate is the reference arc (points without
candidates count as wrong). *Macro F1, precision, recall* = candidate-level binary classification at threshold 0.5 (macro over the two classes).
*Log loss* = candidate-level. *Route continuity* = fraction of consecutive points whose predicted arcs are the same arc or
directly connected (to-node = next from-node).

**Primary selection metric: CV observation-level top-1.** It is the quantity the system is judged on (one arc per GPS fix),
and unlike macro F1 it does not depend on a 0.5 threshold that is never used at inference. Log loss is the secondary criterion,
because the HMM multiplies the probabilities, so they must be calibrated.

---

## 3. Model Experimentation and Comparison

### 3.1 Overall model comparison

Source: `results/model_comparison.csv` (+ `results/model_route_metrics.csv` for continuity), table `results/model_comparison_table.csv`.
CV = 5-fold GroupKFold on clean train (mean ± std over folds). Test = student-test (10 trips, 19,160 points). Inference time is
measured on the full student-test candidate set (366,311 rows) on one laptop (22 logical cores).

<!-- TABLE_3_1 -->

### 3.2 Per-model analysis

<!-- PER_MODEL -->

### 3.3 Visualisations

* `results/figures/models_comparison.png`: CV vs student-test top-1, macro F1 and log loss for every model.
* `results/figures/models_train_cv_test.png`: train-fit vs CV vs student-test top-1 (over/underfitting check).
* `results/figures/models_confusion_matrices.png`: student-test candidate-level confusion matrices.
* `results/figures/models_mlp_loss_curve.png`: MLP training loss and validation accuracy per epoch. This is the only model
  with an epoch history; no curves are drawn for the other models.

### 3.4 Model selection

<!-- SELECTION -->

---

## 4. Bayesian Hyperparameter Optimisation (LightGBM)

<!-- SECTION_4 -->

---

## 5. HMM and Viterbi Route Reconstruction

<!-- SECTION_5 -->

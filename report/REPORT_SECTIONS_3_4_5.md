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

All numbers are student-test unless marked CV. "Train-fit top-1" is measured on 40 sampled training points per
training trip of each fold (`cv_train_top1_mean`); comparing it with CV top-1 is our over/underfitting evidence.
Candidate-level positives are 6.2 % of rows, so recall/precision refer to the positive class (the true arc).

#### 3.2.1 Logistic Regression (linear)
* **Why suitable.** A calibrated linear baseline: most of the signal is monotone in a few features (closer, better aligned,
  present in the neighbouring points), so a linear log-odds model is a reasonable first fit. Its probabilities are directly usable as HMM emissions.
* **Strengths / limitations.** Fast (4 s fit, 0.30 ms per 1k candidates), stable across folds, interpretable coefficients. It cannot
  model interactions, e.g. "the heading cosine matters only when the vehicle is moving", so it confuses the two directions of a
  street when the vehicle is slow.
* **Hyperparameters / objective.** StandardScaler, L2 penalty C = 1.0, lbfgs, max_iter 2000. Objective: binary cross-entropy (log loss) + L2.
* **Results.** Top-1 0.841, macro F1 0.907, log loss 0.046 (CV: top-1 0.840 ± 0.037, macro F1 0.905).
* **Fit.** Train-fit top-1 0.847 vs CV 0.840 vs test 0.841: almost no gap, but it scores below the tree models. That is **underfitting (bias)**, not variance.

#### 3.2.2 Gaussian Naive Bayes (probabilistic generative)
* **Why suitable.** A generative, essentially training-free probabilistic baseline (0.8 s fit). It tests whether per-feature
  class-conditional densities are enough.
* **Strengths / limitations.** Very fast and has no tuning. But our features are strongly dependent (`dist_m`, `dist_gap`, `dist_ratio`,
  `win_mean_dist` all measure distance), so the independence assumption counts the same evidence many times. The probabilities are
  pushed to 0/1 and badly calibrated, and many features are far from Gaussian (bounded cosines, the 150 m fill value).
* **Hyperparameters / objective.** StandardScaler + GaussianNB (var_smoothing 1e-9); maximum-likelihood class-conditional Gaussians.
* **Results.** Top-1 0.772, macro F1 0.738, **log loss 0.797** (10–20× worse than the others). Precision 0.36 at recall 0.97: it calls far too many candidates positive (32,878 false positives).
* **Fit.** Train-fit 0.816 ≈ CV 0.808, but test is lower at 0.772. The model is too simple (underfitting), and its ranking also generalises less
  well. Because of the poor calibration it would be a poor HMM emission. Kept only as a documented baseline.

#### 3.2.3 K-Nearest Neighbours (instance-based)
* **Why suitable.** A non-parametric local method: similar (distance, heading, context) patterns should share labels. It needs no assumption about the
  shape of the decision boundary.
* **Strengths / limitations.** Captures local non-linear structure with no training. But prediction is expensive: every query
  is compared with the stored rows, **57.7 ms per 1k candidates, 170× slower than LightGBM**, and 21 s for the student-test set.
  Because of that we stored only a 120k-row subsample. It is also sensitive to feature scaling and to the many uninformative dimensions.
* **Hyperparameters / objective.** StandardScaler, k = 25, distance weighting, Euclidean metric, 120k-row training subsample. There is no loss
  function; the prediction is a weighted vote of the neighbours.
* **Results.** Top-1 0.826, macro F1 0.896, log loss 0.058 (CV top-1 0.819 ± 0.038).
* **Fit.** Train-fit 0.862 vs CV 0.819: a moderate gap (distance weighting makes the training points their own nearest neighbours).
  It is also limited by the subsample, i.e. it is data-starved rather than over-flexible. It performs worse than the tree ensembles, and inference cost rules it out for the API.

#### 3.2.4 Random Forest (bagging ensemble)
* **Why suitable.** Averaged deep trees handle the feature interactions logistic regression misses: heading × speed,
  distance × number of candidates, and context × distance. They need no scaling and are robust to the fill value (150 m).
* **Strengths / limitations.** Best CV top-1 and good calibration (log loss 0.037). But it is the slowest to train (147 s on all clean trips with 22 cores,
  and ~110–155 s per fold), its model file is 74 MB, and inference is 3.0 ms per 1k candidates (9× LightGBM).
* **Hyperparameters / objective.** 300 trees, min_samples_leaf 5, max_features sqrt, bootstrap; each tree is grown greedily to minimise Gini impurity.
* **Results.** Top-1 0.867, macro F1 0.927, log loss 0.035 (CV top-1 **0.869 ± 0.038**, macro F1 0.927).
* **Fit.** Train-fit top-1 0.947 vs CV 0.869: the largest gap of all models. It is **mildly overfitting / high variance** (deep trees with leaf size 5),
  although bagging keeps the held-out performance at the top. Test (0.867) matches CV, so the CV estimate is honest.

#### 3.2.5 LightGBM (gradient boosting) — selected for tuning
* **Why suitable.** Boosted trees fit the same interactions as the forest but sequentially correct the remaining errors. They are fast on
  millions of rows and expose a rich hyperparameter space, which makes them a good candidate for Bayesian optimisation.
* **Strengths / limitations.** 2.5 s training, 0.34 ms per 1k candidates, small model (0.16 MB). With the library defaults (100 trees, 31 leaves)
  its log loss (0.043) is worse than the forest's: defaults are not tuned for a 6 %-positive problem with 31 correlated features.
* **Hyperparameters / objective.** Defaults: 100 trees, learning rate 0.1, 31 leaves, unlimited depth, min_child_samples 20,
  no subsampling. Objective: binary log loss (gradient boosting on its gradient/Hessian).
* **Results (untuned).** Top-1 0.866, macro F1 0.927, log loss 0.043 (CV top-1 0.860 ± 0.035, macro F1 0.925).
* **Fit.** Train-fit 0.887 vs CV 0.860 vs test 0.866: a small, healthy gap. Neither clearly over- nor underfitting.

#### 3.2.6 Multi-Layer Perceptron (neural network)
* **Why suitable.** A smooth non-linear function of all features jointly, trained directly on log loss. It is the neural-network family and has a real epoch history.
* **Strengths / limitations.** Good calibration (log loss 0.038) at the cost of the longest training time here (249 s for a capped
  300k-row sample, 30 epochs, early stopping). It needs scaling, and its results depend on the optimiser settings and seed.
* **Hyperparameters / objective.** StandardScaler, hidden layers (128, 64), ReLU, Adam (lr 1e-3), batch 1024, ≤ 30 epochs, early stopping
  on a 10 % validation split (patience 4), 300k-row training subsample. Objective: binary cross-entropy.
* **Results.** Top-1 0.866, macro F1 0.923, log loss 0.038 (CV top-1 0.860 ± 0.038, macro F1 0.919).
* **Fit.** Train-fit 0.873 vs CV 0.860: a small gap. The loss curve (`models_mlp_loss_curve.png`) is still decreasing slowly when early
  stopping / the epoch cap ends training, so it is **slightly underfitting** rather than overfitting.

#### Common observation: the per-trip spread is larger than the gap between models
Fold-to-fold standard deviation of CV top-1 is about 0.035–0.038 for every model, larger than the gap between
the four best models (0.860–0.869). Some trips are intrinsically harder (dense networks, long stops, parallel roads), and
every model finds the same trips hard: fold 2 is the best fold and folds 0/1 the worst for every model (`results/model_cv_folds.csv`).
For all models, about 13 % of the remaining errors are not separable point by point (see 5.6): the two directions of the same street at low speed, and arcs meeting at an intersection.
This is what motivates the sequence model in Section 5.

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

Code: `src/tuning/bayes_opt.py`. Full trial log: `results/tuning/optuna_trials.csv`; best configuration:
`results/tuning/best_params.json`; comparison: `results/tuning/untuned_vs_tuned.csv`; plots: `results/figures/tuning_history.png`,
`results/figures/tuning_importance.png`. Library: Optuna 5.1, TPE sampler (seed 42, 8 random start-up trials), median pruner.

**1–2. Search space**

| Hyperparameter | Range / distribution | Default (untuned) | Best |
|---|---|---|---|
| n_estimators | int 100–600, step 50 | 100 | 500 |
| learning_rate | log-uniform 0.01–0.3 | 0.1 | 0.0290 |
| num_leaves | log-uniform int 15–255 | 31 | 194 |
| max_depth | categorical {−1 (unlimited), 4, 6, 8, 10, 12} | −1 | 8 |
| min_child_samples | log-uniform int 5–200 | 20 | 13 |
| subsample (bagging fraction, freq 1) | uniform 0.5–1.0 | 1.0 | 0.563 |
| colsample_bytree | uniform 0.4–1.0 | 1.0 | 0.740 |
| reg_alpha (L1) | log-uniform 1e-4–10 | 0 | 0.0832 |
| reg_lambda (L2) | log-uniform 1e-4–10 | 0 | 1.985 |

**3. Objective.** Mean candidate-level **log loss** over a 3-fold GroupKFold (by trip) on the 78 clean train trips, with the same
per-trip training sample (≤ 800 points/trip) and evaluation sample (≤ 1,000 points/trip) as Section 3. We minimised log loss rather than
top-1 for three reasons: the HMM consumes the probabilities, so calibration matters; log loss is smooth (top-1 changes in
steps of single points); and it is a proper scoring rule. Top-1 and macro F1 were logged for every trial as user attributes. A 3-fold search
(instead of 5) kept 30 trials affordable on one laptop; the final comparison below uses the same 5 folds as Section 3 for both models.
Student-test was not used by the objective.

**4. Trials.** 30 trials (27 complete, 3 pruned by the median pruner after their first fold). Trial 0 was the untuned default
configuration, enqueued so the search starts from the baseline (3-fold log loss 0.0464). Best: trial 8, log loss 0.0353.
Many configurations reach 0.0353–0.0354 (trials 8, 18, 22, 24, 28): the optimum is a flat plateau of "more, slower-learning
trees with more leaves, depth ≈ 6–8, row/column subsampling and L2 regularisation".

During development, one point needed care: an early version of the search space forced `max_depth ∈ [3, 14]`. On one fold, fixing
max_depth = 14 alone raised the log loss from 0.0448 to 0.0523, so the default (−1, unlimited) was added as an option. Without it,
the "untuned" starting point would have been worse than the real default.

**5. Best values.** See the table above.

**6. Untuned vs tuned** (same features, same training rows, same 5 CV folds, same student-test set):

<!-- TUNING_TABLE -->

The tuned model is better on every quality metric. Its CV top-1 (0.873) is also above the untuned random forest (0.869), so after tuning,
LightGBM is the best model overall. The price is inference cost: 500 trees with up to 194 leaves are slower than 100 × 31.
This is still well within the API budget (a whole student-test trip is scored in well under a second; see Section 7 timings).
Parameter importance (fANOVA, `tuning_importance.png`): <!-- IMPORTANCE -->

---

## 5. HMM and Viterbi Route Reconstruction

Code: `src/hmm/decoder.py` (model + Viterbi), `src/hmm/evaluate.py` (tuning + comparison), served by `api/pipeline.py`.

### 5.1 Hidden states
At time t the hidden state is **one of the candidate arcs of GPS point t** (≤ 30, the same candidates the classifier scored).
The lattice keeps, for every candidate row, its arc id and its projection fraction along the arc. This is the
candidate-to-arc mapping, so the decoded state index is turned back into an arc id per point.

### 5.2 Emission probabilities
log e_t(a) = w_emis · log max(P_clf(a is the true arc | features), 1e-6), where P_clf is the tuned LightGBM probability
(Section 4). The classifier already uses distance, heading and context, so no separate Gaussian distance term is added.
As an ablation, we also decode with a purely geometric emission exp(−½ (d/10 m)²) (classical Newson & Krumm) to measure
what the classifier contributes.

### 5.3 Transition probabilities
Following Newson & Krumm (2009): a vehicle's network distance between two consecutive fixes should be close to the
straight-line distance between them. For consecutive candidates a (at t−1, fraction f_a) and b (at t, fraction f_b):

* same arc, moving forward: d_route = (f_b − f_a)·L_a
* same arc, moving backward: d_route = back_mult · (f_a − f_b)·L_a (moving backwards on a directed arc is penalised; this is what
  separates an arc from its reverse twin when the vehicle is moving)
* different arcs: d_route = (1 − f_a)·L_a + SP(to(a), from(b)) + f_b·L_b, where SP is the **shortest path on the directed road graph**
  (scipy Dijkstra, bounded to min(3,000 m, 3·max step + 300 m), on the nodes within 800 m of the trajectory)

log T(a→b) = −|d_route − d_gc| / β, where d_gc is the straight-line distance between the two GPS fixes. All pairwise route distances of a
trip are computed once and vectorised (unique (source, target) node pairs, batched Dijkstra), so tuning β, back_mult or
the floor only re-runs the cheap Viterbi pass. The recursion runs in log space with a (K_{t−1} × K_t) max-plus product per step.
The longest train trip (19,313 points) decodes in about a second.

### 5.4 Missing candidates and large GPS gaps
* **No candidates** (0.68 % of train points, 0.09 % of student-test): the chain is cut, the point gets no arc (−1, counted as
  wrong in every metric), and Viterbi restarts on the next point that has candidates.
* **Large time gaps** (dt > gap_s; 22 train trips have gaps over 60 s, one up to 153,064 s): the chain is cut and decoding restarts, because
  the road distance across a long gap is not informative. gap_s is a tuned decoder parameter.

### 5.5 Disconnected road segments
89 of 90 trips have several weakly-connected network components, and one-way arcs create many more strongly-connected
components. If no directed path exists within the search bound (or the target lies in another component), the
transition gets a **floor log-probability** (`log_floor`) instead of −∞. The decoder can therefore still "jump" between components
when the emissions at the following points clearly insist on it, rather than getting stuck or failing. Such jumps appear as
breaks in the continuity metric.

### 5.6 Did HMM decoding improve the route?

<!-- SECTION_5_RESULTS -->

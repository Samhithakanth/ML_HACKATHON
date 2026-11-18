# PathSnap Technical Report — Sections 3, 4 and 5 (Persons 2 and 3)

Every number in this file is read from a CSV in `results/` produced by the scripts listed in the README
(section 6, "Reproducing the Experiments"); the file next to each table is its source. Seeds are fixed (42).

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

Example (student-test trip 00000007): the true arcs average 7.3 m from the fix with a mean heading cosine of 0.84, against 48.8 m and −0.13 for the other candidates.

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

| Model | Family | Key configuration | CV macro F1 | CV top-1 | Test top-1 (classifier) | Test top-1 (+HMM) | Route continuity (classifier / +HMM) | Test macro F1 | Test log loss | Inference (ms / 1k cand.) |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear | C=1.0, L2, lbfgs, scaled | 0.9045 ± 0.0171 | 0.8402 ± 0.0368 | 0.8410 | 0.9225 | 0.9297 / 0.9836 | 0.9067 | 0.0464 | 0.12 |
| Gaussian Naive Bayes | Probabilistic generative | var_smoothing=1e-9, scaled | 0.7441 ± 0.0071 | 0.8076 ± 0.0280 | 0.7717 | 0.9163 | 0.9190 / 0.9833 | 0.7376 | 0.7969 | 0.30 |
| K-Nearest Neighbours | Instance-based | k=25, distance-weighted, 120k-row sample, scaled | 0.8901 ± 0.0193 | 0.8190 ± 0.0384 | 0.8260 | 0.9181 | 0.9183 / 0.9830 | 0.8961 | 0.0577 | 47.68 |
| Random Forest | Bagging ensemble | 300 trees, min_leaf 5, max_features sqrt | 0.9274 ± 0.0174 | 0.8686 ± 0.0383 | 0.8665 | 0.9523 | 0.9391 / 0.9835 | 0.9271 | 0.0347 | 1.89 |
| LightGBM (untuned) | Gradient boosting | defaults: 100 trees, lr 0.1, 31 leaves | 0.9245 ± 0.0177 | 0.8604 ± 0.0350 | 0.8658 | 0.9582 | 0.9220 / 0.9834 | 0.9265 | 0.0427 | 0.33 |
| MLP | Neural network | (128,64) ReLU, Adam 1e-3, early stop (ep. 19), 300k rows | 0.9191 ± 0.0162 | 0.8603 ± 0.0375 | 0.8662 | 0.9348 | 0.9439 / 0.9831 | 0.9225 | 0.0376 | 0.41 |
| **LightGBM (Bayesian-tuned, final)** | Gradient boosting | 500 trees, lr 0.029, 194 leaves, depth 8 (Sec. 4) | 0.9298 | 0.8733 | 0.8820 | 0.9627 | 0.9464 / 0.9835 | 0.9327 | 0.0320 | 3.66 |

* Every model was trained and evaluated; none is N/A on any metric. All of them output probabilities, so log loss applies to all.
* "Test top-1 (+HMM)" decodes every model's probabilities with the same tuned decoder (`configs/hmm.yaml`, tuned on the tuned
  LightGBM's OOF probabilities; see 3.4 for why this favours LightGBM-shaped probabilities).
* Inference: idle-machine benchmark, median of 3 passes over all 366,311 student-test candidate rows (`results/inference_timing.csv`, `src/models/timing.py`).
* The tuned model's CV values come from the same 5 folds (`results/tuning/untuned_vs_tuned.csv`); the fold std is not stored for it.

### 3.2 Per-model analysis

All numbers are student-test unless marked CV. "Train-fit top-1" is measured on 40 sampled training points per
training trip of each fold (`cv_train_top1_mean`); comparing it with CV top-1 is our over/underfitting evidence.
Candidate-level positives are 6.2 % of rows, so recall/precision refer to the positive class (the true arc).

#### 3.2.1 Logistic Regression (linear)
* **Why suitable.** A calibrated linear baseline: most of the signal is monotone in a few features (closer, better aligned,
  present in the neighbouring points), so a linear log-odds model is a reasonable first fit. Its probabilities are directly usable as HMM emissions.
* **Strengths / limitations.** Fast (4 s fit, 0.12 ms per 1k candidates), stable across folds, interpretable coefficients. It cannot
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
  is compared with the stored rows, **47.7 ms per 1k candidates, 143× slower than untuned LightGBM**, and 17.5 s for the student-test set.
  Because of that we stored only a 120k-row subsample. It is also sensitive to feature scaling and to the many uninformative dimensions.
* **Hyperparameters / objective.** StandardScaler, k = 25, distance weighting, Euclidean metric, 120k-row training subsample. There is no loss
  function; the prediction is a weighted vote of the neighbours.
* **Results.** Top-1 0.826, macro F1 0.896, log loss 0.058 (CV top-1 0.819 ± 0.038).
* **Fit.** Train-fit 0.862 vs CV 0.819: a moderate gap (distance weighting makes the training points their own nearest neighbours).
  It is also limited by the subsample, i.e. it is data-starved rather than over-flexible. It performs worse than the tree ensembles, and inference cost rules it out for the API.

#### 3.2.4 Random Forest (bagging ensemble)
* **Why suitable.** Averaged deep trees handle the feature interactions logistic regression misses: heading × speed,
  distance × number of candidates, and context × distance. They need no scaling and are robust to the fill value (150 m).
* **Strengths / limitations.** Best CV top-1 and good calibration (log loss 0.037). But it is slow to train (158 s on all clean trips with 22 cores,
  ~105–155 s per fold), its model file is 74 MB (not committed; regenerated by the script), and inference is 1.9 ms per 1k candidates (5.7× untuned LightGBM).
* **Hyperparameters / objective.** 300 trees, min_samples_leaf 5, max_features sqrt, bootstrap; each tree is grown greedily to minimise Gini impurity.
* **Results.** Top-1 0.867, macro F1 0.927, log loss 0.035 (CV top-1 **0.869 ± 0.038**, macro F1 0.927).
* **Fit.** Train-fit top-1 0.947 vs CV 0.869: the largest gap of all models. It is **mildly overfitting / high variance** (deep trees with leaf size 5),
  although bagging keeps the held-out performance at the top. Test (0.867) matches CV, so the CV estimate is honest.

#### 3.2.5 LightGBM (gradient boosting) — selected for tuning
* **Why suitable.** Boosted trees fit the same interactions as the forest but sequentially correct the remaining errors. They are fast on
  millions of rows and expose a rich hyperparameter space, which makes them a good candidate for Bayesian optimisation.
* **Strengths / limitations.** 2.5 s training, 0.33 ms per 1k candidates, small model (0.16 MB). With the library defaults (100 trees, 31 leaves)
  its log loss (0.043) is worse than the forest's: defaults are not tuned for a 6 %-positive problem with 31 correlated features.
* **Hyperparameters / objective.** Defaults: 100 trees, learning rate 0.1, 31 leaves, unlimited depth, min_child_samples 20,
  no subsampling. Objective: binary log loss (gradient boosting on its gradient/Hessian).
* **Results (untuned).** Top-1 0.866, macro F1 0.927, log loss 0.043 (CV top-1 0.860 ± 0.035, macro F1 0.925).
* **Fit.** Train-fit 0.887 vs CV 0.860 vs test 0.866: a small, healthy gap. Neither clearly over- nor underfitting.

#### 3.2.6 Multi-Layer Perceptron (neural network)
* **Why suitable.** A smooth non-linear function of all features jointly, trained directly on log loss. It is the neural-network family and has a real epoch history.
* **Strengths / limitations.** Good calibration (log loss 0.038) at the cost of the longest training time here (249 s for a capped
  300k-row sample, early-stopped at epoch 19). It needs scaling, and its results depend on the optimiser settings and seed.
* **Hyperparameters / objective.** StandardScaler, hidden layers (128, 64), ReLU, Adam (lr 1e-3), batch 1024, ≤ 30 epochs, early stopping
  on a 10 % validation split (patience 4), 300k-row training subsample. Objective: binary cross-entropy.
* **Results.** Top-1 0.866, macro F1 0.923, log loss 0.038 (CV top-1 0.860 ± 0.038, macro F1 0.919).
* **Fit.** Train-fit 0.873 vs CV 0.860: a small gap. In the loss curve (`models_mlp_loss_curve.png`, final model) the training loss keeps falling
  (0.085 → 0.036) while the validation accuracy peaks at epoch 14 (0.9861) and then flattens. Early stopping therefore ended training at epoch 19,
  before the 30-epoch cap, at the point where more epochs would start to overfit. So the network is **well fitted**, neither over- nor underfitting.

#### Common observation: the per-trip spread is larger than the gap between models
Fold-to-fold standard deviation of CV top-1 is about 0.035–0.038 for every model, larger than the gap between
the four best models (0.860–0.869). Some trips are intrinsically harder (dense networks, long stops, parallel roads), and
every model finds the same trips hard: fold 2 is the best fold and folds 0/1 the worst for every model (`results/model_cv_folds.csv`).
Even the best classifier (tuned LightGBM) is wrong on 11.8 % of student-test points. 44 % of those errors are the reverse twin
(same street, wrong direction) and 16 % an adjacent arc at an intersection (`results/error_analysis.csv`, Section 5.6). These cases
are hard to separate point by point, but easy to separate over a sequence. This is what motivates the HMM in Section 5.

### 3.3 Visualisations

* `results/figures/models_comparison.png`: CV vs student-test top-1, macro F1 and log loss for every model.
* `results/figures/models_train_cv_test.png`: train-fit vs CV vs student-test top-1 (over/underfitting check).
* `results/figures/models_confusion_matrices.png`: student-test candidate-level confusion matrices.
* `results/figures/models_mlp_loss_curve.png`: MLP training loss and validation accuracy per epoch. This is the only model
  with an epoch history; no curves are drawn for the other models.

### 3.4 Model selection

Ranked by the primary metric (CV top-1), the untuned models are: Random Forest 0.869, LightGBM 0.860, MLP 0.860, Logistic
Regression 0.840, KNN 0.819, Gaussian NB 0.808. The four best are within one fold standard deviation (≈ 0.037) of each other, so
CV alone does not separate RF, LightGBM and MLP. We chose **LightGBM for Bayesian optimisation** for these reasons:
1. **It was tied for best quality before tuning.** Its CV top-1 is 0.009 below RF's, a quarter of the fold std. Its student-test top-1 is the same as RF's (0.866 vs 0.867).
2. **Cost makes tuning feasible.** One LightGBM fit takes ~2.5 s, against ~110–150 s for RF and ~250 s for the MLP. Thirty trials of group CV cost
   minutes for LightGBM and many hours for RF on one laptop.
3. **It has the richest search space** (tree count, learning rate, leaves, depth, sampling, L1/L2), i.e. the most to gain from tuning.
   Logistic regression and naive Bayes are underfitting, so tuning them cannot remove their bias.
4. **Deployment.** The model is small and fast enough for the API.

The choice was confirmed after tuning: the tuned LightGBM is the best model on every quality metric, both in CV (top-1 0.873, macro F1 0.930)
and on student-test (top-1 0.882, macro F1 0.933, log loss 0.032). After HMM decoding it is also the best (0.963, table 3.1).
With HMM decoding, the ranking changes (table 3.1): tuned LightGBM 0.963, untuned LightGBM 0.958, Random Forest 0.952,
MLP 0.935, Logistic Regression 0.923, KNN 0.918, Gaussian NB 0.916. Random Forest and the MLP are as good as untuned
LightGBM point by point, but gain less from the decoder. One reason is a confound in our setup: the decoder parameters (notably the
emission weight 0.25) were tuned on LightGBM probabilities, so they suit that model's probability distribution best. A fair comparison of
the decoded routes would re-tune the decoder per model; we did not do that, so the +HMM column compares the models under the final system's decoder.
The classifier choice still matters with the HMM: the best classifier gives the best route, and the gap between the weakest and the strongest
model shrinks from 11 points (0.772 vs 0.882) to 4.6 points (0.916 vs 0.963), but it does not disappear.

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

| Metric | Untuned | Tuned |
|---|---|---|
| CV macro F1 (5-fold, train) | 0.9245 | 0.9298 |
| CV top-1 | 0.8604 | 0.8733 |
| CV log loss | 0.0481 | 0.0351 |
| Student-test top-1 (classifier only) | 0.8658 | 0.8820 |
| Student-test top-1 (+HMM) | 0.9582 | 0.9627 |
| Student-test macro F1 | 0.9265 | 0.9327 |
| Student-test log loss | 0.0427 | 0.0320 |
| Student-test precision / recall | 0.8674 / 0.8539 | 0.8912 / 0.8541 |
| Inference, ms per 1k candidates (idle benchmark) | 0.33 | 3.66 |
| Inference, all 366,311 test candidates (s) | 0.12 | 1.34 |
| Training time on all clean trips (s, same run) | 13.0 | 83.6 |
| Model file (MB) | 0.16 | 4.08 |

(Training times in this table were both measured in the same run of `bayes_opt.py`, which ran while other jobs were active; compare them with each other, not with Section 3.)

The tuned model is better on every quality metric. Its CV top-1 (0.873) is also above the untuned random forest (0.869), so after tuning,
LightGBM is the best model overall. The price is inference cost: 500 trees with up to 194 leaves are slower than 100 × 31.
It is still cheap in absolute terms: all 366,311 student-test candidates are scored in 1.34 s, i.e. 0.33 s for the largest test
trip (00000056, 90,835 candidates). End-to-end API latency, including the HMM, is 0.68 s per trip on average (Section 5.6, "Cost").
Parameter importance (fANOVA over the 27 completed trials, `results/tuning/param_importance.csv`, `tuning_importance.png`): num_leaves 0.22, min_child_samples 0.21, colsample_bytree 0.19, learning_rate 0.16, n_estimators 0.08, reg_lambda 0.04, reg_alpha 0.04, max_depth 0.03, subsample 0.03. Tree capacity and sampling (num_leaves, min_child_samples, colsample_bytree) and the learning rate matter most; max_depth and row subsampling matter least once the others are set.

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
the floor only re-runs the cheap Viterbi pass. The recursion runs in log space with a (K_{t−1} × K_t) max-plus product per step. The Viterbi pass itself costs about
26 ms per 1,000 points on student-test (`decode_s_per_1k_points` in `results/hmm/hmm_vs_classifier_summary.csv`).

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

**Decoder tuning (train only).** Five decoder parameters were tuned with Optuna TPE (30 trials, `results/hmm/decoder_trials.csv`,
`results/figures/hmm_decoder_tuning.png`). The objective was top-1 over all 183,260 clean-train points, decoded from the tuned LightGBM's
**5-fold out-of-fold** probabilities, so the emissions have the same quality as on unseen trips. Student-test was not used.

| Parameter | Search range | Default | Tuned (`configs/hmm.yaml`) |
|---|---|---|---|
| β (transition scale, m) | log 1–100 | 10 | 5.68 |
| w_emis (emission weight) | log 0.2–3 | 1.0 | 0.248 |
| log_floor (no-path transition) | −40 to −5 | −20 | −36.9 |
| back_mult (backward-on-arc cost) | log 1–20 | 4 | 1.11 |
| gap_s (cut the chain above, s) | 10–120 | 60 | 49.6 |

Train OOF top-1: classifier only 0.8532, HMM default 0.9136, HMM tuned **0.9331**. The tuned decoder trusts the road graph more
than the classifier (w_emis 0.25: the emission enters with a quarter of its log weight) and uses a sharper transition (β 5.7 m).
Jumps without a path are almost forbidden (floor −37). The backward penalty became almost neutral (1.11): with a sharp β, moving backwards
along an arc is already expensive, because it makes d_route differ from d_gc.

**Classifier-only vs classifier + HMM on student-test** (`results/hmm/hmm_vs_classifier_summary.csv`, 10 trips, 19,160 points):

| Method | Top-1 | Top-1 (trip mean) | Continuity | Breaks / 1k pts | Edit distance | Route precision | Route recall | Route F1 |
|---|---|---|---|---|---|---|---|---|
| classifier only (tuned LightGBM) | 0.8820 | 0.8931 | 0.9464 | 53.6 | 0.360 | 0.841 | 0.974 | 0.901 |
| classifier + HMM, default decoder | 0.9343 | 0.9390 | 0.9835 | 16.4 | 0.119 | 0.943 | 0.987 | 0.964 |
| **classifier + HMM, tuned decoder** | **0.9627** | **0.9570** | **0.9835** | **16.5** | **0.077** | **0.959** | **0.989** | **0.973** |
| distance-only HMM (no classifier) | 0.9288 | 0.9287 | 0.9817 | 18.3 | 0.121 | 0.923 | 0.981 | 0.950 |

The HMM raises student-test top-1 by **+8.1 points** (0.882 → 0.963), and cuts the normalised edit distance between the collapsed
predicted and reference arc sequences from 0.36 to 0.08. Route precision rises from 0.84 to 0.96: the classifier-only route
contains many spurious arcs (the other direction of the street, side streets at intersections) that the HMM removes. The improvement on
student-test (+8.1 pp) matches the train-OOF improvement (+8.0 pp, 0.853 → 0.933), so the decoder tuning did not overfit the train trips.

*Ablation: does the classifier matter?* A classical HMM with a purely geometric emission (Gaussian in the point-to-arc distance,
σ = 10 m) reaches 0.929. The classifier-driven HMM is 3.4 points better, so heading, kinematic and context features add information the
transition model alone does not have.

*Where the classifier fails and the HMM fixes it* (`results/error_analysis.csv`):

| Share of all student-test points | classifier only | + HMM |
|---|---|---|
| wrong overall | 11.8 % | 3.7 % |
| reverse twin (right street, wrong direction) | 5.2 % | 1.3 % |
| adjacent arc (wrong arc at an intersection) | 1.9 % | 1.5 % |
| other wrong arc | 4.7 % | 1.0 % |
| error rate when speed < 2 m/s | 38.7 % | 12.5 % |
| error rate when moving | 10.6 % | 3.4 % |

The largest single error type of the classifier is the **reverse twin**. When the vehicle is slow, the GPS heading is noise and both
directions look identical point by point. The HMM resolves most of these, because a route cannot flip direction from one second to the next
without a long detour. Points at an intersection improve least: 1.5 % of points are still assigned the arc adjacent to the true one. Our interpretation
(not separately measured) is that near a node the decoder can switch to the next arc a point or two early or late, because the
projection distance is almost the same on both arcs.

**Cases where the HMM did not help (or helped less)**
* **Trip 00000063 (train, OOF)**: the only one of 78 train trips where the HMM is slightly worse (0.8278 → 0.8252). On every student-test
  trip the HMM is better than the classifier (`hmm_vs_classifier_per_trip.csv`).
* **Trips 00000009 and 00000065 (student-test)**: the tuned decoder is *worse than the default decoder* (0.918 → 0.903 and 0.880 →
  0.866), although both beat the classifier. These are the two lowest-scoring test trips. A likely explanation (a hypothesis, not separately tested):
  the tuned decoder's low emission weight makes it stick to a connected path, so where the graph offers two plausible branches
  (parallel roads, many short arcs) it can follow the wrong one for longer before the classifier evidence pulls it back.
  Tuning optimises the average over the train trips, not every individual trip.
* **Remaining continuity "breaks"** (316 on student-test, `results/hmm/break_analysis.csv`): **all 316** are pairs of arcs joined by a
  directed road path of ≤ 300 m. The vehicle crossed one or more short arcs within one GPS interval, which the strict continuity
  metric (same or directly adjacent arc) counts as a break. None is a jump between unconnected parts of the network, so the decoded
  route is fully connected on the road graph.
* **Label-suspect trips 33 and 54**: top-1 only 0.108 → 0.130, because the reference route itself is up to ~130 m away from the GPS (P1's finding).
  The HMM still makes the route more continuous (0.853 → 0.916) and raises route F1 from 0.82 to 0.89. These trips measure
  the quality of the labels, not of the model.

**Cost.** End-to-end latency of the API pipeline (`predict_trip`, the code behind `POST /predict`), measured on an idle machine for all
10 student-test trips (`python -m api.benchmark` → `results/api_latency.csv`): **0.10–1.50 s per trip, mean 0.68 s**. Of that, on average,
40 ms is candidate generation, 421 ms is features + tuned LightGBM, and 197 ms is the HMM (lattice route distances + Viterbi). The slowest trip
is 00000056 (3,958 points). Decoding therefore adds about 41 % to the classifier-only pipeline time (197 ms vs 483 ms for everything else, on average), for +8 points of top-1.

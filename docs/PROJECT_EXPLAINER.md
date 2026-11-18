# PathSnap explained (simple but detailed)

## 1. The problem in plain words
A car records its position with GPS once per second. GPS is not exact: dots wobble by several metres, drift onto the next street, or disappear for a while.
**PathSnap's job: given those wobbly dots and a map, work out which roads the car really drove.** This is called *map matching*.

The map is made of **nodes** (junctions and bends) joined by **arcs** (short pieces of road between two nodes; a long street is many arcs, and a two-way street has one arc per direction).
The answer for a trip is, for every GPS dot, the arc the car was on.

```
node A ----arc 1----> node B ----arc 2----> node C        <- the map
   .      .    .  .       .   .      .                      <- noisy GPS dots; which arc is each dot on?
```

## 2. The big idea: three stages
1. **Candidates.** A map has 100,000+ arcs, so for each GPS dot keep only arcs within 100 m of it (at most the 30 nearest). These are the *candidates*.
2. **Classifier.** A machine-learning model looks at each (dot, candidate) pair and outputs "probability this candidate is the true road".
   Picking the highest-scoring candidate for every dot is the **classifier-only** result.
3. **HMM + Viterbi smoothing.** The classifier judges each dot alone, so it can pick roads that are not even connected to each other.
   A *Hidden Markov Model* adds common sense: consecutive dots should be on roads that connect, and the distance driven along the road should match the distance between the dots.
   *Viterbi* is the algorithm that finds the single best chain of candidates across the whole trip.

Then an **API** (server) runs all of this, and a **UI** (web page) lets an evaluator pick a trip and see the result on a map.

## 3. The data
- 80 **training** trips, 10 **student-test** trips (we can use these to compare models), and 10 hidden **evaluator** trips (used only at the final demo; we never see them).
- Each trip has four files: `.track` (GPS: lon, lat, time), `.nodes` (junction coordinates), `.arcs` (road pieces joining two nodes), `.route` (the arcs the car actually drove, in order: the answer key).
- Every trip has its **own** map, and the trips come from cities all over the world, so each trip is converted to metres with its own local projection (degrees are angles, so you cannot measure distance with them).
- The answer key lists arcs but does not say which dot belongs to which arc. We derive that by aligning dots to the route with a forward-only best-fit (dynamic programming): the car never goes backwards along its route.
- Two training trips (33 and 54) have GPS about 130 m away from the reference route for long stretches, so their answers are unreliable. They are excluded from training and reported separately.
- Candidate quality: with radius 100 m and at most 30 candidates, the true arc is missing from the list only 0.65 % of the time on clean training trips and 0.09 % on student-test.
  (If the true road is not in the list, no model can pick it, so this is a ceiling.)

## 4. The features (what the classifier looks at)
For every (dot, candidate) pair, 31 numbers (`src/features/build.py`), in six groups:
| Group | Examples | Why |
|---|---|---|
| Geometry | distance to the arc, position along the arc, rank among candidates | The true road is usually the closest |
| Relative | gap to the next-nearest candidate, number of candidates | In busy streets many roads are close, so absolute distance is ambiguous |
| Heading | angle between the direction the car moves and the arc's direction | The two directions of a two-way street are the same distance away; direction tells them apart |
| Kinematics | speed, time since previous/next dot, acceleration | Heading is unreliable when the car is almost stopped; handle gaps |
| Topology | how many roads leave/enter the arc's ends, does it have a reverse twin | Junction versus plain road |
| Context | is the same or a connected arc also close at the previous/next dot | The true road is usually continuous over time |

## 4b. Where it can go wrong, and what the team did
- **Reverse twins:** the same street in the other direction. Fixed with heading features.
- **GPS gaps:** if there is no fix for a while, the HMM restarts instead of connecting across the gap.
- **Disconnected maps:** 89 of 90 trips have maps split into several pieces. When no route exists between two candidates, the HMM uses a floor probability instead of a hard "impossible", so it can still bridge.
- **No candidates:** if no road is within 100 m of a dot, the output for that dot is "no prediction" (arc −1) rather than a guess.

## 5. The models compared (student-test = 10 trips, 19,160 points)
Six models were trained with 5-fold cross-validation grouped by trip (so no trip appears in both train and validation):

| Model | Family | CV macro F1 | Student-test top-1 (classifier only) | Student-test top-1 (+HMM) | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Linear | 0.905 | 84.1 % | 92.3 % | 0.046 |
| Gaussian Naive Bayes | Probabilistic | 0.744 | 77.2 % | 91.6 % | 0.797 |
| K-Nearest Neighbours | Instance-based | 0.890 | 82.6 % | 91.8 % | 0.058 |
| LightGBM (untuned) | Gradient boosting | 0.925 | 86.6 % | 95.8 % | 0.043 |
| MLP | Neural network | 0.919 | 86.6 % | 93.5 % | 0.038 |
| **LightGBM (Bayesian-tuned)** | Gradient boosting | **0.930** | **88.2 %** | **96.3 %** | **0.032** |

"Top-1" = for each GPS dot, is the highest-scoring candidate the reference arc? (Source: `results/model_comparison_table.csv`.)
The final model is the tuned LightGBM. Boosted trees do best because the features mix distances and angles with non-linear rules, and they give well-calibrated probabilities, which the HMM needs.

## 6. Bayesian hyperparameter tuning
Instead of trying settings at random, **Bayesian optimisation** (Optuna, TPE) uses the results of earlier trials to choose what to try next. 30 trials on LightGBM, scored by 3-fold grouped cross-validation log loss on clean training trips.
Result (`results/tuning/untuned_vs_tuned.csv`): student-test top-1 86.6 % → 88.2 %, log loss 0.0427 → 0.0320. The price: inference is slower (about 3 → 51 ms per 1,000 rows) and training took 84 s instead of 13 s.

## 7. The HMM, in simple terms
- **Hidden states:** at each dot, one of its candidate arcs.
- **Emission** ("how well does this candidate fit this dot"): the classifier's probability.
- **Transition** ("how plausible is going from candidate A at the last dot to candidate B at this dot"): compare the straight-line distance between the two dots with the distance you would actually drive on the network from A to B. If they roughly match, that is plausible; if not, penalise. Driving backwards on an arc is penalised more.
- **Viterbi:** finds the best chain over the whole trip without trying every combination.
- The decoder settings (`configs/hmm.yaml`) were tuned on out-of-fold training predictions only, not on test data.

Effect (student-test, 10 trips), from `results/hmm/hmm_vs_classifier_summary.csv`:
| Method | Top-1 accuracy | Route continuity | Breaks per 1,000 points |
|---|---|---|---|
| Classifier only | 88.2 % | 94.6 % | 53.6 |
| Classifier + HMM (default settings) | 93.4 % | 98.4 % | 16.4 |
| **Classifier + HMM (tuned settings)** | **96.3 %** | **98.3 %** | 16.5 |
| HMM using only distance (no classifier) | 92.9 % | 98.2 % | 18.3 |

Two honest observations: the HMM adds about 8 percentage points on top of the classifier, and a distance-only HMM already reaches 92.9 %, so a large part of the gain comes from the road-network smoothing rather than from the machine learning alone.
On the two flagged training trips (33, 54) the accuracy is only 11 to 13 % because their reference route is unreliable.

## 8. The API and UI
- **API (FastAPI, laptop 1)** starts by loading the model, the HMM settings, the 10 trips' maps, and the private reference labels. For `POST /predict` it runs: raw files → candidates → 31 features → LightGBM → HMM/Viterbi, then compares with the reference labels **on the server** and returns:
  the predicted route and the classifier-only route (as lines to draw), which GPS points are correct, and metrics (accuracy, route continuity, macro F1, number of breaks, inference time) plus warnings such as "GPS gap of 210 s".
- **The answer key never leaves the server.** The prediction code reads only the track, nodes and arcs files (never `.route`); `.route` is read only at startup to build the private scoring labels, and the API never returns it.
- **UI (laptop 2)**: a Streamlit page (or a plain web page at `/web/`) lists the 10 trips, sends only the trip ID, and draws GPS, network and predicted route on a Folium/Leaflet map next to the metrics.
- Why two laptops? It proves the system is a real client-server system and keeps the answers off the machine that shows the results. Every request appears in the API's log (`logs/api.log`).

Measured by running it: all 10 student-test trips return HTTP 200; per-trip HMM accuracy 86.6 % to 100 %; response time 0.25 s (262 points) to 3.8 s (3,958 points).

## 9. Repo map
```
api/        FastAPI server (main.py endpoints, pipeline.py the prediction function, config.py settings)
ui/         Streamlit app (app.py) and the plain web UI (index.html, app.js, style.css)
src/data/   loading, projection, reference labels, candidate generation, plus Person 1's EDA scripts (p1_*.py)
src/features/ 31-feature extraction            src/models/ the 6 models and the predictor wrapper
src/tuning/ Bayesian optimisation              src/hmm/ Viterbi decoder and route polylines
src/eval/   shared metrics (used by the API and by the experiments)       src/viz/ Folium maps and EDA plots
configs/    api.yaml (paths, ports), hmm.yaml (tuned decoder settings)
models/     saved models and the scaler       results/ all experiment outputs, plots, maps
report/     the report sections               docs/ guides, data notes, this file
Student folder/   the dataset (train and student_test)
```

## 10. Who did what
| Person | Work |
|---|---|
| 1 | Data exploration, per-trip projection, reference labels from the route, candidate generation, flagged bad trips, EDA, report Section 2.2 |
| 2 | Feature engineering (31 features), training and comparing 6 models, report Section 3 |
| 3 | Bayesian optimisation, HMM/Viterbi decoder, shared metrics, report Sections 4 and 5 |
| 4 | FastAPI backend, UI, Folium maps, deployment guide, report Sections 6 and 7 |

## 11. Limits to be honest about
- The "correct answers" are derived from a reference route, not independently verified; two training trips show that the reference can be off by about 130 m.
- Accuracy is measured against those derived labels, on 10 student-test trips only; the hidden trips may behave differently.
- Part of the gain comes from the road-network smoothing, not the learned model (distance-only HMM: 92.9 %).
- The tuned model is slower at inference (about 51 ms per 1,000 rows versus 3 ms untuned). Fine for this demo; it matters for large-scale use.
- The README's AI-use statement ("programming aid") must be updated to describe the real use of AI.

## 12. Questions you may be asked
| Question | Short answer |
|---|---|
| Why a classifier and an HMM? | The classifier scores each dot alone; the HMM makes the whole route consistent and connected |
| What is a candidate, and why limit to 100 m / 30? | A nearby road. The list must be short for the model, yet almost always contain the true road (0.65 % miss) |
| Why is accuracy measured per GPS dot? | The system returns one arc per dot, so top-1 at dot level is exactly what is judged |
| How do you avoid leakage? | Cross-validation grouped by trip; decoder tuned on training predictions only; the hidden trips are never read; the API keeps labels server-side |
| Why did tuning help only a little? | Features already carry most of the signal; log loss improved 25 %, accuracy by 1.6 points |
| Why does the HMM help so much? | Dots near a junction look alike, but only one chain of arcs is physically connected and matches the distance driven |
| What if there is a GPS gap? | The decoder restarts the chain and the API warns "route split here" |

# Summary

PathSnap reconstructs the sequence of road segments a vehicle drove from noisy 1 Hz GPS. For every GPS fix we generate the candidate
arcs within 100 m, score them with a classifier (31 features), and decode one connected route with an HMM / Viterbi over the directed road
graph. A FastAPI server (laptop 1) runs the pipeline; a Streamlit UI (laptop 2) sends only a trip id.

| Result on the 10 student-test trips (19,160 GPS fixes) | Top-1 | Route continuity | Route F1 |
|---|---|---|---|
| Best untuned classifier (Random Forest), classifier only | 0.8665 | 0.9391 | – |
| Bayesian-tuned LightGBM, classifier only | 0.8820 | 0.9464 | 0.901 |
| **Tuned LightGBM + HMM / Viterbi (deployed)** | **0.9627** | **0.9835** | **0.973** |

End-to-end API latency: 0.10–1.50 s per trip (mean 0.68 s). Every number in this report is produced by a script in the repository
(README, "Reproducing the Experiments") and is stored in `results/`.

**Contents:** 2 Dataset and EDA · 3 Model experimentation and comparison · 4 Bayesian hyperparameter optimisation · 5 HMM and Viterbi
route reconstruction · 6 Geographical visualisation (Folium) · 7 API, UI and deployment · 8 Academic integrity and AI-use disclosure

| Member | Responsibility |
|---|---|
| Person 1: [name, SRN] | data loading, EDA, reference labels, candidate generation (Section 2) |
| Person 2: [name, SRN] | features, model experiments (Section 3) |
| Person 3: [name, SRN] | Bayesian optimisation, HMM / Viterbi, metrics (Sections 4, 5) |
| Person 4: [name, SRN] | API, UI, Folium, deployment, report assembly (Sections 6, 7) |

# 8. Academic Integrity and AI-Use Disclosure

**Data handling.** The evaluator partition was never read by any script. Every fit and every model, hyperparameter and decoder choice used the
train partition only, with group-aware CV by trip. Student-test was used for reporting only. In deployment, the reference routes
stay on laptop 1; responses contain predictions and server-side metrics only.

**Reproducibility.** Seeds are fixed (42). Every table and figure is generated from `results/` by the scripts listed in the README.
Numbers that no script reproduced (an earlier placeholder pipeline) were removed before this report was written.

**AI-use disclosure.** *(The team must review this paragraph and make it accurate for every member before submission.)*
AI assistance (Claude, Anthropic) was used to write a large part of the code for Persons 2 and 3: the feature builder, the model comparison,
the Bayesian optimisation, the HMM decoder, the evaluation scripts and tests. It was also used to draft the text of Sections 2–5 and
this assembled report, under the direction of the team member responsible. AI assistance was also used for [team to complete: other parts,
e.g. API/UI code or report text for Persons 1 and 4]. All results come from running that code on the provided data; none were typed in or
estimated. The team has reviewed the code and the results, and every member is prepared to explain the full pipeline in the viva.

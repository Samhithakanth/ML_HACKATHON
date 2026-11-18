# PathSnap Consolidated Technical Report — Master Assembly Guide

This document defines the consolidated technical report structure in exact accordance with the **PathSnap Deliverables Specification (Sections 1–8)**. Each teammate owns their assigned sections:

| Section # | Title | Lead Owner | Associated Files & Artifacts |
|---|---|---|---|
| **Frontmatter** | Title, Team Details, Abstract, Table of Contents | P4 (Assembly) | `README.md` |
| **Section 2** | Dataset Understanding & Exploratory Data Analysis | **Person 1** | `notebooks/EDA.ipynb`, `results/figures/gps_sampling_distribution.png`, `results/dataset_inventory.csv` |
| **Section 3** | Model Experimentation & Comparison (6 models) | **Person 2** | `src/features/`, `src/models/train_compare.py`, `results/model_comparison_table.csv`, `results/figures/models_*.png`, `report/REPORT_SECTIONS_3_4_5.md` |
| **Section 4** | Bayesian Hyperparameter Optimisation (Optuna) | **Person 3** | `src/tuning/bayes_opt.py`, `results/tuning/*.csv`, `results/figures/tuning_*.png`, `report/REPORT_SECTIONS_3_4_5.md` |
| **Section 5** | HMM & Viterbi Route Reconstruction | **Person 3** | `src/hmm/decoder.py`, `src/hmm/evaluate.py`, `configs/hmm.yaml`, `results/hmm/*.csv`, `results/figures/hmm_*.png`, `report/REPORT_SECTIONS_3_4_5.md` |
| **Section 6** | Geographical Visualisation Using Folium | **Person 4** | `src/viz/folium_map.py`, `results/maps/*.html`, `report/REPORT_SECTION_6_AND_7.md` |
| **Section 7** | API, UI and Two-Laptop Deployment | **Person 4** | `api/`, `ui/`, `logs/api.log`, `report/REPORT_SECTION_6_AND_7.md` |
| **Section 8** | Academic Integrity & AI-Use Disclosure | All Members | `README.md`, Section 8 Policy Statement |

---

## Pre-Submission Verification Checklist for Final PDF

- [x] Single consolidated PDF submitted per team (no separate files per model).
- [x] All 5 distinct algorithm families documented with actual experiment numbers.
- [x] Bayesian optimization trials and search spaces recorded.
- [x] HMM vs Classifier-only comparison table included.
- [x] Folium interactive maps generated and screenshots included.
- [x] Screenshots of UI and API terminal logs included.
- [x] Every number in the tables matches the CSVs in `results/`.
- [x] AI-use disclosure statement included.

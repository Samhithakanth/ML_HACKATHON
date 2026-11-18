"""
PathSnap Model Experimentation & Route Reconstruction Pipeline
Implements:
1. Spatial candidate generation & 10-feature extraction.
2. Training & evaluation of 5 distinct models across algorithm families:
   - Logistic Regression (Linear & Probabilistic)
   - K-Nearest Neighbors (Instance-Based)
   - Random Forest (Ensemble / Bagging)
   - LightGBM (Gradient Boosting)
   - MLP Classifier (Neural Network)
3. Group-aware Cross-Validation on training trips (grouped by trip_id).
4. Out-of-sample evaluation on the 10 student-testing trips:
   - Observation Top-1 Accuracy
   - Macro F1, Precision, Recall, Log Loss
   - Route Continuity & Disconnected Breaks
   - Inference Latency
5. Bayesian Hyperparameter Optimization (Optuna) on winning model.
6. HMM + Viterbi decoding and comparative analysis.
7. Saves artifacts, comparison tables, and plots.
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import scipy.spatial
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# Scikit-learn & LightGBM imports
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
import lightgbm as lgb
import joblib

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "Student folder", "train")
TEST_DIR = os.path.join(BASE_DIR, "Student folder", "student_test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------- GEOMETRY & FEATURE EXTRACTION -----------------

def pt_to_segment(p, a, b):
    """Computes perpendicular distance and projection factor t for point p to segment ab."""
    ab = b - a
    l2 = np.dot(ab, ab)
    if l2 == 0:
        return np.linalg.norm(p - a), 0.0, np.linalg.norm(p - a), np.linalg.norm(p - b)
    t = np.clip(np.dot(p - a, ab) / l2, 0.0, 1.0)
    proj = a + t * ab
    dist = np.linalg.norm(p - proj)
    d_u = np.linalg.norm(p - a)
    d_v = np.linalg.norm(p - b)
    return dist, t, d_u, d_v

def angle_between_vectors(v1, v2):
    """Computes angle difference in radians [0, pi] between two 2D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    dot = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(dot))

def extract_trip_features(tid, partition="train", max_candidates=15):
    """Extracts candidate features and ground-truth labels for a given trip."""
    base_dir = TRAIN_DIR if partition == "train" else TEST_DIR
    tdir = os.path.join(base_dir, tid)
    
    track = np.loadtxt(os.path.join(tdir, f"{tid}.track"))
    nodes = np.loadtxt(os.path.join(tdir, f"{tid}.nodes"))
    arcs = np.loadtxt(os.path.join(tdir, f"{tid}.arcs"), dtype=int)
    route_file = os.path.join(tdir, f"{tid}.route")
    route = np.loadtxt(route_file, dtype=int) if os.path.exists(route_file) else None
    
    mean_lat = np.mean(track[:, 1])
    scale = np.array([111320.0 * np.cos(np.radians(mean_lat)), 110540.0])
    
    nodes_m = nodes * scale
    track_m = track[:, :2] * scale
    ts = track[:, 2]
    
    # Precompute arc geometries
    arc_u = nodes_m[arcs[:, 0]]
    arc_v = nodes_m[arcs[:, 1]]
    arc_mid = (arc_u + arc_v) / 2.0
    arc_p1 = arc_u * 0.75 + arc_v * 0.25
    arc_p2 = arc_u * 0.25 + arc_v * 0.75
    arc_lengths = np.linalg.norm(arc_v - arc_u, axis=1)
    
    # Precompute node degrees
    from collections import Counter
    node_degrees = Counter(arcs[:, 0]) + Counter(arcs[:, 1])
    
    # Sample points for spatial querying: 4 points per arc
    sampled_pts = np.vstack([arc_u, arc_v, arc_p1, arc_p2])
    arc_ids_map = np.tile(np.arange(len(arcs)), 4)
    tree = scipy.spatial.cKDTree(sampled_pts)
    
    # Ground truth mapping if route available
    true_arcs = []
    if route is not None:
        r_arcs = arcs[route]
        curr_r_idx = 0
        for i, pt in enumerate(track_m):
            w_start = max(0, curr_r_idx - 2)
            w_end = min(len(route), curr_r_idx + 10)
            best_dist = 1e9
            best_arc = route[curr_r_idx]
            best_k = curr_r_idx
            for k in range(w_start, w_end):
                u, v = r_arcs[k]
                d, _, _, _ = pt_to_segment(pt, nodes_m[u], nodes_m[v])
                if d < best_dist:
                    best_dist = d
                    best_arc = route[k]
                    best_k = k
            true_arcs.append(best_arc)
            curr_r_idx = best_k
            
    # Compute GPS headings and velocities
    n_pts = len(track)
    gps_headings = np.zeros((n_pts, 2))
    speeds = np.zeros(n_pts)
    for i in range(n_pts):
        prev_idx = max(0, i - 1)
        next_idx = min(n_pts - 1, i + 1)
        dt = ts[next_idx] - ts[prev_idx]
        disp = track_m[next_idx] - track_m[prev_idx]
        if np.linalg.norm(disp) > 0:
            gps_headings[i] = disp / np.linalg.norm(disp)
        if dt > 0:
            speeds[i] = np.linalg.norm(disp) / dt
            
    # Query nearest candidates per GPS point
    _, indices = tree.query(track_m, k=min(len(sampled_pts), 35))
    
    rows = []
    obs_info = [] # store observation metadata for Top-1 calculation
    
    for i in range(n_pts):
        pt = track_m[i]
        cands = list(dict.fromkeys(arc_ids_map[indices[i]]))[:max_candidates]
        true_arc = true_arcs[i] if route is not None else -1
        
        # Ensure true arc is included in candidate training set if missed
        if partition == "train" and true_arc != -1 and true_arc not in cands:
            cands.append(true_arc)
            
        obs_cands = []
        for arc_id in cands:
            u, v = arcs[arc_id]
            u_pos = nodes_m[u]
            v_pos = nodes_m[v]
            
            d_perp, t_proj, d_u, d_v = pt_to_segment(pt, u_pos, v_pos)
            arc_vec = v_pos - u_pos
            angle_diff = angle_between_vectors(gps_headings[i], arc_vec)
            
            label = 1 if (route is not None and arc_id == true_arc) else 0
            
            feat = [
                d_perp,                                # Feature 0: Perpendicular distance (m)
                angle_diff,                            # Feature 1: Heading difference (rad)
                t_proj,                                # Feature 2: Projection ratio [0, 1]
                arc_lengths[arc_id],                   # Feature 3: Road segment length (m)
                d_u,                                   # Feature 4: Distance to start node (m)
                d_v,                                   # Feature 5: Distance to end node (m)
                speeds[i],                             # Feature 6: Vehicle speed (m/s)
                float(node_degrees[u]),                # Feature 7: From-node degree
                float(node_degrees[v]),                # Feature 8: To-node degree
                abs(d_u - d_v)                         # Feature 9: Node symmetry delta (m)
            ]
            rows.append({
                "trip_id": tid,
                "obs_id": i,
                "arc_id": arc_id,
                "features": feat,
                "label": label
            })
            obs_cands.append(arc_id)
            
        obs_info.append({
            "obs_id": i,
            "true_arc": true_arc,
            "candidates": obs_cands
        })
        
    return rows, obs_info, arcs, nodes_m, route

# ----------------- MAIN EXPERIMENT RUNNER -----------------

def run_experiments():
    print("=" * 60)
    print("STEP 1: Extracting Candidates and Features across Dataset")
    print("=" * 60)
    
    # Select 18 representative training trips for balanced, robust multi-model training
    all_train_trips = sorted(os.listdir(TRAIN_DIR))
    np.random.seed(42)
    selected_train = np.random.choice(all_train_trips, size=18, replace=False).tolist()
    
    train_records = []
    train_obs = {}
    for tid in selected_train:
        recs, obs, _, _, _ = extract_trip_features(tid, "train", max_candidates=12)
        train_records.extend(recs)
        train_obs[tid] = obs
        print(f"Processed Train Trip {tid}: {len(recs)} candidate instances")
        
    # Extract features for all 10 student-testing trips
    test_trips = sorted(os.listdir(TEST_DIR))
    test_records = []
    test_obs = {}
    test_trip_meta = {}
    for tid in test_trips:
        recs, obs, arcs, nodes_m, route = extract_trip_features(tid, "student_test", max_candidates=12)
        test_records.extend(recs)
        test_obs[tid] = obs
        test_trip_meta[tid] = {"arcs": arcs, "nodes_m": nodes_m, "route": route}
        print(f"Processed Student-Test Trip {tid}: {len(recs)} candidate instances")
        
    # Build matrices
    X_train = np.array([r["features"] for r in train_records])
    y_train = np.array([r["label"] for r in train_records])
    groups_train = np.array([r["trip_id"] for r in train_records])
    
    X_test = np.array([r["features"] for r in test_records])
    y_test = np.array([r["label"] for r in test_records])
    test_df = pd.DataFrame(test_records)
    
    # Scale features for distance/neural models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    
    print(f"\nTraining Set: {len(X_train)} samples ({np.sum(y_train)} positives), {len(selected_train)} trips")
    print(f"Student-Testing Set: {len(X_test)} samples ({np.sum(y_test)} positives), 10 trips")
    
    # ----------------- 5 DISTINCT MODELS -----------------
    models = {
        "Logistic Regression": {
            "family": "Linear & Probabilistic",
            "config": "C=1.0, solver='lbfgs', max_iter=500",
            "model": LogisticRegression(C=1.0, solver='lbfgs', max_iter=500, random_state=42),
            "use_scaled": True
        },
        "K-Nearest Neighbors": {
            "family": "Instance-Based",
            "config": "k=11, weights='distance', metric='euclidean'",
            "model": KNeighborsClassifier(n_neighbors=11, weights='distance'),
            "use_scaled": True
        },
        "Random Forest": {
            "family": "Ensemble (Bagging)",
            "config": "n_estimators=100, max_depth=12, n_jobs=-1",
            "model": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
            "use_scaled": False
        },
        "LightGBM": {
            "family": "Gradient Boosting",
            "config": "n_estimators=150, lr=0.08, num_leaves=31",
            "model": lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, num_leaves=31, random_state=42, n_jobs=-1),
            "use_scaled": False
        },
        "MLP Classifier": {
            "family": "Neural Networks",
            "config": "layers=(64, 32), act='relu', early_stopping=True",
            "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True),
            "use_scaled": True
        }
    }
    
    print("\n" + "=" * 60)
    print("STEP 2: GroupKFold Cross-Validation on Train Partition")
    print("=" * 60)
    
    gkf = GroupKFold(n_splits=5)
    cv_scores = {}
    
    for name, m_info in models.items():
        clf = m_info["model"]
        X_data = X_train_scaled if m_info["use_scaled"] else X_train
        f1_folds = []
        for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_data, y_train, groups=groups_train)):
            clf_fold = clf.__class__(**clf.get_params())
            clf_fold.fit(X_data[trn_idx], y_train[trn_idx])
            val_preds = clf_fold.predict(X_data[val_idx])
            f1_folds.append(f1_score(y_train[val_idx], val_preds, average='macro'))
        cv_macro_f1 = float(np.mean(f1_folds))
        cv_scores[name] = cv_macro_f1
        print(f"[{name}] 5-Fold Group CV Macro F1: {cv_macro_f1:.4f}")
        
    print("\n" + "=" * 60)
    print("STEP 3: Fitting Models & Evaluating on 10 Student-Test Trips")
    print("=" * 60)
    
    results_summary = []
    fitted_models = {}
    predictions_per_model = {}
    
    for name, m_info in models.items():
        t0 = time.time()
        clf = m_info["model"]
        X_trn = X_train_scaled if m_info["use_scaled"] else X_train
        X_tst = X_test_scaled if m_info["use_scaled"] else X_test
        
        clf.fit(X_trn, y_train)
        fit_time_s = time.time() - t0
        fitted_models[name] = clf
        
        # Save model artifact
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        joblib.dump(clf, os.path.join(MODELS_DIR, f"{clean_name}.joblib"))
        
        # Test inference
        t1 = time.time()
        preds = clf.predict(X_tst)
        probs = clf.predict_proba(X_tst)[:, 1] if hasattr(clf, "predict_proba") else None
        infer_time_ms = int(round((time.time() - t1) * 1000.0))
        predictions_per_model[name] = probs
        
        # Calculate Classification Metrics
        f1_macro = f1_score(y_test, preds, average='macro')
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        loss = log_loss(y_test, probs) if probs is not None else np.nan
        
        # Calculate Observation Top-1 Accuracy across the 10 student trips
        test_df["pred_score"] = probs if probs is not None else preds
        correct_top1 = 0
        total_obs = 0
        
        # Route Continuity calculation:
        total_transitions = 0
        connected_transitions = 0
        
        for tid in test_trips:
            t_slice = test_df[test_df["trip_id"] == tid]
            obs_groups = t_slice.groupby("obs_id")
            arcs = test_trip_meta[tid]["arcs"]
            prev_arc = None
            
            for obs_id, grp in obs_groups:
                best_row = grp.loc[grp["pred_score"].idxmax()]
                if best_row["label"] == 1:
                    correct_top1 += 1
                total_obs += 1
                
                # Check topological continuity
                cur_arc = int(best_row["arc_id"])
                if prev_arc is not None:
                    total_transitions += 1
                    # Connected if staying on same arc or to_node(prev) == from_node(cur)
                    if cur_arc == prev_arc or arcs[prev_arc, 1] == arcs[cur_arc, 0]:
                        connected_transitions += 1
                prev_arc = cur_arc
                
        top1_acc = float(correct_top1 / total_obs)
        route_continuity = float(connected_transitions / total_transitions)
        
        results_summary.append({
            "Model": name,
            "Algorithm Family": m_info["family"],
            "Key Configuration": m_info["config"],
            "CV Macro F1-score": round(cv_scores[name], 4),
            "Top-1 Accuracy": round(top1_acc, 4),
            "Route continuity": round(route_continuity, 4),
            "Student-Test Macro F1-score": round(f1_macro, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "Log Loss": round(loss, 4) if not np.isnan(loss) else "N/A",
            "Inference Time": f"{infer_time_ms} ms",
            "Training Time": f"{fit_time_s:.2f} s"
        })
        
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv(os.path.join(FIG_DIR, "model_comparison_table.csv"), index=False)
    print("\n" + "=" * 60)
    print("CONSOLIDATED MODEL COMPARISON TABLE:")
    print("=" * 60)
    print(df_results[["Model", "Algorithm Family", "CV Macro F1-score", "Top-1 Accuracy", "Route continuity", "Student-Test Macro F1-score", "Log Loss", "Inference Time"]].to_string(index=False))
    
    # ----------------- PLOT COMPARISON CHARTS -----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # CV vs Test Macro F1
    x = np.arange(len(df_results))
    width = 0.35
    axes[0].bar(x - width/2, df_results["CV Macro F1-score"], width, label="CV Macro F1", color="#38bdf8", alpha=0.9)
    axes[0].bar(x + width/2, df_results["Student-Test Macro F1-score"], width, label="Student-Test Macro F1", color="#0284c7", alpha=0.9)
    axes[0].set_ylabel("Macro F1-Score", fontsize=10)
    axes[0].set_title("Cross-Validation vs Student-Test Macro F1", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_results["Model"], rotation=20, ha="right", fontsize=9)
    axes[0].set_ylim(0.65, 1.0)
    axes[0].legend(frameon=True, facecolor="#f8fafc")
    
    # Top-1 Accuracy and Route Continuity
    axes[1].bar(x - width/2, df_results["Top-1 Accuracy"], width, label="Top-1 Accuracy", color="#34d399", alpha=0.9)
    axes[1].bar(x + width/2, df_results["Route continuity"], width, label="Route Continuity (Classifier-Only)", color="#f59e0b", alpha=0.9)
    axes[1].set_ylabel("Score Ratio", fontsize=10)
    axes[1].set_title("Top-1 Accuracy & Route Continuity Comparison", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_results["Model"], rotation=20, ha="right", fontsize=9)
    axes[1].set_ylim(0.65, 1.0)
    axes[1].legend(frameon=True, facecolor="#f8fafc")
    
    plt.tight_layout()
    chart_p = os.path.join(FIG_DIR, "model_performance_comparison.png")
    plt.savefig(chart_p)
    plt.close()
    print(f"\nSaved comparison chart to {chart_p}")
    
    # ----------------- BAYESIAN HYPERPARAMETER OPTIMIZATION -----------------
    print("\n" + "=" * 60)
    print("STEP 4: Bayesian Hyperparameter Optimization (Optuna on LightGBM)")
    print("=" * 60)
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 250),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1
        }
        
        clf_trial = lgb.LGBMClassifier(**params)
        f1_cv = []
        for trn_i, val_i in gkf.split(X_train, y_train, groups=groups_train):
            clf_trial.fit(X_train[trn_i], y_train[trn_i])
            preds_val = clf_trial.predict(X_train[val_i])
            f1_cv.append(f1_score(y_train[val_i], preds_val, average="macro"))
        return float(np.mean(f1_cv))
        
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=20, timeout=120)
    
    best_params = study.best_params
    print(f"\nBest Optuna Trial (F1 = {study.best_value:.4f}):")
    print(json.dumps(best_params, indent=2))
    
    # Train Tuned LightGBM
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    tuned_lgb = lgb.LGBMClassifier(**best_params)
    tuned_lgb.fit(X_train, y_train)
    joblib.dump(tuned_lgb, os.path.join(MODELS_DIR, "lightgbm_tuned_v1.joblib"))
    
    # Evaluate Tuned Model on Student-Testing Set
    t0 = time.time()
    tuned_preds = tuned_lgb.predict(X_test)
    tuned_probs = tuned_lgb.predict_proba(X_test)[:, 1]
    tuned_infer_time_ms = int(round((time.time() - t0) * 1000.0))
    
    tuned_f1 = f1_score(y_test, tuned_preds, average='macro')
    tuned_loss = log_loss(y_test, tuned_probs)
    
    test_df["tuned_score"] = tuned_probs
    c_top1 = 0
    t_obs = 0
    for tid in test_trips:
        for obs_id, grp in test_df[test_df["trip_id"] == tid].groupby("obs_id"):
            best_row = grp.loc[grp["tuned_score"].idxmax()]
            if best_row["label"] == 1:
                c_top1 += 1
            t_obs += 1
    tuned_top1 = float(c_top1 / t_obs)
    
    untuned_row = df_results[df_results["Model"] == "LightGBM"].iloc[0]
    
    tuning_table = pd.DataFrame([
        {
            "Configuration": "Untuned Baseline (Default LightGBM)",
            "CV Macro F1": untuned_row["CV Macro F1-score"],
            "Student-Test Top-1 Accuracy": untuned_row["Top-1 Accuracy"],
            "Student-Test Macro F1": untuned_row["Student-Test Macro F1-score"],
            "Log Loss": untuned_row["Log Loss"],
            "Inference Time": untuned_row["Inference Time"]
        },
        {
            "Configuration": "Bayesian Tuned (Optuna TPE LightGBM)",
            "CV Macro F1": round(study.best_value, 4),
            "Student-Test Top-1 Accuracy": round(tuned_top1, 4),
            "Student-Test Macro F1": round(tuned_f1, 4),
            "Log Loss": round(tuned_loss, 4),
            "Inference Time": f"{tuned_infer_time_ms} ms"
        }
    ])
    tuning_table.to_csv(os.path.join(FIG_DIR, "bayesian_tuning_comparison.csv"), index=False)
    print("\n--- TUNED VS UNTUNED COMPARISON TABLE ---")
    print(tuning_table.to_string(index=False))
    
    # ----------------- HMM + VITERBI ROUTE RECONSTRUCTION -----------------
    print("\n" + "=" * 60)
    print("STEP 5: HMM + Viterbi Route Reconstruction vs Classifier-Only")
    print("=" * 60)
    
    # Evaluate HMM on all 10 student trips
    hmm_results = []
    
    for tid in test_trips:
        arcs = test_trip_meta[tid]["arcs"]
        nodes_m = test_trip_meta[tid]["nodes_m"]
        route = test_trip_meta[tid]["route"]
        t_slice = test_df[test_df["trip_id"] == tid]
        obs_groups = [grp for _, grp in t_slice.groupby("obs_id")]
        
        # Classifier-only path
        clf_selected_arcs = []
        for grp in obs_groups:
            best_arc = int(grp.loc[grp["tuned_score"].idxmax()]["arc_id"])
            clf_selected_arcs.append(best_arc)
            
        # Viterbi decoding
        # States at step t: candidates in grp
        # Emissions: softmax or calibrated prob
        # Transitions: high if connected in graph, penalty otherwise
        n_steps = len(obs_groups)
        V = [{}] # state -> log_prob
        path = {}
        
        # Initialize
        init_grp = obs_groups[0]
        for _, row in init_grp.iterrows():
            arc_id = int(row["arc_id"])
            p_emit = max(1e-5, float(row["tuned_score"]))
            V[0][arc_id] = np.log(p_emit)
            path[arc_id] = [arc_id]
            
        for t in range(1, n_steps):
            V.append({})
            new_path = {}
            cur_grp = obs_groups[t]
            
            for _, row in cur_grp.iterrows():
                cur_arc = int(row["arc_id"])
                p_emit = max(1e-5, float(row["tuned_score"]))
                
                # Maximize over previous candidates
                best_prob = -1e9
                best_prev = None
                for prev_arc, prev_prob in V[t-1].items():
                    # Transition score:
                    if cur_arc == prev_arc:
                        log_trans = -0.1 # highly likely to stay on same arc
                    elif arcs[prev_arc, 1] == arcs[cur_arc, 0]:
                        log_trans = -0.3 # valid forward network transition
                    else:
                        # Topological break penalty
                        log_trans = -6.0
                        
                    tot_prob = prev_prob + log_trans + np.log(p_emit)
                    if tot_prob > best_prob:
                        best_prob = tot_prob
                        best_prev = prev_arc
                        
                V[t][cur_arc] = best_prob
                new_path[cur_arc] = path[best_prev] + [cur_arc] if best_prev in path else [cur_arc]
            path = new_path
            
        # Optimal sequence
        final_best_arc = max(V[-1], key=V[-1].get)
        hmm_selected_arcs = path[final_best_arc]
        
        # Accuracy & Continuity
        true_arcs_trip = [test_obs[tid][i]["true_arc"] for i in range(n_steps)]
        
        clf_acc = np.mean([1 if a == t else 0 for a, t in zip(clf_selected_arcs, true_arcs_trip)])
        hmm_acc = np.mean([1 if a == t else 0 for a, t in zip(hmm_selected_arcs, true_arcs_trip)])
        
        # Continuity
        clf_conn = sum(1 for i in range(len(clf_selected_arcs)-1) if clf_selected_arcs[i] == clf_selected_arcs[i+1] or arcs[clf_selected_arcs[i], 1] == arcs[clf_selected_arcs[i+1], 0])
        clf_cont = clf_conn / max(1, len(clf_selected_arcs) - 1)
        
        hmm_conn = sum(1 for i in range(len(hmm_selected_arcs)-1) if hmm_selected_arcs[i] == hmm_selected_arcs[i+1] or arcs[hmm_selected_arcs[i], 1] == arcs[hmm_selected_arcs[i+1], 0])
        hmm_cont = hmm_conn / max(1, len(hmm_selected_arcs) - 1)
        
        hmm_results.append({
            "trip_id": tid,
            "n_pts": n_steps,
            "classifier_only_acc": round(clf_acc, 4),
            "hmm_acc": round(hmm_acc, 4),
            "acc_improvement": round(hmm_acc - clf_acc, 4),
            "classifier_only_continuity": round(clf_cont, 4),
            "hmm_continuity": round(hmm_cont, 4),
            "continuity_improvement": round(hmm_cont - clf_cont, 4)
        })
        
    df_hmm = pd.DataFrame(hmm_results)
    df_hmm.to_csv(os.path.join(FIG_DIR, "hmm_route_reconstruction_results.csv"), index=False)
    print("\n--- HMM VS CLASSIFIER-ONLY PER-TRIP RESULTS ---")
    print(df_hmm.to_string(index=False))
    print(f"\nMean Classifier-Only Accuracy: {df_hmm['classifier_only_acc'].mean():.4f}")
    print(f"Mean HMM Decoded Accuracy: {df_hmm['hmm_acc'].mean():.4f} (+{(df_hmm['hmm_acc'].mean() - df_hmm['classifier_only_acc'].mean())*100:.2f}%)")
    print(f"Mean Classifier-Only Continuity: {df_hmm['classifier_only_continuity'].mean():.4f}")
    print(f"Mean HMM Decoded Continuity: {df_hmm['hmm_continuity'].mean():.4f} (+{(df_hmm['hmm_continuity'].mean() - df_hmm['classifier_only_continuity'].mean())*100:.2f}%)")
    
    print("\nALL EXPERIMENTS & PIPELINE COMPLETE!")

if __name__ == "__main__":
    run_experiments()

"""
src/hmm/viterbi.py
Hidden Markov Model & Viterbi sequential route decoding.
Optimally reconstructs connected trajectories across the road network graph.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


def viterbi_decode(
    candidates_per_point: List[List[int]],
    candidate_scores: np.ndarray,
    arcs: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    hmm_config: Optional[Dict[str, Any]] = None
) -> Tuple[List[int], List[int]]:
    """
    Decodes the most likely road segment sequence using the Viterbi algorithm.

    Returns:
        (hmm_selected_arcs: List[int],
         classifier_only_arcs: List[int])
    """
    if hmm_config is None:
        hmm_config = {}

    stay_log_trans = hmm_config.get("stay_log_trans", -0.1)
    connected_log_trans = hmm_config.get("connected_log_trans", -0.3)
    disconnected_log_trans = hmm_config.get("disconnected_log_trans", -6.0)
    min_emission_prob = hmm_config.get("min_emission_prob", 1e-5)
    gap_threshold = hmm_config.get("gap_split_threshold_s", 5.0)

    n_steps = len(candidates_per_point)
    if n_steps == 0:
        return [], []

    # Map flat scores back to per-point candidate structure
    cursor = 0
    obs_candidates = []
    obs_scores = []
    classifier_only_arcs = []

    for i in range(n_steps):
        cands = candidates_per_point[i]
        n_c = len(cands)
        scores = candidate_scores[cursor: cursor + n_c] if cursor + n_c <= len(candidate_scores) else np.zeros(n_c)
        cursor += n_c

        obs_candidates.append(cands)
        obs_scores.append(scores)

        # Classifier-only selects candidate with maximum score
        if n_c > 0:
            best_idx = int(np.argmax(scores))
            classifier_only_arcs.append(int(cands[best_idx]))
        else:
            classifier_only_arcs.append(0)

    # Viterbi Recurrence
    V = [{}]  # state -> log_prob
    path = {}

    # Initialize at t = 0
    init_cands = obs_candidates[0]
    init_scores = obs_scores[0]
    for c_id, score in zip(init_cands, init_scores):
        p_emit = max(min_emission_prob, float(score))
        V[0][c_id] = np.log(p_emit)
        path[c_id] = [c_id]

    # Recurrence for t = 1 ... n_steps - 1
    for t in range(1, n_steps):
        V.append({})
        new_path = {}
        cur_cands = obs_candidates[t]
        cur_scores = obs_scores[t]

        is_gap = False
        if timestamps is not None and t < len(timestamps):
            if timestamps[t] - timestamps[t - 1] > gap_threshold:
                is_gap = True

        for cur_arc, score in zip(cur_cands, cur_scores):
            p_emit = max(min_emission_prob, float(score))
            log_emit = np.log(p_emit)

            best_prob = -1e12
            best_prev = None

            for prev_arc, prev_prob in V[t - 1].items():
                if is_gap:
                    # In a large GPS gap, decouple topological transition expectation
                    log_trans = -1.0
                elif cur_arc == prev_arc:
                    log_trans = stay_log_trans
                elif (
                    prev_arc < len(arcs)
                    and cur_arc < len(arcs)
                    and arcs[prev_arc, 1] == arcs[cur_arc, 0]
                ):
                    log_trans = connected_log_trans
                else:
                    log_trans = disconnected_log_trans

                total_prob = prev_prob + log_trans + log_emit
                if total_prob > best_prob:
                    best_prob = total_prob
                    best_prev = prev_arc

            V[t][cur_arc] = best_prob
            if best_prev is not None and best_prev in path:
                new_path[cur_arc] = path[best_prev] + [cur_arc]
            else:
                new_path[cur_arc] = [cur_arc]

        # Handle empty candidates safeguard
        if not V[t]:
            prev_best = max(V[t - 1], key=V[t - 1].get)
            V[t][prev_best] = V[t - 1][prev_best]
            new_path[prev_best] = path[prev_best] + [prev_best]

        path = new_path

    # Traceback optimal state
    final_best_arc = max(V[-1], key=V[-1].get)
    hmm_selected_arcs = path[final_best_arc]

    return hmm_selected_arcs, classifier_only_arcs


def build_route_polylines(
    arcs_seq: List[int],
    arcs: np.ndarray,
    nodes: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    gap_threshold: float = 5.0
) -> Tuple[List[List[List[float]]], int]:
    """
    Constructs polylines from an arc sequence, converting to [lat, lon] coordinates.
    Splits into separate paths whenever there is a topological break or GPS gap.
    Returns:
        (paths: list of [[lat, lon], ...], unique_arcs_count: int)
    """
    if len(arcs_seq) == 0:
        return [], 0

    node_lats = nodes[:, 1]
    node_lons = nodes[:, 0]

    paths = []
    curr_path = []
    last_node = None
    unique_arcs = set()

    for i, arc_id in enumerate(arcs_seq):
        if arc_id >= len(arcs):
            continue
        unique_arcs.add(arc_id)
        u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])

        # Check gap
        has_gap = False
        if timestamps is not None and i > 0 and i < len(timestamps):
            if timestamps[i] - timestamps[i - 1] > gap_threshold:
                has_gap = True

        # Check topological break
        is_break = (last_node is not None and last_node != u) or has_gap

        if is_break:
            if len(curr_path) >= 2:
                paths.append(curr_path)
            curr_path = [
                [float(node_lats[u]), float(node_lons[u])],
                [float(node_lats[v]), float(node_lons[v])]
            ]
        else:
            if not curr_path:
                curr_path.append([float(node_lats[u]), float(node_lons[u])])
            curr_path.append([float(node_lats[v]), float(node_lons[v])])

        last_node = v

    if len(curr_path) >= 2:
        paths.append(curr_path)

    return paths, len(unique_arcs)

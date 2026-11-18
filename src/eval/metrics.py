"""
src/eval/metrics.py
Shared metric definitions for PathSnap:
- Pointwise Top-1 accuracy
- Macro F1-score
- Route continuity
- Disconnected break count
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_top1_accuracy(predicted_arcs, reference_arcs) -> float:
    """Computes observation-level Top-1 accuracy."""
    if len(predicted_arcs) == 0 or len(reference_arcs) == 0:
        return 0.0
    y_true = np.asarray(reference_arcs)
    y_pred = np.asarray(predicted_arcs)
    return float(np.mean(y_true == y_pred))


def compute_macro_f1(predicted_arcs, reference_arcs) -> float:
    """Computes Macro F1-score across all unique arcs present."""
    if len(predicted_arcs) == 0 or len(reference_arcs) == 0:
        return 0.0
    return float(f1_score(reference_arcs, predicted_arcs, average="macro", zero_division=0))


def compute_route_continuity(predicted_arcs, arcs_array) -> tuple[float, int]:
    """
    Computes route continuity and counts disconnected breaks.
    Route continuity is the fraction of consecutive steps that are topologically connected:
    either continuing on the same arc (arc_i == arc_{i+1})
    or moving to a downstream connected arc (arcs[arc_i, 1] == arcs[arc_{i+1}, 0]).

    Returns:
        (continuity_ratio: float, n_breaks: int)
    """
    n = len(predicted_arcs)
    if n <= 1:
        return 1.0, 0

    n_transitions = n - 1
    connected_count = 0
    disconnected_breaks = 0

    for i in range(n_transitions):
        u_arc = int(predicted_arcs[i])
        v_arc = int(predicted_arcs[i + 1])

        if u_arc == v_arc:
            # Same road segment
            connected_count += 1
        else:
            # Check if arcs array contains these arc IDs
            if u_arc < len(arcs_array) and v_arc < len(arcs_array):
                to_node_u = int(arcs_array[u_arc, 1])
                from_node_v = int(arcs_array[v_arc, 0])
                if to_node_u == from_node_v:
                    connected_count += 1
                else:
                    disconnected_breaks += 1
            else:
                disconnected_breaks += 1

    continuity = float(connected_count / n_transitions)
    return continuity, int(disconnected_breaks)


def compute_point_correctness(predicted_arcs, reference_arcs) -> list[bool]:
    """Returns a list of booleans indicating if each point matched reference."""
    return [bool(p == r) for p, r in zip(predicted_arcs, reference_arcs)]

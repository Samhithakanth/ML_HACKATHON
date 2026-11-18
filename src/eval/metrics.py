"""
src/eval/metrics.py  (Person 3, shared with the API and the report)

One definition of every metric, so the API, the experiments and the report agree.

Candidate level (binary: is this candidate the reference arc?)
    candidate_metrics      macro F1, precision / recall of the positive class, log loss, ROC-AUC
Observation level (one predicted arc per GPS point)
    top1_from_scores       best-scoring candidate per point (-1 if the point has no candidates)
    compute_top1_accuracy  fraction of ALL points whose predicted arc == reference arc
                           (points without candidates or without a prediction count as wrong)
Sequence / route level
    compute_route_continuity  fraction of consecutive point pairs whose arcs are the same arc or
                              graph-connected (to_node of arc_i == from_node of arc_i+1)
    route_set_prf             precision / recall / F1 of the SET of traversed arcs vs reference
    normalized_edit_distance  Levenshtein distance between the collapsed (run-length deduplicated)
                              predicted and reference arc sequences, divided by the longer length
The reference sequence is the per-point reference arc from the DP alignment to `.route`
(src.data.labels.align_trip_to_route); it never leaves the server.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, roc_auc_score


# ----------------------------------------------------------------- candidate level
def candidate_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    pred = (prob >= threshold).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_true, prob)) if 0 < y_true.sum() < len(y_true) else float("nan"),
    }


# ----------------------------------------------------------------- observation level
def top1_from_scores(cands: pd.DataFrame, prob, n_points: int | None = None) -> np.ndarray:
    """cands: one trip's candidate rows (point_idx, arc_id). Returns predicted arc per point (-1 = none)."""
    p = cands.point_idx.values
    if n_points is None:
        n_points = int(p.max()) + 1 if len(p) else 0
    out = np.full(n_points, -1, dtype=np.int64)
    if len(p) == 0:
        return out
    order = np.lexsort((-np.asarray(prob), p))          # by point, then highest prob first
    ps = p[order]
    first = np.r_[True, ps[1:] != ps[:-1]]
    out[ps[first]] = cands.arc_id.values[order][first]
    return out


def compute_top1_accuracy(predicted_arcs, reference_arcs) -> float:
    if len(predicted_arcs) == 0 or len(reference_arcs) == 0:
        return 0.0
    return float(np.mean(np.asarray(predicted_arcs) == np.asarray(reference_arcs)))


def compute_macro_f1(predicted_arcs, reference_arcs) -> float:
    """Macro F1 over arc ids, treating every arc as a class (multi-class view of the point labels)."""
    if len(predicted_arcs) == 0 or len(reference_arcs) == 0:
        return 0.0
    return float(f1_score(np.asarray(reference_arcs), np.asarray(predicted_arcs), average="macro", zero_division=0))


def compute_point_correctness(predicted_arcs, reference_arcs) -> list[bool]:
    return [bool(p == r) for p, r in zip(predicted_arcs, reference_arcs)]


# ----------------------------------------------------------------- sequence level
def compute_route_continuity(predicted_arcs, arcs_array) -> tuple[float, int]:
    """(continuity ratio, number of disconnected breaks). arcs_array: (n_arcs, 2) from/to nodes.
    Steps involving a point with no prediction (-1) count as breaks."""
    a = np.asarray(predicted_arcs, dtype=np.int64)
    if len(a) <= 1:
        return 1.0, 0
    arcs_array = np.asarray(arcs_array)
    u, v = a[:-1], a[1:]
    valid = (u >= 0) & (v >= 0) & (u < len(arcs_array)) & (v < len(arcs_array))
    same = valid & (u == v)
    conn = np.zeros(len(u), dtype=bool)
    conn[valid] = arcs_array[u[valid], 1] == arcs_array[v[valid], 0]
    ok = same | conn
    return float(ok.mean()), int((~ok).sum())


def collapse(seq) -> np.ndarray:
    s = np.asarray(seq)
    s = s[s >= 0]
    if len(s) == 0:
        return s
    return s[np.r_[True, s[1:] != s[:-1]]]


def route_set_prf(predicted_arcs, reference_arcs) -> dict:
    P = set(collapse(predicted_arcs).tolist())
    R = set(collapse(reference_arcs).tolist())
    tp = len(P & R)
    prec = tp / len(P) if P else 0.0
    rec = tp / len(R) if R else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}


def normalized_edit_distance(predicted_arcs, reference_arcs) -> float:
    a, b = collapse(predicted_arcs), collapse(reference_arcs)
    n, m = len(a), len(b)
    if max(n, m) == 0:
        return 0.0
    prev = np.arange(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        sub = prev[:-1] + (b != a[i - 1])
        cur = np.empty(m + 1, dtype=np.int64)
        cur[0] = i
        # insertion/deletion recurrence needs a left-to-right pass: cur[j] = min(prev[j]+1, cur[j-1]+1, sub[j-1])
        best = np.minimum(prev[1:] + 1, sub)
        # cur[j] = min(best[j-1], cur[j-1] + 1)  ->  running min of (best[k] + (j-1-k))
        idx = np.arange(m)
        cur[1:] = np.minimum.accumulate(np.r_[best] - idx) + idx
        cur[1:] = np.minimum(cur[1:], i + 1 + idx)
        prev = cur
    return float(prev[m] / max(n, m))


def sequence_metrics(predicted_arcs, reference_arcs, arcs_array) -> dict:
    cont, breaks = compute_route_continuity(predicted_arcs, arcs_array)
    out = {"top1": compute_top1_accuracy(predicted_arcs, reference_arcs),
           "continuity": cont, "n_breaks": breaks,
           "edit_distance": normalized_edit_distance(predicted_arcs, reference_arcs)}
    out.update(route_set_prf(predicted_arcs, reference_arcs))
    return out

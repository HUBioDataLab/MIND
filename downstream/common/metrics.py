"""Multi-label function-prediction metrics (CAFA / DeepFRI conventions).

Protein-centric Fmax: for each threshold t in 0.00, 0.01, ..., 1.00 a term is
predicted when ``score >= t``. Precision is averaged over proteins with at
least one predicted term, recall over all evaluated proteins; Fmax is the
maximum harmonic mean over t. Every evaluated protein must carry at least one
true term (CAFA evaluates annotated benchmark proteins only).

Pair-centric AUPR: average precision over all (protein, term) pairs
(``sklearn.metrics.average_precision_score`` on the flattened arrays).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score

THRESHOLDS = np.round(np.arange(0, 101) / 100.0, 2)


def _check(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError(f"labels {labels.shape} and scores {scores.shape} must be equal 2-D arrays")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or inf")
    empty = np.where(labels.sum(axis=1) == 0)[0]
    if empty.size:
        raise ValueError(f"{empty.size} proteins have no true term (first rows {empty[:5].tolist()})")
    return labels.astype(bool), scores


def protein_fmax(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels, scores = _check(labels, scores)
    best = {"fmax": 0.0, "threshold": float("nan"), "precision": float("nan"), "recall": float("nan")}
    true_counts = labels.sum(axis=1)
    for threshold in THRESHOLDS:
        predicted = scores >= threshold
        predicted_counts = predicted.sum(axis=1)
        covered = predicted_counts > 0
        if not covered.any():
            continue
        hits = (predicted & labels).sum(axis=1)
        precision = float(np.mean(hits[covered] / predicted_counts[covered]))
        recall = float(np.mean(hits / true_counts))
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best["fmax"]:
            best = {"fmax": f1, "threshold": float(threshold), "precision": precision, "recall": recall}
    return best


def pair_aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = _check(labels, scores)
    return float(average_precision_score(labels.ravel(), scores.ravel()))


def top1_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    """Fraction of proteins whose highest-scoring term is a true term."""
    labels, scores = _check(labels, scores)
    return float(labels[np.arange(len(labels)), scores.argmax(axis=1)].mean())


def subset_accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    """Exact-match accuracy of the label set predicted at ``threshold``."""
    labels, scores = _check(labels, scores)
    return float(((scores >= threshold) == labels).all(axis=1).mean())


def evaluate(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    fmax = protein_fmax(labels, scores)
    return {
        "fmax": fmax["fmax"],
        "fmax_threshold": fmax["threshold"],
        "fmax_precision": fmax["precision"],
        "fmax_recall": fmax["recall"],
        "pair_aupr": pair_aupr(labels, scores),
        "top1_accuracy": top1_accuracy(labels, scores),
        "subset_accuracy@0.5": subset_accuracy(labels, scores),
        "num_proteins": int(len(labels)),
    }

"""Multi-label function-prediction metrics (CAFA / DeepFRI conventions).

Protein-centric Fmax: for each threshold t in 0.00, 0.01, ..., 1.00 a term is
predicted when ``score >= t``. Precision is averaged over proteins with at
least one predicted term, recall over all evaluated proteins; Fmax is the
maximum harmonic mean over t. Every evaluated protein must carry at least one
true term (CAFA evaluates annotated benchmark proteins only).

Pair-centric AUPR: average precision over all (protein, term) pairs
(``sklearn.metrics.average_precision_score`` on the flattened arrays); this is the
GearNet definition and keeps the name ``pair_aupr``.

Label-centric (macro) AUPR: one average precision per term over the proteins, then the
mean over terms (the DeepFRI definition), reported as ``macro_aupr``. A term with no true
protein in the evaluated set has no precision-recall curve; it is left out of the mean and
counted in ``macro_aupr_num_terms_skipped``, never counted as zero.

Coverage (DeepFRI): the fraction of test proteins for which the method makes a prediction. Our probe scores
every protein, so coverage is read at the Fmax threshold: the fraction of proteins with at least one term scored
``>= threshold`` (the proteins that enter the precision average).

IC-binned macro AUPR (DeepFRI Fig. 3f): the label-centric AUPR over the terms with information content
``IC(term) > t``, ``IC = -log2(n_term / n_proteins)`` (DeepFRI eq. 1). DeepFRI counts proteins in UniProt-GOA;
that table is not available here, so the probe takes the frequencies from the train split (``train_term_ic``).
Any other IC vector can be passed to ``ic_binned_macro_aupr``.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

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
    best = {"fmax": 0.0, "threshold": float("nan"), "precision": float("nan"), "recall": float("nan"),
            "coverage": float("nan")}
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
            best = {"fmax": f1, "threshold": float(threshold), "precision": precision, "recall": recall,
                    "coverage": float(covered.mean())}
    return best


def pair_aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = _check(labels, scores)
    return float(average_precision_score(labels.ravel(), scores.ravel()))


def label_aupr(labels: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    """Label-centric AUPR: mean over terms of the per-term average precision (DeepFRI)."""
    labels, scores = _check(labels, scores)
    values, skipped = [], []
    for term in range(labels.shape[1]):
        if not labels[:, term].any():
            skipped.append(term)
            continue
        values.append(float(average_precision_score(labels[:, term], scores[:, term])))
    if not values:
        raise ValueError("no term has a true protein; macro AUPR is undefined")
    return {
        "macro_aupr": float(np.mean(values)),
        "num_terms_evaluated": len(values),
        "num_terms_skipped_no_positive": len(skipped),
        "skipped_term_indices": skipped,
    }


def train_term_ic(train_labels: np.ndarray) -> np.ndarray:
    """Information content per term, ``-log2(n_term / n_proteins)`` over the train proteins; ``inf`` for an unseen term."""
    train_labels = np.asarray(train_labels)
    if train_labels.ndim != 2 or train_labels.shape[0] < 1:
        raise ValueError(f"train labels must be a 2-D array with at least one protein, got {train_labels.shape}")
    if not np.isin(train_labels, (0, 1)).all():
        raise ValueError("train labels must be 0/1")
    counts = train_labels.sum(axis=0).astype(np.float64)
    with np.errstate(divide="ignore"):
        return -np.log2(counts / train_labels.shape[0])


def ic_binned_macro_aupr(labels: np.ndarray, scores: np.ndarray, term_ic: np.ndarray,
                         thresholds=(5.0, 10.0)) -> Dict[str, Dict[str, object]]:
    """Label-centric AUPR over the terms with ``IC > t`` for each threshold ``t`` (key ``ic_gt<t>``).

    A term without a true protein is skipped and counted; a bin with no evaluable term has ``macro_aupr`` None.
    """
    labels, scores = _check(labels, scores)
    term_ic = np.asarray(term_ic, dtype=np.float64)
    if term_ic.shape != (labels.shape[1],):
        raise ValueError(f"term_ic needs one value per term ({labels.shape[1]}), got shape {term_ic.shape}")
    if np.isnan(term_ic).any():
        raise ValueError("term_ic contains NaN")
    has_positive = labels.any(axis=0)
    per_term = {term: float(average_precision_score(labels[:, term], scores[:, term]))
                for term in np.where(has_positive)[0]}
    result = {}
    for threshold in thresholds:
        in_bin = term_ic > threshold
        values = [per_term[term] for term in np.where(in_bin & has_positive)[0]]
        result[f"ic_gt{threshold:g}"] = {
            "macro_aupr": float(np.mean(values)) if values else None,
            "num_terms_evaluated": len(values),
            "num_terms_skipped_no_positive": int((in_bin & ~has_positive).sum()),
        }
    return result


def top1_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    """Fraction of proteins whose highest-scoring term is a true term."""
    labels, scores = _check(labels, scores)
    return float(labels[np.arange(len(labels)), scores.argmax(axis=1)].mean())


def subset_accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    """Exact-match accuracy of the label set predicted at ``threshold``."""
    labels, scores = _check(labels, scores)
    return float(((scores >= threshold) == labels).all(axis=1).mean())


def evaluate(labels: np.ndarray, scores: np.ndarray, term_ic: np.ndarray = None,
             ic_thresholds=(5.0, 10.0)) -> Dict[str, float]:
    """All multi-label metrics; ``term_ic`` (one IC per term) adds the IC-binned macro AUPR.

    The first IC threshold is the headline ``ic_binned_macro_aupr``; further ones are
    ``ic_binned_macro_aupr_ic_gt<t>``. Each bin also records its term count (``<name>_num_terms``); a bin
    without an evaluable term has the count 0 and no value.
    """
    fmax = protein_fmax(labels, scores)
    macro = label_aupr(labels, scores)
    result = {
        "fmax": fmax["fmax"],
        "fmax_threshold": fmax["threshold"],
        "fmax_precision": fmax["precision"],
        "fmax_recall": fmax["recall"],
        "pair_aupr": pair_aupr(labels, scores),
        "top1_accuracy": top1_accuracy(labels, scores),
        "subset_accuracy@0.5": subset_accuracy(labels, scores),
        "num_proteins": int(len(labels)),
        "macro_aupr": macro["macro_aupr"],
        "macro_aupr_num_terms": macro["num_terms_evaluated"],
        "macro_aupr_num_terms_skipped": macro["num_terms_skipped_no_positive"],
        "coverage": fmax["coverage"],
    }
    if term_ic is not None:
        bins = ic_binned_macro_aupr(labels, scores, term_ic, ic_thresholds)
        for position, (key, entry) in enumerate(bins.items()):
            name = "ic_binned_macro_aupr" if position == 0 else f"ic_binned_macro_aupr_{key}"
            result[f"{name}_num_terms"] = entry["num_terms_evaluated"]
            if entry["macro_aupr"] is not None:
                result[name] = entry["macro_aupr"]
    return result


# ---------------------------------------------------------------------------
# Regression metrics (ATOM3D LBA: pK). RMSE is the primary number.
# ---------------------------------------------------------------------------

def _check_regression(targets: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if targets.shape != predictions.shape:
        raise ValueError(f"targets {targets.shape} and predictions {predictions.shape} differ")
    if targets.size == 0:
        raise ValueError("regression metrics need at least one sample")
    if not np.isfinite(targets).all():
        raise ValueError("targets contain NaN or inf")
    if not np.isfinite(predictions).all():
        raise ValueError("predictions contain NaN or inf")
    return targets, predictions


def rmse(targets: np.ndarray, predictions: np.ndarray) -> float:
    targets, predictions = _check_regression(targets, predictions)
    return float(np.sqrt(np.mean((targets - predictions) ** 2)))


def mae(targets: np.ndarray, predictions: np.ndarray) -> float:
    targets, predictions = _check_regression(targets, predictions)
    return float(np.mean(np.abs(targets - predictions)))


def _correlation_undefined_reason(targets: np.ndarray, predictions: np.ndarray) -> str | None:
    """Why Pearson / Spearman have no value for these samples (None when they are defined)."""
    if targets.size < 2:
        return "fewer_than_two_samples"
    constant_target = bool(np.ptp(targets) == 0)
    constant_prediction = bool(np.ptp(predictions) == 0)
    if constant_target and constant_prediction:
        return "constant_target_and_prediction"
    if constant_prediction:
        return "constant_prediction"
    if constant_target:
        return "constant_target"
    return None


def pearson_r(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Pearson correlation; a constant prediction or target has no correlation."""
    targets, predictions = _check_regression(targets, predictions)
    if targets.size < 2:
        raise ValueError("Pearson r needs at least two samples")
    if np.ptp(targets) == 0 or np.ptp(predictions) == 0:
        raise ValueError("Pearson r is undefined for a constant target or prediction")
    from scipy.stats import pearsonr

    return float(pearsonr(targets, predictions)[0])


def spearman_rho(targets: np.ndarray, predictions: np.ndarray) -> float:
    targets, predictions = _check_regression(targets, predictions)
    if targets.size < 2:
        raise ValueError("Spearman rho needs at least two samples")
    from scipy.stats import spearmanr

    value = float(spearmanr(targets, predictions).statistic)
    if not np.isfinite(value):
        raise ValueError("Spearman rho is undefined (constant target or prediction)")
    return value


def evaluate_regression(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, object]:
    """RMSE / MAE always; Pearson and Spearman are ``None`` with a named reason when undefined.

    ``correlation_undefined_reason`` is ``None`` when both correlations were computed, otherwise one of
    ``constant_prediction``, ``constant_target``, ``constant_target_and_prediction`` or
    ``fewer_than_two_samples``. NaN/inf inputs and shape mismatches still raise.
    """
    checked_targets, checked_predictions = _check_regression(targets, predictions)
    reason = _correlation_undefined_reason(checked_targets, checked_predictions)
    return {
        "rmse": rmse(checked_targets, checked_predictions),
        "mae": mae(checked_targets, checked_predictions),
        "pearson_r": None if reason else pearson_r(checked_targets, checked_predictions),
        "spearman_rho": None if reason else spearman_rho(checked_targets, checked_predictions),
        "num_samples": int(checked_targets.size),
        "target_mean": float(checked_targets.mean()),
        "target_std": float(checked_targets.std()),
        "correlation_undefined_reason": reason,
    }


# ---------------------------------------------------------------------------
# Single-label multi-class metrics (Fold, Reaction: accuracy is the primary number).
# ---------------------------------------------------------------------------

def _check_multiclass(targets: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets)
    scores = np.asarray(scores, dtype=np.float64)
    if targets.ndim != 1 or scores.ndim != 2 or scores.shape[0] != targets.shape[0]:
        raise ValueError(f"targets {targets.shape} must be 1-D class ids matching scores {scores.shape} (N x classes)")
    if targets.size == 0:
        raise ValueError("multi-class metrics need at least one sample")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError(f"class targets must be integers, got {targets.dtype}")
    if targets.min() < 0 or targets.max() >= scores.shape[1]:
        raise ValueError(f"class ids must lie in [0, {scores.shape[1]}), got [{targets.min()}, {targets.max()}]")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or inf")
    return targets.astype(np.int64), scores


def accuracy(targets: np.ndarray, scores: np.ndarray) -> float:
    """Fraction of samples whose arg-max class equals the target (GearNet / IEConv headline metric)."""
    targets, scores = _check_multiclass(targets, scores)
    return float((scores.argmax(axis=1) == targets).mean())


def balanced_accuracy(targets: np.ndarray, scores: np.ndarray) -> float:
    """Mean per-class recall over the classes that occur in ``targets`` (classes absent from the split are not averaged)."""
    targets, scores = _check_multiclass(targets, scores)
    predicted = scores.argmax(axis=1)
    recalls = [float((predicted[targets == c] == c).mean()) for c in np.unique(targets)]
    return float(np.mean(recalls))


def weighted_f1(targets: np.ndarray, scores: np.ndarray) -> float:
    """Per-class F1 averaged with the class support as weight (PEER's balanced metric, sklearn ``average="weighted"``).

    A class that is only predicted (no true sample) has weight 0; a class with no true positive has F1 0.
    """
    targets, scores = _check_multiclass(targets, scores)
    predicted = scores.argmax(axis=1)
    num_classes = scores.shape[1]
    support = np.bincount(targets, minlength=num_classes).astype(np.float64)
    true_positive = np.bincount(targets[predicted == targets], minlength=num_classes).astype(np.float64)
    predicted_count = np.bincount(predicted, minlength=num_classes).astype(np.float64)
    denominator = support + predicted_count
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros(num_classes), where=denominator > 0)
    return float((support * f1).sum() / support.sum())


def evaluate_multiclass(targets: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    targets, scores = _check_multiclass(targets, scores)
    top5 = np.argsort(-scores, axis=1)[:, :min(5, scores.shape[1])]
    return {
        "accuracy": accuracy(targets, scores),
        "top5_accuracy": float((top5 == targets[:, None]).any(axis=1).mean()),
        "num_proteins": int(len(targets)),
        "num_classes_present": int(len(np.unique(targets))),
        "balanced_accuracy": balanced_accuracy(targets, scores),
        "weighted_f1": weighted_f1(targets, scores),
    }


# ---------------------------------------------------------------------------
# MoleculeNet metrics (pure functions; no task code yet).
#
# Labels of a multi-task data set carry NaN where a molecule was not measured for a task. Every function
# masks those entries per task. A task that cannot be scored (single-class labels) is skipped and counted in
# the result; if no task can be scored the call raises. Scores must be finite everywhere.
# ---------------------------------------------------------------------------

def _as_columns(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D (samples x tasks), got shape {array.shape}")
    return array


def _check_masked(labels, scores, *, label_name: str, score_name: str) -> tuple[np.ndarray, np.ndarray]:
    labels, scores = _as_columns(labels, label_name), _as_columns(scores, score_name)
    if labels.shape != scores.shape:
        raise ValueError(f"{label_name} {labels.shape} and {score_name} {scores.shape} must have equal shapes")
    if labels.shape[0] == 0:
        raise ValueError("metrics need at least one sample")
    if not np.isfinite(scores).all():
        raise ValueError(f"{score_name} contain NaN or inf")
    return labels, scores


def _multitask_binary(labels, scores, metric, key: str) -> Dict[str, object]:
    labels, scores = _check_masked(labels, scores, label_name="labels", score_name="scores")
    observed = ~np.isnan(labels)
    if not np.isin(labels[observed], (0, 1)).all():
        raise ValueError("labels must be 0, 1 or NaN (missing)")
    per_task, skipped = [], []
    for task in range(labels.shape[1]):
        valid = observed[:, task]
        truth = labels[valid, task].astype(int)
        if truth.sum() == 0 or truth.sum() == truth.size:
            per_task.append(None)
            skipped.append(task)
            continue
        per_task.append(float(metric(truth, scores[valid, task])))
    evaluated = [value for value in per_task if value is not None]
    if not evaluated:
        raise ValueError("every task is single-class (needs labelled positives and negatives); nothing to evaluate")
    return {
        key: float(np.mean(evaluated)),
        "per_task": per_task,
        "num_tasks": int(labels.shape[1]),
        "num_tasks_evaluated": len(evaluated),
        "num_tasks_skipped": len(skipped),
        "skipped_task_indices": skipped,
    }


def multitask_roc_auc(labels: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    """Mean over tasks of the ROC-AUC computed on each task's labelled molecules (MoleculeNet / Uni-Mol)."""
    return _multitask_binary(labels, scores, roc_auc_score, "roc_auc")


def multitask_prc_auc(labels: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    """Mean over tasks of the average precision (area under the precision-recall curve), masked per task.

    Average precision is the step-wise sum used by scikit-learn; MoleculeNet's own PRC-AUC may integrate the
    curve with the trapezoid rule, which gives slightly different numbers.
    """
    return _multitask_binary(labels, scores, average_precision_score, "prc_auc")


def multitarget_regression_errors(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, object]:
    """Per-target and target-averaged RMSE and MAE, masking NaN (missing) targets. Units are the caller's."""
    targets, predictions = _check_masked(targets, predictions, label_name="targets", score_name="predictions")
    observed = ~np.isnan(targets)
    if not np.isfinite(targets[observed]).all():
        raise ValueError("targets contain inf")
    rmse_values, mae_values, counts = [], [], []
    for target in range(targets.shape[1]):
        valid = observed[:, target]
        if not valid.any():
            raise ValueError(f"target {target} has no labelled sample; its error is undefined")
        error = predictions[valid, target] - targets[valid, target]
        rmse_values.append(float(np.sqrt(np.mean(error ** 2))))
        mae_values.append(float(np.mean(np.abs(error))))
        counts.append(int(valid.sum()))
    return {
        "rmse": float(np.mean(rmse_values)),
        "mae": float(np.mean(mae_values)),
        "rmse_per_target": rmse_values,
        "mae_per_target": mae_values,
        "num_targets": int(targets.shape[1]),
        "num_labeled_per_target": counts,
    }


def evaluate_multitask_classification(labels: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    """ROC-AUC (Uni-Mol's headline) and PRC-AUC of a multi-task data set in one dict; NaN labels are missing.

    Both metrics skip the same single-class tasks, so ``num_tasks_skipped`` and ``skipped_task_indices`` describe both.
    """
    roc = multitask_roc_auc(labels, scores)
    prc = multitask_prc_auc(labels, scores)
    if roc["skipped_task_indices"] != prc["skipped_task_indices"]:
        raise RuntimeError("ROC-AUC and PRC-AUC skipped different tasks")
    return {
        "roc_auc": roc["roc_auc"], "prc_auc": prc["prc_auc"], "per_task": roc["per_task"],
        "per_task_prc_auc": prc["per_task"], "num_tasks": roc["num_tasks"],
        "num_tasks_evaluated": roc["num_tasks_evaluated"], "num_tasks_skipped": roc["num_tasks_skipped"],
        "skipped_task_indices": roc["skipped_task_indices"],
    }


# ---------------------------------------------------------------------------
# RNA metrics (rnaglib core tasks; pure functions, no task code yet).
#
# Binary and multi-label scores are probabilities; a prediction is ``score >= threshold``. A value that
# cannot be computed is ``None`` and named in ``undefined`` (never a silent 0). MCC with a zero marginal
# (for example an always-negative model) is therefore ``None``, whereas scikit-learn returns 0.
# ---------------------------------------------------------------------------

def _binary_counts(truth: np.ndarray, predicted: np.ndarray) -> tuple[int, int, int, int]:
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    tn = int((~truth & ~predicted).sum())
    return tp, fp, fn, tn


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _balanced_accuracy_from_counts(tp, fp, fn, tn) -> float | None:
    if tp + fn == 0 or tn + fp == 0:
        return None
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def _mcc_from_counts(tp, fp, fn, tn) -> float | None:
    denominator = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return _ratio(tp * tn - fp * fn, denominator)


def _check_binary_labels(labels: np.ndarray) -> np.ndarray:
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return labels.astype(bool)


def binary_classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, object]:
    """Balanced accuracy, F1, MCC, AUC and Jaccard of the positive class for one binary task."""
    labels, scores = np.asarray(labels, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != scores.shape:
        raise ValueError(f"labels {labels.shape} and scores {scores.shape} must have equal shapes (1-D)")
    if labels.size == 0:
        raise ValueError("metrics need at least one sample")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or inf")
    truth = _check_binary_labels(labels)
    tp, fp, fn, tn = _binary_counts(truth, scores >= threshold)
    values = {
        "balanced_accuracy": _balanced_accuracy_from_counts(tp, fp, fn, tn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "mcc": _mcc_from_counts(tp, fp, fn, tn),
        "auc": None if truth.all() or not truth.any() else float(roc_auc_score(truth, scores)),
        "jaccard": _ratio(tp, tp + fp + fn),
    }
    reasons = {"balanced_accuracy": "single_class_labels", "f1": "no_positive_labels_or_predictions",
               "mcc": "zero_marginal", "auc": "single_class_labels", "jaccard": "no_positive_labels_or_predictions"}
    return {
        **values,
        "threshold": float(threshold),
        "num_samples": int(labels.size),
        "num_positive": int(truth.sum()),
        "undefined": {name: reasons[name] for name, value in values.items() if value is None},
    }


def _macro(values: list, name: str, reason: str, undefined: dict) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        undefined[name] = reason
        return None
    return float(np.mean(defined))


def multilabel_classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, object]:
    """F1 and Jaccard (macro over labels, micro over all entries, mean over samples), plus macro balanced accuracy and AUC.

    rnaglib does not state which averaging its RNA-GO table uses, so every variant is returned under its own name.
    A label (or sample) whose metric is 0/0 is left out of the macro (or sample) mean and counted.
    """
    labels, scores = np.asarray(labels, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 2 or labels.shape != scores.shape:
        raise ValueError(f"labels {labels.shape} and scores {scores.shape} must be equal 2-D arrays (samples x labels)")
    if labels.shape[0] == 0:
        raise ValueError("metrics need at least one sample")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or inf")
    truth = _check_binary_labels(labels)
    predicted = scores >= threshold
    per_label = [_binary_counts(truth[:, j], predicted[:, j]) for j in range(truth.shape[1])]
    per_sample = [_binary_counts(truth[i], predicted[i]) for i in range(truth.shape[0])]
    tp, fp, fn = (int((truth & predicted).sum()), int((~truth & predicted).sum()),
                  int((truth & ~predicted).sum()))
    label_auc = [None if truth[:, j].all() or not truth[:, j].any() else float(roc_auc_score(truth[:, j], scores[:, j]))
                 for j in range(truth.shape[1])]
    per_label_values = {
        "f1": [_ratio(2 * c[0], 2 * c[0] + c[1] + c[2]) for c in per_label],
        "jaccard": [_ratio(c[0], c[0] + c[1] + c[2]) for c in per_label],
        "balanced_accuracy": [_balanced_accuracy_from_counts(*c) for c in per_label],
        "auc": label_auc,
    }
    per_sample_values = {
        "f1": [_ratio(2 * c[0], 2 * c[0] + c[1] + c[2]) for c in per_sample],
        "jaccard": [_ratio(c[0], c[0] + c[1] + c[2]) for c in per_sample],
    }
    undefined: Dict[str, str] = {}
    result: Dict[str, object] = {}
    for metric in ("f1", "jaccard"):
        result[f"{metric}_macro"] = _macro(per_label_values[metric], f"{metric}_macro",
                                           "no_positive_labels_or_predictions", undefined)
        result[f"{metric}_samples"] = _macro(per_sample_values[metric], f"{metric}_samples",
                                             "no_positive_labels_or_predictions", undefined)
    result["f1_micro"] = _ratio(2 * tp, 2 * tp + fp + fn)
    result["jaccard_micro"] = _ratio(tp, tp + fp + fn)
    for name in ("f1_micro", "jaccard_micro"):
        if result[name] is None:
            undefined[name] = "no_positive_labels_or_predictions"
    result["balanced_accuracy_macro"] = _macro(per_label_values["balanced_accuracy"], "balanced_accuracy_macro",
                                               "no_label_has_both_classes", undefined)
    result["auc_macro"] = _macro(per_label_values["auc"], "auc_macro", "no_label_has_both_classes", undefined)
    result.update({
        "num_labels": int(truth.shape[1]),
        "num_samples": int(truth.shape[0]),
        "threshold": float(threshold),
        "num_labels_skipped": {name: sum(value is None for value in values) for name, values in per_label_values.items()},
        "num_samples_skipped": {name: sum(value is None for value in values) for name, values in per_sample_values.items()},
        "undefined": undefined,
    })
    return result


def multiclass_classification_metrics(targets: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    """Accuracy (= sequence recovery for nucleotide identity), balanced accuracy, macro F1 / Jaccard, MCC, one-vs-rest AUC.

    Balanced accuracy averages the recall of classes present in ``targets``; macro F1 / Jaccard average over classes
    present in the targets or the predictions; one-vs-rest AUC averages over classes that occur but not in every sample.
    """
    targets, scores = _check_multiclass(targets, scores)
    num_classes = scores.shape[1]
    predicted = scores.argmax(axis=1)
    confusion = np.bincount(targets * num_classes + predicted, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    total = int(confusion.sum())
    tp = np.diag(confusion).astype(np.float64)
    true_counts, predicted_counts = confusion.sum(axis=1), confusion.sum(axis=0)
    present = true_counts > 0
    seen = present | (predicted_counts > 0)
    undefined: Dict[str, str] = {}
    f1 = float(np.mean(2 * tp[seen] / (true_counts[seen] + predicted_counts[seen])))
    jaccard = float(np.mean(tp[seen] / (true_counts[seen] + predicted_counts[seen] - tp[seen])))
    mcc_denominator = math.sqrt(float(total ** 2 - (predicted_counts.astype(float) ** 2).sum())
                                * float(total ** 2 - (true_counts.astype(float) ** 2).sum()))
    mcc = _ratio(float(tp.sum() * total - (predicted_counts.astype(float) * true_counts).sum()), mcc_denominator)
    if mcc is None:
        undefined["mcc"] = "zero_marginal"
    aucs = [float(roc_auc_score(targets == k, scores[:, k])) for k in range(num_classes)
            if 0 < (targets == k).sum() < targets.size]
    auc = None if not aucs else float(np.mean(aucs))
    if auc is None:
        undefined["auc_ovr_macro"] = "single_class_labels"
    accuracy_value = float(tp.sum() / total)
    return {
        "accuracy": accuracy_value,
        "sequence_recovery": accuracy_value,
        "balanced_accuracy": float(np.mean(tp[present] / true_counts[present])),
        "f1_macro": f1,
        "jaccard_macro": jaccard,
        "mcc": mcc,
        "auc_ovr_macro": auc,
        "num_classes": int(num_classes),
        "num_classes_present": int(present.sum()),
        "num_classes_skipped_auc": int(num_classes - len(aucs)),
        "num_samples": total,
        "undefined": undefined,
    }

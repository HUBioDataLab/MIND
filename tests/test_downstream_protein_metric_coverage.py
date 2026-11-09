"""Extra protein-task metrics that the papers report (label-centric AUPR, Fold Avg., balanced accuracy,
LBA with a constant prediction) and their appearance in the report tables.

Every old metric must keep its value: the golden numbers below were recorded from the code before the
extra metrics were added.
"""

import json
import math

import numpy as np
import pytest

from downstream import report
from downstream.common.metrics import (
    balanced_accuracy,
    evaluate,
    evaluate_multiclass,
    evaluate_regression,
    label_aupr,
    pair_aupr,
    pearson_r,
    rmse,
    spearman_rho,
)

LABELS = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
SCORES = np.array([[0.9, 0.2, 0.1], [0.3, 0.8, 0.4], [0.6, 0.1, 0.05]])


def _golden_inputs():
    rng = np.random.default_rng(7)
    labels = (rng.random((30, 6)) < 0.3).astype(int)
    labels[np.arange(30), rng.integers(0, 6, 30)] = 1
    scores = rng.random((30, 6))
    targets = rng.normal(6, 1.5, 25)
    predictions = targets + rng.normal(0, 1, 25)
    classes = rng.integers(0, 5, 40)
    class_scores = rng.random((40, 5))
    return labels, scores, targets, predictions, classes, class_scores


# ---------------------------------------------------------------------------
# Old values are frozen
# ---------------------------------------------------------------------------

def test_old_metric_values_are_unchanged():
    labels, scores, targets, predictions, classes, class_scores = _golden_inputs()
    multilabel = evaluate(labels, scores)
    assert multilabel["fmax"] == pytest.approx(0.575068870523416)
    assert multilabel["fmax_threshold"] == pytest.approx(0.13)
    assert multilabel["fmax_precision"] == pytest.approx(0.4166666666666667)
    assert multilabel["fmax_recall"] == pytest.approx(0.9277777777777777)
    assert multilabel["pair_aupr"] == pytest.approx(0.4164911666108017)
    assert multilabel["top1_accuracy"] == pytest.approx(0.36666666666666664)
    assert multilabel["subset_accuracy@0.5"] == 0.0 and multilabel["num_proteins"] == 30
    regression = evaluate_regression(targets, predictions)
    assert regression["rmse"] == pytest.approx(1.0508880439724344)
    assert regression["mae"] == pytest.approx(0.8587491970915988)
    assert regression["pearson_r"] == pytest.approx(0.7816901613475498)
    assert regression["spearman_rho"] == pytest.approx(0.7392307692307692)
    assert regression["num_samples"] == 25
    assert regression["target_mean"] == pytest.approx(5.912024898028314)
    assert regression["target_std"] == pytest.approx(1.372133881104896)
    multiclass = evaluate_multiclass(classes, class_scores)
    assert multiclass["accuracy"] == 0.2 and multiclass["top5_accuracy"] == 1.0
    assert multiclass["num_proteins"] == 40 and multiclass["num_classes_present"] == 5


# ---------------------------------------------------------------------------
# 1. Label-centric (macro) AUPR, the DeepFRI definition
# ---------------------------------------------------------------------------

def test_label_aupr_hand_example_differs_from_pair_aupr():
    result = label_aupr(LABELS, SCORES)
    # term 0 ranks perfectly (AP 1), term 1 ranks +,-,+ (AP (1 + 2/3) / 2 = 5/6), term 2 ranks perfectly (AP 1)
    assert result["macro_aupr"] == pytest.approx((1 + 5 / 6 + 1) / 3)
    assert result["num_terms_evaluated"] == 3 and result["num_terms_skipped_no_positive"] == 0
    assert result["macro_aupr"] != pytest.approx(pair_aupr(LABELS, SCORES))


def test_label_aupr_skips_terms_without_positives_and_says_so():
    labels = np.hstack([LABELS, np.zeros((3, 2), dtype=int)])
    scores = np.hstack([SCORES, np.full((3, 2), 0.5)])
    result = label_aupr(labels, scores)
    assert result["macro_aupr"] == pytest.approx(17 / 18)  # the two empty terms do not enter the mean
    assert result["num_terms_evaluated"] == 3 and result["num_terms_skipped_no_positive"] == 2
    assert result["skipped_term_indices"] == [3, 4]


def test_label_aupr_fails_loudly_on_bad_input():
    with pytest.raises(ValueError, match="NaN"):
        label_aupr(LABELS, np.where(SCORES > 0.5, np.nan, SCORES))
    with pytest.raises(ValueError, match="must be equal"):
        label_aupr(LABELS, SCORES[:, :2])
    with pytest.raises(ValueError, match="no true term"):
        label_aupr(np.array([[1, 0], [0, 0]]), np.zeros((2, 2)))


def test_evaluate_adds_macro_aupr_next_to_pair_aupr():
    result = evaluate(LABELS, SCORES)
    assert result["macro_aupr"] == pytest.approx(17 / 18)
    assert result["macro_aupr_num_terms"] == 3 and result["macro_aupr_num_terms_skipped"] == 0
    assert result["pair_aupr"] == pytest.approx(pair_aupr(LABELS, SCORES))


# ---------------------------------------------------------------------------
# 2. Fold: balanced accuracy and the Avg. of the three test sets
# ---------------------------------------------------------------------------

def test_balanced_accuracy_hand_example():
    scores = np.eye(4)[[0, 1, 1, 1]]  # predictions 0, 1, 1, 1
    targets = np.array([0, 0, 1, 2])
    # recall: class 0 = 1/2, class 1 = 1, class 2 = 0; class 3 never occurs and is not averaged
    assert balanced_accuracy(targets, scores) == pytest.approx(0.5)
    result = evaluate_multiclass(targets, scores)
    assert result["accuracy"] == pytest.approx(0.5)  # unchanged: sample-weighted top-1
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["num_classes_present"] == 3
    skewed = evaluate_multiclass(np.array([0, 0, 0, 1]), np.eye(2)[[0, 0, 0, 0]])
    assert skewed["accuracy"] == pytest.approx(0.75) and skewed["balanced_accuracy"] == pytest.approx(0.5)


def _write_fold_run(base, seed, accuracies, balanced=None):
    directory = base / f"seed{seed}"
    directory.mkdir(parents=True)
    test = {}
    for name, value in accuracies.items():
        entry = {"accuracy": value, "top5_accuracy": 1.0, "num_proteins": 10, "num_classes_present": 3}
        if balanced is not None:
            entry["balanced_accuracy"] = balanced[name]
        test[name] = entry
    (directory / "metrics.json").write_text(json.dumps({
        "task": "fold", "kind": "multiclass", "seed": seed, "val": {"accuracy": 0.5}, "test": test}))


def test_fold_report_adds_avg_and_balanced_columns(tmp_path):
    base = tmp_path / "downstream" / "fold" / "snap" / "linear_probe.mean_atoms"
    for seed, shift in ((0, 0.0), (1, 0.1)):
        _write_fold_run(base, seed,
                        {"test_fold": 0.2 + shift, "test_superfamily": 0.4 + shift, "test_family": 0.9},
                        balanced={"test_fold": 0.1, "test_superfamily": 0.3, "test_family": 0.8})
    rows = report.collect("fold", tmp_path)
    row = rows[0]
    assert row["test_accuracy_avg"]["mean"] == pytest.approx((0.2 + 0.4 + 0.9) / 3 + 0.2 / 3 / 2)
    assert row["test_balanced_accuracy_avg"]["mean"] == pytest.approx(0.4)
    assert row["test_balanced_accuracy_test_fold"]["mean"] == pytest.approx(0.1)
    assert row["test_accuracy_test_fold"]["mean"] == pytest.approx(0.25)  # old keys stay
    markdown = report.to_markdown("fold", rows)
    header = markdown.splitlines()[0]
    assert "test accuracy avg" in header and "test balanced accuracy avg" in header
    assert "test accuracy test_fold" in header
    assert header.index("val accuracy") < header.index("test accuracy test_fold") < header.index("test accuracy avg")
    assert "72.7" in markdown  # literature Avg. derived from the three accuracies


def test_fold_report_reads_old_results_without_the_new_fields(tmp_path):
    base = tmp_path / "downstream" / "fold" / "snap" / "linear_probe.mean_atoms"
    _write_fold_run(base, 0, {"test_fold": 0.2, "test_superfamily": 0.4, "test_family": 0.9})
    row = report.collect("fold", tmp_path)[0]
    assert row["test_accuracy_avg"]["mean"] == pytest.approx(0.5)
    assert row["test_balanced_accuracy_avg"] is None and row["test_balanced_accuracy_test_fold"] is None
    markdown = report.to_markdown("fold", [row])
    assert "0.500" in markdown and "-" in markdown.splitlines()[2]


def test_reaction_report_has_no_avg_column(tmp_path):
    directory = tmp_path / "downstream" / "reaction" / "snap" / "linear_probe.mean_atoms" / "seed0"
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text(json.dumps({
        "task": "reaction", "kind": "multiclass", "seed": 0, "val": {"accuracy": 0.5},
        "test": {"test": {"accuracy": 0.6, "top5_accuracy": 0.9, "num_proteins": 5, "num_classes_present": 3}}}))
    row = report.collect("reaction", tmp_path)[0]
    assert "test_accuracy_avg" not in row and row["test_accuracy_test"]["mean"] == pytest.approx(0.6)
    assert "avg" not in report.to_markdown("reaction", [row]).splitlines()[0]


def test_fold_run_with_a_missing_test_split_fails_loudly(tmp_path):
    base = tmp_path / "downstream" / "fold" / "snap" / "linear_probe.mean_atoms"
    _write_fold_run(base, 0, {"test_fold": 0.2, "test_family": 0.9})
    with pytest.raises(ValueError, match="test_superfamily"):
        report.collect("fold", tmp_path)


# ---------------------------------------------------------------------------
# 3. LBA: a constant prediction must not abort the evaluation
# ---------------------------------------------------------------------------

def test_constant_prediction_keeps_rmse_and_labels_the_correlations():
    targets = np.array([1.0, 2.0, 3.0, 4.0])
    result = evaluate_regression(targets, np.full(4, 2.5))
    assert result["rmse"] == pytest.approx(math.sqrt((2.25 + 0.25 + 0.25 + 2.25) / 4))
    assert result["mae"] == pytest.approx(1.0)
    assert result["pearson_r"] is None and result["spearman_rho"] is None
    assert result["correlation_undefined_reason"] == "constant_prediction"
    json.dumps(result, allow_nan=False)  # strict JSON: no NaN slipped in


def test_constant_float_prediction_is_detected_exactly():
    # 0.1 * 3 repeated has no rounding noise problem when compared by min == max
    result = evaluate_regression(np.array([1.0, 2.0, 3.0]), np.full(3, 3.3))
    assert result["correlation_undefined_reason"] == "constant_prediction"


def test_constant_target_and_both_constant_are_labelled():
    assert evaluate_regression(np.full(3, 5.0), np.array([1.0, 2.0, 3.0]))[
        "correlation_undefined_reason"] == "constant_target"
    both = evaluate_regression(np.full(3, 5.0), np.full(3, 4.0))
    assert both["correlation_undefined_reason"] == "constant_target_and_prediction" and both["rmse"] == 1.0


def test_single_sample_regression_labels_the_correlations():
    result = evaluate_regression(np.array([2.0]), np.array([3.0]))
    assert result["rmse"] == 1.0 and result["pearson_r"] is None
    assert result["correlation_undefined_reason"] == "fewer_than_two_samples"


def test_normal_regression_has_no_reason_and_old_functions_still_raise():
    targets, predictions = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.5, 1.5, 3.5, 5.0])
    result = evaluate_regression(targets, predictions)
    assert result["correlation_undefined_reason"] is None
    assert result["pearson_r"] == pytest.approx(pearson_r(targets, predictions))
    with pytest.raises(ValueError, match="constant"):
        pearson_r(targets, np.full(4, 3.0))
    with pytest.raises(ValueError, match="undefined"):
        spearman_rho(targets, np.full(4, 3.0))
    assert rmse(targets, predictions) == pytest.approx(result["rmse"])


def test_nan_predictions_still_abort_the_evaluation():
    with pytest.raises(ValueError, match="NaN"):
        evaluate_regression(np.array([1.0, 2.0]), np.array([1.0, float("nan")]))


def _write_lba_run(base, seed, test, val=None):
    directory = base / f"seed{seed}"
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text(json.dumps({
        "task": "lba", "kind": "regression", "seed": seed, "val": val or {"rmse": 1.5}, "test": {"all": test}}))


def test_lba_report_survives_null_correlations_and_names_the_reason(tmp_path):
    base = tmp_path / "downstream" / "lba" / "snap" / "linear_probe.identity30.joint.concat_pocket_ligand"
    _write_lba_run(base, 0, {"rmse": 1.6, "mae": 1.3, "pearson_r": None, "spearman_rho": None,
                             "correlation_undefined_reason": "constant_prediction"})
    _write_lba_run(base, 1, {"rmse": 1.4, "mae": 1.1, "pearson_r": 0.5, "spearman_rho": 0.45,
                             "correlation_undefined_reason": None})
    row = report.collect("lba", tmp_path)[0]
    assert row["test_rmse"]["n"] == 2 and row["test_pearson_r"]["n"] == 1
    assert row["test_correlation_undefined"] == {"constant_prediction": 1}
    markdown = report.to_markdown("lba", [row])
    assert "constant_prediction" in markdown and "1 of 2 seeds" in markdown
    json.dumps(row, allow_nan=False)


def test_lba_report_with_every_correlation_null_prints_dashes(tmp_path):
    base = tmp_path / "downstream" / "lba" / "snap" / "linear_probe.identity30.joint.concat_pocket_ligand"
    _write_lba_run(base, 0, {"rmse": 1.6, "mae": 1.3, "pearson_r": None, "spearman_rho": None,
                             "correlation_undefined_reason": "constant_prediction"})
    row = report.collect("lba", tmp_path)[0]
    assert row["test_pearson_r"] is None and row["test_rmse"]["mean"] == pytest.approx(1.6)
    assert "1.600" in report.to_markdown("lba", [row])


def test_lba_report_reads_old_results_without_the_reason_key(tmp_path):
    base = tmp_path / "downstream" / "lba" / "snap" / "linear_probe.identity30.joint.concat_pocket_ligand"
    _write_lba_run(base, 0, {"rmse": 1.4, "mae": 1.1, "pearson_r": 0.5, "spearman_rho": 0.45})
    row = report.collect("lba", tmp_path)[0]
    assert row["test_correlation_undefined"] == {}
    markdown = report.to_markdown("lba", [row])
    assert "constant_prediction" not in markdown and "Spearman rho" in markdown


def test_probe_headline_survives_a_constant_prediction(tmp_path):
    from downstream import probe

    rng = np.random.default_rng(0)
    outputs = tmp_path / "out"
    for split, count in {"train": 30, "val": 10, "test": 10}.items():
        features = np.zeros((count, 4), dtype=np.float32)  # identical features -> the head predicts a constant
        labels = rng.normal(6.0, 1.0, (count, 1)).astype(np.float32)
        path = outputs / "downstream" / "lba" / "toy" / "embeddings" / f"identity30.joint.{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, chains=np.array([f"{split}{i}" for i in range(count)]), labels=labels,
                 emb_concat_pocket_ligand=features)
        path.with_suffix(".json").write_text(json.dumps({"split": split}))
    assert probe.main(["--task", "lba", "--model-id", "toy", "--head", "linear", "--seeds", "0", "--pooling",
                       "concat_pocket_ligand", "--outputs-root", str(outputs), "--threads", "1"]) == 0
    run = outputs / "downstream" / "lba" / "toy"
    metrics = json.loads(next(run.glob("*/seed0/metrics.json")).read_text())
    assert metrics["test"]["all"]["rmse"] > 0
    assert metrics["test"]["all"]["pearson_r"] is None
    assert metrics["test"]["all"]["correlation_undefined_reason"] == "constant_prediction"
    assert report.main(["--task", "lba", "--outputs-root", str(outputs)]) == 0


# ---------------------------------------------------------------------------
# 6. Multi-label report: macro AUPR is an extra column, old columns keep their place
# ---------------------------------------------------------------------------

def _write_ec_run(base, seed, test_all, cutoff=None):
    directory = base / f"seed{seed}"
    directory.mkdir(parents=True)
    test = {"all": test_all}
    if cutoff is not None:
        test["cutoff_30"] = cutoff
    (directory / "metrics.json").write_text(json.dumps({
        "task": "ec", "kind": "multilabel", "seed": seed, "val": {"fmax": 0.5}, "test": test}))


_OLD_EC = {"fmax": 0.6, "pair_aupr": 0.5, "top1_accuracy": 0.7}


def test_multilabel_report_adds_macro_aupr_after_the_old_columns(tmp_path):
    base = tmp_path / "downstream" / "ec" / "snap" / "linear_probe.mean_atoms"
    new = dict(_OLD_EC, macro_aupr=0.4, macro_aupr_num_terms=3, macro_aupr_num_terms_skipped=1)
    _write_ec_run(base, 0, new, cutoff=dict(new))
    row = report.collect("ec", tmp_path)[0]
    assert row["test_macro_aupr"]["mean"] == pytest.approx(0.4)
    assert row["test_macro_aupr_num_terms_skipped"]["mean"] == 1
    header = report.to_markdown("ec", [row]).splitlines()[0]
    assert "macro AUPR" in header and "pair AUPR" in header
    assert header.index("pair AUPR") < header.index("Fmax <30%") < header.index("macro AUPR")


def test_multilabel_report_reads_old_results_without_macro_aupr(tmp_path):
    base = tmp_path / "downstream" / "ec" / "snap" / "linear_probe.mean_atoms"
    _write_ec_run(base, 0, dict(_OLD_EC), cutoff=dict(_OLD_EC))
    row = report.collect("ec", tmp_path)[0]
    assert row["test_macro_aupr"] is None and row["test_pair_aupr"]["mean"] == pytest.approx(0.5)
    markdown = report.to_markdown("ec", [row])
    assert "0.500" in markdown and "macro AUPR" in markdown
    json.dumps(row, allow_nan=False)

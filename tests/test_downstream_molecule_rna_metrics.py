"""Pure metric functions for the MoleculeNet and RNA (rnaglib) downstream tasks (no task code exists yet).

Hand-computed values; invalid input must raise, and an undefined value is ``None`` with a named reason,
never a silent zero.
"""

import json
import math

import numpy as np
import pytest

from downstream.common.metrics import (
    binary_classification_metrics,
    multiclass_classification_metrics,
    multilabel_classification_metrics,
    multitarget_regression_errors,
    multitask_prc_auc,
    multitask_roc_auc,
)

NAN = float("nan")


# ---------------------------------------------------------------------------
# MoleculeNet: masked multi-task ROC-AUC and PRC-AUC
# ---------------------------------------------------------------------------

def test_multitask_roc_auc_masks_missing_labels_and_averages_tasks():
    labels = np.array([[1, 1], [0, NAN], [1, 0], [0, 1], [NAN, 0]])
    scores = np.array([[0.9, 0.2], [0.1, 0.9], [0.8, 0.6], [0.3, 0.7], [0.5, 0.1]])
    # task 0 uses rows 0-3: positives 0.9, 0.8 above negatives 0.1, 0.3 -> AUC 1
    # task 1 uses rows 0, 2, 3, 4: positives 0.2, 0.7; negatives 0.6, 0.1 -> (1 + 1 + 0 + 1) / 4 = 3/4
    result = multitask_roc_auc(labels, scores)
    assert result["per_task"] == [pytest.approx(1.0), pytest.approx(0.75)]
    assert result["roc_auc"] == pytest.approx(0.875)
    assert result["num_tasks"] == 2 and result["num_tasks_evaluated"] == 2 and result["num_tasks_skipped"] == 0
    assert result["skipped_task_indices"] == []


def test_multitask_roc_auc_skips_single_class_tasks_and_counts_them():
    labels = np.array([[1, 1, NAN], [0, 1, NAN], [1, 1, 0], [0, 1, NAN]])
    scores = np.array([[0.9, 0.2, 0.5], [0.1, 0.9, 0.5], [0.8, 0.6, 0.5], [0.3, 0.7, 0.5]])
    result = multitask_roc_auc(labels, scores)
    # task 1 is all positive, task 2 has one labelled negative only: both undefined
    assert result["per_task"][0] == pytest.approx(1.0) and result["per_task"][1] is None and result["per_task"][2] is None
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["num_tasks_evaluated"] == 1 and result["num_tasks_skipped"] == 2
    assert result["skipped_task_indices"] == [1, 2]
    json.dumps(result, allow_nan=False)


def test_multitask_roc_auc_single_task_one_dimensional_input():
    result = multitask_roc_auc(np.array([1, 0, 1, 0]), np.array([0.9, 0.2, 0.4, 0.5]))
    assert result["roc_auc"] == pytest.approx(0.75) and result["num_tasks"] == 1


@pytest.mark.parametrize("function", [multitask_roc_auc, multitask_prc_auc])
def test_multitask_classification_fails_loudly(function):
    good_scores = np.array([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="every task is single-class"):
        function(np.array([[1, NAN], [1, NAN]]), good_scores)
    with pytest.raises(ValueError, match="NaN or inf"):
        function(np.array([[1, 0], [0, 1]]), np.array([[0.1, NAN], [0.3, 0.4]]))
    with pytest.raises(ValueError, match="must have equal shapes"):
        function(np.array([[1, 0], [0, 1]]), good_scores[:, :1])
    with pytest.raises(ValueError, match="0, 1 or NaN"):
        function(np.array([[2, 0], [0, 1]]), good_scores)
    with pytest.raises(ValueError, match="at least one sample"):
        function(np.zeros((0, 2)), np.zeros((0, 2)))


def test_multitask_prc_auc_is_average_precision_per_task():
    labels = np.array([[1, 1], [0, 0], [1, NAN], [0, 1]])
    scores = np.array([[0.9, 0.5], [0.8, 0.1], [0.4, 0.3], [0.2, 0.6]])
    # task 0: ranking 0.9(+) 0.8(-) 0.4(+) 0.2(-) -> AP = (1 + 2/3) / 2 = 5/6
    # task 1 (rows 0, 1, 3): ranking 0.6(+) 0.5(+) 0.1(-) -> AP = 1
    result = multitask_prc_auc(labels, scores)
    assert result["per_task"] == [pytest.approx(5 / 6), pytest.approx(1.0)]
    assert result["prc_auc"] == pytest.approx((5 / 6 + 1) / 2)
    assert result["num_tasks_skipped"] == 0


def test_multitask_prc_auc_skips_tasks_without_both_classes():
    labels = np.array([[1, 0], [0, 0], [1, 0]])
    scores = np.array([[0.9, 0.5], [0.1, 0.4], [0.8, 0.3]])
    result = multitask_prc_auc(labels, scores)
    assert result["per_task"][1] is None and result["num_tasks_skipped"] == 1
    assert result["prc_auc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MoleculeNet: masked multi-target RMSE / MAE
# ---------------------------------------------------------------------------

def test_multitarget_regression_errors_mask_missing_targets():
    targets = np.array([[1.0, 10.0], [2.0, NAN], [3.0, 14.0], [4.0, 12.0]])
    predictions = np.array([[1.5, 11.0], [2.0, 99.0], [2.0, 14.0], [5.0, 15.0]])
    result = multitarget_regression_errors(targets, predictions)
    # target 0 errors 0.5, 0, 1, 1; target 1 (rows 0, 2, 3) errors 1, 0, 3 (the NaN row, whatever predicted, is ignored)
    assert result["rmse_per_target"] == [pytest.approx(math.sqrt((0.25 + 0 + 1 + 1) / 4)),
                                         pytest.approx(math.sqrt((1 + 0 + 9) / 3))]
    assert result["mae_per_target"] == [pytest.approx(2.5 / 4), pytest.approx(4 / 3)]
    assert result["rmse"] == pytest.approx(np.mean(result["rmse_per_target"]))
    assert result["mae"] == pytest.approx(np.mean(result["mae_per_target"]))
    assert result["num_targets"] == 2 and result["num_labeled_per_target"] == [4, 3]


def test_multitarget_regression_one_dimensional_single_target():
    result = multitarget_regression_errors(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert result["rmse"] == pytest.approx(math.sqrt(4 / 3)) and result["mae"] == pytest.approx(2 / 3)


def test_multitarget_regression_fails_loudly():
    with pytest.raises(ValueError, match="no labelled sample"):
        multitarget_regression_errors(np.array([[1.0, NAN], [2.0, NAN]]), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="NaN or inf"):
        multitarget_regression_errors(np.array([[1.0], [2.0]]), np.array([[1.0], [NAN]]))
    with pytest.raises(ValueError, match="inf"):
        multitarget_regression_errors(np.array([[1.0], [float("inf")]]), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="must have equal shapes"):
        multitarget_regression_errors(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="at least one sample"):
        multitarget_regression_errors(np.zeros((0, 1)), np.zeros((0, 1)))


# ---------------------------------------------------------------------------
# RNA: binary tasks (RNA-Site, RNA-Prot, RNA-CM, RNA-Ligand, RNA-VS)
# ---------------------------------------------------------------------------

def test_binary_metrics_hand_example():
    labels = np.array([1, 1, 0, 0, 0, 1])
    scores = np.array([0.9, 0.4, 0.6, 0.2, 0.1, 0.8])
    # threshold 0.5 -> predictions 1 0 1 0 0 1: TP 2, FN 1, FP 1, TN 2
    result = binary_classification_metrics(labels, scores)
    assert result["balanced_accuracy"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["jaccard"] == pytest.approx(0.5)
    assert result["mcc"] == pytest.approx(1 / 3)
    assert result["auc"] == pytest.approx(8 / 9)
    assert result["threshold"] == 0.5 and result["num_samples"] == 6 and result["num_positive"] == 3
    assert result["undefined"] == {}


def test_binary_metrics_label_undefined_values_instead_of_zero():
    # every label is positive: balanced accuracy and AUC need both classes
    result = binary_classification_metrics(np.array([1, 1, 1]), np.array([0.9, 0.2, 0.7]))
    assert result["balanced_accuracy"] is None and result["auc"] is None
    assert result["undefined"]["balanced_accuracy"] == "single_class_labels"
    assert result["undefined"]["auc"] == "single_class_labels"
    assert result["f1"] == pytest.approx(0.8) and result["jaccard"] == pytest.approx(2 / 3)
    # a model that predicts "all negative": MCC has a zero marginal, F1 and Jaccard are a real 0
    collapsed = binary_classification_metrics(np.array([1, 0, 0, 1]), np.array([0.1, 0.2, 0.1, 0.3]))
    assert collapsed["mcc"] is None and collapsed["undefined"]["mcc"] == "zero_marginal"
    assert collapsed["f1"] == 0.0 and collapsed["jaccard"] == 0.0 and collapsed["balanced_accuracy"] == 0.5
    json.dumps(collapsed, allow_nan=False)


def test_binary_metrics_threshold_is_explicit():
    labels, scores = np.array([1, 0, 1, 0]), np.array([0.6, 0.55, 0.4, 0.1])
    assert binary_classification_metrics(labels, scores)["f1"] == pytest.approx(0.5)
    assert binary_classification_metrics(labels, scores, threshold=0.35)["f1"] == pytest.approx(0.8)
    assert binary_classification_metrics(labels, scores, threshold=0.35)["threshold"] == 0.35


def test_binary_metrics_fail_loudly():
    with pytest.raises(ValueError, match="0/1"):
        binary_classification_metrics(np.array([0, 2]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="NaN or inf"):
        binary_classification_metrics(np.array([0, 1]), np.array([0.1, NAN]))
    with pytest.raises(ValueError, match="must have equal shapes"):
        binary_classification_metrics(np.array([0, 1]), np.array([0.1]))
    with pytest.raises(ValueError, match="at least one sample"):
        binary_classification_metrics(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# RNA: multi-label (RNA-GO)
# ---------------------------------------------------------------------------

_ML_LABELS = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0]])
_ML_SCORES = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.4], [0.6, 0.15, 0.1], [0.1, 0.2, 0.7]])
# predictions at 0.5: [[1,0,1],[0,1,0],[1,0,0],[0,0,1]]


def test_multilabel_metrics_hand_example_names_every_average():
    result = multilabel_classification_metrics(_ML_LABELS, _ML_SCORES)
    # per label F1: 1, 2/3, 2/3; Jaccard: 1, 1/2, 1/2 ; counts tp 4, fp 1, fn 1
    assert result["f1_macro"] == pytest.approx(7 / 9)
    assert result["f1_micro"] == pytest.approx(0.8)
    assert result["jaccard_macro"] == pytest.approx(2 / 3)
    assert result["jaccard_micro"] == pytest.approx(2 / 3)
    # per sample F1: 1, 1, 2/3, 0 ; Jaccard: 1, 1, 1/2, 0
    assert result["f1_samples"] == pytest.approx(2 / 3)
    assert result["jaccard_samples"] == pytest.approx(0.625)
    # per label balanced accuracy 1, 3/4, 5/6 ; AUC 1, 3/4, 1
    assert result["balanced_accuracy_macro"] == pytest.approx(31 / 36)
    assert result["auc_macro"] == pytest.approx(11 / 12)
    assert result["num_labels"] == 3 and result["threshold"] == 0.5
    assert result["undefined"] == {}


def test_multilabel_metrics_count_skipped_labels_and_samples():
    labels = np.hstack([_ML_LABELS, np.zeros((4, 1), dtype=int)])
    scores = np.hstack([_ML_SCORES, np.full((4, 1), 0.1)])  # a label nobody has and nobody predicts
    result = multilabel_classification_metrics(labels, scores)
    assert result["f1_macro"] == pytest.approx(7 / 9)  # the empty label is left out of the mean, not averaged as 0
    assert result["auc_macro"] == pytest.approx(11 / 12) and result["balanced_accuracy_macro"] == pytest.approx(31 / 36)
    assert result["num_labels"] == 4
    assert result["num_labels_skipped"] == {"f1": 1, "jaccard": 1, "balanced_accuracy": 1, "auc": 1}
    empty_rows = multilabel_classification_metrics(np.array([[1, 0], [0, 0]]), np.array([[0.9, 0.1], [0.1, 0.2]]))
    assert empty_rows["num_samples_skipped"] == {"f1": 1, "jaccard": 1}
    assert empty_rows["f1_samples"] == pytest.approx(1.0)  # only the first row is defined
    json.dumps(result, allow_nan=False)


def test_multilabel_metrics_fail_loudly():
    with pytest.raises(ValueError, match="0/1"):
        multilabel_classification_metrics(np.array([[0, 3]]), np.array([[0.1, 0.2]]))
    with pytest.raises(ValueError, match="NaN or inf"):
        multilabel_classification_metrics(np.array([[0, 1]]), np.array([[0.1, NAN]]))
    with pytest.raises(ValueError, match="2-D"):
        multilabel_classification_metrics(np.array([0, 1]), np.array([0.1, 0.2]))


def test_multilabel_auc_is_undefined_not_zero_when_no_label_has_both_classes():
    result = multilabel_classification_metrics(np.array([[1, 1], [1, 1]]), np.array([[0.9, 0.8], [0.7, 0.6]]))
    assert result["auc_macro"] is None and result["balanced_accuracy_macro"] is None
    assert result["undefined"]["auc_macro"] == "no_label_has_both_classes"
    assert result["undefined"]["balanced_accuracy_macro"] == "no_label_has_both_classes"
    assert result["f1_macro"] == pytest.approx(1.0)
    json.dumps(result, allow_nan=False)


# ---------------------------------------------------------------------------
# RNA: multi-class (RNA-IF nucleotide identity, sequence recovery)
# ---------------------------------------------------------------------------

_MC_TARGETS = np.array([0, 0, 1, 2, 2, 2])
_MC_PRED = np.array([0, 1, 1, 2, 2, 0])


def test_multiclass_metrics_hand_example():
    result = multiclass_classification_metrics(_MC_TARGETS, np.eye(4)[_MC_PRED])
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["sequence_recovery"] == result["accuracy"]
    assert result["balanced_accuracy"] == pytest.approx((1 / 2 + 1 + 2 / 3) / 3)
    assert result["f1_macro"] == pytest.approx((0.5 + 2 / 3 + 0.8) / 3)
    assert result["jaccard_macro"] == pytest.approx(0.5)
    assert result["mcc"] == pytest.approx(12 / math.sqrt(24 * 22))
    assert result["num_classes"] == 4 and result["num_classes_present"] == 3
    assert result["undefined"] == {}


def test_multiclass_auc_is_one_vs_rest_over_present_classes():
    perfect = np.eye(4)[_MC_TARGETS]
    result = multiclass_classification_metrics(_MC_TARGETS, perfect)
    assert result["auc_ovr_macro"] == pytest.approx(1.0)
    assert result["num_classes_skipped_auc"] == 1  # class 3 never occurs, so it has no ROC curve
    assert result["accuracy"] == 1.0 and result["mcc"] == pytest.approx(1.0)


def test_multiclass_mcc_is_undefined_when_one_class_is_always_predicted():
    result = multiclass_classification_metrics(np.array([0, 1, 0, 1]), np.array([[1, 0]] * 4))
    assert result["mcc"] is None and result["undefined"]["mcc"] == "zero_marginal"
    assert result["accuracy"] == 0.5 and result["balanced_accuracy"] == 0.5
    json.dumps(result, allow_nan=False)


def test_multiclass_metrics_fail_loudly():
    scores = np.eye(3)[[0, 1, 2]]
    with pytest.raises(ValueError, match="must be integers"):
        multiclass_classification_metrics(np.array([0.0, 1.0, 2.0]), scores)
    with pytest.raises(ValueError, match="class ids must lie in"):
        multiclass_classification_metrics(np.array([0, 1, 3]), scores)
    with pytest.raises(ValueError, match="NaN or inf"):
        multiclass_classification_metrics(np.array([0, 1, 2]), np.where(scores > 0, NAN, scores))

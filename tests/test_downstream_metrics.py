"""Hand-computed checks of protein-centric Fmax, pair AUPR and accuracies."""

import numpy as np
import pytest

from downstream.common.metrics import evaluate, pair_aupr, protein_fmax, subset_accuracy, top1_accuracy

LABELS = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
SCORES = np.array([[0.9, 0.2, 0.1], [0.3, 0.8, 0.4], [0.6, 0.1, 0.05]])


def test_fmax_hand_example():
    # t = 0.31..0.40: P1 {0}, P2 {1, 2}, P3 {0} -> precision 1, recall (1 + 1 + 1/2) / 3 = 5/6.
    result = protein_fmax(LABELS, SCORES)
    assert result["fmax"] == pytest.approx(10 / 11)
    assert result["threshold"] == pytest.approx(0.31)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(5 / 6)


def test_precision_ignores_proteins_without_predictions_but_recall_counts_them():
    labels = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    scores = np.array([[0.9, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    # t >= 0.01: only P1 predicts -> precision 1 (P2 excluded), recall 1/2 -> F = 2/3.
    # Averaging precision over all proteins would give 1/2 -> F = 1/2; t = 0 gives P 1/4, R 1 -> F = 0.4.
    assert protein_fmax(labels, scores)["fmax"] == pytest.approx(2 / 3)


def test_pair_aupr_hand_examples():
    assert pair_aupr(np.array([[1, 0], [0, 1]]), np.array([[0.9, 0.4], [0.3, 0.8]])) == pytest.approx(1.0)
    # ranking 0.9(-) 0.5(-) 0.4(+) 0.2(+): AP = (1/3 + 2/4) / 2
    assert pair_aupr(np.array([[1, 0], [0, 1]]), np.array([[0.2, 0.9], [0.5, 0.4]])) == pytest.approx(5 / 12)


def test_accuracies():
    assert top1_accuracy(LABELS, SCORES) == pytest.approx(1.0)
    assert subset_accuracy(LABELS, SCORES, 0.5) == pytest.approx(1 / 3)
    assert evaluate(LABELS, SCORES)["num_proteins"] == 3


@pytest.mark.parametrize("labels, scores, message", [
    (np.array([[1, 0], [0, 0]]), np.zeros((2, 2)), "no true term"),
    (np.array([[1, 0]]), np.array([[np.nan, 0.1]]), "NaN"),
    (np.array([[1, 0]]), np.zeros((1, 3)), "must be equal"),
    (np.array([[2, 0]]), np.zeros((1, 2)), "0/1"),
])
def test_invalid_inputs_fail_loudly(labels, scores, message):
    with pytest.raises(ValueError, match=message):
        protein_fmax(labels, scores)

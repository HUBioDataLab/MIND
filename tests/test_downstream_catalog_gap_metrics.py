"""The five catalog metrics that had no code: EC coverage, GO IC-binned macro AUPR (go_mf / go_bp / go_cc)
and Fold weighted F1. Hand-computed examples; the catalog must no longer list any required cell as missing.

Definitions (source: DeepFRI, Nat. Commun. 2021 p.5 / Fig. 3 legend / Methods eq. 1; PEER App. B.1):
- coverage: fraction of test proteins with at least one predicted term at the Fmax threshold.
- IC-binned macro AUPR: label-centric AUPR over the terms with IC(term) > t; IC = -log2(n_term / n_proteins).
- weighted F1: per-class F1 averaged with the class support as weight (sklearn ``average="weighted"``).
"""

import json

import numpy as np
import pytest
import yaml

from downstream import ledger, probe
from downstream.common.metrics import (
    evaluate,
    evaluate_multiclass,
    ic_binned_macro_aupr,
    protein_fmax,
    train_term_ic,
    weighted_f1,
)
from downstream.tasks.go import mf
from ledger_helpers import SHA_A, card, ec_metrics, write_probe

# Four proteins; the last one has such low scores that it is not predicted at the best threshold.
COV_LABELS = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
COV_SCORES = np.array([[0.9, 0.1], [0.3, 0.2], [0.8, 0.05], [0.02, 0.01]])


# ---------------------------------------------------------------------------
# 1. EC coverage
# ---------------------------------------------------------------------------

def test_coverage_at_the_fmax_threshold_hand_example():
    # t in (0.10, 0.20]: proteins 0, 1, 2 predict (precision 1, 1/2, 1 -> 5/6), protein 3 predicts nothing,
    # recall = (1 + 1 + 1 + 0) / 4 = 3/4, F = 15/19. Every other threshold gives a smaller F.
    best = protein_fmax(COV_LABELS, COV_SCORES)
    assert best["threshold"] == pytest.approx(0.11)
    assert best["fmax"] == pytest.approx(15 / 19)
    assert best["precision"] == pytest.approx(5 / 6) and best["recall"] == pytest.approx(3 / 4)
    assert best["coverage"] == pytest.approx(3 / 4)


def test_evaluate_reports_coverage_next_to_fmax():
    result = evaluate(COV_LABELS, COV_SCORES)
    assert result["coverage"] == pytest.approx(0.75)
    assert result["fmax"] == pytest.approx(15 / 19)  # unchanged


def test_coverage_is_one_when_every_protein_predicts_at_the_best_threshold():
    labels = np.array([[1, 0], [0, 1], [1, 0]])
    scores = np.array([[0.9, 0.2], [0.3, 0.8], [0.7, 0.1]])
    assert evaluate(labels, scores)["coverage"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. GO IC-binned macro AUPR
# ---------------------------------------------------------------------------

IC_LABELS = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 0, 0]])
IC_SCORES = np.array([[0.7, 0.9, 0.2, 0.5], [0.1, 0.4, 0.8, 0.5], [0.6, 0.5, 0.1, 0.5], [0.2, 0.3, 0.3, 0.5]])
IC_VALUES = np.array([2.0, 6.0, 11.0, 7.0])  # term 3 has IC 7 but no positive protein


def test_train_term_ic_is_minus_log2_of_the_train_frequency():
    train = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0]])  # counts 2, 2, 0 over 4 proteins
    ic = train_term_ic(train)
    assert ic[0] == pytest.approx(1.0) and ic[1] == pytest.approx(1.0)
    assert np.isinf(ic[2]) and ic[2] > 0  # a term never seen in train is the most specific one
    with pytest.raises(ValueError, match="at least one"):
        train_term_ic(np.zeros((0, 3)))


def test_ic_binned_macro_aupr_hand_example():
    # term 1 (IC 6): scores .9(+) .5(-) .4(+) .3(+) -> AP (1 + 2/3 + 3/4) / 3 = 29/36
    result = ic_binned_macro_aupr(IC_LABELS, IC_SCORES, IC_VALUES, thresholds=(5.0, 10.0))
    above5, above10 = result["ic_gt5"], result["ic_gt10"]
    # IC > 5: terms 1 (6), 2 (11), 3 (7). Term 3 has no positive and is skipped, never counted as zero.
    assert above5["num_terms_evaluated"] == 2 and above5["num_terms_skipped_no_positive"] == 1
    # term 2 column: labels [0, 1, 1, 0], scores [.2, .8, .1, .3] -> ranking .8(+) .3(-) .2(-) .1(+): AP (1 + 2/4) / 2 = 3/4
    assert above5["macro_aupr"] == pytest.approx((29 / 36 + 3 / 4) / 2)
    assert above10["macro_aupr"] == pytest.approx(3 / 4) and above10["num_terms_evaluated"] == 1


def test_ic_bin_without_an_evaluable_term_has_no_value():
    result = ic_binned_macro_aupr(IC_LABELS, IC_SCORES, IC_VALUES, thresholds=(50.0,))
    assert result["ic_gt50"] == {"macro_aupr": None, "num_terms_evaluated": 0, "num_terms_skipped_no_positive": 0}


def test_ic_binned_threshold_is_strict():
    ic = np.array([5.0, 5.0, 5.0, 5.0])
    assert ic_binned_macro_aupr(IC_LABELS, IC_SCORES, ic, thresholds=(5.0,))["ic_gt5"]["num_terms_evaluated"] == 0


def test_ic_binned_fails_loudly_on_a_wrong_ic_vector():
    with pytest.raises(ValueError, match="one value per term"):
        ic_binned_macro_aupr(IC_LABELS, IC_SCORES, IC_VALUES[:3])
    with pytest.raises(ValueError, match="NaN"):
        ic_binned_macro_aupr(IC_LABELS, IC_SCORES, np.array([1.0, np.nan, 3.0, 4.0]))


def test_evaluate_adds_the_ic_keys_only_with_a_term_ic():
    assert not any(key.startswith("ic_binned") for key in evaluate(IC_LABELS, IC_SCORES))
    result = evaluate(IC_LABELS, IC_SCORES, term_ic=IC_VALUES, ic_thresholds=(5.0, 10.0))
    assert result["ic_binned_macro_aupr"] == pytest.approx((29 / 36 + 3 / 4) / 2)  # first threshold = headline cell
    assert result["ic_binned_macro_aupr_num_terms"] == 2
    assert result["ic_binned_macro_aupr_ic_gt10"] == pytest.approx(3 / 4)
    assert result["ic_binned_macro_aupr_ic_gt10_num_terms"] == 1
    json.dumps(result, allow_nan=False)


def test_evaluate_counts_an_empty_bin_but_writes_no_value_for_it():
    result = evaluate(IC_LABELS, IC_SCORES, term_ic=IC_VALUES, ic_thresholds=(5.0, 50.0))
    assert result["ic_binned_macro_aupr_ic_gt50_num_terms"] == 0 and "ic_binned_macro_aupr_ic_gt50" not in result


# ---------------------------------------------------------------------------
# 3. Fold weighted F1
# ---------------------------------------------------------------------------

def test_weighted_f1_hand_example_and_sklearn_parity():
    from sklearn.metrics import f1_score

    scores = np.eye(4)[[0, 1, 1, 1]]  # predictions 0, 1, 1, 1
    targets = np.array([0, 0, 1, 2])
    # class 0: tp 1 fp 0 fn 1 -> F1 2/3, support 2; class 1: tp 1 fp 2 fn 0 -> F1 1/2, support 1;
    # class 2: tp 0 -> F1 0, support 1; class 3 never occurs and has no weight
    assert weighted_f1(targets, scores) == pytest.approx((2 * 2 / 3 + 1 * 0.5 + 0) / 4)
    assert weighted_f1(targets, scores) == pytest.approx(
        f1_score(targets, scores.argmax(axis=1), average="weighted", zero_division=0))


def test_weighted_f1_matches_sklearn_on_random_predictions():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(3)
    targets = rng.integers(0, 6, 80)
    scores = rng.random((80, 8))  # classes 6 and 7 can only be predicted, never true
    assert weighted_f1(targets, scores) == pytest.approx(
        f1_score(targets, scores.argmax(axis=1), average="weighted", zero_division=0))


def test_weighted_f1_equals_accuracy_for_perfect_and_differs_from_balanced_accuracy():
    perfect = np.eye(3)[[0, 1, 2, 0]]
    assert weighted_f1(np.array([0, 1, 2, 0]), perfect) == pytest.approx(1.0)
    skewed = evaluate_multiclass(np.array([0, 0, 0, 1]), np.eye(2)[[0, 0, 0, 0]])
    # class 0: tp 3 fp 1 fn 0 -> F1 6/7 (support 3); class 1: F1 0 (support 1) -> 18/28
    assert skewed["weighted_f1"] == pytest.approx(9 / 14)
    assert skewed["accuracy"] == pytest.approx(0.75) and skewed["balanced_accuracy"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. Catalog and ledger recognise the metrics
# ---------------------------------------------------------------------------

def test_catalog_has_no_required_cell_without_code_and_names_the_five_cells():
    catalog = ledger.load_catalog()
    for task, spec in catalog["tasks"].items():
        assert [c for c in spec["required"] if c.get("code", "present") == "missing"] == [], task
    wanted = {"ec": "coverage", "go_mf": "ic_binned_macro_aupr", "go_bp": "ic_binned_macro_aupr",
              "go_cc": "ic_binned_macro_aupr", "fold": "weighted_f1"}
    for task, metric in wanted.items():
        assert metric in {c["metric"] for c in catalog["tasks"][task]["required"]}, (task, metric)


def test_ledger_flattens_the_new_keys_into_rows_and_keeps_counts_as_context(tmp_path):
    outputs = tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    metrics = ec_metrics("model_a", 0, 0.5)
    metrics["task"] = "go_mf"
    for block in (metrics["val"], *metrics["test"].values()):
        block.update(coverage=0.9, ic_binned_macro_aupr=0.3, ic_binned_macro_aupr_num_terms=7,
                     ic_binned_macro_aupr_ic_gt10=0.2, ic_binned_macro_aupr_ic_gt10_num_terms=2)
    write_probe(outputs, metrics)
    root = tmp_path / "results"
    ledger.ingest(root, outputs_root=outputs, command="test")
    rows = ledger.read_records(root, "measurements")
    cell = {r["metric"]: r for r in rows if r["eval_split"] == "test" and r["subset"] == "all"}
    assert cell["ic_binned_macro_aupr"]["value"] == pytest.approx(0.3)
    assert cell["ic_binned_macro_aupr"]["unit"] == "fraction" and cell["ic_binned_macro_aupr"]["is_catalog_metric"]
    assert cell["ic_binned_macro_aupr"]["context"]["ic_binned_macro_aupr_num_terms"] == 7
    assert cell["coverage"]["value"] == pytest.approx(0.9)
    assert not any(r["metric"].endswith("_num_terms") for r in rows)  # counts are context, not metrics
    assert cell["ic_binned_macro_aupr_ic_gt10"]["is_catalog_metric"] is False


# ---------------------------------------------------------------------------
# 5. Probe wiring
# ---------------------------------------------------------------------------

def _write_embeddings(outputs, task, counts, num_classes, kind):
    rng = np.random.default_rng(0)
    for split, count in counts.items():
        ids = np.arange(count)
        labels = np.zeros((count, num_classes), dtype=np.uint8)
        labels[ids, ids % num_classes] = 1
        features = labels.astype(np.float32) * 2 + rng.normal(0, 0.3, labels.shape).astype(np.float32)
        path = outputs / "downstream" / task / "toy" / "embeddings" / f"{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, chains=np.array([f"{split}{i}" for i in range(count)]),
                 labels=labels if kind == "multilabel" else (ids % num_classes).astype(np.int64),
                 emb_mean_atoms=features)
        path.with_suffix(".json").write_text(json.dumps({"split": split}))


def test_go_task_yaml_declares_the_ic_bins():
    config = mf.load_task_config()
    assert config["ic_binned_aupr"]["ic_thresholds"] == [5.0, 10.0]
    assert config["ic_binned_aupr"]["ic_reference"] == "train_frequency"


def test_go_probe_writes_coverage_and_ic_binned_aupr_with_train_frequency_ic(tmp_path, monkeypatch):
    data = yaml.safe_load(mf.TASK_YAML.read_text())
    data["ic_binned_aupr"]["ic_thresholds"] = [1.0, 1.55]  # toy data: train IC is 1.51 / 1.62 / 1.62
    data["label_count"] = 3
    data["probe"].update(epochs=30, patience=30, lr=0.05, batch_size=16)
    config_path = tmp_path / "task.yaml"
    config_path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(mf, "TASK_YAML", config_path)
    outputs = tmp_path / "outputs"
    _write_embeddings(outputs, "go_mf", {"train": 40, "valid": 12, "test": 9}, 3, "multilabel")
    # the checked-in cutoff file is not used here: the toy test chains have no cutoff columns
    monkeypatch.setattr(mf, "test_subsets", lambda config, chains: {})
    assert probe.main(["--task", "go_mf", "--model-id", "toy", "--head", "linear", "--seeds", "0",
                       "--data-root", str(tmp_path / "deepfri" / "go"), "--outputs-root", str(outputs),
                       "--threads", "1"]) == 0
    run = outputs / "downstream" / "go_mf" / "toy" / "linear_probe.mean_atoms" / "seed0"
    test = json.loads((run / "metrics.json").read_text())["test"]["all"]
    assert 0.0 < test["coverage"] <= 1.0
    assert test["ic_binned_macro_aupr_num_terms"] == 3 and test["ic_binned_macro_aupr_ic_gt1.55_num_terms"] == 2
    assert 0.0 < test["ic_binned_macro_aupr"] <= 1.0


def test_ec_probe_has_coverage_but_no_ic_keys(tmp_path, monkeypatch):
    from downstream.tasks.ec import dataset as ec

    data = yaml.safe_load(ec.TASK_YAML.read_text())
    data["probe"].update(epochs=20, patience=20, lr=0.05, batch_size=16)
    config_path = tmp_path / "task.yaml"
    config_path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(ec, "TASK_YAML", config_path)
    monkeypatch.setattr(ec, "test_subsets", lambda config, chains: {})
    outputs = tmp_path / "outputs"
    _write_embeddings(outputs, "ec", {"train": 40, "valid": 12, "test": 9}, 3, "multilabel")
    assert probe.main(["--task", "ec", "--model-id", "toy", "--head", "linear", "--seeds", "0",
                       "--outputs-root", str(outputs), "--threads", "1"]) == 0
    run = outputs / "downstream" / "ec" / "toy" / "linear_probe.mean_atoms" / "seed0"
    test = json.loads((run / "metrics.json").read_text())["test"]["all"]
    assert "coverage" in test and not any(key.startswith("ic_binned") for key in test)


def test_fold_probe_writes_weighted_f1_per_test_split(tmp_path, monkeypatch):
    from downstream.tasks.fold import dataset as fold

    data = yaml.safe_load(fold.TASK_YAML.read_text())
    data["label_count"] = 3
    data["probe"].update(epochs=30, patience=30, lr=0.05, batch_size=16)
    config_path = tmp_path / "task.yaml"
    config_path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(fold, "TASK_YAML", config_path)
    outputs = tmp_path / "outputs"
    _write_embeddings(outputs, "fold", {"train": 60, "valid": 12, "test_fold": 9, "test_family": 6,
                                        "test_superfamily": 3}, 3, "multiclass")
    assert probe.main(["--task", "fold", "--model-id", "toy", "--head", "linear", "--seeds", "0",
                       "--outputs-root", str(outputs), "--threads", "1"]) == 0
    test = json.loads((outputs / "downstream" / "fold" / "toy" / "linear_probe.mean_atoms" / "seed0"
                       / "metrics.json").read_text())["test"]
    assert all(0.0 <= test[name]["weighted_f1"] <= 1.0 for name in ("test_fold", "test_family", "test_superfamily"))

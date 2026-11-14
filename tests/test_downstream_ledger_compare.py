"""Model comparison with a paired bootstrap over predictions.npz (stage 3)."""

from __future__ import annotations

import numpy as np
import pytest

from downstream import ledger, ledger_compare, ledger_report
from downstream.common.metrics import evaluate, evaluate_regression
from ledger_helpers import SHA_A, SHA_B, card, ec_metrics, ec_predictions, lba_metrics, write_probe


def _ec_run(outputs, model, seed, predictions, sha, **kwargs):
    metrics = ec_metrics(model, seed, 0.5)
    block = evaluate(predictions["test_labels"], predictions["test_scores"])
    metrics["test"] = {"all": block}
    metrics["val"] = block
    write_probe(outputs, metrics, sha=sha, predictions=predictions, **kwargs)


def _lba_run(outputs, model, seed, noise, sha, n=200):
    rng = np.random.default_rng(100 + seed)
    labels = np.linspace(2, 10, n).astype(np.float32)
    scores = (labels + rng.normal(0, noise, n)).astype(np.float32)
    metrics = lba_metrics(model, seed, 1.0)
    metrics["test"] = {"all": evaluate_regression(labels, scores)}
    write_probe(outputs, metrics, sha=sha, splits=("train", "val", "test"),
                predictions={"test_chains": np.array([f"L{i}" for i in range(n)]), "test_scores": scores,
                             "test_labels": labels, "valid_scores": scores[:2]})


def _state(tmp_path, build):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "A", SHA_A)
    card(outputs, "B", SHA_B)
    build(outputs)
    ledger.ingest(root, outputs_root=outputs, command="test")
    return ledger_report.load_state(root)


def _pair(result, task, metric, subset="all", eval_split="test"):
    return next(p for p in result["pairs"] if (p["task"], p["metric"], p["subset"], p["eval_split"]) ==
                (task, metric, subset, eval_split))


# -- C1: identical predictions give zero difference -----------------------------------------------------------

def test_identical_predictions_are_indistinguishable(tmp_path):
    def build(outputs):
        for seed in (0, 1, 2):
            predictions = ec_predictions(np.random.default_rng(seed))
            _ec_run(outputs, "A", seed, predictions, SHA_A)
            _ec_run(outputs, "B", seed, predictions, SHA_B)
    result = ledger_compare.compare(_state(tmp_path, build), "A", "B", n_boot=200)
    pair = _pair(result, "ec", "fmax")
    assert pair["diff_b_better"] == 0 and pair["ci_low"] == 0 and pair["ci_high"] == 0
    assert pair["decision"] == "ayırt edilemez" and pair["common_seeds"] == [0, 1, 2]


# -- C2: a known difference has the right sign; lower-is-better metrics flip --------------------------------

def test_known_difference_has_the_right_sign(tmp_path):
    def build(outputs):
        for seed in (0, 1, 2):
            _lba_run(outputs, "A", seed, 1.0, SHA_A)
            _lba_run(outputs, "B", seed, 0.2, SHA_B)
            _ec_run(outputs, "A", seed, ec_predictions(np.random.default_rng(seed), quality=0.2), SHA_A)
            _ec_run(outputs, "B", seed, ec_predictions(np.random.default_rng(seed), quality=0.9), SHA_B)
    result = ledger_compare.compare(_state(tmp_path, build), "A", "B", n_boot=300)
    rmse = _pair(result, "lba", "rmse")
    assert rmse["mean_b"] < rmse["mean_a"] and rmse["diff_b_better"] > 0 and rmse["ci_low"] > 0
    assert rmse["decision"] == "B daha iyi"
    fmax = _pair(result, "ec", "fmax")
    assert fmax["mean_b"] > fmax["mean_a"] and fmax["diff_b_better"] > 0 and fmax["decision"] == "B daha iyi"
    reverse = ledger_compare.compare(_state(tmp_path / "r", build), "B", "A", n_boot=300)
    assert _pair(reverse, "lba", "rmse")["decision"] == "B daha kötü"


# -- C3: rows are aligned by chain id; different test sets are refused -------------------------------------

def test_permuted_predictions_are_aligned(tmp_path):
    def build(outputs):
        for seed in (0, 1):
            predictions = ec_predictions(np.random.default_rng(seed))
            order = np.random.default_rng(99).permutation(len(predictions["test_chains"]))
            _ec_run(outputs, "A", seed, predictions, SHA_A)
            _ec_run(outputs, "B", seed, {k: (v[order] if k != "valid_scores" else v) for k, v in predictions.items()},
                    SHA_B)
    pair = _pair(ledger_compare.compare(_state(tmp_path, build), "A", "B", n_boot=100), "ec", "fmax")
    assert pair["ci_low"] == pytest.approx(0) and pair["ci_high"] == pytest.approx(0)


def test_different_test_chains_are_an_error(tmp_path):
    def build(outputs):
        predictions = ec_predictions(np.random.default_rng(0))
        _ec_run(outputs, "A", 0, predictions, SHA_A)
        other = dict(predictions, test_chains=np.array([f"X{i}" for i in range(len(predictions["test_chains"]))]))
        _ec_run(outputs, "B", 0, other, SHA_B)
    with pytest.raises(ValueError, match="test chains"):
        ledger_compare.compare(_state(tmp_path, build), "A", "B", n_boot=50)


def test_predictions_that_do_not_reproduce_the_ledger_value_are_an_error(tmp_path):
    def build(outputs):
        predictions = ec_predictions(np.random.default_rng(0))
        _ec_run(outputs, "A", 0, predictions, SHA_A)
        _ec_run(outputs, "B", 0, predictions, SHA_B)
        path = outputs / "downstream" / "ec" / "B" / "mlp_probe.mean_atoms" / "seed0" / "predictions.npz"
        np.savez_compressed(path, **dict(predictions, test_scores=1 - predictions["test_scores"]))
    with pytest.raises(ValueError, match="does not reproduce"):
        ledger_compare.compare(_state(tmp_path, build), "A", "B", n_boot=50)


# -- C4: cells only one model has are listed; settings never mix --------------------------------------------

def test_cells_of_only_one_model_are_listed(tmp_path):
    def build(outputs):
        predictions = ec_predictions(np.random.default_rng(0))
        _ec_run(outputs, "A", 0, predictions, SHA_A)
        _ec_run(outputs, "B", 0, predictions, SHA_B)
        _lba_run(outputs, "A", 0, 1.0, SHA_A)
    result = ledger_compare.compare(_state(tmp_path, build), "A", "B", n_boot=50)
    assert any(c["task"] == "lba" and c["metric"] == "rmse" for c in result["only_a"])
    assert not result["only_b"]


def test_different_data_revisions_are_not_compared(tmp_path):
    def build(outputs):
        predictions = ec_predictions(np.random.default_rng(0))
        _ec_run(outputs, "A", 0, predictions, SHA_A)
        _ec_run(outputs, "B", 0, predictions, SHA_B, revision="rev_v1")
    result = ledger_compare.compare(_state(tmp_path, build), "A", "B", n_boot=50)
    assert not [p for p in result["pairs"] if p["task"] == "ec"]
    entry = next(c for c in result["only_a"] if c["metric"] == "fmax" and c["subset"] == "all")
    assert "farklı ayar" in entry["reason"]


def test_subsets_without_masks_get_no_interval(tmp_path):
    def build(outputs):
        for seed in (0, 1):
            predictions = ec_predictions(np.random.default_rng(seed))
            for model, sha in (("A", SHA_A), ("B", SHA_B)):
                metrics = ec_metrics(model, seed, 0.5)
                metrics["test"] = {"all": evaluate(predictions["test_labels"], predictions["test_scores"]),
                                   "cutoff_30": evaluate(predictions["test_labels"][:10], predictions["test_scores"][:10])}
                write_probe(outputs, metrics, sha=sha, predictions=predictions)
    pair = _pair(ledger_compare.compare(_state(tmp_path, build), "A", "B", n_boot=50), "ec", "fmax", "cutoff_30")
    assert pair["ci_low"] is None and pair["decision"].startswith("karar yok")


# -- C5: the report states all three noise sources and the multiple-comparison warning -----------------------

def test_report_states_noise_sources_and_multiple_comparisons(tmp_path):
    def build(outputs):
        for seed in (0, 1, 2):
            predictions = ec_predictions(np.random.default_rng(seed))
            _ec_run(outputs, "A", seed, predictions, SHA_A)
            _ec_run(outputs, "B", seed, predictions, SHA_B)
    state = _state(tmp_path, build)
    result = ledger_compare.compare(state, "A", "B", n_boot=100)
    markdown = ledger_compare.compare_markdown(state, result)
    assert "ön-eğitim tohumu gürültüsü ölçülmedi" in markdown
    assert "1. Probe tohumu" in markdown and "2. Test örneklemi" in markdown and "3. Ön-eğitim koşusu" in markdown
    with_ci = sum(p["ci_low"] is not None for p in result["pairs"])
    assert f"{with_ci} karşılaştırma" in markdown and f"{0.05 * with_ci:.1f}" in markdown
    assert state["header"]["measurements_sha256"] in markdown
    written = ledger_compare.write_compare(state["root"], "A", "B", n_boot=100, state=state)
    assert [p.name for p in written] == ["A__vs__B.md", "A__vs__B.csv"]


def test_single_seed_cells_get_no_decision(tmp_path):
    def build(outputs):
        predictions = ec_predictions(np.random.default_rng(0))
        _ec_run(outputs, "A", 0, predictions, SHA_A)
        _ec_run(outputs, "B", 0, ec_predictions(np.random.default_rng(0), quality=0.9), SHA_B)
    pair = _pair(ledger_compare.compare(_state(tmp_path, build), "A", "B", n_boot=50), "ec", "fmax")
    assert pair["std_a"] is None and pair["ci_low"] is not None
    assert pair["decision"] == "karar yok: tek tohum, probe-tohumu std'si tanımsız"


# -- Seed sets: mean, std and difference come from the seeds both models share, like the interval -----------

def _ledger_values(state, model, task, metric):
    return {r["run_seed"]: r["value"] for r in state["rows"]
            if (r["model_key"], r["task"], r["metric"], r["eval_split"], r["subset"]) == (model, task, metric, "test", "all")}


def test_statistics_use_only_the_common_seeds(tmp_path):
    def build(outputs):
        for seed, noise in ((0, 0.2), (1, 1.0), (2, 1.2)):
            _lba_run(outputs, "A", seed, noise, SHA_A)
        for seed, noise in ((1, 0.4), (2, 0.5), (3, 3.0)):
            _lba_run(outputs, "B", seed, noise, SHA_B)
    state = _state(tmp_path, build)
    pair = _pair(ledger_compare.compare(state, "A", "B", n_boot=100), "lba", "rmse")
    a, b = _ledger_values(state, "A", "lba", "rmse"), _ledger_values(state, "B", "lba", "rmse")
    assert pair["common_seeds"] == [1, 2] and pair["n_seeds_a"] == pair["n_seeds_b"] == 2
    assert pair["mean_a"] == pytest.approx(np.mean([a[1], a[2]]))
    assert pair["mean_b"] == pytest.approx(np.mean([b[1], b[2]]))
    assert pair["std_a"] == pytest.approx(np.std([a[1], a[2]], ddof=ledger_report.STD_DDOF))
    assert pair["diff_b_better"] == pytest.approx(pair["mean_a"] - pair["mean_b"])  # rmse: lower is better
    assert pair["diff_b_better"] > 0 and pair["ci_low"] > 0  # the difference and its interval agree


def test_models_without_a_common_seed_get_no_decision(tmp_path):
    def build(outputs):
        for seed in (0, 1):
            _lba_run(outputs, "A", seed, 1.0, SHA_A)
        for seed in (2, 3):
            _lba_run(outputs, "B", seed, 0.2, SHA_B)
    pair = _pair(ledger_compare.compare(_state(tmp_path, build), "A", "B", n_boot=50), "lba", "rmse")
    assert pair["common_seeds"] == [] and pair["mean_a"] is None and pair["diff_b_better"] is None
    assert pair["decision"] == "karar yok: ortak koşu tohumu yok"


# -- S2: derived_from without a catalog average is an ordinary row (its value is still reproduced) ----------

def test_derived_row_without_catalog_average_is_compared_like_a_plain_row(tmp_path):
    import uuid

    def build(outputs):
        for seed in (0, 1, 2):
            _ec_run(outputs, "A", seed, ec_predictions(np.random.default_rng(seed), quality=0.4), SHA_A)
            _ec_run(outputs, "B", seed, ec_predictions(np.random.default_rng(seed + 10), quality=0.6), SHA_B)
    _state(tmp_path, build)
    root = tmp_path / "results"
    fmax_rows = [r for r in ledger.read_records(root, "measurements")
                 if (r["eval_split"], r["subset"], r["metric"]) == ("test", "all", "fmax")]
    ledger.append_records(root, "measurements", [
        dict(r, row_id=str(uuid.uuid4()), ingested_utc=ledger.now_utc(), derived_from="predictions.npz@x")
        for r in fmax_rows])
    result = ledger_compare.compare(ledger_report.load_state(root), "A", "B", n_boot=50)
    pair = _pair(result, "ec", "fmax")
    assert pair["ci_low"] is not None and pair["common_seeds"] == [0, 1, 2]

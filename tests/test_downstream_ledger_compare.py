"""Model comparison with a paired bootstrap over predictions.npz."""

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
    assert pair["decision"] == "indistinguishable" and pair["common_seeds"] == [0, 1, 2]


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
    assert rmse["decision"] == "B better"
    fmax = _pair(result, "ec", "fmax")
    assert fmax["mean_b"] > fmax["mean_a"] and fmax["diff_b_better"] > 0 and fmax["decision"] == "B better"
    reverse = ledger_compare.compare(_state(tmp_path / "r", build), "B", "A", n_boot=300)
    assert _pair(reverse, "lba", "rmse")["decision"] == "B worse"


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
    assert "different setting" in entry["reason"]


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
    assert pair["ci_low"] is None and pair["decision"].startswith("no decision")


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
    assert "pretraining-seed noise was not measured" in markdown
    assert "1. Probe seed" in markdown and "2. Test sample" in markdown and "3. Pretraining run" in markdown
    with_ci = sum(p["ci_low"] is not None for p in result["pairs"])
    assert f"{with_ci} comparisons" in markdown and f"{0.05 * with_ci:.1f}" in markdown
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
    assert pair["decision"] == "no decision: single seed, probe-seed std undefined"


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
    assert pair["decision"] == "no decision: no common run seed"


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


# -- Vectorised multi-target ROC-AUC / PRC-AUC: identical to the per-resample scikit-learn implementation ---------

def _masked_data(rng, n=60, targets=9, ties=True, missing=0.3):
    """Labels with NaN (missing), single-class and all-missing targets, and scores with many ties."""
    labels = (rng.random((n, targets)) < 0.4).astype(float)
    labels[rng.random((n, targets)) < missing] = np.nan
    labels[:, 0] = 0.0  # a single-class target (no positive)
    labels[:, 1] = 1.0  # a single-class target (no negative)
    labels[:, 2] = np.nan  # a target with no labelled sample
    scores = rng.random((n, targets)) + 0.4 * np.nan_to_num(labels) * (rng.random((n, targets)) < 0.6)
    if ties:
        scores = np.round(scores, 1)
    scores[:, 3] = 0.5  # a constant score: one big tie
    return labels, scores


@pytest.mark.parametrize("metric", ["roc_auc", "prc_auc"])
def test_batched_metric_equals_sklearn_per_resample(metric):
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(3)
    labels, scores = _masked_data(rng)
    function = ledger_compare.metric_fn("multilabel_masked", metric, labels, scores)
    assert isinstance(function, ledger_compare._MaskedBinaryMetric)
    draws = rng.integers(0, len(labels), (200, len(labels)))
    batched = ledger_compare._block_values([function], [0], [draws], workers=1)[0]
    assert len(batched) == 200
    # per resample: the reference function (metrics.multitask_*) and, per target, scikit-learn itself
    sklearn_metric = roc_auc_score if metric == "roc_auc" else average_precision_score
    for row, idx in zip(batched, draws):
        per_target = []
        for t in range(labels.shape[1]):
            keep = ~np.isnan(labels[idx, t])
            truth = labels[idx, t][keep].astype(int)
            if truth.sum() in (0, truth.size):
                continue
            per_target.append(sklearn_metric(truth, scores[idx, t][keep]))
        assert row == pytest.approx(np.mean(per_target), abs=1e-12, rel=0)
        assert row == pytest.approx(function(idx), abs=1e-12, rel=0)
    # the whole sample, as used by the reproduction check
    assert function.full() == pytest.approx(function(np.arange(len(labels))), abs=1e-12, rel=0)


def test_batched_metric_is_undefined_where_the_reference_raises():
    labels = np.array([[1.0, np.nan], [1.0, np.nan], [0.0, 0.0], [1.0, 1.0]])
    scores = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
    function = ledger_compare.metric_fn("multilabel_masked", "roc_auc", labels, scores)
    draws = np.array([[0, 1, 0, 1], [0, 2, 3, 1], [3, 3, 3, 3]])  # the first and last draw have one class only
    values = ledger_compare._block_values([function], [0], [draws], workers=1)[0]
    assert np.isnan(values[0]) and np.isnan(values[2]) and np.isfinite(values[1])
    for row in (0, 2):
        with pytest.raises(ValueError, match="single-class"):
            function(draws[row])
    assert values[1] == pytest.approx(function(draws[1]), abs=1e-12, rel=0)
    with pytest.raises(ValueError, match="single-class"):
        ledger_compare.metric_fn("multilabel_masked", "roc_auc", labels[:2], scores[:2]).full()
    with pytest.raises(ValueError, match="labels must be 0, 1 or NaN"):
        ledger_compare.metric_fn("multilabel_masked", "roc_auc", labels + 2, scores)


def _reference_paired_bootstrap(rows_a, rows_b, *, sign, n_boot, rng_seed):
    """The comparison as it was before vectorisation: one resample at a time, scikit-learn on every target."""
    from downstream.common import metrics as M
    seeds = sorted({r["run_seed"] for r in rows_a} & {r["run_seed"] for r in rows_b})
    by_a, by_b = ({r["run_seed"]: r for r in rows} for rows in (rows_a, rows_b))

    def load(row):
        with np.load(row["source_metrics_path"].with_name("predictions.npz")) as data:
            labels, scores, metric = data["test_labels"], data["test_scores"], row["metric"]
        function = M.multitask_roc_auc if metric == "roc_auc" else M.multitask_prc_auc
        return lambda idx: function(labels[idx], scores[idx])[metric], len(labels)
    functions = [(load(by_a[s])[0], load(by_b[s])[0]) for s in seeds]
    n = load(by_a[seeds[0]])[1]
    rng = np.random.default_rng(rng_seed)
    differences, failed = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            values = [np.mean([fb(idx)]) - np.mean([fa(idx)]) for fa, fb in functions]
        except ValueError:
            failed += 1
            continue
        if not np.all(np.isfinite(values)):
            failed += 1
            continue
        differences.append(sign * float(np.mean(values)))
    return (*np.percentile(differences, [2.5, 97.5]), failed)


def _multitarget_rows(tmp_path, metric, n, targets, seeds=(0, 1, 2)):
    """Ledger-like rows of two models whose predictions.npz hold masked multi-target test predictions."""
    from downstream.common import metrics as M
    function = M.multitask_roc_auc if metric == "roc_auc" else M.multitask_prc_auc
    labels, _ = _masked_data(np.random.default_rng(11), n=n, targets=targets)
    rows = {"a": [], "b": []}
    for model, quality in (("a", 0.2), ("b", 0.5)):
        for seed in seeds:
            rng = np.random.default_rng(1000 * seed + (model == "b"))
            scores = np.round(rng.random(labels.shape) + quality * np.nan_to_num(labels), 1)
            folder = tmp_path / model / f"seed{seed}"
            folder.mkdir(parents=True)
            np.savez(folder / "predictions.npz", test_chains=np.array([f"m{i}" for i in range(n)]),
                     test_scores=scores, test_labels=labels)
            rows[model].append({"run_seed": seed, "value": function(labels, scores)[metric], "metric": metric,
                                "kind": "multilabel_masked", "eval_split": "test", "subset": "all",
                                "derived_from": None, "source_metrics_path": folder / "metrics.json"})
    return rows["a"], rows["b"]


@pytest.mark.parametrize("metric", ["roc_auc", "prc_auc"])
@pytest.mark.parametrize("n, targets, n_boot", [(80, 9, 150), (10, 4, 300)])  # 10 samples: many resamples are undefined
def test_vectorised_bootstrap_equals_the_reference_implementation(tmp_path, monkeypatch, metric, n, targets, n_boot):
    rows_a, rows_b = _multitarget_rows(tmp_path, metric, n, targets)
    for row in rows_a + rows_b:  # a tiny test set cannot reproduce the full-sample check value of the ledger: skip it
        row["value"] = ledger_compare.metric_fn("multilabel_masked", metric, *_arrays(row)).full()
    low, high, failed = _reference_paired_bootstrap(rows_a, rows_b, sign=1.0, n_boot=n_boot, rng_seed=5)
    if n == 10:
        assert failed > 0  # the undefined-resample path is exercised
    runs = []
    for workers, block, min_work in ((1, 10 ** 9, 10 ** 12), (3, 10 ** 9, 0), (3, 3 * n, 0), (1, n, 10 ** 12)):
        monkeypatch.setattr(ledger_compare, "_RESAMPLE_BLOCK", block)
        monkeypatch.setattr(ledger_compare, "_PARALLEL_MIN_WORK", min_work)
        runs.append(ledger_compare.paired_bootstrap(rows_a, rows_b, {}, sign=1.0, n_boot=n_boot, rng_seed=5,
                                                    workers=workers))
    for result in runs:  # processes and block size change nothing
        assert result["ci_low"] == pytest.approx(low, abs=1e-12, rel=0)
        assert result["ci_high"] == pytest.approx(high, abs=1e-12, rel=0)
        assert result["n_boot_failed"] == failed and result["ci_reason"] is None


def _arrays(row):
    with np.load(row["source_metrics_path"].with_name("predictions.npz")) as data:
        return data["test_labels"], data["test_scores"]


# -- A run restricted with --tasks never replaces the all-task files of the same pair ------------------------

def test_restricted_run_does_not_overwrite_the_all_task_files(tmp_path):
    def build(outputs):
        for seed in (0, 1):
            predictions = ec_predictions(np.random.default_rng(seed))
            _ec_run(outputs, "A", seed, predictions, SHA_A)
            _ec_run(outputs, "B", seed, predictions, SHA_B)
            _lba_run(outputs, "A", seed, 1.0, SHA_A)
            _lba_run(outputs, "B", seed, 0.5, SHA_B)
    state = _state(tmp_path, build)
    everything = ledger_compare.write_compare(state["root"], "A", "B", n_boot=20, state=state)
    assert [p.name for p in everything] == ["A__vs__B.md", "A__vs__B.csv"]
    before = [p.read_bytes() for p in everything]
    restricted = ledger_compare.write_compare(state["root"], "A", "B", tasks=["lba"], n_boot=20, state=state)
    assert [p.name for p in restricted] == ["A__vs__B__tasks-lba.md", "A__vs__B__tasks-lba.csv"]
    assert [p.read_bytes() for p in everything] == before  # the all-task files are untouched
    assert "ec" not in restricted[1].read_text().split("\n", 1)[1].split(",")[0]
    # the suffix depends on the task set, not on its order, and stays short for long lists
    assert ledger_compare.compare_stem("A", "B", ["lba", "ec"]) == ledger_compare.compare_stem("A", "B", ["ec", "lba", "ec"])
    assert ledger_compare.compare_stem("A", "B", ["ec", "lba"]) == "A__vs__B__tasks-ec-lba"
    assert ledger_compare.compare_stem("A", "B", None) == ledger_compare.compare_stem("A", "B", []) == "A__vs__B"
    assert len(ledger_compare.compare_stem("A", "B", [f"task_{i:03d}" for i in range(40)])) < 60


def test_worker_count_is_validated_and_bounded(monkeypatch):
    import os
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(4)))
    assert ledger_compare.default_workers() == 4
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(384)))
    assert ledger_compare.default_workers() == ledger_compare.DEFAULT_WORKERS
    with pytest.raises(SystemExit):
        ledger_compare.main(["--a", "A", "--b", "B", "--workers", "385"])
    with pytest.raises(SystemExit):
        ledger_compare.main(["--a", "A", "--b", "B", "--workers", "0"])

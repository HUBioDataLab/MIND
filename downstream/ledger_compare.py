"""Compare two models cell by cell, with a paired bootstrap over the saved test predictions.

    python -m downstream.ledger_compare --a <model_a> --b <model_b> [--tasks ec lba] [--bootstrap 1000] [--workers N]

Writes ``$MIND_RESULTS_ROOT/reports/compare/<A>__vs__<B>.md|.csv`` (``<A>__vs__<B>__tasks-<tasks>`` when
``--tasks`` restricts the run, so that it never replaces the all-task files). Only cells with the same comparable
setting (see ``downstream.ledger_report``: protocol, pooling, data revision and data fingerprint, settings
hash) are compared; every other cell is listed as "only in A" / "only in B" with the reason.

Three noise sources are reported separately:

1. probe seed: std over run seeds of each model (``ledger_report.STD_DDOF``);
2. test sample: paired bootstrap 95 % interval of the difference. Each replicate draws one set of test-sample
   indices, applies it to A and B and to every common run seed, and averages the signed difference over seeds.
   Chains are aligned by id; a different test-chain set or labels is an error, and each seed's predictions must
   reproduce the ledger value before they are used;
3. pretraining run: not measured (one pretraining seed per model); the report says so in a warning box.

Sign: ``diff_b_better`` = B - A for higher-is-better metrics and A - B for lower-is-better ones, so + always
means B is better. Decision: "B better" when the interval's lower bound is > 0 and |diff| > 2 x the larger
seed std; "B worse" symmetrically; otherwise "indistinguishable". Cells without an interval (subsets whose
mask is not in predictions.npz, validation rows, metrics without a bootstrap implementation) get "no decision".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from downstream import ledger, ledger_report
from downstream.common import metrics as M

REPRODUCE_TOLERANCE = 1e-4
DEFAULT_WORKERS = 16  # upper bound of the default; --workers can ask for more (never more than the cores available)
_RESAMPLE_BLOCK = 2_000_000  # resamples x samples handled at once (bounds the memory of one block)
_PARALLEL_MIN_WORK = 20_000_000  # targets x samples x resamples below which a process pool does not pay off
_SECTION_KEY = ("protocol_family", "protocol_id", "pooling", "data_revision", "split_tag", "representation",
                "protocol_settings_sha256", "data_fingerprint")


# ---------------------------------------------------------------------------------------------------------
# Resampled metric functions: prepare(labels, scores) -> f(indices) -> value (ValueError when undefined)
# ---------------------------------------------------------------------------------------------------------

def _fmax_fn(labels: np.ndarray, scores: np.ndarray) -> Callable:
    """Protein-centric Fmax as in ``metrics.protein_fmax``, from per-protein counts so resampling is cheap."""
    labels, scores = M._check(labels, scores)
    hits = np.stack([((scores >= t) & labels).sum(axis=1) for t in M.THRESHOLDS]).astype(np.float64)
    predicted = np.stack([(scores >= t).sum(axis=1) for t in M.THRESHOLDS]).astype(np.float64)
    true = labels.sum(axis=1).astype(np.float64)

    def fmax(idx):
        h, p = hits[:, idx], predicted[:, idx]
        covered = p > 0
        n_covered = covered.sum(axis=1)
        precision = np.where(n_covered > 0, (np.where(covered, h / np.maximum(p, 1), 0)).sum(axis=1)
                             / np.maximum(n_covered, 1), 0.0)
        recall = (h / true[idx]).mean(axis=1)
        valid = (n_covered > 0) & (precision + recall > 0)
        f1 = np.where(valid, 2 * precision * recall / np.where(valid, precision + recall, 1), 0.0)
        return float(f1.max())
    return fmax


# ---------------------------------------------------------------------------------------------------------
# Batched ROC-AUC / average precision of masked multi-target labels for many bootstrap resamples at once
# ---------------------------------------------------------------------------------------------------------

class _MaskedBinaryMetric:
    """Mean over targets of ROC-AUC or average precision, as ``metrics.multitask_roc_auc`` / ``multitask_prc_auc``.

    ``__call__(idx)`` is the reference implementation (one resample, scikit-learn). ``block_values`` evaluates many
    resamples at once: a resample of the test set is a vector of integer weights (how often each sample was drawn),
    and the AUC or average precision of weighted samples equals the value of the repeated samples. The scores of
    each target are sorted once; the weights are then accumulated in sorted order, with tied scores merged into one
    group (a tie counts half in ROC-AUC and forms one threshold in average precision), exactly as scikit-learn
    treats them. A target with no positive or no negative sample in a resample is skipped for that resample; a
    resample in which every target is skipped is undefined (NaN), where the reference raises ``ValueError``.
    """

    def __init__(self, metric: str, labels: np.ndarray, scores: np.ndarray):
        self.metric = metric
        self.labels, self.scores = M._check_masked(labels, scores, label_name="labels", score_name="scores")
        observed = ~np.isnan(self.labels)
        if not np.isin(self.labels[observed], (0, 1)).all():
            raise ValueError("labels must be 0, 1 or NaN (missing)")
        self.n_samples, self.n_targets = self.labels.shape
        self.targets = []  # per target: (rows in ascending score order, 0/1 labels there, group starts or None)
        for target in range(self.n_targets):
            rows = np.flatnonzero(observed[:, target])
            order = rows[np.argsort(self.scores[rows, target], kind="stable")]
            sorted_scores = self.scores[order, target]
            starts = np.flatnonzero(np.concatenate([[True], sorted_scores[1:] != sorted_scores[:-1]]))
            self.targets.append((order, self.labels[order, target], None if len(starts) == len(order) else starts))

    def __call__(self, idx):
        function = M.multitask_roc_auc if self.metric == "roc_auc" else M.multitask_prc_auc
        return function(self.labels[idx], self.scores[idx])[self.metric]

    def _target_values(self, counts: np.ndarray, target: int) -> np.ndarray:
        """Metric of one target for every resample; ``counts`` is (samples, resamples), NaN where one class is missing."""
        order, truth, starts = self.targets[target]
        if len(order) == 0:
            return np.full(counts.shape[1], np.nan)
        weight = counts[order]
        positive = weight * truth[:, None]
        negative = weight - positive
        if starts is not None:
            positive, negative = np.add.reduceat(positive, starts, axis=0), np.add.reduceat(negative, starts, axis=0)
        n_pos, n_neg = positive.sum(axis=0), negative.sum(axis=0)
        defined = (n_pos > 0) & (n_neg > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.metric == "roc_auc":
                below = np.cumsum(negative, axis=0) - negative  # negatives with a lower score than the group
                value = (positive * (below + 0.5 * negative)).sum(axis=0) / (n_pos * n_neg)
            else:
                true_pos = n_pos - np.cumsum(positive, axis=0) + positive  # positives scored >= the group threshold
                predicted = true_pos + (n_neg - np.cumsum(negative, axis=0) + negative)
                precision = np.divide(true_pos, predicted, out=np.zeros_like(true_pos), where=positive > 0)
                value = (positive * precision).sum(axis=0) / n_pos
        return np.where(defined, value, np.nan)

    def full(self) -> float:
        """The metric on the whole sample (every weight 1); ``ValueError`` when no target can be scored."""
        total, defined = self.partial_sum(np.ones((self.n_samples, 1)), 0, self.n_targets)
        if defined[0] == 0:
            raise ValueError("every task is single-class (needs labelled positives and negatives); nothing to evaluate")
        return float(total[0] / defined[0])

    def partial_sum(self, counts: np.ndarray, lo: int, hi: int):
        """(sum over targets lo..hi of the defined values, number of defined targets), per resample."""
        total, defined = np.zeros(counts.shape[1]), np.zeros(counts.shape[1])
        for target in range(lo, hi):
            value = self._target_values(counts, target)
            ok = ~np.isnan(value)
            total += np.where(ok, value, 0.0)
            defined += ok
        return total, defined


def _weights(draws: np.ndarray) -> np.ndarray:
    """(samples, resamples) float64 table: how often each sample was drawn by each resample (draws: resamples x n)."""
    n_boot, n = draws.shape
    flat = (draws * n_boot + np.arange(n_boot)[:, None]).ravel()
    return np.bincount(flat, minlength=n * n_boot).reshape(n, n_boot).astype(np.float64)


_SHARED: dict = {}  # read by forked pool workers; set before the pool is created


def _run_unit(unit):
    function, part, lo, hi = unit
    return _SHARED["functions"][function].partial_sum(_SHARED["counts"][part], lo, hi)


def _block_values(functions: list, parts: list, draws: list, workers: int) -> list:
    """Values of every function on every resample of one block: list of (resamples,) arrays, NaN where undefined.

    ``functions[i]`` is evaluated on the draws of ``parts[i]``.
    """
    values: list = [None] * len(functions)
    batched = [i for i, f in enumerate(functions) if isinstance(f, _MaskedBinaryMetric)]
    for i in set(range(len(functions))) - set(batched):
        out = np.full(len(draws[parts[i]]), np.nan)
        for r, idx in enumerate(draws[parts[i]]):
            try:
                out[r] = functions[i](idx)
            except ValueError:
                pass  # undefined on this resample (the resample counts as failed)
        values[i] = out
    if batched:
        counts = {p: _weights(draws[p]) for p in {parts[i] for i in batched}}
        work = sum(functions[i].n_targets * counts[parts[i]].size for i in batched)
        units = []  # (function, part, first target, end target); about 4 units per worker keep the pool balanced
        step = max(1, -(-sum(functions[i].n_targets for i in batched) // (4 * max(workers, 1))))
        for i in batched:
            units += [(i, parts[i], lo, min(lo + step, functions[i].n_targets))
                      for lo in range(0, functions[i].n_targets, step)]
        if workers > 1 and work >= _PARALLEL_MIN_WORK:
            _SHARED.update(functions=functions, counts=counts)
            try:
                with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as pool:
                    results = list(pool.map(_run_unit, units))
            finally:
                _SHARED.clear()
        else:
            _SHARED.update(functions=functions, counts=counts)
            try:
                results = [_run_unit(unit) for unit in units]
            finally:
                _SHARED.clear()
        total = {i: np.zeros(len(draws[parts[i]])) for i in batched}
        defined = {i: np.zeros(len(draws[parts[i]])) for i in batched}
        for unit, (part_sum, part_defined) in zip(units, results):
            total[unit[0]] += part_sum
            defined[unit[0]] += part_defined
        for i in batched:
            with np.errstate(divide="ignore", invalid="ignore"):
                values[i] = np.where(defined[i] > 0, total[i] / defined[i], np.nan)
    return values


def metric_fn(kind: str, metric: str, labels: np.ndarray, scores: np.ndarray) -> Optional[Callable]:
    """Resampling function for one (kind, metric), or None when no bootstrap is implemented for it."""
    if kind == "multilabel" and metric == "fmax":
        return _fmax_fn(labels, scores)
    if kind == "multilabel" and metric == "top1_accuracy":
        checked_labels, checked_scores = M._check(labels, scores)
        correct = checked_labels[np.arange(len(checked_labels)), checked_scores.argmax(axis=1)].astype(np.float64)
        return lambda idx: float(correct[idx].mean())
    if kind == "multiclass" and metric in ("accuracy", "balanced_accuracy"):
        targets, checked_scores = M._check_multiclass(labels, scores)
        correct = (checked_scores.argmax(axis=1) == targets).astype(np.float64)
        if metric == "accuracy":
            return lambda idx: float(correct[idx].mean())

        def balanced(idx):  # mean recall over the classes present in the resample, as metrics.balanced_accuracy
            counts = np.bincount(targets[idx])
            hits = np.bincount(targets[idx], weights=correct[idx], minlength=len(counts))
            present = counts > 0
            return float(np.mean(hits[present] / counts[present]))
        return balanced
    if kind == "regression" and metric in ("rmse", "mae", "pearson_r", "spearman_rho"):
        function = {"rmse": M.rmse, "mae": M.mae, "pearson_r": M.pearson_r, "spearman_rho": M.spearman_rho}[metric]
        return lambda idx: function(labels.reshape(-1)[idx], scores.reshape(-1)[idx])
    if kind == "multilabel_masked" and metric in ("roc_auc", "prc_auc"):
        return _MaskedBinaryMetric(metric, labels, scores)
    if kind == "multitarget_regression" and metric in ("rmse", "mae"):
        return lambda idx: M.multitarget_regression_errors(labels[idx], scores[idx])[metric]
    return None


def _split_arrays(path: Path, eval_split: str):
    """(chains, scores, labels) of one test split from predictions.npz (multi-class extras are prefixed)."""
    with np.load(path) as data:
        if f"{eval_split}_chains" in data:
            prefix = eval_split
        elif "test_chains" in data:
            prefix = "test"  # the first test split of the task (the integrity check below confirms the pairing)
        else:
            raise ValueError(f"{path} has no test predictions")
        return (data[f"{prefix}_chains"].astype(str), data[f"{prefix}_scores"], data[f"{prefix}_labels"])


def _aligned(path: Path, eval_split: str, order: Optional[np.ndarray]):
    chains, scores, labels = _split_arrays(path, eval_split)
    if order is None:
        return chains, scores, labels
    position = {chain: i for i, chain in enumerate(chains)}
    if len(position) != len(chains) or set(position) != set(order.tolist()):
        raise ValueError(f"{path}: test chains differ from the other model's ({len(chains)} vs {len(order)} ids, "
                         f"{len(set(position) ^ set(order.tolist()))} not shared); paired comparison is impossible")
    index = np.array([position[chain] for chain in order])
    return order, scores[index], labels[index]


def _ci_reason(row: dict, spec: dict) -> Optional[str]:
    if row["eval_split"] == "val":
        return "validation labels are not in predictions.npz"
    if row["subset"] != "all":
        return "subset mask is not in predictions.npz"
    return None


def _parts(row: dict, spec: dict) -> list:
    """Test splits behind a row: the catalog's derived average (e.g. Fold test_avg) or the row's own split.

    Only a catalog average for this split and metric expands; any other ``derived_from`` row is resampled on its
    own split, and the reproduction check in ``paired_bootstrap`` still compares it with the predictions.
    """
    if row["derived_from"]:
        for derived in spec.get("derived_averages") or []:
            if derived["eval_split"] == row["eval_split"] and row["metric"] in derived["metrics"]:
                return list(derived["of"])
    return [row["eval_split"]]


def paired_bootstrap(rows_a: list, rows_b: list, spec: dict, *, sign: float, n_boot: int,
                     rng_seed: int, workers: int = 1) -> dict:
    """Interval of the signed (B better = +) difference, paired over test samples and common seeds.

    The resample indices are drawn from ``default_rng(rng_seed)`` one resample after the other (one draw per test
    part), so the interval depends neither on ``workers`` (processes that share the per-target metric work) nor on
    how many resamples are evaluated at once.
    """
    by_seed_a = {r["run_seed"]: r for r in rows_a}
    by_seed_b = {r["run_seed"]: r for r in rows_b}
    seeds = sorted(set(by_seed_a) & set(by_seed_b))
    first = rows_a[0]
    reason = _ci_reason(first, spec)
    if reason is None and not seeds:
        reason = "no common run seed"
    if reason is None and any(by_seed_a[s]["value"] is None or by_seed_b[s]["value"] is None for s in seeds):
        reason = "value undefined for some seeds"
    if reason is not None:
        return {"ci_low": None, "ci_high": None, "n_boot_failed": None, "common_seeds": seeds, "ci_reason": reason}
    parts = _parts(first, spec)
    orders = {}
    functions = []  # per seed: (f_a, f_b) lists over parts
    for seed in seeds:
        per_model = []
        for row in (by_seed_a[seed], by_seed_b[seed]):
            path = Path(row["source_metrics_path"]).with_name("predictions.npz")
            if not path.is_file():
                return {"ci_low": None, "ci_high": None, "n_boot_failed": None, "common_seeds": seeds,
                        "ci_reason": f"predictions.npz missing: {path}"}
            fns = []
            for part in parts:
                chains, scores, labels = _aligned(path, part, orders.get(part))
                if part not in orders:
                    orders[part] = chains
                    orders[part, "labels"] = labels
                elif not np.array_equal(labels, orders[part, "labels"], equal_nan=True):
                    raise ValueError(f"{path}: labels of {part} differ from the other run's for the same chains")
                function = metric_fn(row["kind"], row["metric"], labels, scores)
                if function is None:
                    return {"ci_low": None, "ci_high": None, "n_boot_failed": None, "common_seeds": seeds,
                            "ci_reason": f"no bootstrap defined for {row['kind']}/{row['metric']}"}
                fns.append(function)
            full = float(np.mean([f.full() if isinstance(f, _MaskedBinaryMetric) else f(np.arange(len(orders[p])))
                                  for f, p in zip(fns, parts)]))
            if abs(full - row["value"]) > REPRODUCE_TOLERANCE:
                raise ValueError(f"{path}: predictions.npz does not reproduce the ledger value of "
                                 f"{row['metric']} {row['eval_split']} (seed {seed}): {full:.6f} vs {row['value']:.6f}")
            per_model.append(fns)
        functions.append(per_model)
    rng = np.random.default_rng(rng_seed)
    sizes = [len(orders[p]) for p in parts]
    flat = [(f, k) for per_model in functions for fns in per_model for k, f in enumerate(fns)]  # seed, model, part
    block = max(1, min(n_boot, _RESAMPLE_BLOCK // max(sizes)))
    differences, failed = [], 0
    for first in range(0, n_boot, block):
        n_block = min(block, n_boot - first)
        draws = [np.empty((n_block, n), dtype=np.int64) for n in sizes]
        for r in range(n_block):  # one draw per part and resample, in order: the stream of a resample-by-resample loop
            for k, n in enumerate(sizes):
                draws[k][r] = rng.integers(0, n, n)
        values = iter(_block_values([f for f, _ in flat], [k for _, k in flat], draws, workers))
        per_seed = []
        for per_model in functions:  # per_model = [functions of A, functions of B], each over the parts
            mean_a, mean_b = (np.mean([next(values) for _ in fns], axis=0) for fns in per_model)
            per_seed.append(mean_b - mean_a)
        difference = np.mean(per_seed, axis=0)  # over the common seeds
        finite = np.isfinite(difference)
        failed += int((~finite).sum())
        differences.extend((sign * difference[finite]).tolist())
    if not differences:
        return {"ci_low": None, "ci_high": None, "n_boot_failed": failed, "common_seeds": seeds,
                "ci_reason": "metric undefined in every bootstrap sample"}
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"ci_low": float(low), "ci_high": float(high), "n_boot_failed": failed, "common_seeds": seeds,
            "ci_reason": None}


# ---------------------------------------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------------------------------------

def _cells(state: dict, model: str, tasks: Optional[list]) -> dict:
    cells = defaultdict(list)
    for task in tasks or sorted({r["task"] for r in state["rows"]}):
        for section in ledger_report.sections(state, task):
            for row in section["rows"]:
                if row["model_key"] == model:
                    cells[(task,) + tuple(section[f] for f in _SECTION_KEY)
                          + (row["eval_split"], row["subset"], row["metric"])].append(row)
    return cells


def _common_seed_stats(rows: list, seeds: set) -> dict:
    common = [r for r in rows if r["run_seed"] in seeds]
    return ledger_report.cell_stats(common) if common else {"mean": None, "std": None, "n_seeds": 0}


def _decision(diff, low, high, std_a, std_b, reason) -> str:
    if diff is None:
        return "no decision: value undefined"
    if low is None:
        return f"no decision: {reason}"
    if std_a is None or std_b is None:
        return "no decision: single seed, probe-seed std undefined"
    margin = 2 * max(std_a, std_b)
    if low > 0 and abs(diff) > margin:
        return "B better"
    if high < 0 and abs(diff) > margin:
        return "B worse"
    return "indistinguishable"


def default_workers() -> int:
    """Processes of a run without ``--workers``: ``DEFAULT_WORKERS``, never more than the cores this process may use."""
    return max(1, min(DEFAULT_WORKERS, len(os.sched_getaffinity(0))))


def compare(state: dict, a: str, b: str, *, tasks: Optional[list] = None, n_boot: int = 1000,
            rng_seed: int = 0, workers: int = 1) -> dict:
    if a == b:
        raise ValueError("A and B must be different models")
    cells_a, cells_b = _cells(state, a, tasks), _cells(state, b, tasks)
    if not cells_a and not cells_b:
        raise ValueError(f"no valid ledger rows for {a!r} or {b!r}" + (f" in tasks {tasks}" if tasks else ""))
    pairs = []
    for key in sorted(set(cells_a) & set(cells_b), key=str):
        rows_a, rows_b = cells_a[key], cells_b[key]
        task, metric = key[0], key[-1]
        info = ledger.metric_info(state["catalog"], task, metric)
        # mean, std and difference over the seeds both models have: the seeds of the paired interval
        common = {r["run_seed"] for r in rows_a} & {r["run_seed"] for r in rows_b}
        stats_a, stats_b = _common_seed_stats(rows_a, common), _common_seed_stats(rows_b, common)
        if rows_a[0]["unit"] != rows_b[0]["unit"]:
            raise ValueError(f"{key}: units differ ({rows_a[0]['unit']} vs {rows_b[0]['unit']})")
        if info["direction"] is None:
            continue  # metrics outside the catalog have no direction; they stay in the task tables only
        sign = 1.0 if info["direction"] == "higher" else -1.0
        diff = (None if stats_a["mean"] is None or stats_b["mean"] is None
                else sign * (stats_b["mean"] - stats_a["mean"]))
        spec = ledger.task_spec(state["catalog"], task)
        boot = paired_bootstrap(rows_a, rows_b, spec, sign=sign, n_boot=n_boot, rng_seed=rng_seed, workers=workers)
        pairs.append({"task": task, **dict(zip(_SECTION_KEY, key[1:9])), "eval_split": key[9], "subset": key[10],
                      "metric": metric, "direction": info["direction"], "unit": info["unit"],
                      "is_primary": rows_a[0]["is_primary"], "mean_a": stats_a["mean"], "mean_b": stats_b["mean"],
                      "std_a": stats_a["std"], "std_b": stats_b["std"], "n_seeds_a": stats_a["n_seeds"],
                      "n_seeds_b": stats_b["n_seeds"], "diff_b_better": diff, **boot, "n_boot": n_boot,
                      "rng_seed": rng_seed,
                      "decision": (_decision(diff, boot["ci_low"], boot["ci_high"], stats_a["std"], stats_b["std"],
                                             boot["ci_reason"]) if common else f"no decision: {boot['ci_reason']}")})

    def only(mine, other, name):
        loose = {(k[0], k[2], k[9], k[10], k[11]) for k in other}
        return [{"task": k[0], "protocol_id": k[2], "pooling": k[3], "data_revision": k[4], "eval_split": k[9],
                 "subset": k[10], "metric": k[11],
                 "reason": ("different setting (pooling / data revision / data fingerprint / protocol settings differ)"
                            if (k[0], k[2], k[9], k[10], k[11]) in loose else f"not in {name}")}
                for k in sorted(set(mine) - set(other), key=str)]
    return {"a": a, "b": b, "pairs": pairs, "only_a": only(cells_a, cells_b, "B"), "only_b": only(cells_b, cells_a, "A"),
            "n_boot": n_boot, "rng_seed": rng_seed}


def _num(value, digits=3) -> str:
    return "undefined" if value is None else f"{value:.{digits}f}"


def compare_markdown(state: dict, result: dict) -> str:
    a, b = result["a"], result["b"]
    lines = ledger_report.header_lines(state, f"Comparison: A = {a}, B = {b}")
    with_ci = [p for p in result["pairs"] if p["ci_low"] is not None]
    counts = defaultdict(int)
    for pair in result["pairs"]:
        counts[pair["decision"].split(":")[0]] += 1
    lines += [f"> WARNING: {ledger_report.PRETRAIN_SEED_WARNING} This report cannot say \"B improved X\"; at most it "
              "says \"between these two runs there is a difference beyond probe and test noise\".", "",
              "Difference = B - A, signed by direction: + always means B is better (the sign is flipped for "
              "lower-is-better metrics such as RMSE/MAE).", "", "Three noise sources, separately:", "",
              f"1. Probe seed (random initialization of the head): columns std_A and std_B (across seeds, std: ddof="
              f"{ledger_report.STD_DDOF} (sample std); undefined for a single seed).",
              f"2. Test sample: paired bootstrap 95 % confidence interval ({result['n_boot']} resamples, RNG seed "
              f"{result['rng_seed']}; the same sample indices are applied to both models and to all common seeds).",
              "3. Pretraining run (same settings, another pretraining seed): NOT MEASURED (warning above).", "",
              "Decision: \"B better\" = interval lower bound > 0 and |diff| > 2 x max(std_A, std_B); \"B worse\" is "
              "symmetric; every other case is \"indistinguishable\". Without an interval: \"no decision\" and the "
              "reason.", "",
              f"Multiple comparisons: {len(with_ci)} comparisons were made at the 95 % level; even with no real "
              f"difference, about {0.05 * len(with_ci):.1f} of them can have an interval that excludes zero by chance "
              "(expected false alarms).", "",
              "Decision counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
              "| task | protocol | pooling | data | eval_split | subset | metric | unit | A | B | diff (+ = B better) | "
              "std_A | std_B | 95 % CI | common seeds | decision |", "|" + "---|" * 16]
    for p in sorted(result["pairs"], key=lambda p: (p["task"], not p["is_primary"], p["eval_split"], p["subset"],
                                                    p["metric"])):
        interval = "-" if p["ci_low"] is None else f"[{p['ci_low']:+.3f}, {p['ci_high']:+.3f}]"
        if p["n_boot_failed"]:
            interval += f" ({p['n_boot_failed']} samples undefined)"
        lines.append(f"| {p['task']} | {p['protocol_id']} | {p['pooling']} | {p['data_revision']} | {p['eval_split']} | "
                     f"{p['subset']} | {p['metric']} | {p['unit']} | {_num(p['mean_a'])} | {_num(p['mean_b'])} | "
                     f"{'undefined' if p['diff_b_better'] is None else format(p['diff_b_better'], '+.3f')} | "
                     f"{_num(p['std_a'])} | {_num(p['std_b'])} | {interval} | {len(p['common_seeds'])} | {p['decision']} |")
    for name, entries in (("A", result["only_a"]), ("B", result["only_b"])):
        lines += ["", f"## Cells only in {name} ({len(entries)})", ""]
        lines += [f"- {e['task']} / {e['protocol_id']} / {e['pooling']} / {e['data_revision']} / {e['eval_split']} / "
                  f"{e['subset']} / {e['metric']}: {e['reason']}" for e in entries] or ["- none"]
    return "\n".join(lines) + "\n"


def compare_stem(a: str, b: str, tasks: Optional[list] = None) -> str:
    """File name (without extension) of a comparison: ``<A>__vs__<B>`` for all tasks of the ledger, with a
    ``__tasks-<task>-<task>`` suffix when the run is restricted by ``--tasks``, so that a restricted run never
    replaces the all-task files of the same pair."""
    stem = f"{a}__vs__{b}"
    if not tasks:
        return stem
    bad = [t for t in tasks if not t or any(c in t for c in "/\\") or t.startswith(".")]
    if bad:
        raise ValueError(f"invalid task name(s) for a file name: {bad}")
    names = "-".join(sorted(set(tasks)))
    if len(names) > 80:  # keep the file name short for long task lists; the full list is in the report
        names = f"{len(set(tasks))}tasks-{hashlib.sha256(names.encode()).hexdigest()[:8]}"
    return f"{stem}__tasks-{names}"


def write_compare(root: Optional[Path], a: str, b: str, *, tasks: Optional[list] = None, n_boot: int = 1000,
                  rng_seed: int = 0, state: Optional[dict] = None, workers: int = 1) -> list:
    root = ledger.results_root(root)
    state = state or ledger_report.load_state(root)
    result = compare(state, a, b, tasks=tasks, n_boot=n_boot, rng_seed=rng_seed, workers=workers)
    written: list = []
    directory = root / "reports" / "compare"
    stem = compare_stem(a, b, tasks)
    ledger_report._write(directory / f"{stem}.md", compare_markdown(state, result), written)
    rows = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in p.items()} for p in result["pairs"]]
    ledger_report._write(directory / f"{stem}.csv", ledger_report._csv_text(rows), written)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None,
                        help="processes that share the per-target metric work of one comparison (results agree to "
                             f"floating-point rounding, about 1e-16, whatever the number); default: min({DEFAULT_WORKERS}, usable cores); at most the usable cores")
    parser.add_argument("--results-root", type=Path, default=None, help=f"default: ${ledger.ENV_VAR}")
    args = parser.parse_args(argv)
    if args.bootstrap < 1:
        parser.error("--bootstrap must be at least 1")
    usable = len(os.sched_getaffinity(0))
    if args.workers is not None and not 1 <= args.workers <= usable:
        parser.error(f"--workers must be between 1 and the {usable} usable cores")
    workers = default_workers() if args.workers is None else args.workers
    print(f"ds-compare: {workers} worker process(es) of {usable} usable cores", flush=True)
    written = write_compare(args.results_root, args.a, args.b, tasks=args.tasks, n_boot=args.bootstrap,
                            rng_seed=args.rng_seed, workers=workers)
    print(json.dumps([str(p) for p in written], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

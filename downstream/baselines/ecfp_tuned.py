"""Tuned scikit-learn baseline on the ECFP4 fingerprints (model id ``ecfp4_tuned``, CPU only).

    python -m downstream.baselines.ecfp_tuned --task mn_bbbp [--feature bits|counts] [--n-bits 2048]
        [--threads 8] [--outputs-root DIR] [--split-tag seed0]

The classical fingerprint baseline: L2-regularized logistic regression (classification) or ridge regression
(regression), one model per target, on the same fingerprints (the ``ecfp4`` embedding files written by
``downstream.baselines.ecfp``) and the same stored scaffold split as the probes. The only hyper-parameter, ``C`` or
``alpha``, is chosen from a small grid on the VALIDATION split by the task's primary metric; the test split is
scored once with the chosen value and never enters the selection. The fit is deterministic, so exactly one seed (0)
is written and the ledger records ``n_seeds = 1``. A task with one split per seed (NPASS) is fitted once per split
tag (``--split-tag seed<k>`` required): the result of tag ``seed<k>`` is the run of seed ``k``, so the ledger cell has
one seed per tag and its spread is the split-to-split variance.

* classification: ``LogisticRegression(solver="liblinear", max_iter=2000)`` per target on its labelled rows, grid
  ``C in (0.01, 0.1, 1.0)`` (``(0.05, 0.5)`` when the train split has 30,000 molecules or more); selected on the
  validation mean ROC-AUC. A target with a single class in train raises (none of the 14 tasks has one).
* regression: ``Ridge(alpha)`` per target on its labelled rows, targets in original units (no standardization), grid
  ``alpha in (0.1, 1, 10, 100, 1000)``; selected on the validation primary error (RMSE or MAE).
* no feature scaling, no class weights, no subsampling of the train split.

Output layout and metrics are those of ``downstream.probe`` (protocol ``sklearn_tuned``): ``metrics.json`` with the same
validation / test blocks and QC subsets, ``predictions.npz``, ``config.json`` with the embedding records (model id
renamed to ``ecfp4_tuned``), the grid, the chosen value and the library versions, and ``run.log`` with every
candidate's validation score. Targets are fitted in parallel with ``--threads`` worker processes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from downstream.baselines.ecfp import (
    MODEL_ID as FEATURE_MODEL_ID, TUNED_MODEL_ID, TUNED_PROTOCOL, TUNED_SEED, FingerprintSpec,
)
from downstream.common import split_replicates
from downstream.common.metrics import evaluate_multitask_classification, multitarget_regression_errors
from downstream.common.registry import embeddings_path, new_results_dir, outputs_root, run_record
from downstream.probe import evaluate_masked_subsets, validation_score
from downstream.tasks import SMILES_TASKS, load_task

MODEL_ID, PROTOCOL, SEED = TUNED_MODEL_ID, TUNED_PROTOCOL, TUNED_SEED
FEATURE_SCALING = "none"
LARGE_TRAIN = 30000
CLASSIFICATION_GRIDS = {"small_train": (0.01, 0.1, 1.0), "large_train": (0.05, 0.5)}
REGRESSION_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
MAX_ITER = 2000


def grid_for(kind: str, n_train: int) -> tuple:
    if kind == "multilabel_masked":
        return CLASSIFICATION_GRIDS["large_train" if n_train >= LARGE_TRAIN else "small_train"]
    if kind == "multitarget_regression":
        return REGRESSION_GRID
    raise ValueError(f"the tuned fingerprint baseline covers the MoleculeNet kinds only, got {kind!r}")


def settings_for(kind: str, grid: Sequence[float], metric: str, seeds: Sequence[int] = (SEED,)) -> dict:
    """The protocol settings of one task (their sha256 identifies the setting in the ledger)."""
    if kind == "multilabel_masked":
        estimator = {"estimator": "LogisticRegression", "solver": "liblinear", "penalty": "l2", "max_iter": MAX_ITER,
                     "class_weight": None, "parameter": "C", "selection": "validation mean ROC-AUC (highest)"}
    else:
        estimator = {"estimator": "Ridge", "solver": "auto", "parameter": "alpha", "target_standardization": False,
                     "selection": f"validation {metric} in original units (lowest)"}
    return {**estimator, "grid": [float(v) for v in grid], "one_model_per_target": True, "seeds": list(seeds),
            "feature_scaling": FEATURE_SCALING, "train_subsampling": None}


def load_features(path: Path, pooling: str):
    """``(chains, labels, csr float32 features)`` of one embedding file; the dense array is never held as float32."""
    from scipy import sparse

    with np.load(path) as data:
        key = f"emb_{pooling}"
        if key not in data:
            raise KeyError(f"{path} has no pooling {pooling!r} (has {[k for k in data if k.startswith('emb_')]})")
        dense = data[key]
        blocks = [sparse.csr_matrix(dense[i:i + 20000].astype(np.float32)) for i in range(0, len(dense), 20000)]
        return data["chains"].tolist(), data["labels"].astype(np.float32), sparse.vstack(blocks, format="csr")


def _fit_target(estimator: str, parameter: float, x_train, y_train, x_eval: Sequence) -> list:
    """Predictions on each matrix of ``x_eval`` of one target fitted on its labelled train rows."""
    if estimator == "logreg":
        from sklearn.linear_model import LogisticRegression

        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            raise ValueError("a train target has a single class; the logistic regression is undefined")
        model = LogisticRegression(C=parameter, solver="liblinear", max_iter=MAX_ITER).fit(x_train, y_train)
        return [model.predict_proba(x)[:, 1] for x in x_eval]
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=parameter).fit(x_train, y_train)
    return [model.predict(x) for x in x_eval]


def fit_candidate(kind: str, parameter: float, x_train, y_train, x_eval: Sequence, *, threads: int) -> list:
    """Predictions ``[eval matrix][molecule, target]`` (float32) of one grid value; targets fitted in parallel."""
    from joblib import Parallel, delayed

    estimator = "logreg" if kind == "multilabel_masked" else "ridge"
    jobs = []
    for target in range(y_train.shape[1]):
        labelled = ~np.isnan(y_train[:, target])
        if not labelled.any():
            raise ValueError(f"train target {target} has no labelled value")
        jobs.append((x_train[labelled], y_train[labelled, target].astype(np.float64)))
    fitted = Parallel(n_jobs=threads)(delayed(_fit_target)(estimator, parameter, x, y, x_eval) for x, y in jobs)
    return [np.stack([fitted[t][i] for t in range(len(fitted))], axis=1).astype(np.float32)
            for i in range(len(x_eval))]


def run_task(task_name: str, spec: FingerprintSpec, root: Path, *, threads: int, grid: Optional[Sequence[float]] = None,
             write: bool = True, split_tag: Optional[str] = None) -> dict:
    """Fit, select on validation, score test; with ``write`` the files of a probe seed go to the results directory.

    ``grid`` overrides the default grid (used for ablations, never by the runner). ``split_tag`` is required for a task
    with one split per seed (fit on that tag's split, written as the run of the tag's seed) and refused otherwise.
    Returns the numbers.
    """
    import joblib
    import sklearn

    task = load_task(task_name)
    config = task.load_task_config(task.TASK_YAML)
    kind, metric = task.KIND, config["primary_metric"]
    train_split, valid_split, test_split = config["splits"]
    replicated = split_replicates.is_replicated(config)
    if replicated:
        if split_tag is None:
            raise ValueError(f"{task_name} has one split per seed; pass the split tag (seed<k>)")
        seed = split_replicates.seed_for_tag(split_tag)
        split_replicates.tag_for_run(config, seed, explicit=split_tag)
        seeds, prefix = list(config["split_seeds"]), f"{split_tag}."
        test_subsets = lambda chains: task.test_subsets(config, chains, split_tag)  # noqa: E731
    else:
        if split_tag is not None:
            raise ValueError(f"{task_name} has no per-seed splits; --split-tag does not apply")
        seed, seeds, prefix = SEED, [SEED], ""
        test_subsets = lambda chains: task.test_subsets(config, chains)  # noqa: E731
    data = {name: load_features(embeddings_path(task_name, FEATURE_MODEL_ID, prefix + name, root=root), spec.pooling)
            for name in (train_split, valid_split, test_split)}
    for name, (chains, labels, _) in data.items():
        if len(chains) != labels.shape[0]:
            raise ValueError(f"{task_name} {name}: {len(chains)} chains but {labels.shape[0]} label rows")
    x_train, y_train = data[train_split][2], data[train_split][1]
    grid = tuple(grid) if grid is not None else grid_for(kind, x_train.shape[0])
    started = time.monotonic()
    lines, best = [], None
    evaluator = evaluate_multitask_classification if kind == "multilabel_masked" else multitarget_regression_errors
    for parameter in grid:
        valid_scores, test_scores = fit_candidate(kind, parameter, x_train, y_train,
                                                  [data[valid_split][2], data[test_split][2]], threads=threads)
        score = validation_score(kind, data[valid_split][1], valid_scores, metric)  # higher is better for both kinds
        shown = score if kind == "multilabel_masked" else -score
        lines.append(f"{'C' if kind == 'multilabel_masked' else 'alpha'} {parameter:g} validation {metric} {shown:.6f}")
        if best is None or score > best["score"]:  # strict: the first of equal candidates wins
            best = {"parameter": float(parameter), "score": score, "valid": valid_scores, "test": test_scores}
    metrics_val = evaluator(data[valid_split][1], best["valid"])
    subsets = test_subsets(data[test_split][0])
    metrics_test = {"all": evaluator(data[test_split][1], best["test"]),
                    "subsets": evaluate_masked_subsets(evaluator, data[test_split][1], best["test"], subsets)}
    result = {"task": task_name, "kind": kind, "metric": metric, "grid": list(grid), "selected": best["parameter"],
              "val": metrics_val, "test": metrics_test, "seconds": round(time.monotonic() - started, 1)}
    if not write:
        return result
    run = run_record()
    settings = settings_for(kind, grid, metric, seeds)
    out = new_results_dir(task_name, MODEL_ID, PROTOCOL, seed, pooling=spec.pooling, root=root)
    lines.append(f"selected {settings['parameter']} {best['parameter']:g} on validation; test scored once")
    (out / "run.log").write_text("\n".join(lines) + "\n")
    records = {}
    for name in (train_split, valid_split, test_split):
        record = json.loads(embeddings_path(task_name, FEATURE_MODEL_ID, prefix + name, root=root)
                            .with_suffix(".json").read_text())
        records[name] = dict(record, model_id=MODEL_ID, features_from_model_id=FEATURE_MODEL_ID)
    np.savez_compressed(out / "predictions.npz", test_chains=np.array(data[test_split][0]),
                        test_scores=best["test"], test_labels=data[test_split][1].astype(np.float32),
                        valid_scores=best["valid"])
    metrics = {"task": task_name, "kind": kind, "model_id": MODEL_ID, "protocol": PROTOCOL, "pooling": spec.pooling,
               "seed": seed, "best_epoch": None, "split_tag": split_tag, "representation": None,
               "val": metrics_val, "test": metrics_test}
    (out / "config.json").write_text(json.dumps({
        "args": {"task": task_name, "model_id": MODEL_ID, "head": PROTOCOL, "pooling": spec.pooling, "seeds": [seed],
                 "feature_scaling": FEATURE_SCALING, "threads": threads, "outputs_root": str(root)},
        "probe": settings, "selected": {settings["parameter"]: best["parameter"], "grid": [float(v) for v in grid],
                                        "validation_scores": lines[:-1]},
        "libraries": {"scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
                      "numpy": np.__version__, "python": sys.version.split()[0]},
        "embeddings": records, "run": run}, indent=1))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, choices=SMILES_TASKS)
    parser.add_argument("--feature", default="bits", choices=("bits", "counts"))
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8, help="worker processes fitting targets in parallel")
    parser.add_argument("--outputs-root", type=Path, default=None, help="default: $MIND_OUTPUTS_ROOT or <repo>/outputs")
    parser.add_argument("--split-tag", default=None, help="NPASS (one split per seed): seed<k>, required there")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    result = run_task(args.task, FingerprintSpec(args.feature, args.n_bits), outputs_root(args.outputs_root),
                      threads=args.threads, split_tag=args.split_tag)
    primary = result["test"]["all"][result["metric"]]
    print(f"{args.task}: selected {result['selected']:g} of {result['grid']} on validation; test "
          f"{result['metric']} {primary:.4f} ({result['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

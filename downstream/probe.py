"""Train probe heads on cached embeddings (CPU is enough).

    python -m downstream.probe --task ec  --model-id <snapshot|snapshot.random_init|esm2_...> --head linear
    python -m downstream.probe --task lba --model-id <snapshot> --head mlp [--split-tag identity30]

Per seed: features are standardized with train statistics, the head is trained,
the epoch with the best validation metric is kept (early stopping on
``patience``), and the test split is evaluated.

``ec``  multi-label, BCE loss, validation Fmax maximized, test also evaluated
        per sequence-identity cutoff.
``fold`` / ``reaction`` single-label multi-class, cross-entropy loss, validation accuracy maximized; one
        accuracy per test split (Fold has test_fold / test_superfamily / test_family).
``lba`` single-value regression, MSE (or Huber) loss, validation **RMSE
        minimized** (lower is better), test reported as RMSE / Pearson r /
        Spearman rho.
``rsr`` ranking regression (ATOM3D RNA Structure Ranking): one RMSD per model, MSE (or Huber) loss on RMSD standardized
        with train statistics, validation = mean per-puzzle Spearman maximized, test = mean per-puzzle and global
        Spearman / Pearson (+ mean top-1 loss, RMSE); the puzzle of a model is read from its id.
``ss_q3`` / ``ss_q8`` per-residue multi-class (rows = labelled residues, embeddings of ``downstream.embed``), same loss and
        validation accuracy as ``fold``; each test block also carries the coverage of its literature set.
``mn_*`` MoleculeNet. ``multilabel_masked`` (classification, NaN = missing label): masked BCE loss, validation
        mean ROC-AUC maximized, test ROC-AUC + PRC-AUC. ``multitarget_regression``: masked MSE on targets
        standardized with train statistics, validation = the task's primary error (RMSE or MAE, original units)
        minimized, test RMSE + MAE averaged over targets. Both also report subsets of the test molecules that
        the processing QC flagged (flat conformers, rare elements, ...); undefined subsets say why.

``--feature-scaling none`` is for the sparse ECFP4 fingerprints of ``downstream.baselines.ecfp`` (which use their own
model id and pooling name); encoder embeddings are always standardized, and the two are refused when mixed up.

Results go to ``outputs/downstream/<task>/<model_id>/<protocol>.<pooling>/seed<k>/``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from downstream.common.heads import HEADS, build_head
from downstream.common.metrics import (
    accuracy, evaluate, evaluate_multiclass, evaluate_multitask_classification, evaluate_ranking_regression,
    evaluate_regression, multitarget_regression_errors, multitask_roc_auc, protein_fmax, rmse, train_term_ic,
)
from downstream.common.registry import embeddings_path, new_results_dir, run_record, write_text_atomic
from downstream.tasks import KINDS, TASKS, load_task
from downstream.tasks.multiclass import split_roles
from downstream.tasks.ss.dataset import coverage_text as task_coverage_text


def load_embeddings(task: str, model_id: str, split: str, pooling: str, root=None, *, kind: str = "multilabel"):
    path = embeddings_path(task, model_id, split, root=root)
    with np.load(path) as data:
        key = f"emb_{pooling}"
        if key not in data:
            raise KeyError(f"{path} has no pooling {pooling!r} (has {[k for k in data if k.startswith('emb_')]})")
        labels = data["labels"].astype(np.int64 if kind == "multiclass" else np.float32)
        return data["chains"].tolist(), labels, data[key].astype(np.float32), path


MASKED_KINDS = ("multilabel_masked", "multitarget_regression")
SCALED_KINDS = ("multitarget_regression", "ranking_regression")  # heads fitted on train-standardized targets
SINGLE_VALUE_KINDS = ("regression", "ranking_regression")        # one output per sample
FEATURE_SCALINGS = ("standardize", "none")


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE over the labelled cells only (NaN target = missing); a batch without a labelled cell has zero loss and gradient."""

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        observed = ~torch.isnan(target)
        if not bool(observed.any()):
            return output.sum() * 0.0
        per_cell = nn.functional.binary_cross_entropy_with_logits(
            output, torch.where(observed, target, torch.zeros_like(target)), reduction="none")
        return (per_cell * observed).sum() / observed.sum()


class MaskedMSELoss(nn.Module):
    """MSE over the labelled cells only (NaN target = missing)."""

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        observed = ~torch.isnan(target)
        if not bool(observed.any()):
            return output.sum() * 0.0
        error = (output - torch.where(observed, target, torch.zeros_like(target))) ** 2
        return (error * observed).sum() / observed.sum()


def target_scale(labels: np.ndarray):
    """Per-target ``(mean, std)`` of the labelled train values; a target with no value or a constant one raises."""
    mean, std = np.zeros(labels.shape[1], np.float32), np.ones(labels.shape[1], np.float32)
    for target in range(labels.shape[1]):
        values = labels[~np.isnan(labels[:, target]), target]
        if values.size == 0:
            raise ValueError(f"train target {target} has no labelled value; it cannot be standardized")
        if float(values.std()) == 0.0:
            raise ValueError(f"train target {target} is constant ({float(values[0])}); standardization is undefined")
        mean[target], std[target] = values.mean(), values.std()
    return mean, std


def evaluate_masked_subsets(evaluator, labels: np.ndarray, scores: np.ndarray, subsets: Dict[str, np.ndarray]) -> dict:
    """Metrics of each flagged subset; a subset the metric cannot score is recorded as ``undefined`` with the reason."""
    result = {}
    for name, mask in sorted(subsets.items()):
        entry = {"molecules": int(mask.sum())}
        try:
            entry.update(evaluator(labels[mask], scores[mask]))
        except ValueError as exc:
            entry["undefined"] = str(exc)
        result[name] = entry
    return result


def _format_optional(value) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def evaluate_with_cutoffs(labels: np.ndarray, scores: np.ndarray, subsets: Dict[int, np.ndarray],
                          **ic_options) -> dict:
    result = {"all": evaluate(labels, scores, **ic_options)}
    for cutoff, mask in sorted(subsets.items()):
        result[f"cutoff_{cutoff}"] = evaluate(labels[mask], scores[mask], **ic_options) if mask.any() else None
    return result


def ic_options(config: dict, train_labels: np.ndarray) -> dict:
    """``evaluate`` keywords for the IC-binned macro AUPR: only tasks whose task.yaml has ``ic_binned_aupr`` (GO)."""
    settings = config.get("ic_binned_aupr")
    if settings is None:
        return {}
    if settings.get("ic_reference") != "train_frequency":
        raise ValueError(f"ic_binned_aupr.ic_reference must be 'train_frequency', got {settings.get('ic_reference')!r}")
    return {"term_ic": train_term_ic(train_labels), "ic_thresholds": tuple(float(t) for t in settings["ic_thresholds"])}


def output_scores(output: torch.Tensor, *, kind: str, scale=None) -> np.ndarray:
    """Head output (CPU) -> the scores the metrics read: probabilities, or targets in their original units."""
    if kind in ("multilabel", "multilabel_masked"):
        return torch.sigmoid(output).numpy()
    if kind == "multiclass":
        return torch.softmax(output, dim=-1).numpy()
    if kind in SCALED_KINDS:
        if scale is None:
            raise ValueError(f"a {kind} head predicts standardized targets; pass scale=(mean, std)")
        values = output.squeeze(-1).numpy() if kind == "ranking_regression" else output.numpy()
        return values * scale[1] + scale[0]
    if kind != "regression":
        raise ValueError(f"unknown task kind {kind!r}; choose from {KINDS}")
    return output.squeeze(-1).numpy()


@torch.no_grad()
def predict(head: nn.Module, features: torch.Tensor, *, kind: str = "multilabel", scale=None) -> np.ndarray:
    head.eval()
    return output_scores(head(features), kind=kind, scale=scale)


def output_dim(kind: str, num_classes, train_labels: np.ndarray) -> int:
    """Width of the head's last layer: classes (multiclass), 1 (regression) or one column per label/target."""
    if kind == "multiclass":
        if not num_classes:
            raise ValueError("a multiclass head needs num_classes (task.yaml label_count)")
        return int(num_classes)
    return 1 if kind in SINGLE_VALUE_KINDS else int(train_labels.shape[1])


def evaluate_task(kind: str, config: dict, *, train_labels: np.ndarray, val_labels: np.ndarray,
                  val_scores: np.ndarray, test_labels: Dict[str, np.ndarray], test_scores: Dict[str, np.ndarray],
                  subsets: dict, val_groups: Optional[np.ndarray] = None,
                  test_groups: Optional[Dict[str, np.ndarray]] = None) -> tuple:
    """``(metrics_val, metrics_test, headline)`` of one trained model; shared by the probe and fine-tuning.

    ``test_labels`` / ``test_scores`` map each test split name to its arrays, the primary test split first (only
    multiclass tasks have more than one). ``subsets`` are the primary test split's row masks (identity cutoffs or
    QC flags); empty for kinds without subsets. ``val_groups`` / ``test_groups`` (per-sample group labels, by test split
    name) are required by ``ranking_regression`` and refused elsewhere.
    """
    if kind != "ranking_regression" and (val_groups is not None or test_groups is not None):
        raise ValueError(f"group labels apply to ranking_regression tasks only, not {kind}")
    if list(test_scores) != list(test_labels) or not test_labels:
        raise ValueError(f"test splits of labels {list(test_labels)} and scores {list(test_scores)} differ")
    if kind != "multiclass" and len(test_labels) != 1:
        raise ValueError(f"a {kind} task has exactly one test split, got {list(test_labels)}")
    test_split = next(iter(test_labels))
    labels, scores = test_labels[test_split], test_scores[test_split]
    if kind == "multilabel":
        ic = ic_options(config, train_labels)
        metrics_val = evaluate(val_labels, val_scores, **ic)
        metrics_test = evaluate_with_cutoffs(labels, scores, subsets, **ic)
        headline = f"val fmax {metrics_val['fmax']:.4f} test fmax {metrics_test['all']['fmax']:.4f}"
    elif kind == "multiclass":
        metrics_val = evaluate_multiclass(val_labels, val_scores)
        metrics_test = {name: evaluate_multiclass(test_labels[name], test_scores[name]) for name in test_labels}
        headline = f"val accuracy {metrics_val['accuracy']:.4f} " + " ".join(
            f"{name} {metrics_test[name]['accuracy']:.4f}" for name in test_labels)
    elif kind in MASKED_KINDS:
        evaluator = (evaluate_multitask_classification if kind == "multilabel_masked"
                     else multitarget_regression_errors)
        metrics_val = evaluator(val_labels, val_scores)
        metrics_test = {"all": evaluator(labels, scores),
                        "subsets": evaluate_masked_subsets(evaluator, labels, scores, subsets)}
        headline = (f"val roc_auc {metrics_val['roc_auc']:.4f} test roc_auc {metrics_test['all']['roc_auc']:.4f} "
                    f"prc_auc {metrics_test['all']['prc_auc']:.4f} "
                    f"(tasks skipped {metrics_test['all']['num_tasks_skipped']})"
                    if kind == "multilabel_masked" else
                    f"val rmse {metrics_val['rmse']:.4f} mae {metrics_val['mae']:.4f} "
                    f"test rmse {metrics_test['all']['rmse']:.4f} mae {metrics_test['all']['mae']:.4f}")
    elif kind == "ranking_regression":
        if val_groups is None or test_groups is None or test_split not in test_groups:
            raise ValueError("ranking_regression evaluation needs val_groups and test_groups for the test split")
        metrics_val = evaluate_ranking_regression(val_labels.reshape(-1), val_scores, val_groups)
        metrics_test = evaluate_ranking_regression(labels.reshape(-1), scores, test_groups[test_split])
        mean_block, pooled = metrics_test["mean_per_puzzle"], metrics_test["global"]
        headline = (f"val mean spearman {_format_optional(metrics_val['mean_per_puzzle']['spearman_rho'])} test mean "
                    f"spearman {_format_optional(mean_block['spearman_rho'])} global spearman "
                    f"{_format_optional(pooled['spearman_rho'])} mean pearson {_format_optional(mean_block['pearson_r'])} "
                    f"global pearson {_format_optional(pooled['pearson_r'])} top1 loss {mean_block['top1_loss']:.3f}")
    elif kind == "regression":
        metrics_val = evaluate_regression(val_labels.reshape(-1), val_scores)
        metrics_test = {"all": evaluate_regression(labels.reshape(-1), scores)}
        headline = (f"val rmse {metrics_val['rmse']:.4f} test rmse {metrics_test['all']['rmse']:.4f} "
                    f"pearson {_format_optional(metrics_test['all']['pearson_r'])} "
                    f"spearman {_format_optional(metrics_test['all']['spearman_rho'])}")
        if metrics_test["all"]["correlation_undefined_reason"]:
            headline += f" (correlation undefined: {metrics_test['all']['correlation_undefined_reason']})"
    else:
        raise ValueError(f"unknown task kind {kind!r}; choose from {KINDS}")
    return metrics_val, metrics_test, headline


def save_predictions(path: Path, kind: str, *, test_chains: Dict[str, list], test_labels: Dict[str, np.ndarray],
                     test_scores: Dict[str, np.ndarray], val_scores: np.ndarray) -> None:
    """``predictions.npz``: the primary test split as ``test_*``, further (multiclass) test splits as ``<name>_*``."""
    names = list(test_labels)
    label_dtype = np.int64 if kind == "multiclass" else np.float32
    extra = {f"{name}_{field}": value for name in names[1:] for field, value in (
        ("chains", np.array(test_chains[name])), ("scores", test_scores[name].astype(np.float32)),
        ("labels", test_labels[name]))}
    np.savez_compressed(path, test_chains=np.array(test_chains[names[0]]),
                        test_scores=test_scores[names[0]].astype(np.float32),
                        test_labels=test_labels[names[0]].astype(label_dtype),
                        valid_scores=val_scores.astype(np.float32), **extra)


def build_loss(kind: str, settings: dict) -> nn.Module:
    if kind == "multilabel":
        return nn.BCEWithLogitsLoss()
    if kind == "multiclass":
        return nn.CrossEntropyLoss()
    if kind == "multilabel_masked":
        return MaskedBCEWithLogitsLoss()
    if kind == "multitarget_regression":
        return MaskedMSELoss()
    if kind not in SINGLE_VALUE_KINDS:
        raise ValueError(f"unknown task kind {kind!r}; choose from {KINDS}")
    loss = settings.get("loss", "mse")
    if loss == "mse":
        return nn.MSELoss()
    if loss == "huber":
        return nn.HuberLoss(delta=float(settings["huber_delta"]))
    raise ValueError(f"unknown regression loss {loss!r}; choose from mse, huber")


def validation_score(kind: str, targets: np.ndarray, predictions: np.ndarray, metric: str = None,
                     groups: Optional[np.ndarray] = None) -> float:
    """Higher is better, so one early-stopping rule covers both task kinds.

    EC maximizes Fmax directly; LBA minimizes RMSE, so the negative RMSE is
    returned. Never compare these two numbers across tasks. ``ranking_regression`` returns the mean per-puzzle
    Spearman (needs ``groups``); an undefined one (a puzzle with constant predictions) is ``-inf``, so such an epoch
    is never selected and a run with no defined epoch fails loudly.
    """
    if kind == "ranking_regression":
        if groups is None:
            raise ValueError("ranking_regression validation needs the group (puzzle) of every sample")
        value = evaluate_ranking_regression(targets.reshape(-1), predictions.reshape(-1), groups)["mean_per_puzzle"]
        return -float("inf") if value["spearman_rho"] is None else float(value["spearman_rho"])
    if kind == "multilabel":
        return protein_fmax(targets, predictions)["fmax"]
    if kind == "multiclass":
        return accuracy(targets, predictions)
    if kind == "multilabel_masked":
        return multitask_roc_auc(targets, predictions)["roc_auc"]
    if kind == "multitarget_regression":
        if metric not in ("rmse", "mae"):
            raise ValueError(f"multitarget_regression validation needs metric 'rmse' or 'mae', got {metric!r}")
        return -multitarget_regression_errors(targets, predictions)[metric]
    if kind != "regression":
        raise ValueError(f"unknown task kind {kind!r}; choose from {KINDS}")
    return -rmse(targets.reshape(-1), predictions.reshape(-1))


def validation_label(kind: str, score: float, metric: str = None) -> tuple:
    """``(name, value)`` of a ``validation_score`` for the run log; errors are shown with their own sign."""
    return (("val_fmax", score) if kind == "multilabel" else
            ("val_accuracy", score) if kind == "multiclass" else
            ("val_roc_auc", score) if kind == "multilabel_masked" else
            ("val_mean_spearman", score) if kind == "ranking_regression" else
            (f"val_{metric}", -score) if kind == "multitarget_regression" else ("val_rmse", -score))


def train_probe(train_x, train_y, val_x, val_y, *, head_kind, settings, seed, log, kind="multilabel",
                num_classes=None, scale=None, metric=None, val_groups=None):
    torch.manual_seed(seed)
    fit_y = train_y
    if kind in SCALED_KINDS:
        if scale is None:
            raise ValueError(f"a {kind} probe needs scale=(mean, std) of the train targets")
        fit_y = (train_y - torch.from_numpy(scale[0])) / torch.from_numpy(scale[1])  # NaN stays NaN (missing)
    if kind == "multiclass" and not num_classes:
        raise ValueError("a multiclass probe needs num_classes (task.yaml label_count)")
    out_dim = output_dim(kind, num_classes, train_y)
    head = build_head(head_kind, train_x.shape[1], out_dim,
                      hidden_dim=settings["mlp_hidden"], dropout=settings["mlp_dropout"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    loss_fn = build_loss(kind, settings)
    generator = torch.Generator().manual_seed(seed)
    best = {"score": -float("inf"), "epoch": -1, "state": None}
    for epoch in range(settings["epochs"]):
        head.train()
        order = torch.randperm(len(train_x), generator=generator)
        total = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            rows = order[start:start + settings["batch_size"]]
            optimizer.zero_grad()
            output = head(train_x[rows])
            target = fit_y[rows]
            loss = (loss_fn(output.squeeze(-1), target.reshape(-1)) if kind in SINGLE_VALUE_KINDS
                    else loss_fn(output, target))
            loss.backward()
            optimizer.step()
            total += float(loss) * len(rows)
        score = validation_score(kind, val_y.numpy(), predict(head, val_x, kind=kind, scale=scale), metric,
                                 groups=val_groups)
        label, shown = validation_label(kind, score, metric)
        log(f"epoch {epoch} train_loss {total / len(order):.5f} {label} {shown:.4f}")
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "state": copy.deepcopy(head.state_dict())}
        elif epoch - best["epoch"] >= settings["patience"]:
            log(f"early stop at epoch {epoch}; best epoch {best['epoch']}")
            break
    if best["state"] is None:
        raise RuntimeError("probe training produced no validated epoch")
    head.load_state_dict(best["state"])
    best["fmax"] = best["score"]  # kept for the EC callers that read this key
    return head, best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--head", required=True, choices=HEADS)
    parser.add_argument("--pooling", default=None, help="default: probe.pooling in task.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--outputs-root", type=Path, default=None)
    parser.add_argument("--split-tag", default=None, help="LBA only: identity30 (default) | identity60")
    parser.add_argument("--representation", default=None, help="LBA only: joint (default) | separate")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for torch; the machine is shared with training runs")
    parser.add_argument("--feature-scaling", default="standardize", choices=FEATURE_SCALINGS,
                        help="standardize: train mean/std per feature (every encoder embedding); none: raw features "
                             "(sparse fingerprints, whose rare bits a train std would blow up)")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    run = run_record()
    torch.set_num_threads(args.threads)
    task = load_task(args.task)
    kind = task.KIND
    if args.task != "lba" and (args.split_tag or args.representation):
        parser.error("--split-tag and --representation apply to the lba task only")
    config = task.load_task_config(task.TASK_YAML, data_root=args.data_root)
    settings = config["probe"]
    pooling = args.pooling or settings["pooling"]
    protocol = f"{args.head}_probe"
    train_split, val_split, test_splits = split_roles(config)
    test_split = test_splits[0]
    prefix = ""
    split_tag = representation = None
    if args.task == "lba":
        split_tag = args.split_tag or config["primary_split_tag"]
        representation = args.representation or config["representation"]
        prefix = f"{split_tag}.{representation}."
        protocol = f"{protocol}.{split_tag}.{representation}"

    data = {name: load_embeddings(args.task, args.model_id, prefix + name, pooling, args.outputs_root, kind=kind)
            for name in (train_split, val_split, *test_splits)}
    provenance = {s: json.loads(Path(p).with_suffix(".json").read_text()) for s, (_, _, _, p) in data.items()}
    fingerprint = {"fingerprint" in record for record in provenance.values()}
    if len(fingerprint) != 1:
        parser.error("the embedding splits disagree on being a fingerprint")
    if fingerprint.pop() != (args.feature_scaling == "none"):
        parser.error("fingerprint features need --feature-scaling none and encoder embeddings --feature-scaling "
                     "standardize (the ledger groups probes by their settings and cannot tell the two apart)")
    if args.feature_scaling == "standardize":
        mean = data[train_split][2].mean(axis=0)
        std = data[train_split][2].std(axis=0).clip(min=1e-6)
    else:
        mean = np.zeros(data[train_split][2].shape[1], np.float32)
        std = np.ones_like(mean)
    tensors = {name: (torch.from_numpy((x - mean) / std), torch.from_numpy(y))
               for name, (_, y, x, _) in data.items()}
    subsets = task.test_subsets(config, data[test_split][0]) if kind in ("multilabel", *MASKED_KINDS) else {}
    scale = target_scale(data[train_split][1]) if kind in SCALED_KINDS else None
    groups = ({name: task.group_labels(data[name][0]) for name in (val_split, *test_splits)}
              if kind == "ranking_regression" else {})
    for seed in args.seeds or settings["seeds"]:
        out = new_results_dir(args.task, args.model_id, protocol, seed, pooling=pooling,
                              root=args.outputs_root)
        log_handle = open(out / "run.log", "w")

        def log(message):
            print(message, file=log_handle, flush=True)

        head, best = train_probe(*tensors[train_split], *tensors[val_split], head_kind=args.head,
                                 settings=settings, seed=seed, log=log, kind=kind,
                                 num_classes=config.get("label_count"), scale=scale,
                                 metric=config.get("primary_metric"), val_groups=groups.get(val_split))
        val_scores = predict(head, tensors[val_split][0], kind=kind, scale=scale)
        test_scores = {name: predict(head, tensors[name][0], kind=kind, scale=scale) for name in test_splits}
        test_labels = {name: data[name][1] for name in test_splits}
        metrics_val, metrics_test, headline = evaluate_task(
            kind, config, train_labels=data[train_split][1], val_labels=data[val_split][1], val_scores=val_scores,
            test_labels=test_labels, test_scores=test_scores, subsets=subsets, val_groups=groups.get(val_split),
            test_groups={name: groups[name] for name in test_splits} if groups else None)
        if hasattr(task, "augment_test_metrics"):  # per-residue tasks: attach each test set's coverage (subset marker)
            coverage = {name: provenance[name]["coverage"] for name in test_splits}
            task.augment_test_metrics(metrics_test, coverage)
            headline += " | " + task_coverage_text(coverage)
            (out / "coverage.json").write_text(json.dumps(coverage, indent=1))
        metrics = {
            "task": args.task, "kind": kind, "model_id": args.model_id, "protocol": protocol,
            "pooling": pooling, "seed": seed, "best_epoch": best["epoch"],
            "split_tag": split_tag, "representation": representation,
            "val": metrics_val, "test": metrics_test,
        }
        if scale is not None:
            metrics["target_standardization"] = {"mean": scale[0].tolist(), "std": scale[1].tolist(),
                                                 "primary_metric": config.get("primary_metric")}
        save_predictions(out / "predictions.npz", kind, test_chains={name: data[name][0] for name in test_splits},
                         test_labels=test_labels, test_scores=test_scores, val_scores=val_scores)
        torch.save({"state_dict": head.state_dict(), "feature_mean": mean, "feature_std": std,
                    "head": args.head, "in_dim": int(mean.shape[0]),
                    **({"target_mean": scale[0], "target_std": scale[1]} if scale is not None else {})}, out / "head.pt")
        (out / "config.json").write_text(json.dumps({"args": vars(args) | {"data_root": str(args.data_root),
                                                     "outputs_root": str(args.outputs_root)},
                                                     "probe": settings, "embeddings": provenance, "run": run},
                                                    indent=1))
        write_text_atomic(out / "metrics.json", json.dumps(metrics, indent=1))  # last: marks the seed complete
        log_handle.close()
        print(f"seed {seed}: {headline} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

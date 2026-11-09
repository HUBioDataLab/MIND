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
``mn_*`` MoleculeNet. ``multilabel_masked`` (classification, NaN = missing label): masked BCE loss, validation
        mean ROC-AUC maximized, test ROC-AUC + PRC-AUC. ``multitarget_regression``: masked MSE on targets
        standardized with train statistics, validation = the task's primary error (RMSE or MAE, original units)
        minimized, test RMSE + MAE averaged over targets. Both also report subsets of the test molecules that
        the processing QC flagged (flat conformers, rare elements, ...); undefined subsets say why.

Results go to ``outputs/downstream/<task>/<model_id>/<protocol>.<pooling>/seed<k>/``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn

from downstream.common.heads import HEADS, build_head
from downstream.common.metrics import (
    accuracy, evaluate, evaluate_multiclass, evaluate_multitask_classification, evaluate_regression,
    multitarget_regression_errors, multitask_roc_auc, protein_fmax, rmse,
)
from downstream.common.registry import embeddings_path, new_results_dir
from downstream.tasks import KINDS, TASKS, load_task
from downstream.tasks.multiclass import split_roles


def load_embeddings(task: str, model_id: str, split: str, pooling: str, root=None, *, kind: str = "multilabel"):
    path = embeddings_path(task, model_id, split, root=root)
    with np.load(path) as data:
        key = f"emb_{pooling}"
        if key not in data:
            raise KeyError(f"{path} has no pooling {pooling!r} (has {[k for k in data if k.startswith('emb_')]})")
        labels = data["labels"].astype(np.int64 if kind == "multiclass" else np.float32)
        return data["chains"].tolist(), labels, data[key].astype(np.float32), path


MASKED_KINDS = ("multilabel_masked", "multitarget_regression")


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


def evaluate_with_cutoffs(labels: np.ndarray, scores: np.ndarray, subsets: Dict[int, np.ndarray]) -> dict:
    result = {"all": evaluate(labels, scores)}
    for cutoff, mask in sorted(subsets.items()):
        result[f"cutoff_{cutoff}"] = evaluate(labels[mask], scores[mask]) if mask.any() else None
    return result


@torch.no_grad()
def predict(head: nn.Module, features: torch.Tensor, *, kind: str = "multilabel", scale=None) -> np.ndarray:
    head.eval()
    output = head(features)
    if kind in ("multilabel", "multilabel_masked"):
        return torch.sigmoid(output).numpy()
    if kind == "multiclass":
        return torch.softmax(output, dim=-1).numpy()
    if kind == "multitarget_regression":
        if scale is None:
            raise ValueError("a multitarget_regression head predicts standardized targets; pass scale=(mean, std)")
        return output.numpy() * scale[1] + scale[0]
    if kind != "regression":
        raise ValueError(f"unknown task kind {kind!r}; choose from {KINDS}")
    return output.squeeze(-1).numpy()


def build_loss(kind: str, settings: dict) -> nn.Module:
    if kind == "multilabel":
        return nn.BCEWithLogitsLoss()
    if kind == "multiclass":
        return nn.CrossEntropyLoss()
    if kind == "multilabel_masked":
        return MaskedBCEWithLogitsLoss()
    if kind == "multitarget_regression":
        return MaskedMSELoss()
    if kind != "regression":
        raise ValueError(f"unknown task kind {kind!r}; choose from {KINDS}")
    loss = settings.get("loss", "mse")
    if loss == "mse":
        return nn.MSELoss()
    if loss == "huber":
        return nn.HuberLoss(delta=float(settings["huber_delta"]))
    raise ValueError(f"unknown regression loss {loss!r}; choose from mse, huber")


def validation_score(kind: str, targets: np.ndarray, predictions: np.ndarray, metric: str = None) -> float:
    """Higher is better, so one early-stopping rule covers both task kinds.

    EC maximizes Fmax directly; LBA minimizes RMSE, so the negative RMSE is
    returned. Never compare these two numbers across tasks.
    """
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


def train_probe(train_x, train_y, val_x, val_y, *, head_kind, settings, seed, log, kind="multilabel",
                num_classes=None, scale=None, metric=None):
    torch.manual_seed(seed)
    fit_y = train_y
    if kind == "multitarget_regression":
        if scale is None:
            raise ValueError("a multitarget_regression probe needs scale=(mean, std) of the train targets")
        fit_y = (train_y - torch.from_numpy(scale[0])) / torch.from_numpy(scale[1])  # NaN stays NaN (missing)
    if kind == "multiclass":
        if not num_classes:
            raise ValueError("a multiclass probe needs num_classes (task.yaml label_count)")
        out_dim = int(num_classes)
    else:
        out_dim = 1 if kind == "regression" else train_y.shape[1]
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
            loss = (loss_fn(output.squeeze(-1), target.reshape(-1)) if kind == "regression"
                    else loss_fn(output, target))
            loss.backward()
            optimizer.step()
            total += float(loss) * len(rows)
        score = validation_score(kind, val_y.numpy(), predict(head, val_x, kind=kind, scale=scale), metric)
        label, shown = (("val_fmax", score) if kind == "multilabel" else
                        ("val_accuracy", score) if kind == "multiclass" else
                        ("val_roc_auc", score) if kind == "multilabel_masked" else
                        (f"val_{metric}", -score) if kind == "multitarget_regression" else ("val_rmse", -score))
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
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
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
    mean = data[train_split][2].mean(axis=0)
    std = data[train_split][2].std(axis=0).clip(min=1e-6)
    tensors = {name: (torch.from_numpy((x - mean) / std), torch.from_numpy(y))
               for name, (_, y, x, _) in data.items()}
    subsets = task.test_subsets(config, data[test_split][0]) if kind in ("multilabel", *MASKED_KINDS) else {}
    scale = target_scale(data[train_split][1]) if kind == "multitarget_regression" else None
    for seed in args.seeds or settings["seeds"]:
        out = new_results_dir(args.task, args.model_id, protocol, seed, pooling=pooling,
                              root=args.outputs_root)
        log_handle = open(out / "run.log", "w")

        def log(message):
            print(message, file=log_handle, flush=True)

        head, best = train_probe(*tensors[train_split], *tensors[val_split], head_kind=args.head,
                                 settings=settings, seed=seed, log=log, kind=kind,
                                 num_classes=config.get("label_count"), scale=scale,
                                 metric=config.get("primary_metric"))
        test_scores = predict(head, tensors[test_split][0], kind=kind, scale=scale)
        val_scores = predict(head, tensors[val_split][0], kind=kind, scale=scale)
        if kind == "multilabel":
            metrics_val = evaluate(data[val_split][1], val_scores)
            metrics_test = evaluate_with_cutoffs(data[test_split][1], test_scores, subsets)
            headline = f"val fmax {metrics_val['fmax']:.4f} test fmax {metrics_test['all']['fmax']:.4f}"
        elif kind == "multiclass":
            metrics_val = evaluate_multiclass(data[val_split][1], val_scores)
            all_test_scores = {name: predict(head, tensors[name][0], kind=kind) for name in test_splits}
            metrics_test = {name: evaluate_multiclass(data[name][1], all_test_scores[name]) for name in test_splits}
            headline = f"val accuracy {metrics_val['accuracy']:.4f} " + " ".join(
                f"{name} {metrics_test[name]['accuracy']:.4f}" for name in test_splits)
        elif kind in MASKED_KINDS:
            evaluator = (evaluate_multitask_classification if kind == "multilabel_masked"
                         else multitarget_regression_errors)
            metrics_val = evaluator(data[val_split][1], val_scores)
            metrics_test = {"all": evaluator(data[test_split][1], test_scores),
                            "subsets": evaluate_masked_subsets(evaluator, data[test_split][1], test_scores, subsets)}
            headline = (f"val roc_auc {metrics_val['roc_auc']:.4f} test roc_auc {metrics_test['all']['roc_auc']:.4f} "
                        f"prc_auc {metrics_test['all']['prc_auc']:.4f} "
                        f"(tasks skipped {metrics_test['all']['num_tasks_skipped']})"
                        if kind == "multilabel_masked" else
                        f"val rmse {metrics_val['rmse']:.4f} mae {metrics_val['mae']:.4f} "
                        f"test rmse {metrics_test['all']['rmse']:.4f} mae {metrics_test['all']['mae']:.4f}")
        else:
            metrics_val = evaluate_regression(data[val_split][1].reshape(-1), val_scores)
            metrics_test = {"all": evaluate_regression(data[test_split][1].reshape(-1), test_scores)}
            headline = (f"val rmse {metrics_val['rmse']:.4f} test rmse {metrics_test['all']['rmse']:.4f} "
                        f"pearson {_format_optional(metrics_test['all']['pearson_r'])} "
                        f"spearman {_format_optional(metrics_test['all']['spearman_rho'])}")
            if metrics_test["all"]["correlation_undefined_reason"]:
                headline += f" (correlation undefined: {metrics_test['all']['correlation_undefined_reason']})"
        metrics = {
            "task": args.task, "kind": kind, "model_id": args.model_id, "protocol": protocol,
            "pooling": pooling, "seed": seed, "best_epoch": best["epoch"],
            "split_tag": split_tag, "representation": representation,
            "val": metrics_val, "test": metrics_test,
        }
        if scale is not None:
            metrics["target_standardization"] = {"mean": scale[0].tolist(), "std": scale[1].tolist(),
                                                 "primary_metric": config["primary_metric"]}
        extra = ({f"{name}_{field}": value for name in test_splits[1:] for field, value in (
            ("chains", np.array(data[name][0])), ("scores", all_test_scores[name].astype(np.float32)),
            ("labels", data[name][1]))} if kind == "multiclass" else {})
        np.savez_compressed(out / "predictions.npz", test_chains=np.array(data[test_split][0]),
                            test_scores=test_scores.astype(np.float32),
                            test_labels=data[test_split][1].astype(np.float32 if kind != "multiclass" else np.int64),
                            valid_scores=val_scores.astype(np.float32), **extra)
        torch.save({"state_dict": head.state_dict(), "feature_mean": mean, "feature_std": std,
                    "head": args.head, "in_dim": int(mean.shape[0]),
                    **({"target_mean": scale[0], "target_std": scale[1]} if scale is not None else {})}, out / "head.pt")
        provenance = {s: json.loads(Path(p).with_suffix(".json").read_text()) for s, (_, _, _, p) in data.items()}
        (out / "config.json").write_text(json.dumps({"args": vars(args) | {"data_root": str(args.data_root),
                                                     "outputs_root": str(args.outputs_root)},
                                                     "probe": settings, "embeddings": provenance}, indent=1))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        log_handle.close()
        print(f"seed {seed}: {headline} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

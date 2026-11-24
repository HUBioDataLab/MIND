"""End-to-end fine-tuning of a MIND encoder plus head (GPU training workload; user-run via Make only).

    python -m downstream.finetune --task <task> --model <snapshot> --device cuda:0 [--random-init] [--seeds 0 1 2]
    python -m downstream.finetune --task lba --model <snapshot> --device cuda:0 [--split-tag identity30]

Every task kind is fine-tuned with the probe's loss, validation score and evaluation code
(``downstream.probe.build_loss`` / ``validation_score`` / ``evaluate_task``), so a fine-tune result and a probe result
of one task carry the same metrics, computed the same way:

``multilabel`` (EC, GO)  BCE; validation Fmax; test per identity cutoff (GO also IC-binned macro AUPR).
``multiclass`` (Fold, Reaction)  cross-entropy; validation accuracy; one metric block per test split.
``regression`` (LBA)  MSE (or Huber); validation RMSE minimized; test RMSE / Pearson / Spearman.
``multilabel_masked`` (MoleculeNet classification)  BCE over labelled cells only (NaN = missing); validation mean
    ROC-AUC; test ROC-AUC / PRC-AUC averaged over tasks, plus the QC-flagged test subsets.
``multitarget_regression`` (MoleculeNet regression)  masked MSE on targets standardized with train statistics;
    validation = the task's primary error (original units) minimized; test RMSE / MAE plus the QC subsets.
``ranking_regression`` (RSR)  MSE on the RMSD standardized with train statistics; validation = mean per-puzzle Spearman
    maximized; test mean per-puzzle and global Spearman / Pearson, mean top-1 loss, RMSE.

Protocol (``finetune`` block of task.yaml, resolved by ``downstream.finetune_settings``; CLI overrides are recorded in
config.json): AdamW with ``lr_encoder`` for the encoder and ``lr_head`` for the head, gradients clipped to
``grad_clip_norm`` (global norm over encoder and head; null = no clipping). As in the probe, the head reads pooled
features standardized with the train mean / std; here the statistics come from the initial encoder (before any update)
and are fixed buffers of the head. ``--random-init`` builds the same architecture with untrained weights (seeded with
the run seed) and trains it from scratch with ``lr_encoder_random_init``: a learning rate that suits a pretrained
encoder is far too small for an untrained one, so each arm uses its own. Training runs ``epochs`` epochs
(``patience: null``) or stops after ``patience`` epochs without validation improvement; the epoch with the best
validation score, computed on the valid split only, is restored and evaluated on the test split(s). A run whose best
epoch is its last one says so in run.log (that arm may still have been improving).

Batches hold at most ``max_atoms_per_batch`` graph nodes and, when set, at most ``max_samples_per_batch`` samples
(MoleculeNet: 32 molecules, the per-process batch of the Uni-Mol fine-tuning recipe; without the cap a 15,000-node
budget packs hundreds of small molecules into one optimizer step). Training samples get fresh random long-range edges
every epoch (seed, epoch), as in pretraining; validation/test use the fixed embedding seed and run with deterministic
CUDA algorithms, as ``downstream.embed`` does; training does not (``core.deterministic``: atomic adds in the backward
pass). Windows of a cropped chain are separate training examples carrying the chain labels; at evaluation their pooled
embeddings are residue-weighted per chain before the head. LBA fine-tuning supports the ``joint`` representation only.

One process trains on one device (``--device``; CUDA only, flash_varlen attention has no CPU kernel). There is no
multi-GPU (DDP) path. Results go to ``outputs/downstream/<task>/<model_id>/<protocol>.<pooling>/seed<k>/``
(protocol ``finetune``, LBA ``finetune.<split_tag>.joint``); ``model_id`` is the snapshot name, plus
``.random_init`` for the untrained control. Every requested seed is checked before training starts: an existing
``metrics.json`` is never overwritten. ``metrics.json`` is written last and atomically; it marks a finished seed.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

from downstream.common import encoder as mind_encoder
from downstream.common.heads import build_head
from downstream.common.registry import (
    RANDOM_INIT_SUFFIX, git_commit, new_results_dir, resolve_model, results_dir, run_record, sha256_file,
    write_text_atomic,
)
from downstream.embed import embed_split_mind
from downstream.finetune_settings import OVERRIDES, resolve_settings
from downstream.probe import (
    MASKED_KINDS, SCALED_KINDS, SINGLE_VALUE_KINDS, build_loss, evaluate_task, output_dim, output_scores,
    save_predictions, target_scale, validation_label, validation_score,
)
from downstream.tasks import TASKS, load_task
from downstream.tasks.multiclass import split_roles


class StandardizedHead(nn.Module):
    """``head((x - feature_mean) / feature_std)``; the statistics are fixed buffers, never trained."""

    def __init__(self, head: nn.Module, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.head = head
        self.register_buffer("feature_mean", torch.as_tensor(np.asarray(mean), dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(np.asarray(std), dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head((features - self.feature_mean) / self.feature_std)


def feature_statistics(features: np.ndarray) -> tuple:
    """Train mean and std per feature, the probe's rule (std clipped at 1e-6)."""
    return features.mean(axis=0), features.std(axis=0).clip(min=1e-6)


def training_targets(labels: np.ndarray, kind: str, scale=None) -> torch.Tensor:
    """Label rows as the loss reads them: class ids, or float rows (NaN = missing), standardized for regression."""
    labels = np.asarray(labels)
    if kind == "multiclass":
        return torch.from_numpy(labels.astype(np.int64))
    values = labels.astype(np.float32)
    if kind in SCALED_KINDS:
        if scale is None:
            raise ValueError(f"{kind} fine-tuning needs scale=(mean, std) of the train targets")
        values = (values - scale[0]) / scale[1]  # NaN stays NaN (missing)
    return torch.from_numpy(values)


def compute_loss(loss_fn: nn.Module, output: torch.Tensor, targets: torch.Tensor, kind: str) -> torch.Tensor:
    if kind in SINGLE_VALUE_KINDS:
        return loss_fn(output.squeeze(-1), targets.reshape(-1))
    return loss_fn(output, targets)


def build_optimizer(model: nn.Module, head: nn.Module, settings: dict) -> torch.optim.Optimizer:
    """AdamW over every encoder parameter (``lr_encoder``) and every head parameter (``lr_head``)."""
    return torch.optim.AdamW([
        {"params": list(model.parameters()), "lr": float(settings["lr_encoder"])},
        {"params": list(head.parameters()), "lr": float(settings["lr_head"])},
    ], weight_decay=float(settings["weight_decay"]))


@contextlib.contextmanager
def deterministic_scoring(device: torch.device):
    """Deterministic CUDA kernels while scoring (as ``core.deterministic`` for embedding); the mode is restored."""
    if device.type != "cuda":
        yield
        return
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def train_epoch(model, head, split, optimizer, device, *, epoch_seed: int, settings: dict, loss_fn: nn.Module,
                kind: str, targets: torch.Tensor) -> float:
    """One pass over the training samples in a seeded order, in batches within the node and sample caps."""
    model.train()
    head.train()
    pooling, max_atoms = settings["pooling"], int(settings["max_atoms_per_batch"])
    max_samples, clip = settings["max_samples_per_batch"], settings["grad_clip_norm"]
    parameters = list(model.parameters()) + list(head.parameters())
    sample_chain = torch.as_tensor(split.sample_chain).long()
    if len(sample_chain) != len(split):
        raise ValueError(f"split has {len(split)} samples but sample_chain has {len(sample_chain)} entries")
    order = torch.randperm(len(split), generator=torch.Generator().manual_seed(epoch_seed)).tolist()
    indices, samples, atoms, total, seen = [], [], 0, 0.0, 0

    def step():
        graph = mind_encoder.collate(samples).to(device)
        node_embeddings, pool_mask = mind_encoder.encode(model, graph)
        pooled = mind_encoder.pool(node_embeddings, graph, pool_mask, [pooling])[pooling]
        loss = compute_loss(loss_fn, head(pooled), targets[sample_chain[indices]].to(device), kind)
        optimizer.zero_grad()
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters, float(clip))
        optimizer.step()
        return float(loss) * len(indices)

    for index in order:
        sample = split.runtime(index, model.config, seed=epoch_seed)
        full = max_samples is not None and len(samples) >= int(max_samples)
        if samples and (full or atoms + sample.num_nodes > max_atoms):
            total += step()
            seen += len(indices)
            indices, samples, atoms = [], [], 0
        indices.append(index)
        samples.append(sample)
        atoms += sample.num_nodes
    if samples:
        total += step()
        seen += len(indices)
    return total / max(seen, 1)


@torch.no_grad()
def split_features(model, split, device, *, seed: int, settings: dict) -> np.ndarray:
    """Pooled per-chain features of the current encoder (eval mode, fixed edge seed, deterministic kernels)."""
    model.eval()
    with deterministic_scoring(torch.device(device)):
        return embed_split_mind(model, split, device, seed=seed, poolings=[settings["pooling"]],
                                max_atoms=int(settings["max_atoms_per_batch"]))[settings["pooling"]]


@torch.no_grad()
def split_scores(model, head, split, device, *, seed: int, settings: dict, kind: str, scale=None) -> np.ndarray:
    """Per-chain scores of the current encoder + head, in the units ``downstream.probe.output_scores`` defines."""
    features = split_features(model, split, device, seed=seed, settings=settings)
    head.eval()
    output = head(torch.from_numpy(features).to(device)).float().cpu()
    return output_scores(output, kind=kind, scale=scale)


def fit(model, head, splits: dict, *, train_split: str, val_split: str, device, settings: dict, seed: int,
        eval_seed: int, kind: str, scale, metric: Optional[str], log: Callable[[str], None],
        val_groups=None) -> dict:
    """Train encoder + head and restore the epoch with the best validation score; returns that epoch's record."""
    optimizer = build_optimizer(model, head, settings)
    loss_fn = build_loss(kind, settings)
    targets = training_targets(splits[train_split].labels, kind, scale)
    val_labels = np.asarray(splits[val_split].labels)
    epochs, patience = int(settings["epochs"]), settings["patience"]
    best = {"score": -float("inf"), "epoch": -1}
    for epoch in range(epochs):
        started = time.monotonic()
        loss = train_epoch(model, head, splits[train_split], optimizer, device, epoch_seed=seed * 100003 + epoch,
                           settings=settings, loss_fn=loss_fn, kind=kind, targets=targets)
        trained = time.monotonic()
        scores = split_scores(model, head, splits[val_split], device, seed=eval_seed, settings=settings, kind=kind,
                              scale=scale)
        score = validation_score(kind, val_labels, scores, metric, groups=val_groups)
        label, shown = validation_label(kind, score, metric)
        log(f"epoch {epoch} train_loss {loss:.5f} {label} {shown:.4f} (train {trained - started:.1f} s, "
            f"validation {time.monotonic() - trained:.1f} s)")
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "encoder": copy.deepcopy(model.state_dict()),
                    "head": copy.deepcopy(head.state_dict())}
        elif patience is not None and epoch - best["epoch"] >= int(patience):
            log(f"early stop at epoch {epoch}; best epoch {best['epoch']}")
            break
    if "encoder" not in best:
        raise RuntimeError("fine-tuning produced no validated epoch")
    if best["epoch"] == epochs - 1:
        log(f"best epoch is the last epoch ({best['epoch']}): validation was still improving; the run may be cut short")
    model.load_state_dict(best["encoder"])
    head.load_state_dict(best["head"])
    return best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--model", required=True, help="snapshot name or alias")
    parser.add_argument("--random-init", action="store_true",
                        help="same architecture, untrained weights seeded with the run seed (lr_encoder_random_init)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="default: finetune.seeds in task.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--outputs-root", type=Path, default=None)
    parser.add_argument("--split-tag", default=None, help="LBA only: identity30 (default) | identity60")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for torch; the machine is shared with training runs")
    parser.add_argument("--lr-encoder", type=float, default=None,
                        help="override the encoder LR of the arm (finetune.lr_encoder or lr_encoder_random_init)")
    parser.add_argument("--lr-head", type=float, default=None, help="override finetune.lr_head")
    parser.add_argument("--weight-decay", type=float, default=None, help="override finetune.weight_decay (AdamW, >= 0)")
    parser.add_argument("--mlp-dropout", type=float, default=None, help="override finetune.mlp_dropout, in [0, 1)")
    parser.add_argument("--pooling", default=None, choices=mind_encoder.ALL_POOLINGS,
                        help="override finetune.pooling (the task's embedding.poolings limit the choice)")
    parser.add_argument("--epochs", type=int, default=None, help="override finetune.epochs")
    parser.add_argument("--patience", type=int, default=None, help="override finetune.patience (task.yaml: null)")
    return parser


def check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    for name in ("lr_encoder", "lr_head", "epochs", "patience"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.weight_decay is not None and args.weight_decay < 0:
        parser.error("--weight-decay must be >= 0")
    if args.mlp_dropout is not None and not 0 <= args.mlp_dropout < 1:
        parser.error("--mlp-dropout must be in [0, 1)")
    if args.task != "lba" and args.split_tag:
        parser.error("--split-tag applies to the lba task only")
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds lists a seed twice")


def run(args: argparse.Namespace) -> list:
    """Fine-tune every requested seed and return the results directories (the device is the caller's choice)."""
    record = run_record()
    task = load_task(args.task)
    kind = task.KIND
    config = task.load_task_config(task.TASK_YAML, data_root=args.data_root)
    settings = resolve_settings(config["finetune"], random_init=args.random_init,
                                overrides={name: getattr(args, name) for name in OVERRIDES},
                                allowed_poolings=config["embedding"]["poolings"])
    device = torch.device(args.device)
    pooling = settings["pooling"]
    eval_seed = int(config["embedding"]["seed"])
    train_split, val_split, test_splits = split_roles(config)
    resolved = resolve_model(args.model, root=args.outputs_root)
    model_id = resolved.model_id + (RANDOM_INIT_SUFFIX if args.random_init else "")
    protocol, split_kwargs, split_tag, representation = "finetune", {}, None, None
    if args.task == "lba":
        if config["representation"] != "joint":
            raise ValueError("LBA fine-tuning supports representation=joint only")
        split_tag, representation = args.split_tag or config["primary_split_tag"], "joint"
        split_kwargs = {"representation": representation}
        protocol = f"finetune.{split_tag}.{representation}"
        source_dir = task.processed_dir(config, split_tag)
    else:
        source_dir = task.processed_dir(config)
    seeds = list(args.seeds or settings["seeds"])
    directories = {seed: results_dir(args.task, model_id, protocol, seed, pooling=pooling, root=args.outputs_root)
                   for seed in seeds}
    finished = [str(d) for d in directories.values() if (d / "metrics.json").exists()]
    if finished:  # checked before any training, so a later seed never fails after hours of work
        raise FileExistsError(f"refusing to overwrite finished seeds: {finished}")
    if any(getattr(args, name) is not None for name in OVERRIDES):
        print("note: CLI overrides change the task.yaml settings; ds-run reports these seeds as trained with other "
              "settings. Use a separate --outputs-root for trial runs.", flush=True)
    names = (train_split, val_split, *test_splits)
    splits = {name: task.ProcessedSplit(source_dir / f"{name}.pt", **split_kwargs) for name in names}
    processed = {name: {"path": str(source_dir / f"{name}.pt"), "sha256": sha256_file(source_dir / f"{name}.pt")}
                 for name in names}
    train_labels = np.asarray(splits[train_split].labels)
    subsets = (task.test_subsets(config, splits[test_splits[0]].chains)
               if kind in ("multilabel", *MASKED_KINDS) else {})
    scale = target_scale(train_labels) if kind in SCALED_KINDS else None
    groups = ({name: task.group_labels(splits[name].chains) for name in (val_split, *test_splits)}
              if kind == "ranking_regression" else {})
    metric = config.get("primary_metric")
    out_dim = output_dim(kind, config.get("label_count"), train_labels)
    outputs = []

    for seed in seeds:
        out = new_results_dir(args.task, model_id, protocol, seed, pooling=pooling, root=args.outputs_root)
        with open(out / "run.log", "w") as log_handle:
            def log(message, handle=log_handle):
                print(message, file=handle, flush=True)
                print(message, flush=True)

            torch.manual_seed(seed)
            model = mind_encoder.load_mind_encoder(str(resolved.ckpt_path), device,
                                                   random_init=args.random_init, init_seed=seed,
                                                   domains=mind_encoder.geometry_domains(config.get("modality")))
            mean, std = feature_statistics(split_features(model, splits[train_split], device, seed=eval_seed,
                                                          settings=settings))
            head = StandardizedHead(build_head(settings["head"], int(mean.shape[0]), out_dim,
                                               hidden_dim=settings["mlp_hidden"], dropout=settings["mlp_dropout"]),
                                    mean, std).to(device)
            log(f"{args.task} {protocol} {model_id} seed {seed}: encoder {settings['encoder_init']}, lr_encoder "
                f"{settings['lr_encoder']}, lr_head {settings['lr_head']}, grad_clip_norm {settings['grad_clip_norm']}, "
                f"epochs {settings['epochs']}, patience {settings['patience']}, batch <= "
                f"{settings['max_atoms_per_batch']} nodes / {settings['max_samples_per_batch']} samples")
            best = fit(model, head, splits, train_split=train_split, val_split=val_split, device=device,
                       settings=settings, seed=seed, eval_seed=eval_seed, kind=kind, scale=scale, metric=metric,
                       log=log, val_groups=groups.get(val_split))
            scores = {name: split_scores(model, head, splits[name], device, seed=eval_seed, settings=settings,
                                         kind=kind, scale=scale)
                      for name in (val_split, *test_splits)}
            test_labels = {name: np.asarray(splits[name].labels) for name in test_splits}
            test_scores = {name: scores[name] for name in test_splits}
            metrics_val, metrics_test, headline = evaluate_task(
                kind, config, train_labels=train_labels, val_labels=np.asarray(splits[val_split].labels),
                val_scores=scores[val_split], test_labels=test_labels, test_scores=test_scores, subsets=subsets,
                val_groups=groups.get(val_split),
                test_groups={name: groups[name] for name in test_splits} if groups else None)
            metrics = {
                "task": args.task, "kind": kind, "model_id": model_id, "protocol": protocol, "pooling": pooling,
                "seed": seed, "best_epoch": best["epoch"], "split_tag": split_tag, "representation": representation,
                "val": metrics_val, "test": metrics_test,
            }
            if scale is not None:
                metrics["target_standardization"] = {"mean": scale[0].tolist(), "std": scale[1].tolist(),
                                                     "primary_metric": metric}
            save_predictions(out / "predictions.npz", kind, test_chains={n: splits[n].chains for n in test_splits},
                             test_labels=test_labels, test_scores=test_scores, val_scores=scores[val_split])
            torch.save({"state_dict": best["head"], "head": settings["head"], "in_dim": int(mean.shape[0]),
                        "feature_mean": mean, "feature_std": std,
                        **({"target_mean": scale[0], "target_std": scale[1]} if scale is not None else {})},
                       out / "head.pt")
            torch.save(best["encoder"], out / "encoder.pt")
            (out / "config.json").write_text(json.dumps({
                "args": {k: str(v) for k, v in vars(args).items()}, "finetune": settings,
                "snapshot": resolved.model_id, "sha256": resolved.card["sha256"], "git": git_commit(),
                "eval_seed": eval_seed, "run": record, "processed": processed,
            }, indent=1))
            write_text_atomic(out / "metrics.json", json.dumps(metrics, indent=1))  # last: marks the seed finished
            log(f"seed {seed}: {headline} (best epoch {best['epoch']}) -> {out}")
        outputs.append(out)
    return outputs


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    check_args(parser, args)
    if torch.device(args.device).type != "cuda":
        parser.error("fine-tuning needs CUDA (flash_varlen attention has no CPU kernel)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before the first CUDA call (core.deterministic)
    torch.set_num_threads(args.threads)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-end fine-tuning of a MIND encoder plus head (training workload; user-run via Make only).

    python -m downstream.finetune --task ec  --model current_best --device cuda:0 [--random-init]
    python -m downstream.finetune --task lba --model current_best --device cuda:0 [--split-tag identity30]

Training samples get fresh random long-range edges every epoch (seed, epoch),
as in pretraining; validation/test use the fixed embedding seed. EC windows of a
cropped chain are separate training examples carrying the chain labels; at
evaluation their pooled embeddings are residue-weighted per chain before the
head. Early stopping on the validation metric: Fmax (maximized) for EC, RMSE
(minimized) for LBA. Results go to
``outputs/downstream/<task>/<model_id>/<protocol>.<pooling>/seed<k>/``.

LBA fine-tuning supports the ``joint`` representation only; ``separate`` needs
two encoder passes per sample and is a probe-only ablation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from downstream.common import encoder as mind_encoder
from downstream.common.heads import HEADS, build_head
from downstream.common.metrics import evaluate, evaluate_regression
from downstream.common.registry import RANDOM_INIT_SUFFIX, git_commit, new_results_dir, resolve_model
from downstream.embed import embed_split_mind
from downstream.probe import build_loss, evaluate_with_cutoffs, validation_score
from downstream.tasks import TASKS, load_task


@torch.no_grad()
def chain_scores(model, head, split, device, *, seed, pooling, max_atoms, kind="multilabel") -> np.ndarray:
    model.eval()
    head.eval()
    features = embed_split_mind(model, split, device, seed=seed, poolings=[pooling], max_atoms=max_atoms)[pooling]
    output = head(torch.from_numpy(features).to(device))
    if kind == "multilabel":
        return torch.sigmoid(output).float().cpu().numpy()
    return output.squeeze(-1).float().cpu().numpy()


def train_epoch(model, head, split, optimizer, device, *, epoch_seed, pooling, max_atoms,
                loss_fn, kind="multilabel"):
    model.train()
    head.train()
    targets_all = torch.from_numpy(np.asarray(split.labels)).float()
    order = torch.randperm(len(split), generator=torch.Generator().manual_seed(epoch_seed)).tolist()
    batch, atoms, total, seen = [], 0, 0.0, 0

    def step(indices):
        samples = [split.runtime(i, model.config, seed=epoch_seed) for i in indices]
        graph = mind_encoder.collate(samples).to(device)
        node_embeddings, pool_mask = mind_encoder.encode(model, graph)
        pooled = mind_encoder.pool(node_embeddings, graph, pool_mask, [pooling])[pooling]
        targets = targets_all[split.sample_chain[indices]].to(device)
        output = head(pooled)
        loss = (loss_fn(output.squeeze(-1), targets.reshape(-1)) if kind == "regression"
                else loss_fn(output, targets))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss)

    for index in order:
        size = int(split.payload["samples"][index].z.numel())
        if batch and atoms + size > max_atoms:
            total += step(batch) * len(batch)
            seen += len(batch)
            batch, atoms = [], 0
        batch.append(index)
        atoms += size
    if batch:
        total += step(batch) * len(batch)
        seen += len(batch)
    return total / max(seen, 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--model", required=True, help="snapshot name or alias")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--outputs-root", type=Path, default=None)
    parser.add_argument("--split-tag", default=None, help="LBA only: identity30 (default) | identity60")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for torch; the machine is shared with training runs")
    parser.add_argument("--lr-encoder", type=float, default=None,
                        help="override finetune.lr_encoder; a random-init encoder needs a much larger "
                             "value than the pretrained one (task.yaml's default suits a pretrained encoder)")
    parser.add_argument("--epochs", type=int, default=None, help="override finetune.epochs")
    parser.add_argument("--patience", type=int, default=None, help="override finetune.patience")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    for name in ("lr_encoder", "epochs", "patience"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    torch.set_num_threads(args.threads)
    task = load_task(args.task)
    kind = task.KIND
    if args.task != "lba" and args.split_tag:
        parser.error("--split-tag applies to the lba task only")
    config = task.load_task_config(task.TASK_YAML, data_root=args.data_root)
    settings = dict(config["finetune"])
    # CLI overrides are recorded in config.json below, so a run always states the
    # values it actually used rather than the file's defaults.
    for name in ("lr_encoder", "epochs", "patience"):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("fine-tuning needs CUDA (flash_varlen attention)")
    if settings["head"] not in HEADS:
        raise ValueError(f"finetune.head must be one of {HEADS}")
    pooling, max_atoms = settings["pooling"], int(settings["max_atoms_per_batch"])
    eval_seed = int(config["embedding"]["seed"])
    train_split, val_split, test_split = config["splits"]
    resolved = resolve_model(args.model, root=args.outputs_root)
    model_id = resolved.model_id + (RANDOM_INIT_SUFFIX if args.random_init else "")
    protocol = "finetune"
    split_kwargs = {}
    if args.task == "lba":
        split_tag = args.split_tag or config["primary_split_tag"]
        if config["representation"] != "joint":
            raise ValueError("LBA fine-tuning supports representation=joint only")
        split_kwargs = {"representation": "joint"}
        protocol = f"finetune.{split_tag}.joint"
        source_dir = task.processed_dir(config, split_tag)
    else:
        source_dir = task.processed_dir(config)
    splits = {name: task.ProcessedSplit(source_dir / f"{name}.pt", **split_kwargs)
              for name in (train_split, val_split, test_split)}
    subsets = task.test_subsets(config, splits[test_split].chains) if kind == "multilabel" else {}
    out_dim = 1 if kind == "regression" else int(config["label_count"])
    loss_fn = build_loss(kind, settings)

    for seed in args.seeds or settings["seeds"]:
        out = new_results_dir(args.task, model_id, protocol, seed, pooling=pooling, root=args.outputs_root)
        log_handle = open(out / "run.log", "w")

        def log(message):
            print(message, file=log_handle, flush=True)
            print(message, flush=True)

        torch.manual_seed(seed)
        model = mind_encoder.load_mind_encoder(str(resolved.ckpt_path), device,
                                               random_init=args.random_init, init_seed=seed)
        probe_sample = splits[train_split].runtime(0, model.config, seed=eval_seed)
        dim = mind_encoder.embed_batch(model, [probe_sample], device, [pooling])[pooling].shape[1]
        head = build_head(settings["head"], dim, out_dim,
                          hidden_dim=settings["mlp_hidden"], dropout=settings["mlp_dropout"]).to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "lr": settings["lr_encoder"]},
            {"params": head.parameters(), "lr": settings["lr_head"]},
        ], weight_decay=settings["weight_decay"])
        best = {"score": -float("inf"), "epoch": -1}
        for epoch in range(settings["epochs"]):
            loss = train_epoch(model, head, splits[train_split], optimizer, device,
                               epoch_seed=seed * 100003 + epoch, pooling=pooling, max_atoms=max_atoms,
                               loss_fn=loss_fn, kind=kind)
            val_scores = chain_scores(model, head, splits[val_split], device, seed=eval_seed,
                                      pooling=pooling, max_atoms=max_atoms, kind=kind)
            score = validation_score(kind, np.asarray(splits[val_split].labels), val_scores)
            label, shown = ("val_fmax", score) if kind == "multilabel" else ("val_rmse", -score)
            log(f"epoch {epoch} train_loss {loss:.5f} {label} {shown:.4f}")
            if score > best["score"]:
                best = {"score": score, "epoch": epoch, "encoder": copy.deepcopy(model.state_dict()),
                        "head": copy.deepcopy(head.state_dict())}
            elif epoch - best["epoch"] >= settings["patience"]:
                log(f"early stop at epoch {epoch}; best epoch {best['epoch']}")
                break
        if "encoder" not in best:
            raise RuntimeError("fine-tuning produced no validated epoch")
        model.load_state_dict(best["encoder"])
        head.load_state_dict(best["head"])
        scores = {name: chain_scores(model, head, splits[name], device, seed=eval_seed, pooling=pooling,
                                     max_atoms=max_atoms, kind=kind) for name in (val_split, test_split)}
        if kind == "multilabel":
            metrics_val = evaluate(splits[val_split].labels, scores[val_split])
            metrics_test = evaluate_with_cutoffs(splits[test_split].labels, scores[test_split], subsets)
        else:
            metrics_val = evaluate_regression(np.asarray(splits[val_split].labels).reshape(-1),
                                              scores[val_split])
            metrics_test = {"all": evaluate_regression(
                np.asarray(splits[test_split].labels).reshape(-1), scores[test_split])}
        metrics = {
            "task": args.task, "kind": kind, "model_id": model_id, "protocol": protocol,
            "pooling": pooling, "seed": seed, "best_epoch": best["epoch"],
            "split_tag": args.split_tag, "val": metrics_val, "test": metrics_test,
        }
        np.savez_compressed(out / "predictions.npz", test_chains=np.array(splits[test_split].chains),
                            test_scores=scores[test_split],
                            test_labels=np.asarray(splits[test_split].labels),
                            valid_scores=scores[val_split])
        torch.save({"state_dict": best["head"], "head": settings["head"]}, out / "head.pt")
        torch.save(best["encoder"], out / "encoder.pt")
        (out / "config.json").write_text(json.dumps({
            "args": {k: str(v) for k, v in vars(args).items()}, "finetune": settings,
            "snapshot": resolved.model_id, "sha256": resolved.card["sha256"], "git": git_commit(),
            "eval_seed": eval_seed,
        }, indent=1))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

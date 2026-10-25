"""End-to-end fine-tuning of a MIND encoder plus head (training workload; user-run via Make only).

    python -m downstream.finetune --task ec --model current_best --device cuda:0 [--random-init]

Training samples get fresh random long-range edges every epoch (seed, epoch),
as in pretraining; validation/test use the fixed embedding seed. Windows of a
cropped chain are separate training examples carrying the chain labels; at
evaluation their pooled embeddings are residue-weighted per chain before the
head. Early stopping on validation Fmax; results go to
``outputs/downstream/<task>/<model_id>/finetune.<pooling>/seed<k>/``.
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
from downstream.common.metrics import evaluate, protein_fmax
from downstream.common.registry import RANDOM_INIT_SUFFIX, git_commit, new_results_dir, resolve_model
from downstream.embed import embed_split_mind
from downstream.probe import evaluate_with_cutoffs
from downstream.tasks.ec.dataset import TASK_YAML, ProcessedSplit, load_task_config, processed_dir, test_subsets

TASK_CONFIGS = {"ec": TASK_YAML}


@torch.no_grad()
def chain_scores(model, head, split, device, *, seed, pooling, max_atoms) -> np.ndarray:
    model.eval()
    head.eval()
    features = embed_split_mind(model, split, device, seed=seed, poolings=[pooling], max_atoms=max_atoms)[pooling]
    return torch.sigmoid(head(torch.from_numpy(features).to(device))).float().cpu().numpy()


def train_epoch(model, head, split, optimizer, device, *, epoch_seed, pooling, max_atoms):
    model.train()
    head.train()
    labels = torch.from_numpy(split.labels).float()
    order = torch.randperm(len(split), generator=torch.Generator().manual_seed(epoch_seed)).tolist()
    loss_fn = nn.BCEWithLogitsLoss()
    batch, atoms, total, seen = [], 0, 0.0, 0

    def step(indices):
        samples = [split.runtime(i, model.config, seed=epoch_seed) for i in indices]
        graph = mind_encoder.collate(samples).to(device)
        node_embeddings, pool_mask = mind_encoder.encode(model, graph)
        pooled = mind_encoder.pool(node_embeddings, graph, pool_mask, [pooling])[pooling]
        targets = labels[split.sample_chain[indices]].to(device)
        loss = loss_fn(head(pooled), targets)
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
    parser.add_argument("--task", required=True, choices=sorted(TASK_CONFIGS))
    parser.add_argument("--model", required=True, help="snapshot name or alias")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--outputs-root", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_task_config(TASK_CONFIGS[args.task], data_root=args.data_root)
    settings = config["finetune"]
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("fine-tuning needs CUDA (flash_varlen attention)")
    if settings["head"] not in HEADS:
        raise ValueError(f"finetune.head must be one of {HEADS}")
    pooling, max_atoms = settings["pooling"], int(settings["max_atoms_per_batch"])
    eval_seed = int(config["embedding"]["seed"])
    resolved = resolve_model(args.model, root=args.outputs_root)
    model_id = resolved.model_id + (RANDOM_INIT_SUFFIX if args.random_init else "")
    splits = {s: ProcessedSplit(processed_dir(config) / f"{s}.pt") for s in ("train", "valid", "test")}
    subsets = test_subsets(config, splits["test"].chains)

    for seed in args.seeds or settings["seeds"]:
        out = new_results_dir(args.task, model_id, "finetune", seed, pooling=pooling, root=args.outputs_root)
        log_handle = open(out / "run.log", "w")

        def log(message):
            print(message, file=log_handle, flush=True)
            print(message, flush=True)

        torch.manual_seed(seed)
        model = mind_encoder.load_mind_encoder(str(resolved.ckpt_path), device,
                                               random_init=args.random_init, init_seed=seed)
        probe_sample = splits["train"].runtime(0, model.config, seed=eval_seed)
        dim = mind_encoder.embed_batch(model, [probe_sample], device, [pooling])[pooling].shape[1]
        head = build_head(settings["head"], dim, config["label_count"],
                          hidden_dim=settings["mlp_hidden"], dropout=settings["mlp_dropout"]).to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "lr": settings["lr_encoder"]},
            {"params": head.parameters(), "lr": settings["lr_head"]},
        ], weight_decay=settings["weight_decay"])
        best = {"fmax": -1.0, "epoch": -1}
        for epoch in range(settings["epochs"]):
            loss = train_epoch(model, head, splits["train"], optimizer, device, epoch_seed=seed * 100003 + epoch,
                               pooling=pooling, max_atoms=max_atoms)
            val_scores = chain_scores(model, head, splits["valid"], device, seed=eval_seed,
                                      pooling=pooling, max_atoms=max_atoms)
            val_fmax = protein_fmax(splits["valid"].labels, val_scores)["fmax"]
            log(f"epoch {epoch} train_bce {loss:.5f} val_fmax {val_fmax:.4f}")
            if val_fmax > best["fmax"]:
                best = {"fmax": val_fmax, "epoch": epoch, "encoder": copy.deepcopy(model.state_dict()),
                        "head": copy.deepcopy(head.state_dict())}
            elif epoch - best["epoch"] >= settings["patience"]:
                log(f"early stop at epoch {epoch}; best epoch {best['epoch']}")
                break
        model.load_state_dict(best["encoder"])
        head.load_state_dict(best["head"])
        scores = {s: chain_scores(model, head, splits[s], device, seed=eval_seed, pooling=pooling,
                                  max_atoms=max_atoms) for s in ("valid", "test")}
        metrics = {
            "task": args.task, "model_id": model_id, "protocol": "finetune", "pooling": pooling, "seed": seed,
            "best_epoch": best["epoch"], "val": evaluate(splits["valid"].labels, scores["valid"]),
            "test": evaluate_with_cutoffs(splits["test"].labels, scores["test"], subsets),
        }
        np.savez_compressed(out / "predictions.npz", test_chains=np.array(splits["test"].chains),
                            test_scores=scores["test"], test_labels=splits["test"].labels,
                            valid_scores=scores["valid"])
        torch.save({"state_dict": best["head"], "head": settings["head"]}, out / "head.pt")
        torch.save(best["encoder"], out / "encoder.pt")
        (out / "config.json").write_text(json.dumps({
            "args": {k: str(v) for k, v in vars(args).items()}, "finetune": settings, "snapshot": resolved.model_id,
            "sha256": resolved.card["sha256"], "git": git_commit(), "eval_seed": eval_seed,
        }, indent=1))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

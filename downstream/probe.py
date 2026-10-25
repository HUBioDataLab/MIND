"""Train probe heads on cached chain embeddings (CPU is enough).

    python -m downstream.probe --task ec --model-id <snapshot|snapshot.random_init|esm2_...> --head linear

Per seed: features are standardized with train statistics, the head is trained
with BCE, the epoch with the best validation Fmax is kept (early stopping on
``patience``), and the test split is evaluated overall and per identity cutoff.
Results go to ``outputs/downstream/<task>/<model_id>/<head>_probe.<pooling>/seed<k>/``.
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
from downstream.common.metrics import evaluate, protein_fmax
from downstream.common.registry import embeddings_path, new_results_dir
from downstream.tasks.ec.dataset import TASK_YAML, load_task_config, test_subsets

TASK_CONFIGS = {"ec": TASK_YAML}


def load_embeddings(task: str, model_id: str, split: str, pooling: str, root=None):
    path = embeddings_path(task, model_id, split, root=root)
    with np.load(path) as data:
        key = f"emb_{pooling}"
        if key not in data:
            raise KeyError(f"{path} has no pooling {pooling!r} (has {[k for k in data if k.startswith('emb_')]})")
        return data["chains"].tolist(), data["labels"].astype(np.float32), data[key].astype(np.float32), path


def evaluate_with_cutoffs(labels: np.ndarray, scores: np.ndarray, subsets: Dict[int, np.ndarray]) -> dict:
    result = {"all": evaluate(labels, scores)}
    for cutoff, mask in sorted(subsets.items()):
        result[f"cutoff_{cutoff}"] = evaluate(labels[mask], scores[mask]) if mask.any() else None
    return result


@torch.no_grad()
def predict(head: nn.Module, features: torch.Tensor) -> np.ndarray:
    head.eval()
    return torch.sigmoid(head(features)).numpy()


def train_probe(train_x, train_y, val_x, val_y, *, head_kind, settings, seed, log):
    torch.manual_seed(seed)
    head = build_head(head_kind, train_x.shape[1], train_y.shape[1],
                      hidden_dim=settings["mlp_hidden"], dropout=settings["mlp_dropout"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    best = {"fmax": -1.0, "epoch": -1, "state": None}
    for epoch in range(settings["epochs"]):
        head.train()
        order = torch.randperm(len(train_x), generator=generator)
        total = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            rows = order[start:start + settings["batch_size"]]
            optimizer.zero_grad()
            loss = loss_fn(head(train_x[rows]), train_y[rows])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(rows)
        val_fmax = protein_fmax(val_y.numpy(), predict(head, val_x))["fmax"]
        log(f"epoch {epoch} train_bce {total / len(order):.5f} val_fmax {val_fmax:.4f}")
        if val_fmax > best["fmax"]:
            best = {"fmax": val_fmax, "epoch": epoch, "state": copy.deepcopy(head.state_dict())}
        elif epoch - best["epoch"] >= settings["patience"]:
            log(f"early stop at epoch {epoch}; best epoch {best['epoch']}")
            break
    head.load_state_dict(best["state"])
    return head, best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, choices=sorted(TASK_CONFIGS))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--head", required=True, choices=HEADS)
    parser.add_argument("--pooling", default=None, help="default: probe.pooling in task.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--outputs-root", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_task_config(TASK_CONFIGS[args.task], data_root=args.data_root)
    settings = config["probe"]
    pooling = args.pooling or settings["pooling"]
    protocol = f"{args.head}_probe"

    data = {s: load_embeddings(args.task, args.model_id, s, pooling, args.outputs_root) for s in ("train", "valid", "test")}
    mean = data["train"][2].mean(axis=0)
    std = data["train"][2].std(axis=0).clip(min=1e-6)
    tensors = {s: (torch.from_numpy((x - mean) / std), torch.from_numpy(y)) for s, (_, y, x, _) in data.items()}
    subsets = test_subsets(config, data["test"][0])
    for seed in args.seeds or settings["seeds"]:
        out = new_results_dir(args.task, args.model_id, protocol, seed, pooling=pooling, root=args.outputs_root)
        log_handle = open(out / "run.log", "w")

        def log(message):
            print(message, file=log_handle, flush=True)

        head, best = train_probe(*tensors["train"], *tensors["valid"], head_kind=args.head,
                                 settings=settings, seed=seed, log=log)
        test_scores = predict(head, tensors["test"][0])
        val_scores = predict(head, tensors["valid"][0])
        metrics = {
            "task": args.task, "model_id": args.model_id, "protocol": protocol, "pooling": pooling,
            "seed": seed, "best_epoch": best["epoch"], "val": evaluate(data["valid"][1], val_scores),
            "test": evaluate_with_cutoffs(data["test"][1], test_scores, subsets),
        }
        np.savez_compressed(out / "predictions.npz", test_chains=np.array(data["test"][0]),
                            test_scores=test_scores.astype(np.float32), test_labels=data["test"][1].astype(np.uint8),
                            valid_scores=val_scores.astype(np.float32))
        torch.save({"state_dict": head.state_dict(), "feature_mean": mean, "feature_std": std,
                    "head": args.head, "in_dim": int(mean.shape[0])}, out / "head.pt")
        provenance = {s: json.loads(Path(p).with_suffix(".json").read_text()) for s, (_, _, _, p) in data.items()}
        (out / "config.json").write_text(json.dumps({"args": vars(args) | {"data_root": str(args.data_root),
                                                     "outputs_root": str(args.outputs_root)},
                                                     "probe": settings, "embeddings": provenance}, indent=1))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        log_handle.close()
        print(f"seed {seed}: val fmax {metrics['val']['fmax']:.4f} test fmax {metrics['test']['all']['fmax']:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

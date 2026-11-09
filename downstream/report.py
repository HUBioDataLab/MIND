"""Collect downstream ``metrics.json`` files into one table (markdown + json).

    python -m downstream.report --task ec  [--outputs-root DIR]
    python -m downstream.report --task lba [--outputs-root DIR]

Rows are (model_id, protocol.pooling) with mean +- std over seeds. Literature
numbers are appended and labelled; they come from fine-tuned models under
their authors' protocols and are not directly comparable to frozen probes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from downstream.common.registry import outputs_root

LITERATURE = {
    "ec": [  # test Fmax at 95 % identity (GearNet / SaProt papers)
        ("GearNet-Edge (scratch)", 0.810),
        ("GearNet-Edge Multiview Contrast", 0.874),
        ("ESM-1b", 0.864),
        ("ESM-2 650M", 0.868),
        ("SaProt", 0.882),
    ],
    # GO test Fmax at 95 % identity; numbers copied from the task-selection notes (SaProt paper table), not re-verified
    "go_mf": [("ESM-2 650M", 0.670), ("SaProt", 0.682)],
    "go_bp": [("ESM-2 650M", 0.473), ("SaProt", 0.486)],
    "go_cc": [("ESM-2 650M", 0.470), ("SaProt", 0.479)],
    "lba": [  # ATOM3D LBA, 30 % sequence-identity split, test RMSE (lower is better)
        ("ATOM3D 3DCNN", 1.416),
        ("ATOM3D ENN", 1.568),
        ("ATOM3D GNN", 1.601),
        ("ProNet", 1.463),
    ],
}
# Single-label accuracy tasks: {task: [(model, {test split: accuracy %})]}; GearNet paper Table 2 (copied from the
# literature table of the task-selection notes), fine-tuned end-to-end, not comparable to frozen probes.
MULTICLASS_LITERATURE = {
    "fold": [
        ("GearNet-Edge-IEConv", {"test_fold": 48.3, "test_superfamily": 70.3, "test_family": 99.5}),
        ("GearNet-Edge Multiview Contrast", {"test_fold": 54.1, "test_superfamily": 80.5, "test_family": 99.9}),
        ("ESM-1b", {"test_fold": 26.8, "test_superfamily": 60.1, "test_family": 97.8}),
    ],
    "reaction": [
        ("GearNet-Edge-IEConv", {"test": 85.3}),
        ("GearNet-Edge Multiview Contrast", {"test": 87.5}),
        ("ESM-1b", {"test": 83.1}),
    ],
}
MULTICLASS_TASKS = tuple(MULTICLASS_LITERATURE)
LITERATURE_NOTE = "literature, fine-tuned, different protocol"
LBA_LITERATURE_NOTE = (
    "ATOM3D LBA 30 % split, test RMSE from the authors' own training protocols; "
    "GET and EPT report the current state of the art on this task. Not comparable "
    "to a frozen-encoder probe."
)
REGRESSION_TASKS = ("lba",)


def _stat(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}


def collect(task: str, root=None) -> list:
    base = outputs_root(root) / "downstream" / task
    groups = defaultdict(list)
    for path in sorted(base.glob("*/*/seed*/metrics.json")):
        metrics = json.loads(path.read_text())
        groups[(path.parents[2].name, path.parents[1].name)].append(metrics)
    rows = []
    for (model_id, protocol), runs in sorted(groups.items()):
        if all(run.get("kind") == "multiclass" for run in runs):
            test_splits = sorted({k for run in runs for k in run["test"]})
            rows.append({
                "model_id": model_id, "protocol": protocol, "seeds": sorted(run["seed"] for run in runs),
                "val_accuracy": _stat([run["val"]["accuracy"] for run in runs]),
                **{f"test_accuracy_{name}": _stat([run["test"][name]["accuracy"] for run in runs]) for name in test_splits},
            })
            continue
        if all(run.get("kind") == "regression" for run in runs):
            rows.append({
                "model_id": model_id, "protocol": protocol,
                "seeds": sorted(run["seed"] for run in runs),
                "val_rmse": _stat([run["val"]["rmse"] for run in runs]),
                "test_rmse": _stat([run["test"]["all"]["rmse"] for run in runs]),
                "test_pearson_r": _stat([run["test"]["all"]["pearson_r"] for run in runs]),
                "test_spearman_rho": _stat([run["test"]["all"]["spearman_rho"] for run in runs]),
                "test_mae": _stat([run["test"]["all"]["mae"] for run in runs]),
            })
            continue
        cutoffs = sorted({k for run in runs for k in run["test"] if k.startswith("cutoff_")},
                         key=lambda k: int(k.split("_")[1]))
        rows.append({
            "model_id": model_id, "protocol": protocol, "seeds": sorted(run["seed"] for run in runs),
            "val_fmax": _stat([run["val"]["fmax"] for run in runs]),
            "test_fmax": _stat([run["test"]["all"]["fmax"] for run in runs]),
            "test_pair_aupr": _stat([run["test"]["all"]["pair_aupr"] for run in runs]),
            "test_top1_accuracy": _stat([run["test"]["all"]["top1_accuracy"] for run in runs]),
            **{f"test_fmax_{c}": _stat([(run["test"].get(c) or {}).get("fmax") for run in runs]) for c in cutoffs},
        })
    return rows


def _fmt(stat):
    return "-" if stat is None else f"{stat['mean']:.3f} ± {stat['std']:.3f} (n={stat['n']})"


def regression_markdown(task: str, rows: list) -> str:
    header = ["model_id", "protocol", "val RMSE", "test RMSE", "Pearson r", "Spearman rho", "test MAE"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join([
            row["model_id"], row["protocol"], _fmt(row.get("val_rmse")), _fmt(row.get("test_rmse")),
            _fmt(row.get("test_pearson_r")), _fmt(row.get("test_spearman_rho")), _fmt(row.get("test_mae")),
        ]) + " |")
    for name, value in LITERATURE.get(task, []):
        lines.append("| " + " | ".join([name, "literature", "-", f"{value:.3f}"] + ["-"] * 3) + " |")
    return "\n".join(lines) + "\n\n" + LBA_LITERATURE_NOTE + "\n"


def multiclass_markdown(task: str, rows: list) -> str:
    splits = sorted({k[len("test_accuracy_"):] for row in rows for k in row if k.startswith("test_accuracy_")})
    literature = MULTICLASS_LITERATURE.get(task, [])
    splits = sorted(set(splits) | {k for _, values in literature for k in values})
    header = ["model_id", "protocol", "val accuracy"] + [f"test accuracy {name}" for name in splits]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join([row["model_id"], row["protocol"], _fmt(row.get("val_accuracy"))]
                                       + [_fmt(row.get(f"test_accuracy_{name}")) for name in splits]) + " |")
    for name, values in literature:
        lines.append("| " + " | ".join([name, LITERATURE_NOTE, "-"]
                                       + [(f"{values[s]:.1f} %" if s in values else "-") for s in splits]) + " |")
    return "\n".join(lines) + "\n\nAccuracies of probes are fractions (0-1); literature rows are percent.\n"


def to_markdown(task: str, rows: list) -> str:
    if task in REGRESSION_TASKS:
        return regression_markdown(task, rows)
    if task in MULTICLASS_TASKS:
        return multiclass_markdown(task, rows)
    columns = sorted({k for row in rows for k in row if k.startswith("test_fmax_cutoff")},
                     key=lambda k: int(k.rsplit("_", 1)[1]))
    header = ["model_id", "protocol", "val Fmax", "test Fmax", "pair AUPR", "top-1 acc"] + [
        f"Fmax <{c.rsplit('_', 1)[1]}%" for c in columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [row["model_id"], row["protocol"], _fmt(row["val_fmax"]), _fmt(row["test_fmax"]),
                 _fmt(row["test_pair_aupr"]), _fmt(row["test_top1_accuracy"])] + [_fmt(row.get(c)) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    for name, value in LITERATURE.get(task, []):
        lines.append("| " + " | ".join([name, LITERATURE_NOTE, "-", f"{value:.3f}"] + ["-"] * (len(header) - 4)) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--outputs-root", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = collect(args.task, args.outputs_root)
    if not rows:
        raise SystemExit(f"no metrics.json under {outputs_root(args.outputs_root) / 'downstream' / args.task}")
    base = outputs_root(args.outputs_root) / "downstream" / args.task
    markdown = to_markdown(args.task, rows)
    (base / "results.md").write_text(markdown)
    (base / "results.json").write_text(json.dumps({
        "rows": rows,
        "literature": [
            {"model": n, ("test_rmse" if args.task in REGRESSION_TASKS else "test_fmax"): v,
             "note": LBA_LITERATURE_NOTE if args.task in REGRESSION_TASKS else LITERATURE_NOTE}
            for n, v in LITERATURE.get(args.task, [])
        ] + [{"model": n, "test_accuracy_percent": v, "note": LITERATURE_NOTE}
             for n, v in MULTICLASS_LITERATURE.get(args.task, [])],
    }, indent=1))
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

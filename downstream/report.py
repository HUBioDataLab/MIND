"""Collect downstream ``metrics.json`` files into one table (markdown + json).

    python -m downstream.report --task ec [--outputs-root DIR]

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
}
LITERATURE_NOTE = "literature, fine-tuned, different protocol"


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


def to_markdown(task: str, rows: list) -> str:
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
        "literature": [{"model": n, "test_fmax": v, "note": LITERATURE_NOTE} for n, v in LITERATURE.get(args.task, [])],
    }, indent=1))
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

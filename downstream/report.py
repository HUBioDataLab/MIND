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
MASKED_KINDS = ("multilabel_masked", "multitarget_regression")
MOLNET_NOTE = (
    "MoleculeNet (Uni-Mol LMDB scaffold split, first conformer, H dropped). Frozen-encoder probes; no literature rows: "
    "Uni-Mol's published numbers are fully fine-tuned, conformer-averaged and 3-seed, so they are not comparable and "
    "are not entered here. Classification: mean over targets, per-target masking of missing labels; targets with a "
    "single class in the split are skipped and counted. Regression: errors in original units, averaged over targets. "
    "Subset rows (flat conformers, rare elements, ...) are reported in addition to, never instead of, the full test set."
)
# Fold has three test sets; GearNet's table adds an "Avg." column over them.
FOLD_TEST_SPLITS = ("test_fold", "test_superfamily", "test_family")


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
        if all(run.get("kind") in MASKED_KINDS for run in runs):
            rows.append(_masked_row(model_id, protocol, runs))
            continue
        if all(run.get("kind") == "multiclass" for run in runs):
            test_splits = sorted({k for run in runs for k in run["test"]})
            row = {
                "model_id": model_id, "protocol": protocol, "seeds": sorted(run["seed"] for run in runs),
                "val_accuracy": _stat([run["val"]["accuracy"] for run in runs]),
                **{f"test_accuracy_{name}": _stat([run["test"][name]["accuracy"] for run in runs]) for name in test_splits},
                **{f"test_balanced_accuracy_{name}": _stat([run["test"][name].get("balanced_accuracy") for run in runs])
                   for name in test_splits},
            }
            if task == "fold":
                row.update(_fold_average_stats(model_id, protocol, runs))
            rows.append(row)
            continue
        if all(run.get("kind") == "regression" for run in runs):
            undefined = defaultdict(int)
            for run in runs:
                reason = run["test"]["all"].get("correlation_undefined_reason")
                if reason is not None:
                    undefined[reason] += 1
            rows.append({
                "model_id": model_id, "protocol": protocol,
                "seeds": sorted(run["seed"] for run in runs),
                "val_rmse": _stat([run["val"]["rmse"] for run in runs]),
                "test_rmse": _stat([run["test"]["all"]["rmse"] for run in runs]),
                "test_pearson_r": _stat([run["test"]["all"]["pearson_r"] for run in runs]),
                "test_spearman_rho": _stat([run["test"]["all"]["spearman_rho"] for run in runs]),
                "test_mae": _stat([run["test"]["all"]["mae"] for run in runs]),
                "test_correlation_undefined": dict(undefined),
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
            "test_macro_aupr": _stat([run["test"]["all"].get("macro_aupr") for run in runs]),
            "test_macro_aupr_num_terms_skipped": _stat(
                [run["test"]["all"].get("macro_aupr_num_terms_skipped") for run in runs]),
        })
    return rows


def _masked_row(model_id: str, protocol: str, runs: list) -> dict:
    """Row of a MoleculeNet probe (``multilabel_masked`` or ``multitarget_regression``) over its seeds."""
    kinds = {run["kind"] for run in runs}
    if len(kinds) != 1:
        raise ValueError(f"{model_id}/{protocol}: runs mix the kinds {sorted(kinds)}")
    kind = kinds.pop()
    fields = ("roc_auc", "prc_auc") if kind == "multilabel_masked" else ("rmse", "mae")
    row = {"model_id": model_id, "protocol": protocol, "kind": kind, "seeds": sorted(run["seed"] for run in runs),
           **{f"val_{f}": _stat([run["val"][f] for run in runs]) for f in fields},
           **{f"test_{f}": _stat([run["test"]["all"][f] for run in runs]) for f in fields}}
    if kind == "multilabel_masked":
        row["test_num_tasks_evaluated"] = _stat([run["test"]["all"]["num_tasks_evaluated"] for run in runs])
        row["test_num_tasks_skipped"] = _stat([run["test"]["all"]["num_tasks_skipped"] for run in runs])
    names = sorted({name for run in runs for name in run["test"].get("subsets", {})})
    row["test_subsets"] = {}
    for name in names:
        entries = [run["test"]["subsets"][name] for run in runs if name in run["test"]["subsets"]]
        row["test_subsets"][name] = {
            "molecules": entries[0]["molecules"],
            "undefined_seeds": sum("undefined" in entry for entry in entries),
            **{f: _stat([entry.get(f) for entry in entries]) for f in fields},
        }
    return row


def masked_markdown(rows: list) -> str:
    classification = rows[0]["kind"] == "multilabel_masked"
    header = (["model_id", "protocol", "val ROC-AUC", "test ROC-AUC", "test PRC-AUC", "tasks evaluated", "tasks skipped"]
              if classification else ["model_id", "protocol", "val RMSE", "val MAE", "test RMSE", "test MAE"])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = ([row["model_id"], row["protocol"], _fmt(row["val_roc_auc"]), _fmt(row["test_roc_auc"]),
                  _fmt(row["test_prc_auc"]), _fmt(row["test_num_tasks_evaluated"]), _fmt(row["test_num_tasks_skipped"])]
                 if classification else
                 [row["model_id"], row["protocol"], _fmt(row["val_rmse"]), _fmt(row["val_mae"]),
                  _fmt(row["test_rmse"]), _fmt(row["test_mae"])])
        lines.append("| " + " | ".join(cells) + " |")
    subset_lines = []
    for row in rows:
        for name, entry in sorted(row["test_subsets"].items()):
            values = ", ".join(f"{f} {_fmt(entry[f])}" for f in (("roc_auc", "prc_auc") if classification else ("rmse", "mae")))
            subset_lines.append(f"- {row['model_id']} / {row['protocol']} / {name} ({entry['molecules']} test molecules): "
                                f"{values}; undefined in {entry['undefined_seeds']} of {len(row['seeds'])} seeds")
    return "\n".join(lines) + "\n\n" + MOLNET_NOTE + "\n" + ("\nFlagged test subsets:\n" + "\n".join(subset_lines) + "\n"
                                                           if subset_lines else "")


def _fold_average_stats(model_id: str, protocol: str, runs: list) -> dict:
    """Per-seed mean over the three Fold test sets (GearNet's Avg. column); old runs without balanced accuracy give None."""
    accuracy_means, balanced_means = [], []
    for run in runs:
        missing = [name for name in FOLD_TEST_SPLITS if name not in run["test"]]
        if missing:
            raise ValueError(f"{model_id}/{protocol} seed {run['seed']}: Fold run lacks test splits {missing}; "
                             "the Avg. over the three test sets is undefined")
        accuracy_means.append(float(np.mean([run["test"][n]["accuracy"] for n in FOLD_TEST_SPLITS])))
        balanced = [run["test"][n].get("balanced_accuracy") for n in FOLD_TEST_SPLITS]
        balanced_means.append(None if any(b is None for b in balanced) else float(np.mean(balanced)))
    return {"test_accuracy_avg": _stat(accuracy_means), "test_balanced_accuracy_avg": _stat(balanced_means)}


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
    notes = []
    for row in rows:
        for reason, count in sorted(row.get("test_correlation_undefined", {}).items()):
            notes.append(f"Correlation undefined ({reason}) in {count} of {len(row['seeds'])} seeds for "
                         f"{row['model_id']} / {row['protocol']}; RMSE and MAE are still reported.")
    return "\n".join(lines) + "\n\n" + LBA_LITERATURE_NOTE + "\n" + "".join(f"\n{note}\n" for note in notes)


def multiclass_markdown(task: str, rows: list) -> str:
    splits = sorted({k[len("test_accuracy_"):] for row in rows for k in row
                     if k.startswith("test_accuracy_") and k != "test_accuracy_avg"})
    literature = MULTICLASS_LITERATURE.get(task, [])
    splits = sorted(set(splits) | {k for _, values in literature for k in values})
    with_average = task == "fold"
    header = (["model_id", "protocol", "val accuracy"] + [f"test accuracy {name}" for name in splits]
              + [f"test balanced accuracy {name}" for name in splits])
    if with_average:
        header += ["test accuracy avg", "test balanced accuracy avg"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = ([row["model_id"], row["protocol"], _fmt(row.get("val_accuracy"))]
                 + [_fmt(row.get(f"test_accuracy_{name}")) for name in splits]
                 + [_fmt(row.get(f"test_balanced_accuracy_{name}")) for name in splits])
        if with_average:
            cells += [_fmt(row.get("test_accuracy_avg")), _fmt(row.get("test_balanced_accuracy_avg"))]
        lines.append("| " + " | ".join(cells) + " |")
    for name, values in literature:
        cells = ([name, LITERATURE_NOTE, "-"] + [(f"{values[s]:.1f} %" if s in values else "-") for s in splits]
                 + ["-"] * len(splits))
        if with_average:
            cells += [f"{np.mean(list(values.values())):.1f} %" if set(values) == set(FOLD_TEST_SPLITS) else "-", "-"]
        lines.append("| " + " | ".join(cells) + " |")
    notes = ("Accuracies of probes are fractions (0-1); literature rows are percent. Accuracy is the share of correctly "
             "classified proteins; balanced accuracy is the mean per-class recall over the classes present in that test "
             "split.")
    if with_average:
        notes += (" Avg. is the plain mean of the three test-set values per seed (the literature Avg. is derived the same "
                  "way). GearNet's 'mean accuracy' is not shown to be either the plain or the balanced accuracy, so both "
                  "are reported.")
    return "\n".join(lines) + "\n\n" + notes + "\n"


def to_markdown(task: str, rows: list) -> str:
    if rows and rows[0].get("kind") in MASKED_KINDS:
        return masked_markdown(rows)
    if task in REGRESSION_TASKS:
        return regression_markdown(task, rows)
    if task in MULTICLASS_TASKS:
        return multiclass_markdown(task, rows)
    columns = sorted({k for row in rows for k in row if k.startswith("test_fmax_cutoff")},
                     key=lambda k: int(k.rsplit("_", 1)[1]))
    header = ["model_id", "protocol", "val Fmax", "test Fmax", "pair AUPR", "top-1 acc"] + [
        f"Fmax <{c.rsplit('_', 1)[1]}%" for c in columns] + ["macro AUPR", "macro AUPR skipped terms"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [row["model_id"], row["protocol"], _fmt(row["val_fmax"]), _fmt(row["test_fmax"]),
                 _fmt(row["test_pair_aupr"]), _fmt(row["test_top1_accuracy"])] + [_fmt(row.get(c)) for c in columns] + [
                     _fmt(row.get("test_macro_aupr")), _fmt(row.get("test_macro_aupr_num_terms_skipped"))]
        lines.append("| " + " | ".join(cells) + " |")
    for name, value in LITERATURE.get(task, []):
        lines.append("| " + " | ".join([name, LITERATURE_NOTE, "-", f"{value:.3f}"] + ["-"] * (len(header) - 4)) + " |")
    return ("\n".join(lines) + "\n\npair AUPR: one curve over all (protein, term) pairs (GearNet). macro AUPR: one curve per "
            "term, averaged over terms (DeepFRI); terms without a true protein are skipped and counted.\n")


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

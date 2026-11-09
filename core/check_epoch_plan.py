"""Build a config's training batch plan for several epochs from chunk metadata only.

A fixed ``epoch_num_batches`` raises at the start of any epoch whose natural
greedy plan needs more batches, which would stop a long run mid-way. This check
draws the natural plan for the first epochs (no GPU, no sample loading) and
exits non-zero when the margin is not positive.

    python -m core.check_epoch_plan --config-yaml-path core/pretraining_config_protein_only.yaml --epochs 5
"""

import argparse
import json
import statistics
import sys

from core.pretraining_config import check_yaml_keys, create_pretraining_config, load_config_yaml
from core.train_pretrain import _build_train_batch_sampler, load_universal_dataset, split_dataset


def natural_plan_counts(config, epochs: int) -> tuple[int | None, list[dict]]:
    multi_domain = getattr(config, "multi_domain", None) or {}
    is_multi_domain = bool(multi_domain.get("enabled", False))
    dataset = load_universal_dataset(config, config.dataset, config.dataset_download_dir)
    train_dataset, _, _ = split_dataset(dataset, config.seed)
    sampler = _build_train_batch_sampler(
        train_dataset,
        config,
        config.max_atoms_per_batch,
        is_multi_domain,
        getattr(config, "use_improved_sampler", True),
    )
    if sampler.epoch_atom_budget is None:
        raise ValueError("config has no multi_domain.atom_sampling.epoch_atom_budget; nothing to check")
    fixed = sampler.epoch_num_batches
    sampler.epoch_num_batches = None
    rows = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        plan = sampler._get_batch_plan()
        stats = sampler.plan_stats
        rows.append({
            "epoch": epoch,
            "natural_batches": len(plan),
            "actual_total_atoms": stats.get("actual_total_atoms"),
            "mean_atoms_per_batch": stats.get("actual_total_atoms", 0) / max(len(plan), 1),
        })
    return fixed, rows


def plan_summary_line(summary: dict, ok: bool) -> str:
    """One greppable line; the Makefile target prints only this one to the terminal."""
    return (
        f"PLAN_SUMMARY: config={summary['config']} seed={summary['seed']} "
        f"epoch_num_batches={summary['epoch_num_batches']} natural_max={summary['natural_max']} "
        f"margin={summary['margin_batches']} status={'OK' if ok else 'FAIL'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-yaml-path", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    config_dict = load_config_yaml(args.config_yaml_path)
    check_yaml_keys(config_dict, args.config_yaml_path)
    if args.seed is not None:
        config_dict["seed"] = args.seed
    config = create_pretraining_config(**config_dict)
    fixed, rows = natural_plan_counts(config, args.epochs)
    counts = [row["natural_batches"] for row in rows]
    summary = {
        "config": args.config_yaml_path,
        "seed": config.seed,
        "max_atoms_per_batch": config.max_atoms_per_batch,
        "epoch_num_batches": fixed,
        "natural_min": min(counts),
        "natural_max": max(counts),
        "natural_stdev": statistics.pstdev(counts),
        "margin_batches": None if fixed is None else fixed - max(counts),
        "epochs": rows,
    }
    print(json.dumps(summary, indent=2))
    ok = fixed is None or max(counts) <= fixed
    print(plan_summary_line(summary, ok))
    if not ok:
        print(f"FAIL: natural plan needs up to {max(counts)} batches, above epoch_num_batches={fixed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

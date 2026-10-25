"""Frozen model snapshots and the downstream results layout (all under gitignored ``outputs/``).

Layout (root = ``$MIND_OUTPUTS_ROOT`` or ``<repo>/outputs``)::

    pretrain/<group>/<run_name>/                       new pretraining runs (existing run dirs stay as-is)
    models/<name>/{model.ckpt,card.json,hparams.json}  read-only snapshot used downstream
    models/<alias> -> <name>                           e.g. current_best (relative symlink)
    downstream/<task>/<model_id>/embeddings/<split>.npz
    downstream/<task>/<model_id>/<protocol>.<pooling>/seed<k>/{metrics.json,predictions.npz,head.pt,run.log,config.json}

A snapshot only reads its source: the checkpoint is streamed into a new
directory, hashed, re-hashed after copying, and refused if the source changed
size or mtime meanwhile (a live run rewriting ``last.ckpt``). Existing
snapshots are never overwritten. ``model_id`` in results is always the
concrete snapshot name, so repointing an alias never relabels old results.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAIN_GROUPS = ("multidomain", "protein_only", "rna_only", "unimol_only", "coconut_only")
PROTOCOLS = ("linear_probe", "mlp_probe", "finetune")
RANDOM_INIT_SUFFIX = ".random_init"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def outputs_root(root: Optional[Path] = None) -> Path:
    if root is not None:
        return Path(root)
    return Path(os.environ.get("MIND_OUTPUTS_ROOT", REPO_ROOT / "outputs"))


def _check_name(kind: str, value: str) -> str:
    if not isinstance(value, str) or not _NAME.match(value):
        raise ValueError(f"invalid {kind} {value!r}: use letters, digits, '.', '_' or '-'")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: Path = REPO_ROOT) -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot read git commit of {repo}: {exc}") from exc
    return f"{sha}+dirty" if dirty else sha


def pretrain_run_dir(group: str, run_name: str, root: Optional[Path] = None) -> Path:
    if group not in PRETRAIN_GROUPS:
        raise ValueError(f"pretrain group must be one of {PRETRAIN_GROUPS}, got {group!r}")
    return outputs_root(root) / "pretrain" / group / _check_name("run name", run_name)


def _training_provenance(src: Path) -> dict:
    """Git commit and CLI args recorded by W&B next to the checkpoint, if any."""
    records = {}
    for path in sorted(glob.glob(str(src.parent / "wandb" / "run-*" / "files" / "wandb-metadata.json"))):
        with open(path) as handle:
            meta = json.load(handle)
        records[path] = {"git": meta.get("git"), "args": meta.get("args"), "program": meta.get("program")}
    commits = sorted({str((r["git"] or {}).get("commit")) for r in records.values()})
    return {"wandb_runs": records, "training_git_commits": commits}


def _checkpoint_summary(src: Path) -> dict:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    config = checkpoint["hyper_parameters"]["config"]
    datasets = (config.get("multi_domain") or {}).get("datasets") or {}
    domains = sorted(name for name, entry in datasets.items() if (entry or {}).get("enabled"))
    val_metrics = {}
    for key, state in (checkpoint.get("callbacks") or {}).items():
        if key.startswith("ModelCheckpoint") and isinstance(state, dict):
            for field in ("monitor", "best_model_score", "current_score", "best_model_path"):
                value = state.get(field)
                val_metrics[field] = float(value) if torch.is_tensor(value) else value
    return {
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "pytorch_lightning_version": checkpoint.get("pytorch-lightning_version"),
        "domains": domains,
        "val_metrics": val_metrics,
        "hparams": config,
    }


def snapshot_model(
    src_ckpt: Path, name: str, *, root: Optional[Path] = None,
    config_path: Optional[Path] = None, notes: str = "",
) -> Path:
    """Copy ``src_ckpt`` into ``models/<name>/`` with a card; never touches the source."""
    src = Path(src_ckpt).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"checkpoint not found: {src}")
    models = outputs_root(root) / "models"
    dest = models / _check_name("snapshot name", name)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f"refusing to overwrite snapshot {dest}")
    before = src.stat()
    summary = _checkpoint_summary(src)
    partial = models / f".{name}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    digest = hashlib.sha256()
    with open(src, "rb") as reader, open(partial / "model.ckpt", "wb") as writer:
        for block in iter(lambda: reader.read(16 * 1024 * 1024), b""):
            digest.update(block)
            writer.write(block)
    after = src.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        shutil.rmtree(partial)
        raise RuntimeError(f"{src} changed while copying (live run?); snapshot aborted")
    sha = digest.hexdigest()
    if sha256_file(partial / "model.ckpt") != sha:
        shutil.rmtree(partial)
        raise RuntimeError(f"copy of {src} does not match its sha256")
    os.chmod(partial / "model.ckpt", 0o444)
    with open(partial / "hparams.json", "w") as handle:
        json.dump(summary.pop("hparams"), handle, indent=1, sort_keys=True, default=str)
    card = {
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(src),
        "source_size": before.st_size,
        "source_mtime_ns": before.st_mtime_ns,
        "sha256": sha,
        **summary,
        "config_path": str(config_path) if config_path else None,
        "config_sha256": sha256_file(Path(config_path)) if config_path else None,
        "snapshot_repo_commit": git_commit(),
        **_training_provenance(src),
        "notes": notes,
    }
    with open(partial / "card.json", "w") as handle:
        json.dump(card, handle, indent=1)
    os.replace(partial, dest)
    return dest


def set_alias(alias: str, target: str, *, root: Optional[Path] = None, repoint: bool = False) -> Path:
    """Point ``models/<alias>`` at snapshot ``target`` (relative symlink)."""
    models = outputs_root(root) / "models"
    link = models / _check_name("alias", alias)
    if not (models / _check_name("snapshot name", target) / "card.json").is_file():
        raise FileNotFoundError(f"no snapshot named {target!r} in {models}")
    if link.exists() or link.is_symlink():
        if not link.is_symlink():
            raise FileExistsError(f"{link} is a real directory, not an alias")
        if not repoint:
            raise FileExistsError(f"alias {alias!r} exists ({os.readlink(link)}); pass repoint=True")
        link.unlink()
    link.symlink_to(target)
    return link


@dataclass(frozen=True)
class ResolvedModel:
    model_id: str
    ckpt_path: Path
    card: dict
    requested: str


def resolve_model(ref: str, *, root: Optional[Path] = None, verify: bool = True) -> ResolvedModel:
    """Resolve a snapshot name or alias; the checkpoint hash must match its card."""
    entry = outputs_root(root) / "models" / _check_name("model reference", ref)
    if not entry.exists():
        raise FileNotFoundError(f"no snapshot or alias {ref!r} under {entry.parent}")
    directory = entry.resolve()
    with open(directory / "card.json") as handle:
        card = json.load(handle)
    ckpt = directory / "model.ckpt"
    if verify and sha256_file(ckpt) != card["sha256"]:
        raise RuntimeError(f"{ckpt} does not match the sha256 in its card")
    return ResolvedModel(directory.name, ckpt, card, ref)


def embeddings_path(task: str, model_id: str, split: str, *, root: Optional[Path] = None) -> Path:
    return (outputs_root(root) / "downstream" / _check_name("task", task) / _check_name("model id", model_id)
            / "embeddings" / f"{_check_name('split', split)}.npz")


def results_dir(
    task: str, model_id: str, protocol: str, seed: int, *, pooling: str, root: Optional[Path] = None
) -> Path:
    """``<protocol>.<pooling>/seed<k>``; the pooling suffix keeps pooling variants apart."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    return (outputs_root(root) / "downstream" / _check_name("task", task) / _check_name("model id", model_id)
            / f"{protocol}.{_check_name('pooling', pooling)}" / f"seed{int(seed)}")


def new_results_dir(
    task: str, model_id: str, protocol: str, seed: int, *, pooling: str, root: Optional[Path] = None
) -> Path:
    """Create a results dir; refuses one that already holds metrics."""
    path = results_dir(task, model_id, protocol, seed, pooling=pooling, root=root)
    if (path / "metrics.json").exists():
        raise FileExistsError(f"refusing to overwrite results in {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Snapshot a checkpoint or point an alias at a snapshot.")
    parser.add_argument("--outputs-root", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("src_ckpt", type=Path)
    snap.add_argument("name")
    snap.add_argument("--config-path", type=Path, default=None)
    snap.add_argument("--notes", default="")
    alias = commands.add_parser("alias")
    alias.add_argument("alias")
    alias.add_argument("target")
    alias.add_argument("--repoint", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        dest = snapshot_model(args.src_ckpt, args.name, root=args.outputs_root,
                              config_path=args.config_path, notes=args.notes)
        print(json.dumps(json.loads((dest / "card.json").read_text()), indent=1))
    else:
        print(set_alias(args.alias, args.target, root=args.outputs_root, repoint=args.repoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

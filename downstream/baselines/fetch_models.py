"""Download the pinned baseline models of ``registry.yaml`` into MIND_MODELS_ROOT.

Hugging Face repos go to ``<root>/huggingface/hub`` (the HF cache layout, so
``from_pretrained(..., cache_dir=<root>/huggingface/hub, local_files_only=True)``
finds them); URL files go to ``<root>/other/<name>/`` and are md5-checked.
``<root>/MANIFEST.json`` records the revision, license and sha256 of every file.
A failed model is reported and the command exits 1.

    python -m downstream.baselines.fetch_models --models esm2_t33_650M_UR50D nafm
    python -m downstream.baselines.fetch_models --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

REGISTRY = Path(__file__).with_name("registry.yaml")
DEFAULT_ROOT = Path(os.environ.get("MIND_MODELS_ROOT", "/home/yusuf/models"))


def load_registry(path: Path = REGISTRY) -> dict:
    models = yaml.safe_load(path.read_text())["models"]
    for name, spec in models.items():
        if spec["kind"] == "hf" and not spec.get("revision"):
            raise ValueError(f"{name}: Hugging Face models must pin a revision")
        if spec["kind"] == "url" and not spec.get("md5"):
            raise ValueError(f"{name}: URL models must give an md5")
        if spec["kind"] not in ("hf", "url"):
            raise ValueError(f"{name}: unknown kind {spec['kind']!r}")
    return models


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_hf(spec: dict, root: Path) -> list[Path]:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        spec["repo"], revision=spec["revision"], cache_dir=str(root / "huggingface" / "hub"),
        allow_patterns=spec.get("allow"), max_workers=4,
    ))
    files = sorted(p for p in snapshot.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"{spec['repo']}@{spec['revision']}: no files matched {spec.get('allow')}")
    return files


def fetch_url(name: str, spec: dict, root: Path) -> list[Path]:
    target = root / "other" / name / spec["url"].rsplit("/", 1)[-1]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        urllib.request.urlretrieve(spec["url"], partial)
        partial.rename(target)
    md5 = _digest(target, "md5")
    if md5 != spec["md5"]:
        raise RuntimeError(f"{target}: md5 {md5} != registry {spec['md5']}")
    return [target]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", nargs="+")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)

    registry = load_registry()
    names = list(registry) if args.all else args.models
    unknown = sorted(set(names) - set(registry))
    if unknown:
        parser.error(f"not in {REGISTRY.name}: {unknown}")

    manifest_path = args.root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    failures = {}
    for name in names:
        spec = registry[name]
        print(f"[{name}] {spec.get('repo') or spec['url']}", flush=True)
        try:
            files = fetch_hf(spec, args.root) if spec["kind"] == "hf" else fetch_url(name, spec, args.root)
        except Exception as error:  # recorded and reported; the command exits 1
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"[{name}] FAILED {failures[name]}", flush=True)
            continue
        manifest[name] = {
            "source": spec.get("repo") or spec["url"], "revision": spec.get("revision"),
            "license": spec["license"], "modality": spec["modality"],
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "files": {str(p.relative_to(args.root)): {"bytes": p.stat().st_size, "sha256": _digest(p, "sha256")}
                      for p in files},
        }
        size = sum(f["bytes"] for f in manifest[name]["files"].values())
        print(f"[{name}] ok, {len(files)} files, {size / 1e9:.2f} GB", flush=True)
        args.root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    if failures:
        print(json.dumps({"failed": failures}, indent=1), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

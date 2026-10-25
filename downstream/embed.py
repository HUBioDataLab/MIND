"""Extract and cache chain embeddings for a downstream task (GPU workload; user-run via Make).

    python -m downstream.embed --task ec --model current_best --device cuda:0
    python -m downstream.embed --task ec --model current_best --random-init --device cuda:0
    python -m downstream.embed --task ec --model esm2_t12_35M_UR50D --hf-cache-dir DIR --device cuda:0

Writes ``outputs/downstream/<task>/<model_id>/embeddings/<split>.npz`` (chains,
labels, one ``emb_<pooling>`` array per pooling) plus a ``.json`` provenance
record; existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from core.deterministic import enable_deterministic_algorithms
from downstream.baselines import esm2
from downstream.common import encoder as mind_encoder
from downstream.common.registry import (
    RANDOM_INIT_SUFFIX, embeddings_path, git_commit, resolve_model, sha256_file,
)
from downstream.tasks.ec.dataset import TASK_YAML, ProcessedSplit, load_task_config, processed_dir

TASK_CONFIGS = {"ec": TASK_YAML}


def batches_by_atoms(split: ProcessedSplit, model_config, *, seed: int, max_atoms: int):
    """Yield ``(indices, samples)`` in stored order with at most ``max_atoms`` nodes per batch."""
    indices, samples, atoms = [], [], 0
    for index, sample in split.iter_runtime(model_config, seed=seed):
        if samples and atoms + sample.num_nodes > max_atoms:
            yield indices, samples
            indices, samples, atoms = [], [], 0
        indices.append(index)
        samples.append(sample)
        atoms += sample.num_nodes
    if samples:
        yield indices, samples


def embed_split_mind(model, split: ProcessedSplit, device, *, seed: int, poolings: Sequence[str],
                     max_atoms: int) -> Dict[str, np.ndarray]:
    """Chain embeddings ``{pooling: [num_chains, D]}`` for one processed split."""
    per_sample = {mode: [None] * len(split) for mode in poolings}
    for indices, samples in batches_by_atoms(split, model.config, seed=seed, max_atoms=max_atoms):
        pooled = mind_encoder.embed_batch(model, samples, device, poolings)
        for mode, values in pooled.items():
            for row, index in enumerate(indices):
                per_sample[mode][index] = values[row].numpy()
    return {mode: split.chain_embeddings(np.stack(rows)) for mode, rows in per_sample.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, choices=sorted(TASK_CONFIGS))
    parser.add_argument("--model", required=True, help="snapshot name/alias, or an ESM-2 name")
    parser.add_argument("--random-init", action="store_true", help="same architecture, untrained weights")
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--outputs-root", type=Path, default=None)
    parser.add_argument("--hf-cache-dir", default=None, help="required for ESM-2")
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--esm-batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for torch; the machine is shared with training runs")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    torch.set_num_threads(args.threads)

    config = load_task_config(TASK_CONFIGS[args.task], data_root=args.data_root)
    device = torch.device(args.device)
    if device.type == "cuda":
        enable_deterministic_algorithms()
    seed = int(config["embedding"]["seed"])
    is_esm = args.model in esm2.ESM2_MODELS
    provenance = {"task": args.task, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "git": git_commit(), "device": str(device), "seed": seed,
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if is_esm:
        if args.random_init:
            parser.error("--random-init applies to MIND snapshots only")
        if not args.hf_cache_dir:
            parser.error("--hf-cache-dir is required for ESM-2 (the default HF cache is not writable)")
        tokenizer, model = esm2.load_esm2(args.model, device, cache_dir=args.hf_cache_dir,
                                          local_files_only=args.hf_local_files_only)
        model_id, poolings = args.model, [esm2.POOLING]
        provenance.update(model=esm2.ESM2_MODELS[args.model],
                          hf_revision=getattr(model.config, "_commit_hash", None))
    else:
        if device.type != "cuda":
            parser.error("the MIND encoder needs CUDA (flash_varlen attention has no CPU kernel)")
        resolved = resolve_model(args.model, root=args.outputs_root)
        model = mind_encoder.load_mind_encoder(str(resolved.ckpt_path), device,
                                               random_init=args.random_init, init_seed=args.init_seed)
        model_id = resolved.model_id
        if args.random_init:
            model_id += RANDOM_INIT_SUFFIX + (f"_s{args.init_seed}" if args.init_seed else "")
        poolings = list(config["embedding"]["poolings"])
        provenance.update(model_ref=args.model, snapshot=resolved.model_id, sha256=resolved.card["sha256"],
                          random_init=args.random_init, init_seed=args.init_seed)

    for split_name in args.splits or config["splits"]:
        out = embeddings_path(args.task, model_id, split_name, root=args.outputs_root)
        if out.exists():
            raise FileExistsError(f"refusing to overwrite {out}")
        source = processed_dir(config) / f"{split_name}.pt"
        split = ProcessedSplit(source)
        started = time.monotonic()
        if is_esm:
            arrays = {esm2.POOLING: esm2.embed_sequences(tokenizer, model, split.sequences, device,
                                                       batch_size=args.esm_batch_size)}
        else:
            arrays = embed_split_mind(model, split, device, seed=seed, poolings=poolings,
                                      max_atoms=int(config["embedding"]["max_atoms_per_batch"]))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, chains=np.array(split.chains), labels=split.labels,
                 **{f"emb_{mode}": value for mode, value in arrays.items()})
        record = dict(provenance, split=split_name, model_id=model_id, processed=str(source),
                      processed_sha256=sha256_file(source), chains=len(split.chains),
                      poolings=sorted(arrays), dim=int(next(iter(arrays.values())).shape[1]),
                      seconds=round(time.monotonic() - started, 1))
        out.with_suffix(".json").write_text(json.dumps(record, indent=1))
        print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

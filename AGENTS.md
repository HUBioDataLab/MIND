# Repository Guidelines

## Project Overview

MIND (Molecular INtelligence through Domains) is a multi-domain molecular
foundation-model project. A shared ESA graph transformer learns masked atom
types and distance bins from protein, RNA, and UniMol graphs.

Current contracts:

- Protein caches have 4 backbone + 10 sidechain slots; RNA has 12 backbone +
  11 base slots. V is padding target id 120, not a physical runtime atom.
- Geometry-neutral encoding removes visible V and all masked sidechain/base
  nodes from ESA; post-ESA latent slots read their own backbone context.
- Protein/RNA MLM masks complete blocks; UniMol uses atom-level masking.
- Protein/RNA chunks store no edges: protein Tier-1 CA edges are regenerated at load time and
  every other edge is rebuilt at runtime. Configs that need stored edges are refused loudly.
- Invariant features are enabled; the optional equivariant branch is disabled.
  M layers are graph-local and S layers are molecule-global.
- Protein hybrid edges combine CA sequence, real spatial, and dynamic random
  long-range connections; they are not a complete chemical-bond graph.
- Full-pass sampling is requested, but the current Lightning epoch limit can
  truncate longer rebuilt plans. See the current handoff before claiming full
  data coverage.

## Project Structure & Module Organization

- `core/`: training entrypoint, model/encoder/tasks, configs and metrics.
- `data_loading/`: adapters, fixed-slot conversion, chunk metadata, loading and sampling.
- `esa/`: ESA backbone and attention kernels.
- `tests/`: production-contract, geometry, attention and sampler regression tests.
- `Docs/`: canonical English documentation; start at [Docs/README.md](Docs/README.md).
- `.agents/skills/`: the shared source for MIND skills; `.claude/skills/` links
  to these folders. `CLAUDE.md` imports this file for Claude Code.
- `GET/`, `papers/`, `template/`: vendor/reference material, not current operating instructions.
- `logs/`, `outputs/`: local run evidence and checkpoints, outside Git.
- `Makefile`: user interface for training and supported local workflows.

## Data Locations

Datasets live outside Git. `chunk_dir` is often a prefix whose sibling
`_chunk_*` directories exist; it need not be a literal directory.

Protein and RNA chunks are edge-free fixed-slot chunks (`mind_pt_schema`, built by
`data_loading/build_edge_free_chunks.py`); the legacy edge-stored chunks were deleted on
9 October 2026 after the equivalence checks in the
[scaling plan](Docs/training/FULL_DATA_SCALING_PLAN_20260923.md) and
[CURRENT_TODOS.md](CURRENT_TODOS.md).

| Dataset | Current Atum base / source |
|---|---|
| Protein 1M (1,117,868; train/val/test) | `/home/yusuf/data/proteins/processed_14atom_1m_edgefree_{train,val,test}`, split by structure-ID hash |
| Protein 40k baseline | `/home/yusuf/data/proteins/processed_14atom_edgefree`, chunks 0–9 |
| Raw 40k / 1M proteins | `/home/yusuf/data/proteins/raw_structures_hq_40k/`, `/home/yusuf/data/protein_1m/raw_structures_1m_1024/` |
| Protein manifests | `/home/yusuf/data/proteins/afdb_clusters/`, `/home/yusuf/data/protein_1m/afdb_clusters/` |
| Full RNA corpus (12,869; split by PDB entry) | `/home/yusuf/data/rna/processed_23slot_full_edgefree_{train,val,test}` |
| June-equivalent RNA (baseline) | `/home/yusuf/data/rna/virtual_v1/june_equivalent_23slot_v2/processed_23slot_edgefree` |
| Fast RNA prefix | `/home/yusuf/data/rna_small/processed_23slot_5k_edgefree` |
| UniMol prefix | `/home/yusuf/data/small_molecules/unimol/universal_unimol`; 200 available chunks; the logged full baseline loader slice is 0, 1, 10, 100–106 because discovery is lexicographic |
| Metabolite / QM9 | COCONUT is at `/home/yusuf/data/metabolites/processed`; it remains disabled because mixed-domain coordination is defective. QM9 is unavailable. |

See [migration inventory](Docs/operations/NON_GIT_INVENTORY.md) for sizes,
all raw/derived data, user-local tools, temporary evidence and symlinks.

## Build, Test, and Development Commands

Use the working `mind_esa3d` environment. For Atum, export the actual environment
as described in [migration](Docs/operations/MIGRATION.md); `GET/env.yml` is an
older vendor environment and does not reproduce the current setup.

```bash
make help
make test-unit
make train CONFIG=f LOG=fast_check
make train GPU=6 CONFIG=m LOG=multidomain_run
make train GPU=auto CONFIG=f EPOCHS=3 SEED=42 LOG=smoke
make status
make logs
```

Training commands are user-executed. Fast duration depends on hardware/data;
do not promise ten minutes. Logs go to `logs/<MonthYear>/`, outputs to a
run-specific directory or explicit `OUTPUT_DIR`.

`make build-edge-free` has no default paths and refuses to overwrite a chunk; it still
writes data, so run it only on an explicit user request with a new destination prefix.

## Execution Authority and Process Verification

These rules are mandatory for every Codex session in this repository:

- Training, training-style validation, smoke training, and full benchmark workloads
  remain user-controlled and must be exposed through a Make target. Codex must not
  start, stop, resume, or background those workloads.
- After giving a training-related Make command, Codex must ask whether the user ran it
  and wait for confirmation. Only after confirmation may Codex monitor the workload.
- Data audits, data processing, dataset conversion, and immutable derived-dataset
  generation do not require Make targets. Codex may run them directly only when the
  user explicitly requests execution. The exact command, inputs, output directory,
  log path, and result must be reported; generated data remains outside Git.
- Codex may directly run small, bounded tests that it wrote or focused existing unit
  tests, as well as read-only inspection and monitoring commands.
- Codex must never infer that a workload stopped from a remembered PID, a stale PID,
  a sandbox-local process list, or one unchanged log snapshot.
- To verify a user-started workload, Codex must inspect the host process table using
  the full command/output-directory identity, map the live PID to the GPU with
  `nvidia-smi`, and check that the intended log's size/mtime or progress advances
  across two observations. A process in `D` state is still alive and requires log/I/O
  checks; it must not be reported as stopped merely because GPU utilization is 0%.
- If host-level process visibility is unavailable, Codex must say that status is
  unverified and request permission for the read-only host check. It must not report
  the process as stopped.
- Monitoring must distinguish the launcher shell PID, Python worker PID, GPU PID,
  output directory, and log path. Conclusions must be based on the Python worker and
  current log progress, not only the launcher.
- Generated data and long-run outputs remain outside Git unless explicitly requested.

Developer references:
```bash
python -m core.train_pretrain --config-yaml-path core/pretraining_config_fast.yaml
pytest -v tests/test_adapters.py
```

Training commands above remain developer references only and must be exposed through
a Make target when the user will run them. Explicitly requested data-processing
commands may be run directly under the rules above.

## Known Issues & Active Work

The September 1–12 full run completed 100 epochs. Protein real-sidechain accuracy
is 85.77%; RNA real-base accuracy is 93.04% at final validation, not held-out test.

Open issues:

- Rare protein chemistry and late-slot real/V confusion remain, particularly
  CYS, HIS, GLN and TRP. The old geometry-driven CB collapse is not current.
- Mask token `0` is incorrectly reused as the RBF padding mask for retained
  real backbone nodes.
- Batch-wide explicit-bond precedence can zero protein/RNA chemical-coordination
  features in batches containing bonded small molecules.
- Lightning can retain the initial batch limit while sampler plan length changes;
  epoch 99 consumed 19,081 of 19,122 planned batches.
- The invariant atom-MLM objective and validation results do not establish
  coordinate generation or downstream foundation-model quality.
- Metabolite/QM9 remain disabled; enabling another domain requires its own
  data-contract and mixed-batch verification.

Read [the evidence and next steps](Docs/training/STATUS_AND_NEXT_STEPS_20260912.md)
before changing the model. Do not restart superseded CB-stacking, synthetic
template or element-balanced-loss plans from archived notes.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and descriptive `snake_case` names for modules, functions, variables, and config keys. Keep YAML config names explicit and prefixed by task/domain (e.g., `pretraining_config_protein.yaml`). Follow existing local patterns before adding abstractions. Prefer existing config objects, and keep data paths in YAML or Make targets.

## Testing Guidelines

Use focused checked-in tests for each change and inspect skips. Adapter tests:
`pytest -v tests/test_adapters.py`; the previous standalone adapter validator
was removed. Geometry, attention, mixed-domain and sampler tests live in
`tests/`. `make test-unit` exposes pytest through the working environment.

Training changes require user-started fast and matched multidomain checks; see
[validation](Docs/training/VALIDATION.md). Data-processing checks require explicit
execution authorization, a separate output prefix, and provenance records.

Record exact command, config, dataset selection, seed, expected change, result
and skipped checks. Sampler-only tests do not prove trainer consumption, and
single-domain tests do not prove mixed-batch feature independence.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries:
- `Fix RNA leakage in protein per-atom metric tables`
- `Speed up .pkl to .pt conversion: vectorize edge construction`

Keep commits focused. Never mix config experiments with code fixes. Do not write Claude as co-author.

## Security & Configuration Tips

Do not commit generated datasets, model checkpoints, W&B logs, local `.log` files (they live in `logs/` which is gitignored), or machine-specific absolute paths unless they are intentionally part of a documented Make target. Keep credentials and API tokens out of configs and scripts.

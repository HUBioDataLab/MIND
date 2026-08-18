# Repository Guidelines

## Project Overview

MIND (Molecular INtelligence through Domains) is a multi-domain molecular foundation model pre-trained on proteins, RNA, and small molecules simultaneously. It uses a shared graph transformer (ESA backbone) with Masked Language Modeling (MLM) and distance prediction tasks. Think of it like BERT but for 3D molecular graphs — each atom is a token, and the model learns chemistry by predicting masked atoms in their 3D context.

Key design choices:
- **14-atom uniform protein representation**: Every residue gets exactly 4 backbone + 10 sidechain slots. Empty sidechain slots are filled with virtual 'V' atoms placed at CB position (prevents sequence leakage from atom count).
- **Block-MLM for proteins**: Entire residues (all 14 atoms) are masked together, not random atoms.
- **Cross-modal batching**: Proteins, RNA, and small molecules appear in the same batch, with atom-aware sampling to balance domain contributions.
- **Hybrid edges (3-tier)**: Tier 1 = sequence neighbors (i±k), Tier 2 = spatial within 5Å, Tier 3 = random long-range.

## Project Structure & Module Organization

```
MIND/
├── core/                          # Training code
│   ├── train_pretrain.py          # Main training entry point
│   ├── metrics/                   # Per-atom metric callbacks
│   │   ├── protein_per_atom_metrics_callback.py
│   │   └── rna_per_atom_metrics_callback.py
│   ├── pretraining_config_fast.yaml       # Fast sanity-check config (~10 min)
│   └── pretraining_config_multidomain.yaml  # Main training config
│
├── data_loading/                  # Data pipeline
│   ├── adapters/                  # Per-domain raw→UniversalMolecule converters
│   │   ├── protein_adapter.py     # PDB/CIF → 14-atom uniform repr
│   │   ├── rna_adapter.py
│   │   └── unimol_adapter.py
│   ├── cache_to_pyg.py            # UniversalMolecule → PyG graph (.pt chunks)
│   ├── process_chunked_dataset.py # Orchestrates chunked processing
│   ├── create_chunk_metadata.py   # Generates metadata.json for each chunk dir
│   └── validate_adapters.py       # Quick smoke-test for adapters
│
├── esa/                           # ESA graph transformer backbone + layers
├── GET/                           # Imported GET model tree + conda env file
├── data/                          # Data acquisition scripts (not raw data)
├── Docs/                          # Design notes and pipeline docs
│   ├── CB_DUSUKLUGU_ACIKLAMA.md   # CB recall collapse explanation (Turkish)
│   ├── MIND_OVERVIEW.md           # Interview-ready project overview
│   ├── features/                  # Feature-specific design docs
│   │   ├── HYBRID_EDGE_CONSTRUCTION.md
│   │   ├── DYNAMIC_RANDOM_EDGES.md
│   │   └── ATOM_TYPES.md
│   └── proteins/
│       └── PROTEIN_REPRESENTATION_PIPELINE.md
│
├── logs/                          # Training logs (organized by month)
│   ├── June2026/
│   ├── May2026/
│   └── April2026/
│
├── Makefile                       # Primary interface for training + processing
└── AGENTS.md                      # This file
```

## Data Locations

All raw and processed data lives **outside** the repo:

| Dataset | Path | Notes |
|---------|------|-------|
| Proteins (processed .pt chunks) | `/home/yusuf/data/proteins/processed_14atom/` | 10 chunks, 14-atom uniform repr |
| Proteins (raw PDB/CIF) | `/home/yusuf/data/proteins/raw_structures_hq_40k/` | ~40k HQ structures |
| Proteins (manifest) | `/home/yusuf/data/proteins/afdb_clusters/manifest_hq_40k_len512.csv` | Filters to ≤512 residue chains |
| RNA (processed .pt chunks) | `/home/yusuf/data/rna/processed_graphs/` | Ready to use |
| UniMol small molecules | `/home/arin/8TB_arin/unimol_pyg_full/universal_unimol/` | 10+ chunks, on Arin's disk |
| Metabolites | `/media/ubuntu/8TB/ege/processed/` | 497 chunks; currently disabled (bond_edge_index collision in PyG collate) |

## Build, Test, and Development Commands

Create the expected environment with:

```bash
conda env create -f GET/env.yml
conda activate mind_esa3d
```

Use `make help` to list supported workflows.

```bash
make train CONFIG=f LOG=my_test       # Fast sanity check (~10 min), saves to logs/June2026/my_testfast_<datetime>.log
make train CONFIG=m LOG=my_run        # Full multidomain training
make train GPU=6 CONFIG=m             # Force specific GPU
make train GPU=auto CONFIG=m          # Auto-pick GPU with most free VRAM (≥15 GB)
make process-protein NUM_CHUNKS=10    # Process protein data (sequential)
make process-protein PARALLEL=2       # Process in 2 parallel groups
make status                           # Show training processes + GPU usage
make logs                             # List 10 most recent log files
```

Log files are auto-routed to `logs/<MonthYear>/<name>_<YYYYMMDD_HHMMSS>.log` for multidomain or `logs/<MonthYear>/<name>fast_<YYYYMMDD_HHMMSS>.log` for fast runs.

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
python data_loading/validate_adapters.py --adapter protein --max_samples 5
```

Training commands above remain developer references only and must be exposed through
a Make target when the user will run them. Explicitly requested data-processing
commands may be run directly under the rules above.

## Known Issues & Active Work

**CB recall collapse (~0.017–0.28)**: Virtual atoms (9 per GLY, 9 per ALA, …) are all placed at CB's 3D coordinate. After masking removes type signal, the model learns "this position ≈ V" and also predicts CB as V. TRP is the only exception (10 real sidechain atoms, no virtuals, CB is correctly recalled). Tried 2Å offset fix — failed (corrupted edge graph geometry). Next step: evaluate RNA-style backbone stacking (virtuals on CA instead of CB), then BoltzGen if that fails.

**Metabolite disabled**: `bond_edge_index` key in coconut chunks collides with PyG's cross-modal batcher. Keep `metabolite: enabled: false` in configs until resolved.

**Epoch imbalance**: When protein/RNA data exhausts during an epoch, late batches become UniMol-only. Fix planned: remove per-domain epoch boundary, instead interleave from start.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and descriptive `snake_case` names for modules, functions, variables, and config keys. Keep YAML config names explicit and prefixed by task/domain (e.g., `pretraining_config_protein.yaml`). Follow existing local patterns before adding abstractions. Prefer existing config objects, and keep data paths in YAML or Make targets.

## Testing Guidelines

There is no dedicated checked-in unit test suite. Validate changes with:

1. **Adapter changes**: `python data_loading/validate_adapters.py --adapter <name> --max_samples 5`
2. **Training changes**: `make train CONFIG=f LOG=<feature_name>` — check log metrics (see `.claude/commands/validate_feature.md` or `.codex/validate_feature.md` for the full checklist)
3. **Pipeline changes**: `make process-protein NUM_CHUNKS=1` then spot-check the `.pt` output

Record the command, config, dataset subset, and any expected metric/log change in your PR.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries:
- `Fix RNA leakage in protein per-atom metric tables`
- `Speed up .pkl to .pt conversion: vectorize edge construction`

Keep commits focused. Never mix config experiments with code fixes. Do not write Claude as co-author.

## Security & Configuration Tips

Do not commit generated datasets, model checkpoints, W&B logs, local `.log` files (they live in `logs/` which is gitignored), or machine-specific absolute paths unless they are intentionally part of a documented Make target. Keep credentials and API tokens out of configs and scripts.

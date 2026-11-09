# MIND training guide

Training is started through the Makefile (`make train`). Run `make help` for the
full target list.

## Commands

```bash
make train CONFIG=f LOG=fast_check
make train CONFIG=m LOG=multidomain_run
make train GPU=6 CONFIG=m LOG=multidomain_run
make train GPU=auto CONFIG=f LOG=fast_check
make train GPU=auto CONFIG=pretraining_config_multidomain_1m_coconut EPOCHS=100 SEED=42 LOG=md_1m_coconut_100ep
make test-unit
make status
make logs
```

`CONFIG=f` selects [`core/pretraining_config_fast.yaml`](../../core/pretraining_config_fast.yaml)
and `CONFIG=m` selects [`core/pretraining_config_multidomain.yaml`](../../core/pretraining_config_multidomain.yaml).
Any other value is a file name in `core/` without `.yaml`
(`CONFIG=pretraining_config_fast_1m`).

Optional variables (see the header of the [Makefile](../../Makefile)):

| Variable | Effect |
|---|---|
| `GPU` | GPU index or `auto` (the GPU with the most free memory, at least `MIN_VRAM_MB`, default 15000) |
| `LOG` | log name prefix; the file is `logs/<MonthYear>/<LOG>_<datetime>.log` (fast configs add `fast`) |
| `EPOCHS` | overrides `max_epochs` of the config (`--max-epochs`) |
| `SEED` | overrides the config seed (`--seed`) |
| `CKPT` | resumes full checkpoint state (`--ckpt-path`); combine with a larger `EPOCHS` |
| `MAX_ATOMS` | overrides `max_atoms_per_batch` (`--max-atoms-per-batch`) |
| `OUTPUT_DIR` | checkpoint directory; default `outputs/<LOG>_<datetime>`, unique per run |

The Make target chooses and prints the GPU, log path, output directory, config,
epoch override, and seed, then starts the run in the background. `make status`
and `make logs` are read-only inspection commands. The Makefile activates the
conda environment given by `MIND_CONDA_ENV` (default: the authors' environment);
override `MIND_CONDA_ENV`, `CONDA_ROOT` and `MIND_HF_HOME` on another machine.

## Configuration files

`core/` holds 13 YAML files. The six multi-domain files share one model and task
setup (7 layers, width 256, 16 heads, `pretraining_tasks` long-range distance,
short-range distance and MLM, `precision: bfloat16`, `xformers_or_torch_attn:
flash_varlen`) and differ in data and sampling. Protein and RNA chunks are
edge-free fixed-slot chunks; UniMol and COCONUT chunks store radius-graph edges.

| YAML | Domains enabled | Data selection | Sampling and loader |
|---|---|---|---|
| `pretraining_config_multidomain.yaml` (`CONFIG=m`) | protein (40k), UniMol, RNA (June-equivalent); metabolite, QM9, DNA off | protein chunks 0-9, UniMol chunks 0-9, all RNA chunks; one pooled random 80/10/10 split | natural full pass (`atom_sampling.enabled: false`), `max_atoms_per_batch: 6000`, `num_workers: 0`, 100 epochs |
| `pretraining_config_fast.yaml` (`CONFIG=f`) | same three domains | chunk 0 of each domain, 5k-structure RNA | budgeted epoch (700k atoms, 140 batches), 6,000 atoms, 3 epochs |
| `pretraining_config_multidomain_1m.yaml` | protein (1M), full RNA corpus, UniMol; COCONUT off | fixed `splits:` for all three (UniMol train chunks 0-197, val 198, test 199); `oversized_graphs: skip` for protein and RNA | budgeted virtual epochs (100M atoms, 6,400 batches, weights 65/15/20), 18,000 atoms, `num_workers: 0`, 100 epochs |
| `pretraining_config_multidomain_1m_coconut.yaml` | **four domains**: protein (1M), RNA, UniMol, COCONUT | as above plus COCONUT chunks 0-494 train, 495 val, 496 test | budgeted virtual epochs (weights 55/15/20/10), 18,000 atoms, `num_workers: 8`, `esa_packed_rows: true`, 100 epochs |
| `pretraining_config_fast_1m.yaml` | protein (1M), RNA, UniMol | one chunk per protein split, RNA train chunks 0-1, UniMol chunk 0 | budgeted (2.4M atoms, 160 batches), 18,000 atoms, 3 epochs |
| `pretraining_config_fast_1m_coconut.yaml` | four domains | as `fast_1m` plus COCONUT chunks 0-1 (random split) | budgeted (weights 55/15/20/10), 18,000 atoms, `num_workers: 8`, `esa_packed_rows: true`, 3 epochs |
| `pretraining_config_protein.yaml`, `..._protein_flash_varlen_dynamic.yaml`, `..._rna.yaml`, `..._unimol.yaml`, `..._coconut.yaml`, `..._lba.yaml`, `..._qm9.yaml` | one domain each (`dataset:` key, `multi_domain` not used) | `dataset_download_dir` with older or other-user paths | older single-dataset templates with their own model sizes (RNA width 192, UniMol/COCONUT/LBA/QM9 width 512); UniMol, COCONUT and QM9 also use `coordinate_denoising`; QM9 data is unavailable |

The dataset paths in every YAML are the authors' paths and must be edited
(see the [data pipeline](../data/DATA_PIPELINE.md)). The 10 October graph-local
coordination repair is required for the two COCONUT-enabled four-domain configs;
their learning effect still needs a matched from-scratch comparison.

## Multidomain epoch policy

`pretraining_config_multidomain.yaml` (`CONFIG=m`) enables PDB, UniMol, and
June-equivalent 23-slot RNA; QM9, metabolite, and DNA are disabled. It requests a
natural full pass (`atom_sampling.enabled: false`); the 1M and fast configs set
`atom_sampling.enabled: true` and run budgeted virtual epochs (a fixed atom budget
split by `domain_weights` and a fixed batch count), not a pass over the corpus.
Metabolite (COCONUT) is enabled only in the two four-domain configs; the other
configs keep it off.

A natural full pass is an intended policy, not proof that every run completed its
plan: the training loop can cache the initial dataloader batch count while the
sampler rebuilds a different-length plan at `set_epoch`. The sampler, the two epoch
policies, the splits and the per-domain graph-size limit are described in the
[data pipeline](../data/DATA_PIPELINE.md#loading-and-batching), together with how to
check a log (`Ready! ... batches created` against consumed batches and per-domain
totals) before calling a run full-pass.

## Model and objectives

The entry point is [`core/train_pretrain.py`](../../core/train_pretrain.py). The
encoder, the ESA backbone, the three configured tasks (long-range distance,
short-range distance, MLM; weight `1.0` each, set by `pretraining_tasks` and
`task_weights`), the block masking and the virtual-target loss weight are described in
the [model overview](../architecture/MODEL_OVERVIEW.md); the geometry-neutral graph
view (which replaced the historical `masked_slot_template` experiment, disabled in the
production configs) is in [virtual slots](../architecture/VIRTUAL_SLOTS.md).

## Logging and checkpoints

The configs use explicit W&B logging. Training prints the resolved logger before
model creation, records loss and per-domain metrics, and adds protein/RNA
per-atom callbacks. The geometry-neutral diagnostics are observational: context
counterfactuals and representation-gradient audits do not add losses.

Validation cadence comes from `val_check_interval`; inspect the resolved value
in the config and interpret old logs carefully because older sampler lengths
could make fractional cadence misleading. A run is not evidence of a healthy
training process unless logs show finite losses, advancing epochs, and consumed
loader plans. A sampler line reporting `mode=full_pass` describes the plan; it
does not establish that Lightning consumed the whole plan.

## Data and resource prerequisites

Every enabled chunk should have a matching metadata sidecar. A missing or unreadable
sidecar makes the default configs fail at start-up (atom-budget batching and
`max_graph_atoms` need it; details in the
[data pipeline](../data/DATA_PIPELINE.md#chunk-conversion-and-metadata)). The
`num_workers` value differs by config (0 for `m`, `fast`, `multidomain_1m` and
`fast_1m`; 8 for the two four-domain configs, set after a measured loader speed-up);
chunks are loaded lazily. Lower `max_atoms_per_batch` if the selected GPU cannot hold
the configured graph mix.

Some graphs carry explicit bonds (`bond_edge_index`) and others do not (protein
graphs carry none); since the 10 October repair the encoder chooses the coordination
source per graph (`has_explicit_bonds`, see the
[model overview](../architecture/MODEL_OVERVIEW.md)), so a graph never mixes the two
sources inside a mixed batch. CPU contract tests pass;
the learning effect of enabling COCONUT still needs a matched from-scratch comparison.

Before the first training run in a fresh worktree, run the focused tests
(`make test-unit`), then a short `make train CONFIG=f` run.

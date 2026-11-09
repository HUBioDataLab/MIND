# MIND — Molecular INtelligence through Domains

MIND is a self-supervised **foundation-model pretraining framework for molecules**.
One shared graph transformer is trained on **proteins, RNA and small molecules at the
same time**, so that a single encoder learns atom-level chemistry and 3D geometry across
molecular domains instead of one model per domain.

> **Status: research code, pretraining stage.** The model is pretrained and evaluated
> on its own masked-atom task. Downstream benchmarks and a coordinate-generation
> capability are *not* part of the current results. See [Results](#results) and
> [Limitations](#limitations).

## What it does

MIND represents every molecule as a 3D atom graph and trains the encoder to recover
what was hidden:

| Objective | Question the model answers |
|---|---|
| Masked atom type (MLM) | Which element is this masked atom? (121-class head) |
| Long-range distance | In which distance bin is a sampled atom pair? |
| Short-range distance | In which distance bin is a pair of atoms joined by a graph edge of at most `short_range_max_distance` (5 Å)? |

The three losses are summed with equal weight. Protein residues and RNA nucleotides
are masked **as whole blocks** (all sidechain / base atoms at once); small molecules
are masked atom by atom.

## How it works

```text
PDB / mmCIF / SDF / LMDB
   │  adapters (protein, RNA, UniMol, COCONUT, LBA, QM9)
   ▼
fixed-slot PyG graph chunks          protein: 14 slots / residue (4 backbone + 10 sidechain)
   │                                  RNA:     23 slots / nucleotide (12 backbone + 11 base)
   ▼
block / atom masking  →  geometry-neutral graph view  →  molecular encoder
                                                              │
                                        ESA backbone  M → S → M → S → M → S → P
                                                              │
                                                 MLM head · distance heads
```

- **Backbone: ESA graph transformer** ([esa/](esa/)), adapted from the Edge-Set Attention
  codebase (Buterez et al.) and applied to atom nodes. `M` layers
  attend over graph edges within one molecule, `S` layers attend globally within a
  molecule, `P` is attention pooling. Default: 7 layers, width 256, 16 heads,
  **6.3 M parameters**, FlashAttention variable-length kernels, bf16.
- **Shared fixed-slot layout.** Protein and RNA are padded to a fixed number of atom
  slots per block; padding is a dedicated virtual type `V`, so every block is a
  constant-size prediction target. Details: [data representation](Docs/data/DATA_REPRESENTATION.md).
- **Geometry-neutral masking.** Masked sidechain/base atoms and padding are removed from
  the encoder's graph, so their coordinates cannot leak the answer. Their identities are
  predicted after the encoder from the owning block's backbone context. Details:
  [virtual slots](Docs/architecture/VIRTUAL_SLOTS.md).
- **Hybrid protein edges.** Cα sequence neighbours, spatial neighbours and random long-range
  connections, rebuilt at load time, so the stored chunks contain no edges.
  Details: [hybrid edges](Docs/architecture/HYBRID_EDGES.md).
- **Invariant features.** Atom-type and period embeddings (the period is known for H, C, N, O, F, P, S and Cl) plus rotation/translation-invariant
  geometric descriptors and Gaussian radial-basis features. An optional equivariant
  branch exists but is disabled.
- **Batching.** Mixed-domain batches are packed to an atom budget (`max_atoms_per_batch`)
  rather than a graph count.

## Data

| Domain | Source | Used in the reported run |
|---|---|---|
| Protein | AlphaFold DB representatives (pLDDT-filtered) | 40k structures (32,034 train graphs) |
| RNA | PDB/RNA 3D structures | 8,556 structures (6,833 train graphs) |
| Small molecules | [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) 209M conformation set | 10 of 200 LMDB-derived chunks (686,908 train graphs) |
| Metabolites | [COCONUT](https://coconut.naturalproducts.net/) natural products | not used in the reported run (the four-domain configs enable it) |
| QM9, LBA | adapters only | not used in pretraining |

Datasets are **not** distributed with this repository. Download and preprocessing
scripts are in [data/](data/) and [data_loading/](data_loading/); the full procedure
is in the [data pipeline guide](Docs/data/DATA_PIPELINE.md). Dataset paths in the YAML
configs point to the authors' machine and must be edited.

## Installation

Requirements: Linux, an NVIDIA GPU of Ampere generation or newer (FlashAttention and
bf16), CUDA 12.8 drivers, Python 3.11. The reference setup trained with 6,000 atoms per
batch on a 32 GB GPU (larger batches were used on 96 GB cards).

```bash
conda create -n mind python=3.11 -y && conda activate mind

# 1. PyTorch (CUDA 12.8 build)
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# 2. Everything else (PyG extensions are fetched from the matching wheel index)
pip install -r requirements.txt

# 3. FlashAttention needs torch at build time
pip install flash-attn==2.8.0.post2 --no-build-isolation
```

`pip install -r requirements-dev.txt` additionally installs `pytest`. Versions are pinned
to the environment that produced the results below; other CUDA/torch combinations
need matching PyG and FlashAttention builds. `rdkit-pypi` ships wheels only up to
Python 3.11.

## Quick start

**1. Check the install (no data, no GPU needed for most tests):**

```bash
make test-unit            # or: python -m pytest -q tests --ignore=tests/test_multigpu_training.py
```

**2. Prepare data.** There is no bundled sample dataset. The smallest end-to-end path is
protein-only: fetch AlphaFold DB cluster metadata, build a manifest, download structures,
and convert them to chunks.

```bash
python data/download_protein_clusters.py
python data/protein_pipeline/1_filter_and_create_manifest.py --mode manifest \
    --metadata-file <metadata.tsv.gz> --output manifest.csv --plddt 70 --max-len 1024
python data/protein_pipeline/2_download_pdbs_from_manifest.py \
    --manifest-file manifest.csv --structures-outdir raw_structures --workers 16
make build-edge-free DOMAIN=protein MANIFEST=manifest.csv RAW_DIR=raw_structures \
    DEST_PREFIX=<new output prefix> CHUNK_SIZE=4000
```

**3. Point a config at your chunks and train.** Copy
[core/pretraining_config_fast.yaml](core/pretraining_config_fast.yaml) (a short sanity
run) to `core/my_config.yaml`, set each enabled dataset's `chunk_dir`, disable the
domains you did not prepare, and set `use_wandb: false` if you do not use Weights & Biases.

```bash
make train GPU=auto CONFIG=my_config EPOCHS=3 SEED=42 LOG=my_run
```

`CONFIG` is either a shortcut (`m` = multidomain, `f` = fast) or a file name in `core/`
without `.yaml`. `make train` selects the GPU (`GPU=auto` picks one with at least 15 GB
free), starts training in the background, writes the log to `logs/<MonthYear>/` and
checkpoints to a run-specific directory under `outputs/` (override with `OUTPUT_DIR`).
Inspect a run with `make status` and `make logs`. The `Makefile` activates the authors'
conda environment by default; override `MIND_CONDA_ENV`, `CONDA_ROOT` and `MIND_HF_HOME`
to use it elsewhere. Which YAML does what is listed in the
[training guide](Docs/training/TRAINING_GUIDE.md#configuration-files); the full four-domain
configuration (protein, RNA, Uni-Mol, COCONUT) is
[core/pretraining_config_multidomain_1m_coconut.yaml](core/pretraining_config_multidomain_1m_coconut.yaml).
Pretrained checkpoints are not published yet.

## Results

Run results are kept locally.

## Limitations

- Evaluated only on the pretraining task; no downstream benchmark is reported yet.
- Weak spots: rare sidechain chemistry — Cys, His, Gln, Trp, and Ser/Thr/Tyr hydroxyls
  (for example, Trp ring atoms are recovered far less reliably than backbone atoms).
- The model predicts atom types and distance bins. It is **not** a coordinate or structure
  generator.
- The sampler is designed for one full pass over the data per epoch, but a known
  Lightning epoch-length issue can drop a fraction of a percent of the planned batches.
- Pretraining results come from a single seed and a subset of the available graphs.
- Metabolite (COCONUT) and QM9 domains are implemented but not enabled in the default configs.

## Repository layout

```text
core/           training entry point, model, masking, tasks, configs, metrics
esa/            ESA graph-attention backbone and FlashAttention kernels
data_loading/   adapters, chunk building, samplers, PS_300 fragment tokenizer
data/           dataset download and preparation scripts
tests/          contract, geometry, attention and sampler tests
Docs/           architecture, data and training documentation
```

## Documentation

- [Model overview](Docs/architecture/MODEL_OVERVIEW.md)
- [Data pipeline](Docs/data/DATA_PIPELINE.md)
- [Data representation](Docs/data/DATA_REPRESENTATION.md)
- [Training guide](Docs/training/TRAINING_GUIDE.md)
- [All documentation](Docs/README.md)

## Acknowledgements

MIND builds on the ESA (Edge-Set Attention) codebase, uses the PS_300 fragment tokenizer from
[GET](https://github.com/THUNLP-MT/GET) (MIT, kept in
[data_loading/ps_tokenizer/](data_loading/ps_tokenizer/)), and takes inspiration for protein graph
construction from Salad. Training data comes from AlphaFold DB, the PDB, Uni-Mol and COCONUT;
please follow their licenses.

## License and citation

No license has been chosen yet; until one is added, all rights are reserved by the authors.
A paper and a citation entry are pending.

Developed in the [HUBioDataLab](https://github.com/HUBioDataLab).

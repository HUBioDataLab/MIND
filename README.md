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
| Short-range distance | In which distance bin is a bonded / near-neighbour pair? |

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
  constant-size prediction target.
- **Geometry-neutral masking.** Masked sidechain/base atoms and padding are removed from
  the encoder's graph, so their coordinates cannot leak the answer. Their identities are
  predicted after the encoder from the owning block's backbone context. Details:
  [virtual slots](Docs/architecture/VIRTUAL_SLOTS.md).
- **Hybrid protein edges.** Cα sequence neighbours, spatial neighbours and random long-range
  connections, rebuilt at load time, so the stored chunks contain no edges.
  Details: [hybrid edges](Docs/architecture/HYBRID_EDGES.md).
- **Invariant features.** Atom-type and periodic-table embeddings plus rotation/translation-invariant
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
| Metabolites | [COCONUT](https://coconut.naturalproducts.net/) natural products | implemented, disabled by default |
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
run), set each enabled dataset's `chunk_dir`, disable the domains you did not prepare,
and set `use_wandb: false` if you do not use Weights & Biases.

```bash
python -m core.train_pretrain \
    --config-yaml-path core/my_config.yaml \
    --output-dir outputs/my_run --max-epochs 3 --seed 42
```

The full four-domain configuration is
[core/pretraining_config_multidomain.yaml](core/pretraining_config_multidomain.yaml).
The `Makefile` wraps the same entry point (`make train CONFIG=f`, `make status`,
`make logs`) but contains the authors' conda path; override `MIND_CONDA_ENV` and
`CONDA_ROOT` to use it elsewhere. Training logs and checkpoints are written to
`logs/` and `outputs/`. Pretrained checkpoints are not published yet.

## Results

Final validation of the 100-epoch multi-domain run (protein + RNA + Uni-Mol, seed 42,
one run). These are **validation** numbers on masked-atom prediction, not held-out test
or downstream-task results.

| Metric | Value | Majority-slot baseline |
|---|---:|---:|
| Protein: real sidechain atom accuracy | **85.77 %** | 59.95 % |
| Protein: backbone atom accuracy | 100 % | – |
| Protein: whole-residue exact match (real atoms) | 71.22 % | – |
| RNA: real base atom accuracy | **93.04 %** | 57.72 % |
| RNA: whole-nucleotide exact match (real atoms) | 82.28 % | – |
| Validation loss — total / long-range / short-range / MLM | 0.796 / 0.406 / 0.230 / 0.160 | – |

The majority-slot baseline always predicts the most common element for a given slot.
Removing or shuffling the owning block's context drops protein sidechain accuracy by
about 22 points, so the model uses context rather than only slot priors (audit on a small
fixed sample).

An independent rerun on different hardware (batch budget 18,000 atoms, same config and
seed) reproduced the picture: protein 85.45 %, RNA 93.67 %, validation loss 0.788.
Throughput there was about 52 k atoms/s, 2.2× faster end to end than the first run.

## Limitations

- Evaluated only on the pretraining task; no downstream benchmark is reported yet.
- Weak spots: rare sidechain chemistry — Cys, His, Gln, Trp, and Ser/Thr/Tyr hydroxyls
  (for example Trp ring atoms are recovered under 11 % of the time).
- The model predicts atom types and distance bins. It is **not** a coordinate or structure
  generator.
- The sampler is designed for one full pass over the data per epoch, but a known
  Lightning epoch-length issue can drop a fraction of a percent of the planned batches.
- Reported numbers come from a single seed and a data subset (about 0.9 M of the available graphs).
- Metabolite (COCONUT) and QM9 domains are implemented but not enabled in the reported run.

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

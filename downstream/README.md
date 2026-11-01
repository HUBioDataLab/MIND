# Downstream evaluation

Frozen-encoder probes, fine-tuning, a random-init control and ESM-2 baselines
for MIND encoders.

| Task | Kind | Data | Primary metric |
|---|---|---|---|
| `ec` | multi-label, 538 EC terms | DeepFRI/GearNet split, 15,550 / 1,729 / 1,919 chains, test cutoffs 30/40/50/70/95 % | Fmax (higher) |
| `lba` | regression, pK | ATOM3D LBA (PDBbind 2019 refined, 4,463 complexes), 30 % split 3507/466/490, 60 % split 3563/448/452 | RMSE (lower) |

## Pipeline

| Step | Command | Kind |
|---|---|---|
| download | `python -m downstream.tasks.<task>.prepare download` | data processing, CPU, network |
| process | `python -m downstream.tasks.ec.prepare process --workers 16` / `...lba.prepare process --threads 8` | data processing, CPU |
| snapshot | `python -m downstream.common.registry snapshot <ckpt> <name>` and `... alias current_best <name>` | file copy, CPU |
| embed | `python -m downstream.embed --task <task> --model <snapshot> [--random-init]` | GPU inference |
| ESM-2 | `python -m downstream.embed --task ec --model esm2_t12_35M_UR50D --hf-cache-dir DIR` | GPU inference |
| probe | `python -m downstream.probe --task <task> --model-id <id> --head linear\|mlp [--pooling P]` | CPU training |
| fine-tune | `python -m downstream.finetune --task <task> --model <snapshot> [--random-init]` | GPU training |
| report | `python -m downstream.report --task <task>` | CPU |

For `lba`, `embed`, `probe` and `finetune` also take `--split-tag identity30|identity60`;
`embed` and `probe` additionally take `--representation joint|separate`. Each
combination writes its own embeddings and results directory.

Training-like steps (embed, probe, fine-tune) are user-run through Make targets.

## Contracts

- `downstream/tasks/ec/task.yaml` must name a missing-atom policy explicitly
  (`missing_atoms`); the checked-in value `UNDECIDED` stops processing.
  See `downstream/common/missing_atoms.py` for `strict`,
  `drop_residue_missing_backbone` + `mask` / `drop_chain_if_fraction_above`.
- Experimental structures never reach `ProteinAdapter` unchecked: the adapter
  pads a missing sidechain atom with V and shifts later atoms into wrong slots.
  Incomplete residues are masked like a pretraining MLM block (all 14 slots,
  mask token 0, sidechain slots latent) or the chain is dropped.
- Graphs are built with the training functions (`_standardize_residue_to_14_atoms`,
  `_universal_to_pyg`, `preposition_visible_virtuals`, edge-free encode/decode,
  `attach_runtime_template_metadata`, protein random edges under a fixed per-chain seed).
- Only pre-PMA node states are used; the PMA graph embedding is untrained.
  Pooling covers real atoms the encoder saw with their true type
  (`active_node_mask & ~mlm_mask`): `mean_atoms`, `ca`, `mean_residues`.
- Chains longer than 1024 residues are skipped and recorded (`length_policy: skip`)
  or split into contiguous windows (`crop_windows`).
- Each processed split has a per-chain QC table (`qc_<split>.tsv`) and summary
  (`qc_<split>.json`); above `qc.max_failed_chain_fraction` the split is not written
  and the command exits 1.

## Layout

Data (outside Git): `/home/yusuf/data/downstream/<task>/<modality>/{raw,processed,splits,provenance.json}`.

Models and results (under `outputs/`, or `$MIND_OUTPUTS_ROOT`):

```
models/<snapshot>/{model.ckpt,card.json,hparams.json}   read-only copy, sha256 in card
models/<alias> -> <snapshot>                            e.g. current_best
downstream/<task>/<model_id>/embeddings/<split>.{npz,json}
downstream/<task>/<model_id>/<protocol>.<pooling>/seed<k>/{metrics.json,predictions.npz,head.pt,run.log,config.json}
```

`model_id` is the snapshot name, `<snapshot>.random_init` for the untrained
control, or the ESM-2 name. Literature EC numbers in the report are fine-tuned
results under other protocols and are labelled as such.


## LBA (ATOM3D ligand binding affinity)

Target: pK = -log10(Kd or Ki) of a protein-ligand complex. Source: Zenodo record
4914718, md5-pinned in `downstream/tasks/lba/task.yaml`. The LMDB values are
gzip-compressed JSON with `to_dict(orient="split")` DataFrames, read with the
`lmdb` package only; the `atom3d` package is not a dependency. An LMDB's real
length is its `num_examples` key: `stat()["entries"]` is three too large because
of the `id_to_idx` / `num_examples` / `serialization_format` metadata keys.

### Representation

`joint` (default) is the ATOM3D setting: one graph with the pocket and the
ligand, `dataset_type="protein"` so `build_geometry_neutral_graph_view` removes
the visible V padding and rebuilds the radius graph over **every** node of the
graph. That rebuild is what produces the protein-ligand interface edges; the
stored edges are CA-only and cross none of them. Protein and ligand are told
apart by `entity_idx` (0 / 1) and by their node features: pocket atoms sit in
14-slot residue blocks and receive domain-local slot ids 1..14, while each
ligand atom is its own block and therefore receives slot id 0 ("no reliable
fixed slot"), the embedding's padding index. The encoder does **not** read
`entity_idx`; it is used by the LBA poolings and by the QC.

`separate` encodes the pocket (`dataset_type="protein"`) and the ligand
(`dataset_type="unimol"`, radius edges, `pos_code="sm"`) as two graphs and
concatenates the pooled vectors. It has no interface edges, so the gap to
`joint` measures how much the model uses the interaction. Probe-only;
fine-tuning is `joint` only.

### Contracts

- Pocket = every standard residue with at least one atom within 6 A of a ligand
  atom, kept **in full**. Hydrogens count for the contact test and are never
  stored as nodes; that reproduces ATOM3D's own `atoms_pocket` residue set
  exactly. Whole residues mean the 14-slot adapter never shifts atoms into
  wrong slots, so the cutoff itself needs no masking.
- The 600-atom cap counts real heavy atoms (pocket + ligand) and drops whole
  residues farthest-first; it never splits a residue and never truncates the
  ligand. It does not bind on this dataset (max 458 real atoms measured).
- Block ids are sequence-faithful (`chain_ordinal * 1_000_000 + residue index in
  that chain`), so Tier-1 CA edges only join real sequence neighbours and never
  cross chains.
- No explicit ligand bonds. `resolve_bond_sources` reports explicit bonds per
  *graph*, so attaching them would make `_compute_chemical_coordination` drop
  the <=1.85 A distance proxy for every pocket atom too. Pretraining has no
  explicit bonds for proteins or for UniMol, so the proxy is the in-distribution
  choice. `ligand.bonds: explicit` is refused by the config loader.
- Missing atoms use the EC policy: residues missing a backbone atom are dropped,
  a residue with an incomplete sidechain keeps its backbone and has its whole
  14-slot block masked.
- Regression, not multi-label: the head outputs one value, the loss is MSE
  (`probe.loss`, `huber` available), early stopping minimizes validation RMSE,
  and the report prints RMSE / Pearson r / Spearman rho. The shared
  `validation_score` helper returns a higher-is-better number for both kinds
  (`-RMSE` for LBA), so one early-stopping rule covers EC and LBA.

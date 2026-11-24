# Downstream evaluation

Frozen-encoder probes, fine-tuning, a random-init control and ESM-2 baselines
for MIND encoders.

| Task | Kind | Data | Primary metric |
|---|---|---|---|
| `ec` | multi-label, 538 EC terms | DeepFRI/GearNet split, 15,550 / 1,729 / 1,919 chains, test cutoffs 30/40/50/70/95 % | Fmax (higher) |
| `go_mf` / `go_bp` / `go_cc` | multi-label, 489 / 1,943 / 320 GO terms | DeepFRI nrPDB-GO split as packaged by TorchDrug (Zenodo 6622158 GeneOntology.zip), test cutoffs 30/40/50/70/95 % | Fmax (higher) |
| `fold` | multi-class, 1,195 SCOPe 1.75 folds | GearNet Fold3D (Zenodo 7593591, fold3d.zip, CC BY 4.0), 12,312 / 736 / 718 / 1,272 / 1,254 (train / valid / test_fold / test_family / test_superfamily) | accuracy, one per test split |
| `reaction` | multi-class, 384 EC numbers | IEConv ProtFunct (github.com/phermosilla/IEConv_proteins, Google Drive archive), 29,215 / 2,562 / 5,651 | accuracy |
| `ss_q3` / `ss_q8` | per-residue multi-class, 3 (helix / strand / coil) / 8 (DSSP) classes | TAPE / PEER release of the NetSurfP-2.0 data (S3 `secondary_structure.tar.gz`), train 8,678 / valid 2,170, test CB513 513 / TS115 115 / CASP12 21 chains, mapped to RCSB structures | per-residue accuracy (Q3 / Q8) on CB513, macro-F1, coverage |
| `lba` | regression, pK | ATOM3D LBA (PDBbind 2019 refined, 4,463 complexes), 30 % split 3507/466/490, 60 % split 3563/448/452 | RMSE (lower) |
| `rsr` | ranking regression, RMSD of 3D RNA models | ATOM3D RNA Structure Ranking (Zenodo 4961085, candidates split by time): FARFAR2-Puzzles models of 21 RNA-Puzzles, 12,479 / 4,000 / 4,000 models (train / val / test; 13 / 4 / 4 puzzles) | mean per-puzzle Spearman (higher) |
| `mn_bbbp` ... `mn_qm8` (14 MoleculeNet tasks, QM9 excluded) | `multilabel_masked` (BBBP, BACE, ClinTox, Tox21, ToxCast, SIDER, HIV, PCBA, MUV) or `multitarget_regression` (ESOL, FreeSolv, Lipo, QM7, QM8) | Uni-Mol LMDB release, scaffold split as shipped; first conformer, hydrogens dropped, no cropping | ROC-AUC + PRC-AUC (classification); RMSE + MAE (regression; primary: RMSE for ESOL/FreeSolv/Lipo, MAE for QM7/QM8) |
| `npass_ptp1b`, `npass_ache`, `npass_cox2`, `npass_hivrt`, `npass_tyrosinase`, `npass_cyp3a4`, `npass_mrp4`, `npass_cox1` (8 NPASS tasks) | `multitarget_regression`, one target | NaFM data release (Figshare 10.6084/m9.figshare.28980254.v1, `regression_data.csv`, CC BY 4.0), 612 / 341 / 189 (paper: 190) / 186 / 186 / 178 / 177 / 140 molecules; random 80/10/10 split of the NaFM reference code (seed 0); RDKit ETKDGv3 + MMFF94 conformer, hydrogens dropped | RMSE (primary) + MAE |

## Quick use

Two locations are set outside the code: `MIND_DATA_ROOT` (where task data is downloaded and processed, laid out as
`<MIND_DATA_ROOT>/downstream/<task>/...`) and `MIND_RESULTS_ROOT` (the results ledger and all generated reports). Pass
them as environment variables or as Make variables, or keep them in an untracked `local.mk` (see the Makefile header).
Baseline weights (ESM-2, SaProt) are fetched with `make ds-fetch-models BASELINES="<name> ..."` into `MIND_MODELS_ROOT`.

```bash
make ds-prepare [TASKS=all|molnet|ec,go_mf,...] [EXCLUDE="reaction mn_pcba"] [PLAN=1]   # all data steps in one run
make ds-download TASK=<task>                 # one task: fetch and verify the source data (CPU, network)
make ds-process TASK=<task>                  # one task: build the processed files (CPU)
make ds-run MODEL=<snapshot | path/to.ckpt> NAME=<snapshot> GPU=0 PLAN=1   # print the plan, write nothing
make ds-run MODEL=<snapshot | path/to.ckpt> NAME=<snapshot> GPU=0 TASKS=mn_bbbp,mn_bace
make ds-status                               # state of this month's ds_* logs
```

`ds-run` embeds, probes, adds the results to the ledger and regenerates the reports. Results are stored under
`$MIND_RESULTS_ROOT/reports`: `summary.md` (primary metric per task and head, counts of missing cells),
`missing_cells.md` and `missing_cells.csv` (every missing cell with its category and the command that fills it),
`tasks/<task>.md|csv` (every metric the catalog requires, such as ROC-AUC and PRC-AUC or RMSE and MAE) and
`timeline/<task>.md|csv`. Details are in "One command per model" below.

## Pipeline

| Step | Command | Kind |
|---|---|---|
| download | `python -m downstream.tasks.<task>.prepare download` | data processing, CPU, network |
| process | `python -m downstream.tasks.ec.prepare process --workers 16` / `...lba.prepare process --threads 8` | data processing, CPU |
| snapshot | `python -m downstream.common.registry snapshot <ckpt> <name>` and `... alias current_best <name>` | file copy, CPU |
| embed | `python -m downstream.embed --task <task> --model <snapshot> [--random-init]` | GPU inference |
| ESM-2 | `python -m downstream.embed --task ec --model esm2_t12_35M_UR50D --hf-cache-dir DIR` | GPU inference |
| Foldseek 3Di | `python -m downstream.baselines.foldseek_3di --task ec --foldseek BIN --threads 16` | data processing, CPU |
| SaProt | `python -m downstream.embed --task ec --model saprot_650m_pdb --hf-cache-dir DIR` | GPU inference |
| probe | `python -m downstream.probe --task <task> --model-id <id> --head linear\|mlp [--pooling P]` | CPU training |
| fine-tune | `python -m downstream.finetune --task <task> --model <snapshot> [--random-init]` | GPU training, one GPU |
| report | `python -m downstream.report --task <task>` | CPU |
| ledger | `python -m downstream.ledger add --outputs-root DIR`, `downstream.ledger_report report`, `downstream.ledger_compare --a A --b B` | CPU, writes only under `$MIND_RESULTS_ROOT` |

For `lba`, `embed`, `probe` and `finetune` also take `--split-tag identity30|identity60`;
`embed` and `probe` additionally take `--representation joint|separate`. Each
combination writes its own embeddings and results directory.

Training-like steps (embed, probe, fine-tune) are user-run through Make targets.

### Fine-tuning

`downstream.finetune` trains encoder and head together for every task kind, with the probe's loss, validation score and
evaluation function (`downstream.probe.evaluate_task`), so a `finetune` row in the results ledger carries the same
metrics as the probe rows of that task (EC/GO Fmax per identity cutoff and AUPRs, GO's IC-binned AUPR, Fold/Reaction
accuracy per test split, LBA RMSE/Pearson/Spearman, RSR per-puzzle and global Spearman/Pearson, MoleculeNet ROC-AUC/PRC-AUC or RMSE/MAE with the QC subsets).
MoleculeNet classification uses BCE over the labelled cells only (NaN = missing); MoleculeNet regression standardizes
the targets with train statistics and reports errors in the original units. Settings come from the `finetune` block of
each `task.yaml` (`downstream/finetune_settings.py` resolves them): AdamW, `lr_encoder` (pretrained encoder) below
`lr_head`; gradients clipped to `grad_clip_norm` 0.5 (the pretraining value); `--random-init` trains the same
architecture from untrained weights (seeded with the run seed) with `lr_encoder_random_init`; `epochs` epochs
(`patience: null`), the epoch with the best validation score is kept and evaluated on the test split(s); seeds 0, 1, 2.
As in the probe, the head reads pooled features standardized with the train mean / std, taken once from the initial
encoder and kept as fixed head buffers (saved in `head.pt`). A batch holds at most `max_atoms_per_batch` graph nodes
and, for MoleculeNet, at most `max_samples_per_batch` 32 molecules (the per-process batch of the Uni-Mol recipe;
without it a 15,000-node budget packs hundreds of molecules into one step). MoleculeNet per-dataset changes live in
`finetune_overrides` of its task.yaml (PCBA: 20 epochs; the others keep 50). Validation and test scoring use
deterministic CUDA kernels, training does not. `run.log` gives the train and validation seconds of every epoch and
notes when the best epoch is the last one (the arm may still have been improving). `--lr-encoder`, `--lr-head`,
`--weight-decay`, `--mlp-dropout`, `--pooling` (one of the task's `embedding.poolings`), `--epochs`, `--patience`,
`--lr-schedule {constant,cosine,plateau}` and `--warmup-ratio` override the file and are recorded in `config.json`
(the schedule keys only when they differ from the default constant lr without warmup, so older records still match); ds-run reports seeds trained with overrides as other settings, so
use a separate `OUTPUTS_ROOT` for such trial runs.
Learning-rate schedules (`downstream/lr_schedule.py`, off by default) scale the encoder and head rates together:
`--warmup-ratio r` ramps linearly from 0 to the peak over `r` of the planned optimizer steps; `cosine` then decays the rate
to 0.1x of the peak at the last planned step; `plateau` halves it (factor 0.5, patience 3 epochs, minimum 1e-6) when the
validation score that selects the best epoch has not improved, stepped once per epoch after validation (not combinable
with warmup). Planned steps = `epochs` x steps per epoch, counted before training with the epoch-0 batching; an early stop
does not shorten the plan. `run.log` prints the rates each epoch starts with.
All requested seeds are checked before training, and `metrics.json` is written last and atomically (it marks a
finished seed). One process uses one GPU; there is no multi-GPU path. `make ds-finetune TASK=<task> MODEL=<snapshot> GPU=<id> [RANDOM_INIT=1] [SEEDS="0 1 2"]
[OUTPUTS_ROOT=<dir>] [FT_ARGS="--epochs 2"]` runs one task detached; `make ds-run ... HEADS=linear,mlp,finetune` (or
`HEADS=finetune`) runs it for every task (below).

### MoleculeNet (`mn_*`)

`make ds-download TASK=mn_esol` fetches the Uni-Mol MoleculeNet archive (size and checksums pinned in
`downstream/tasks/molnet/task.yaml`, verified after the download) and extracts it; all 14 tasks share one archive.
`make ds-process TASK=mn_esol` (and `ds-validate`, a read-only dry run) reads the extracted Uni-Mol LMDB files. Output: `<source_root>/<task>/processed/<revision>/{train,valid,test}.pt` plus `qc_<split>.json/.tsv`
and a `provenance.json`. Rules (`downstream/tasks/molnet/task.yaml`): first of the 11 conformers; hydrogens dropped;
no atom-count cropping; molecules are never dropped for chemistry reasons but flagged and counted (flat conformers,
rare elements mapped to id 119, one heavy atom, no heavy atom, more than 256 heavy atoms). A molecule left with no
heavy atom is kept but cannot be embedded: loading its graph raises. Missing classification labels (-1 in the LMDB) become
NaN with a mask. Coordinates are stored in Angstrom: every dataset declares the unit of its LMDB coordinates in
`task.yaml` (`coordinate_unit`, no default; QM7 stores Bohr and is converted by 0.529177210903), and processing stops
when the median nearest-neighbour distance of a split is outside `plausible_median_nn_angstrom` (1.0-2.0; the real
sets give 1.37-1.46), naming the task. QM7 is processed revision `v2` (the other 13 stay `v1`); a QM7 `v1` file is
refused at load time. The ligand graph is the one of the LBA "separate" ligand path (`dataset_type="unimol"`, 5 A radius, 64
neighbours). The probe trains a masked BCE head (classification) or a masked MSE head on train-standardized targets
(regression), selects the epoch by the primary metric and also reports the QC-flagged subsets of the test set.
`downstream.finetune` fine-tunes both kinds with the same losses and metrics. Embedding with a MIND checkpoint requires CUDA and a checkpoint trained with
`geometry_neutral_virtual_slots` (protein flag set).

### NPASS (`npass_*`)

Eight natural-product bioactivity regressions of the NaFM paper (Table 2). `make ds-download TASK=npass_ptp1b` fetches
the 0.46 MiB `regression_data.csv` (url, size, md5, sha256 and row count pinned in `downstream/tasks/npass/task.yaml`; a
difference stops the run) into `<source_root>/raw`, shared by the eight tasks. `make ds-process TASK=npass_ptp1b` (or
`TASK=npass` for all eight; `ds-validate` is the read-only dry run) selects the rows of the target id, splits them with
the NaFM reference code's rule (`np.random.default_rng(0).permutation`, `round(0.8 n)` / `round(0.1 n)` / rest), builds one
RDKit ETKDGv3 conformer with explicit hydrogens and seed 0, MMFF94-optimizes it and drops the hydrogens. Output:
`<source_root>/<task>/processed/v1/{train,valid,test}.pt`, `qc_<split>.json/.tsv`, `provenance.json`. The payload and loader
are the MoleculeNet ones, so `embed`, `probe`, `finetune` and the ECFP4 baselines (`make ds-ecfp TASKS=npass`) apply
unchanged; `TASKS=npass` selects the family in `ds-run`.

Every molecule is kept. One whose embedding fails (also after a retry with random start coordinates) gets flat 2-D
coordinates, the flag `all_conformers_flat`, and is counted and listed in `qc_<split>.json` / `.tsv`; the test subsets
report it separately. MMFF outcomes (converged, not converged, no parameters) are counted. The counts must equal the
pinned ones: 7 targets equal the paper's Table 2, COX-2 has 189 molecules in the release against 190 in the paper (the
cause is not documented; `count_note`).

What the primary sources fix and what they do not (details in the `task.yaml` header): the data file, the per-target
counts and the reference code's random 80/10/10 split are established; the reference code draws the split from the run
seed, while here one split (seed 0) serves all three training seeds, so our standard deviations cover initialization
only. The unit and transform of `activity_value`, the NPASS activity type and filtering behind the file, the paper's seed
list and the split of its Table 2 are not stated anywhere (the paper says only "three random seeds"); errors are therefore
in the unit of that column. Literature numbers are not strictly comparable to ours. Per-seed splits would need a split tag
like LBA's through embed, probe, finetune and the ledger; that is not implemented.

SaProt (`saprot_650m_pdb`, `saprot_650m_af2`, `saprot_35m_af2`) sees the residues
MIND kept, one `<AA><3Di>` token each; the 3Di state comes from Foldseek on the
same structure file, aligned to the kept residues without mismatches (rules in
`downstream/baselines/foldseek_3di.py`). Residues MIND masks for an incomplete
sidechain, and residues without a Foldseek state, get SaProt's unknown-structure
letter `#`; the amino acid is kept. The 3Di cache lives in `<data_root>/foldseek_3di_<processed_revision>/`
and is valid for that processed revision only.

## Contracts

- `downstream/tasks/ec/task.yaml` must name a missing-atom policy explicitly
  (`missing_atoms`); processing stops if it is absent or unknown.
  See `downstream/common/missing_atoms.py` for `strict`,
  `drop_residue_missing_backbone` + `mask` / `drop_chain_if_fraction_above`.
- Experimental structures never reach `ProteinAdapter` unchecked: its 14-slot
  routine pads a missing sidechain atom with V and shifts later atoms into wrong
  slots. (Its raw-file entry, `convert_to_universal`, used by the pretraining
  builders, refuses any incomplete or unknown residue with a `ValueError`.)
  Incomplete residues are masked like a pretraining MLM block (all 14 slots,
  mask token 0, sidechain slots latent) or the chain is dropped.
- `block_idx` of a chain graph is the residue's position in the parsed chain, so a
  residue dropped for a missing backbone atom leaves a gap and Tier-1 sequence edges
  join only true sequence neighbours. Processed files built before this rule (no
  `block_index` in the schema `source`) are refused at load time.
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

Data (outside Git): `<data_root>/downstream/<task>/<modality>/{raw,processed,splits,provenance.json}`.
`processed_revision` in `task.yaml` (EC and GO: `v3`; Fold, Reaction and MoleculeNet QM7: `v2`) is appended to the processed
directory name. EC/GO v2 reads PDB chain ids of two or three characters, which the unsuffixed revision dropped as
`parse_error`; EC/GO v3 and Fold/Reaction v2 keep the sequence gap of dropped residues in `block_idx` (schema
`source.block_index`), and the earlier directories are refused at load time. Embedding paths carry no data revision,
so `ds-run` reports an embedding built from another processed directory as an error instead of skipping it: embed a
new revision under a new `OUTPUTS_ROOT`. Every `process`
run appends its own record (command, config/input/output sha256, QC summaries) to `provenance.json`.
EC test chains missing from `test_cutoffs.csv` raise, except the documented `test_cutoff_unlisted_chains`.

Models and results (under `outputs/`, or `$MIND_OUTPUTS_ROOT`):

```
models/<snapshot>/{model.ckpt,card.json,hparams.json}   read-only copy, sha256 in card
models/<alias> -> <snapshot>                            e.g. current_best
downstream/<task>/<model_id>/embeddings/<split>.{npz,json}
downstream/<task>/<model_id>/<protocol>.<pooling>/seed<k>/{metrics.json,predictions.npz,head.pt,run.log,config.json}
```

`model_id` is the snapshot name, `<snapshot>.random_init` for the untrained
control, or the ESM-2 / SaProt name. Literature EC numbers in the report are fine-tuned
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


## RSR (ATOM3D RNA structure ranking)

Target: all-atom RMSD (Angstrom) of a candidate 3D model of an RNA to its experimental structure; the task ranks the
candidates of a puzzle. Source: Zenodo record 4961085, `RSR-candidates-split-by-time.tar.gz` (1,269,970,448 bytes, md5
pinned in `downstream/tasks/rsr/task.yaml` from the record API, plus the record's README, LICENSE_SA and split-index
files). Zenodo gives md5 only: `download` records the sha256 of every file in `provenance.json` on the first run and
refuses a later run whose file differs. The LMDB layout is read like LBA's (gzip + JSON or msgpack, `to_dict(orient="split")`
frames, no `atom3d` dependency, pickle refused), but the archive's folder names are not documented, so the three LMDBs
are located by name (`train`, `val`, `test` holding `data.mdb`) and exactly one set must exist.

- Split: ATOM3D's temporal split by puzzle (the models of a puzzle never cross splits). `expected_counts` 12,479 / 4,000 /
  4,000 are the line counts of the record's index files; `download` and `process` compare them with the LMDB
  `num_examples` and stop on a difference. 13 x 1000 would be 13,000 training models: some training puzzle(s) have fewer
  than 1000, per-puzzle counts are in `qc_<split>.json`. The paper text calls the validation puzzles "14-17 (excluding
  16)" (three), the record has four (4,000 models); processing follows the record and `expected_puzzles` (13 / 4 / 4).
- Graph: the RNA training adapter's own 23-slot routine (`downstream/common/rna_graph.py`, a parallel of the protein
  `graph_builder`): hydrogens, the listed placeholder atoms and non-blank/`A` alternate locations are skipped, atom names go through the adapter's
  legacy aliases, residues must be A/C/G/U. Per model, counted in `qc_<split>.tsv`: nucleotides with an incomplete
  backbone (the 5' end has no phosphate; kept, a quarantined block as in pretraining), missing base atoms (not repaired:
  the base slots take the real atoms in priority order and the rest become V at C1', so atoms behind a gap move to an
  earlier slot, as in pretraining), removed unexpected atoms, dropped terminal OP3. A model that cannot be converted
  (unsupported residue, duplicated atom names, bad RMSD, no/mixed puzzle, too few nucleotides) is a counted failure with its
  reason; above `qc.max_failed_structure_fraction`, `qc.max_incomplete_base_fraction`, a puzzle count other than expected or
  a puzzle with no usable model the split is not written and the command exits 1. Puzzle overlap between processed
  splits exits 1. RNA graphs have no random edges, so embeddings do not depend on the seed.
- Ids are `<puzzle>/<LMDB index>`; the puzzle is the first element of the LMDB `id` (the text of a tuple `(puzzle, model)`, e.g. `rna_puzzle_14_bound`; the atoms' `ensemble` column is the model's file name, not the puzzle). The raw
  LMDB id is kept in the QC table and the processed file.
- Metrics (`downstream/common/metrics.py: evaluate_ranking_regression`): per puzzle Spearman and Pearson, averaged over the
  puzzles (`mean_per_puzzle`), and pooled over all models (`global`); `mean_per_puzzle` also has the mean top-1 loss
  (true RMSD of the model ranked first minus the best true RMSD of its puzzle; ATOM3D does not report it) and the
  per-puzzle lists; `global` also RMSE and MAE. A puzzle with an undefined correlation makes that mean `null` with the
  reason, never a mean over the rest. The ledger cells are `(test, mean_per_puzzle|global, spearman_rho|pearson_r)`, the
  same names as the literature file.
- Probe: the head reads the pooled embedding (`mean_atoms`), MSE on the RMSD standardized with train statistics (scores
  are mapped back to Angstrom), epoch selection by the validation mean per-puzzle Spearman. Fine-tuning
  (`downstream.finetune`) uses the same loss, selection and metrics; `finetune.lr_encoder` is 1e-4 (small-task recipe), the
  random-init arm 3e-4. Embedding needs a checkpoint trained with `geometry_neutral_virtual_slots` for RNA (`rna: true`);
  `ds-run` runs `rsr` only for models whose card lists the `rna` domain.
- Commands: `make ds-download TASK=rsr`, `make ds-validate TASK=rsr` (converts 20 models per split, writes nothing),
  `make ds-process TASK=rsr WORKERS=8`; embed / probe / fine-tune take `--task rsr`.
- Read on the real data: json serialization, `scores.rms`, LMDB counts equal the index files; 21 puzzles (13 / 4 / 4;
  `rna_puzzle_2_hack` has 479 models, all others 1000). Rosetta placeholder atoms RPA1/RPA2/RPB1/RPB2 (empty element, 0-7
  per model, no RNA atoms) are removed and counted as `placeholder_atoms_removed`; no other unknown atoms occur.

## GO (go_mf, go_bp, go_cc)

One task family, three tasks, one processed file per split holding all three label matrices
(`downstream/tasks/go/`; `task.yaml` is the single config, `mf.py`/`bp.py`/`cc.py` select the branch).

- Source: Zenodo record 6622158, `GeneOntology.zip` (md5 and size pinned in `task.yaml`), the same record that
  holds EC's archive. Shared raw data lives once in `<data_root>/downstream/deepfri/raw/`;
  splits, processed files and `provenance.json` are task-specific under `deepfri/go/`.
- Annotation file: per ontology a 4-row block (terms, names) in the order MF, BP, CC, then one row per chain.
  `read_go_annotations` rejects any deviation (header text, GO id shape, duplicate terms or chains, term
  counts, terms outside the branch's list).
- Chains with no term in a branch are excluded from that branch's train/valid/test
  (`unannotated_policy: exclude`, counts in `GoProcessedSplit.report`). The CAFA Fmax in
  `downstream/common/metrics.py` raises on a protein without a true term; TorchDrug instead keeps them as
  all-zero rows, so GearNet-style numbers can differ slightly. `keep_zero_rows` is refused by the loader.
- Missing atoms: same policy and code path as EC (`process_entry`); the adapter's V-padding/slot shift for a
  missing sidechain atom (see Contracts) is never reached with an incomplete residue.
- Commands: `make ds-download TASK=go_mf`, `make ds-validate TASK=go_mf` (read-only),
  `make ds-process TASK=go_mf WORKERS=16`; embed/probe/finetune take `--task go_mf|go_bp|go_cc`.

## Leakage registry

`downstream/common/leakage.py` + `downstream/leakage_references.yaml` compare the ids of a task's splits with the
pretraining plans (protein 1M plan, protein 40k manifest, RNA full plan) and write a `leakage` record into the
task's `provenance.json` plus a per-id TSV. Categories: `seen_train`, `seen_val` (used for checkpoint
selection), `seen_test` (pretraining hold-out), `unseen`, `unmapped`. PDB ids against the AFDB (UniProt) set are
refused unless a `pdb_chain -> uniprot` map is given; it never reports "0 leaked" for incomparable namespaces.
Id matching does not detect homologs. `make ds-leakage ...` (see the Makefile header).


## Secondary structure (ss_q3, ss_q8)

`downstream/tasks/ss/` (`task.yaml`, `dataset.py`, `mapping.py`, `tape_lmdb.py`, `structures.py`, `prepare.py`); one data directory,
one download and one processed directory serve both tasks.

- Protocol (the TAPE / PEER one): train and valid from Klausen et al. (NetSurfP-2.0), 25 % identity filtered against the test sets;
  tests CB513, TS115, CASP12; per-residue accuracy. Labels are the release's own: `ss8` index 0..7 is DSSP `GHIBESTL`, `ss3` is
  helix {G,H,I} / strand {B,E} / coil {S,T,L}; the code checks `ss3 == collapse(ss8)` for every chain and never recomputes DSSP.
  The TAPE LMDB values are pickles of numpy arrays: `tape_lmdb.py` unpickles with an allow-list of four numpy globals.
- Structure mapping (`mapping.py`): each chain id `<pdb>-<chain>` is fetched from RCSB (`files.rcsb.org/download/<ID>.cif.gz`,
  current coordinates, author chain id), its standard residues are aligned to the dataset sequence (global, end gaps free, `X` neutral)
  and the dataset label is put on every structure residue with an identical amino acid. A chain with identity below
  `alignment.min_identity` or aligned fraction below `alignment.min_aligned_fraction` is dropped and counted. Every dataset residue
  ends in exactly one bucket (`evaluated`, `not_in_structure_flagged_unresolved`, `not_in_structure_unflagged`,
  `sequence_mismatch`, `dropped_missing_backbone`, `chain_not_processed`); the sum is checked. Unlabelled structure residues stay in
  the graph as context. Residues with a missing backbone atom are dropped and incomplete sidechains masked as in EC; a chain longer
  than 1024 residues becomes contiguous windows.
- Coverage is part of every result: `qc_<split>.json` and `<seed dir>/coverage.json`, and each test block of `metrics.json` holds
  `residue_coverage`, `chain_coverage`, `literature_residues`, `literature_chains`. Accuracy and macro-F1 are over the evaluated
  residues only, so a coverage below 1 is a SUBSET of the literature set (the log line says so).
  `accuracy_all_residues_coil_fallback` has the literature's denominator and counts every residue that could not be evaluated as
  predicted coil: the release labels unresolved residues coil (`ss8` 7, `valid_mask` 1.0), whereas NetSurfP-2.0 treats them as
  unlabelled and does not score them.
- Residue vector (`downstream.common.encoder.residue_states`): the CA node state after ESA. Geometry-neutral encoding keeps N, CA, C
  of every residue as real ESA nodes even when its sidechain block is masked, so every residue has one; `mean_backbone` (mean of N,
  CA, C) is the alternative. A residue of a masked block carries a state computed from mask-token backbone nodes; those labelled rows
  are counted in the embedding record (`labelled_rows_in_masked_blocks`).
- Probe: embeddings hold one row per labelled residue (`downstream.embed`, MIND snapshots only); the probe is the Fold one (cross
  entropy, validation accuracy, train-standardized features) with residues as rows. Fine-tuning (`downstream.finetune`): cross
  entropy over the labelled residues of each batch, the head reads every residue's CA state, encoder LR 1e-4, epoch chosen by
  validation accuracy. Both tasks embed separately (two embedding sets of the same residues).
- Commands: `make ds-download TASK=ss_q3 SS_ARGS=--records-only` (archive + id census, no structures), then `make ds-download TASK=ss_q3`
  (structures, resumable, 4 requests per second), `make ds-validate TASK=ss_q3`, `make ds-process TASK=ss_q3 WORKERS=16`.
  A split with more than `qc.max_failed_chain_fraction` unusable chains is not written (`SS_ARGS="--max-failed-fraction 0.4"` accepts more).

## Fold and Reaction (multiclass)

`downstream/tasks/{fold,reaction}/` (`task.yaml`, `dataset.py`, `prepare.py`) on top of `downstream/tasks/multiclass.py`.

- Structures are the archives' own `.hdf5` files (`structure_io.read_hdf5_chain`, needs `h5py`, which is required to process Fold and Reaction). Atom elements come from atom names; `atom_types` is not used. Chains go through the EC/GO code path
  (`process_entry` + missing-atom policy), so a CA-only domain is dropped and counted, never kept. `ProcessedSplit.report`
  and `qc_<split>.json` carry listed / kept / dropped counts.
- Source notes and pinned checksums are at the top of each `task.yaml`. Reaction's Google Drive archive has no published
  size or md5: `download` stops until `archive_sha256` / `archive_bytes` are pinned after reviewing `provenance.json`.
- Probe: cross-entropy head, validation accuracy, one accuracy per test split in `metrics.json`; `downstream.report`
  adds the GearNet-paper literature rows. Fine-tuning uses the same loss and reports the same blocks.
- Commands: `make ds-download|ds-validate|ds-process TASK=fold` (same for `reaction`).
- Leakage: Fold ids are SCOPe domain ids (`scop_domain`, no map to the AFDB UniProt reference, so the tool refuses);
  Reaction ids are `4xi6.A` (`pdb_chain`, needs `--id-map`). See `queries:` in `leakage_references.yaml`.

## Results ledger

`downstream/ledger.py`, `ledger_report.py`, `ledger_compare.py`; the required-metric catalog is
`downstream/ledger_catalog.yaml` (every metric a reference paper reports for a task; uncomputed ones are marked
`code: missing` and appear as NO-CODE; currently none).

Metrics added for the catalog's former NO-CODE cells (all in `downstream/common/metrics.py`, written to `metrics.json` by
the probe): EC/GO `coverage` (fraction of test proteins with at least one term scored at or above the Fmax threshold;
DeepFRI's own threshold for its legend number is not stated); GO `ic_binned_macro_aupr` (label-centric AUPR over terms
with IC > 5, plus `..._ic_gt10` for IC > 10, with the term counts; IC = -log2(train frequency) because DeepFRI's UniProt-GOA
table is not available, thresholds in `go/task.yaml`); Fold/Reaction `weighted_f1` (per-class F1 weighted by class support,
PEER Table 5). Runs written before these metrics lack them (`MISSING:metric-not-in-file`); `coverage` and `weighted_f1` can be
recomputed from `predictions.npz`, `ic_binned_macro_aupr` additionally needs the train labels (the probe's `embeddings/train.npz`).

What goes where: Git holds only the code (the three modules, the catalog, the Make targets and the tests). The ledger
data and every generated report stay outside Git under `$MIND_RESULTS_ROOT` (Python has no default; the Makefile passes
its `MIND_RESULTS_ROOT`), as does `literature/literature.csv`; committing the code never commits a result.

Std: ledger reports and comparisons use the sample std over run seeds (ddof=1, undefined for one seed), while
`downstream.report` (`ds-report`) keeps ddof=0, so the same seeds can show two different std values.

- `ledger/measurements.jsonl`: one row per model x run seed x (eval split, subset, metric), with checkpoint sha256,
  training card fields (epoch, `selection_rule`), data revision and processed-file sha256, protocol and its settings
  hash, the probe's own commit (`config.json` `run`; older runs list `run_git` in `missing_provenance`).
  `models.jsonl`, `annotations.jsonl` (superseded / invalid / caveat / smoke / valid), `ingests.jsonl`.
- Append-only: `MANIFEST.json` holds each file's line count and sha256; an edit outside the module stops every
  write and `verify`. Writes hold an exclusive `flock`. Re-ingesting a file (same metrics and config sha256) adds nothing.
  One model name with two checkpoint hashes is an error.
- Reports (`reports/`, regenerated): `tasks/<task>.md|csv` (one "Setting" section per comparable setting: protocol,
  pooling, data revision and fingerprint, settings hash; superseded rows excluded), `timeline/<task>.md|csv` (all
  statuses), `summary.md` (primary metric per task, counts of required / computed / missing cells),
  `missing_cells.md|csv` (each missing cell with its category and the command that fills it). Literature comes from
  `<root>/literature/literature.csv` (local) and is shown in its own column, unit and protocol.
- `ledger_compare`: cell-by-cell A vs B with probe-seed std, a paired bootstrap interval over `predictions.npz` (chains
  aligned by id; each seed's predictions must reproduce the ledger value) and the unmeasured pretraining-seed noise
  stated in a warning box, plus the expected number of chance "differences". Output: `reports/compare/<A>__vs__<B>.md|csv`
  for a run over all tasks; a run restricted with `TASKS=...` writes `<A>__vs__<B>__tasks-<task>-<task>.md|csv` (sorted
  task names; a hash for long lists), so it can never replace the all-task files of the pair. Speed: per-target ROC-AUC
  and PRC-AUC of the multi-target tasks (ToxCast, PCBA, Tox21, MUV, SIDER) are computed for all resamples at once from
  integer resample weights over the scores sorted once per target (ties handled as in scikit-learn); the random
  resample indices, the decision rule and the files are unchanged, and the interval equals the earlier per-resample
  scikit-learn computation to about 1e-15 (`tests/test_downstream_ledger_compare.py`). `COMPARE_WORKERS=N`
  (`--workers N`) lets one comparison use N processes (one core each) for the per-target work of the large tasks; the
  default is min(16, usable cores), the maximum is the number of usable cores, and the result agrees to floating-point rounding (about 1e-16) whatever the number.
  The command prints the worker count it uses.
- Commands: `make ds-ledger-add OUTPUTS_ROOT=<dir> [EVAL_DIRS=...]`, `ds-ledger-annotate`, `ds-ledger-report`,
  `ds-summary`, `ds-missing`, `ds-compare A=<model> B=<model>`, `ds-ledger-verify`. `ds-report` is unchanged.
  Report models: `MODELS=...`, else `summary_models` in `<root>/comparisons.yaml`, else every model of the ledger;
  a listed model the ledger has never seen appears as `MISSING:model-not-in-ledger`, never skipped.
- `make ds-ledger-recompute MODEL=<m> CONTROL=<m2> DATA_REVISION=<rev> [TASK=ec] [DRY_RUN=1]`: adds the macro AUPR a
  frozen probe did not record (EC/GO), computed from each seed's `predictions.npz` with the probe's own function. It
  writes nothing unless the metrics files are unchanged since ingestion, fmax and pair AUPR are reproduced exactly,
  the CONTROL model's recorded macro AUPR is reproduced by the same formula, and no macro AUPR row exists yet. New
  rows name their source in `recomputed_from` and stay in the "Setting" section of the file they come from.

## One command for the data (`make ds-prepare`)

```bash
make ds-prepare PLAN=1                                  # plan, sizes, disk check; writes nothing
make ds-prepare                                         # every registered task, detached
make ds-prepare TASKS=molnet EXCLUDE="mn_pcba" WORKERS=16
make ds-prepare TASKS=ec,go_mf,fold FOREGROUND=1        # in the terminal
```

`TASKS` takes `all`, a family (`molnet`, `go`, ...) or task names; `EXCLUDE` removes names or families again (an unknown
name is refused). The tasks come from `downstream.tasks.TASKS`, so a task registered the same way (module
`downstream.tasks.<family>.*` plus a `prepare` CLI with `download` and `process`) is included without a change to the
command. A prepare module declares its pinned archives and outputs in `download_spec(config)` and its extra CLI
arguments in `cli_args(...)` (`downstream/common/prepare_spec.py`); `disk_estimate_gib` in task.yaml feeds the disk
estimate.

* Plan: tasks that use the same archive share one `download` step (the 14 MoleculeNet tasks, the three GO
  branches), tasks that write the same processed directory share one `process` step; all downloads run before any
  processing. `make` prints the plan (steps, states, archive size in GiB, estimated disk need, free space) in the
  terminal first and stops on a problem; the run itself is detached and writes one log under `logs/<MonthYear>/`
  (`ds_prepare_*.log`) that ends with `EXIT_CODE=<n>` (`make ds-status`).
* Idempotent: a finished step is re-verified, not redone. Archives are checked against their pinned size and
  checksums; processed split files against the sha256 recorded in `provenance.json` (without a record only their
  existence is checked, and the log says so). Re-running on a finished tree writes nothing. If every processed
  directory of a task exists, its raw download is reported as not needed and is not fetched again.
* Never overwritten: a `<archive>.partial` file, an archive of the wrong size, or a processed directory with only some
  of its split files is a plan error that names the path; nothing runs until you remove or move it.
* Unavailable sources: a task whose download cannot be fetched (today `reaction`: the Google Drive link returns
  HTTP 404; task.yaml `download_unavailable`) is listed as `unavailable` and skipped under `all`/a family, and is an
  error (exit 2, nothing runs) when named in `TASKS`.
* Failure: a failing step is reported and the steps that do not depend on it go on; the exit code is 1 and the
  summary lists what failed or was skipped. Exit code 0 without the expected outputs also counts as a failure.
* `ds-run` stops with `no processed data ... make ds-prepare TASKS=<task>` for a task without processed files.

## One command per model (`make ds-run`)

`downstream/run_all.py`: `make ds-run MODEL=<snapshot | path/to.ckpt> [NAME=<snapshot>] [TASKS=all|ec,go_mf,...]
GPU=2,3 [PER_GPU=1] [HEADS=linear,mlp] [PROBE_JOBS=1] [RANDOM_INIT=1] [OUTPUTS_ROOT=<dir>] [PLAN=1]
[COMPARE_TO=<snapshot[.random_init],...>] [BOOTSTRAP=1000] [COUNTERFACTUAL=1 [CF_MODES="..."] [EVAL_CONFIG=<config>]]
[FOREGROUND=1]`.
A checkpoint path is first snapshotted as NAME (a NAME that already names another checkpoint, in the snapshot
store or in the ledger, stops the run). Then one embedding job per task on the GPUs (at most PER_GPU per GPU; splits
already embedded are skipped), then, after every embedding job has ended, the probes on CPU (SEEDS; seeds already
done are skipped), `ledger add`, and a table of the primary metric per task and head plus the missing required
cells. `all` is EC, GO, Fold, LBA (identity30 and identity60, joint), RSR, the 14 MoleculeNet and the 8 NPASS tasks (`TASKS=molnet`, `TASKS=npass`);
Reaction only when named. A task outside the model's pretraining domains (protein needs `pdb`, RSR `rna`, MoleculeNet and NPASS `unimol`) is skipped
and listed under `all`, refused when named. A failing job is reported with its log and the others continue; the
run ends with the list of what is missing and exit code 1. `PLAN=1` prints what would run (and recorded embedding
times of other models, when there are any) and writes nothing. Logs and `summary.md` go to
`logs/<Month><Year>/ds_run_<model>_<stamp>_<pid>/`.

- `HEADS=linear,mlp,finetune` adds fine-tuning: after the probes, one `downstream.finetune` job per task (SEEDS),
  at most `FT_PER_GPU` jobs per GPU (default 1; PER_GPU applies to embeddings only), the longest first (estimate:
  task.yaml `finetune_seconds_per_epoch` x epochs x seeds), each next job on the first free slot. Rows are tagged
  `protocol_family=finetune`, `protocol_id=finetune` (LBA `finetune.<split_tag>.joint`). A seed counts as done only
  when its `metrics.json` exists and its `config.json` records the current task.yaml settings and processed directory;
  otherwise the task is reported as an error ("use a new OUTPUTS_ROOT"), never silently skipped or overwritten.
  `HEADS=finetune` alone skips the embeddings and probes; a fine-tune problem never blocks the probes.
  `RANDOM_INIT=1`, `MODEL=<snapshot>.random_init` and `COMPARE_TO=<snapshot>.random_init` fine-tune the untrained
  control the same way.
- After every finished probe or fine-tune job, ds-run adds the outputs root to the ledger (one add at a time;
  `ledger add` skips files already ingested), so an interrupted multi-day run keeps what finished.
- `MODELS="a b a.random_init"` (instead of MODEL) runs ds-run for each model one after another in one detached
  command and one log; each model ends with `MODEL_EXIT_CODE <model>=<n>`, the log with `EXIT_CODE`.

- `COMPARE_TO=a,b.random_init`: every listed model goes through the same tasks, heads and seeds (what is complete
  is skipped; `.random_init` is the RANDOM_INIT=1 control of that snapshot), gets its own table, and at the end
  `ds-compare A=<each> B=MODEL` runs (BOOTSTRAP resamples). The last step of every ds-run is the ledger report
  (`make ds-ledger-report`: `tasks/`, `timeline/`, `summary.md`, `missing_cells.*` under `$MIND_RESULTS_ROOT/reports`).
- `COUNTERFACTUAL=1` (MODEL must be a `.ckpt` path): the `make eval-counterfactual` sweep (same arguments; CF_MODES,
  SEED, DOMAINS=unimol, EVAL_CONFIG) on the snapshot copy, one mode per GPU job after the embeddings, written to
  `<outputs root>/eval/cf_<NAME>_<stamp>/<mode>` and added to the ledger. A failing mode is listed, the others go on.
- Example (one model, two references, all tasks, two GPUs):
  `make ds-run MODEL=/path/to/best-checkpoint.ckpt NAME=<new snapshot> OUTPUTS_ROOT=<outputs root> GPU=2,3
  COMPARE_TO=<reference snapshot>,<reference snapshot>.random_init` (first with `PLAN=1`).

**Every long ds-* job keeps running when the terminal or SSH connection closes.** `ds-run`, `ds-ecfp`, `ds-probe` and `ds-finetune` start in
their own session (`setsid nohup`, no terminal) and return at once with the log path; the log's last line is
`EXIT_CODE=<n>` of the Python command (non-zero when it failed). `make ds-status [N=15]`
lists this month's `ds_*.log` files with that line (no line: still running or killed; check the process with
`make status`). `FOREGROUND=1` runs them in the terminal (`| tee`); `PLAN=1` always runs in the terminal.
`ds-finetune` is detached the same way. `ds-embed`, `eval-pretrain` and `eval-counterfactual` run as `nohup ... &`. Short CPU steps
(`ds-snapshot`, `ds-report`, `ds-ledger-*`, `ds-compare`, `ds-summary`, `ds-missing`) and the data steps
(`ds-download`/`ds-process`/`ds-validate`/`ds-leakage`) stay in the foreground.

## ECFP4 fingerprint baselines (`make ds-ecfp`)

`downstream/baselines/ecfp.py` and `ecfp_tuned.py`: two baselines for the 14 MoleculeNet and the 8 NPASS tasks on the same features, the
Morgan fingerprint (radius 2 = ECFP4, 2048 bits, RDKit `rdFingerprintGenerator`) of the SMILES stored in the processed
splits, and on the same fixed split (Uni-Mol scaffold split; NPASS: the seed-0 random split). CPU only, no checkpoint.

| model id | protocol | what it is | seeds |
|---|---|---|---|
| `ecfp4` | `linear_probe`, `mlp_probe` | the MIND probe (`downstream.probe`: same heads, masked losses, validation-metric epoch selection, `task.yaml` probe settings) with the fingerprint as the feature | 0-2 |
| `ecfp4_tuned` | `sklearn_tuned` | scikit-learn `LogisticRegression` (liblinear; grid C in 0.01, 0.1, 1; 0.05, 0.5 from 30,000 train molecules) or `Ridge` (alpha in 0.1 ... 1000) per target, the parameter chosen on the validation split by the primary metric, test scored once, no feature scaling, no subsampling | 0 only (the fit is deterministic) |

```bash
make ds-ecfp PLAN=1                                   # what would run; writes nothing
make ds-ecfp TASKS=all [VARIANT=probe|tuned|both] [HEADS=linear,mlp] [SEEDS="0 1 2"] [PROBE_JOBS=2] [DS_THREADS=8] [OUTPUTS_ROOT=<dir>]
```

Per task: fingerprints -> `<outputs>/downstream/<task>/ecfp4/embeddings/<split>.{npz,json}` (pooling name
`ecfp4_bits2048`; the json holds the RDKit version, radius, size and variant), then the probes (VARIANT `probe` or `both`)
for the seeds without a `metrics.json`, and the tuned fit (VARIANT `tuned` or `both`, default `both`) whose files go to
`<outputs>/downstream/<task>/ecfp4_tuned/sklearn_tuned.ecfp4_bits2048/seed0/` (the same `metrics.json`, `predictions.npz`,
`config.json`, `run.log`; the config holds the grid, the chosen value, the library versions and the embedding records of
`ecfp4`), then `ledger add` and the ledger reports of `$MIND_RESULTS_ROOT`. Detached like `ds-run` (log ends with
`EXIT_CODE=<n>`; `NO_LEDGER=1` stops before the ledger, `FOREGROUND=1` runs in the terminal). A SMILES RDKit cannot parse
fails the task (it is never replaced by an all-zero row); a rerun with another fingerprint definition into the same outputs
root is refused. `--feature counts` (count vector) and `--n-bits N` are available on the module's command line.
`PROBE_JOBS` processes run at a time (largest tasks first); `DS_THREADS` is the thread count of a probe and the number of
worker processes that fit the targets of a tuned model.

The probe baseline does not standardize the fingerprint with train statistics (`downstream.probe --feature-scaling none`),
because a bit set in one train molecule would become a value of about 40 and a bit never set in train 1e6 when a test
molecule has it. `downstream.probe` and the ledger refuse a fingerprint that is standardized and an embedding that is not.

Ledger and reports: rows have `model_key` `ecfp4` or `ecfp4_tuned`, `model_kind` `fingerprint_baseline`, no checkpoint
hash, and an `input_feature` field (fingerprint definition and feature scaling). The catalog's `fingerprint_baselines` entries
list the task families a baseline is required on (`molecule_property`), so `missing_cells` and `summary.md` never ask for
it on EC, GO, Fold, Reaction or LBA (`n/a`); `ecfp4_tuned` is required and summarized under its own protocol
`sklearn_tuned`. In the task tables and in `ds-compare` the probe baseline `ecfp4` sits in the same setting as the encoder
probes (the report maps its pooling to the catalog's `table_pooling`, `mean_atoms`; a footnote names the real input);
`ecfp4_tuned` has its own setting because its protocol differs, so `ds-compare` lists its cells as only in one model instead of
comparing it. Add the models to `summary_models` of `<root>/comparisons.yaml` to give them a column in `summary.md`.

# Downstream evaluation

Frozen-encoder probes, fine-tuning, a random-init control and ESM-2 baselines
for MIND encoders.

| Task | Kind | Data | Primary metric |
|---|---|---|---|
| `ec` | multi-label, 538 EC terms | DeepFRI/GearNet split, 15,550 / 1,729 / 1,919 chains, test cutoffs 30/40/50/70/95 % | Fmax (higher) |
| `go_mf` / `go_bp` / `go_cc` | multi-label, 489 / 1,943 / 320 GO terms (counts UNVERIFIED until download) | DeepFRI nrPDB-GO split as packaged by TorchDrug (Zenodo 6622158 GeneOntology.zip), test cutoffs 30/40/50/70/95 % | Fmax (higher) |
| `fold` | multi-class, 1,195 SCOPe 1.75 folds | GearNet Fold3D (Zenodo 7593591, fold3d.zip, CC BY 4.0), 12,312 / 736 / 718 / 1,272 / 1,254 (train / valid / test_fold / test_family / test_superfamily) | accuracy, one per test split |
| `reaction` | multi-class, 384 EC numbers | IEConv ProtFunct (github.com/phermosilla/IEConv_proteins, Google Drive archive), 29,215 / 2,562 / 5,651 | accuracy |
| `lba` | regression, pK | ATOM3D LBA (PDBbind 2019 refined, 4,463 complexes), 30 % split 3507/466/490, 60 % split 3563/448/452 | RMSE (lower) |
| `mn_bbbp` ... `mn_qm8` (14 MoleculeNet tasks, QM9 excluded) | `multilabel_masked` (BBBP, BACE, ClinTox, Tox21, ToxCast, SIDER, HIV, PCBA, MUV) or `multitarget_regression` (ESOL, FreeSolv, Lipo, QM7, QM8) | Uni-Mol LMDB release, scaffold split as shipped; first conformer, hydrogens dropped, no cropping | ROC-AUC + PRC-AUC (classification); RMSE + MAE (regression; primary: RMSE for ESOL/FreeSolv/Lipo, MAE for QM7/QM8) |

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
| fine-tune | `python -m downstream.finetune --task <task> --model <snapshot> [--random-init]` | GPU training |
| report | `python -m downstream.report --task <task>` | CPU |
| ledger | `python -m downstream.ledger add --outputs-root DIR`, `downstream.ledger_report report`, `downstream.ledger_compare --a A --b B` | CPU, writes only under `$MIND_RESULTS_ROOT` |

For `lba`, `embed`, `probe` and `finetune` also take `--split-tag identity30|identity60`;
`embed` and `probe` additionally take `--representation joint|separate`. Each
combination writes its own embeddings and results directory.

Training-like steps (embed, probe, fine-tune) are user-run through Make targets.

### MoleculeNet (`mn_*`)

`make ds-process TASK=mn_esol` (and `ds-validate`, a read-only dry run) reads the already extracted Uni-Mol LMDB files; there is
no download step. Output: `<source_root>/<task>/processed/<revision>/{train,valid,test}.pt` plus `qc_<split>.json/.tsv`
and a `provenance.json`. Rules (`downstream/tasks/molnet/task.yaml`): first of the 11 conformers; hydrogens dropped;
no atom-count cropping; molecules are never dropped for chemistry reasons but flagged and counted (flat conformers,
rare elements mapped to id 119, one heavy atom, no heavy atom, more than 256 heavy atoms). A molecule left with no
heavy atom is kept but cannot be embedded: loading its graph raises. Missing classification labels (-1 in the LMDB) become
NaN with a mask. The ligand graph is the one of the LBA "separate" ligand path (`dataset_type="unimol"`, 5 A radius, 64
neighbours). The probe trains a masked BCE head (classification) or a masked MSE head on train-standardized targets
(regression), selects the epoch by the primary metric and also reports the QC-flagged subsets of the test set.
`downstream.finetune` refuses these kinds. Embedding with a MIND checkpoint requires CUDA and a checkpoint trained with
`geometry_neutral_virtual_slots` (protein flag set); this path has not been run on GPU.

SaProt (`saprot_650m_pdb`, `saprot_650m_af2`, `saprot_35m_af2`) sees the residues
MIND kept, one `<AA><3Di>` token each; the 3Di state comes from Foldseek on the
same structure file, aligned to the kept residues without mismatches (rules in
`downstream/baselines/foldseek_3di.py`). Residues MIND masks for an incomplete
sidechain, and residues without a Foldseek state, get SaProt's unknown-structure
letter `#`; the amino acid is kept. The 3Di cache lives in `<data_root>/foldseek_3di_<processed_revision>/`
and is valid for that processed revision only.

## Contracts

- `downstream/tasks/ec/task.yaml` must name a missing-atom policy explicitly
  (`missing_atoms`); the checked-in value `UNDECIDED` stops processing.
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
`processed_revision` in `task.yaml` (EC and GO: `v2`) is appended to the processed directory name; v2 reads PDB
chain ids of two or three characters (EC v1 lost 46 chains, GO v1 lost 1,171 as `parse_error`). Every `process`
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
- Missing atoms: same policy and code path as EC (`process_entry`); the adapter's V-padding/slot shift (G5)
  is never reached with an incomplete residue.
- Commands: `make ds-download TASK=go_mf`, `make ds-validate TASK=go_mf` (read-only),
  `make ds-process TASK=go_mf WORKERS=16`; embed/probe/finetune take `--task go_mf|go_bp|go_cc`.

## Leakage registry

`downstream/common/leakage.py` + `downstream/leakage_references.yaml` compare the ids of a task's splits with the
pretraining plans (protein 1M plan, protein 40k manifest, RNA full plan) and write a `leakage` record into the
task's `provenance.json` plus a per-id TSV. Categories: `seen_train`, `seen_val` (used for checkpoint
selection), `seen_test` (pretraining hold-out), `unseen`, `unmapped`. PDB ids against the AFDB (UniProt) set are
refused unless a `pdb_chain -> uniprot` map is given; it never reports "0 leaked" for incomparable namespaces.
Id matching does not detect homologs. `make ds-leakage ...` (see the Makefile header).


## Fold and Reaction (multiclass)

`downstream/tasks/{fold,reaction}/` (`task.yaml`, `dataset.py`, `prepare.py`) on top of `downstream/tasks/multiclass.py`.

- Structures are the archives' own `.hdf5` files (`structure_io.read_hdf5_chain`, needs `h5py`, installed in
  `mind_esa3d` as 3.16.0 and required to process Fold and Reaction). Atom elements come from atom names; `atom_types` is not used. Chains go through the EC/GO code path
  (`process_entry` + missing-atom policy), so a CA-only domain is dropped and counted, never kept. `ProcessedSplit.report`
  and `qc_<split>.json` carry listed / kept / dropped counts.
- Source and unverified items are written at the top of each `task.yaml`. Reaction's Google Drive archive has no published
  size or md5: `download` stops until `archive_sha256` / `archive_bytes` are pinned after reviewing `provenance.json`.
- Probe: cross-entropy head, validation accuracy, one accuracy per test split in `metrics.json`; `downstream.report`
  adds the GearNet-paper literature rows. Fine-tuning refuses multiclass tasks (not implemented).
- Commands: `make ds-download|ds-validate|ds-process TASK=fold` (same for `reaction`).
- Leakage: Fold ids are SCOPe domain ids (`scop_domain`, no map to the AFDB UniProt reference, so the tool refuses);
  Reaction ids are `4xi6.A` (`pdb_chain`, needs `--id-map`). See `queries:` in `leakage_references.yaml`.

## Results ledger

`downstream/ledger.py`, `ledger_report.py`, `ledger_compare.py`; the required-metric catalog is
`downstream/ledger_catalog.yaml` (every metric a reference paper reports for a task; uncomputed ones are marked
`code: missing` and appear as KOD-YOK).

What goes where: Git holds only the code (the three modules, the catalog, the Make targets and the tests). The ledger
data and every generated report stay local under `$MIND_RESULTS_ROOT` (no default in Python; the Makefile passes
`/home/yusuf/mind_results`), as does `literature/literature.csv`; committing the code never commits a result.

Std: ledger reports and comparisons use the sample std over run seeds (ddof=1, undefined for one seed), while
`downstream.report` (`ds-report`) keeps ddof=0, so the same seeds can show two different std values.

- `ledger/measurements.jsonl`: one row per model x run seed x (eval split, subset, metric), with checkpoint sha256,
  training card fields (epoch, `selection_rule`), data revision and processed-file sha256, protocol and its settings
  hash, the probe's own commit (`config.json` `run`; older runs list `run_git` in `missing_provenance`).
  `models.jsonl`, `annotations.jsonl` (superseded / invalid / caveat / smoke / valid), `ingests.jsonl`.
- Append-only: `MANIFEST.json` holds each file's line count and sha256; an edit outside the module stops every
  write and `verify`. Writes hold an exclusive `flock`. Re-ingesting a file (same metrics and config sha256) adds nothing.
  One model name with two checkpoint hashes is an error.
- Reports (`reports/`, regenerated): `tasks/<task>.md|csv` (one "Ayar" section per comparable setting: protocol,
  pooling, data revision and fingerprint, settings hash; superseded rows excluded), `timeline/<task>.md|csv` (all
  statuses), `ozet_hoca.md` (primary metric per task, counts of required / computed / missing cells),
  `eksik_hucreler.md|csv` (each missing cell with its category and the command that fills it). Literature comes from
  `<root>/literature/literature.csv` (local) and is shown in its own column, unit and protocol.
- `ledger_compare`: cell-by-cell A vs B with probe-seed std, a paired bootstrap interval over `predictions.npz` (chains
  aligned by id; each seed's predictions must reproduce the ledger value) and the unmeasured pretraining-seed noise
  stated in a warning box, plus the expected number of chance "differences".
- Commands: `make ds-ledger-add OUTPUTS_ROOT=<dir> [EVAL_DIRS=...]`, `ds-ledger-annotate`, `ds-ledger-report`,
  `ds-summary`, `ds-missing`, `ds-compare A=<model> B=<model>`, `ds-ledger-verify`. `ds-report` is unchanged.
  Report models: `MODELS=...`, else `summary_models` in `<root>/comparisons.yaml`, else `DEFAULT_SUMMARY_MODELS` in
  `ledger_report.py`; a listed model the ledger has never seen appears as `EKSİK:model-henüz-yok`, never skipped.
- `make ds-ledger-recompute MODEL=<m> CONTROL=<m2> DATA_REVISION=<rev> [TASK=ec] [DRY_RUN=1]`: adds the macro AUPR a
  frozen probe did not record (EC/GO), computed from each seed's `predictions.npz` with the probe's own function. It
  writes nothing unless the metrics files are unchanged since ingestion, fmax and pair AUPR are reproduced exactly,
  the CONTROL model's recorded macro AUPR is reproduced by the same formula, and no macro AUPR row exists yet. New
  rows name their source in `recomputed_from` and stay in the "Ayar" section of the file they come from.

## One command per model (`make ds-run`)

`downstream/run_all.py`: `make ds-run MODEL=<snapshot | path/to.ckpt> [NAME=<snapshot>] [TASKS=all|ec,go_mf,...]
GPU=2,3 [PER_GPU=1] [HEADS=linear,mlp] [PROBE_JOBS=1] [RANDOM_INIT=1] [OUTPUTS_ROOT=<dir>] [PLAN=1]
[COMPARE_TO=<snapshot[.random_init],...>] [BOOTSTRAP=1000] [COUNTERFACTUAL=1 [CF_MODES="..."] [EVAL_CONFIG=<config>]]
[FOREGROUND=1]`.
A checkpoint path is first snapshotted as NAME (a NAME that already names another checkpoint, in the snapshot
store or in the ledger, stops the run). Then one embedding job per task on the GPUs (at most PER_GPU per GPU; splits
already embedded are skipped), then, after every embedding job has ended, the probes on CPU (SEEDS; seeds already
done are skipped), `ledger add`, and a table of the primary metric per task and head plus the missing required
cells. `all` is EC, GO, Fold, LBA (identity30 and identity60, joint) and the 14 MoleculeNet tasks; Reaction only
when named. A task outside the model's pretraining domains (protein needs `pdb`, MoleculeNet `unimol`) is skipped
and listed under `all`, refused when named. A failing job is reported with its log and the others continue; the
run ends with the list of what is missing and exit code 1. `PLAN=1` prints what would run (and recorded embedding
times of other models, when there are any) and writes nothing. Logs and `summary.md` go to
`logs/<Month><Year>/ds_run_<model>_<stamp>_<pid>/`; fine-tuning is not part of this command.

- `COMPARE_TO=a,b.random_init`: every listed model goes through the same tasks, heads and seeds (what is complete
  is skipped; `.random_init` is the RANDOM_INIT=1 control of that snapshot), gets its own table, and at the end
  `ds-compare A=<each> B=MODEL` runs (BOOTSTRAP resamples). The last step of every ds-run is the ledger report
  (`make ds-ledger-report`: `tasks/`, `timeline/`, `ozet_hoca.md`, `eksik_hucreler.*` under `$MIND_RESULTS_ROOT/reports`).
- `COUNTERFACTUAL=1` (MODEL must be a `.ckpt` path): the `make eval-counterfactual` sweep (same arguments; CF_MODES,
  SEED, DOMAINS=unimol, EVAL_CONFIG) on the snapshot copy, one mode per GPU job after the embeddings, written to
  `<outputs root>/eval/cf_<NAME>_<stamp>/<mode>` and added to the ledger. A failing mode is listed, the others go on.
- Example (one model, two references, all tasks, two GPUs):
  `make ds-run MODEL=/path/to/best-checkpoint.ckpt NAME=<new snapshot> OUTPUTS_ROOT=<outputs root> GPU=2,3
  COMPARE_TO=<reference snapshot>,<reference snapshot>.random_init` (first with `PLAN=1`).

**Every long ds-* job keeps running when the terminal or SSH connection closes.** `ds-run` and `ds-probe` start in
their own session (`setsid nohup`, no terminal) and return at once with the log path; the log's last line is
`EXIT_CODE=<n>` of the Python command (non-zero when it failed, as `pipefail` gave before). `make ds-status [N=15]`
lists this month's `ds_*.log` files with that line (no line: still running or killed; check the process with
`make status`). `FOREGROUND=1` runs them in the terminal as before (`| tee`); `PLAN=1` always runs in the terminal.
`ds-embed`, `ds-finetune`, `eval-pretrain` and `eval-counterfactual` were already `nohup ... &`. Short CPU steps
(`ds-snapshot`, `ds-report`, `ds-ledger-*`, `ds-compare`, `ds-summary`, `ds-missing`) and the data steps
(`ds-download`/`ds-process`/`ds-validate`/`ds-leakage`, run on explicit decision) stay in the foreground.

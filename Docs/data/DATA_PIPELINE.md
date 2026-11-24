# MIND data pipeline

This page describes the data path used by the current training code. Raw and
processed datasets live outside this repository; the paths below are examples
from the checked-in configurations.

## Pipeline

```text
raw structures / molecule records
        │
        ├─ adapter → UniversalMolecule / UniversalBlock / UniversalAtom
        ├─ streaming pickle cache (.pkl)
        ├─ cache_to_pyg.py → one PyG container per chunk (.pt)
        ├─ create_chunk_metadata.py → sidecar metadata (.json)
        └─ train_pretrain.py → LazyUniversalDataset → dynamic batches
```

For proteins and RNA, the preparation workflow builds edge-free chunks directly from raw
structures (`mind_pt_schema`; no `.pkl` cache and no stored edges):

```bash
make build-edge-free DOMAIN=protein MANIFEST=<csv> RAW_DIR=<dir> DEST_PREFIX=<new prefix> CHUNK_SIZE=4000
make build-edge-free DOMAIN=rna MANIFEST=<csv> RAW_DIR=<dir> DEST_PREFIX=<new prefix> CHUNK_SIZE=200 SPLIT_GROUP=pdb-entry
```

[`data_loading/build_edge_free_chunks.py`](../../data_loading/build_edge_free_chunks.py) assigns
train/val/test by SHA-256 of the structure ID (or PDB entry), shuffles within each split,
writes resumable chunks `<prefix>_<split>_chunk_NNN` and records failures in the metadata.
The loader is described in [`data_loading/pt_schema.py`](../../data_loading/pt_schema.py).

### RNA: from the public RNA3DB release to chunks

The RNA source is [RNA3DB](https://github.com/marcellszi/rna3db), which publishes one mmCIF
per chain for each release. Three commands take a release to chunks; none asks for input.

```bash
make rna-prepare RELEASE=2026-01 DEST=<new dir> WORKERS=24      # download, extract, manifest (detached, CPU)
make build-edge-free DOMAIN=rna MANIFEST=<DEST>/manifest_rna.csv RAW_DIR=<DEST>/raw_structures \
     DEST_PREFIX=<new prefix> CHUNK_SIZE=200 SPLIT_GROUP=pdb-entry WORKERS=24
```

`rna-prepare` runs [`data/rna/download_rna3db.py`](../../data/rna/download_rna3db.py) and then
[`data/rna/create_rna_manifest.py`](../../data/rna/create_rna_manifest.py); its log ends with
`EXIT_CODE=<n>`. `DEST` must not exist (`RESUME=1` continues an interrupted run of the same release).
Nothing is overwritten. `RESUME=1` on a finished destination re-verifies it and exits 0: the download is
re-checked against `split.json`, and the manifest is checked against the current inputs (release, `split.json`
digest, raw file count, thresholds, manifest digest, chain set). A manifest that does not match fails.

| `RELEASE` | upstream tag | chains in `split.json` |
|---|---|---|
| `2024-01` | `full-release` | 11,176 |
| `2024-04` | `incremental-update` | 11,622 |
| `2024-12` | `2024-12-04-full-release` | 12,892 | <!-- main-ready: allow RNA3DB release tag (data-source provenance) -->
| `2025-10` | `2025-10-01-incremental-release` | 14,774 | <!-- main-ready: allow RNA3DB release tag (data-source provenance) -->
| `2026-01` | `2026-01-05-full-release` | 15,441 | <!-- main-ready: allow RNA3DB release tag (data-source provenance) -->

The release is a required argument, because releases differ in size and content and the choice
must be explicit. Releases are almost cumulative: each one contains nearly all chains of the previous one (a few chains are dropped between releases).

- **Download and integrity.** The tarball sizes are pinned for every release. SHA-256 is pinned for
  `2025-10` and `2026-01` (the digests the release host publishes); for the others the digest is
  computed on the first download and stored in `download_provenance.json`, and a later run that sees a
  different digest is refused. `--expected-sha256` supplies a known digest. Extraction never uses archive
  paths: a member name with `..`, an absolute path, a link or an unexpected file type is an error.
- **Lowercase chain ids.** The tarball writes a lowercase run in a chain id between dashes
  (`7taw_-m-.cif` is chain `m`) to survive case-insensitive file systems. The tool writes the decoded
  name `7taw_m.cif`, which is what `split.json` and the builder use. Chains that differ only by case
  (`1abc_A`, `1abc_a`) are distinct files, so the destination must be on a case-sensitive file system
  (checked before extraction). Skipping the decoding loses every lowercase chain (11.6 % to 13.0 % of the chains, depending on the release).
- **Completeness check.** The decoded file names must equal the chain set of the release `split.json`
  and the pinned chain count; any difference fails the run. The record of archives, digests and counts is
  `download_provenance.json`.
- **Manifest.** Columns: `name` (`<pdb>_<chain>`, read by the builder), `pdb_id`, `chain`,
  `rna3db_set`, `release_date`, `structure_method`, `resolution`, `length`. Rows are sorted by name, so
  one release and one set of thresholds give one file. Two stages select chains. First, the RNA3DB
  filters with the release defaults: length >= 32 nt, resolution <= 9.0 A (NMR counts as infinite),
  at most 80 % of one nucleotide, at most 30 % `N`; the RNA3DB chain selection itself (every
  polyribonucleotide chain, DNA/RNA hybrids included, no pure DNA) happens upstream. A published
  `split.json` passes these filters unchanged, so they reject nothing unless the thresholds are
  tightened. Second, every remaining chain goes through the builder's own worker, and chains it rejects
  (a P-only phosphate trace is the known case) are dropped with the reason, because every builder
  failure counts against the per-chunk failure limit. Any other conversion error stops the run. Third,
mirror-image (left-handed) chains are dropped with the criterion of
[`audit_rna_anchor_selection.py`](../../data_loading/audit_rna_anchor_selection.py): in blocks with an exact
12-atom backbone, O2' lies on the non-positive side of the frame built from C2', C3', C4', and a molecule is a
mirror image when more than half of those blocks do. Blocks with degenerate anchors are not counted.
  `manifest_rna_provenance.json` holds the thresholds, input digests, counts per rejection reason (including
  `mirror_image_chains`) and the rejected chains.
- **Not applied.** The manifest applies no size limit; training-time atom limits are set in the training configs.
- **Resources.** The mmCIF tarball is about 2 GB (xz) and extracts to text files of roughly 1.3 MB per
  chain on average, so plan for a few tens of GB. The manifest check converts every chain once, which
  takes about as long as the chunk build itself; it parallelizes over `WORKERS`.

Also in `data/rna/`: `filter_rna_manifest.py` (an optional atom-count filter for an existing
manifest) and `filter_dataset.py` (a legacy length filter with hard-coded paths, unused) are not part of this
workflow.

COCONUT is built from the raw 3D SDF by one recorded command, `make build-coconut` (see
[COCONUT build](#coconut-build)). UniMol still uses the two-stage pipeline orchestrated by
[`data_loading/process_chunked_dataset.py`](../../data_loading/process_chunked_dataset.py).
It creates a universal cache with
[`data_loading/cache_universal_datasets.py`](../../data_loading/cache_universal_datasets.py)
and converts it with [`data_loading/cache_to_pyg.py`](../../data_loading/cache_to_pyg.py).

Data processing is separate from training: the scripts above run on their own
(`make build-edge-free` for protein and RNA) and never start a training run.
Generated caches remain outside Git.

## COCONUT build

```bash
make build-coconut INPUT=<coconut.sdf|coconut.zip> DEST_PREFIX=<new prefix> WORKERS=32
python -m data_loading.build_coconut --verify-only --dest-prefix <prefix>
```

[`data_loading/build_coconut.py`](../../data_loading/build_coconut.py) goes from the raw SDF (or
the official zip, extracted next to the destination and removed on success) to chunks in
one run. It never overwrites; an interrupted run resumes only when its recorded plan and
parameters are identical.

- A record is one conformer; a molecule has several. A record is rejected, in this order,
  when it has fewer than 10 heavy atoms, an element outside H, C, O, N, S, Cl, Br, P, I, Na,
  F, or only z = 0 heavy atoms (a 2D depiction). Each reason is counted. A record that cannot
  be parsed aborts the build with its byte offset.
- Splits are assigned per molecule (SHA-256 of salt and SMILES), so all conformers of a
  molecule share one split; the SDF is ordered by molecule size, so the split never depends
  on the record position. Records are shuffled within a split and cut into balanced chunks.
  Related stereoisomers or tautomers with different SMILES strings can still fall in
  different splits.
- A record that passes the filters but fails the adapter (PS_300 tokenizer: for example any
  sodium-containing molecule, or an invalid valence) is dropped and recorded with its id
  and reason in the chunk metadata and `<prefix>_failures.csv`. Failures are not
  uniform in size (they grow with heavy-atom count), so the built set slightly
  under-represents large molecules.
- The books must close exactly: raw records = rejected + failed + built, planned chunks =
  built chunks, and every chunk holds exactly its planned records minus its recorded
  failures. The run exits non-zero otherwise (`status: failed_verification`).
- Output: `<prefix>_{train,val,test}_chunk_NNN/processed/` in the stored-edge layout of
  `cache_to_pyg.py` (radius edges at 5 A with 64 neighbours, `bond_edge_index`) plus the
  metadata sidecar (with `mol_id = coconut_<record index in the SDF>`), and next to them
  `<prefix>_plan.csv.gz` (record, split, chunk, molecule, conformer), `_rejected.csv`,
  `_failures.csv` and `_build_manifest.json` (input and SDF SHA-256, filter version,
  parameters, tokenizer vocabulary hash, counts per stage and split, git revision).

A config reads the result with fixed splits and no chunk ranges:

```yaml
metabolite:
  enabled: true
  splits:
    train: {chunk_dir: <prefix>_train, chunk_range: null}
    val:   {chunk_dir: <prefix>_val,   chunk_range: null}
    test:  {chunk_dir: <prefix>_test,  chunk_range: null}
```

Covered by [`tests/test_build_coconut.py`](../../tests/test_build_coconut.py).

## Representation contracts

Adapters produce `UniversalMolecule` objects defined in
[`data_loading/data_types.py`](../../data_loading/data_types.py). A processed
PyG graph is expected to carry, where applicable, `z`, `pos`, `edge_index`,
`edge_attr`, `block_idx`, `entity_idx`, `pos_code`, `mol_id`, and
`dataset_type`. The production tests use these fields rather than synthetic
aliases; see [`tests/test_adapters.py`](../../tests/test_adapters.py).

Protein graphs use fixed 14-slot residues (4 backbone + 10 sidechain) and RNA
graphs fixed 23-slot nucleotides (12 backbone + 11 base); slots without a real atom
are virtual `V` padding. The contract, the PyG fields, the atom-id vocabulary and the
legacy-id repair (done in memory by the loader; production `.pt` files are never
rewritten) are defined once in [data representation](DATA_REPRESENTATION.md). The
code is [`protein_adapter.py`](../../data_loading/adapters/protein_adapter.py),
[`rna_adapter.py`](../../data_loading/adapters/rna_adapter.py),
[`cache_to_pyg.py`](../../data_loading/cache_to_pyg.py) and
[`lazy_universal_dataset.py`](../../data_loading/lazy_universal_dataset.py).

## RNA subset derivation

An RNA subset can be derived from existing PyG chunks without changing the production
source. [`data_loading/build_rna_baseline_subset_manifest.py`](../../data_loading/build_rna_baseline_subset_manifest.py)
counts real non-hydrogen atoms (virtual padding excluded) and records stable molecule
IDs with source chunk/index provenance;
[`data_loading/derive_rna_baseline_subset_pt.py`](../../data_loading/derive_rna_baseline_subset_pt.py)
re-collates only the manifest-selected graphs into a new immutable output and refuses
to overwrite an existing one. The selection is defined on real atoms, not on
`num_nodes` (a 23-slot graph contains padding), and the source PTs are checked for
unchanged fingerprints. `tests/test_rna_baseline_subset.py` covers the manifest
semantics and source immutability.

## Chunk conversion and metadata

`cache_universal_datasets.py` streams molecules into pickle rather than holding
the whole source corpus in RAM. `cache_to_pyg.py` converts each molecule to a
PyG graph (radius-graph edges) and writes the collated chunk; this is the route for
UniMol and COCONUT. Protein and RNA do not use it: `make build-edge-free` writes
edge-free chunks and the runtime rebuilds the three-tier protein graph (see
[hybrid edges](../architecture/HYBRID_EDGES.md)).

Every chunk used by the improved sampler needs a metadata sidecar (the default
sampler; `build_edge_free_chunks.py` writes it for protein and RNA). Generate it
for other chunks with:

```bash
python data_loading/create_chunk_metadata.py \
  --chunk-dir /path/to/processed_chunk/processed
```

The sidecar records dataset type, sample count, atom and edge totals, and
per-sample atom/edge counts. These counts are authoritative for atom-aware
packing. Sidecar failure behavior, as implemented:

- `LazyUniversalDataset` does **not** fail on a missing or unreadable sidecar (an
  invalid JSON file is caught and counted as missing): it prints a warning and falls
  back to metadata-free mode
  ([`lazy_universal_dataset.py`](../../data_loading/lazy_universal_dataset.py)).
- With atom-budget batching (`use_dynamic_batching` and the default
  `use_improved_sampler: true`, as in every multi-domain config)
  `ImprovedDynamicBatchSampler` then raises `Metadata is required for atom-aware
  batching`, and `max_graph_atoms` raises `max_graph_atoms requires complete chunk
  metadata`. A run with these configs therefore stops at start-up.
- A chunk whose PT schema (`mind_pt_schema` or not) disagrees with its sidecar raises
  when the chunk is loaded.
- Edge totals in the sidecar of an edge-free protein chunk count only the Tier-1
  edges stored at load time; atom counts are exact.

## Loading and batching

[`LazyUniversalDataset`](../../data_loading/lazy_universal_dataset.py) builds a
selection-specific index map and loads chunk contents on demand with an LRU
cache. `train_pretrain.py` discovers sibling directories named
`<base>_chunk_*` (sorted by name, so UniMol chunk 10 sorts before chunk 2),
applies an inclusive `chunk_range`, and combines enabled domains into one
dataset. An enabled domain whose `chunk_dir` has no chunks stops the run with an error
(`FileNotFoundError`); it is not skipped, so a run never trains on less data than the
config says. An unfinished `.partial` chunk
directory raises.

Instead of `chunk_dir`/`chunk_range`, a domain may give fixed splits:

```yaml
splits:
  train: {chunk_dir: <prefix>_train, chunk_range: null}
  val:   {chunk_dir: <prefix>_val,   chunk_range: [0, 0]}
  test:  {chunk_dir: <prefix>_test,  chunk_range: [0, 0]}
```

Every chunk of a split prefix then belongs to that split as a whole. `splits` and
`chunk_dir`/`chunk_range` cannot be combined, `train` is required, only
`train`/`val`/`test` keys are accepted and an empty split raises. Domains without
`splits:` are pooled and split randomly 80/10/10 with the run seed
(`split_dataset` in [`train_pretrain.py`](../../core/train_pretrain.py)). The
1M protein, full RNA, UniMol (train chunks 0-197, val 198, test 199) and COCONUT
(0-494 / 495 / 496) configs use fixed splits.

With dynamic batching, [`ImprovedDynamicBatchSampler`](../../data_loading/improved_dynamic_sampler.py)
packs complete graphs up to `max_atoms_per_batch`. That value is a packing
target, not a safe limit for an individual graph. `max_graph_atoms` is the
separate per-domain limit, checked before train/validation/test splitting. By
default (`oversized_graphs: error`) an oversized graph raises with its structure id;
with `oversized_graphs: skip` (protein and RNA in the 1M configs) it is logged,
excluded from every split and counted in the start-up message. The fallback sampler remains available for compatibility and
must be treated as legacy when interpreting old runs.

Two epoch policies exist, selected by `multi_domain.atom_sampling`:

- Natural full pass (`pretraining_config_multidomain.yaml`, the three-domain
  baseline; `atom_sampling.enabled:
  false`): the sampler plan targets every selected train graph once per epoch, with
  deterministic proportional interleaving while domains remain available. It does not
  cycle short domains or enforce atom quotas; any tail after a domain is exhausted is a
  consequence of the natural dataset sizes.
- Budgeted virtual epochs (`atom_sampling.enabled: true`: the 1M and fast configs):
  a fixed `epoch_atom_budget` is split by `domain_weights` (for example protein 55 /
  RNA 15 / UniMol 20 / COCONUT 10), the plan has a fixed `epoch_num_batches`, and
  `cycle_to_quota: true` repeats a short domain to reach its quota. An epoch is not a
  pass over the corpus.

Validation and test are single-pass and unshuffled. A full-pass claim requires log
verification: compare each epoch plan's batch count and per-domain selected/unique
counts with the consumed progress and callback summary. A plan can be longer than the
batch count cached by the training loop: Lightning keeps the initial dataloader batch
count while the sampler rebuilds a plan of a different length at `set_epoch` (the
Lightning epoch-length issue); the
constant plan length of the budgeted configs is meant to avoid it.

How to check it in a log: the sampler prints `Ready! ... batches created` for each
epoch plan; compare that number with the consumed progress count and with the
per-domain selected versus tracked totals. A sampler line reporting `mode=full_pass`
describes the plan only; it does not show that Lightning consumed all of it. With
`atom_sampling.enabled: false`, `epoch_atom_budget` and `domain_weights` are unused
and `cycle_to_quota` is `false`. Dynamic batching is also used for validation and test
when enabled; those loaders are deterministic and single-pass.

## Integrity checks

Before using a new chunk, check metadata, graph tensor lengths, edge bounds,
finite coordinates, molecule IDs, and domain labels. For a requested adapter
or pipeline change, the focused checks are:

```bash
pytest -v tests/test_adapters.py tests/test_build_edge_free_chunks.py tests/test_build_coconut.py tests/test_rna_baseline_subset.py
```

Then inspect one generated graph with `torch.load` and confirm `pos` has shape
`[N, 3]`, all node-aligned fields have length `N`, and a protein residue still
has 14 slots.

## Related code

- [`data_loading/lazy_universal_dataset.py`](../../data_loading/lazy_universal_dataset.py)
- [`data_loading/chunk_sampler.py`](../../data_loading/chunk_sampler.py)
- [`data_loading/improved_dynamic_sampler.py`](../../data_loading/improved_dynamic_sampler.py)
- [`data_loading/create_chunk_metadata.py`](../../data_loading/create_chunk_metadata.py)
- [`core/train_pretrain.py`](../../core/train_pretrain.py)

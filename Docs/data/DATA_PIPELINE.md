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

For proteins, the supported preparation workflow is exposed by the Makefile:

```bash
make process-protein NUM_CHUNKS=10
make process-protein NUM_CHUNKS=10 PARALLEL=2
```

The target paths and arguments are defined in [`Makefile`](../../Makefile).
Its default includes `--force` and reuses the 40k output prefix. Review and
override those paths for a new experiment; do not run it as a migration check.
The underlying orchestrator is
[`data_loading/process_chunked_dataset.py`](../../data_loading/process_chunked_dataset.py).
It creates a universal cache with
[`data_loading/cache_universal_datasets.py`](../../data_loading/cache_universal_datasets.py)
and converts it with [`data_loading/cache_to_pyg.py`](../../data_loading/cache_to_pyg.py).

Data processing is separate from training. Processing may be run directly when
the user explicitly requests it; generated caches remain outside Git.

## Representation contracts

Adapters produce `UniversalMolecule` objects defined in
[`data_loading/data_types.py`](../../data_loading/data_types.py). A processed
PyG graph is expected to carry, where applicable, `z`, `pos`, `edge_index`,
`edge_attr`, `block_idx`, `entity_idx`, `pos_code`, `mol_id`, and
`dataset_type`. The production tests use these fields rather than synthetic
aliases; see [`tests/test_adapters.py`](../../tests/test_adapters.py).

Protein graphs use the fixed 14-slot residue representation: four backbone
slots plus ten sidechain slots. Missing sidechain slots are virtual `V` atoms.
RNA graphs in the current June-equivalent dataset use 23 slots per nucleotide:
12 backbone/sugar/phosphate slots plus 11 base slots, filled with real atoms or
virtual padding as available. See
[`data_loading/adapters/protein_adapter.py`](../../data_loading/adapters/protein_adapter.py)
and [`data_loading/adapters/rna_adapter.py`](../../data_loading/adapters/rna_adapter.py).

The model distinguishes virtual padding from rare real elements. Current
conversion uses the explicit virtual model id and the loader repairs legacy
cache ids in memory; it does not rewrite production `.pt` files. This behavior
is implemented by [`data_loading/cache_to_pyg.py`](../../data_loading/cache_to_pyg.py)
and [`data_loading/lazy_universal_dataset.py`](../../data_loading/lazy_universal_dataset.py).

## Chunk conversion and metadata

`cache_universal_datasets.py` streams molecules into pickle rather than holding
the whole source corpus in RAM. `cache_to_pyg.py` converts each molecule to a
PyG graph and writes the collated chunk. Protein preprocessing can add the
three-tier hybrid graph (sequence, spatial, and random long-range edges); the
parameters are in the YAML config and the implementation is in
[`data_loading/cache_to_pyg.py`](../../data_loading/cache_to_pyg.py).

Every chunk used by the improved sampler needs a metadata sidecar. Generate it
with:

```bash
python data_loading/create_chunk_metadata.py \
  --chunk-dir /path/to/processed_chunk/processed
```

The sidecar records dataset type, sample count, atom and edge totals, and
per-sample atom/edge counts. These counts are authoritative for atom-aware
packing; a missing or inconsistent sidecar should fail loudly.

## Loading and batching

[`LazyUniversalDataset`](../../data_loading/lazy_universal_dataset.py) builds a
selection-specific index map and loads chunk contents on demand with an LRU
cache. `train_pretrain.py` discovers sibling directories named
`<base>_chunk_*`, applies an inclusive `chunk_range`, and combines enabled
domains into one dataset.

With dynamic batching, [`ImprovedDynamicBatchSampler`](../../data_loading/improved_dynamic_sampler.py)
packs complete graphs up to `max_atoms_per_batch`. That value is a packing
target, not a safe limit for an individual graph. `max_graph_atoms` is the
separate hard pre-split invariant; oversized graphs fail before train/validation/
test splitting. The fallback sampler remains available for compatibility and
must be treated as legacy when interpreting old runs.

The approved full multidomain policy sets `atom_sampling.enabled: false` and
requests a natural full pass: the sampler plan targets every selected train
graph once per epoch, with deterministic proportional interleaving while
domains remain available. It does not cycle
short domains or enforce the retired atom quotas; any tail after a domain is
exhausted is a consequence of the natural dataset sizes, not quota cycling.
Validation and test are also single-pass and unshuffled. Quota/cycling code is
kept only for focused regression tests and historical experiments. A full-pass
claim still requires log verification: compare each epoch plan's batch count
and per-domain selected/unique counts with the consumed progress and callback
summary. A plan can be longer than the batch count cached by the training loop.

## Integrity checks

Before using a new chunk, check metadata, graph tensor lengths, edge bounds,
finite coordinates, molecule IDs, and domain labels. For a requested adapter
or pipeline change, the focused checks are:

```bash
pytest -v tests/test_adapters.py
make process-protein NUM_CHUNKS=1
```

Then inspect one generated graph with `torch.load` and confirm `pos` has shape
`[N, 3]`, all node-aligned fields have length `N`, and a protein residue still
has 14 slots. Do not use the removed `data_loading/validate_adapters.py` path.

## Related code

- [`data_loading/lazy_universal_dataset.py`](../../data_loading/lazy_universal_dataset.py)
- [`data_loading/chunk_sampler.py`](../../data_loading/chunk_sampler.py)
- [`data_loading/improved_dynamic_sampler.py`](../../data_loading/improved_dynamic_sampler.py)
- [`data_loading/create_chunk_metadata.py`](../../data_loading/create_chunk_metadata.py)
- [`core/train_pretrain.py`](../../core/train_pretrain.py)

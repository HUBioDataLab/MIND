# RNA datasets and current 23-slot input

## Current locations

The current approved multidomain configuration
([`core/pretraining_config_multidomain.yaml`](../../core/pretraining_config_multidomain.yaml))
uses:

| Item | Path / policy |
|---|---|
| Raw RNA source | `/media/16TB/archive/yusuf/MIND/data/raw/rna/arya_rna_data/` |
| Current full-training base | `/home/yusuf/data/rna/virtual_v1/june_equivalent_23slot_v2/processed_23slot` |
| Full-training chunks | Sibling directories `processed_23slot_chunk_*` discovered by the loader |
| Protein comparison | `/home/yusuf/data/proteins/processed_14atom` |
| Small-molecule comparison | `/home/yusuf/data/small_molecules/unimol/universal_unimol` |

The fast YAML currently points at the separate development base
`/home/yusuf/data/rna_small/processed_23slot_5k` and selects chunk `0`. This is
an intentionally smaller development artifact, not the full June-equivalent
RNA corpus. Always inspect the resolved config and loader output before a run.

## June-equivalent selection

The current RNA selection was derived from existing PyG chunks without changing
the production source. The manifest builder
[`data_loading/build_rna_june_equivalent_manifest.py`](../../data_loading/build_rna_june_equivalent_manifest.py)
counts real non-hydrogen atoms, excluding virtual padding, and records stable
molecule IDs and source chunk/index provenance. The derivation tool
[`data_loading/derive_rna_june_equivalent_pt.py`](../../data_loading/derive_rna_june_equivalent_pt.py)
re-collates only the manifest-selected graphs into a new immutable output.

The selection is not equivalent to filtering by `num_nodes`: a 23-slot graph
contains padding, while the historical policy is defined on real non-hydrogen
atoms. Outputs are written to a new base and the source PTs are checked for
unchanged fingerprints. The tools refuse to overwrite an existing immutable
output.

The checked-in test covers both manifest semantics and source immutability:

```bash
pytest -q tests/test_rna_june_equivalent.py
```

## 23-slot contract

Each accepted canonical nucleotide has 12 backbone/sugar/phosphate slots and
11 base slots. Standard bases have the following real/V base-slot pattern:

| Base | Real base slots | Virtual slots |
|---|---:|---:|
| A | 10 | 1 |
| G | 11 | 0 |
| C | 8 | 3 |
| U | 8 | 3 |

The 11-slot capacity is a per-nucleotide limit. The metric position vocabulary
is a union across bases and may contain more than 11 names; it must not be
confused with the physical slot count. Real modified/OOV positions must not be
silently relabeled as virtual.

The loader keeps legacy compatibility in memory: semantic virtual position
codes are mapped to the explicit virtual model id, and legacy `BR` labels are
restored to bromine. Persisted source PTs are not modified by this repair.

## Long graphs and coverage

`max_graph_atoms` is checked before splitting. The current full config uses an
RNA graph limit of 6,000 and a June-equivalent derived selection designed to
exclude unsafe oversized graphs. `max_atoms_per_batch` remains a packing target;
it cannot make one graph smaller. Long-RNA windowing is not part of the current
input contract, so a future all-corpus expansion must first define and validate
an explicit window/subgraph representation.

Do not infer corpus size from the number of output chunks: one source group can
be re-chunked into many output directories. Old RNA caches and old handoff
counts are historical evidence only; their exact molecule overlap with current
outputs is not assumed.

## RNA validation

Use focused adapter and selection tests before any training command:

```bash
pytest -v tests/test_adapters.py tests/test_rna_june_equivalent.py
```

For a training smoke check, use the repository Make target and inspect the
resolved RNA path, chunk count, metadata, finite losses, and RNA metrics in the
log. Training is user-started; this page does not launch it.

Related implementation: [`data_loading/adapters/rna_adapter.py`](../../data_loading/adapters/rna_adapter.py),
[`data_loading/atom_position_vocab.py`](../../data_loading/atom_position_vocab.py),
[`core/metrics/rna_per_atom_metrics_callback.py`](../../core/metrics/rna_per_atom_metrics_callback.py).

# Data representation: fixed slots, atom ids and RNA blocks

This page is the single source for the fixed-slot contract (protein 14 slots, RNA 23
slots), the model atom-id vocabulary and the RNA block layout. How the raw data become
chunks and batches is in the [data pipeline](DATA_PIPELINE.md); what the model does
with the slots at run time (the geometry-neutral view) is in
[virtual slots](../architecture/VIRTUAL_SLOTS.md).

## Fixed-slot contract

| Domain | Block | Backbone slots | Sidechain / base slots | Slots per block |
|---|---|---|---|---|
| Protein | canonical residue | 4 (`N`, `CA`, `C`, `O`) | 10 | 14 |
| RNA | nucleotide | 12 | 11 | 23 |
| UniMol, COCONUT | none (atom-level) | - | - | prepared atom sets, no padding |

Slots that hold no real atom are virtual padding atoms: target class `V`, model id 120,
`is_virtual` true, `pos_code` `V<slot number>`. The fixed width is an output/cache
contract, not a statement that all 14 or 23 nodes are physical atoms during model
encoding: the model removes visible `V` nodes and all masked sidechain/base nodes before
ESA (see [virtual slots](../architecture/VIRTUAL_SLOTS.md)). The shared encoder sees
protein, RNA and small-molecule graphs; protein and RNA block MLM masks complete blocks,
UniMol is masked atom by atom.

## Protein blocks: raw structures to fixed-width blocks

`data_loading/adapters/protein_adapter.py` reads a `.pdb` or `.cif` file with
BioPython (any other extension yields no blocks) and walks all models in the file;
AlphaFold DB files contain a single model. Atom filtering:

- atoms whose `element` field (PDB columns 77-78, mmCIF `type_symbol`, upper-cased)
  is `H` are dropped; the atom name is not inspected, an empty element is treated as
  `C`, and deuterium `D` is kept;
- of the alternate locations, only blank or `A` are kept.

Residues whose hetero flag is blank, or that `is_aa(standard=True)` accepts, become
`UniversalBlock`s; heteroatom residues are skipped unless the adapter is built with
`include_hetatms=True` (a constructor argument, not a YAML key). Each chain gets its
own `entity_idx`, so sequence edges cannot cross chains.

`use_14atom_uniform` is likewise a constructor argument (default `True`), not a YAML
key; it is the processed protein data contract. With it, each standard residue is
emitted in this order:

```text
N, CA, C, O, [up to ten canonical sidechain positions]
```

Backbone atoms are taken in file order (N, CA, C, O in AlphaFold files); the adapter
does not check for a missing backbone atom. The sidechain atoms are sorted by the
shared priority list in `data_loading/atom_position_vocab.py` (names not in the list go
last), real atoms first and virtual atoms after them, so slot i holds the i-th
available atom rather than a fixed atom name; if a real atom is absent, later atoms
shift up. More than ten sidechain atoms are silently dropped (the canonical amino
acids have at most ten, TRP). The padding slots are virtual atoms whose `pos_code` is
`V<slot number>` (`V1` ... `V10`; for example ALA, with only CB, carries `V2` ... `V10`),
`is_virtual` is true, and model id is 120. Their coordinates are CB for residues with CB, or CA for glycine
and fallback cases. `OXT` is excluded from the ten sidechain slots; legacy
terminal records are handled separately by masking policy.

The fixed-width invariant is four backbone plus ten sidechain slots = 14 nodes
per canonical residue. A malformed block that gets
masked is quarantined by `core/masked_slot_mapping.py`, and the geometry-neutral
view then raises a `RuntimeError`.

## RNA blocks

RNA uses twelve backbone positions:

```text
P, OP1, OP2, O5', C5', C4', O4', C3', O3', C2', O2', C1'
```

Each nucleotide has up to eleven base positions; the shared vocabulary also
recognizes supported modified-base labels. RNA fixed blocks therefore have
12 + 11 = 23 slots when standardized. Legacy aliases such as `O1P` and
`C1*` are normalized on the RNA path only. Optional phosphate `OP3` is
terminal metadata, not a fixed base slot. Each accepted canonical nucleotide has 12
backbone/sugar/phosphate slots and 11 base slots; standard bases have the following
real/V base-slot pattern:

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

Long graphs: `max_graph_atoms` is checked before splitting (see the
[data pipeline](DATA_PIPELINE.md#loading-and-batching)). The multi-domain
configs set the RNA limit to 6,000 nodes (5,000 in `pretraining_config_fast.yaml`); the full RNA corpus (12,869 structures, split
by PDB entry) has a largest graph of 102,349 nodes in its sidecar metadata (`num_atoms`,
virtual padding slots included). `max_atoms_per_batch` remains a packing target; it cannot make one graph
smaller. Long-RNA windowing is not part of the current input contract, so a future
all-corpus expansion must first define and validate an explicit window/subgraph
representation. Do not infer corpus size from the number of output chunks: one source
group can be re-chunked into many output directories. The adapter is
[`data_loading/adapters/rna_adapter.py`](../../data_loading/adapters/rna_adapter.py).

## PyG graph contract

`data_loading/cache_to_pyg.py` flattens each molecule into a PyG `Data` object:

| Field | Meaning |
|---|---|
| `z` | model atomic ids, one per cached node |
| `pos` | float32 coordinates, shape `[N, 3]` |
| `edge_index` | directed sparse graph edges, shape `[2, E]` |
| `edge_attr` | distance for each cached edge |
| `block_idx` | residue index per node |
| `entity_idx` | chain/entity index per node |
| `pos_code` | canonical atom position or `Vn` |
| `block_symbols` | residue symbols |
| `dataset_type`, `mol_id` | routing and provenance |

The table above is the legacy edge-stored layout (still written for UniMol and
COCONUT). Protein and RNA chunks are edge-free (`mind_pt_schema`): `edge_index` and
`edge_attr` are not stored and the runtime builds the hybrid graph described in
[hybrid edges](../architecture/HYBRID_EDGES.md). `cache_to_pyg.py` always writes a
`bond_edge_index` field (an empty tensor when the source supplies no bonds); the
protein adapter produces no bonds, so protein structures rely on spatial/hybrid
edges rather than a complete chemical-bond graph.

## Model atomic ids

MIND uses a compact id space of 121 entries (`atom_types: 121`):

| Id | Meaning |
|---:|---|
| `0` | input mask token; never a physical atom label |
| `1`, `6`, `7`, `8`, `9`, `11`, `15`, `16`, `17`, `35`, `53` | atomic numbers of the only 11 recognized elements: H, C, N, O, F, Na, P, S, Cl, Br, I |
| `119` | rare/unknown real element bucket: every other element (B, V, Se, Zn, Mg, Fe, ...) |
| `120` | virtual padding target `V` |

`data_loading/cache_to_pyg.py::atom_to_model_id` checks virtual metadata and
`Vn` position codes before element lookup, so a real vanadium atom is not
confused with adapter-generated virtual padding. Legacy caches with the old
virtual id are normalized by `LazyUniversalDataset`; this repair is made in memory and
production `.pt` files are not rewritten. The single MLM head
predicts these ids; metric callbacks map them to domain-readable labels.

The other ids in `2`-`118` are never produced by the current conversion
(`data_loading/cache_to_pyg.py::ELEMENT_TO_ATOMIC_NUMBER`). Hydrogens are removed by
the protein and RNA adapters (by the `element` field) and by the COCONUT and LBA
adapters; the UniMol adapter has no hydrogen filter. Unrecognized real elements must
remain distinguishable from `V` through id 119 and should not silently become
padding.

## Position codes

### Protein

The canonical backbone tuple is `(N, CA, C, O)`. The sidechain union is held in
`PROTEIN_SIDECHAIN_ATOM_PRIORITY` and includes names such as `CB`, `CG1`,
`OD1`, `NE2`, `NZ`, `SG`, and `SD`. Residue-specific sidechain maps
(`PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE`) list the expected positions of the 20
canonical amino acids; the adapter does not filter with them (they are used by an
import-time consistency assertion, the audit scripts and the tests). The adapter
fills the ten sidechain slots with the real atoms found, sorted by the priority
list, followed by virtual padding `V<slot number>` (see
[protein blocks](#protein-blocks-raw-structures-to-fixed-width-blocks)).

`OXT` is a recognized terminal atom for compatibility but is not a sidechain
slot in newly built 14-atom blocks. Metrics may report it in legacy data, while
protein block MLM can exclude legacy OXT-containing blocks via
`protein_block_mlm_exclude_legacy_oxt`.

### RNA

The twelve backbone positions, the base positions and the legacy aliases are listed in
[RNA blocks](#rna-blocks).

### Position embedding versus chemistry

Raw position codes identify named atoms and can reveal padding through `Vn`.
They are cache/metric metadata, not safe masked-target input features. The
geometry-neutral latent readout instead uses domain-local ordinal slot ids
that do not depend on the hidden atom name, element, or occupancy. `V=120`
is a target class; a `Vn` position code is padding metadata. Neither supplies
physical geometry in the runtime graph.

## Loading and masking

`LazyUniversalDataset` loads chunked `.pt` files on demand. The training path
uses an atom-budget sampler, then PyG collates samples while preserving
`batch`, `block_idx`, `pos_code`, and domain metadata. In the multidomain
configuration, protein block MLM selects complete residue blocks (default
ratio 0.15), saves `original_types`, and writes zero as the masked input id.
The zero is an input mask token, not a physical element.

When geometry-neutral slots are enabled, the model does not feed the raw fixed
graph directly to ESA. Visible V padding and all masked sidechain slots are
removed; masked backbone atoms remain. Real-only spatial neighbors are
completed at runtime, and latent sidechain states are reconstructed after ESA.
See [virtual slots](../architecture/VIRTUAL_SLOTS.md).

## Validation and evaluation contract

For a valid canonical protein sample, validate exact 14-node blocks, unique
backbone positions, finite coordinates, aligned `z`/`pos`/`pos_code` lengths,
chain-local sequence edges, and no cross-graph edges after collation. Tests and
audits should use these production fields rather than synthetic alternatives.
Malformed target-domain data should fail loudly with molecule id and domain
context; non-protein transforms should pass through unchanged.

The fixed representation prevents a direct variable-node-count shortcut, but
it does not guarantee that MLM predictions use residue context. Report
per-slot/per-role metrics and slot-only or owner-shuffle baselines. For
evaluation, separate support and recall for backbone, real sidechain/base,
virtual padding, and rare/hetero atoms. Overall accuracy is dominated by easy
backbone and V targets and is insufficient as the sole quality measure.

Use focused adapter tests before any training command:

```bash
pytest -v tests/test_adapters.py
```

For a training smoke check, use `make train CONFIG=f` and inspect the resolved
RNA path, chunk count, metadata, finite losses, and RNA metrics in the log.

## Related code

- [`data_loading/adapters/protein_adapter.py`](../../data_loading/adapters/protein_adapter.py)
- [`data_loading/adapters/rna_adapter.py`](../../data_loading/adapters/rna_adapter.py)
- [`data_loading/atom_position_vocab.py`](../../data_loading/atom_position_vocab.py)
- [`data_loading/cache_to_pyg.py`](../../data_loading/cache_to_pyg.py)
- [`core/metrics/rna_per_atom_metrics_callback.py`](../../core/metrics/rna_per_atom_metrics_callback.py)

# Protein representation and data flow

## Raw structures to fixed-width blocks

`data_loading/adapters/protein_adapter.py` parses PDB or mmCIF with BioPython.
Hydrogen atoms are dropped. Standard amino-acid residues become
`UniversalBlock`s; chain/entity ids are retained so sequence edges cannot cross
chains. Optional heteroatom inclusion is controlled by the adapter.

With `use_14atom_uniform: true` (the processed protein data contract), each
canonical residue is emitted in this order:

```text
N, CA, C, O, [up to ten canonical sidechain positions]
```

The sidechain positions are sorted by the shared priority list in
`data_loading/atom_position_vocab.py`. Missing positions are filled with
virtual atoms whose `pos_code` is `V1` … `V10`, `is_virtual` is true, and model
id is 120. Their coordinates are CB for residues with CB, or CA for glycine
and fallback cases. `OXT` is excluded from the ten sidechain slots; legacy
terminal records are handled separately by masking policy.

The fixed-width invariant is four backbone plus ten sidechain slots = 14 nodes
per canonical residue. It is an output/cache contract, not a statement that
all 14 nodes are physical atoms during model encoding.

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

Protein caches normally use the hybrid edge builder described in
[hybrid edges](../architecture/HYBRID_EDGES.md). Legacy radius graphs remain
supported. `bond_edge_index` is present when the source molecule supplies
explicit bonds, but protein structure caches generally rely on spatial/hybrid
edges rather than a complete chemical-bond graph.

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

## Validation contract

For a valid canonical protein sample, validate exact 14-node blocks, unique
backbone positions, finite coordinates, aligned `z`/`pos`/`pos_code` lengths,
chain-local sequence edges, and no cross-graph edges after collation. Tests and
audits should use these production fields rather than synthetic alternatives.
Malformed target-domain data should fail loudly with molecule id and domain
context; non-protein transforms should pass through unchanged.

The fixed representation prevents a direct variable-node-count shortcut, but
it does not guarantee that MLM predictions use residue context. Report
per-slot/per-role metrics and slot-only or owner-shuffle baselines.

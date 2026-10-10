# Hybrid and runtime protein edges

## The three-tier protein graph

Protein and RNA chunks store no edges (`mind_pt_schema`); the former offline builder in
`cache_to_pyg.py` was removed in October 2026. Protein Tier 1 is regenerated when a chunk
is loaded and every other edge is built at runtime. The tiers are:

1. Tier 1 connects CA atoms at sequence offsets ±1 through ±`k`, constrained to
   one chain/entity. The default is `sequence_neighbors_k: 3`.
2. Tier 2 is an all-atom radius graph with a 5 Å cutoff and up to
   `max_spatial_neighbors: 48` neighbors per source; Tier-1 duplicates are
   removed.
3. Tier 3 samples CA–CA pairs at least 10 Å apart with inverse-distance-cubed
   weighting. The default is `num_random_edges: 8` per CA source, stored in both
   directions.

The combined graph is one directed `edge_index` with a distance `edge_attr`. It
carries no tier identifier, so tier semantics are inferred from coordinates,
block/entity metadata and configuration. Old edge-stored caches (deleted on
9 October 2026) were built with a radius graph limited by `max_neighbors`.

Virtual nodes are present in old fixed-width cache graphs, so an offline radius
neighbor limit may have been consumed by virtual padding. The geometry-neutral
runtime path therefore filters invalid endpoints and completes real-only local
neighbors before ESA. It should not be described as simply deleting V edges.

## Runtime Tier 3

When `use_dynamic_random_edges: true`,
`OptimizedProteinDynamicRandomEdges` runs on protein samples during loading.
It identifies CA atoms from production `pos_code`, optionally removes existing
long-range CA–CA edges while preserving sequence neighbors, then samples new
CA–CA pairs with the same minimum-distance and inverse-cubic rules. With
`replace_existing_random_edges: true`, only the regenerated Tier 3 set remains
alongside preserved non-Tier-3 edges. `edge_attr` is updated and the result is
coalesced.

The transform is protein-specific and passes non-protein samples through. It
requires protein `pos`, `edge_index`, `pos_code`, `block_idx`, and `entity_idx`
and fails loudly when those production fields are absent. Dynamic sampling is
runtime data augmentation; it does not rewrite `.pt` caches.

## What ESA uses

In the default node-attention path, `M` layers receive the compact physical
edge index and perform graph-local attention. `S` layers perform global
self-attention within each molecule. Both tiers and edge types are represented
to invariant attention primarily by connectivity; the default node path does
not pass a discrete tier label. The encoder’s RBF and invariant summaries are
computed from runtime edges, but the chemical-coordination implementation has
an explicit-bond precedence caveat documented in [the model overview](MODEL_OVERVIEW.md).

This is sparse graph routing, not a claim that every computation is O(NK):
global S attention and padded intermediate tensors can still scale with the
largest graph in a batch. An optional distance-bias path exists but is disabled
by default.

## Configuration reference

```yaml
use_hybrid_edges: true
sequence_neighbors_k: 3
max_spatial_neighbors: 48
num_random_edges: 8
random_edge_min_distance: 10.0
use_dynamic_random_edges: true
replace_existing_random_edges: true
```

The exact effective edge set depends on molecule size, chain/entity boundaries,
cached edges, and runtime geometry-neutral filtering. Report edge counts and
invalid-endpoint counts for any benchmark rather than assuming the nominal
6+48+8 count is reached for every residue.

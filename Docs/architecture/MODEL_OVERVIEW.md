# MIND model overview

MIND is a multi-domain molecular pretraining system. Its current data path can
train one ESA-based encoder on protein, RNA, and UniMol small-molecule graphs in
the same run. A graph contains atom nodes, coordinates, block metadata, and a
directed sparse `edge_index`; PyG batching keeps molecules separate.

This is an implementation description, not a claim of downstream or generative
performance. The production multidomain configuration is
`core/pretraining_config_multidomain.yaml`.

## End-to-end computation

```text
PDB/CIF or prepared molecular cache
  -> UniversalMolecule / UniversalBlock / UniversalAtom
  -> PyG graph cache (.pt)
  -> LazyUniversalDataset and atom-budget sampler
  -> block/atom masking at training time
  -> geometry-neutral runtime graph view (protein/RNA)
  -> molecular encoder
  -> ESA M/S/P backbone
  -> MLM and distance task heads
```

The protein adapter removes hydrogen and standardizes canonical residues to 14
slots. RNA has a separate 23-slot adapter representation. Small molecules keep
their prepared atom sets. See [protein representation](../data/PROTEIN_REPRESENTATION.md)
and [atom vocabulary](../data/ATOM_VOCABULARY.md).

## Encoder and ESA backbone

`core/molecular_encoder.py` combines atomic-number and periodic-period
embeddings with invariant geometric summaries and Gaussian radial-basis (RBF)
features computed from the runtime graph. Coordinates are centered per graph;
the geometry summaries are intended to be translation/rotation-invariant.

The ESA implementation is in `esa/masked_layers.py`. In the default
multidomain config, `layer_types` is `['M', 'S', 'M', 'S', 'M', 'S', 'P']`,
with width 256 and 16 heads per layer:

- `M` is graph-local attention over the supplied edge index.
- `S` is global self-attention within each molecule.
- `P` is PMA pooling to 32 learned seed outputs for graph-level embeddings.

With `xformers_or_torch_attn: flash_varlen`, graph-local M attention uses the
sparse `forward_graph` path and S attention uses variable-length FlashAttention
within molecule boundaries. The model still creates some padded intermediate
representations, so “padding-free” should be read as an attention-path detail,
not a guarantee that every intermediate is sparse.

The default production setting is `use_equivariant_features: false`. An
equivariant `EquivariantSAB` branch exists and requires working NequIP/e3nn
dependencies; enabling it is a separate configuration and validation choice.

## Geometry-neutral fixed slots

Protein/RNA fixed-width caches retain all slots, including virtual padding, but
the enabled runtime view in `core/virtual_slot_graph.py` changes what ESA sees:

1. visible virtual `V` nodes are removed from the encoder node set;
2. all sidechain/base slots in a selected masked block are removed, regardless
   of whether their target is real or `V`;
3. real masked backbone atoms remain as encoder nodes;
4. physical edges are filtered/completed using geometry-valid nodes; and
5. removed side/base targets are reconstructed after ESA as
   `LayerNorm(mean(owner-backbone states) + domain-local slot embedding)`.

Protein slots read from four backbone atoms (`N`, `CA`, `C`, `O`); RNA slots
read from its 12 backbone atoms. These slot readout links are post-ESA routing,
not ESA attention edges. The full-index states are restored only so the task
heads can score the fixed 14/23 target layout. Clean coordinates for removed
real slots are supervision data only, never encoder geometry.

## Active pretraining tasks

The default task list is:

- long-range distance classification on sampled graph-local atom pairs;
- short-range distance classification on selected physical graph edges; and
- one 121-class atomic-number MLM head.

Protein and RNA whole-block masking is controlled independently by
`protein_block_mlm` and `rna_block_mlm`; the default ratio is 0.15. Other domains
use atom masking when the per-sample transform is active. MLM loss is ordinary
cross-entropy separated into a mean real-target term plus
`virtual_atom_loss_weight` times a mean `V`-target term. The default virtual
weight is 0.2. There is no active masked-slot distance objective in the current
runtime contract. Coordinate denoising code exists but is not in the default
task list.

## Boundaries and known caveats

The geometry-neutral path removes the original V-coordinate shortcut, but that
does not prove that the learned representation uses residue or nucleotide
context. Slot-majority and owner-shuffle diagnostics remain necessary. A recent
long run provides bounded evidence (final protein `SCr` about 0.8577, RNA
`BaseR` about 0.9304, and heteroatom recalls about 0.6277/0.9155); these are
run-specific observations, not a proof that the representation is solved.

Two current encoder details deserve explicit caution:

- `_compute_geometric_features` masks padded rows using `x_dense == 0`; masked
  real backbone nodes can therefore have their RBF aggregate removed even when
  they remain geometry-valid. The invariant summary path and RBF path should
  not be assumed identical.
- `_compute_chemical_coordination` prefers `bond_edge_index` when present. In a
  mixed batch, that explicit edge set must be domain-safe; otherwise protein or
  RNA nodes can receive zero/incorrect chemical-coordination values when the
  batch contains molecules with explicit bonds.

The metabolite/COCONUT domain remains disabled in the production comparison
configuration. See [hybrid edges](HYBRID_EDGES.md) for graph construction.

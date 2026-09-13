# Virtual slots and geometry-neutral masking

## Why fixed slots exist

The protein adapter emits four backbone slots plus ten sidechain slots per
canonical residue. RNA uses twelve backbone plus eleven base slots. Fixed width
keeps a masked residue/nucleotide from revealing its identity through the
number of nodes presented to the task head. The raw cache therefore remains
14 or 23 nodes per block even when some slots are padding.

Virtual padding has one target class, `V`, represented by model id 120. It is a
label/padding convention, not a physical atom. In the protein adapter, virtual
positions are placed at `CB` (or `CA` for glycine); this is retained in old
caches but is not trusted as encoder geometry.

## Runtime roles

The geometry-neutral view makes these roles explicit:

| Role | Encoder node? | Geometry input? | MLM target? |
|---|---:|---:|---:|
| visible real atom | yes | yes | no |
| visible virtual padding | no | no | no |
| masked backbone atom | yes | yes, from its real coordinate | yes |
| masked sidechain/base slot | no | no | yes (real or `V`) |

The target decision is made from saved `original_types`; a target being real or
virtual never changes whether its masked side/base slot is removed. This avoids
reintroducing occupancy information through a mask or geometry branch.

## Current forward contract

`core/virtual_slot_graph.py::build_geometry_neutral_graph_view` requires
`z`, `pos`, `batch`, `block_idx`, `edge_index`, `mlm_mask`, and pre-mask
`original_types`. It returns masks and index maps for the full cache and a
compact physical graph for ESA:

```text
geometry-valid nodes
  = visible real atoms + retained masked backbone atoms

ESA input
  = compact geometry-valid nodes and their physical edges

post-ESA slot state
  = LayerNorm(mean(owner backbone hidden states) + slot embedding)
```

The owner readout is differentiable and domain-local: each protein sidechain
slot reads its own residue’s four backbone states; each RNA base slot reads its
own nucleotide’s twelve backbone states. The readout edge list is never passed
to ESA attention, RBF aggregation, equivariant vectors, or distance targets.
Full-index scatter restores 10 or 11 latent states for the existing MLM head.

Coordinates for removed real slots are kept only for optional supervision code;
the current active task set does not add a masked-slot distance objective. `V`
slots never create physical distance targets.

## Graph and centering invariants

The runtime view filters cached edges to valid endpoints and completes local
radius neighbors over geometry-valid coordinates. Protein CA sequence and
long-range structural edges may be retained under the configured hybrid-edge
rules; RNA local neighbors are similarly completed. The important invariant is
that a physical or attention edge cannot end at a removed V or side/base slot.
Each graph is centered using geometry-valid atoms only, so changing an inactive
placeholder cannot move the real structure.

These invariants are tested in the focused geometry-neutral test modules. They
establish data routing and leakage resistance; they do not establish that the
MLM has learned chemically informative residue context.

## What is not enabled

`masked_slot_template` is a deprecated synthetic-coordinate experiment and is
off in the production multidomain config. Do not describe it as part of the
current representation. The current system also has no occupancy head, residue
classifier, or BoltzGen-style coordinate decoder.

Equivariant features are optional and disabled by default. If enabled, the
equivariant branch must receive only geometry-valid edges and vectors; the
geometry-neutral masks do not turn the default invariant model into an
equivariant one.

## Validation guidance

The minimum useful checks are:

- placeholder perturbation leaves encoder logits unchanged;
- swapping real versus `V` targets leaves the encoder input unchanged;
- physical edges have no invalid endpoint;
- owner removal and owner shuffling are explicit counterfactuals; and
- slot-majority, per-role, and exact-block metrics accompany overall accuracy.

The focused tests prove the first three mechanical properties. Learning
acceptance still requires multiple seeds in which correct-owner performance
beats slot-only and shuffled-owner baselines.

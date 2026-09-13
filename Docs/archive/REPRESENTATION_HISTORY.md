# Representation history

The original design documents are preserved in
`Docs/archive/pre_reorganization_20260912.tar.gz`. This page keeps the useful
decision trail without treating rejected experiments as current behavior.

## Fixed-slot design

The protein adapter adopted 14 slots per residue (4 backbone + 10 sidechain)
and one generic virtual class `V`. Missing sidechain slots were placed at CB,
or CA for glycine. The intent was to prevent residue identity from being
revealed by variable atom count. The unintended consequence was that a real CB
could share a coordinate with many V fillers, making the masked element target
ambiguous. This is a representation trade-off, not evidence that the model is
generally unable to encode proteins.

The RNA adapter follows a separate 23-slot convention (12 backbone + 11 base
slots). A temporary C1′-stacking experiment was not retained in the production
representation.

## Experiments and decisions

### CB-offset experiment

A recorded small protein A/B experiment moved virtual atoms into a 2 Å sphere
around CB. The archived report recorded worse loss and broad backbone/sidechain
regressions (for example, CG about 0.99 to 0.63 and N about 0.63 to 0.00),
while CB remained poor. The likely mechanism was synthetic nodes entering the
spatial graph and consuming neighbor capacity. These values are historical
records, not a current benchmark.

### RNA C1′ stacking experiment

The archived A/B record reported lower RNA overall accuracy (0.7252 to 0.5569),
C1′ recall (0.9414 to 0.8043), backbone recall (0.8665 to 0.7106), real-base
accuracy (0.5459 to 0.4613), and nucleotide exact accuracy (0.3895 to 0.1802)
when virtual nodes were stacked at C1′. It also recorded 868,574 zero-distance
cache edges. The result rejected anchor stacking as a safe generic RNA fix.

### Synthetic template-v1

The project also tried type-independent synthetic masked-slot coordinates.
`masked_slot_template.enabled` is false in the current configuration. A
synthetic coordinate template is not part of the present encoder contract and
must not be described as an active solution.

### Geometry-neutral runtime view

The accepted mechanical direction is to preserve the fixed 14/23 target layout
while removing visible V and masked side/base slots from the ESA input graph.
Masked slots are reconstructed after ESA from their own owner-backbone mean plus
a slot embedding, followed by LayerNorm. Runtime edge completion and physical
centering use geometry-valid nodes only. This removes the direct V-coordinate
and masked-slot geometry channel, but it does not prove that training uses
residue/nucleotide context.

Focused tests recorded in the former decision document covered compaction,
real-only edge restoration, owner routing, and placeholder invariance. Those are
mechanical checks. Learning acceptance still requires owner-shuffle and
slot-only baselines across multiple seeds.

## Current evidence and remaining caveats

A later long run recorded final protein real-sidechain recall (`SCr`) around
0.8577, RNA real-base accuracy (`BaseR`) around 0.9304, and heteroatom recalls
around 0.6277 and 0.9155. These are bounded observations from one run, not a
claim that the representation problem is solved; CB alone is especially weak
evidence because carbon/slot priors can be easy shortcuts.

Current code still has two geometry-feature caveats: padded-row masking based on
`x_dense == 0` can suppress RBF features for masked real backbone nodes, and
explicit `bond_edge_index` precedence in chemical coordination must be domain
safe in mixed batches. These are documented in [the model overview](../architecture/MODEL_OVERVIEW.md)
and remain open validation points.

## Scope of rejected proposals

Literal BoltzGen marker encoding would require a coordinate diffusion objective,
equivariant coordinate output, dynamic/noisy graph handling, and a decoder. It
is not an adapter-only patch and has no supplied RNA marker scheme. Learned
per-residue pseudo-atoms, occupancy heads, and masked-slot distance objectives
are likewise not current MIND behavior.

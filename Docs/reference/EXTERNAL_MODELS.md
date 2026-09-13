# External model references and comparison boundaries

This page records why external projects appear in MIND documentation. It keeps
claims verifiable: statements about the local implementation are checked
against this repository; statements about external papers or vendor code are
only summaries of the referenced artifact and are not MIND measurements.

## GET

The vendored `GET/` tree identifies itself as *Generalist Equivariant Transformer
Towards 3D Molecular Interaction Learning* and provides a `get_clean` API with
atom/block ids, coordinates, and edges. Its README is the source for the local
vendor description. MIND imports its own ESA implementation from `esa/`; it is
not accurate to call MIND’s current pretraining model a drop-in GET model.

## ESA graph attention

The local code in `esa/masked_layers.py` exposes MAB/SAB/PMA-style components
and configurable `layer_types`. In the default MIND pretraining config, M/S/P
semantics are implemented as graph-local M attention, molecule-global S
attention, and PMA pooling. This is a code-level comparison only. The archived
`papers/ESA_COMPARISON.md` attributed broader architectural and performance
claims to an external paper without pinning each claim to a verified source;
those claims are intentionally not repeated here.

## Salad

MIND’s protein cache builder uses a Salad-inspired naming and three-tier idea:
CA sequence neighbors, local spatial neighbors, and CA long-range random edges.
The exact MIND implementation is in
[`HYBRID_EDGES.md`](../architecture/HYBRID_EDGES.md), including its different
defaults and runtime geometry-neutral rebuild. Similarity of terminology does
not establish equivalent training objectives, generative behavior, or
performance. The external Salad paper should be consulted directly for those
claims.

## BoltzGen

The archived BoltzGen note describes a protein design representation in which
fixed atom14 coordinates and marker placement support coordinate generation and
decoding. That is a different task from MIND’s masked atomic-number prediction.
MIND does not currently implement a BoltzGen coordinate diffusion process,
marker-count decoder, or RNA analogue. See [representation history](../archive/REPRESENTATION_HISTORY.md)
for the local decision and its limitations.

## Other literature named in archived material

The former overview/report also mentioned Uni-Mol, RoseTTAFold All-Atom,
ATOMICA, EPT, ChemBERTa, SaProt, SELFormer, and RNA3DB. Those references are
contextual pointers, not validated baselines in this repository. No numerical
comparison, citation-derived capability, or “state of the art” conclusion
should be inferred unless a reproducible experiment and a primary source are
added.

## Safe comparison language

Use “inspired by,” “shares a component with,” or “the local code implements”
when describing these relationships. Reserve “matches,” “improves,” or
“outperforms” for an explicitly reproduced, controlled benchmark with its
dataset, configuration, and metric definition recorded.

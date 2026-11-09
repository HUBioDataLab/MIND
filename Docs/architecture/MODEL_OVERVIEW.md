# MIND model overview

MIND is a multi-domain molecular pretraining system. Its current data path can
train one ESA-based encoder on protein, RNA, and UniMol small-molecule graphs in
the same run. A graph contains atom nodes, coordinates, block metadata, and a
directed sparse `edge_index`; PyG batching keeps molecules separate.

This is an implementation description, not a claim of downstream or generative
performance. The `core/` YAML files and what each one selects are listed in the
[training guide](../training/TRAINING_GUIDE.md#configuration-files); the Makefile
shortcut `CONFIG=m` is `core/pretraining_config_multidomain_1m_coconut.yaml`, the
four-domain configuration; the old three-domain 40k baseline is
`core/pretraining_config_multidomain.yaml` (`CONFIG=pretraining_config_multidomain`).

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
their prepared atom sets. The slot contract, the atom-id vocabulary and the RNA
blocks are defined in [data representation](../data/DATA_REPRESENTATION.md).

The entry point is [`core/train_pretrain.py`](../../core/train_pretrain.py),
which builds a `LazyUniversalDataset`, loaders, the ESA model, callbacks, and a
PyTorch Lightning trainer. The shared encoder is implemented in
[`core/molecular_encoder.py`](../../core/molecular_encoder.py) and the model
and task dispatcher in [`core/pretraining_model.py`](../../core/pretraining_model.py).

## Encoder and ESA backbone

`core/molecular_encoder.py` combines atomic-number and period embeddings (the
period is known for H, C, N, O, F, P, S and Cl; every other id gets period 0)
with invariant geometric summaries and Gaussian radial-basis (RBF)
features computed from the runtime graph. Coordinates are centered per graph;
the geometry summaries are intended to be translation/rotation-invariant.

The ESA implementation is in `esa/masked_layers.py`. In the default
multidomain config, `layer_types` is `['M', 'S', 'M', 'S', 'M', 'S', 'P']`,
with width 256 and 16 heads per layer (6,271,600 parameters for the multidomain
configs):

- `M` is graph-local attention over the supplied edge index.
- `S` is global self-attention within each molecule.
- `P` is PMA pooling to 32 learned seed outputs for graph-level embeddings.

With the `flash_varlen` backend (the only one, and the code default of the
`xformers_or_torch_attn` field, so the YAMLs do not set it), graph-local M attention uses the
sparse `forward_graph` path and S attention uses variable-length FlashAttention
within molecule boundaries. `flash_varlen` is the only supported attention
backend; the torch and xformers backends were removed. With
`esa_packed_rows: false` (the dataclass default, and the setting of the
three-domain and `fast` configs) the model still creates padded
`[B, N, D]` intermediate representations, so “padding-free” should be read as an
attention-path detail. With `esa_packed_rows: true` (the COCONUT-enabled four-domain configs and the four `*_only` configs)
ESA runs on packed `[T, D]` rows and pads once, before PMA; this mode is not
supported together with equivariant features or the distance bias.

The default production setting is `use_equivariant_features: false`. An
equivariant `EquivariantSAB` branch exists and requires working NequIP/e3nn
dependencies; enabling it is a separate configuration and validation choice.

## Geometry-neutral fixed slots

Protein/RNA fixed-width caches retain all slots, including virtual padding, but
the enabled runtime view in
[`core/virtual_slot_graph.py`](../../core/virtual_slot_graph.py) changes what ESA
sees: visible virtual `V` nodes and all sidechain/base slots of a masked block are
removed from the encoder graph, masked backbone atoms stay, and the removed targets
are read after ESA from their owner backbone states. The roles, the readout formula
and the invariants are described once, in
[virtual slots](VIRTUAL_SLOTS.md).

## Active pretraining tasks

The default task list contains exactly these configured tasks, each with weight
`1.0`:

- long-range distance classification on sampled graph-local atom pairs;
- short-range distance classification on selected physical graph edges; and
- one 121-class atomic-number MLM head.

```yaml
pretraining_tasks: [long_range_distance, short_range_distance, mlm]
task_weights:
  long_range_distance: 1.0
  short_range_distance: 1.0
  mlm: 1.0
```

Protein and RNA whole-block masking is controlled independently by
`protein_block_mlm` and `rna_block_mlm` (off in the dataclass, on in the multidomain
configs); the ratio is 0.15. Other domains
use atom masking when the per-sample transform is active. MLM loss is ordinary
cross-entropy separated into a mean real-target term plus
`virtual_atom_loss_weight` times a mean `V`-target term. The multidomain
configs set this weight to 0.2 (the dataclass default is 1.0); this is a weighting
inside MLM, not an additional element-balanced objective. There is no active
masked-real-slot (masked-slot) distance objective and no separate element-balanced
loss in the current runtime contract. Coordinate denoising code exists but is not in
the default task list.

## Boundaries and known caveats

The geometry-neutral path removes the original V-coordinate shortcut, but that
does not prove that the learned representation uses residue or nucleotide
context. Slot-majority and owner-shuffle diagnostics remain necessary. Run
results are kept locally.

One encoder detail deserves explicit caution: `_compute_geometric_features`
masks padded rows using `x_dense == 0`; masked real backbone nodes can therefore
have their RBF aggregate removed even when they remain geometry-valid. The
invariant summary path and RBF path should not be assumed identical.

`_compute_chemical_coordination` chooses its source per graph: graphs that carry
explicit bonds (`bond_edge_index`, flagged by `has_explicit_bonds`) count those
bonds, and only graphs without an explicit source use the distance proxy
(edges of 0.5-1.85 Å). Protein and RNA graphs therefore keep the distance proxy
in mixed batches even when other graphs carry explicit bonds (repaired on
10 October; CPU contract tests pass and CUDA COCONUT runs have completed, but the
matched reference-versus-repaired comparison has no log on this machine, so it was not
run or could not be verified).

The metabolite/COCONUT domain is disabled in the three-domain comparison
configuration and enabled in the three four-domain configurations and in `coconut_only`.
See [hybrid edges](HYBRID_EDGES.md) for graph construction.

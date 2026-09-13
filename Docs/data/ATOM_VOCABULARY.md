# Atom and position vocabulary

## Model atomic ids

MIND uses a compact id space of 121 entries (`atom_types: 121`):

| Id | Meaning |
|---:|---|
| `0` | input mask token; never a physical atom label |
| `1`–`118` | standard periodic-table atomic numbers when recognized |
| `119` | rare/unknown real element bucket |
| `120` | virtual padding target `V` |

`data_loading/cache_to_pyg.py::atom_to_model_id` checks virtual metadata and
`Vn` position codes before element lookup, so a real vanadium atom is not
confused with adapter-generated virtual padding. Legacy caches with the old
virtual id are normalized by `LazyUniversalDataset`. The single MLM head
predicts these ids; metric callbacks map them to domain-readable labels.

Hydrogens are normally removed from PDB protein structures. RNA and small
molecules may have their own source filtering. Unrecognized real elements must
remain distinguishable from `V` through id 119 and should not silently become
padding.

## Protein position codes

The canonical backbone tuple is `(N, CA, C, O)`. The sidechain union is held in
`PROTEIN_SIDECHAIN_ATOM_PRIORITY` and includes names such as `CB`, `CG1`,
`OD1`, `NE2`, `NZ`, `SG`, and `SD`. Residue-specific sidechain maps define
which of those positions are real for each of the 20 canonical amino acids.
Missing positions become `V1` … `V10` in their fixed ordinal slots.

`OXT` is a recognized terminal atom for compatibility but is not a sidechain
slot in newly built 14-atom blocks. Metrics may report it in legacy data, while
protein block MLM can exclude legacy OXT-containing blocks via
`protein_block_mlm_exclude_legacy_oxt`.

## RNA position codes

RNA uses twelve backbone positions:

```text
P, OP1, OP2, O5', C5', C4', O4', C3', O3', C2', O2', C1'
```

Each nucleotide has up to eleven base positions; the shared vocabulary also
recognizes supported modified-base labels. RNA fixed blocks therefore have
12 + 11 = 23 slots when standardized. Legacy aliases such as `O1P` and
`C1*` are normalized on the RNA path only. Optional phosphate `OP3` is
terminal metadata, not a fixed base slot.

## Position embedding versus chemistry

Raw position codes identify named atoms and can reveal padding through `Vn`.
They are cache/metric metadata, not safe masked-target input features. The
geometry-neutral latent readout instead uses domain-local ordinal slot ids
that do not depend on the hidden atom name, element, or occupancy. `V=120`
is a target class; a `Vn` position code is padding metadata. Neither supplies
physical geometry in the runtime graph.

For evaluation, separate support and recall for backbone, real sidechain/base,
virtual padding, and rare/hetero atoms. Overall accuracy is dominated by easy
backbone and V targets and is insufficient as the sole quality measure.

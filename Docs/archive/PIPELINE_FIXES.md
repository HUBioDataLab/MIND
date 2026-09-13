# Pipeline fixes and remaining risks

This is a historical record of pipeline investigations. “Fixed” means the
current code has an implementation or focused test; it does not mean every old
cache, configuration, or external dataset has been regenerated. Claims based
only on old logs are labelled historical.

## Current contracts

| Area | Current interpretation |
|---|---|
| Metadata | Required by the improved atom-aware sampler; missing/inconsistent metadata fails loudly. |
| Graph size | `max_graph_atoms` is a pre-split hard guard; `max_atoms_per_batch` is only a packing target. |
| Epochs | Approved multidomain config requests full-pass, natural-domain sampling with zero quota cycling; each run must verify consumed counts against its epoch plan. |
| Virtual slots | Geometry-neutral encoder view is active in approved configs; synthetic masked-slot templates are off. |
| Losses | Long-range distance, short-range distance, and MLM have weight 1.0; MLM virtual targets have weight 0.2. |
| Metabolite | Disabled in the approved comparison until mixed `bond_edge_index` collation is explicitly validated. |

## Investigated issues

### Cross-chain protein sequence edges

The historical audit found that assigning every protein residue the same
`entity_idx` could create sequence edges across chain boundaries. The current
hybrid-edge path checks graph/entity membership, but old processed chunks retain
the semantics with which they were built. Treat this as a reprocessing and
audit requirement, not as proof that every existing cache is corrected. The
relevant code is [`data_loading/adapters/protein_adapter.py`](../../data_loading/adapters/protein_adapter.py)
and [`data_loading/cache_to_pyg.py`](../../data_loading/cache_to_pyg.py).

### Distance bins and validation masks

The coarse five-bin distance experiment is superseded by configured nonuniform
long-range and short-range boundaries. Validation/test distance masks are
deterministic in the current loss dispatcher, while training short-range masks
remain stochastic. Verify the resolved YAML when comparing historical logs.

### Sampler length and oversized graphs

Older experiments used a fixed atom-per-sample estimate in `__len__`, allowed
short-domain cycling, and silently emitted an oversized one-graph batch. The
current sampler materializes one deterministic plan per epoch and the approved
full config requests single-pass natural sampling. The hard graph guard now
rejects oversized metadata before splitting. The quota/cycling implementation
remains for regression coverage, so old quota logs must not be described as
current policy.

The September 2026 100-epoch run exposed an operational gap: epoch 99 created
19,122 planned batches with all selected samples unique, but Lightning consumed
19,081 batches. Its cached training-batch limit therefore truncated the plan;
sampler construction alone cannot certify a full pass. Treat this as an active
validation risk until a run verifies plan length against tracked totals for
every epoch.

### Runtime geometry and virtual slots

The historical synthetic-template path moved masked slots to canonical local
coordinates and rebuilt edges. Controlled experiments exposed shortcut and
performance risks, so it is disabled in the current configs. The active
geometry-neutral view removes visible virtual and masked sidechain/base nodes
from the encoder graph and supplies latent slot readouts from physical owner
context. It also suppresses invalid geometry from distance objectives. See
[`core/virtual_slot_graph.py`](../../core/virtual_slot_graph.py) and
[`tests/test_geometry_neutral_virtual_slots.py`](../../tests/test_geometry_neutral_virtual_slots.py).

### Chemical bonds and masking

Some molecule adapters provide explicit `bond_edge_index`; other domains may
carry an empty tensor. Current mixed-batch masking selects explicit chemical
bonds when they are present and does not silently invent a protein/RNA bond
graph from unrelated edges. The behavior is covered by
[`data_loading/masking_utils.py`](../../data_loading/masking_utils.py) and
[`tests/test_chemical_coordination.py`](../../tests/test_chemical_coordination.py).
Keep any remaining fallback or RBF edge-suppression behavior under focused
tests; do not infer that all historical bond issues are closed.

### Performance telemetry

Old logs contained hard-coded cache-hit and throughput messages and compared
dynamic batches using it/s alone. Current callback work records measured graph,
atom, edge, cache, wait, and step statistics. Compare atoms/s and quantiles, and
retain the exact config/data selection alongside a run log.

## Verification commands

```bash
make test-unit
pytest -v tests/test_adapters.py
pytest -v tests/test_dynamic_sampler_epoch_plan.py
pytest -v tests/test_geometry_neutral_virtual_slots.py
```

Training validation remains user-only and must go through `make train`; these
commands do not start a workload.

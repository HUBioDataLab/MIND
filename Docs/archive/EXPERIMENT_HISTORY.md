# Experiment history

This archive preserves decisions and results from superseded experiments. It is
not a current configuration guide. Current run status belongs in the
[`Docs/training/STATUS_AND_NEXT_STEPS_20260912.md`](../training/STATUS_AND_NEXT_STEPS_20260912.md)
report.

## Protein-specific loss experiments

The residue-reconstruction series tested amino-acid prediction, atom-element
prediction, coordinate noise, and BERT-style edge masking. Full edge isolation
plateaued near the observed residue-loss baseline; retaining unmasked-to-masked
context improved the protein-specific loss but could harm non-protein geometry
when centering was applied globally. A two-stage element/coordinate experiment
reported the best historical protein-specific loss while keeping COCONUT metrics
healthy.

All residue-loss heads and experiments were removed. The current objective is
the shared three-task set: long-range distance, short-range distance, and MLM.
Do not use the historical residue-loss values as a current model metric. The
source investigation is retained in the archived pre-reorganization material;
the current task implementation is
[`core/pretraining_tasks.py`](../../core/pretraining_tasks.py).

## Virtual-slot investigation

The earlier plan explored fixed protein 14-slot and RNA 23-slot representations,
local-frame templates, leakage probes, runtime edge rebuilding, and controlled
protein/RNA A/B runs. The synthetic `masked_slot_template` path was eventually
superseded: coordinates were not a reliable slot identity for a masked real/V
target, and runtime rebuilding added measurable complexity. Its flag remains
off in the approved configs.

The current design is the geometry-neutral latent view. It keeps fixed-width
targets but removes visible virtual padding and masked sidechain/base targets
from encoder attention. The target slots are read from encoded physical
backbone context. Local slot identity is supplied by domain-local slot features,
not by target `z`, virtual labels, or synthetic target coordinates. Relevant
code and tests:

- [`core/virtual_slot_graph.py`](../../core/virtual_slot_graph.py)
- [`core/pretraining_model.py`](../../core/pretraining_model.py)
- [`tests/test_geometry_neutral_virtual_slots.py`](../../tests/test_geometry_neutral_virtual_slots.py)
- [`tests/test_virtual_atom_stage4.py`](../../tests/test_virtual_atom_stage4.py)

The historical RNA C1'-stacking handoff found that old RNA caches could contain
no real virtual nodes and that metric vocabulary could confuse modified real
atoms with V. The current June-equivalent dataset and RNA vocabulary/loader
contracts supersede those old cache assumptions. Exact old molecule overlap is
not reconstructed from filename order.

## Multidomain sampler history

An earlier bounded-epoch experiment introduced atom-deficit quotas and cycling
to prevent short domains disappearing near an epoch tail. It was tested, but the
resulting epoch saw only a fraction of selected data and repeated samples. That
policy is superseded. The approved full run disables atom quotas and cycles,
consumes every selected graph once, and reports the natural domain atom mix.

The corresponding audit also identified these historical failure modes:

- graph-count proportions are not atom proportions;
- a batch budget does not cap an individual graph;
- a fixed-500-atom `__len__` estimate can disagree with yielded batches;
- output chunk count does not measure corpus size;
- raw it/s is not a safe comparison for dynamic batches;
- validation cadence can be distorted when a sampler reports the wrong length.

These are interpretation rules for old logs. Current behavior should be checked
against [`data_loading/improved_dynamic_sampler.py`](../../data_loading/improved_dynamic_sampler.py),
[`core/train_pretrain.py`](../../core/train_pretrain.py), and the resolved YAML.

## Historical sources and provenance

The pre-reorganization source files are preserved in
[`Docs/archive/pre_reorganization_20260912.tar.gz`](pre_reorganization_20260912.tar.gz),
including the former Turkish/English virtual-atom plan and RNA fast-experiment
handoff. They are retained for provenance, not as active instructions. The
consolidated current documents are:

- [`Docs/data/DATA_PIPELINE.md`](../data/DATA_PIPELINE.md)
- [`Docs/data/RNA_DATASETS.md`](../data/RNA_DATASETS.md)
- [`Docs/training/TRAINING_GUIDE.md`](../training/TRAINING_GUIDE.md)
- [`Docs/training/VALIDATION.md`](../training/VALIDATION.md)

Historical claims from external reports, remembered baselines, or unavailable
old caches should be qualified unless reproduced from current code and data.

## Safe interpretation of experiment results

When comparing experiments, hold fixed the seed, selected molecule IDs, chunk
paths, sampler policy, graph limits, task weights, validation set, and GPU. Keep
real-only protein/RNA metrics separate from V-inclusive aggregates, and report
support denominators. A falling total loss alone does not prove that virtual
slot confusion or chemistry has been solved.

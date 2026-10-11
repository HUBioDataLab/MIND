# MIND documentation

Current-code documentation and operational handoff, evidence reviewed 12 September
2026 and documentation finalized 13 September 2026.
Read current documents first; archived experiment proposals are not active plans.

## Start here

- [Final 100-epoch results, open issues, and next development steps](training/STATUS_AND_NEXT_STEPS_20260912.md)
- [Full-data scaling plan](training/FULL_DATA_SCALING_PLAN_20260923.md)
- [Encoder feature defects: scope, consequences and repair plan](training/ENCODER_FEATURE_DEFECTS_20261009.md)
- [Model overview](architecture/MODEL_OVERVIEW.md)

## Structure

```text
Docs/
├── architecture/   Model, virtual slots, hybrid edges
├── data/           Representation, vocabulary, loading, RNA selection
├── training/       Commands, validation, dated results and CPU evidence
├── development/    Shared coding-agent setup and workflow
├── reference/      External-model comparison boundaries
└── archive/        Superseded experiments and preserved original documents
```

| Topic | Documents |
|---|---|
| Architecture | [Overview](architecture/MODEL_OVERVIEW.md), [virtual slots](architecture/VIRTUAL_SLOTS.md), [hybrid edges](architecture/HYBRID_EDGES.md) |
| Data | [Pipeline](data/DATA_PIPELINE.md), [protein representation](data/PROTEIN_REPRESENTATION.md), [atom vocabulary](data/ATOM_VOCABULARY.md), [RNA datasets](data/RNA_DATASETS.md) |
| Training | [Guide](training/TRAINING_GUIDE.md), [validation](training/VALIDATION.md), [current results/handoff](training/STATUS_AND_NEXT_STEPS_20260912.md), [bounded encoder reproduction](training/reproduce_encoder_findings.py), [encoder defect report](training/ENCODER_FEATURE_DEFECTS_20261009.md) |
| Development | [Agent workflow](development/AGENT_WORKFLOW.md), [validate-feature skill](../.agents/skills/validate-feature/SKILL.md), [strict-contract skill](../.agents/skills/strict-contracts-and-validation/SKILL.md) |
| References | [External-model references](reference/EXTERNAL_MODELS.md) |
| History | [Representation decisions](archive/REPRESENTATION_HISTORY.md), [pipeline fixes/risks](archive/PIPELINE_FIXES.md), [experiments](archive/EXPERIMENT_HISTORY.md) |

Root `README.md` and `CODEX.md` point here or
to the relevant canonical document. `AGENTS.md` contains shared rules and
`CLAUDE.md` imports them. All MIND skills live in `.agents/skills/`; Claude's
`.claude/skills/` contains relative symlinks to those same folders. There are no
separate Codex/OpenCode copies. See the agent workflow for discovery and checks.

## Preservation and review coverage

[The original-document archive](archive/pre_reorganization_20260912.tar.gz)
preserves the former Docs tree, root migration/planning notes, agent validation
notes, repository instructions/skill, the ESA comparison, and the other-group
report as they stood before reorganization. Restore to a **separate directory**
when consulting history; do not overwrite current docs with archived proposals.

All original first-party project documentation was read and assessed against
current code or explicitly retained as historical evidence. Vendor
`GET/README.md` and `template/dream_mar2026/report.txt` were read as external
references; vendor/license/source artifacts are left in place and are not
rewritten to imply they describe current MIND. PDFs/source snapshots are
reference artifacts, not a newly validated external benchmark.

Main replacements:

| Former document(s) | Current location |
|---|---|
| `MIND_OVERVIEW`, `features/HYBRID_EDGE_CONSTRUCTION` | `architecture/` |
| `features/VIRTUAL_ATOMS_IN_GRAPH_TRANSFORMER`, `CB_DUSUKLUGU_ACIKLAMA` | `architecture/VIRTUAL_SLOTS` and `archive/REPRESENTATION_HISTORY` |
| `features/BOLTZGEN_GEOMETRIC_RESIDUE_ENCODING`, `papers/ESA_COMPARISON` | `reference/EXTERNAL_MODELS` and archive |
| `proteins/ATOM_TYPES`, `PROTEIN_REPRESENTATION_PIPELINE`, `DYNAMIC_RANDOM_EDGES` | `data/` and `architecture/HYBRID_EDGES` |
| `DATA_LOADING_DOCUMENTATION`, `DATA_LOCATIONS` | `data/DATA_PIPELINE`, `data/RNA_DATASETS` |
| `FAST_TRAINING_GUIDE`, `TRAINING_DOCUMENTATION` | `training/TRAINING_GUIDE`, `training/VALIDATION` |
| `PIPELINE_BUGS`, `PROTEIN_LOSS_EXPERIMENTS`, `features/EPOCH_BALANCED_MULTIMODAL_BATCHING` | `archive/PIPELINE_FIXES`, `archive/EXPERIMENT_HISTORY` |
| `features/MULTIDOMAIN_VIRTUAL_V1_RUN_AUDIT`, root RNA/V handoffs | `training/STATUS_AND_NEXT_STEPS_20260912`, data/archive docs |

Project docs, original-document archive, root entrypoints and shared MIND skills
are **versioned in Git**. A clone includes them. Logs, checkpoints, external data,
private tool state and reference trees still have to be copied separately.

# MIND documentation

Feature and architecture documentation for the current code. Run records, migration
notes, task lists and dated evidence are kept outside Git.

## Start here

- [Model overview](architecture/MODEL_OVERVIEW.md)
- [Data pipeline](data/DATA_PIPELINE.md)
- [Data representation](data/DATA_REPRESENTATION.md)
- [Training guide](training/TRAINING_GUIDE.md)
- [Downstream evaluation](../downstream/README.md)

## Structure

```text
Docs/
├── architecture/   Model, virtual slots, hybrid edges
├── data/           Data pipeline, fixed-slot representation, atom vocabulary
├── training/       Training commands and the bounded encoder reproduction script
└── reference/      External-model comparison boundaries
```

| Topic | Documents |
|---|---|
| Architecture | [Overview](architecture/MODEL_OVERVIEW.md), [virtual slots](architecture/VIRTUAL_SLOTS.md), [hybrid edges](architecture/HYBRID_EDGES.md) |
| Data | [Pipeline](data/DATA_PIPELINE.md), [data representation](data/DATA_REPRESENTATION.md) (protein and RNA slots, atom ids, position codes) |
| Training | [Guide](training/TRAINING_GUIDE.md), [bounded encoder reproduction](training/reproduce_encoder_findings.py) |
| Downstream | [Downstream evaluation](../downstream/README.md) (tasks, protocols, results ledger) |
| References | [External-model references](reference/EXTERNAL_MODELS.md) |

Logs, checkpoints, external data and local notes are not in Git.

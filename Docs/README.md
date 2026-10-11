# MIND documentation

Feature and architecture documentation for the current code. Run records, migration
notes, task lists and dated evidence are kept outside Git.

## Start here

- [Model overview](architecture/MODEL_OVERVIEW.md)
- [Data pipeline](data/DATA_PIPELINE.md)
- [Training guide](training/TRAINING_GUIDE.md)

## Structure

```text
Docs/
├── architecture/   Model, virtual slots, hybrid edges
├── data/           Representation, vocabulary, loading, RNA selection
├── training/       Training commands and the bounded encoder reproduction script
├── development/    Shared coding-agent setup and workflow
└── reference/      External-model comparison boundaries
```

| Topic | Documents |
|---|---|
| Architecture | [Overview](architecture/MODEL_OVERVIEW.md), [virtual slots](architecture/VIRTUAL_SLOTS.md), [hybrid edges](architecture/HYBRID_EDGES.md) |
| Data | [Pipeline](data/DATA_PIPELINE.md), [protein representation](data/PROTEIN_REPRESENTATION.md), [atom vocabulary](data/ATOM_VOCABULARY.md), [RNA datasets](data/RNA_DATASETS.md) |
| Training | [Guide](training/TRAINING_GUIDE.md), [bounded encoder reproduction](training/reproduce_encoder_findings.py) |
| Development | [Agent workflow](development/AGENT_WORKFLOW.md), [validate-feature skill](../.agents/skills/validate-feature/SKILL.md), [strict-contract skill](../.agents/skills/strict-contracts-and-validation/SKILL.md) |
| References | [External-model references](reference/EXTERNAL_MODELS.md) |

Root `README.md` and `CODEX.md` point here or to the relevant canonical document.
`AGENTS.md` contains shared rules and `CLAUDE.md` imports them. All MIND skills live in
`.agents/skills/`; Claude's `.claude/skills/` contains relative symlinks to those same
folders. There are no separate Codex/OpenCode copies. Logs, checkpoints, external data
and private tool state are not in Git.

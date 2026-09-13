# MIND

Molecular INtelligence through Domains: a shared ESA graph transformer pretrained
on protein, RNA, and small-molecule graphs with atom masking and distance tasks.
The current protein/RNA runtime uses physical atoms for encoding and latent
backbone readout for masked fixed-width slots. It is not a coordinate generator.

Start with the [documentation index](Docs/README.md).

- [Architecture](Docs/architecture/MODEL_OVERVIEW.md)
- [Data pipeline](Docs/data/DATA_PIPELINE.md)
- [Training and validation](Docs/training/TRAINING_GUIDE.md)
- [Latest training results and development handoff](Docs/training/STATUS_AND_NEXT_STEPS_20260912.md)
- [Anzu → Atum migration](Docs/operations/MIGRATION.md)

## Commands

Use the working `mind_esa3d` environment. For migration, export the actual
source environment; `GET/env.yml` is an older vendor environment, not the
current MIND lockfile.

```bash
make help
make test-unit
make train CONFIG=f LOG=fast_check
make train CONFIG=m LOG=multidomain_run
```

Training is user-started. See [AGENTS.md](AGENTS.md) for execution authority and
[validation](Docs/training/VALIDATION.md) for interpretation and verification.
Project docs, shared rules and MIND skills arrive through Git. Data, logs,
outputs and private user-tool state require the full migration procedure.

## Current status

The September 1–12, 2026 run completed 100 epochs. Final validation real-sidechain
accuracy is 85.77% for protein and real-base accuracy is 93.04% for RNA. These
are not held-out test or downstream results. Rare protein chemistry, late-slot
V competition, two encoder-feature defects, and an epoch-consumption issue
remain open; the linked handoff provides evidence and ordered next steps.

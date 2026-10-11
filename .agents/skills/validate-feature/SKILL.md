---
name: validate-feature
description: Validate MIND training, data, masking, attention, metrics, and configuration changes using production-contract tests, user-started Make smoke runs, and matched multidomain comparisons.
---

# Validate a MIND feature across coding agents

## Read the current repository contract

Locate the MIND repository and read its root `AGENTS.md`. Read
`Docs/training/TRAINING_GUIDE.md` and any local validation or status notes kept outside Git.
Use repository paths rather than hard-coding the source server's home directory.
Read `.agents/skills/strict-contracts-and-validation/SKILL.md` for applicable
data/transform changes. Report missing documents rather than inventing baselines.

## Establish a narrow comparison

Record changed files, intended behavior, exact config, dataset selection, seed,
masking policy, graph limits, tasks and baseline run/checkpoint. Preserve unrelated
work. Separate code evidence, unit-test evidence and learning evidence. Do not
assume archived CB/virtual-stacking thresholds remain valid.

## Run bounded checks

Run focused existing tests or new minimal regression tests for the change.
Production graphs use `z`, `pos`, `edge_index`, `pos_code`, `block_idx`, and
domain metadata; do not substitute synthetic-only field aliases. Test malformed
target-domain inputs explicitly and legitimate non-target-domain pass-through.
For adapter work use `tests/test_adapters.py`, not the removed
`data_loading/validate_adapters.py`. Inspect skipped real-data/CUDA cases.

For geometry/masking changes, test placeholder perturbation, target-independent
masked input, valid physical endpoints, compact/full reindexing, mixed-domain
batch independence, and real masked nodes versus actual padding. For samplers,
test planned versus consumed identities over changing epoch lengths, including
the trainer integration; sampler-only length tests are insufficient.

## Keep workloads user-controlled

Never start, stop, resume or background training, training-style validation, or
benchmarks. Offer an existing Make target, ask whether the user ran it, and wait
for confirmation before monitoring. An evaluation-only workflow that does not
exist must be implemented and reviewed before proposing its command.

Data audits, processing, and conversion require explicit user execution
authorization; record exact inputs, destination, log and result, and never
overwrite production PTs by default. `make build-edge-free` has no defaults and
refuses to overwrite; give it a new destination prefix and inspect the arguments.

## Interpret runs honestly

After user confirmation, inspect host Python-worker identity, command/output
directory, GPU mapping, and two log-progress observations. If host visibility
is unavailable, report status unverified and request the read-only check.

Inspect finite losses/gradients, actual domain membership, edge validity,
real/V confusion, protein real-sidechain and heteroatom recall, RNA real-base
recall, exact-block metrics, support, and resource use. Overall accuracy, CB
alone, or a decreasing loss is not enough. Distinguish last validation, best
checkpoint validation, and held-out test. Per-task loss magnitudes are not
gradient-share measurements; a first-batch norm audit is only that batch.

Use fast runs for routing/crash checks, not as substitutes for controlled
multidomain quality comparisons. Hold datasets, masks, seeds and training budget
fixed when judging a change. Report regressions and uncertainty; do not promise
that a correctness fix must improve every metric. Commit/push only when the
user's requested verification and authorization gates are satisfied.

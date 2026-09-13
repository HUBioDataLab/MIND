# MIND validation and run checks

Validation has two parts: bounded tests that Codex may run directly, and
training workloads that remain user-controlled. This document records checks,
not an automatic launch procedure.

## Focused tests

Run the complete unit suite through the repository interface:

```bash
make test-unit
```

For adapter changes, run the production-contract tests directly as needed:

```bash
pytest -v tests/test_adapters.py
```

For the current RNA selection and geometry-neutral paths:

```bash
pytest -v tests/test_rna_june_equivalent.py
pytest -v tests/test_geometry_neutral_virtual_slots.py
pytest -v tests/test_dynamic_sampler_epoch_plan.py
```

These tests use production field contracts (`z`, `pos`, `edge_index`,
`pos_code`, `block_idx`, and `dataset_type`) and should fail loudly on malformed
target-domain data. The old `data_loading/validate_adapters.py` command was
removed; do not document or invoke it.

## Fast training gate

After focused tests pass, the user can run the fast Make target:

```bash
make train GPU=6 CONFIG=f LOG=feature_fast_check
```

The fast config is a three-epoch, reduced-chunk run with the same model/task
family as the full config. It is a smoke gate, not a quality claim. Before the
user starts it, compare the resolved config and verify the selected RNA base,
chunk range, graph limits, task weights, W&B mode, and geometry-neutral flags.

## What to inspect

First check the log for a traceback, OOM, missing metadata, NaN/Inf, or an
unexpected domain/path. Then check:

- loader discovery and `dataset_type` for every enabled domain;
- complete train/validation/test batch plans and `len(sampler)` matching yielded
  batches;
- no graph over its configured `max_graph_atoms` limit;
- full-pass train membership: selected, unique, and available counts agree and
  repeated count is zero;
- finite `train_total_loss`/`val_total_loss` and a broadly decreasing early
  curve;
- protein block-MLM and RNA nucleotide-MLM mask counts;
- real-only protein/RNA metric denominators and per-domain losses;
- measured batch/graph sizes and throughput rather than hard-coded it/s claims.

For a natural full-pass claim, do not stop at the sampler's `mode=full_pass` or
`Ready!` line. Compare that epoch's planned batch count with the progress bar
and `Total batches tracked`, then compare each domain's selected/unique counts
with its tracked count. Lightning may retain the initial
`num_training_batches` while an epoch sampler plan changes length; a mismatch
means the run was truncated (or ended early) and is not a verified full pass.

Historical CB/V and RNA anchor values are context, not acceptance gates for a
new representation. In particular, inspect protein backbone, CB, real
sidechain (`SCr`), and virtual recall separately; inspect RNA C1', backbone,
real base (`BaseR`), virtual, and real-only nucleotide metrics separately.
Do not treat aggregate `SC`/`Base` values that include padding as chemistry
accuracy.

## Data and pipeline checks

For a requested protein processing change, the supported sequence is:

```bash
pytest -v tests/test_adapters.py
make process-protein NUM_CHUNKS=1
```

Review the target before execution: its default includes `--force` and the
existing 40k output prefix. Use isolated output/cache paths for a new check.

Spot-check one generated `.pt` graph for `[N, 3]` coordinates, node-aligned
metadata, finite tensors, in-range edges, and exact 14-slot protein residues.
For RNA selection changes, use:

```bash
pytest -v tests/test_rna_june_equivalent.py
```

The June-equivalent derivation tools write immutable outputs and preserve source
fingerprints. Do not overwrite production PTs to make a test pass.

## Full-run readiness

When the fast gate is clean, the user may run:

```bash
make train GPU=6 CONFIG=m LOG=feature_multidomain_check
```

Before a long run, record the exact config, selected chunk files, seed, GPU,
metadata totals, graph-size quantiles, and expected validation cadence. Compare
future runs by atoms/s, edges/s, graphs/s, data wait, and step-time quantiles;
optimizer batches/s alone is not comparable under dynamic batching.

Training must not be started, stopped, resumed, or backgrounded by Codex. Once
the user confirms a run was started, status checks must identify the live Python
worker, its output directory, GPU mapping, and advancing log progress—not just a
remembered PID or one unchanged log snapshot.

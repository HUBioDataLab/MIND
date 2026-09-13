# MIND training guide

Training is started by the user through the Makefile. The agent may inspect
configs, logs, and tests, but must not launch or stop a training workload.

## Commands

```bash
make train CONFIG=f LOG=fast_check
make train CONFIG=m LOG=multidomain_run
make train GPU=6 CONFIG=m LOG=multidomain_run
make train GPU=auto CONFIG=f LOG=fast_check
```

`f` selects [`core/pretraining_config_fast.yaml`](../../core/pretraining_config_fast.yaml);
`m` selects [`core/pretraining_config_multidomain.yaml`](../../core/pretraining_config_multidomain.yaml).
The Make target chooses/prints the GPU, log path, output directory, config,
epoch override, and seed. Logs are written below `logs/<MonthYear>/` and
checkpoints below a run-specific `outputs/` directory. Use `make status` and
`make logs` for read-only inspection.

## Current multidomain setup

The approved full config enables PDB, UniMol, and June-equivalent 23-slot RNA;
QM9, metabolite, and DNA are disabled. Metabolite remains outside this
comparison until its graph contract is explicitly validated.

The current full config requests a natural full pass:

- `atom_sampling.enabled: false`;
- `epoch_atom_budget` and `domain_weights` are unused;
- `cycle_to_quota: false`;
- the sampler's epoch plan targets each selected train graph once per epoch;
- domains are proportionally interleaved while samples remain, with no quota
  cycling; any natural tail reflects the selected dataset sizes.

This is an intended policy, not proof that every run completed its plan. The
training loop can cache the initial dataloader batch count while the sampler
rebuilds a different-length plan at `set_epoch`. For each epoch, compare the
`Ready! ... batches created` plan with the progress-bar/summary count and the
per-domain selected-versus-tracked totals before calling the run full-pass.

The dynamic batch sampler uses metadata and packs graphs toward
`max_atoms_per_batch`. Per-domain `max_graph_atoms` checks happen before the
80/10/10 split. Dynamic batching is also used for validation/test when enabled,
but those loaders are deterministic and single-pass.

## Model and objectives

The entry point is [`core/train_pretrain.py`](../../core/train_pretrain.py),
which builds a `LazyUniversalDataset`, loaders, the ESA model, callbacks, and a
PyTorch Lightning trainer. The shared encoder is implemented in
[`core/molecular_encoder.py`](../../core/molecular_encoder.py) and the model
and task dispatcher in [`core/pretraining_model.py`](../../core/pretraining_model.py).

The current objective contains exactly these configured tasks, each with weight
`1.0`:

```yaml
pretraining_tasks: [long_range_distance, short_range_distance, mlm]
task_weights:
  long_range_distance: 1.0
  short_range_distance: 1.0
  mlm: 1.0
```

Protein MLM masks complete residues (14 slots) and RNA MLM masks complete
nucleotides (23 slots) when their block flags are enabled. UniMol remains
atom-level. The MLM loss gives virtual targets weight `0.2` through
`virtual_atom_loss_weight`; this is a weighting inside MLM, not an additional
element-balanced objective. There is currently no masked-real-slot distance
loss and no separate element-balanced loss.

The active virtual-slot design is the geometry-neutral graph view. For enabled
protein/RNA graphs it removes visible virtual nodes and masked sidechain/base
nodes from the encoder graph, then reads target slots from their encoded
backbone owner context. Physical distance objectives use geometry-valid nodes.
Removed sidechain/base coordinates are excluded from encoder geometry; retained
masked backbone coordinates still provide physical input. This path is implemented in
[`core/virtual_slot_graph.py`](../../core/virtual_slot_graph.py). The older
synthetic `masked_slot_template` flag remains explicitly disabled in the
approved configs and is historical experiment machinery, not the current
encoder input.

## Logging and checkpoints

The configs use explicit W&B logging. Training prints the resolved logger before
model creation, records loss and per-domain metrics, and adds protein/RNA
per-atom callbacks. The geometry-neutral diagnostics are observational: context
counterfactuals and representation-gradient audits do not add losses.

Validation cadence comes from `val_check_interval`; inspect the resolved value
in the config and interpret old logs carefully because older sampler lengths
could make fractional cadence misleading. A run is not evidence of a healthy
training process unless logs show finite losses, advancing epochs, and consumed
loader plans. A sampler line reporting `mode=full_pass` describes the plan; it
does not establish that Lightning consumed the whole plan.

## Data and resource prerequisites

All enabled chunks need matching metadata sidecars. The default loader uses
`num_workers: 0` with lazy chunk loading. Lower `max_atoms_per_batch` if the
selected GPU cannot hold the configured graph mix, and keep metabolite disabled
until its `bond_edge_index` contract is validated for mixed collation.

For a fresh worktree, run the focused tests first, then ask the user to run a
Make training command. Do not substitute a direct `python -m core.train_pretrain`
launch for the Make workflow.

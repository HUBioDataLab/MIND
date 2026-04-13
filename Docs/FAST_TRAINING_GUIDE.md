# Fast Training Guide

Quick sanity check config for validating code changes before full runs.

## Usage

```bash
# Fast test (~5-8 min, 3 epochs, 1 chunk per dataset)
make train GPU=6 CONFIG=f LOG=fast_test

# Full multidomain training
make train GPU=6 CONFIG=m LOG=my_run

# Other commands
make stop GPU=6        # Kill training on GPU 6
make status            # Show running processes + GPU usage
make logs              # List recent log files
```

## What the Fast Config Does

`core/pretraining_config_fast.yaml` is identical to the full multidomain config except:

| Parameter | Full | Fast |
|-----------|------|------|
| `chunk_range` (all datasets) | all chunks | `[0, 0]` (1 chunk each) |
| `max_epochs` | 50 | 3 |
| `val_check_interval` | 0.25 (4x/epoch) | 0.5 (2x/epoch) |
| `max_cache_chunks` | 10 | 4 |
| `batch_stats_log_frequency` | 25 | 10 |
| `log_per_type_frequency` | 25 | 10 |
| `wandb_run_name` | multidomain-run-1 | fast-sanity-check |

Model architecture, hyperparameters, and pretraining tasks are **exactly the same** — only data volume and epoch count differ.

## When to Use

- After any code change, before pushing to main
- Testing new loss functions or masking strategies
- Verifying non-protein losses are not degraded
- Checking that training starts, logs correctly, and loss decreases

## What to Check

1. **No errors/OOM** in the log file
2. **val_total_loss decreasing** across validations
3. **Per-type losses** (PROTEIN, COCONUT, QM9, UNIMOL) all reasonable
4. **Batch composition** shows mixed datasets

# Epoch-Balanced Multimodal Batching

## Problem

Multimodal pretraining uses protein, RNA, and UniMol in the same training stream. Before this change, each domain was consumed once per epoch. If protein or RNA had fewer samples than UniMol, the end of the epoch naturally became UniMol-only.

That was bad for two reasons:

- Late-epoch optimizer steps no longer saw all modalities.
- Batch statistics looked mixed early in the epoch but drifted to one domain near the end.

## Changed Files

- `data_loading/improved_dynamic_sampler.py`
  - Main sampler used when `use_improved_sampler: true`.
- `data_loading/dynamic_chunk_sampler.py`
  - Fallback sampler used when `use_improved_sampler: false`.
- `core/train_pretrain.py`
  - Enables cycling only for training loaders.

## Train vs Eval Behavior

Training:

```python
cycle_short_domains=is_multi_domain
```

Validation and test:

```python
cycle_short_domains=False
```

So only training reuses short-domain samples. Validation/test remain single-pass so metrics are not biased by repeated examples.

## How Samples Are Chosen

At epoch start, samples are first grouped by domain:

```text
protein -> [protein sample indices]
rna     -> [rna sample indices]
unimol  -> [unimol sample indices]
```

The chunk order and sample order inside chunks are still shuffled with the existing epoch-dependent seeds:

```text
seed + epoch
seed + epoch * 10000 + chunk_idx
```

After that, the sampler builds a weighted domain schedule. The schedule uses the domain proportions from the current split and a small `interleave_window = 10`.

Example proportions:

```text
protein: 10%
rna:     10%
unimol:  80%
```

This becomes roughly:

```text
protein: 1 slot
rna:     1 slot
unimol:  8 slots
```

The smooth weighted scheduler spreads those slots across the round instead of placing all protein/RNA at the beginning.

## Are Ratios Fixed During The Epoch?

The ratios are computed once when the sampler builds the epoch stream. They come from the number of samples available in the current train split. There is no YAML setting that controls these ratios.

Example:

```text
protein samples: 100
rna samples:      100
unimol samples:   800
```

This gives:

```text
protein: 10%
rna:     10%
unimol:  80%
```

Those ratios are not recomputed later inside the same epoch. If one domain reaches the end of its shuffled list, it wraps around; the domain schedule itself stays the same.

So if UniMol were the short domain and protein/RNA were longer, UniMol would also wrap around. The implementation does not switch to a new ratio like "only remaining domains"; it keeps the epoch-level multimodal mix stable.

## What Happens When Protein/RNA Run Out?

Short domains wrap around with modulo indexing:

```python
sample_pos = type_positions[dtype] % len(samples)
mixed_samples.append(samples[sample_pos])
type_positions[dtype] += 1
```

This means the replacement is not newly random at exhaustion time. It continues through the already-shuffled per-domain list and wraps back to the beginning when it reaches the end.

So if RNA has already consumed:

```text
rna_samples = [r7, r2, r5]
```

the next RNA picks are:

```text
r7, r2, r5, r7, r2, r5, ...
```

Because `rna_samples` was shuffled at epoch start, the repeated order changes across epochs, but it is deterministic inside one epoch.

## Batch Construction

After the mixed sample stream is built, batches are still atom-budgeted:

```text
add next sample while current_atoms + sample_atoms <= max_atoms_per_batch
otherwise start a new batch
```

So the domain scheduler decides candidate order; `max_atoms_per_batch` decides where batch boundaries fall.

## Important Edge Cases

- If only one domain is available, cycling is disabled automatically by `len(samples_by_type) > 1`.
- If validation/test run in multimodal mode, cycling stays disabled.
- UniMol can still dominate sample count, but protein/RNA should keep appearing near the end of training epochs instead of disappearing.

## How To Check Logs

Use batch statistics:

```bash
grep -n "Batch .*Composition\|PROTEIN\|RNA\|UNIMOL" logs/June2026/<run>.log | tail -160
```

Near the end of training epochs, expect `PROTEIN`, `RNA`, and `UNIMOL` to still appear in logged batch composition blocks.

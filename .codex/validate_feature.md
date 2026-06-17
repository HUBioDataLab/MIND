# Validate Feature — MIND Implementation Checklist

When implementing any new feature in MIND, follow this workflow from fast sanity check through multidomain validation.

## 1. Fast Sanity Check (CONFIG=f)

Run:
```
make train CONFIG=f LOG=<feature_name>
```

This runs `core/pretraining_config_fast.yaml` for ~10–20 min on a small slice of data (1 chunk each of pdb, unimol, rna; metabolite disabled). Check the log under `logs/<MonthYear>/`.

### Metrics to verify in the log

**Protein per-atom table** (search `PER-ATOM METRICS (Protein)` or `📊 PER-ATOM`):
- Backbone atoms (N/CA/C/O): Recall should be ≥0.90, typically >0.95 after a few epochs
- CB recall: Known issue (~0.017–0.28 due to virtual atom stacking) — do NOT treat a drop here as your fault unless you touched `protein_adapter.py`
- Virtual (V) recall: Should stay high (>0.80); a drop to ~0.50 or below suggests the model is now predicting real elements at virtual positions — check if you changed any masking or embedding logic
- CG/CD/CE atoms: Should show recall ≥0.70 after 10+ epochs
- Precision column: Structurally 1.0 for element-aware preds (out-of-vocab → index 35) — do not chase this

**Per-AA all-or-nothing table** (search `PER-AMINO-ACID`):
- GLY: Should show all 4 backbone real (0 virtual slots); SCr=0 (no real sidechain)
- TRP: Should show SCr=10 (10 real sidechain atoms, no virtuals) — highest recall expected
- ALA: Should show SCr=1 (only CB is real sidechain)

**RNA per-atom table** (search `RNA` or `rna_atom_metrics`):
- C1' recall: Should be ≥0.85 (C1' is backbone, virtuals stacked on it)
- Base atoms (N1/C2/N3/C4/C5/C6): Should show clear per-nucleotide separation

**Loss values** (search `train_total_loss` in wandb or log):
- Should decrease from epoch 1 to 5
- If loss stays flat or increases, check batch composition — `log_batch_statistics: true` prints domain breakdown

**Common fast-train errors to watch for:**
- `ValueError: Metadata is required for atom-aware batching!` → run `python data_loading/create_chunk_metadata.py --chunk-dir <dir>`
- `CUDA out of memory` → lower `max_atoms_per_batch` in config (current safe value for 32GB: 6000)
- `KeyError: bond_edge_index` → metabolite chunks have this key; keep `metabolite: enabled: false`
- `ValueError: out-of-range integer` in element embedding → element type not in vocab; check new atom types with `data_loading/validate_adapters.py`

## 2. Adapter/Data Changes

If you changed any adapter (`data_loading/adapters/*.py`) or the cache pipeline (`data_loading/cache_to_pyg.py`):

```
python data_loading/validate_adapters.py --adapter <name> --max_samples 5
```

Then re-process at least 1 chunk:
```
make process-protein NUM_CHUNKS=1
```

And spot-check atom counts in the resulting `.pt` file:
```python
import torch; g = torch.load('path/to/chunk_0.pt')[0]; print(g.x.shape, g.pos.shape)
```

Expected: `x` shape `[N, F]` where N is total atoms, F is feature dim; each protein residue should contribute exactly 14 atoms if 14-atom uniform repr is on.

## 3. Metric Callback Changes

If you changed `core/metrics/protein_per_atom_metrics_callback.py` or `rna_per_atom_metrics_callback.py`:

- Run fast train and compare the per-atom tables in the log to a known-good baseline (e.g., `logs/June2026/multidomain_100ep_pdb_unimol_rna_20260609_065900.log`)
- Specifically check: atom counts per AA match expected (GLY=4bb+0sc+10v, ALA=4+1+9, TRP=4+10+0)
- Check that "SCr" (sidechain real) counts exclude OXT and other terminal atoms — the fix in commit `603a7fd` uses vocab membership, not `is_virtual_atom()`

## 4. Multidomain Validation (CONFIG=m)

Only after fast train passes:
```
make train CONFIG=m LOG=<feature_name>
```

Watch for the first few validation cycles (every 0.25 epoch). Key indicators:
- `val_total_loss` should decrease
- `val_mlm_loss` for proteins should be lower than for unimol/rna (block-MLM is easier per-atom due to context)
- Per-domain loss breakdown: check `compute_per_type_losses: true` output

## 5. Known Stable Baseline

100-epoch multidomain run (pdb+unimol+rna, no metabolite):
`logs/June2026/multidomain_100ep_pdb_unimol_rna_20260609_065900.log`

CB recall at epoch 100: ~0.017–0.28 (known issue, not a regression)
Virtual recall at epoch 100: ~0.81 (dropped from 0.9998 — expected, model became more aggressive)
CG recall: ~0.70+, N/CA/C/O: ~0.95+

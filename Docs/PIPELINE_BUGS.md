# MIND Pipeline Bugs & Improvements

Identified during protein loss investigation and full pipeline audit (April 2025).

---

## Bug 1: Cross-Chain Sequence Edges in Proteins [CRITICAL]

**Status**: Fixed in `protein_adapter.py` (requires data reprocessing)

**Problem**: The hybrid edge construction (Tier 1: sequence edges) checks `entity_idx` to keep edges within the same chain. However, `protein_adapter.py` assigns `entity_idx=0` to ALL standard amino acids regardless of which chain they belong to. Combined with globally contiguous `block_idx`, this creates **false backbone edges across chain boundaries** in multi-chain proteins.

**Root Cause** (`protein_adapter.py` line 364):
```python
entity_idx = 0 if is_standard_aa else 1  # No chain distinction!
```

**Edge check** (`cache_to_pyg.py` line 360-362):
```python
if (neighbor_entity == current_entity and       # both 0 → passes
    neighbor_block == current_block + offset):   # contiguous → passes
```

**Example** (Hemoglobin, PDB 1HHO — 4 chains):
```
Chain A: ALA(block=0) - GLY(block=1) - ... - HIS(block=140)   entity_idx=0
Chain B: VAL(block=141) - HIS(block=142) - ... - ARG(block=287) entity_idx=0
                    ↑ FALSE EDGE ↑
HIS(140) ↔ VAL(141): same entity (0==0), sequential block (141==140+1) → edge created
But these residues are on separate chains, potentially 30+ Å apart.
```

With `sequence_neighbors_k=3`, each chain boundary generates up to 6 false directed edges.

**Fix**: Assign per-chain `entity_idx` in `protein_adapter.py`. Each chain gets a unique integer, heteroatoms get a separate value.

**Impact**: Affects all multi-chain proteins in `processed_14atom` chunks. Requires full reprocessing of protein data (pkl → pt).

---

## Bug 2: Distance Bins Too Coarse (5 bins / 20Å)

**Status**: ✅ Fixed

**Problem**: Config `distance_bins: 5` with `max_distance: 20.0` gives 4Å per bin. Covalent bonds (1.5Å), hydrogen bonds (2.8Å), and van der Waals contacts (3.5Å) all fall in **bin 0**. The distance prediction task cannot distinguish fundamental chemical interactions.

**Fix applied**: Replaced uniform linear binning with non-uniform bins via `torch.bucketize`. 16 total bins: 8 fine bins for 0–5Å (chemically critical range) and 8 coarse bins for 5–20Å.

```python
# pretraining_tasks.py — _distances_to_bins()
torch.bucketize(distances, self.bin_boundaries)

# pretraining_config_multidomain.yaml
distance_bins: 16
distance_bin_boundaries: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.5, 17.5, 20.0]
# Bin 0: [0,1Å)  Bin 1: [1,1.5Å)  ...  Bin 7: [4,5Å)  Bin 8: [5,7Å)  ... Bin 15: [20Å+)
```

**Impact**: Covalent bonds, H-bonds and vdW contacts now map to distinct bins; model can learn meaningful distance semantics.

---

## Bug 3: Random Edge Mask During Validation (Noisy Metrics)

**Status**: ✅ Fixed

**Problem**: `short_range_distance` loss applies a random 15% edge mask. This runs during both training AND validation via `_compute_pretraining_losses`. Validation loss is stochastic — same model, same data, different val loss each time.

**Fix applied**: Added `is_training: bool = True` parameter to `_compute_pretraining_losses` and `_compute_per_type_losses`. `validation_step` and `test_step` now pass `is_training=False`, which switches to an all-edges mask.

```python
# is_training=True  (training_step)  → random 15% mask
mask = (torch.rand(edge_index.size(1)) < 0.15).to(distances.device)
# is_training=False (validation/test) → all edges used deterministically
mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=distances.device)
```

**Impact**: Validation loss is now deterministic and comparable across runs and epochs.

---

## Bug 4: Edge Remapping via Python Loops (Performance)

**Status**: Not yet fixed

**Problem**: Edge-attention path converts edge indices through Python dict + list comprehension over ALL edges, every forward pass.

**Code** (`pretraining_model.py` lines 213-221):
```python
node_id_map = {int(n.item()): i for i, n in enumerate(unique_nodes)}
local_edge_index[0] = torch.tensor(
    [node_id_map[int(i)] for i in edge_index[0].tolist()], device=edge_index.device
)
```

**Fix**: Replace with `torch.unique(edge_index, return_inverse=True)` — single tensor op.

**Impact**: Can dominate step time for protein-sized graphs. Expected 1.3-2x speedup on forward pass.

---

## Bug 5: Coordinate Denoising Frame Mismatch (Latent)

**Status**: Not yet fixed (currently disabled in config)

**Problem**: `clean_pos` is saved before `forward()` centering, but `pos` gets centered. Loss target `true_noise = data.pos - data.clean_pos` mixes coordinate frames.

**Code**:
- `_ensure_coord_denoising_attributes` saves `clean_pos = pos.clone()` (uncentered)
- `forward()` overwrites `batch.pos` with centered coordinates
- `coordinate_denoising_loss` computes `data.pos - data.clean_pos` (centered - uncentered)

**Fix**: Center `clean_pos` alongside `pos` in `forward()`, or save `clean_pos` after centering.

**Impact**: Will produce incorrect training signal if coordinate denoising task is enabled.

---

## Bug 6: Per-Graph Python Loops in Critical Path

**Status**: Not yet fixed

**Problem**: CoM centering, MLM masking, and coord denoising all use Python for-loops over graphs. These run every training/validation step.

**Code** (`pretraining_model.py` line 136):
```python
for graph_id in torch.unique(batch_mapping):
    mask = batch_mapping == graph_id
    ...
```

**Fix**: Use `scatter_mean` for CoM; segment-based operations for masking.

**Impact**: Moderate speedup, especially with many small graphs per batch.

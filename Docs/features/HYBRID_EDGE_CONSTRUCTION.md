# Hybrid Edge Construction for Protein Graphs

## Overview

MIND implements a hybrid edge construction strategy inspired by the [Salad paper](https://www.nature.com/articles/s42256-025-01100-z) (Efficient protein structure generation with sparse denoising models). This approach replaces simple radius-based edges with a more sophisticated 3-tier system that captures both local backbone topology and long-range folding interactions.

## Motivation

Traditional graph neural networks for proteins use uniform spatial cutoffs (e.g., 5Å radius), resulting in O(N²) edges for dense structures. Salad proposes a sparse approach with O(N·K) complexity that maintains rich connectivity while being computationally tractable for large proteins.

## Edge Construction Strategy

For each residue, the model selects a fixed number of neighbors (default: **64 total**) using three complementary strategies:

### 1. Sequence-Based Edges (Tier 1)
- **Logic**: Residues close in the amino acid sequence are connected via peptide bonds and local backbone interactions.
- **Selection**: For each residue at position `i`, connect to nearest **k neighbors** in sequence (i±1, i±2, ..., i±k).
- **Purpose**: Preserves local chain geometry and secondary structure elements (α-helices, β-sheets).
- **MIND Implementation**: `sequence_neighbors_k = 3` (6 edges per residue)

### 2. Spatial Edges (Tier 2)
- **Logic**: Protein folding brings distant sequence positions into close 3D proximity (e.g., β-sheet formation, domain interfaces).
- **Selection**: k-nearest neighbors based on Euclidean distance between Cα atoms.
- **Constraint**: Edges already in Tier 1 are excluded to maximize information diversity.
- **Purpose**: Captures tertiary structure and long-range contacts.
- **MIND Implementation**: `max_spatial_neighbors = 48` (filtered for Tier 1 duplicates)

### 3. Random/Long-Range Edges (Tier 3)
- **Logic**: Pure local connectivity limits global information flow. Random edges enable efficient message passing across the entire protein.
- **Selection**: Randomly sample from remaining residue pairs with inverse distance weighting (1/d³).
- **Purpose**: Provides global context without full attention complexity.
- **MIND Implementation**: `num_random_edges = 8` (scaled down for smaller proteins vs. Salad's 32)

### Edge Count Distribution

| Tier | Salad (Default) | MIND (Proteins) | Purpose |
|------|-----------------|-----------------|---------|
| Sequence | 16 | 6 (k=3) | Backbone continuity |
| Spatial | 16 | 48 | Folding geometry |
| Random | 32 | 8 | Global context |
| **Total** | **64** | **~62** | Sparse graph |

**Note**: MIND adjusts counts to better suit the discriminative pretraining tasks (MLM, distance prediction) rather than generative denoising.

---

## Dynamic vs. Static Graphs

### Salad's Dynamic Approach
The original Salad paper uses **dynamic graph updates**:
- Edges (especially Tier 2 spatial neighbors) are recalculated at each denoising block
- As noisy coordinates are refined, the spatial neighbor list evolves
- Critical for generative models where atom positions change during forward pass

### MIND's Static Approach (Current Implementation)
MIND currently uses **static edges**:
- Edges are computed once during data preprocessing (`cache_to_pyg.py`)
- Coordinates are fixed during forward pass (discriminative tasks: MLM, distance prediction)
- Dynamic updates provide no benefit for tasks without coordinate denoising

---

## Implementation Details

### Stage 1: Static Hybrid Edges (✅ Implemented)

**Location**: `data_loading/cache_to_pyg.py`

**Functions**:
- `_build_sequence_edges()`: Connects residues via sequence proximity
- `_build_spatial_edges()`: k-NN based on Cα positions (using PyG's `radius_graph`)
- `_build_random_edges()`: Inverse-distance weighted sampling
- `_filter_duplicate_edges()`: Removes Tier 1 duplicates from Tier 2
- `_build_hybrid_edges()`: Orchestrates all three tiers

**Configuration** (`pretraining_config_multidomain.yaml`):
```yaml
use_hybrid_edges: true
sequence_neighbors_k: 3          # ±3 neighbors in sequence
max_spatial_neighbors: 48        # k-NN spatial edges
num_random_edges: 8              # Long-range connections
random_edge_min_distance: 10.0   # Minimum distance for random edges (Å)
```

**Cache Signature**: Hybrid edge parameters are included in the processed filename to ensure cache invalidation when parameters change.

**Advantages**:
- ✅ Computed once at data loading (no runtime overhead)
- ✅ Backward compatible (no model changes required)
- ✅ Fast and cacheable
- ✅ Sufficient for discriminative pretraining tasks

**Limitations**:
- ❌ No dynamic graph updates
- ❌ Not optimal for generative/denoising models

---

### Stage 2: Dynamic Hybrid Edges (⏸️ Not Implemented)

**Proposed Location**: `core/pretraining_model.py` → `UniversalMolecularEncoder.forward()`

**Logic**:
1. Tier 1 (sequence) and Tier 3 (random) remain static (coordinate-independent)
2. Tier 2 (spatial) is recalculated every N blocks using current atom positions
3. Requires coordinate denoising task (similar to Salad's generative objective)

**Why Not Implemented**:
- MIND's current tasks (MLM, distance prediction) use **fixed coordinates**
- Dynamic updates add computational overhead without benefit
- Stage 2 only makes sense when coordinates evolve during forward pass (e.g., structure generation, denoising)

**Future Consideration**:
If MIND adds coordinate denoising or generative tasks, Stage 2 can be implemented with:
```python
# Pseudo-code for dynamic spatial edges
for block_idx in range(num_blocks):
    x, pos = transformer_block(x, pos, edge_index)
    
    if block_idx % update_frequency == 0:
        # Recalculate Tier 2 spatial edges
        edge_index = update_spatial_edges(pos, edge_index_static)
```

---

## Comparison to Legacy System

| Aspect | Legacy (Radius Graph) | Hybrid Edges (Current) |
|--------|----------------------|------------------------|
| **Edge Selection** | Uniform 5Å cutoff | 3-tier (sequence + spatial + random) |
| **Complexity** | O(N²) worst case | O(N·K) guaranteed |
| **Backbone Info** | ❌ No sequence bias | ✅ Explicit sequence edges |
| **Global Context** | ❌ Limited by cutoff | ✅ Random long-range edges |
| **Edge Diversity** | ⚠️ Spatially clustered | ✅ Multi-scale connectivity |
| **Avg Edges/Residue** | ~38 (variable) | ~62 (fixed) |

---

**Dataset Status**:
- ✅ **Proteins**: All 10 chunks reprocessed with hybrid edges
- ⏳ **QM9**: Not yet implemented (future work)
- ⏳ **RNA**: Not yet implemented (future work)

---

## References

- **Paper**: [Efficient protein structure generation with sparse denoising models](https://www.nature.com/articles/s42256-025-01100-z)
- **Authors**: Salad Team (Nature Machine Intelligence, 2025)
- **Key Insight**: Sparse neighbor selection (sequence + spatial + random) achieves comparable accuracy to full attention with O(N·K) complexity

---

## Future Work

1. **Adaptive Edge Density**: Scale `max_spatial_neighbors` based on molecule size
   - Small molecules (QM9): Nearly complete graph
   - Metabolites: Dense connectivity
   - Proteins: Current strategy (48 neighbors)
   - RNA: Further sparsification

2. **Dynamic Edges for Generative Tasks**: Implement Stage 2 when coordinate denoising is added

3. **Multi-Domain Optimization**: Tune edge parameters separately for proteins, RNA, small molecules

4. **Alternative Spatial Metrics**: Explore distance-based edge weights (continuous) vs. binary connectivity

---

## Configuration Quick Reference

**Enable Hybrid Edges**:
```yaml
# core/pretraining_config_multidomain.yaml
use_hybrid_edges: true
sequence_neighbors_k: 3
max_spatial_neighbors: 48
num_random_edges: 8
random_edge_min_distance: 10.0
```

**Reprocess Dataset**:
```bash
python data_loading/process_chunked_dataset.py \
    --config-yaml-path core/pretraining_config_multidomain.yaml \
    --dataset-type pdb \
    --num-chunks 10
```

**Verify Implementation**:
```bash
# Check processed files include hybrid edge signature
ls ../data/proteins/processed_14atom_chunk_0/processed/*.pt

# Should see: processed_data_pdb_[...hash...]_hybrid_seq3_spat48_rand8.pt
```

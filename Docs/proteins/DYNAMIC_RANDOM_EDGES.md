# Dynamic Random Edges for Protein Graphs

## Overview

This implementation enables **epoch-varying random long-range edges** (Tier 3) in protein graph neural networks. Unlike static preprocessing, edges are generated at runtime, ensuring different connectivity patterns each epoch to prevent overfitting to specific long-range relationships.

## Key Design: CA-Atom Only

### Why CA (Alpha Carbon) Atoms?

Random edges are created **exclusively between CA atoms**, matching the preprocessing logic in `cache_to_pyg._build_random_edges()`.

**Rationale:**

1. **Structural Representation**: CA atoms form the protein backbone, with one CA per residue. They capture the essential 3D structure and long-range relationships without requiring all-atom edges.

2. **Computational Efficiency**: 
   - 300-residue protein: 300 CA atoms vs 4,200 total atoms (14-atom uniform)
   - CA-CA edges: 300² = 90K possible edges
   - All-atom edges: 4,200² ≈ 17.6M possible edges (~190× overhead)

3. **Information Density**: Side-chain atoms provide local information (captured by Tier 1/2 edges). CA-CA long-range edges capture global structural relationships relevant for folding and function.

4. **Literature Support**: Widely used in protein structure prediction (AlphaFold, ESM-Fold) and graph-based models (Salad, ProteinMPNN).

## Implementation

### File Structure

```
data_loading/
├── protein_dynamic_random_edges.py    # Runtime transform
├── cache_to_pyg.py                    # Preprocessing (static edges)
└── rebuild_edges.py                   # Tool to rebuild edges from cache

core/
├── train_pretrain.py                  # Training pipeline integration
└── pretraining_config_multidomain.yaml # Configuration
```

### Configuration

```yaml
# Static Tier 3 (preprocessing)
use_hybrid_edges: true
num_random_edges: 8
random_edge_min_distance: 10.0

# Dynamic Tier 3 (runtime)
use_dynamic_random_edges: false         # Enable for epoch-varying edges
replace_existing_random_edges: false    # Remove static Tier 3 if true
```

### Usage Modes

**Mode 1: Static Only (Default)**
```yaml
use_dynamic_random_edges: false
```
- Tier 3 edges from preprocessing
- Same edges every epoch
- Faster (no runtime overhead)

**Mode 2: Dynamic Additive**
```yaml
use_dynamic_random_edges: true
replace_existing_random_edges: false
```
- Static Tier 3 + Dynamic Tier 3
- ~10-15% more edges
- Both static and varying patterns

**Mode 3: Dynamic Replacement (Recommended)**
```yaml
use_dynamic_random_edges: true
replace_existing_random_edges: true
```
- Only dynamic Tier 3
- Same edge count, different each epoch
- Prevents bias from static patterns

## Performance

### Overhead Analysis (40K proteins)

```
Per-Protein Cost:
- Small (100 CA):    ~0.5ms
- Medium (300 CA):   ~2ms
- Large (1000 CA):   ~10ms

Total per Epoch:
- 40K × 2ms = 80 seconds ≈ 1.3 minutes

Impact:
- Baseline: 15-20 minutes/epoch
- With dynamic: 16-21 minutes/epoch
- Overhead: ~6-8% (acceptable)
```

### Optimization

The implementation uses:
- **Vectorized distance computation**: GPU-accelerated pairwise distances
- **Inverse distance³ weighting**: Matches preprocessing, biases toward closer long-range edges
- **Efficient sampling**: PyTorch's `multinomial` for weighted random selection
- **CA-only filtering**: Reduces search space by ~14× (14-atom uniform representation)

## Testing

Run the test suite:
```bash
python data_loading/protein_dynamic_random_edges.py
```

Expected output:
```
✅ Test 1: Additive Mode
   - Added: ~1,600 edges (8 per CA × 100 CA × 2 bidirectional)
   
✅ Test 2: Replacement Mode
   - Static Tier 3 removed, dynamic added

✅ Verification:
   - All edges connect CA atoms
   - All distances ≥ 10.0Å
```

## Protein-Specific Design

This implementation is **specifically designed for proteins** with the 14-atom uniform representation:

**Compatible:**
- ✅ PDB proteins (14-atom uniform)
- ✅ AlphaFold structures (after conversion)
- ✅ Any protein with `position_indices` attribute

**Not Compatible:**
- ❌ Small molecules (QM9, COCONUT)
- ❌ RNA/DNA (different backbone structure)
- ❌ Non-uniform atom representations

For small molecules, all-atom random edges may be appropriate (see deleted `dynamic_random_edges.py` for reference).

## Future Work

### 1. Multi-Domain Support
Extend to other molecular types with domain-specific strategies:
- **RNA/DNA**: Phosphate atom-based edges
- **Small molecules**: Distance-weighted all-atom edges
- **Complexes**: Interface-focused random edges

### 2. Adaptive Edge Density
Dynamically adjust `num_random_edges` based on protein size:
```python
num_edges = min(8, max(4, num_residues // 100))
```

### 3. Learned Edge Selection
Replace random sampling with learned attention:
- Predict which long-range edges are structurally important
- Use RL to optimize edge selection for downstream tasks

### 4. Hierarchical Random Edges
Multi-scale random edges:
- **Residue-level**: CA-CA (current)
- **Domain-level**: Between secondary structure elements
- **Chain-level**: For multi-chain complexes

### 5. Temporal Consistency
For molecular dynamics:
- Maintain edge consistency across trajectory frames
- Gradual edge evolution rather than complete randomization

## References

1. **Salad**: Structure-Aware Language-Augmented Diffusion for Protein Design

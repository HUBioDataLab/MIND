# Protein Loss Experiments — Summary of All Attempts

All experiments aimed to add a protein-specific pretraining loss to MIND's multi-domain pipeline.
MIND is a general molecular foundation model — the loss must teach atom-level and inter-molecular
interactions, not just protein-specific structural features.

Base tasks (`long_range_distance`, `short_range_distance`, `mlm`) work well across all molecule types.

---

## Run 1: Residue Type Prediction (`residue_loss.log`)

**Date**: Mar 6 | **Epochs**: 50 | **Steps/epoch**: ~7137

**What**: Mask random residues (all atoms set to `z=0`, `pos=0.0`) → predict amino acid type (20-class).

**Config**: `pos=0.0`, full edge removal for masked atoms.

**Result**: Loss decreased from ~2.29 to ~1.21 and plateaued there. The model learned something
but the task was **abandoned** — predicting 20 amino acid classes is a protein-structure-specific
task. MIND should learn general atom-level and inter-molecular interactions, not protein ontology.

**COCONUT final**: ~1.88 (healthy)

**Log evidence**:
```
Step 0   — residue_atom_reconstruction: 2.2892
Step 50  — residue_atom_reconstruction: 1.2053
Step 6250 — residue_atom_reconstruction: 1.2074  (50 epochs later, no change)
```

---

## Run 2: Atom Type Prediction + Coordinate Noise (`residue_loss_0_3.log`)

**Date**: Mar 11 | **Epochs**: 50 | **Steps/epoch**: ~5710

**What**: Switched prediction target from amino acid type to **atom element type**. Added Gaussian
noise (σ=0.3Å) to masked atom coordinates instead of `pos=0.0`. Applied CoM fix (exclude `z=0`
atoms from center calculation).

**Result**: Loss plateau at ~1.20 — essentially identical to Run 1. Changing the prediction target
and adding noise made **no measurable difference**.

**Why**: Edge isolation was still in place. Masked atoms received zero information from neighbors,
making prediction impossible regardless of coordinate handling.

**Log evidence**:
```
Step 0    — residue_atom_reconstruction: 2.2918
Step 50   — residue_atom_reconstruction: 1.2069
Step 2850 — residue_atom_reconstruction: 1.1949  (no improvement)
```

---

## Run 3a: BERT-style Edge Masking (`residue_loss_edge.log`)

**Date**: Mar 12 | **Epochs**: 10 | **Steps/epoch**: ~5710

**What**: Kept `unmasked→masked` edges (neighbors can see masked atoms), removed only
`masked→unmasked` (masked atoms can't copy). Real coordinates (no noise). CoM fix applied.

**Result**: Residue loss improved to ~1.13 (first meaningful improvement). However,
**COCONUT loss degraded** from ~1.88 to ~2.47 because the CoM fix was applied globally
(also excluded `z=0` MLM-masked atoms from CoM in small molecules, distorting their geometry).

**Log evidence**:
```
Step 0    — residue_atom_reconstruction: 2.2186
Step 4900 — residue_atom_reconstruction: 1.1568
COCONUT final — Total: 2.47  (was 1.88 in Run 1)
```

---

## Run 3b: Real Coordinates Only (`residue_loss_coordinat.log`)

**Date**: Mar 12 | **Epochs**: 10 | **Steps/epoch**: ~7137

**What**: Same as Run 1 but with real coordinates (no noise, no `pos=0.0`). No BERT edge masking
(full edge isolation still active). CoM fix applied globally.

**Result**: Loss at ~1.21 — no improvement over Run 1. Confirms that coordinates alone don't help
without edge connectivity. COCONUT also degraded (~2.58) due to global CoM bug.

**Log evidence**:
```
Step 0    — residue_atom_reconstruction: 2.2881
Step 6250 — residue_atom_reconstruction: 1.2258
COCONUT final — Total: 2.58  (degraded)
```

---

## Run 4: BERT Edge Final (`residue_loss_bert_final.log`)

**Date**: Mar 17 | **Epochs**: ~10 | **Steps/epoch**: ~10706

**What**: BERT edge masking with protein-scoped CoM fix (only exclude `z=0` for protein graphs).

**Result**: Residue loss at ~1.13. COCONUT improved but still slightly elevated.

**Log evidence**:
```
Step 9950 — residue_atom_reconstruction: 1.1260
```

---

## Run 5: Final BERT variant (`residue_loss_finalbert.log`)

**Date**: Mar 17

**Result**: Loss reverted to ~1.21 — likely the BERT edge fix was accidentally not applied
in this variant.

**Log evidence**:
```
Step 6250 — residue_atom_reconstruction: 1.2114
```

---

## Run 6: Two-Stage Atom Reconstruction (`residue_loss_melih.log`)

**Date**: Mar 27 | **Epochs**: 50 | **Steps/epoch**: ~10706

**What**: Complete redesign:
- **Stage 1**: Predict atom element type (11 classes: C, N, O, S, H, P, ..., Other)
- **Stage 2**: Predict coordinate displacement (Δpos) conditioned on predicted atom type embedding
- BERT-style edge masking
- Protein-scoped CoM fix
- Covalent boundary edges marked with sentinel value `-1.0`

**Result**: Atom type loss improved to ~1.04 (best across all runs). Coordinate loss: ~0.07.
Other losses (mlm: 0.21, short_range: 0.22) remained healthy. COCONUT: ~1.88 (healthy).

**Log evidence**:
```
Step 0    — residue_atom_type: 2.29 | residue_atom_coord: 0.28
Step 9950 — residue_atom_type: 1.0317 | residue_atom_coord: 0.0686
COCONUT final — Total: 1.88  (healthy)
train_epoch — mlm: 0.211 | short_range: 0.220  (healthy)
```

---

## Summary Table

| Run | Log file | Epochs | Residue Loss (final) | COCONUT (final) | Key Change |
|-----|----------|--------|---------------------|-----------------|------------|
| 1 | residue_loss.log | 50 | 1.21 | 1.88 | Baseline: amino acid prediction, z=0, edge isolation |
| 2 | residue_loss_0_3.log | 50 | 1.20 | — | Atom type prediction, noise σ=0.3 |
| 3a | residue_loss_edge.log | 10 | 1.15 | 2.47 | BERT edges (global CoM bug) |
| 3b | residue_loss_coordinat.log | 10 | 1.21 | 2.58 | Real coords, no BERT edges (global CoM bug) |
| 4 | residue_loss_bert_final.log | ~10 | 1.13 | — | BERT edges + protein-scoped CoM |
| 5 | residue_loss_finalbert.log | — | 1.21 | — | BERT edge fix accidentally missing |
| 6 | residue_loss_melih.log | 50 | 1.04 (type) + 0.07 (coord) | 1.88 | Two-stage, type-conditioned coords |

---

## Key Lessons

| Issue | Impact | Prevention |
|-------|--------|------------|
| Full edge removal for masked atoms | Blocks gradient flow, causes plateau at ~1.2 | Use BERT-style: keep unmasked→masked edges |
| Global CoM fix | Breaks COCONUT/QM9 (+30% loss increase) | Scope protein-only changes with `dataset_type` check |
| `pos=0.0` for masked atoms | Destroys geometric features | Use real coordinates or light noise |
| Too many simultaneous changes | Impossible to debug when things go wrong | One change at a time, validate with fast config |
| Modifying return types of shared functions | Breaks other callers | Keep interfaces stable |

---

## Final Decision

All residue loss code was removed. Codebase is clean with 3 base tasks:
`long_range_distance`, `short_range_distance`, `mlm`.

Next protein loss design must:
1. Keep masked atom coordinates real
2. Use BERT-style edge masking
3. Scope all changes to protein graphs only
4. Validate each change independently with `make train GPU=X CONFIG=f`
5. Verify non-protein losses are not degraded

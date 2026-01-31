# Atom Types in Protein Structures

## Dataset Analysis

**Dataset**: 40,000 AlphaFold protein structures from `/home/yusuf/data/proteins/raw_structures_hq_40k/`

### Found Elements

| Element | Count | Percentage | Description |
|---------|-------|------------|-------------|
| **C** | 28,166,466 | 63.43% | Carbon (backbone + sidechain) |
| **N** | 7,666,790 | 17.27% | Nitrogen (backbone + sidechain) |
| **O** | 8,353,663 | 18.81% | Oxygen (backbone + sidechain) |
| **S** | 217,132 | 0.49% | Sulfur (Cys, Met) |

**Total unique elements: 4** (C, N, O, S)

**Note**: Hydrogen atoms are not included in X-ray crystallography structures.

---

## Element Categories in Proteins

### Standard Elements (99.99% of proteins)
```python
["H", "C", "N", "O", "S"]  # 5 elements
```

### Common Additions (10-30% of proteins)
```python
["P", "Fe", "Zn", "Mg", "Ca"]  # Metal cofactors, phosphorylation
```

### Uncommon Elements (1-10% of proteins)
```python
["Cu", "Mn", "Co", "Ni", "Mo", "Se"]  # Metalloproteins, selenoproteins
```

### Rare Elements (<1%)
```python
["Cl", "K", "Na", "I", "Br", "F"]  # Ion channels, modifications
```

### Very Rare (<0.1%)
```python
["W", "V", "Cd", "As", "Hg"]  # Exotic metalloproteins
```

---

## Complete Element List

```python
ALL_PROTEIN_ELEMENTS = [
    "H",   # Z=1  | Standard
    "C",   # Z=6  | Standard
    "N",   # Z=7  | Standard
    "O",   # Z=8  | Standard
    "F",   # Z=9  | Rare
    "Na",  # Z=11 | Rare
    "Mg",  # Z=12 | Common
    "P",   # Z=15 | Common
    "S",   # Z=16 | Standard
    "Cl",  # Z=17 | Rare
    "K",   # Z=19 | Rare
    "Ca",  # Z=20 | Common
    "V",   # Z=23 | Very Rare
    "Mn",  # Z=25 | Uncommon
    "Fe",  # Z=26 | Common
    "Co",  # Z=27 | Uncommon
    "Ni",  # Z=28 | Uncommon
    "Cu",  # Z=29 | Uncommon
    "Zn",  # Z=30 | Common
    "As",  # Z=33 | Very Rare
    "Se",  # Z=34 | Uncommon
    "Br",  # Z=35 | Rare
    "Mo",  # Z=42 | Uncommon
    "Cd",  # Z=48 | Very Rare
    "I",   # Z=53 | Rare
    "W",   # Z=74 | Very Rare
    "Hg",  # Z=80 | Very Rare
]  # Total: 27 elements
```

---

## Recommended Sets

### Current Dataset (AlphaFold)
```python
CURRENT = ["C", "N", "O", "S"]  # 4 elements
```

### Realistic Set (covers most real proteins)
```python
REALISTIC = ["H", "C", "N", "O", "S", "P", "Fe", "Zn", "Mg", "Ca"]  # 10 elements
```

### Extended Set (includes metalloproteins)
```python
EXTENDED = [
    "H", "C", "N", "O", "S",           # Standard
    "P", "Fe", "Zn", "Mg", "Ca",       # Common
    "Cu", "Mn", "Co", "Ni", "Mo", "Se" # Uncommon
]  # 16 elements
```

---

## Why Only 4 Elements in Current Dataset?

1. **AlphaFold structures**: Clean protein backbones and sidechains only
2. **No cofactors**: Metal ions (Fe, Zn, etc.) filtered out (HETATM records excluded)
3. **No hydrogen**: X-ray crystallography cannot resolve H atoms
4. **Standard amino acids**: Only 20 canonical amino acids (C, N, O, S)

To include metal cofactors and modified amino acids, use experimental PDB structures with HETATM records.

---

## Model Support

**MIND current configuration:**
- `max_atomic_num: 118` (supports all elements in periodic table)
- Current dataset uses: **4 elements** (C, N, O, S)

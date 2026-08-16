"""Shared positional atom vocabularies for fixed-slot protein and RNA data.

The adapter priority lists and metric labels must describe the same positions.
Residue/base maps below make the expected canonical union explicit, while the
assertions prevent a copied or incomplete list from silently mapping real atoms
to the virtual metric class.
"""

from collections import Counter
from typing import Dict, Iterable, Tuple


# Sparse model ids. Common elements retain their physical atomic numbers;
# uncommon real elements collapse to R=119, while adapter-generated virtual
# padding is a separate prediction target at V=120. Input masking uses id 0,
# so PretrainingConfig.atom_types=121 covers the full 0..120 range.
RARE_ATOMIC_NUMBER = 119
VIRTUAL_ATOMIC_NUMBER = 120


PROTEIN_BACKBONE_ATOMS: Tuple[str, ...] = ('N', 'CA', 'C', 'O')

PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE: Dict[str, Tuple[str, ...]] = {
    'ALA': ('CB',),
    'ARG': ('CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'),
    'ASN': ('CB', 'CG', 'OD1', 'ND2'),
    'ASP': ('CB', 'CG', 'OD1', 'OD2'),
    'CYS': ('CB', 'SG'),
    'GLN': ('CB', 'CG', 'CD', 'OE1', 'NE2'),
    'GLU': ('CB', 'CG', 'CD', 'OE1', 'OE2'),
    'GLY': (),
    'HIS': ('CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'),
    'ILE': ('CB', 'CG1', 'CG2', 'CD1'),
    'LEU': ('CB', 'CG', 'CD1', 'CD2'),
    'LYS': ('CB', 'CG', 'CD', 'CE', 'NZ'),
    'MET': ('CB', 'CG', 'SD', 'CE'),
    'PHE': ('CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'),
    'PRO': ('CB', 'CG', 'CD'),
    'SER': ('CB', 'OG'),
    'THR': ('CB', 'OG1', 'CG2'),
    'TRP': ('CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'),
    'TYR': ('CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'),
    'VAL': ('CB', 'CG1', 'CG2'),
}

# Stable adapter order. NH1/NH2 are explicit ARG nitrogen positions, not H atoms.
PROTEIN_SIDECHAIN_ATOM_PRIORITY: Tuple[str, ...] = (
    'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2',
    'CE', 'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3', 'CH2',
    'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NZ', 'NH1', 'NH2',
    'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH',
    'SG', 'SD',
)

# OXT is an optional terminal backbone oxygen. It is reported by metrics but
# must not consume one of the ten fixed sidechain slots in newly built caches.
PROTEIN_TERMINAL_ATOMS: Tuple[str, ...] = ('OXT',)


RNA_BACKBONE_ATOMS: Tuple[str, ...] = (
    'P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
    "C2'", "O2'", "C1'",
)

# Legacy PDB atom aliases occasionally survive in mmCIF-derived structures.
# Normalize them only on the RNA path; protein position names are untouched.
RNA_POSITION_CODE_ALIASES = {
    'O1P': 'OP1',
    'O2P': 'OP2',
    'O3P': 'OP3',
    "C1*": "C1'",
    "C2*": "C2'",
    "C3*": "C3'",
    "C4*": "C4'",
    "C5*": "C5'",
    "O2*": "O2'",
    "O3*": "O3'",
    "O4*": "O4'",
    "O5*": "O5'",
}

# Optional terminal phosphate oxygen. Like protein OXT, it is useful to report
# for old caches but must not consume a fixed base slot in newly adapted data.
RNA_TERMINAL_ATOMS: Tuple[str, ...] = ('OP3',)

RNA_BASE_ATOMS_BY_NUCLEOTIDE: Dict[str, Tuple[str, ...]] = {
    'A': ('N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'),
    'G': ('N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'),
    'C': ('N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'),
    'U': ('N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6'),
}

# A nucleotide has at most 11 base atoms, but the union of positional labels
# across A/G/C/U has 15 names. Slot count and metric vocabulary size differ.
RNA_STANDARD_BASE_ATOM_PRIORITY: Tuple[str, ...] = (
    'N9', 'C8', 'N7', 'C6', 'N1', 'C2', 'N3', 'C4', 'C5',
    'N4', 'O4', 'N6', 'O6', 'N2', 'O2',
)

# Recognized positions seen in supported modified nucleotides. These retain a
# deterministic adapter order and their own metric rows instead of becoming V.
RNA_MODIFIED_BASE_ATOM_PRIORITY: Tuple[str, ...] = ('N5', 'N8', 'C7')
RNA_BASE_ATOM_PRIORITY = (
    RNA_STANDARD_BASE_ATOM_PRIORITY + RNA_MODIFIED_BASE_ATOM_PRIORITY
)


def normalize_rna_position_code(pos_code: str) -> str:
    """Normalize known legacy RNA atom labels to modern PDB/mmCIF names."""
    code = str(pos_code or '').strip()
    return RNA_POSITION_CODE_ALIASES.get(code, code)


def is_virtual_position_code(pos_code: str) -> bool:
    """Return true only for adapter-generated V or V<number> padding labels."""
    code = str(pos_code or '')
    return code == 'V' or (code.startswith('V') and code[1:].isdigit())


def rna_backbone_contract_violations(
    pos_codes: Iterable[str],
) -> Tuple[Tuple[str, ...], Dict[str, int]]:
    """Return missing and duplicated canonical RNA backbone positions."""
    counts = Counter(normalize_rna_position_code(code) for code in pos_codes)
    missing = tuple(code for code in RNA_BACKBONE_ATOMS if counts[code] == 0)
    duplicates = {
        code: counts[code]
        for code in RNA_BACKBONE_ATOMS
        if counts[code] > 1
    }
    return missing, duplicates


def has_exact_rna_backbone(pos_codes: Iterable[str]) -> bool:
    """Return whether every canonical RNA backbone position occurs once."""
    missing, duplicates = rna_backbone_contract_violations(pos_codes)
    return not missing and not duplicates


def is_p_only_rna_block(atoms: Iterable[object]) -> bool:
    """Return whether a block has exactly one real atom and that atom is P.

    Adapter-generated virtual padding is deliberately ignored. This detects
    both historical one-atom ``[P]`` blocks and future standardized forms such
    as ``[P, V1, ..., V11]`` without confusing a real atom whose element happens
    to be vanadium with a virtual positional slot.
    """
    real_position_codes = []
    for atom in atoms:
        position_code = normalize_rna_position_code(
            getattr(atom, 'pos_code', '')
        )
        is_virtual = bool(getattr(atom, 'is_virtual', False))
        if is_virtual or is_virtual_position_code(position_code):
            continue
        real_position_codes.append(position_code)
    return real_position_codes == ['P']


def count_p_only_rna_blocks(molecule: object) -> int:
    """Count coarse P-only nucleotide blocks in a UniversalMolecule-like object."""
    return sum(
        is_p_only_rna_block(getattr(block, 'atoms', ()))
        for block in getattr(molecule, 'blocks', ())
    )


def rna_molecule_contains_p_only_block(molecule: object) -> bool:
    """Return whether an RNA molecule mixes in any coarse P-only block."""
    return count_p_only_rna_blocks(molecule) > 0


def _union(values: Dict[str, Tuple[str, ...]]) -> set:
    return {atom for atom_names in values.values() for atom in atom_names}


assert len(PROTEIN_SIDECHAIN_ATOM_PRIORITY) == len(set(PROTEIN_SIDECHAIN_ATOM_PRIORITY))
assert set(PROTEIN_SIDECHAIN_ATOM_PRIORITY) == _union(PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE)
assert max(map(len, PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE.values())) == 10

assert len(RNA_BASE_ATOM_PRIORITY) == len(set(RNA_BASE_ATOM_PRIORITY))
assert set(RNA_STANDARD_BASE_ATOM_PRIORITY) == _union(RNA_BASE_ATOMS_BY_NUCLEOTIDE)
assert max(map(len, RNA_BASE_ATOMS_BY_NUCLEOTIDE.values())) == 11

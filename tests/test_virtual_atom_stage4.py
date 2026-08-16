from data_loading.adapters.protein_adapter import ProteinAdapter
from data_loading.adapters.rna_adapter import RNAAdapter
from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS,
    PROTEIN_SIDECHAIN_ATOM_PRIORITY,
    PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE,
    RNA_BACKBONE_ATOMS,
    RNA_BASE_ATOMS_BY_NUCLEOTIDE,
    VIRTUAL_ATOMIC_NUMBER,
    is_virtual_position_code,
)
from data_loading.cache_to_pyg import atom_to_model_id
from data_loading.data_types import UniversalAtom


def _element_for_position(code):
    if code == "P":
        return "P"
    if code.startswith("O"):
        return "O"
    if code.startswith("N"):
        return "N"
    if code.startswith("S"):
        return "S"
    return "C"


def _atom(code, index, block_index=0):
    return UniversalAtom(
        element=_element_for_position(code),
        position=(float(index), float(index % 3), float(index % 5)),
        pos_code=code,
        block_idx=block_index,
        atom_idx_in_block=index,
        entity_idx=0,
    )


def test_protein_canonical_gly_ala_trp_target_slots():
    adapter = ProteinAdapter(use_14atom_uniform=True)
    expected_real_sidechains = {"GLY": 0, "ALA": 1, "TRP": 10}

    for residue, expected_real_count in expected_real_sidechains.items():
        codes = list(PROTEIN_BACKBONE_ATOMS)
        codes.extend(PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE[residue])
        atoms = [_atom(code, index) for index, code in enumerate(codes)]
        standardized = adapter._standardize_residue_to_14_atoms(
            atoms, residue, block_idx=0
        )
        side_slots = standardized[4:]

        assert len(standardized) == 14
        assert len(side_slots) == 10
        assert sum(not atom.is_virtual for atom in side_slots) == expected_real_count
        assert sum(atom.is_virtual for atom in side_slots) == 10 - expected_real_count
        assert all(
            atom_to_model_id(atom) == VIRTUAL_ATOMIC_NUMBER
            for atom in side_slots
            if atom.is_virtual
        )

        if residue == "GLY":
            assert [atom.pos_code for atom in side_slots] == [
                f"V{index}" for index in range(1, 11)
            ]
        elif residue == "ALA":
            assert side_slots[0].pos_code == "CB"
            assert atom_to_model_id(side_slots[0]) == 6
            assert [atom.pos_code for atom in side_slots[1:]] == [
                f"V{index}" for index in range(2, 11)
            ]
        else:
            assert [atom.pos_code for atom in side_slots] == list(
                code
                for code in PROTEIN_SIDECHAIN_ATOM_PRIORITY
                if code in PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE["TRP"]
            )
            assert not any(atom.is_virtual for atom in side_slots)


def test_rna_canonical_a_c_g_u_target_slots():
    adapter = RNAAdapter(use_23atom_uniform=True)
    expected_real_counts = {"A": 10, "C": 8, "G": 11, "U": 8}

    for nucleotide, expected_real_count in expected_real_counts.items():
        codes = list(RNA_BACKBONE_ATOMS)
        codes.extend(RNA_BASE_ATOMS_BY_NUCLEOTIDE[nucleotide])
        atoms = [_atom(code, index) for index, code in enumerate(codes)]
        standardized = adapter._standardize_residue_to_23_atoms(
            atoms, nucleotide, block_idx=0
        )
        base_slots = standardized[12:]

        assert len(standardized) == 23
        assert len(base_slots) == 11
        assert sum(not atom.is_virtual for atom in base_slots) == expected_real_count
        assert sum(atom.is_virtual for atom in base_slots) == 11 - expected_real_count
        assert all(
            atom_to_model_id(atom) == VIRTUAL_ATOMIC_NUMBER
            for atom in base_slots
            if atom.is_virtual
        )
        assert all(
            is_virtual_position_code(atom.pos_code) == atom.is_virtual
            for atom in base_slots
        )

        assert {atom.pos_code for atom in base_slots if not atom.is_virtual} == set(
            RNA_BASE_ATOMS_BY_NUCLEOTIDE[nucleotide]
        )
        if expected_real_count < 11:
            assert [atom.pos_code for atom in base_slots[expected_real_count:]] == [
                f"V{index}" for index in range(expected_real_count + 1, 12)
            ]

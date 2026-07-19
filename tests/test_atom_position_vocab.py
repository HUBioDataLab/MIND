import unittest

from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS,
    PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE,
    PROTEIN_SIDECHAIN_ATOM_PRIORITY,
    RNA_BACKBONE_ATOMS,
    RNA_BASE_ATOMS_BY_NUCLEOTIDE,
    RNA_STANDARD_BASE_ATOM_PRIORITY,
    is_virtual_position_code,
    normalize_rna_position_code,
)
from data_loading.adapters.protein_adapter import ProteinAdapter
from data_loading.adapters.rna_adapter import RNAAdapter
from data_loading.data_types import UniversalAtom
from core.metrics.protein_atom_types import (
    PROTEIN_UNKNOWN_ATOM,
    PROTEIN_VIRTUAL_ATOM,
    pos_code_to_atom_class as protein_atom_class,
)
from core.metrics.rna_atom_types import (
    RNA_UNKNOWN_ATOM,
    RNA_VIRTUAL_ATOM,
    pos_code_to_atom_class as rna_atom_class,
)
from core.vocabulary import (
    map_lba_position_codes_to_get,
    map_rna_position_codes_to_get,
)


def _union(position_map):
    return {atom for atoms in position_map.values() for atom in atoms}


def _atom(pos_code, index):
    return UniversalAtom(
        element=pos_code[0],
        position=(float(index), float(index + 1), float(index + 2)),
        pos_code=pos_code,
        block_idx=0,
        atom_idx_in_block=index,
        entity_idx=0,
    )


class AtomPositionVocabularyTest(unittest.TestCase):
    def test_protein_priority_covers_every_standard_residue_position(self):
        self.assertEqual(
            set(PROTEIN_SIDECHAIN_ATOM_PRIORITY),
            _union(PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE),
        )
        self.assertLessEqual({'NH1', 'NH2'}, set(PROTEIN_SIDECHAIN_ATOM_PRIORITY))
        self.assertEqual(max(map(len, PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE.values())), 10)

    def test_rna_vocabulary_covers_all_standard_positions_with_11_slots(self):
        self.assertEqual(
            set(RNA_STANDARD_BASE_ATOM_PRIORITY),
            _union(RNA_BASE_ATOMS_BY_NUCLEOTIDE),
        )
        self.assertLessEqual({'N4', 'N6', 'O2', 'O4'}, set(RNA_STANDARD_BASE_ATOM_PRIORITY))
        self.assertEqual(len(RNA_STANDARD_BASE_ATOM_PRIORITY), 15)
        self.assertEqual(max(map(len, RNA_BASE_ATOMS_BY_NUCLEOTIDE.values())), 11)

    def test_get_mappings_cover_every_standard_position(self):
        protein_positions = PROTEIN_BACKBONE_ATOMS + PROTEIN_SIDECHAIN_ATOM_PRIORITY
        protein_fallbacks = [
            code for code in protein_positions
            if map_lba_position_codes_to_get(code) == 'sm'
        ]
        self.assertEqual(protein_fallbacks, [])

        rna_positions = RNA_BACKBONE_ATOMS + RNA_STANDARD_BASE_ATOM_PRIORITY
        rna_fallbacks = [
            code for code in rna_positions
            if map_rna_position_codes_to_get(code) == "'"
        ]
        self.assertEqual(rna_fallbacks, [])

    def test_only_generated_virtual_position_codes_map_to_virtual(self):
        for code in ('V', 'V1', 'V10'):
            with self.subTest(code=code):
                self.assertTrue(is_virtual_position_code(code))
                self.assertEqual(protein_atom_class(code), PROTEIN_VIRTUAL_ATOM)
                self.assertEqual(rna_atom_class(code), RNA_VIRTUAL_ATOM)

        for code in ('', 'VAL', 'VIRTUAL', 'MG'):
            with self.subTest(code=code):
                self.assertFalse(is_virtual_position_code(code))
                self.assertEqual(protein_atom_class(code), PROTEIN_UNKNOWN_ATOM)
                self.assertEqual(rna_atom_class(code), RNA_UNKNOWN_ATOM)

    def test_known_real_positions_never_map_to_virtual(self):
        for code in ('NH1', 'NH2', 'OXT'):
            with self.subTest(modality='protein', code=code):
                self.assertEqual(protein_atom_class(code), code)
                self.assertNotEqual(protein_atom_class(code), PROTEIN_VIRTUAL_ATOM)

        for code in ('N4', 'N6', 'O2', 'O4', 'N5', 'N8', 'C7'):
            with self.subTest(modality='rna', code=code):
                self.assertEqual(rna_atom_class(code), code)
                self.assertNotEqual(rna_atom_class(code), RNA_VIRTUAL_ATOM)

    def test_rna_legacy_names_are_normalized_without_touching_protein(self):
        self.assertEqual(normalize_rna_position_code('O1P'), 'OP1')
        self.assertEqual(normalize_rna_position_code('O2P'), 'OP2')
        self.assertEqual(normalize_rna_position_code('O3P'), 'OP3')
        self.assertEqual(normalize_rna_position_code('C1*'), "C1'")
        self.assertEqual(rna_atom_class('O1P'), 'OP1')
        self.assertEqual(rna_atom_class('O3P'), 'OP3')

    def test_protein_standardization_keeps_arg_positions_and_drops_oxt(self):
        names = list(PROTEIN_BACKBONE_ATOMS)
        names += list(PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE['ARG']) + ['OXT']
        atoms = [_atom(name, index) for index, name in enumerate(names)]

        standardized = ProteinAdapter.__new__(ProteinAdapter)._standardize_residue_to_14_atoms(
            atoms, 'ARG', block_idx=0
        )
        codes = [atom.pos_code for atom in standardized]

        self.assertEqual(len(codes), 14)
        self.assertIn('NH1', codes)
        self.assertIn('NH2', codes)
        self.assertNotIn('OXT', codes)
        self.assertEqual(sum(is_virtual_position_code(code) for code in codes), 3)

    def test_rna_standardization_keeps_11_slots_and_drops_op3(self):
        names = list(RNA_BACKBONE_ATOMS)
        names += list(RNA_BASE_ATOMS_BY_NUCLEOTIDE['C']) + ['OP3']
        atoms = [_atom(name, index) for index, name in enumerate(names)]
        c1_position = next(atom.position for atom in atoms if atom.pos_code == "C1'")

        standardized = RNAAdapter.__new__(RNAAdapter)._standardize_residue_to_23_atoms(
            atoms, 'C', block_idx=0
        )
        codes = [atom.pos_code for atom in standardized]
        virtuals = [atom for atom in standardized if atom.is_virtual]

        self.assertEqual(len(codes), 23)
        self.assertNotIn('OP3', codes)
        self.assertEqual(len(virtuals), 3)
        self.assertTrue(all(atom.position == c1_position for atom in virtuals))

if __name__ == '__main__':
    unittest.main()

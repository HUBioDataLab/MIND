import tempfile
import unittest
from pathlib import Path

from data_loading.adapters.protein_adapter import ProteinAdapter
from data_loading.adapters.rna_adapter import RNAAdapter
from data_loading.atom_position_vocab import has_exact_rna_backbone
from data_loading.data_types import UniversalAtom, UniversalBlock, UniversalMolecule


def _assert_universal_molecule_invariants(mol: UniversalMolecule, expected_dataset_type: str) -> None:
    """Validate core structural invariants of any UniversalMolecule."""
    assert isinstance(mol, UniversalMolecule), f"Expected UniversalMolecule, got {type(mol)}"
    assert isinstance(mol.id, str) and len(mol.id) > 0, "mol.id must be a non-empty string"
    assert mol.dataset_type == expected_dataset_type, (
        f"Expected dataset_type '{expected_dataset_type}', got '{mol.dataset_type}'"
    )
    assert isinstance(mol.blocks, list), "mol.blocks must be a list"
    assert isinstance(mol.properties, dict), "mol.properties must be a dict"

    all_atoms = mol.get_all_atoms()
    assert len(all_atoms) > 0, f"Molecule {mol.id} has no atoms"

    for block_idx, block in enumerate(mol.blocks):
        assert isinstance(block, UniversalBlock), f"Block {block_idx} must be UniversalBlock"
        assert isinstance(block.symbol, str), f"Block {block_idx} symbol must be str"
        assert len(block) == len(block.atoms), (
            f"Block {block_idx} len mismatch: len(block)={len(block)} vs len(atoms)={len(block.atoms)}"
        )
        for atom_idx, atom in enumerate(block.atoms):
            assert isinstance(atom, UniversalAtom), f"Atom {atom_idx} in block {block_idx} must be UniversalAtom"
            assert isinstance(atom.element, str), "atom.element must be str"
            assert isinstance(atom.position, tuple) and len(atom.position) == 3, (
                f"atom.position must be 3-tuple, got {atom.position}"
            )
            assert atom.block_idx == block_idx, (
                f"atom.block_idx mismatch: {atom.block_idx} != {block_idx}"
            )
            assert atom.atom_idx_in_block == atom_idx, (
                f"atom.atom_idx_in_block mismatch: {atom.atom_idx_in_block} != {atom_idx}"
            )
            assert isinstance(atom.entity_idx, int), "atom.entity_idx must be int"


class TestAdapters(unittest.TestCase):
    """Production-contract tests for dataset adapters."""

    def test_protein_adapter_14atom_uniform_representation_synthetic(self):
        """Test ProteinAdapter on synthetic PDB with ALA (has CB) and GLY (no CB)."""
        pdb_text = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.009   1.424   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.246   2.392   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       2.000  -0.765  -1.200  1.00 20.00           C
ATOM      6  N   GLY A   2       3.322   1.542   0.000  1.00 20.00           N
ATOM      7  CA  GLY A   2       3.985   2.846   0.000  1.00 20.00           C
ATOM      8  C   GLY A   2       5.485   2.695   0.000  1.00 20.00           C
ATOM      9  O   GLY A   2       6.082   1.621   0.000  1.00 20.00           O
TER      10      GLY A   2
END
"""
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            f.write(pdb_text.strip())
            temp_path = Path(f.name)

        try:
            adapter = ProteinAdapter(use_14atom_uniform=True)
            mol = adapter.convert_to_universal(temp_path)

            _assert_universal_molecule_invariants(mol, expected_dataset_type="protein")
            self.assertEqual(len(mol.blocks), 2)

            # Block 0: ALA
            ala_block = mol.blocks[0]
            self.assertEqual(ala_block.symbol, "ALA")
            self.assertEqual(len(ala_block.atoms), 14, "Every residue must have exactly 14 atoms")
            self.assertEqual([a.pos_code for a in ala_block.atoms[:4]], ["N", "CA", "C", "O"])
            self.assertEqual(ala_block.atoms[4].pos_code, "CB")
            self.assertEqual(ala_block.atoms[4].element, "C")
            # Remaining 9 atoms must be virtual 'V'
            for virtual_atom in ala_block.atoms[5:]:
                self.assertEqual(virtual_atom.element, "V")
                self.assertEqual(virtual_atom.position, ala_block.atoms[4].position, "Virtual atoms must be at CB")

            # Block 1: GLY
            gly_block = mol.blocks[1]
            self.assertEqual(gly_block.symbol, "GLY")
            self.assertEqual(len(gly_block.atoms), 14, "GLY residue must have exactly 14 atoms")
            self.assertEqual([a.pos_code for a in gly_block.atoms[:4]], ["N", "CA", "C", "O"])
            ca_pos = gly_block.atoms[1].position
            # For GLY, all 10 sidechain slots are virtual 'V' placed at CA
            for virtual_atom in gly_block.atoms[4:]:
                self.assertEqual(virtual_atom.element, "V")
                self.assertEqual(virtual_atom.position, ca_pos, "GLY virtual atoms must be at CA")

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_protein_adapter_real_sample_if_present(self):
        """Test ProteinAdapter on real structure if local raw data is available."""
        real_data_dir = Path("/home/yusuf/data/proteins/raw_structures_hq_40k")
        if not real_data_dir.is_dir():
            self.skipTest(f"Real protein data dir not present: {real_data_dir}")

        pdb_files = list(real_data_dir.glob("*.pdb"))[:2]
        if not pdb_files:
            self.skipTest("No .pdb files found in data dir")

        adapter = ProteinAdapter(use_14atom_uniform=True)
        for pdb_file in pdb_files:
            mol = adapter.convert_to_universal(pdb_file)
            _assert_universal_molecule_invariants(mol, expected_dataset_type="protein")
            for block in mol.blocks:
                self.assertEqual(len(block.atoms), 14, f"Residue {block.symbol} in {mol.id} must have 14 atoms")
                self.assertEqual([a.pos_code for a in block.atoms[:4]], ["N", "CA", "C", "O"])

    def test_rna_adapter_23slot_uniform_representation_if_present(self):
        """Test RNAAdapter on real RNA sample if local raw data is available."""
        real_rna_dir = Path("/home/yusuf/data/rna/raw_structures")
        if not real_rna_dir.is_dir():
            self.skipTest(f"Real RNA data dir not present: {real_rna_dir}")

        cif_files = list(real_rna_dir.glob("*.cif"))[:2]
        if not cif_files:
            self.skipTest("No .cif files found in RNA data dir")

        adapter = RNAAdapter(use_23atom_uniform=True)
        for cif_file in cif_files:
            mol = adapter.convert_to_universal(cif_file)
            _assert_universal_molecule_invariants(mol, expected_dataset_type="rna")
            for block in mol.blocks:
                # Production contract: Complete backbone nucleotides have exactly 23 atoms.
                # Incomplete backbone nucleotides (e.g. terminal missing P/OP1/OP2) have len(backbone) + 11.
                if has_exact_rna_backbone(a.pos_code for a in block.atoms):
                    self.assertEqual(
                        len(block.atoms),
                        23,
                        f"Complete backbone nucleotide {block.symbol} in {mol.id} must have 23 atoms",
                    )
                else:
                    self.assertGreaterEqual(
                        len(block.atoms),
                        11,
                        f"Incomplete nucleotide {block.symbol} in {mol.id} must have at least 11 sidechain slots",
                    )


    def test_adapter_fail_loud_on_invalid_path_or_manifest(self):
        """Test fail-loud contract when non-existent path or bad manifest is provided."""
        adapter = ProteinAdapter()
        with self.assertRaises(FileNotFoundError):
            adapter.load_raw_data("/non/existent/protein/path")

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("wrong_column_header\nvalue1\n")
            bad_manifest_path = Path(f.name)

        try:
            with self.assertRaises(ValueError):
                # load_raw_data requires 'repId' column in manifest
                adapter.load_raw_data("/tmp", manifest_file=str(bad_manifest_path))
        finally:
            if bad_manifest_path.exists():
                bad_manifest_path.unlink()


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

"""
Protein Adapter

Adapter to parse raw PDB/CIF files into the UniversalMolecule format.
Can be configured to handle proteins only or protein-heteroatom complexes.

Supports 14-atom uniform representation to prevent sequence leakage during MLM training.
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to sys.path to allow imports like 'data_loading.adapters'
sys.path.append(str(Path(__file__).resolve().parents[2]))

from data_loading.adapters.base_adapter import BaseAdapter
from data_loading.data_types import UniversalAtom, UniversalBlock, UniversalMolecule

# BioPython for reading PDB and CIF files
from Bio.PDB import PDBParser, MMCIFParser, is_aa

# Constants for 14-atom uniform representation
BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']  # 4 backbone atoms (standard for all amino acids)
MAX_SIDECHAIN_ATOMS = 10  # Tryptophan has 10 non-backbone heavy atoms
TOTAL_ATOMS_PER_RESIDUE = 14  # 4 backbone + 10 sidechain

# Canonical ordering for sidechain atoms (priority-based)
# This ensures consistent ordering across different PDB files
SIDECHAIN_ATOM_PRIORITY = [
    'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2', 
    'CE', 'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3', 'CH2',
    'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NZ', 
    'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH',
    'SG', 'SD'
]

# Virtual atom type (single generic type for all virtual atoms)
# This prevents sequence leakage: model cannot infer residue type from virtual atom type
VIRTUAL_ATOM = 'V'  # All virtual atoms have the same generic type

class ProteinAdapter(BaseAdapter):
    """
    Adapter to parse raw PDB/CIF files into the UniversalMolecule format.
    
    Supports both PDB and CIF formats, with optional filtering via manifest files.
    Can be configured to include or exclude heteroatoms (ligands, ions, etc.).
    """

    def __init__(self, include_hetatms: bool = False, use_14atom_uniform: bool = True) -> None:
        """
        Initialize ProteinAdapter.
        
        Args:
            include_hetatms: Include heteroatoms (ligands, ions) if True, only proteins if False
            use_14atom_uniform: Use 14-atom uniform representation to prevent sequence leakage
        """
        super().__init__("protein")
        self.include_hetatms = include_hetatms
        self.use_14atom_uniform = use_14atom_uniform
        # Initialize parsers with QUIET=True to suppress standard warnings
        self.pdb_parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)
        
        if self.use_14atom_uniform:
            print(f"✨ 14-atom uniform representation ENABLED")
            print(f"   Each residue will have exactly {TOTAL_ATOMS_PER_RESIDUE} atoms")
            print(f"   ({len(BACKBONE_ATOMS)} backbone + {MAX_SIDECHAIN_ATOMS} sidechain)")
            print(f"   Virtual atoms use generic type '{VIRTUAL_ATOM}' to prevent sequence leakage")

    def _get_cb_or_ca_position(self, atoms: List[UniversalAtom], res_name: str) -> Tuple[float, float, float]:
        """
        Get CB position for virtual atoms, or CA for Glycine.
        
        Args:
            atoms: List of atoms in the residue
            res_name: Residue name (e.g., 'GLY', 'ALA')
            
        Returns:
            Position tuple (x, y, z)
        """
        # For Glycine, use CA (no CB)
        if res_name == 'GLY':
            for atom in atoms:
                if atom.pos_code == 'CA':
                    return atom.position
        
        # For other residues, try to find CB
        for atom in atoms:
            if atom.pos_code == 'CB':
                return atom.position
        
        # Fallback to CA if CB not found
        for atom in atoms:
            if atom.pos_code == 'CA':
                return atom.position
        
        # Final fallback: first atom position
        if atoms:
            return atoms[0].position
        
        return (0.0, 0.0, 0.0)
    
    def _sort_sidechain_atoms(self, atoms: List[UniversalAtom]) -> List[UniversalAtom]:
        """
        Sort sidechain atoms by priority order for canonical representation.
        
        Args:
            atoms: List of sidechain atoms
            
        Returns:
            Sorted list of sidechain atoms
        """
        def get_priority(atom: UniversalAtom) -> int:
            """Get priority index for atom (lower = higher priority)"""
            try:
                return SIDECHAIN_ATOM_PRIORITY.index(atom.pos_code)
            except ValueError:
                # Atom not in priority list, put at end
                return len(SIDECHAIN_ATOM_PRIORITY)
        
        return sorted(atoms, key=get_priority)
    
    def _get_virtual_atom_type(self, res_name: str, virtual_index: int, real_sidechain_count: int) -> str:
        """
        Get virtual atom type (always generic to prevent sequence leakage).
        
        All virtual atoms have the same type 'V' to prevent the model from inferring
        residue identity from virtual atom types. This ensures the model learns from
        geometric and chemical features of REAL atoms only.
        
        Disambiguation between similar residues (e.g., Serine vs Cysteine) happens
        through their REAL atoms (OG vs SG), not virtual atoms.
        
        Args:
            res_name: Residue name (unused, kept for API compatibility)
            virtual_index: Index of this virtual atom (unused)
            real_sidechain_count: How many real sidechain atoms (unused)
            
        Returns:
            Generic virtual atom type 'V'
        """
        return VIRTUAL_ATOM
    
    def _standardize_residue_to_14_atoms(
        self, 
        atoms: List[UniversalAtom], 
        res_name: str,
        block_idx: int
    ) -> List[UniversalAtom]:
        """
        Standardize a residue to exactly 14 atoms (4 backbone + 10 sidechain).
        
        For residues with < 10 sidechain atoms, virtual atoms are added at CB position
        (or CA for Glycine). This prevents the model from inferring amino acid type
        from atom count during masked language modeling.
        
        Args:
            atoms: List of atoms in the residue
            res_name: Residue name (e.g., 'ALA', 'TRP')
            block_idx: Block index for atom creation
            
        Returns:
            List of exactly 14 atoms (4 backbone + 10 sidechain)
        """
        # Separate backbone and sidechain atoms
        backbone_atoms = []
        sidechain_atoms = []
        
        for atom in atoms:
            if atom.pos_code in BACKBONE_ATOMS:
                backbone_atoms.append(atom)
            else:
                sidechain_atoms.append(atom)
        
        # Sort sidechain atoms by canonical order
        sidechain_atoms = self._sort_sidechain_atoms(sidechain_atoms)
        
        # Get CB/CA position for virtual atoms
        virtual_position = self._get_cb_or_ca_position(atoms, res_name)
        
        # Create standardized sidechain (exactly 10 atoms)
        standardized_sidechain = []
        real_sidechain_count = len(sidechain_atoms)
        
        for i in range(MAX_SIDECHAIN_ATOMS):
            if i < len(sidechain_atoms):
                # Real atom exists
                atom = sidechain_atoms[i]
                # Update atom_idx_in_block to reflect new position
                atom.atom_idx_in_block = len(backbone_atoms) + i
                standardized_sidechain.append(atom)
            else:
                # Need to add virtual atom
                virtual_idx = i - real_sidechain_count
                virtual_type = self._get_virtual_atom_type(res_name, virtual_idx, real_sidechain_count)
                
                virtual_atom = UniversalAtom(
                    element=virtual_type,
                    position=virtual_position,
                    pos_code=f'V{i+1}',  # V1, V2, V3, ... for virtual atoms
                    block_idx=block_idx,
                    atom_idx_in_block=len(backbone_atoms) + i,
                    entity_idx=0,  # Protein entity
                    is_virtual=True,
                    virtual_type=virtual_type
                )
                standardized_sidechain.append(virtual_atom)
        
        # Combine: 4 backbone + 10 sidechain = 14 atoms
        standardized_atoms = backbone_atoms + standardized_sidechain
        
        # Ensure backbone atoms have correct atom_idx_in_block
        for i, atom in enumerate(standardized_atoms):
            atom.atom_idx_in_block = i
        
        return standardized_atoms

    def load_raw_data(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        manifest_file: Optional[str] = None,
        **kwargs
    ) -> List[Path]:
        """
        Load protein structure files, optionally filtering by manifest file.
        
        Args:
            data_path: Directory containing .pdb/.cif files
            max_samples: Limit number of files (applied after manifest filtering)
            manifest_file: Optional CSV file with 'repId' column for filtering
            
        Returns:
            List of Path objects to protein structure files
        """
        print(f"Scanning protein structures: {data_path}")
        data_path_obj = Path(data_path)
        
        if not data_path_obj.is_dir():
            raise FileNotFoundError(f"Specified data_path is not a directory: {data_path}")

        # If manifest provided, use it to filter specific files
        if manifest_file:
            import pandas as pd
            print(f"📋 Loading manifest: {manifest_file}")
            
            try:
                manifest_df = pd.read_csv(manifest_file)
            except Exception as e:
                raise ValueError(f"Failed to load manifest file: {e}") from e
            
            if 'repId' not in manifest_df.columns:
                raise ValueError(
                    f"Manifest file must have 'repId' column, "
                    f"found: {manifest_df.columns.tolist()}"
                )
            
            # Extract protein IDs
            protein_ids = manifest_df['repId'].tolist()
            print(f"📋 Manifest contains {len(protein_ids):,} protein IDs")
            
            files: List[Path] = []
            missing_files: List[str] = []
            
            for protein_id in protein_ids:
                # Try AlphaFold naming convention first
                pdb_path = data_path_obj / f"AF-{protein_id}-F1-model_v4.pdb"
                cif_path = data_path_obj / f"AF-{protein_id}-F1-model_v4.cif"
                # Also try direct filename match
                direct_pdb = data_path_obj / f"{protein_id}.pdb"
                direct_cif = data_path_obj / f"{protein_id}.cif"
                
                if pdb_path.exists():
                    files.append(pdb_path)
                elif cif_path.exists():
                    files.append(cif_path)
                elif direct_pdb.exists():
                    files.append(direct_pdb)
                elif direct_cif.exists():
                    files.append(direct_cif)
                else:
                    missing_files.append(protein_id)
            
            if missing_files:
                print(f"⚠️  {len(missing_files):,} files not found (will be skipped)")
                if len(missing_files) <= 5:
                    print(f"   Missing: {missing_files}")
                else:
                    print(f"   First 5 missing: {missing_files[:5]}")
            
            print(f"✅ Found {len(files):,}/{len(protein_ids):,} files from manifest")
        else:
            # Directory scan mode
            files = sorted(list(data_path_obj.glob("*.pdb")))
            files.extend(sorted(list(data_path_obj.glob("*.cif"))))
            
            if not files:
                raise FileNotFoundError(
                    f"No .pdb or .cif files found in directory: {data_path}"
                )
            
            print(f"Found {len(files):,} structure files in total.")
        
        # Apply max_samples limit
        if max_samples:
            print(f"Processing limited to {max_samples:,} samples.")
            return files[:max_samples]
        
        return files

    def create_blocks(self, raw_item: Path) -> List[UniversalBlock]:
        """
        Parse a single PDB/CIF file into a list of UniversalBlock objects.
        
        Each amino acid residue becomes one UniversalBlock. If `include_hetatms`
        is True, non-standard residues (ligands, ions) are also converted to blocks.
        
        Args:
            raw_item: Path to PDB or CIF file
            
        Returns:
            List of UniversalBlock objects (one per residue)
        """
        try:
            suffix = raw_item.suffix.lower()
            if suffix == ".pdb":
                structure = self.pdb_parser.get_structure(
                    id=raw_item.stem,
                    file=str(raw_item)
                )
            elif suffix == ".cif":
                structure = self.cif_parser.get_structure(
                    structure_id=raw_item.stem,
                    file=str(raw_item)
                )
            else:
                return []
        except Exception as e:
            print(f"ERROR: File cannot be read {raw_item.name}: {type(e).__name__} - {e}")
            return []

        blocks: List[UniversalBlock] = []
        
        # Iterate through hierarchy: model -> chain -> residue
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Check if residue is a standard amino acid
                    is_standard_aa = (
                        residue.get_id()[0] == ' ' or
                        is_aa(residue, standard=True)
                    )
                    
                    # Skip heteroatoms if not included
                    if not self.include_hetatms and not is_standard_aa:
                        continue
                    
                    block_atoms: List[UniversalAtom] = []
                    res_name = residue.get_resname()
                    entity_idx = 0 if is_standard_aa else 1
                    
                    for atom in residue:
                        # Skip hydrogen atoms to reduce complexity
                        element = (atom.element or "C").strip().upper()
                        if element == 'H':
                            continue
                        
                        # Handle alternate locations (keep only primary or 'A')
                        altloc = atom.get_altloc()
                        if altloc not in (" ", "A"):
                            continue
                        
                        uni_atom = UniversalAtom(
                            element=element,
                            position=tuple(atom.get_coord().tolist()),
                            pos_code=atom.get_name(),
                            block_idx=len(blocks),
                            atom_idx_in_block=len(block_atoms),
                            entity_idx=entity_idx
                        )
                        block_atoms.append(uni_atom)
                    
                    if block_atoms:
                        # Apply 14-atom standardization for standard amino acids
                        if self.use_14atom_uniform and is_standard_aa:
                            block_atoms = self._standardize_residue_to_14_atoms(
                                block_atoms, 
                                res_name, 
                                len(blocks)
                            )
                        
                        # Create a UniversalBlock for the residue if it contains any atoms after filtering.
                        block = UniversalBlock(
                            symbol=res_name, # e.g., 'ALA', 'LYS', or 'ZN', 'HEM' for heteroatoms
                            atoms=block_atoms
                        )
                        blocks.append(block)
        
        return blocks

    def convert_to_universal(self, raw_item: Path) -> UniversalMolecule:
        """
        Convert raw PDB/CIF file to UniversalMolecule.
        
        Args:
            raw_item: Path to PDB or CIF file
            
        Returns:
            UniversalMolecule with blocks representing residues
        """
        blocks = self.create_blocks(raw_item)
        return UniversalMolecule(
            id=raw_item.stem,
            dataset_type=self.dataset_type,
            blocks=blocks,
            properties={}
        )
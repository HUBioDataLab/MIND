#!/usr/bin/env python3

"""
RNA Adapter

Adapter to parse raw RNA CIF files into the UniversalMolecule format.
Processes RNA structures from databases like RNA3DB, filtering for
standard RNA nucleotides (A, U, G, C) and their modified variants.
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path to allow imports like 'data_loading.adapters'
sys.path.append(str(Path(__file__).resolve().parents[2]))

from data_loading.adapters.base_adapter import BaseAdapter
from data_loading.data_types import UniversalAtom, UniversalBlock, UniversalMolecule

# BioPython for reading CIF files
from Bio.PDB import MMCIFParser

class RNAAdapter(BaseAdapter):
    """
    Adapter to parse raw RNA CIF files into the UniversalMolecule format.
    
    Supports mmCIF format with optional filtering via manifest files.
    Processes standard RNA nucleotides (A, U, G, C) and common modified bases.
    """

    # Standard RNA nucleotides and common modified variants
    STANDARD_RNA_BASES = {
        'A', 'U', 'G', 'C',  # Standard bases
        'DA', 'DT', 'DG', 'DC',  # DNA bases (sometimes present)
        'ADE', 'URA', 'GUA', 'CYT',  # Full names
        'PSU', 'I', '1MA', '5MC', '7MG',  # Common modifications
        'M2G', 'OMC', 'OMG', 'YG', 'H2U'
    }

    def __init__(self, include_modified: bool = True, include_ions: bool = False) -> None:
        """
        Initialize RNAAdapter.
        
        Args:
            include_modified: Include modified RNA bases if True
            include_ions: Include metal ions and small molecules if True
        """
        super().__init__("rna")
        self.include_modified = include_modified
        self.include_ions = include_ions
        # Initialize parser with QUIET=True to suppress warnings
        self.cif_parser = MMCIFParser(QUIET=True)

    def load_raw_data(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        manifest_file: Optional[str] = None,
        **kwargs
    ) -> List[Path]:
        """
        Load RNA structure files, optionally filtering by manifest file.
        
        Args:
            data_path: Directory containing .cif files
            max_samples: Limit number of files (applied after manifest filtering)
            manifest_file: Optional CSV file with 'name' column for filtering
            
        Returns:
            List of Path objects to RNA structure files
        """
        print(f"Scanning RNA structures: {data_path}")
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
            
            if 'name' not in manifest_df.columns:
                raise ValueError(
                    f"Manifest file must have 'name' column, "
                    f"found: {manifest_df.columns.tolist()}"
                )
            
            # Extract RNA IDs
            rna_ids = manifest_df['name'].tolist()
            print(f"📋 Manifest contains {len(rna_ids):,} RNA structure IDs")
            
            files: List[Path] = []
            missing_files: List[str] = []
            
            for rna_id in rna_ids:
                # Try with .cif extension
                cif_path = data_path_obj / f"{rna_id}.cif"
                
                if cif_path.exists():
                    files.append(cif_path)
                else:
                    missing_files.append(rna_id)
            
            if missing_files:
                print(f"⚠️  {len(missing_files):,} files not found (will be skipped)")
                if len(missing_files) <= 5:
                    print(f"   Missing: {missing_files}")
                else:
                    print(f"   First 5 missing: {missing_files[:5]}")
            
            print(f"✅ Found {len(files):,}/{len(rna_ids):,} files from manifest")
        else:
            # Directory scan mode
            files = sorted(list(data_path_obj.glob("*.cif")))
            
            if not files:
                raise FileNotFoundError(
                    f"No .cif files found in directory: {data_path}"
                )
            
            print(f"Found {len(files):,} structure files in total.")
        
        # Apply max_samples limit
        if max_samples:
            print(f"Processing limited to {max_samples:,} samples.")
            return files[:max_samples]
        
        return files

    def _is_rna_residue(self, residue) -> bool:
        """
        Check if a residue is an RNA nucleotide.
        
        Args:
            residue: BioPython residue object
            
        Returns:
            True if residue is RNA, False otherwise
        """
        res_name = residue.get_resname().strip()
        
        # Check for standard bases
        if res_name in {'A', 'U', 'G', 'C'}:
            return True
        
        # Check for modified bases if enabled
        if self.include_modified and res_name in self.STANDARD_RNA_BASES:
            return True
        
        return False

    def create_blocks(self, raw_item: Path) -> List[UniversalBlock]:
        """
        Parse a single RNA CIF file into a list of UniversalBlock objects.
        
        Each RNA nucleotide becomes one UniversalBlock. If `include_ions`
        is True, metal ions and cofactors are also converted to blocks.
        
        Args:
            raw_item: Path to CIF file
            
        Returns:
            List of UniversalBlock objects (one per nucleotide)
        """
        try:
            structure = self.cif_parser.get_structure(
                structure_id=raw_item.stem,
                filename=str(raw_item)
            )
        except Exception as e:
            print(f"ERROR: File cannot be read {raw_item.name}: {type(e).__name__} - {e}")
            return []

        blocks: List[UniversalBlock] = []
        
        # Iterate through hierarchy: model -> chain -> residue
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Check if residue is RNA
                    is_rna = self._is_rna_residue(residue)
                    
                    # Check for heteroatoms (ions, cofactors)
                    is_hetatm = residue.get_id()[0].startswith('H_')
                    
                    # Skip non-RNA and unwanted heteroatoms
                    if not is_rna and not (self.include_ions and is_hetatm):
                        continue
                    
                    block_atoms: List[UniversalAtom] = []
                    res_name = residue.get_resname().strip()
                    entity_idx = 0 if is_rna else 1
                    
                    for atom in residue:
                        # Get element, default to carbon if missing
                        element = (atom.element or "C").strip().upper()
                        
                        # Skip hydrogen atoms to reduce complexity
                        if element == 'H':
                            continue
                        
                        # Handle alternate locations (keep only primary or 'A')
                        altloc = atom.get_altloc()
                        if altloc not in (" ", "A"):
                            continue
                        
                        uni_atom = UniversalAtom(
                            element=element,
                            position=tuple(atom.get_coord().tolist()),
                            pos_code=atom.get_name().strip(),
                            block_idx=len(blocks),
                            atom_idx_in_block=len(block_atoms),
                            entity_idx=entity_idx
                        )
                        block_atoms.append(uni_atom)
                    
                    if block_atoms:
                        # Create a UniversalBlock for the nucleotide if it contains atoms
                        block = UniversalBlock(
                            symbol=res_name,  # e.g., 'A', 'U', 'G', 'C', or modified bases
                            atoms=block_atoms
                        )
                        blocks.append(block)
        
        return blocks

    def convert_to_universal(self, raw_item: Path) -> UniversalMolecule:
        """
        Convert raw RNA CIF file to UniversalMolecule.
        
        Args:
            raw_item: Path to CIF file
            
        Returns:
            UniversalMolecule with blocks representing nucleotides
        """
        blocks = self.create_blocks(raw_item)
        return UniversalMolecule(
            id=raw_item.stem,
            dataset_type=self.dataset_type,
            blocks=blocks,
            properties={}
        )


        # Add this to the END of your rna_adapter.py file

if __name__ == "__main__":
    import argparse
    import pickle
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description="Process RNA structures to Universal format")
    parser.add_argument("--data-path", required=True, help="Directory containing .cif files")
    parser.add_argument("--output-path", required=True, help="Output .pkl file path")
    parser.add_argument("--manifest-file", default=None, help="Optional manifest CSV file")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--include-modified", action="store_true", default=True, 
                       help="Include modified RNA bases")
    parser.add_argument("--include-ions", action="store_true", default=False,
                       help="Include metal ions")
    
    args = parser.parse_args()
    
    print("="*60)
    print("RNA Adapter - Universal Molecule Conversion")
    print("="*60)
    
    # Create adapter
    adapter = RNAAdapter(
        include_modified=args.include_modified,
        include_ions=args.include_ions
    )
    
    # Load raw data
    raw_files = adapter.load_raw_data(
        data_path=args.data_path,
        max_samples=args.max_samples,
        manifest_file=args.manifest_file
    )
    
    print(f"\n🔄 Converting {len(raw_files)} RNA structures to Universal format...")
    
    # Convert to universal format
    molecules = []
    skipped = 0
    
    for raw_file in tqdm(raw_files, desc="Processing RNA structures"):
        try:
            mol = adapter.convert_to_universal(raw_file)
            if mol.blocks:  # Only add if it has blocks
                molecules.append(mol)
            else:
                skipped += 1
        except Exception as e:
            print(f"\n⚠️  Error processing {raw_file.name}: {e}")
            skipped += 1
    
    print(f"\n✅ Successfully converted {len(molecules)} structures")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} structures due to errors")
    
    # Save to pickle
    print(f"\n💾 Saving to {args.output_path}...")
    import os
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, 'wb') as f:
        for mol in molecules:
            pickle.dump(mol, f)
    
    print(f"✅ Saved {len(molecules)} molecules to {args.output_path}")
    print(f"📊 File size: {os.path.getsize(args.output_path) / 1024 / 1024:.2f} MB")
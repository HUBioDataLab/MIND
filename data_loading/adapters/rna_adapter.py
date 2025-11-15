import sys
import json
from pathlib import Path
from typing import List, Any, Dict
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from data_loading.adapters.base_adapter import BaseAdapter
from data_loading.data_types import UniversalAtom, UniversalBlock, UniversalMolecule
from Bio import PDB


class RNAAdapter(BaseAdapter):
    """
    Adapter to parse RNA CIF files from filtered_rna_cifs folder 
    into UniversalMolecule format.
    Each nucleotide is treated as a separate block.
    """

    def __init__(self):
        super().__init__("rna")
        self.mmcif_parser = PDB.MMCIFParser()

    def load_raw_data(self, data_path: str, max_samples: int = None, **kwargs) -> List[Dict]:
        """
        Load RNA structures from CIF files in filtered_rna_cifs directory.
        Returns a list of file paths to process.
        """
        print(f"Loading RNA structures from: {data_path}")
        
        cif_dir = Path(data_path)
        if not cif_dir.is_dir():
            raise FileNotFoundError(f"CIF directory not found: {data_path}")
        
        # Find all .cif files
        cif_files = list(cif_dir.glob("*.cif"))
        print(f"Found {len(cif_files)} CIF files")
        
        if not cif_files:
            raise ValueError(f"No .cif files found in {data_path}")
        
        # Return as list of dicts with file paths
        items = [{"file_path": str(f), "id": f.stem} for f in sorted(cif_files)]
        
        if max_samples:
            print(f"Processing limited to {max_samples} samples")
            return items[:max_samples]
        
        return items

    def create_blocks(self, raw_item: Dict) -> List[UniversalBlock]:
        """
        Parse RNA structure from CIF file into UniversalBlock objects.
        Each nucleotide = one block.
        """
        try:
            file_path = raw_item.get("file_path")
            if not file_path:
                return []
            
            # Parse CIF file
            structure = self.mmcif_parser.get_structure("RNA", file_path)
            blocks = []
            nucleotide_index = 0
            
            # Extract sequence and coordinates from structure
            for model in structure:
                for chain in model:
                    residues = list(chain)
                    
                    for residue in residues:
                        block_atoms = []
                        res_name = residue.resname.strip()
                        
                        # Map 3-letter code to 1-letter code
                        nucleotide_symbol = self._map_nucleotide(res_name)
                        
                        # Extract atoms from this nucleotide
                        for atom in residue:
                            try:
                                element = atom.element.strip()
                                if not element or element == 'H':
                                    continue
                                
                                position = tuple(atom.coord)
                                pos_code = atom.name.strip()
                                
                                uni_atom = UniversalAtom(
                                    element=element,
                                    position=position,
                                    pos_code=pos_code,
                                    block_idx=nucleotide_index,
                                    atom_idx_in_block=len(block_atoms),
                                    entity_idx=0  # RNA is single entity
                                )
                                block_atoms.append(uni_atom)
                            except Exception:
                                continue
                        
                        # Create block for this nucleotide
                        if block_atoms:
                            block = UniversalBlock(
                                symbol=nucleotide_symbol,
                                atoms=block_atoms
                            )
                            blocks.append(block)
                            nucleotide_index += 1
            
            return blocks
            
        except Exception as e:
            print(f"ERROR processing {raw_item.get('id', 'unknown')}: {e}")
            return []

    def convert_to_universal(self, raw_item: Dict) -> UniversalMolecule:
        """Wrap blocks into UniversalMolecule with metadata."""
        blocks = self.create_blocks(raw_item)
        
        # Extract sequence from blocks
        sequence = "".join([block.symbol for block in blocks])
        
        # Try to extract metadata from CIF file
        metadata = self._extract_metadata(raw_item.get("file_path", ""))
        
        return UniversalMolecule(
            id=raw_item.get("id", "unknown"),
            dataset_type=self.dataset_type,
            blocks=blocks,
            properties={
                "sequence": sequence,
                "length": len(sequence),
                "resolution": metadata.get("resolution"),
                "structure_method": metadata.get("structure_method"),
                "release_date": metadata.get("release_date"),
            }
        )

    def _map_nucleotide(self, three_letter_code: str) -> str:
        """Convert 3-letter nucleotide code to 1-letter code."""
        mapping = {
            'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C',
            'DA': 'A', 'DU': 'U', 'DG': 'G', 'DC': 'C',
        }
        return mapping.get(three_letter_code.upper(), 'N')

    def _extract_metadata(self, file_path: str) -> Dict:
        """Extract metadata from CIF file header."""
        metadata = {
            "resolution": None,
            "structure_method": None,
            "release_date": None,
        }
        
        try:
            if not file_path:
                return metadata
            
            with open(file_path, 'r') as f:
                for line in f:
                    if '_reflns.d_resolution_high' in line:
                        parts = line.split()
                        if len(parts) > 1:
                            try:
                                metadata["resolution"] = float(parts[-1])
                            except:
                                pass
                    elif '_exptl.method' in line:
                        parts = line.split()
                        if len(parts) > 1:
                            metadata["structure_method"] = parts[-1].strip("'\"")
                    elif '_pdbx_database_status.recvd_initial_deposition_date' in line:
                        parts = line.split()
                        if len(parts) > 1:
                            metadata["release_date"] = parts[-1]
        except Exception:
            pass
        
        return metadata
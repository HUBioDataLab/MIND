# data_loading/adapters/protein_adapter.py

import sys
import os
from pathlib import Path
from typing import List, Any
from tqdm import tqdm

# Ana proje dizinini path'e ekleyerek diğer modüllere erişimi sağlıyoruz
sys.path.append(str(Path(__file__).resolve().parents[2]))

from data_loading.adapters.base_adapter import BaseAdapter
from data_loading.data_types import UniversalAtom, UniversalBlock, UniversalMolecule

# BioPython, PDB dosyalarını okumak için
from Bio.PDB import PDBParser, MMCIFParser, is_aa

class ProteinAdapter(BaseAdapter):
    """
    Ham PDB/CIF dosyalarını okuyup UniversalMolecule formatına dönüştüren adaptör.
    """

    def __init__(self):
        super().__init__("protein")
        self.pdb_parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)

    def load_raw_data(self, data_path: str, max_samples: int = None) -> List[Path]:
        """
        Verilen klasördeki tüm .pdb ve .cif dosyalarının yollarını bir liste olarak döndürür.
        """
        print(f"Protein yapıları taranıyor: {data_path}")
        if not Path(data_path).is_dir():
            raise FileNotFoundError(f"Belirtilen data_path bir klasör değil: {data_path}")

        files = sorted(list(Path(data_path).glob("*.pdb")))
        files.extend(sorted(list(Path(data_path).glob("*.cif"))))

        if not files:
            raise FileNotFoundError(f"Klasörde .pdb veya .cif dosyası bulunamadı: {data_path}")

        print(f"Toplam {len(files):,} adet yapı dosyası bulundu.")
        
        if max_samples:
            print(f"İşlem {max_samples:,} örnek ile sınırlandırıldı.")
            return files[:max_samples]
        
        return files

    def create_blocks(self, raw_item: Path) -> List[UniversalBlock]:
        """
        Tek bir PDB/CIF dosyasını okur ve UniversalBlock listesine dönüştürür.
        Bu implementasyonda her amino asit bir "blok" olarak kabul edilir.
        """
        try:
            suffix = raw_item.suffix.lower()
            if suffix == ".pdb":
                structure = self.pdb_parser.get_structure(id=raw_item.stem, file=str(raw_item))
            elif suffix == ".cif":
                structure = self.cif_parser.get_structure(id=raw_item.stem, file=str(raw_item))
            else:
                return [] # Desteklenmeyen format
        except Exception as e:
            print(f"UYARI: Dosya okunamadı {raw_item.name}: {e}")
            return []

        blocks = []
        atom_count_in_mol = 0

        for model in structure:
            for chain in model:
                for residue in chain:
                    # Sadece standart 20 amino asidi ve bilinmeyenleri (UNK) alıyoruz
                    if residue.get_id()[0] != ' ' and not is_aa(residue, standard=True):
                        continue

                    block_atoms = []
                    res_name = residue.get_resname()

                    for atom in residue:
                        # Hidrojenleri atla
                        element = (atom.element or "C").strip().upper()
                        if element == 'H':
                            continue
                        
                        # Alternatif konumları atla ('A' veya boş olanı tut)
                        altloc = atom.get_altloc()
                        if altloc not in (" ", "A"):
                            continue

                        uni_atom = UniversalAtom(
                            element=element,
                            position=tuple(atom.get_coord().tolist()),
                            pos_code=atom.get_name(), # örn: 'CA', 'CB', 'N'
                            block_idx=len(blocks),
                            atom_idx_in_block=len(block_atoms),
                            entity_idx=0 # Tek bir molekül olduğu için entity_idx = 0
                        )
                        block_atoms.append(uni_atom)
                        atom_count_in_mol += 1
                    
                    if block_atoms:
                        block = UniversalBlock(
                            symbol=res_name, # örn: 'ALA', 'LYS'
                            atoms=block_atoms
                        )
                        blocks.append(block)
        
        return blocks

    def convert_to_universal(self, raw_item: Path) -> UniversalMolecule:
        """
        Tek bir PDB/CIF dosya yolunu tam bir UniversalMolecule nesnesine dönüştürür.
        """
        blocks = self.create_blocks(raw_item)
        return UniversalMolecule(
            id=raw_item.stem, # Dosya adını ID olarak kullan (örn: '1a2b')
            dataset_type=self.dataset_type,
            blocks=blocks,
            properties={}
        )
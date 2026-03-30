#!/usr/bin/env python3
"""
UniMol Adapter

Streaming adapter for the UniMol LMDB ligand dataset.
This version keeps UniMol flat: one block per molecule, with atoms and
3D coordinates preserved and no fragment/motif decomposition.
"""

import os
import pickle
from typing import Any, Dict, List, Optional

import lmdb
from tqdm import tqdm

from data_loading.adapters.base_adapter import BaseAdapter
from data_loading.data_types import UniversalAtom, UniversalBlock, UniversalMolecule


class UniMolAdapter(BaseAdapter):
    """Adapter for UniMol ligand LMDB records."""

    def __init__(self):
        super().__init__("unimol")

    def _resolve_lmdb_path(self, data_path: str) -> str:
        """Accept either the LMDB file itself or a directory containing one."""
        if os.path.isdir(data_path):
            candidate = os.path.join(data_path, "train.lmdb")
            if os.path.isfile(candidate):
                return candidate
            lmdb_files = [
                os.path.join(data_path, name)
                for name in os.listdir(data_path)
                if name.endswith(".lmdb")
            ]
            if lmdb_files:
                return sorted(lmdb_files)[0]
            raise FileNotFoundError(f"No .lmdb file found in directory: {data_path}")
        return data_path

    def _load_lmdb_item(self, value: bytes) -> Optional[Dict[str, Any]]:
        try:
            item = pickle.loads(value)
        except Exception:
            return None

        if not isinstance(item, dict):
            return None

        atoms = item.get("atoms") or item.get("atom_symbols") or item.get("elements")
        coordinates = item.get("coordinates") or item.get("coords") or item.get("positions")
        smiles = item.get("smi") or item.get("smiles") or item.get("smile")

        if atoms is None or coordinates is None:
            return None

        try:
            atom_list = [str(atom).strip() for atom in atoms]
            coord_source = coordinates

            if isinstance(coord_source, (list, tuple)) and coord_source:
                first_coord = coord_source[0]
                if hasattr(first_coord, "__len__") and len(first_coord) == len(atom_list):
                    coord_source = first_coord

            if hasattr(coord_source, "tolist"):
                coord_source = coord_source.tolist()

            coord_list = [tuple(map(float, coord[:3])) for coord in coord_source]
        except Exception:
            return None

        if len(atom_list) != len(coord_list):
            return None

        return {
            "atoms": atom_list,
            "coordinates": coord_list,
            "smi": str(smiles).strip() if smiles is not None else "",
        }

    def _item_to_universal(self, item: Dict[str, Any], mol_id: str) -> UniversalMolecule:
        atoms: List[UniversalAtom] = []
        for atom_idx, (element, position) in enumerate(zip(item["atoms"], item["coordinates"])):
            atoms.append(
                UniversalAtom(
                    element=element,
                    position=position,
                    pos_code="sm",
                    block_idx=0,
                    atom_idx_in_block=atom_idx,
                    entity_idx=0,
                )
            )

        block = UniversalBlock(symbol="UNIMOL_MOL", atoms=atoms)
        properties = {"num_atoms": len(atoms)}
        if item.get("smi"):
            properties["smi"] = item["smi"]

        return UniversalMolecule(
            id=mol_id,
            dataset_type=self.dataset_type,
            blocks=[block],
            properties=properties,
        )

    def load_raw_data(self, data_path: str, max_samples: int = None, **kwargs) -> List[Any]:
        """Load a limited UniMol sample list for debugging or small runs."""
        lmdb_path = self._resolve_lmdb_path(data_path)
        num_chunks = kwargs.get("num_chunks")
        chunk_index = kwargs.get("chunk_index")

        env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False)
        raw_items: List[Any] = []
        try:
            with env.begin() as txn:
                total_entries = txn.stat().get("entries", 0)
                start_idx = 0
                end_idx = total_entries
                if num_chunks is not None and chunk_index is not None:
                    chunk_size = (total_entries + num_chunks - 1) // num_chunks
                    start_idx = chunk_index * chunk_size
                    end_idx = min(start_idx + chunk_size, total_entries)

                for row_idx, (_key, value) in enumerate(txn.cursor()):
                    if row_idx < start_idx:
                        continue
                    if row_idx >= end_idx:
                        break

                    item = self._load_lmdb_item(value)
                    if item is None:
                        continue

                    raw_items.append(item)
                    if max_samples is not None and len(raw_items) >= max_samples:
                        break
        finally:
            env.close()

        return raw_items

    def create_blocks(self, raw_item: Any) -> List[UniversalBlock]:
        """Create a single flat block from a UniMol record."""
        if not isinstance(raw_item, dict):
            return []
        return self._item_to_universal(raw_item, mol_id=raw_item.get("smi", "unimol_unknown")).blocks

    def convert_to_universal(self, raw_item: Any) -> UniversalMolecule:
        """Convert a raw UniMol record to a flat UniversalMolecule."""
        if not isinstance(raw_item, dict):
            raise ValueError("UniMol raw item must be a dictionary")

        mol_id = raw_item.get("smi") or raw_item.get("id") or "unimol_unknown"
        return self._item_to_universal(raw_item, mol_id=str(mol_id))

    def process_dataset(self, data_path: str, cache_path: str = None, **kwargs) -> int:
        """Stream UniMol LMDB records directly into a pickle cache."""
        lmdb_path = self._resolve_lmdb_path(data_path)
        max_samples = kwargs.get("max_samples")
        num_chunks = kwargs.get("num_chunks")
        chunk_index = kwargs.get("chunk_index")

        env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False)
        total_written = 0
        total_seen = 0
        skipped = 0
        out_f = None

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            out_f = open(cache_path, "wb")

        try:
            with env.begin() as txn:
                total_entries = txn.stat().get("entries", 0)
                start_idx = 0
                end_idx = total_entries
                if num_chunks is not None and chunk_index is not None:
                    chunk_size = (total_entries + num_chunks - 1) // num_chunks
                    start_idx = chunk_index * chunk_size
                    end_idx = min(start_idx + chunk_size, total_entries)

                progress = tqdm(total=max(0, end_idx - start_idx), desc=f"Processing {self.dataset_type}")
                for row_idx, (_key, value) in enumerate(txn.cursor()):
                    if row_idx < start_idx:
                        continue
                    if row_idx >= end_idx:
                        break

                    total_seen += 1
                    item = self._load_lmdb_item(value)
                    if item is None:
                        skipped += 1
                        progress.update(1)
                        continue

                    mol_id = item.get("smi") or f"unimol_{row_idx}"
                    universal_item = self._item_to_universal(item, mol_id=str(mol_id))
                    if out_f is not None:
                        pickle.dump(universal_item, out_f, protocol=pickle.HIGHEST_PROTOCOL)
                    total_written += 1
                    progress.update(1)

                    if max_samples is not None and total_written >= max_samples:
                        break

                progress.close()
        finally:
            if out_f is not None:
                out_f.close()
            env.close()

        print(
            f"✅ UniMol processing complete: seen={total_seen:,}, written={total_written:,}, skipped={skipped:,}"
        )
        return total_written
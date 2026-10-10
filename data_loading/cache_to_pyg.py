#!/usr/bin/env python3

# data_loading/cache_to_pyg.py

"""
Universal Representation Datasets using InMemoryDataset

This module provides dataset classes that cache PyTorch Geometric
tensors for instant loading.
"""

import os
import sys
import torch
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Iterator, Callable
from itertools import islice
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import warnings

# Add universal representation imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_loading.data_types import UniversalMolecule

# Adapters pickle objects with __module__='data_types'; make unpickling work from project root
if 'data_types' not in sys.modules:
    import data_loading.data_types as _data_types_module
    sys.modules['data_types'] = _data_types_module

from torch_cluster import radius_graph
from data_loading.atom_position_vocab import (
    RARE_ATOMIC_NUMBER,
    VIRTUAL_ATOMIC_NUMBER,
    is_virtual_position_code,
)
warnings.filterwarnings('ignore')

# Periodic table whitelist: all other atoms collapse to R (rare atom).
ELEMENT_TO_ATOMIC_NUMBER = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'Na': 11,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Br': 35,
    'I': 53,
}
DEFAULT_ATOMIC_NUMBER = RARE_ATOMIC_NUMBER


def element_to_atomic_number(element: str) -> int:
    """Map allowed atoms to their atomic numbers and collapse everything else to R."""
    s = str(element).strip()
    element_normalized = s.capitalize() if len(s) <= 2 else s
    return ELEMENT_TO_ATOMIC_NUMBER.get(element_normalized, RARE_ATOMIC_NUMBER)


def atom_to_model_id(atom) -> int:
    """Map an atom to the model id without confusing vanadium with virtual V."""
    pos_code = str(getattr(atom, 'pos_code', ''))
    if bool(getattr(atom, 'is_virtual', False)) or is_virtual_position_code(pos_code):
        return VIRTUAL_ATOMIC_NUMBER
    return element_to_atomic_number(str(getattr(atom, 'element', 'C')))


def load_molecules_iteratively(file_path: str) -> Iterator[UniversalMolecule]:
    """Generator to load molecules one by one from a pickle stream."""
    with open(file_path, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


# --- Parallel per-molecule conversion (used by process()) ---------------------
# Global per-worker converter, initialized once per process to avoid repeatedly
# rebuilding the config-only helper instance.
_WORKER_CONVERTER = None


def _init_conversion_worker(config: dict) -> None:
    """ProcessPool initializer: build a lightweight, config-only converter.

    We bypass InMemoryDataset.__init__ (which would load/process data) via
    object.__new__ and just set the attributes the conversion methods read.
    The conversion logic is identical across all OptimizedUniversal* subclasses,
    so the base class is sufficient.
    """
    global _WORKER_CONVERTER
    # Each worker process is single-threaded for torch ops; parallelism comes
    # from the process pool. Without this, torch_cluster's radius_graph and BLAS
    # spawn one thread pool per worker, massively oversubscribing the CPU
    # (context-switch thrash) and running SLOWER than the serial baseline.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    obj = object.__new__(OptimizedUniversalDataset)
    for key, val in config.items():
        setattr(obj, key, val)
    # pre_filter / pre_transform are not picklable in general; conversion path
    # in the chunk pipeline does not use them.
    obj.pre_filter = None
    obj.pre_transform = None
    _WORKER_CONVERTER = obj


def _convert_one_molecule(mol):
    """ProcessPool worker: convert a single molecule, returning (data, reason).

    Returns (Data, None) on success, (None, reason_str) on skip/failure so the
    parent can keep the original skip-accounting and first-fail diagnostics.
    """
    conv = _WORKER_CONVERTER
    try:
        pyg_data = conv._create_pyg_data_object(mol)
        if pyg_data is not None:
            return pyg_data, None
        return None, conv._diagnose_skip(mol)
    except Exception as e:  # noqa: BLE001 - mirror original broad catch
        return None, f"Exception: {e}"


class OptimizedUniversalDataset(InMemoryDataset):
    """
    Universal Dataset using InMemoryDataset for tensor caching.
    It processes data in chunks to handle large datasets that don't fit in RAM.
    """

    def __init__(
        self,
        root: str,
        universal_cache_path: str,
        max_samples: Optional[int] = None,
        molecule_max_atoms: Optional[int] = None,
        cutoff_distance: float = 5.0,
        max_neighbors: int = 32,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        conversion_num_workers: Optional[int] = None,
    ):
        """
        Initialize Optimized Universal Dataset

        Args:
            root: Root directory for processed tensor cache
            universal_cache_path: Path to cached universal representations (.pkl file)
            max_samples: Maximum number of samples to load (None for all)
            molecule_max_atoms: Maximum number of atoms per molecule (None for no limit)
            cutoff_distance: Distance cutoff for edge construction
            max_neighbors: Maximum number of neighbors per atom
            transform: Transform to apply to each sample
            pre_transform: Pre-transform to apply during processing
            pre_filter: Pre-filter to apply during processing
            conversion_num_workers: Optional explicit worker cap for converting
                molecules inside one cache file
        """
        self.universal_cache_path = universal_cache_path
        self.max_samples = max_samples
        self.molecule_max_atoms = molecule_max_atoms
        # Coerce to safe defaults so int()/float() never see None (config may pass null)
        self.cutoff_distance = cutoff_distance if cutoff_distance is not None else 5.0
        self.max_neighbors = max_neighbors if max_neighbors is not None else 32
        if conversion_num_workers is not None and conversion_num_workers < 1:
            raise ValueError("conversion_num_workers must be at least 1")
        self.conversion_num_workers = conversion_num_workers

        os.makedirs(root, exist_ok=True)
        super().__init__(root, transform, pre_transform, pre_filter)
        # Trusted file written by this class (PyG objects need weights_only=False on torch >= 2.6).
        self.data, self.slices = torch.load(
            self.processed_paths[0], map_location='cpu', weights_only=False
        )

    @property
    def raw_file_names(self) -> List[str]:
        return [os.path.basename(self.universal_cache_path)]

    @property
    def processed_file_names(self) -> List[str]:
        """Generates a unique filename for the cache based on processing parameters."""
        cache_name = os.path.basename(self.universal_cache_path).replace('.pkl', '')
        if self.max_samples:
            cache_name += f"_samples_{self.max_samples}"
        if self.molecule_max_atoms:
            cache_name += f"_maxatoms_{self.molecule_max_atoms}"

        config_sig = f"cutoff_{self.cutoff_distance}_neighbors_{self.max_neighbors}"
        return [f"optimized_{cache_name}_{config_sig}.pt"]

    def download(self) -> None:
        """Copies the universal .pkl cache to the raw_dir for PyG to find."""
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not os.path.exists(raw_path):
            print(f"📋 Copying universal cache to raw directory...")
            shutil.copy2(self.universal_cache_path, raw_path)

    def process(self) -> None:
        """
        Process molecules from .pkl cache and convert to PyG format.
        
        MEMORY NOTE: This processes all molecules in one pass.
        For large datasets (>50K molecules), use external chunking:
        - Split dataset at manifest level (e.g., 1M → 50 chunks of 20K)
        - Process each chunk separately using this method
        - Use LazyUniversalDataset for training (loads chunks on-demand)
        
        Recommended chunk sizes for processing:
        - 10K molecules: ~200MB RAM
        - 20K molecules: ~400MB RAM  
        - 50K molecules: ~1GB RAM
        """
        print(f"🔄 Processing universal representations...")

        # 1. Load molecules from .pkl file
        molecule_iterator = load_molecules_iteratively(self.raw_paths[0])

        # 2. Apply max_samples limit if specified
        if self.max_samples is not None:
            molecule_iterator = islice(molecule_iterator, self.max_samples)

        # 3. Convert all molecules to PyG Data objects (parallelized).
        #    Order is preserved (executor.map / ordered collection), so the
        #    collated dataset is deterministic w.r.t. the input molecule order.
        molecules = list(molecule_iterator)

        data_list: List[Data] = []
        skipped_count = 0
        first_fail_reason: Optional[str] = None

        cpu = os.cpu_count() or 2
        if self.conversion_num_workers is None:
            num_workers = max(1, min(16, cpu - 2))
        else:
            num_workers = self.conversion_num_workers
        # Don't spin up more workers than molecules.
        num_workers = max(1, min(num_workers, len(molecules)))

        worker_config = {
            "universal_cache_path": self.universal_cache_path,
            "max_samples": self.max_samples,
            "molecule_max_atoms": self.molecule_max_atoms,
            "cutoff_distance": self.cutoff_distance,
            "max_neighbors": self.max_neighbors,
        }

        if num_workers == 1:
            # Fall back to in-process conversion (e.g. tiny inputs).
            _init_conversion_worker(worker_config)
            results = (
                _convert_one_molecule(mol)
                for mol in tqdm(molecules, desc="Converting molecules")
            )
            for pyg_data, reason in results:
                if pyg_data is not None:
                    data_list.append(pyg_data)
                else:
                    if first_fail_reason is None:
                        first_fail_reason = reason
                    skipped_count += 1
        else:
            print(f"🚀 Converting with {num_workers} parallel workers...")
            # chunksize amortizes IPC; map preserves input order.
            # An explicit worker cap is normally used for very large caches.
            # Keep each IPC payload to one molecule in that mode; batching tens
            # of ribosome-sized RNA objects can otherwise exhaust host memory.
            chunksize = (
                1
                if self.conversion_num_workers is not None
                else max(1, len(molecules) // (num_workers * 8))
            )
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_conversion_worker,
                initargs=(worker_config,),
            ) as executor:
                for pyg_data, reason in tqdm(
                    executor.map(_convert_one_molecule, molecules, chunksize=chunksize),
                    total=len(molecules),
                    desc="Converting molecules",
                ):
                    if pyg_data is not None:
                        data_list.append(pyg_data)
                    else:
                        if first_fail_reason is None:
                            first_fail_reason = reason
                        skipped_count += 1

        if not data_list:
            msg = (
                f"No molecules were processed successfully. Skipped {skipped_count} molecules."
            )
            if first_fail_reason:
                msg += f" First skip reason: {first_fail_reason}"
            raise RuntimeError(msg)

        # Memory usage info
        ram_mb = len(data_list) * 20 / 1024
        print(f"✅ Converted {len(data_list):,} molecules to PyG format")
        if skipped_count > 0:
            print(f"⚠️  Skipped {skipped_count} molecules")
        print(f"💾 Estimated RAM usage: ~{ram_mb:.0f}MB")
        if ram_mb > 2000:
            print(f"⚠️  Large dataset! Consider using smaller chunks for 1M+ proteins.")

        # 4. Collate and save the final dataset
        print(f"🔄 Collating dataset...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"✅ Processing complete! Saved to: {self.processed_paths[0]}")

    def _diagnose_skip(self, mol: UniversalMolecule) -> str:
        """Return a short reason why this molecule would be skipped (for error messages)."""
        try:
            atoms = mol.get_all_atoms()
            if not atoms:
                return "get_all_atoms() returned empty (molecule has no blocks or blocks have no atoms)"
            if self.molecule_max_atoms is not None and len(atoms) > self.molecule_max_atoms:
                return f"molecule has {len(atoms)} atoms > molecule_max_atoms={self.molecule_max_atoms}"
            # Try actual conversion with raise_on_error to capture the real error
            try:
                out = self._universal_to_pyg(mol, raise_on_error=True)
                if out is None:
                    return "_universal_to_pyg returned None (empty atoms)"
            except Exception as e:
                return str(e)
            return "unknown"
        except Exception as e:
            return str(e)

    def _create_pyg_data_object(self, mol: UniversalMolecule) -> Optional[Data]:
        """Helper function to filter and convert a single UniversalMolecule."""
        if self.molecule_max_atoms is not None and len(mol.get_all_atoms()) > self.molecule_max_atoms:
            return None

        pyg_data = self._universal_to_pyg(mol)
        if pyg_data is None:
            return None
        if self.pre_filter is not None and not self.pre_filter(pyg_data):
            return None
        if self.pre_transform is not None:
            pyg_data = self.pre_transform(pyg_data)
        return pyg_data

    def _universal_to_pyg(self, mol: UniversalMolecule, raise_on_error: bool = False) -> Optional[Data]:
        """Convert a UniversalMolecule to PyTorch Geometric Data object."""
        try:
            atoms = mol.get_all_atoms()
            if not atoms:
                return None

            # Support position as tuple, list, or numpy (unpickled format may vary)
            def _pos_to_list(p):
                if hasattr(p, '__iter__') and not isinstance(p, str):
                    return list(p)[:3]
                return [0.0, 0.0, 0.0]
            positions = torch.tensor([_pos_to_list(atom.position) for atom in atoms], dtype=torch.float32)
            atomic_numbers = torch.tensor(
                [atom_to_model_id(atom) for atom in atoms],
                dtype=torch.long
            )
            def _safe_int(v, default: int = 0) -> int:
                return int(v) if v is not None else default
            block_indices = torch.tensor([_safe_int(getattr(atom, 'block_idx', None)) for atom in atoms], dtype=torch.long)
            entity_indices = torch.tensor([_safe_int(getattr(atom, 'entity_idx', None)) for atom in atoms], dtype=torch.long)
            pos_codes = [atom.pos_code for atom in atoms]
            block_symbols = [str(getattr(block, 'symbol', '')) for block in mol.blocks]

            num_atoms = len(atoms)
            if num_atoms > 1:
                edge_index = radius_graph(
                    positions, r=float(self.cutoff_distance), batch=None, loop=False,
                    max_num_neighbors=int(self.max_neighbors)
                )
                edge_attr = self._calculate_edge_features(positions, edge_index)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float32)

            bond_edge_index = torch.empty((2, 0), dtype=torch.long)
            bond_edges = mol.properties.get('bond_edges') if mol.properties else None
            if bond_edges:
                edges_bidir = []
                for i, j in bond_edges:
                    edges_bidir.append([i, j])
                    edges_bidir.append([j, i])
                if edges_bidir:
                    bond_edge_index = torch.tensor(edges_bidir, dtype=torch.long).t().contiguous()

            data = Data(
                pos=positions,
                z=atomic_numbers,
                edge_index=edge_index,
                edge_attr=edge_attr,
                bond_edge_index=bond_edge_index,
                block_idx=block_indices,
                entity_idx=entity_indices,
                pos_code=pos_codes,
                block_symbols=block_symbols,
                mol_id=str(getattr(mol, 'id', 'unknown')),
                dataset_type=str(getattr(mol, 'dataset_type', 'unknown')),
                num_nodes=len(atoms),
                num_edges=edge_index.size(1),
            )
            return data
        except (ValueError, IndexError, TypeError) as e:
            if raise_on_error:
                raise
            warnings.warn(f"Error converting molecule {getattr(mol, 'id', '?')}: {e}", UserWarning)
            return None
        except Exception as e:
            if raise_on_error:
                raise
            warnings.warn(f"Unexpected error converting molecule {getattr(mol, 'id', '?')}: {e}", UserWarning)
            return None

    def _element_to_atomic_number(self, element: str) -> int:
        """Convert element symbol to atomic number using dictionary lookup."""
        return element_to_atomic_number(element)

    def _calculate_edge_features(self, positions: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Calculate edge features (distances) between connected atoms"""
        row, col = edge_index
        diff = positions[row] - positions[col]
        distances = torch.norm(diff, dim=1, keepdim=True)
        return distances


class OptimizedUniversalQM9Dataset(OptimizedUniversalDataset):
    """QM9 Dataset using cached PyTorch Geometric tensors."""

    def __init__(
        self,
        root: Optional[str] = None,
        universal_cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        molecule_max_atoms: Optional[int] = None,
        cutoff_distance: float = 5.0,
        max_neighbors: int = 32,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        if root is None:
            root = os.path.join(os.path.dirname(__file__), 'processed', 'qm9_optimized')

        # Default cache path for QM9
        if universal_cache_path is None:
            universal_cache_path = os.path.join(os.path.dirname(__file__), 'cache', 'universal_qm9_all.pkl')
        super().__init__(
            root, universal_cache_path, max_samples, molecule_max_atoms, cutoff_distance, max_neighbors,
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter,
        )


class OptimizedUniversalLBADataset(OptimizedUniversalDataset):
    """LBA Dataset using cached PyTorch Geometric tensors."""

    def __init__(
        self,
        root: Optional[str] = None,
        universal_cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        molecule_max_atoms: Optional[int] = None,
        cutoff_distance: float = 5.0,
        max_neighbors: int = 32,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        if root is None:
            root = os.path.join(os.path.dirname(__file__), 'processed', 'lba_optimized')
        if universal_cache_path is None:
            universal_cache_path = os.path.join(os.path.dirname(__file__), 'cache', 'universal_lba_all.pkl')
        super().__init__(
            root, universal_cache_path, max_samples, molecule_max_atoms, cutoff_distance, max_neighbors,
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter,
        )


class OptimizedUniversalCOCONUTDataset(OptimizedUniversalDataset):
    """COCONUT Dataset using cached PyTorch Geometric tensors."""
    
    def __init__(
        self,
        root: Optional[str] = None,
        universal_cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        cutoff_distance: float = 5.0,
        max_neighbors: int = 32,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        if root is None:
            root = os.path.join(os.path.dirname(__file__), 'processed', 'coconut_optimized')
        if universal_cache_path is None:
            universal_cache_path = os.path.join(os.path.dirname(__file__), 'cache', 'universal_coconut_all.pkl')
        super().__init__(
            root, universal_cache_path, max_samples, None, cutoff_distance, max_neighbors,
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter,
        )

class OptimizedUniversalRNADataset(OptimizedUniversalDataset):
    """
    RNA Dataset using cached PyTorch Geometric tensors.
    
    RNA-specific optimizations:
    - Larger cutoff distance (8.0Å) for base pairing interactions
    - Fewer max neighbors (32) as RNA has sparser structure than proteins
    """

    def __init__(
        self,
        root: Optional[str] = None,
        universal_cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        molecule_max_atoms: Optional[int] = None,
        cutoff_distance: float = 8.0,  # Changed from 6.0 to 8.0 for base pairing
        max_neighbors: int = 32,       # Changed from 20 to 32 for consistency
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        conversion_num_workers: Optional[int] = None,
    ):
        if root is None:
            root = os.path.join(os.path.dirname(__file__), 'processed', 'rna_optimized')
        if universal_cache_path is None:
            universal_cache_path = os.path.join(os.path.dirname(__file__), 'cache', 'universal_rna_all.pkl')
        super().__init__(
            root,
            universal_cache_path,
            max_samples,
            molecule_max_atoms,
            cutoff_distance,
            max_neighbors,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            conversion_num_workers=conversion_num_workers,
        )


class OptimizedUniversalUniMolDataset(OptimizedUniversalDataset):
    """UniMol Dataset using cached PyTorch Geometric tensors."""

    def __init__(
        self,
        root: Optional[str] = None,
        universal_cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        molecule_max_atoms: Optional[int] = None,
        cutoff_distance: float = 5.0,
        max_neighbors: int = 32,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        if root is None:
            root = os.path.join(os.path.dirname(__file__), 'processed', 'unimol_optimized')
        if universal_cache_path is None:
            universal_cache_path = os.path.join(os.path.dirname(__file__), 'cache', 'universal_unimol_all.pkl')
        super().__init__(
            root, universal_cache_path, max_samples, molecule_max_atoms, cutoff_distance, max_neighbors,
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter,
        )


# One registry for every conversion route (single file and directory mode).
DATASET_CLASSES = {
    "qm9": OptimizedUniversalQM9Dataset,
    "lba": OptimizedUniversalLBADataset,
    "pdb": OptimizedUniversalDataset,
    "coconut": OptimizedUniversalCOCONUTDataset,
    "rna": OptimizedUniversalRNADataset,
    "unimol": OptimizedUniversalUniMolDataset,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert universal .pkl cache to PyG .pt format")
    parser.add_argument("--input-pkl", required=True, help="Path to input .pkl file")
    parser.add_argument("--output-dir", required=True, help="Output directory for .pt file")
    parser.add_argument("--dataset-type", choices=sorted(DATASET_CLASSES), required=True, help="Dataset type")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Distance cutoff for edges (Å)")
    parser.add_argument("--max-neighbors", type=int, default=64, help="Maximum neighbors per atom")
    parser.add_argument("--force", action="store_true", help="Force rebuild if output exists")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers (also caps workers inside one RNA cache)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check if output already exists (look for any .pt file in processed subdir)
    processed_dir = os.path.join(args.output_dir, "processed")
    if os.path.exists(processed_dir) and not args.force:
        pt_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
        if pt_files:
            print(f"✅ Output already exists: {os.path.join(processed_dir, pt_files[0])}")
            print(f"💡 Use --force to rebuild")
            sys.exit(0)

    dataset_class = DATASET_CLASSES[args.dataset_type]
    dataset_kwargs = dict(
        root=args.output_dir,
        universal_cache_path=args.input_pkl,
        max_samples=args.max_samples,
        cutoff_distance=args.cutoff,
        max_neighbors=args.max_neighbors,
    )
    if args.dataset_type == "rna":
        dataset_kwargs["conversion_num_workers"] = args.num_workers
    dataset = dataset_class(**dataset_kwargs)
    print(f"✅ Created {args.dataset_type.upper()} dataset: {len(dataset)} samples")

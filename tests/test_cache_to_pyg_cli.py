"""The UniMol/COCONUT cache -> PyG conversion CLI (second stage of the chunk pipeline)."""

import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from data_loading.data_types import UniversalAtom, UniversalBlock, UniversalMolecule

REPO_ROOT = Path(__file__).resolve().parents[1]


def _molecule(mol_id: str, dataset_type: str, num_atoms: int) -> UniversalMolecule:
    atoms = [
        UniversalAtom(
            element="C" if index % 2 == 0 else "O",
            position=(1.5 * index, 0.1 * index, 0.0),
            pos_code="sm",
            block_idx=0,
            atom_idx_in_block=index,
            entity_idx=0,
        )
        for index in range(num_atoms)
    ]
    return UniversalMolecule(
        id=mol_id,
        dataset_type=dataset_type,
        blocks=[UniversalBlock(symbol="FRAGMENT", atoms=atoms)],
        properties={"bond_edges": [(i, i + 1) for i in range(num_atoms - 1)]},
    )


@pytest.mark.parametrize("dataset_type", ["unimol", "coconut"])
def test_single_file_conversion_cli_writes_loadable_chunk(tmp_path, dataset_type):
    """Both live routes convert, exit 0 and write a chunk that PyG can read back."""
    source = tmp_path / f"universal_{dataset_type}_chunk_0.pkl"
    with open(source, "wb") as handle:
        for index, num_atoms in enumerate((4, 6, 5)):
            pickle.dump(_molecule(f"{dataset_type}-{index}", dataset_type, num_atoms), handle)
    output_dir = tmp_path / "chunk_0"

    completed = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "data_loading" / "cache_to_pyg.py"),
            "--input-pkl", str(source),
            "--output-dir", str(output_dir),
            "--dataset-type", dataset_type,
            "--cutoff", "5.0",
            "--max-neighbors", "16",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert completed.returncode == 0, completed.stdout[-800:] + completed.stderr[-800:]

    (chunk_file,) = (output_dir / "processed").glob(f"optimized_universal_{dataset_type}_chunk_0_*.pt")
    data, slices = torch.load(chunk_file, map_location="cpu", weights_only=False)
    assert slices["z"].tolist() == [0, 4, 10, 15]
    assert data.bond_edge_index.shape[0] == 2
    assert data.edge_index.shape[0] == 2

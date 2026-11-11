"""Synthetic chunk writer shared by the fail-loud chunk-metadata tests."""

import json

import torch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate


def write_chunk(root, name, n=4, with_metadata=True):
    processed = root / f"{name}_chunk_0" / "processed"
    processed.mkdir(parents=True)
    samples = [
        Data(z=torch.full((5,), 6, dtype=torch.long), pos=torch.randn(5, 3),
             edge_index=torch.empty((2, 0), dtype=torch.long),
             mol_id=f"{name}_{i}", dataset_type="unimol")
        for i in range(n)
    ]
    data, slices, _ = collate(Data, data_list=samples, increment=False, add_batch=False)
    pt_file = processed / f"{name}.pt"
    torch.save((data, slices), pt_file)
    if with_metadata:
        metadata = {
            "chunk_file": pt_file.name, "num_samples": n, "dataset_type": "unimol",
            "total_atoms": 5 * n, "total_edges": 0,
            "samples": [{"idx": i, "num_atoms": 5, "num_edges": 0, "bucket": 0,
                         "mol_id": f"{name}_{i}"} for i in range(n)],
        }
        (processed / f"{name}_metadata.json").write_text(json.dumps(metadata))
    return pt_file

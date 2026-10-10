"""Bounded CPU reproductions for the 2026-09-12 documentation audit.

Run from the repository root with the MIND Python environment. No dataset,
checkpoint, training loop, network request, or file output is used.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch

from core.molecular_encoder import UniversalMolecularEncoder
from core.pretraining_config import PretrainingConfig
from data_loading.lazy_universal_dataset import LazyUniversalDataset


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = PretrainingConfig(hidden_dims=[32], atom_types=121,
                               gaussian_kernels=8, use_posenc=False)
    encoder = UniversalMolecularEncoder(config).eval()
    protein = Data(
        z=torch.tensor([7, 6, 6, 8]),
        pos=torch.tensor([[0., 0., 0.], [1.45, 0., 0.],
                          [2.9, 0., 0.], [4.15, 0., 0.]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        block_idx=torch.zeros(4, dtype=torch.long),
        pos_code=['N', 'CA', 'C', 'O'], dataset_type='protein', num_nodes=4,
    )
    # A compact backbone graph, after removal of masked sidechain slots.
    protein = LazyUniversalDataset._normalize_sample_for_collate(protein)
    small = Data(
        z=torch.tensor([6, 8]), pos=torch.tensor([[0., 0., 0.], [1.4, 0., 0.]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        bond_edge_index=torch.tensor([[0, 1], [1, 0]]),
        block_idx=torch.zeros(2, dtype=torch.long), pos_code=['sm', 'sm'],
        dataset_type='unimol', num_nodes=2,
    )
    small = LazyUniversalDataset._normalize_sample_for_collate(small)

    def coordination(batch):
        x_dense, _ = to_dense_batch(batch.z, batch.batch)
        src, dst = batch.edge_index
        distances = (batch.pos[src] - batch.pos[dst]).norm(dim=-1)
        return encoder._compute_chemical_coordination(
            x_dense, batch, edge_distances=distances).tolist()

    single = Batch.from_data_list([protein])
    mixed = Batch.from_data_list([protein, small])
    result = {
        'protein_coordination_alone': coordination(single)[0],
        'protein_coordination_with_unimol': coordination(mixed)[0],
    }
    captured = {}

    def capture_gaussian(module, args, output):
        captured['raw_gaussian_abs_sum'] = float(output.detach().abs().sum())

    def capture_projection(module, args):
        features = args[0].detach()
        captured['masked_backbone_rbf_abs_sum'] = float(features[..., 5:].abs().sum())
        captured['masked_backbone_scalar_abs_sum'] = float(features[..., :5].abs().sum())

    h1 = encoder.gaussian_layer.register_forward_hook(capture_gaussian)
    h2 = encoder.coordinate_projection.register_forward_pre_hook(capture_projection)
    try:
        with torch.no_grad():
            encoder._compute_geometric_features(
                torch.zeros_like(single.z), single.pos, single.batch, single)
    finally:
        h1.remove()
        h2.remove()
    result.update(captured)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset

from data_loading.filter_pyg_chunks_by_num_nodes import filter_chunk
from data_loading.lazy_universal_dataset import LazyUniversalDataset


def _graph(mol_id: str, num_nodes: int) -> Data:
    edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(
        pos=torch.zeros((num_nodes, 3)),
        z=torch.ones(num_nodes, dtype=torch.long),
        edge_index=edge_index,
        edge_attr=torch.empty((0, 1)),
        block_idx=torch.arange(num_nodes),
        entity_idx=torch.zeros(num_nodes, dtype=torch.long),
        pos_code=["P"] * num_nodes,
        block_symbols=["A"] * num_nodes,
        mol_id=mol_id,
        dataset_type="rna",
        num_edges=torch.tensor([0]),
    )


class FilterPygChunksTest(unittest.TestCase):
    def test_filter_keeps_exact_limit_and_drops_one_over(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source" / "processed"
            source_dir.mkdir(parents=True)
            source_pt = source_dir / "source.pt"
            graphs = [_graph("exact", 5), _graph("over", 6)]
            torch.save(InMemoryDataset.collate(graphs), source_pt)
            source_pt.with_name("source_metadata.json").write_text(
                json.dumps(
                    {
                        "chunk_file": source_pt.name,
                        "num_samples": 2,
                        "dataset_type": "rna",
                        "total_atoms": 11,
                        "total_edges": 0,
                        "samples": [
                            {"idx": 0, "num_atoms": 5, "num_edges": 0, "bucket": 0},
                            {"idx": 1, "num_atoms": 6, "num_edges": 0, "bucket": 0},
                        ],
                    }
                )
            )
            output_pt = temp_path / "output" / "processed" / "filtered.pt"

            report = filter_chunk(
                source_pt, output_pt, max_nodes=5, source_chunk_index=0
            )

            self.assertEqual(report["kept_graphs"], 1)
            self.assertEqual(report["dropped_graphs"], 1)
            self.assertEqual(report["kept"][0]["mol_id"], "exact")
            self.assertEqual(report["dropped"][0]["mol_id"], "over")

    def test_loader_guard_reports_oversized_graph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source" / "processed"
            source_dir.mkdir(parents=True)
            source_pt = source_dir / "source.pt"
            torch.save(
                InMemoryDataset.collate([_graph("over", 6), _graph("exact", 5)]),
                source_pt,
            )
            source_pt.with_name("source_metadata.json").write_text(
                json.dumps(
                    {
                        "chunk_file": source_pt.name,
                        "num_samples": 2,
                        "dataset_type": "rna",
                        "total_atoms": 11,
                        "total_edges": 0,
                        "samples": [
                            {
                                "idx": 0,
                                "num_atoms": 6,
                                "num_edges": 0,
                                "bucket": 0,
                                "mol_id": "over",
                            },
                            {
                                "idx": 1,
                                "num_atoms": 5,
                                "num_edges": 0,
                                "bucket": 0,
                                "mol_id": "exact",
                            },
                        ],
                    }
                )
            )

            with self.assertRaisesRegex(
                ValueError, "structure_id=over.*node_count=6"
            ):
                LazyUniversalDataset(
                    [str(source_pt)],
                    verbose=False,
                    load_metadata=True,
                    max_graph_atoms_by_domain={"rna": 5},
                )


if __name__ == "__main__":
    unittest.main()

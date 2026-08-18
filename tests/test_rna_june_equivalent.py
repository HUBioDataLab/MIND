"""Focused tests for June-equivalent RNA selection and PT derivation."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate

from data_loading.build_rna_june_equivalent_manifest import (
    build_manifest,
    real_non_hydrogen_count,
)
from data_loading.derive_rna_june_equivalent_pt import derive_all_chunks


def _write_manifest(path: Path, names: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name"])
        writer.writeheader()
        writer.writerows({"name": name} for name in names)


def _graph(molecule_id: str, atomic_numbers: list[int]) -> Data:
    nodes = len(atomic_numbers)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return Data(
        pos=torch.randn(nodes, 3),
        z=torch.tensor(atomic_numbers, dtype=torch.long),
        edge_index=edge_index,
        edge_attr=torch.ones(2, 1),
        bond_edge_index=torch.empty((2, 0), dtype=torch.long),
        block_idx=torch.zeros(nodes, dtype=torch.long),
        entity_idx=torch.zeros(nodes, dtype=torch.long),
        pos_code=["C1'"] * nodes,
        block_symbols=["A"],
        mol_id=molecule_id,
        dataset_type="rna",
        num_nodes=nodes,
        num_edges=2,
    )


def _write_source_chunk(source_base: Path, chunk_index: int, graphs: list[Data]) -> Path:
    processed = (
        source_base.parent
        / f"{source_base.name}_chunk_{chunk_index:03d}"
        / "processed"
    )
    processed.mkdir(parents=True)
    pt_path = processed / f"source_{chunk_index:03d}.pt"
    batch, slices, _ = collate(
        cls=Data, data_list=graphs, increment=False, add_batch=False
    )
    torch.save((batch, slices), pt_path)
    metadata = {
        "chunk_file": pt_path.name,
        "num_samples": len(graphs),
        "dataset_type": "rna",
        "total_atoms": sum(int(graph.num_nodes) for graph in graphs),
        "total_edges": sum(int(graph.num_edges) for graph in graphs),
        "samples": [
            {
                "idx": index,
                "num_atoms": int(graph.num_nodes),
                "num_edges": int(graph.num_edges),
                "bucket": 0,
            }
            for index, graph in enumerate(graphs)
        ],
    }
    (processed / f"{pt_path.stem}_metadata.json").write_text(
        json.dumps(metadata) + "\n"
    )
    return pt_path


class TestRnaJuneEquivalent(unittest.TestCase):
    def test_manifest_uses_real_atoms_and_audits_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_base = root / "source"
            _write_source_chunk(
                source_base,
                0,
                [_graph("a", [6, 1]), _graph("b", [7, 120])],
            )
            _write_source_chunk(
                source_base,
                1,
                [_graph("c", [8, 15, 120]), _graph("x", [6, 6, 6])],
            )
            raw_manifest = root / "raw.csv"
            current_manifest = root / "current.csv"
            _write_manifest(raw_manifest, ["a", "b", "c", "legacy_only"])
            _write_manifest(current_manifest, ["a", "b", "c", "x"])
            filtered_dir = root / "filtered"
            filtered_dir.mkdir()
            for name in ["a", "b", "c", "legacy_only", "stale"]:
                (filtered_dir / f"{name}.cif").touch()

            output_manifest = root / "june_v2.csv"
            output_report = root / "june_v2.json"
            report = build_manifest(
                source_base=source_base,
                historical_raw_manifest=raw_manifest,
                legacy_filtered_dir=filtered_dir,
                current_manifest=current_manifest,
                output_manifest=output_manifest,
                output_report=output_report,
                first_chunk=0,
                last_chunk=1,
                max_real_non_h_atoms=2,
                historical_june_graphs=3,
                expected_historical_raw=4,
                expected_current=4,
                expected_raw_intersection=3,
                expected_raw_only=1,
                expected_current_only=1,
                expected_stale_cifs=1,
                expected_raw_only_ids=("legacy_only",),
            )

            self.assertEqual(report["counts"]["selected"], 3)
            self.assertEqual(report["counts"]["rejected"], 1)
            self.assertEqual(report["counts"]["selected_minus_historical_june"], 0)
            with output_manifest.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["name"] for row in rows], ["a", "b", "c"])
            self.assertEqual(rows[0]["real_non_hydrogen_atom_count"], "1")
            self.assertEqual(rows[1]["num_nodes"], "2")
            self.assertEqual(real_non_hydrogen_count(torch.tensor([1, 6, 120, 8])), 2)

            with self.assertRaises(FileExistsError):
                build_manifest(
                    source_base=source_base,
                    historical_raw_manifest=raw_manifest,
                    legacy_filtered_dir=filtered_dir,
                    current_manifest=current_manifest,
                    output_manifest=output_manifest,
                    output_report=output_report,
                    first_chunk=0,
                    last_chunk=1,
                    expected_historical_raw=4,
                )

    def test_pt_derivation_is_manifest_based_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_base = root / "source"
            source_paths = [
                _write_source_chunk(source_base, 0, [_graph("a", [6, 6]), _graph("b", [6, 6])]),
                _write_source_chunk(source_base, 1, [_graph("c", [6, 6, 6]), _graph("d", [6, 6])]),
            ]
            source_fingerprints = [
                (path.stat().st_size, path.stat().st_mtime_ns) for path in source_paths
            ]

            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "name",
                        "real_non_hydrogen_atom_count",
                        "num_nodes",
                        "source_chunk",
                        "source_idx",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"name": "a", "real_non_hydrogen_atom_count": 2, "num_nodes": 2, "source_chunk": 0, "source_idx": 0},
                        {"name": "c", "real_non_hydrogen_atom_count": 3, "num_nodes": 3, "source_chunk": 1, "source_idx": 0},
                        {"name": "d", "real_non_hydrogen_atom_count": 2, "num_nodes": 2, "source_chunk": 1, "source_idx": 1},
                    ]
                )

            output_root = root / "derived_v2"
            report = derive_all_chunks(
                source_base=source_base,
                output_root=output_root,
                manifest_path=manifest,
                first_chunk=0,
                last_chunk=1,
                expected_selected_count=3,
            )
            self.assertEqual(report["aggregate"]["source_graphs"], 4)
            self.assertEqual(report["aggregate"]["kept_graphs"], 3)
            self.assertEqual(report["aggregate"]["unique_kept_ids"], 3)
            self.assertTrue((output_root / "derivation_report.json").is_file())
            self.assertEqual(
                source_fingerprints,
                [(path.stat().st_size, path.stat().st_mtime_ns) for path in source_paths],
            )

            with self.assertRaises(FileExistsError):
                derive_all_chunks(
                    source_base=source_base,
                    output_root=output_root,
                    manifest_path=manifest,
                    first_chunk=0,
                    last_chunk=1,
                    expected_selected_count=3,
                )


if __name__ == "__main__":
    unittest.main()

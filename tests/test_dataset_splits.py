"""Split assignment and graph-size policy for chunked multi-domain datasets."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.data.collate import collate

from core.train_pretrain import load_universal_dataset, split_dataset
from data_loading.lazy_universal_dataset import LazyUniversalDataset


def _write_chunk(root, prefix, index, sizes, dataset_type, split=None):
    processed = root / f"{prefix}_chunk_{index:03d}" / "processed"
    processed.mkdir(parents=True)
    samples = [
        Data(z=torch.full((n,), 6, dtype=torch.long), pos=torch.randn(n, 3),
             edge_index=torch.empty((2, 0), dtype=torch.long),
             mol_id=f"{prefix}{index}_{i}", dataset_type=dataset_type)
        for i, n in enumerate(sizes)
    ]
    data, slices, _ = collate(Data, data_list=samples, increment=False, add_batch=False)
    pt_file = processed / f"{prefix}_{index}.pt"
    torch.save((data, slices), pt_file)
    metadata = {
        "chunk_file": pt_file.name, "num_samples": len(sizes), "dataset_type": dataset_type,
        "total_atoms": sum(sizes), "total_edges": 0,
        "samples": [{"idx": i, "num_atoms": n, "num_edges": 0, "bucket": 0,
                     "mol_id": f"{prefix}{index}_{i}"} for i, n in enumerate(sizes)],
    }
    if split is not None:
        metadata["split"] = split
    (processed / f"{prefix}_{index}_metadata.json").write_text(json.dumps(metadata))
    return str(pt_file)


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setenv("MIND_INDEX_MAP_CACHE_DIR", str(tmp_path / "maps"))
    return [
        _write_chunk(tmp_path, "a", 0, [5] * 40, "unimol"),
        _write_chunk(tmp_path, "a", 1, [5] * 30, "unimol"),
        _write_chunk(tmp_path, "b", 0, [5] * 20, "unimol", split="val"),
    ]


def _dataset(files, **kwargs):
    return LazyUniversalDataset(chunk_pt_files=files, verbose=False, load_metadata=True, **kwargs)


def test_untagged_split_is_exactly_legacy_random_split(files):
    dataset = _dataset(files[:2])
    ours = split_dataset(dataset, seed=42)
    sizes = [int(0.8 * 70), int(0.1 * 70), 70 - int(0.8 * 70) - int(0.1 * 70)]
    legacy = random_split(dataset, sizes, generator=torch.Generator().manual_seed(42))
    assert [list(a.indices) for a in ours] == [list(b.indices) for b in legacy]


def test_exclusions_keep_legacy_membership(files):
    dataset = _dataset(files[:2])
    dataset.excluded_indices = {3, 17, 60}
    ours = split_dataset(dataset, seed=42)
    legacy = split_dataset(_dataset(files[:2]), seed=42)
    for a, b in zip(ours, legacy):
        assert list(a.indices) == sorted(set(b.indices) - {3, 17, 60})


def test_tagged_chunks_go_to_their_split_and_pool_is_random(files):
    dataset = _dataset(files, chunk_split_tags=[None, "train", "val"])
    train, val, test = split_dataset(dataset, seed=1)
    chunk = {name: set(range(start, end)) for name, (start, end, _) in zip("xyz", dataset.file_ranges)}
    assert chunk["y"] <= set(train.indices) and chunk["z"] <= set(val.indices)
    pooled = [set(s.indices) & chunk["x"] for s in (train, val, test)]
    assert [len(p) for p in pooled] == [32, 4, 4]
    everything = set(train.indices) | set(val.indices) | set(test.indices)
    assert everything == set(range(len(dataset)))
    assert not set(train.indices) & set(val.indices)


def test_metadata_split_must_match_config(files):
    with pytest.raises(ValueError, match="declares split 'val'"):
        _dataset(files, chunk_split_tags=[None, None, "train"])
    with pytest.raises(ValueError, match="declares split 'val'"):
        _dataset(files)


def test_oversized_skip_policy_records_and_excludes(tmp_path, monkeypatch):
    monkeypatch.setenv("MIND_INDEX_MAP_CACHE_DIR", str(tmp_path / "maps"))
    pt = _write_chunk(tmp_path, "p", 0, [10, 50, 10, 70], "protein")
    with pytest.raises(ValueError, match="Oversized graph"):
        _dataset([pt], max_graph_atoms_by_domain={"protein": 40})
    dataset = _dataset([pt], max_graph_atoms_by_domain={"protein": 40},
                       oversized_graph_policy_by_domain={"protein": "skip"})
    assert dataset.excluded_indices == {1, 3}
    assert [r["mol_id"] for r in dataset.excluded_records] == ["p0_1", "p0_3"]
    subsets = split_dataset(dataset, seed=0)
    assert not ({1, 3} & set().union(*(set(s.indices) for s in subsets)))


def test_size_limit_for_unloaded_domain_is_refused(files):
    with pytest.raises(ValueError, match=r"configured for \['pdb'\]"):
        _dataset(files[:1], max_graph_atoms_by_domain={"pdb": 4})


def _config(datasets):
    return SimpleNamespace(
        multi_domain={"enabled": True, "datasets": datasets, "max_cache_chunks": 2},
        pretraining_tasks=[], use_improved_sampler=True,
    )


def test_config_splits_and_pdb_limit_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("MIND_INDEX_MAP_CACHE_DIR", str(tmp_path / "maps"))
    for split, count in (("train", 2), ("val", 1)):
        for index in range(count):
            _write_chunk(tmp_path, f"prot_{split}", index, [10, 30], "protein", split=split)
    _write_chunk(tmp_path, "mol", 0, [5] * 10, "unimol")
    config = _config({
        "pdb": {
            "enabled": True, "max_graph_atoms": 20, "oversized_graphs": "skip",
            "splits": {
                "train": {"chunk_dir": str(tmp_path / "prot_train"), "chunk_range": None},
                "val": {"chunk_dir": str(tmp_path / "prot_val"), "chunk_range": [0, 0]},
            },
        },
        "unimol": {"enabled": True, "chunk_dir": str(tmp_path / "mol"), "chunk_range": None},
    })
    dataset = load_universal_dataset(config, "PDB", "")
    assert dataset.chunk_split_tags == ["train", "train", "val", None]
    # 'pdb' config key maps to metadata 'protein': the 30-atom graphs are skipped.
    assert len(dataset.excluded_indices) == 3
    train, val, test = split_dataset(dataset, seed=0)
    protein_train = [i for i in train.indices if dataset.sample_metadata_index[i]["dataset_type"] == "protein"]
    assert len(protein_train) == 2 and len(val.indices) >= 1


def test_config_rejects_mixed_split_layouts(tmp_path):
    config = _config({"pdb": {"enabled": True, "chunk_dir": "x", "splits": {"train": {"chunk_dir": "y"}}}})
    with pytest.raises(ValueError, match="either 'splits'"):
        load_universal_dataset(config, "PDB", "")
    config = _config({"pdb": {"enabled": True, "splits": {"val": {"chunk_dir": "y"}}}})
    with pytest.raises(ValueError, match="needs a train entry"):
        load_universal_dataset(config, "PDB", "")

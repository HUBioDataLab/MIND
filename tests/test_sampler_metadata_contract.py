"""The batch sampler reads atom counts and dataset types from metadata only; both are required."""

import json

import pytest

from data_loading.improved_dynamic_sampler import ImprovedDynamicBatchSampler
from data_loading.lazy_universal_dataset import LazyUniversalDataset
from tests.chunk_fixtures import write_chunk


def _dataset(tmp_path, monkeypatch, drop_chunk_key=None, drop_sample_key=None):
    monkeypatch.setenv("MIND_INDEX_MAP_CACHE_DIR", str(tmp_path / "maps"))
    pt_file = write_chunk(tmp_path, "s", n=4)
    meta_path = pt_file.parent / "s_metadata.json"
    metadata = json.loads(meta_path.read_text())
    if drop_chunk_key:
        del metadata[drop_chunk_key]
    if drop_sample_key:
        for sample in metadata["samples"]:
            del sample[drop_sample_key]
    meta_path.write_text(json.dumps(metadata))
    return LazyUniversalDataset(chunk_pt_files=[str(pt_file)], verbose=False, load_metadata=True)


@pytest.mark.parametrize("kwargs, key", [({"drop_chunk_key": "dataset_type"}, "dataset_type"),
                                          ({"drop_sample_key": "num_atoms"}, "num_atoms")])
def test_metadata_without_required_keys_fails_at_load(tmp_path, monkeypatch, kwargs, key):
    with pytest.raises(KeyError, match=key):
        _dataset(tmp_path, monkeypatch, **kwargs)


def test_sampler_reads_counts_and_types_straight_from_metadata(tmp_path, monkeypatch):
    sampler = ImprovedDynamicBatchSampler(_dataset(tmp_path, monkeypatch), max_atoms_per_batch=10)
    assert sampler._get_atom_count_fast(2) == 5
    assert sampler._get_dataset_type(0) == "unimol"
    assert [len(batch) for batch in sampler] == [2, 2]

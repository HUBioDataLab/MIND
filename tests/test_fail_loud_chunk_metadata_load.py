"""Missing or unreadable chunk metadata must raise when metadata is requested (DS-43)."""

import pytest

from chunk_fixtures import write_chunk
from data_loading.lazy_universal_dataset import LazyUniversalDataset


@pytest.fixture
def index_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MIND_INDEX_MAP_CACHE_DIR", str(tmp_path / "maps"))


def test_requested_metadata_that_is_missing_raises(tmp_path, index_cache):
    good = write_chunk(tmp_path, "good")
    bad = write_chunk(tmp_path, "nometa", with_metadata=False)
    with pytest.raises(FileNotFoundError, match="nometa.pt"):
        LazyUniversalDataset(chunk_pt_files=[str(good), str(bad)], verbose=False, load_metadata=True)


def test_requested_metadata_that_is_corrupt_raises(tmp_path, index_cache):
    good = write_chunk(tmp_path, "good")
    (good.parent / "good_metadata.json").write_text("{not json")
    with pytest.raises(ValueError, match="good_metadata.json"):
        LazyUniversalDataset(chunk_pt_files=[str(good)], verbose=False, load_metadata=True)


def test_metadata_is_still_optional_when_not_requested(tmp_path, index_cache):
    bad = write_chunk(tmp_path, "nometa", with_metadata=False)
    dataset = LazyUniversalDataset(chunk_pt_files=[str(bad)], verbose=False, load_metadata=False)
    assert len(dataset) == 4 and not dataset.metadata_loaded

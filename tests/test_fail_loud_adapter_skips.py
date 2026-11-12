"""Adapters may skip a bad item only with a count, a reason, a report and a failed run."""

import json
import pickle
from pathlib import Path

import lmdb
import pytest

from data_loading.adapters.base_adapter import BaseAdapter
from data_loading.adapters.protein_adapter import ProteinAdapter
from data_loading.adapters.rna_adapter import RNAAdapter
from data_loading.adapters.unimol_adapter import UniMolAdapter
from data_loading.data_types import UniversalAtom, UniversalBlock, UniversalMolecule


class _ToyAdapter(BaseAdapter):
    def __init__(self):
        super().__init__("toy")

    def load_raw_data(self, data_path, max_samples=None, **kwargs):
        return ["ok1", "explodes", "empty", "ok2"]

    def create_blocks(self, raw_item):
        if raw_item == "empty":
            return []  # what the protein/RNA adapters return for an unreadable file
        atom = UniversalAtom(element="C", position=(0.0, 0.0, 0.0), pos_code="sm",
                             block_idx=0, atom_idx_in_block=0, entity_idx=0)
        return [UniversalBlock(symbol="TOY", atoms=[atom])]

    def convert_to_universal(self, raw_item):
        if raw_item == "explodes":
            raise ValueError("bad coordinates")
        return UniversalMolecule(id=raw_item, dataset_type="toy", blocks=self.create_blocks(raw_item))


def test_base_adapter_skips_are_counted_with_reason_and_fail_the_run(tmp_path):
    cache = tmp_path / "toy.pkl"
    with pytest.raises(RuntimeError, match=r"skipped 2 of 4") as caught:
        _ToyAdapter().process_dataset("unused", cache_path=str(cache))
    assert "bad coordinates" in str(caught.value) and "no blocks" in str(caught.value)
    report = json.loads(Path(str(cache) + ".skipped.json").read_text())
    assert report["skipped"] == 2 and report["seen"] == 4
    assert {item["item"] for item in report["items"]} == {"explodes", "empty"}
    # The two good molecules were still written before the failure was raised.
    with open(cache, "rb") as handle:
        assert [pickle.load(handle).id, pickle.load(handle).id] == ["ok1", "ok2"]


def test_base_adapter_without_skips_succeeds_and_writes_no_report(tmp_path):
    class _Clean(_ToyAdapter):
        def load_raw_data(self, data_path, max_samples=None, **kwargs):
            return ["ok1", "ok2"]

    cache = tmp_path / "clean.pkl"
    assert _Clean().process_dataset("unused", cache_path=str(cache)) == 2
    assert not Path(str(cache) + ".skipped.json").exists()


def _lmdb_with(tmp_path, records):
    path = tmp_path / "train.lmdb"
    env = lmdb.open(str(path), subdir=False, map_size=1 << 24)
    with env.begin(write=True) as txn:
        for i, value in enumerate(records):
            txn.put(f"{i}".encode(), value)
    env.close()
    return str(path)


GOOD_RECORD = pickle.dumps({"atoms": ["C", "O"], "coordinates": [[0, 0, 0], [1, 0, 0]], "smi": "CO"})


@pytest.mark.parametrize("value, reason", [
    (b"\x80garbage", "not a pickle"),
    (pickle.dumps(["a list"]), "not a dict"),
    (pickle.dumps({"smi": "C"}), "missing atoms or coordinates"),
    (pickle.dumps({"atoms": ["C"], "coordinates": [[0, 0, 0], [1, 1, 1]]}), "atom and coordinate counts differ"),
    (pickle.dumps({"atoms": ["C"], "coordinates": [["x", 0, 0]]}), "unreadable coordinates"),
], ids=lambda x: x if isinstance(x, str) else "record")
def test_unimol_bad_record_is_reported_with_its_reason(value, reason):
    with pytest.raises(ValueError, match=reason):
        UniMolAdapter()._load_lmdb_item(value)


def test_unimol_process_dataset_counts_skips_by_reason_and_fails(tmp_path):
    path = _lmdb_with(tmp_path, [GOOD_RECORD, b"\x80garbage", pickle.dumps({"smi": "C"}), GOOD_RECORD])
    cache = tmp_path / "out" / "unimol.pkl"
    with pytest.raises(RuntimeError, match=r"skipped 2 of 4") as caught:
        UniMolAdapter().process_dataset(path, cache_path=str(cache))
    assert "not a pickle" in str(caught.value) and "missing atoms or coordinates" in str(caught.value)
    report = json.loads(Path(str(cache) + ".skipped.json").read_text())
    assert report["by_reason"] == {"not a pickle": 1, "missing atoms or coordinates": 1}


def test_unimol_clean_dataset_still_processes(tmp_path):
    path = _lmdb_with(tmp_path, [GOOD_RECORD, GOOD_RECORD])
    assert UniMolAdapter().process_dataset(path, cache_path=str(tmp_path / "o" / "u.pkl")) == 2
    assert len(UniMolAdapter().load_raw_data(path)) == 2


def test_unimol_load_raw_data_raises_on_a_bad_record(tmp_path):
    path = _lmdb_with(tmp_path, [GOOD_RECORD, b"\x80garbage"])
    with pytest.raises(ValueError, match="not a pickle"):
        UniMolAdapter().load_raw_data(path)


@pytest.mark.parametrize("adapter_cls, name", [
    (ProteinAdapter, "missing.pdb"),
    (ProteinAdapter, "missing.cif"),
    (RNAAdapter, "missing.cif"),
])
def test_unreadable_structure_raises_instead_of_returning_no_blocks(tmp_path, adapter_cls, name):
    with pytest.raises(Exception, match=r"missing"):
        adapter_cls().create_blocks(tmp_path / name)


def test_protein_adapter_rejects_an_unsupported_suffix(tmp_path):
    path = tmp_path / "structure.xyz"
    path.write_text("not a structure")
    with pytest.raises(ValueError, match=r"unsupported structure file suffix.*structure\.xyz"):
        ProteinAdapter().create_blocks(path)


def test_unreadable_structure_is_reported_with_its_parse_error(tmp_path):
    class _OneMissingFile(ProteinAdapter):
        def load_raw_data(self, data_path, max_samples=None, **kwargs):
            return [tmp_path / "missing.pdb"]

    cache = tmp_path / "out" / "protein.pkl"
    with pytest.raises(RuntimeError, match=r"skipped 1 of 1") as caught:
        _OneMissingFile().process_dataset("unused", cache_path=str(cache))
    assert "FileNotFoundError" in str(caught.value) and "missing.pdb" in str(caught.value)

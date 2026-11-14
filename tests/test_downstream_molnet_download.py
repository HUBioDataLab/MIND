"""``prepare download`` of the MoleculeNet tasks: pinned archive, no network (``requests.get`` is replaced)."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile

import pytest
import yaml

from downstream.tasks.ec import prepare as ec_prepare
from downstream.tasks.molnet import common, prepare

TOP = "molecular_property_prediction"


def _archive_bytes(datasets=None) -> bytes:
    """A tar.gz with the release layout: <TOP>/<dataset>/{train,valid,test}.lmdb (tiny placeholder files)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for dataset in sorted(common.TASK_OF_DATASET) if datasets is None else datasets:
            for split in ("train", "valid", "test"):
                payload = f"{dataset}/{split}".encode()
                info = tarfile.TarInfo(f"{TOP}/{dataset}/{split}.lmdb")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _pins(data: bytes) -> dict:
    return {"url": "https://example.invalid/molecular_property_prediction.tar.gz",
            "name": "molecular_property_prediction.tar.gz", "bytes": len(data),
            "md5": hashlib.md5(data).hexdigest(), "sha256": hashlib.sha256(data).hexdigest(),
            "extract_subdir": "raw/extracted"}


class _Response:
    def __init__(self, data: bytes):
        self.status_code, self._data = 200, data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, size):
        for start in range(0, len(self._data), size):
            yield self._data[start:start + size]


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("download must not touch the network here")
    monkeypatch.setattr(ec_prepare.requests, "get", refuse)


def _config(tmp_path, data: bytes, **pin_overrides) -> dict:
    config = common.load_task_config(data_root=tmp_path / "unimol_molnet" / "mn_esol", dataset="esol")
    config["archive"] = {**_pins(data), **pin_overrides}
    return config


def test_checked_in_archive_pins_the_release():
    archive = yaml.safe_load(common.TASK_YAML.read_text())["archive"]
    assert archive["bytes"] == 3_764_044_131
    assert archive["md5"] == "007286c6367984876cfd05475d2489ac"
    assert archive["sha256"] == "a59d465eb82d9498e2696a23b687a5e53d4b8545a8437816c9f7bc537ade8246"
    assert archive["url"].startswith("https://") and archive["url"].endswith("/" + archive["name"])
    assert common.load_task_config(dataset="esol")["raw_subdir"] == f"{archive['extract_subdir']}/{TOP}"


def test_missing_archive_pin_is_refused(tmp_path):
    raw = yaml.safe_load(common.TASK_YAML.read_text())
    raw["archive"]["sha256"] = None
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="archive lacks"):
        common.load_task_config(path, dataset="esol")


def test_download_fetches_checks_and_extracts_then_reuses_the_archive(tmp_path, monkeypatch):
    data = _archive_bytes()
    calls = []
    monkeypatch.setattr(ec_prepare.requests, "get", lambda url, **kw: calls.append(url) or _Response(data))
    config = _config(tmp_path, data)
    result = prepare.download(config)
    assert calls == [config["archive"]["url"]]
    assert result["archive"]["downloaded"] is True and result["extracted"] is True
    raw = common.task_paths(config)["raw"]
    assert sorted(p.name for p in raw.iterdir()) == sorted(common.TASK_OF_DATASET)
    assert (raw / "esol" / "test.lmdb").read_bytes() == b"esol/test"

    def refuse(*args, **kwargs):
        raise AssertionError("an archive in place must not be fetched again")
    monkeypatch.setattr(ec_prepare.requests, "get", refuse)
    again = prepare.download(config)
    assert again["archive"]["downloaded"] is False and again["extracted"] is False


def test_existing_archive_with_wrong_checksum_is_refused_before_extraction(tmp_path, no_network):
    data = _archive_bytes()
    config = _config(tmp_path, data, sha256="0" * 64)
    raw_dir = tmp_path / "unimol_molnet" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / config["archive"]["name"]).write_bytes(data)
    with pytest.raises(RuntimeError, match="sha256"):
        prepare.download(config)
    assert not (raw_dir / "extracted").exists()
    with pytest.raises(RuntimeError, match="bytes"):
        prepare.download(_config(tmp_path, data, bytes=len(data) + 1))
    with pytest.raises(RuntimeError, match="md5"):
        prepare.download(_config(tmp_path, data, md5="0" * 32))


def test_incomplete_archive_or_foreign_extract_dir_raises(tmp_path, no_network):
    data = _archive_bytes(datasets=["esol"])
    config = _config(tmp_path, data)
    raw_dir = tmp_path / "unimol_molnet" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / config["archive"]["name"]).write_bytes(data)
    with pytest.raises(FileNotFoundError, match="lacks 39 LMDB"):
        prepare.download(config)

    other = tmp_path / "other"
    data = _archive_bytes()
    config = _config(other, data)
    (other / "unimol_molnet" / "raw" / "extracted").mkdir(parents=True)
    (other / "unimol_molnet" / "raw" / config["archive"]["name"]).write_bytes(data)
    with pytest.raises(RuntimeError, match="holds no"):
        prepare.download(config)


def test_download_cli_appends_a_record_and_keeps_existing_provenance(tmp_path, no_network):
    data = _archive_bytes()
    source = tmp_path / "unimol_molnet"
    raw = yaml.safe_load(common.TASK_YAML.read_text())
    raw["source_root"] = str(source)
    raw["archive"] = _pins(data)
    config_path = tmp_path / "task.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    (source / "raw").mkdir(parents=True)
    (source / "raw" / raw["archive"]["name"]).write_bytes(data)
    (source / "provenance.json").write_text(json.dumps({"source_url": "earlier manual record"}))
    assert prepare.main(["download", "--task", "mn_bbbp", "--config", str(config_path)]) == 0
    provenance = json.loads((source / "provenance.json").read_text())
    assert provenance["source_url"] == "earlier manual record"
    (record,) = provenance["runs"]
    assert record["kind"] == "download" and record["md5"] == raw["archive"]["md5"]
    assert record["archive"]["sha256"] == raw["archive"]["sha256"] and record["extracted"] is True
    assert len(record["datasets"]) == 14

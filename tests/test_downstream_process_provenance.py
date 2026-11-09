"""The ``process`` step writes its own provenance record (EC, GO and Fold share one code path)."""

import hashlib
import json

import pytest
import yaml

from downstream.common import provenance
from downstream.tasks.ec import dataset as ec_dataset
from downstream.tasks.ec import prepare as ec_prepare
from downstream.tasks.fold import dataset as fold_dataset
from downstream.tasks.fold import prepare as fold_prepare
from downstream.tasks.go import common as go_common
from downstream.tasks.go import prepare as go_prepare
from test_downstream_ec_dataset import _write_task
from test_downstream_fold_reaction import _need_h5py, make_fold
from test_downstream_go import _write_family


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(config, path):
    path.write_text(yaml.safe_dump(config))
    return path


def _only_process_record(provenance_path, existing_runs=0):
    runs = json.loads(provenance_path.read_text())["runs"]
    assert len(runs) == existing_runs + 1
    record = runs[-1]
    assert record["kind"] == "process"
    return record


def test_record_keeps_other_keys_and_earlier_runs_and_raises_without_qc(tmp_path):
    out = tmp_path / "processed"
    out.mkdir()
    (out / "qc_train.json").write_text(json.dumps({"split": "train", "chains": 3}))
    (out / "qc_train.tsv").write_text("chain\n")
    (out / "train.pt").write_bytes(b"payload")
    (tmp_path / "config.yaml").write_text("task: x\n")
    (tmp_path / "in.txt").write_text("a\n")
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({"runs": [{"kind": "download", "note": "manual process entry stays"}],
                                "leakage": [{"id": 1}]}))
    record = provenance.record_process_run(
        path, started_utc="2026-11-09T00:00:00+00:00", config_path=tmp_path / "config.yaml", processed_dir=out,
        splits=["train"], exit_code=0, inputs=[tmp_path / "in.txt"], command="prepare process")
    stored = json.loads(path.read_text())
    assert stored["leakage"] == [{"id": 1}] and stored["runs"][0]["note"] == "manual process entry stays"
    assert stored["runs"][1] == record
    assert record["outputs_sha256"] == {"train.pt": _sha(out / "train.pt"), "qc_train.tsv": _sha(out / "qc_train.tsv")}
    assert record["inputs_sha256"] == {str(tmp_path / "in.txt"): _sha(tmp_path / "in.txt")}
    assert record["qc"]["train"]["chains"] == 3 and record["exit_code"] == 0
    assert record["config_sha256"] == _sha(tmp_path / "config.yaml") and record["git"]
    with pytest.raises(FileNotFoundError, match="qc_valid.json"):
        provenance.process_run_record(started_utc="x", config_path=tmp_path / "config.yaml", processed_dir=out,
                                      splits=["valid"], exit_code=0, inputs=[])
    with pytest.raises(FileNotFoundError, match="process inputs"):
        provenance.process_run_record(started_utc="x", config_path=tmp_path / "config.yaml", processed_dir=out,
                                      splits=["train"], exit_code=0, inputs=[tmp_path / "absent.txt"])


def test_record_of_a_qc_failed_split_has_no_pt_hash_but_the_exit_code(tmp_path):
    out = tmp_path / "processed"
    out.mkdir()
    (out / "qc_test.json").write_text(json.dumps({"split": "test", "failed_or_dropped_fraction": 0.5}))
    (out / "qc_test.tsv").write_text("chain\n")
    (tmp_path / "config.yaml").write_text("task: x\n")
    record = provenance.process_run_record(started_utc="x", config_path=tmp_path / "config.yaml", processed_dir=out,
                                           splits=["test"], exit_code=1, inputs=[])
    assert record["exit_code"] == 1 and list(record["outputs_sha256"]) == ["qc_test.tsv"]


def test_ec_process_command_records_itself(tmp_path):
    config = _write_task(tmp_path)
    config_path = _dump(config, tmp_path / "task.yaml")
    code = ec_prepare.main(["process", "--config", str(config_path), "--data-root", str(tmp_path),
                            "--splits", "train", "valid", "--workers", "1"])
    assert code == 0
    record = _only_process_record(tmp_path / "provenance.json")
    out = ec_dataset.processed_dir(config)
    assert record["processed_dir"] == str(out) and record["exit_code"] == 0 and record["splits"] == ["train", "valid"]
    assert record["outputs_sha256"]["train.pt"] == _sha(out / "train.pt")
    assert record["qc"]["train"]["status"] == {"ok": 2, "failed": 1}
    assert record["config_sha256"] == _sha(config_path)
    assert str(tmp_path / "splits" / "train.txt") in record["inputs_sha256"]
    assert str(tmp_path / "raw" / "annotations.tsv") in record["inputs_sha256"]


def test_ec_process_records_a_failed_qc_run_too(tmp_path):
    config = _write_task(tmp_path)
    config_path = _dump(config, tmp_path / "task.yaml")
    code = ec_prepare.main(["process", "--config", str(config_path), "--data-root", str(tmp_path),
                            "--splits", "test", "--workers", "1"])
    assert code == 1  # 1 of 2 test chains is too long: QC limit 0.4
    record = _only_process_record(tmp_path / "provenance.json")
    assert record["exit_code"] == 1 and "test.pt" not in record["outputs_sha256"]


def test_go_process_command_records_itself(tmp_path):
    config = _write_family(tmp_path)
    config_path = _dump(config, tmp_path / "task.yaml")
    code = go_prepare.main(["process", "--config", str(config_path), "--data-root", str(tmp_path / "deepfri" / "go"),
                            "--workers", "1"])
    assert code == 0
    record = _only_process_record(tmp_path / "deepfri" / "go" / "provenance.json")
    out = go_common.processed_dir(config)
    assert record["processed_dir"] == str(out) and set(record["qc"]) == {"train", "valid", "test"}
    assert record["outputs_sha256"]["test.pt"] == _sha(out / "test.pt")


def test_fold_process_command_records_itself_and_keeps_the_manual_entry(tmp_path):
    _need_h5py()
    config = make_fold(tmp_path)
    config_path = _dump(config, tmp_path / "task.yaml")
    provenance_path = tmp_path / "gearnet" / "fold" / "provenance.json"
    manual = {"kind": "process", "command": "added by hand", "note": "manual entry"}
    provenance_path.write_text(json.dumps({"runs": [manual]}))
    code = fold_prepare.main(["process", "--config", str(config_path), "--data-root", str(tmp_path / "gearnet" / "fold"),
                              "--workers", "1"])
    assert code == 0
    runs = json.loads(provenance_path.read_text())["runs"]
    assert runs[0] == manual and len(runs) == 2
    record = runs[1]
    out = fold_dataset.processed_dir(config)
    assert record["processed_dir"] == str(out) and record["exit_code"] == 0
    assert set(record["qc"]) == set(config["splits"])
    assert record["outputs_sha256"]["train.pt"] == _sha(out / "train.pt")

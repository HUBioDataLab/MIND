"""Fold and Reaction tasks (single-label multi-class, IEConv-style .hdf5 structures).

Synthetic inputs only (no network, no real archive): hdf5 files are written with h5py when it is installed
(tests that need it skip visibly otherwise), archives are tiny zips built in ``tmp_path``.
"""

import hashlib
import importlib
import json
import zipfile

import numpy as np
import pytest
import yaml

from downstream import finetune, probe, report
from downstream.common import leakage, structure_io
from downstream.common.metrics import accuracy, evaluate_multiclass
from downstream.tasks import KINDS, TASKS, load_task, multiclass
from downstream.tasks.fold import dataset as fold
from downstream.tasks.fold import prepare as fold_prepare
from downstream.tasks.reaction import dataset as reaction
from downstream.tasks.reaction import prepare as reaction_prepare
from downstream_helpers import chain_of

CLASSES = ["a.1", "a.2", "b.1"]
FOLD_SPLITS = {
    "train": [("d1aaaa_", "a.1"), ("d1aaab1", "a.2"), ("d1aaac_", "b.1")],
    "valid": [("d2aaaa_", "a.1")],
    "test_fold": [("d3aaaa_", "a.2")],
    "test_family": [("d4aaaa_", "b.1")],
    "test_superfamily": [("d5aaaa_", "a.1")],
}
STRUCTURES = {  # per protein: residues (resname list) and optional missing atoms
    "d1aaaa_": (["ALA", "SER", "GLY", "TRP"] * 3, {}),
    "d1aaab1": (["LYS", "ALA"] * 5, {0: ("NZ",), 3: ("CA",)}),  # missing sidechain atom (mask) + missing CA (drop residue)
    "d1aaac_": (["VAL", "GLU", "PHE"] * 4, {}),
    "d2aaaa_": (["HIS", "CYS", "MET"] * 4, {}),
    "d3aaaa_": (["ALA", "GLY"] * 6, {}),
    "d4aaaa_": (["SER", "THR"] * 6, {}),
    "d5aaaa_": (["LEU", "ILE"] * 6, {}),
}


def _need_h5py():
    return pytest.importorskip("h5py")


def write_hdf5(path, residues, *, drops=None, frames=1, chain="A", atom_types_junk=True, extra=None):
    """One IEConv-style hdf5 file from ``chain_of`` residues; returns the atom count."""
    h5py = _need_h5py()
    names, resnames, ids, positions, chains = [], [], [], [], []
    for number, (resname, atoms, _record) in enumerate(chain_of(residues, drops=drops or {}), start=1):
        for name, _element, xyz in atoms:
            names.append(name)
            resnames.append(resname)
            ids.append(number)
            positions.append(xyz)
            chains.append(chain)
    pos = np.asarray(positions, dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle["atom_pos"] = np.repeat(pos[None], frames, axis=0)
        handle["atom_names"] = np.array([n.encode() for n in names], dtype="S4")
        handle["atom_residue_names"] = np.array([n.encode() for n in resnames], dtype="S3")
        handle["atom_residue_id"] = np.array(ids, dtype=np.int64)
        handle["atom_chain_names"] = np.array([c.encode() for c in chains], dtype="S2")
        if atom_types_junk:
            handle["atom_types"] = np.full(len(names), 99, dtype=np.int64)  # must never be used for elements
        for key, value in (extra or {}).items():
            handle[key] = value
    return len(names)


def _fold_class_map():
    return "".join(f"{name}\t{i}\n" for i, name in enumerate(CLASSES))


def _fold_label_text(rows, *, four=False):
    return "".join((f"{n}\t7\t{c}\t{c}\n" if four else f"{n}\t{c}\t{c}\n") for n, c in rows)


def make_fold(tmp_path, *, with_hdf5=True):
    """Extracted-archive fixture: <tmp>/gearnet/raw/fold3d/fold3d/{class_map.txt,<split>.txt,<split>/<name>.hdf5}."""
    source = tmp_path / "gearnet"
    root = source / "raw" / "fold3d" / "fold3d"
    root.mkdir(parents=True)
    (root / "class_map.txt").write_text(_fold_class_map())
    index = {}
    for split, rows in FOLD_SPLITS.items():
        (root / f"{split}.txt").write_text(_fold_label_text(rows, four=split in ("test_fold", "test_superfamily")))
        (root / split).mkdir()
        for name, _ in rows:
            if with_hdf5:
                residues, drops = STRUCTURES[name]
                write_hdf5(root / split / f"{name}.hdf5", residues, drops=drops)
                index[name] = str(root / split / f"{name}.hdf5")
    config = fold.load_task_config(data_root=source / "fold")
    config["label_count"] = len(CLASSES)
    config["expected_chain_counts"] = {s: len(r) for s, r in FOLD_SPLITS.items()}
    (source / "fold" / "splits").mkdir(parents=True)
    for split, rows in FOLD_SPLITS.items():
        (source / "fold" / "splits" / f"{split}.txt").write_text("".join(f"{n}\n" for n, _ in rows))
    if with_hdf5:
        (source / "raw" / "fold_file_index.json").write_text(json.dumps(index))
    return config


# ---------------------------------------------------------------------------
# Checked-in configs and registry
# ---------------------------------------------------------------------------

def test_checked_in_yaml_defines_fold_and_reaction_with_verified_numbers():
    assert "multiclass" in KINDS and "fold" in TASKS and "reaction" in TASKS
    fold_config = fold.load_task_config()
    assert fold_config["label_count"] == 1195 and fold_config["metric"] == "accuracy"
    assert fold_config["expected_chain_counts"] == {
        "train": 12312, "valid": 736, "test_fold": 718, "test_family": 1272, "test_superfamily": 1254}
    assert fold_config["test_splits"] == ["test_fold", "test_family", "test_superfamily"]
    source = fold_config["sources"][fold_config["source"]]
    assert (source["md5"], source["bytes"]) == ("7b052a94afa4c66f9bebeb9efd769186", 670898225)
    assert "zenodo.org/api/records/7593591/files/fold3d.zip" in source["url"]
    assert fold_config["data_root"].endswith("/gearnet/fold") and fold_config["source_root"].endswith("/gearnet")
    reaction_config = reaction.load_task_config()
    assert reaction_config["label_count"] == 384 and reaction_config["test_splits"] == ["test"]
    assert reaction_config["expected_chain_counts"] == {"train": 29215, "valid": 2562, "test": 5651}
    reaction_source = reaction_config["sources"][reaction_config["source"]]
    assert reaction_source["archive_sha256"] is None  # unpinned until the first download is reviewed
    assert len(reaction_source["lists"]) == 5 and all(len(v) == 64 for v in reaction_source["lists"].values())
    assert reaction_config["data_root"].endswith("/ieconv/reaction")
    for name, module in (("fold", fold), ("reaction", reaction)):
        loaded = load_task(name)
        assert loaded.KIND == "multiclass" and loaded.ProcessedSplit is module.ProcessedSplit
        assert loaded.load_task_config()["missing_atoms"]["policy"] != "UNDECIDED"
        prepare = importlib.import_module(f"downstream.tasks.{name}.prepare")
        assert callable(prepare.main)  # `make ds-download|ds-process|ds-validate TASK=<name>` runs this module


def test_config_loader_rejects_bad_decisions(tmp_path):
    base = yaml.safe_load(fold.TASK_YAML.read_text())

    def write(**overrides):
        path = tmp_path / "task.yaml"
        path.write_text(yaml.safe_dump({**base, **overrides}))
        return path

    with pytest.raises(ValueError, match="metric must be 'accuracy'"):
        fold.load_task_config(write(metric="fmax"))
    with pytest.raises(ValueError, match="must name every split"):
        fold.load_task_config(write(test_splits=["test_fold", "test_family"]))
    with pytest.raises(ValueError, match="expected_chain_counts must name every split"):
        fold.load_task_config(write(expected_chain_counts={"train": 1}))
    with pytest.raises(ValueError, match="at least one test split|must name every split"):
        fold.load_task_config(write(test_splits=[], splits=["train", "valid"]))
    with pytest.raises(Exception, match="UNDECIDED|policy"):
        fold.load_task_config(write(missing_atoms={"policy": "UNDECIDED"}))
    with pytest.raises(ValueError, match="task is 'fold', expected 'reaction'"):
        multiclass.load_config(fold.TASK_YAML, expected_task="reaction")
    with pytest.raises(ValueError, match="missing required keys"):
        fold.load_task_config(_without(tmp_path, base, "missing_atoms"))


def _without(tmp_path, base, key):
    path = tmp_path / "stripped.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in base.items() if k != key}))
    return path


# ---------------------------------------------------------------------------
# Label and list readers
# ---------------------------------------------------------------------------

def test_fold_class_map_and_label_files_fail_loud(tmp_path):
    path = tmp_path / "class_map.txt"
    path.write_text(_fold_class_map())
    assert fold.read_class_map(path, 3) == CLASSES
    for text, message in [
        (_fold_class_map() + "c.1\t3\n", "exactly 0..2"),
        (_fold_class_map().replace("a.2\t1", "a.2\t0"), "appears twice"),
        (_fold_class_map().replace("a.2\t1", "a.2"), "expected '<fold id><TAB><index>'"),
        (_fold_class_map().replace("a.2\t1", "a.1\t1"), "duplicate class names"),
    ]:
        path.write_text(text)
        with pytest.raises(ValueError, match=message):
            fold.read_class_map(path, 3)
    index = {n: i for i, n in enumerate(CLASSES)}
    labels = tmp_path / "labels.txt"
    labels.write_text("d1aaaa_\ta.1\ta.1\nd1aaab1\t5\ta.2\ta.2\n")
    assert fold.read_label_file(labels, index) == [("d1aaaa_", 0), ("d1aaab1", 1)]  # label = last column
    for text, message in [
        ("d1aaaa_\ta.1\n", "2 columns"), ("d1aaaa_\tz.9\tz.9\n", "not in class_map"),
        ("d1aaaa_\ta.1\ta.1\nd1aaaa_\ta.1\ta.1\n", "duplicate protein"), ("\n", "no rows"),
    ]:
        labels.write_text(text)
        with pytest.raises(ValueError, match=message):
            fold.read_label_file(labels, index)


def _write_reaction_lists(tmp_path, *, classes=("1.1.1.1", "2.7.7.7", "3.5.1.1")):
    source = tmp_path / "ieconv"
    lists = source / "raw" / "lists"
    lists.mkdir(parents=True)
    (lists / "unique_functions.txt").write_text("\n".join(classes) + "\n")
    names = {"train": ["1abc.A", "1abc.B", "2xyz.A"], "valid": ["3def.A"], "test": ["4ghi.A", "4ghi.B"]}
    (lists / "chain_functions.txt").write_text("".join(
        f"{n},{i % len(classes)}\n" for i, n in enumerate(sum(names.values(), []))))
    for split, filename in (("train", "training.txt"), ("valid", "validation.txt"), ("test", "testing.txt")):
        (lists / filename).write_text("".join(f"{n}\n" for n in names[split]))
    config = reaction.load_task_config(data_root=source / "reaction")
    config["label_count"] = len(classes)
    config["expected_chain_counts"] = {s: len(v) for s, v in names.items()}
    return config, names


def test_reaction_lists_are_read_and_checked(tmp_path):
    config, names = _write_reaction_lists(tmp_path)
    classes, labels, split_names = reaction.read_labels(config)
    assert classes == ["1.1.1.1", "2.7.7.7", "3.5.1.1"] and split_names == names
    assert labels["1abc.A"] == 0 and labels["4ghi.B"] == 5 % 3
    info = multiclass.validate_inputs(config, split_names, labels, classes)
    assert info["split_counts"] == {"train": 3, "valid": 1, "test": 2} and info["classes"] == 3
    assert info["per_split"]["test"]["classes_absent_from_train"] == 0
    lists = reaction.lists_dir(config)
    (lists / "unique_functions.txt").write_text("1.1.1.1\n1.1.1.1\n3.5.1.1\n")
    with pytest.raises(ValueError, match="3 classes \\(2 unique\\)"):
        reaction.read_labels(config)
    (lists / "unique_functions.txt").write_text("1.1.1.1\n2.7.7.7\n3.5.1.1\n")
    (lists / "chain_functions.txt").write_text("1abc.A,0\n1abc.A,1\n")
    with pytest.raises(ValueError, match="duplicate protein"):
        reaction.read_labels(config)
    (lists / "chain_functions.txt").write_text("1abc.A,9\n")
    with pytest.raises(ValueError, match="class index 9 >= 3"):
        reaction.read_labels(config)
    (lists / "chain_functions.txt").write_text("1abc.A,0\n")
    with pytest.raises(ValueError, match="no chain_functions row"):
        reaction.read_labels(config)
    config, names = _write_reaction_lists(tmp_path / "two")
    classes, labels, split_names = reaction.read_labels(config)
    config["expected_chain_counts"]["test"] = 3
    with pytest.raises(ValueError, match="differ from expected"):
        multiclass.validate_inputs(config, split_names, labels, classes)
    config["expected_chain_counts"]["test"] = 2
    split_names["valid"] = ["1abc.A"]
    config["expected_chain_counts"]["valid"] = 1
    with pytest.raises(ValueError, match="is in both"):
        multiclass.validate_inputs(config, split_names, labels, classes)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_accuracy_metrics_contract():
    scores = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.6, 0.3, 0.1]])
    targets = np.array([0, 1, 1, 2])
    assert accuracy(targets, scores) == pytest.approx(0.5)
    result = evaluate_multiclass(targets, scores)
    assert result["accuracy"] == pytest.approx(0.5) and result["top5_accuracy"] == 1.0
    assert result["num_proteins"] == 4 and result["num_classes_present"] == 3
    with pytest.raises(ValueError, match="must be integers"):
        accuracy(targets.astype(float), scores)
    with pytest.raises(ValueError, match="class ids must lie in"):
        accuracy(np.array([0, 1, 3, 2]), scores)
    with pytest.raises(ValueError, match="1-D class ids"):
        accuracy(np.eye(3)[[0, 1]], scores[:2])
    with pytest.raises(ValueError, match="NaN or inf"):
        accuracy(targets, np.where(scores > 0.65, np.nan, scores))
    with pytest.raises(ValueError, match="at least one sample"):
        accuracy(np.zeros(0, dtype=int), np.zeros((0, 3)))


# ---------------------------------------------------------------------------
# hdf5 reader
# ---------------------------------------------------------------------------

def test_hdf5_reader_builds_chain_records_from_names_not_atom_types(tmp_path):
    path = tmp_path / "d1aaaa_.hdf5"
    residues = ["ALA", "SER", "GLY", "TRP"]
    write_hdf5(path, residues, frames=3)
    records = structure_io.read_chains(path, mse_to_met=True)
    assert list(records) == ["d1aaaa_"]
    record = records["d1aaaa_"]
    assert [r.resname for r in record.residues] == residues and record.report["num_models_in_file"] == 3
    assert record.sequence == "ASGW"
    ca = record.residues[1].atoms["CA"]
    assert ca[0] == "C" and ca[1].shape == (3,)  # element from the name although atom_types is junk (99)
    assert record.residues[0].atoms["N"][0] == "N" and record.residues[1].atoms["OG"][0] == "O"


def test_hdf5_reader_skips_hydrogens_waters_ligands_and_converts_mse(tmp_path):
    h5py = _need_h5py()
    path = tmp_path / "x.hdf5"
    names, resnames, ids, pos = [], [], [], []

    def add(resname, rid, atoms):
        for k, name in enumerate(atoms):
            names.append(name), resnames.append(resname), ids.append(rid), pos.append((rid * 3.8, k * 0.5, 0.0))

    add("MSE", 1, ["N", "CA", "C", "O", "CB", "CG", "SE", "CE", "1HB", "H"])
    add("HOH", 2, ["O"])
    add("ZN", 3, ["ZN"])
    add("ALA", 4, ["N", "CA", "C", "O", "CB", "HA"])
    with h5py.File(path, "w") as handle:
        handle["atom_pos"] = np.asarray(pos, dtype=np.float32)  # (atoms, 3) is accepted
        handle["atom_names"] = np.array([n.encode() for n in names])
        handle["atom_residue_names"] = np.array([n.encode() for n in resnames])
        handle["atom_residue_id"] = np.array(ids)
    record = structure_io.read_chains(path, mse_to_met=True)["x"]
    assert [r.resname for r in record.residues] == ["MET", "ALA"] and record.residues[0].source_resname == "MSE"
    assert "SD" in record.residues[0].atoms and "SE" not in record.residues[0].atoms
    assert record.report["hydrogens_skipped"] == 3 and record.report["waters_skipped"] == 1
    assert record.report["ligand_residues_skipped"] == 1 and record.report["mse_converted"] == 1
    assert "1HB" not in record.residues[0].atoms and "HA" not in record.residues[1].atoms


def test_hdf5_reader_fails_loud_on_bad_files(tmp_path):
    h5py = _need_h5py()
    path = tmp_path / "bad.hdf5"
    base = {"atom_pos": np.zeros((1, 2, 3), np.float32), "atom_names": np.array([b"N", b"CA"]),
            "atom_residue_names": np.array([b"ALA", b"ALA"]), "atom_residue_id": np.array([1, 1])}

    def write(**overrides):
        with h5py.File(path, "w") as handle:
            for key, value in {**base, **overrides}.items():
                if value is not None:
                    handle[key] = value

    write(atom_names=None)
    with pytest.raises(ValueError, match="lacks datasets \\['atom_names'\\]"):
        structure_io.read_chains(path, mse_to_met=True)
    write(atom_residue_id=np.array([1]))
    with pytest.raises(ValueError, match="inconsistent atom counts"):
        structure_io.read_chains(path, mse_to_met=True)
    write(atom_pos=np.full((1, 2, 3), np.nan, np.float32))
    with pytest.raises(ValueError, match="NaN or inf"):
        structure_io.read_chains(path, mse_to_met=True)
    write(atom_pos=np.zeros((1, 2, 4), np.float32))
    with pytest.raises(ValueError, match="expected \\(frames, atoms, 3\\)"):
        structure_io.read_chains(path, mse_to_met=True)
    write(atom_names=np.array([b"CA", b"CA"]))
    with pytest.raises(ValueError, match="duplicate atom name CA"):
        structure_io.read_chains(path, mse_to_met=True)
    write(atom_names=np.array([b"N", b"1"]))
    with pytest.raises(ValueError, match="no element letter"):
        structure_io.read_chains(path, mse_to_met=True)


def test_hdf5_reader_separates_chains_and_repeated_residue_ids(tmp_path):
    h5py = _need_h5py()
    path = tmp_path / "two.hdf5"
    atoms = ["N", "CA", "C", "O"]
    names = atoms * 2
    with h5py.File(path, "w") as handle:
        handle["atom_pos"] = np.arange(24, dtype=np.float32).reshape(1, 8, 3)
        handle["atom_names"] = np.array([n.encode() for n in names])
        handle["atom_residue_names"] = np.array([b"GLY"] * 8)
        handle["atom_residue_id"] = np.array([1] * 8)  # same id, different chain
        handle["atom_chain_names"] = np.array([b"A"] * 4 + [b"B"] * 4)
    assert len(structure_io.read_chains(path, mse_to_met=True)["two"].residues) == 2


def test_missing_h5py_is_a_loud_import_error(tmp_path, monkeypatch):
    import builtins

    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("blocked")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="needs h5py"):
        structure_io.read_hdf5_chain(tmp_path / "any.hdf5", mse_to_met=True)


# ---------------------------------------------------------------------------
# Processing: same missing-atom policy as EC/GO, int labels, QC
# ---------------------------------------------------------------------------

def test_fold_processing_applies_policy_and_keeps_integer_labels(tmp_path):
    _need_h5py()
    config = make_fold(tmp_path)
    classes, labels, _ = fold.read_labels(config)
    assert classes == CLASSES and labels["d1aaab1"] == 1
    assert multiclass.build_processed(config, labels, classes, splits=list(config["splits"]), workers=1,
                                      builder="test") == 0
    out = fold.processed_dir(config)
    assert out.name == "fold3d_zenodo_zip_dropbb_masksc_skip1024"
    qc = json.loads((out / "qc_train.json").read_text())
    assert qc["status"] == {"ok": 3} and qc["listed_chains"] == 3 and qc["kept_chains"] == 3
    assert qc["masked_residues"] == 1 and qc["residues_missing_backbone"] == 1  # same counts as the GO fixture
    train = fold.ProcessedSplit(out / "train.pt")
    assert train.labels.dtype == np.int64 and train.labels.tolist() == [0, 1, 2] and train.classes == CLASSES
    assert train.report == {"listed_chains": 3, "kept_chains": 3, "dropped_chains": 0}
    assert train.sequences[1] == "KAKKAKAKA"  # residue 3 of d1aaab1 (no CA) is dropped, never padded with V
    masked = train.payload["samples"][1].downstream_masked_block
    assert masked.numel() == 9 * 14 and bool(masked[:14].all()) and int(masked.sum()) == 14
    embeddings = np.arange(6, dtype=np.float32).reshape(3, 2)
    np.testing.assert_allclose(train.chain_embeddings(embeddings), embeddings)
    with pytest.raises(ValueError, match="expected 3 sample embeddings"):
        train.chain_embeddings(embeddings[:1])
    with pytest.raises(FileExistsError):
        multiclass.build_processed(config, labels, classes, splits=["train"], workers=1, builder="test")


def test_ca_only_domain_is_dropped_and_counted_never_kept(tmp_path):
    h5py = _need_h5py()
    config = make_fold(tmp_path)
    config["qc"]["max_failed_chain_fraction"] = 0.5
    root = tmp_path / "gearnet" / "raw" / "fold3d" / "fold3d"
    target = root / "train" / "d1aaac_.hdf5"
    target.unlink()
    n = 12
    with h5py.File(target, "w") as handle:  # a CA-only domain (the IEConv loader special-cases these)
        handle["atom_pos"] = np.arange(n * 3, dtype=np.float32).reshape(1, n, 3)
        handle["atom_names"] = np.array([b"CA"] * n)
        handle["atom_residue_names"] = np.array([b"ALA"] * n)
        handle["atom_residue_id"] = np.arange(n)
    classes, labels, _ = fold.read_labels(config)
    assert multiclass.build_processed(config, labels, classes, splits=["train"], workers=1, builder="test") == 0
    out = fold.processed_dir(config)
    qc = json.loads((out / "qc_train.json").read_text())
    assert qc["kept_chains"] == 2 and qc["listed_chains"] == 3 and qc["status"] == {"ok": 2, "dropped": 1}
    train = fold.ProcessedSplit(out / "train.pt")
    assert train.chains == ["d1aaaa_", "d1aaab1"] and train.labels.tolist() == [0, 1]
    assert train.report["dropped_chains"] == 1  # visible to every consumer, not silent
    config["qc"]["max_failed_chain_fraction"] = 0.2
    assert multiclass.build_processed(config, labels, classes, splits=["valid", "test_fold"], workers=1,
                                      builder="test") == 0  # these two have no CA-only domain


def test_qc_threshold_blocks_the_processed_file_and_missing_files_raise(tmp_path):
    _need_h5py()
    config = make_fold(tmp_path)
    config["qc"]["max_failed_chain_fraction"] = 0.2
    (tmp_path / "gearnet" / "raw" / "fold3d" / "fold3d" / "train" / "d1aaaa_.hdf5").unlink()
    classes, labels, _ = fold.read_labels(config)
    assert multiclass.build_processed(config, labels, classes, splits=["train"], workers=1, builder="test") == 1
    out = fold.processed_dir(config)
    assert not (out / "train.pt").exists()
    assert json.loads((out / "qc_train.json").read_text())["reasons"] == {"structure_missing": 1}
    index_path = multiclass.file_index_path(config)
    index = json.loads(index_path.read_text())
    del index["d2aaaa_"]
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="no structure file, first \\['d2aaaa_'\\]"):
        multiclass.build_processed(config, labels, classes, splits=["valid"], workers=1, builder="test")
    with pytest.raises(ValueError, match="unknown split 'dev'"):
        multiclass.build_processed(config, labels, classes, splits=["dev"], workers=1, builder="test")


def test_processed_split_rejects_corrupt_payloads(tmp_path):
    import torch

    _need_h5py()
    config = make_fold(tmp_path)
    classes, labels, _ = fold.read_labels(config)
    assert multiclass.build_processed(config, labels, classes, splits=["valid"], workers=1, builder="test") == 0
    path = fold.processed_dir(config) / "valid.pt"
    payload = torch.load(path, weights_only=False)
    cases = {
        "lacks 'classes'": {k: v for k, v in payload.items() if k != "classes"},
        "1-D int64 vector": {**payload, "labels": payload["labels"].float()},
        "class id outside": {**payload, "labels": payload["labels"] + 7},
        "inconsistent chain/sample": {**payload, "sample_chain": payload["sample_chain"][:0]},
    }
    for message, broken in cases.items():
        target = tmp_path / "broken.pt"
        torch.save(broken, target)
        with pytest.raises(ValueError, match=message):
            fold.ProcessedSplit(target)


# ---------------------------------------------------------------------------
# download (offline: archives are pre-placed, the md5/size pins are overridden)
# ---------------------------------------------------------------------------

def _build_fold_archive(tmp_path, *, drop_hdf5=None, extra_hdf5=False):
    """fold3d.zip with a nested layout: fold3d/{class_map.txt,<split>.txt,<split>.zip -> <split>/<name>.hdf5}."""
    _need_h5py()
    staging = tmp_path / "staging"
    (staging / "fold3d").mkdir(parents=True)
    (staging / "fold3d" / "class_map.txt").write_text(_fold_class_map())
    for split, rows in FOLD_SPLITS.items():
        (staging / "fold3d" / f"{split}.txt").write_text(_fold_label_text(rows))
        (staging / split).mkdir(parents=True)
        for name, _ in rows:
            if name != drop_hdf5:
                write_hdf5(staging / split / f"{name}.hdf5", STRUCTURES[name][0])
        if extra_hdf5 and split == "valid":
            write_hdf5(staging / split / "dextra_.hdf5", ["ALA", "GLY", "SER"])
        with zipfile.ZipFile(staging / "fold3d" / f"{split}.zip", "w") as inner:  # members <split>/<name>.hdf5
            for file in sorted((staging / split).glob("*.hdf5")):
                inner.write(file, f"{split}/{file.name}")
    archive = tmp_path / "fold3d.zip"
    with zipfile.ZipFile(archive, "w") as outer:
        for file in sorted((staging / "fold3d").iterdir()):
            outer.write(file, f"fold3d/{file.name}")
    return archive


def _fold_config_for_archive(tmp_path, archive):
    source = tmp_path / "gearnet"
    (source / "raw").mkdir(parents=True)
    (source / "raw" / "fold3d.zip").write_bytes(archive.read_bytes())  # present => _fetch never touches the network
    config = fold.load_task_config(data_root=source / "fold")
    entry = config["sources"][config["source"]]
    entry["md5"], entry["bytes"] = hashlib.md5(archive.read_bytes()).hexdigest(), archive.stat().st_size
    config["label_count"] = len(CLASSES)
    config["expected_chain_counts"] = {s: len(r) for s, r in FOLD_SPLITS.items()}
    return config


def test_fold_download_extracts_nested_archive_indexes_files_and_validates(tmp_path):
    archive = _build_fold_archive(tmp_path)
    config = _fold_config_for_archive(tmp_path, archive)
    result = fold_prepare.download(config)
    assert result["classes"] == 3 and result["proteins"]["train"] == 3 and result["md5"] == config["sources"][config["source"]]["md5"]
    from downstream.tasks.multiclass_prepare import validate_with
    info = validate_with(config, fold.read_labels)
    assert info["split_counts"]["test_fold"] == 1 and info["structure_files_indexed"] == 7
    paths = multiclass.task_paths(config)
    assert (paths["splits"] / "train.txt").read_text() == "d1aaaa_\nd1aaab1\nd1aaac_\n"
    # idempotent rerun: nothing is overwritten, same content passes
    assert fold_prepare.download(config)["proteins"] == result["proteins"]
    config["sources"][config["source"]]["md5"] = "0" * 32
    with pytest.raises(RuntimeError, match="md5"):
        fold_prepare.download(config)
    config["sources"][config["source"]]["md5"] = hashlib.md5((paths["raw"] / "fold3d.zip").read_bytes()).hexdigest()
    config["sources"][config["source"]]["bytes"] += 1
    with pytest.raises(RuntimeError, match="bytes, pinned"):
        fold_prepare.download(config)


def test_fold_download_refuses_missing_or_unlisted_structure_files(tmp_path):
    archive = _build_fold_archive(tmp_path / "a", drop_hdf5="d3aaaa_")
    config = _fold_config_for_archive(tmp_path / "a", archive)
    with pytest.raises(RuntimeError, match="1 of 7 listed proteins have no hdf5 file.*d3aaaa_"):
        fold_prepare.download(config)
    archive = _build_fold_archive(tmp_path / "b", extra_hdf5=True)
    config = _fold_config_for_archive(tmp_path / "b", archive)
    with pytest.raises(RuntimeError, match="not listed, first \\['dextra_'\\]"):
        fold_prepare.download(config)


def test_archive_extraction_refuses_path_escape_and_unknown_formats(tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as handle:
        handle.writestr("../outside.txt", "x")
    with pytest.raises(RuntimeError, match="escapes"):
        multiclass.safe_extract_archive(evil, tmp_path / "dest")
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not an archive")
    with pytest.raises(RuntimeError, match="neither a zip nor a tar"):
        multiclass.safe_extract_archive(junk, tmp_path / "dest")


def test_reaction_download_requires_pinned_archive_and_exact_hdf5_set(tmp_path):
    _need_h5py()
    config, names = _write_reaction_lists(tmp_path)
    source = config["sources"][config["source"]]
    lists = reaction.lists_dir(config)
    source["lists"] = {name: hashlib.sha256((lists / name).read_bytes()).hexdigest() for name in source["lists"]}
    staging = tmp_path / "staging"
    staging.mkdir()
    archive = tmp_path / "ieconv" / "raw" / source["archive_name"]
    with zipfile.ZipFile(archive, "w") as handle:
        for name in sum(names.values(), []):
            write_hdf5(staging / f"{name}.hdf5", ["ALA", "SER", "GLY", "TRP"])
            handle.write(staging / f"{name}.hdf5", f"data/{name}.hdf5")
    with pytest.raises(RuntimeError, match="UNPINNED archive"):
        reaction_prepare.download(config)
    source["archive_sha256"], source["archive_bytes"] = "0" * 64, archive.stat().st_size
    with pytest.raises(RuntimeError, match="differ from the pins"):
        reaction_prepare.download(config)
    source["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = reaction_prepare.download(config)
    assert result["proteins"] == {s: len(v) for s, v in names.items()} and result["classes"] == 3
    assert result["archive"]["sha256"] == source["archive_sha256"]
    index = json.loads(multiclass.file_index_path(config).read_text())
    assert sorted(index) == sorted(sum(names.values(), []))
    # processed end to end with the same policy code path
    classes, labels, _ = reaction.read_labels(config)
    assert multiclass.build_processed(config, labels, classes, splits=["train", "valid", "test"], workers=1,
                                      builder="test") == 0
    test = reaction.ProcessedSplit(reaction.processed_dir(config) / "test.pt")
    assert test.chains == ["4ghi.A", "4ghi.B"] and test.labels.tolist() == [labels["4ghi.A"], labels["4ghi.B"]]


# ---------------------------------------------------------------------------
# Probe CLI, report, fine-tune refusal
# ---------------------------------------------------------------------------

def _write_embeddings(outputs, task, model, splits, num_classes, *, seed=0):
    rng = np.random.default_rng(seed)
    for split, count in splits.items():
        labels = (np.arange(count) % num_classes).astype(np.int64)
        features = np.eye(num_classes, dtype=np.float32)[labels] * 3 + rng.normal(0, 0.3, (count, num_classes)).astype(np.float32)
        path = outputs / "downstream" / task / model / "embeddings" / f"{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, chains=np.array([f"{split}{i}" for i in range(count)]), labels=labels, emb_mean_atoms=features)
        path.with_suffix(".json").write_text(json.dumps({"split": split}))


def test_probe_cli_reports_one_accuracy_per_fold_test_split_and_report_table(tmp_path):
    config = fold.load_task_config()
    config_path = tmp_path / "task.yaml"
    data = yaml.safe_load(fold.TASK_YAML.read_text())
    data["label_count"] = 3
    data["probe"].update(epochs=40, patience=40, lr=0.05, batch_size=16)
    config_path.write_text(yaml.safe_dump(data))
    outputs = tmp_path / "outputs"
    _write_embeddings(outputs, "fold", "toy", {"train": 60, "valid": 12, "test_fold": 9, "test_family": 6,
                                               "test_superfamily": 3}, 3)
    original = fold.TASK_YAML
    fold.TASK_YAML = config_path  # the CLI reads task.TASK_YAML
    try:
        assert probe.main(["--task", "fold", "--model-id", "toy", "--head", "linear", "--seeds", "0", "1",
                           "--outputs-root", str(outputs), "--threads", "1"]) == 0
    finally:
        fold.TASK_YAML = original
    run = outputs / "downstream" / "fold" / "toy" / "linear_probe.mean_atoms" / "seed0"
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["kind"] == "multiclass" and sorted(metrics["test"]) == ["test_family", "test_fold", "test_superfamily"]
    assert metrics["test"]["test_fold"]["num_proteins"] == 9 and metrics["test"]["test_superfamily"]["num_proteins"] == 3
    assert metrics["val"]["accuracy"] > 0.8 and metrics["test"]["test_fold"]["accuracy"] > 0.8
    predictions = np.load(run / "predictions.npz")
    assert predictions["test_labels"].dtype == np.int64 and predictions["test_family_scores"].shape == (6, 3)
    assert predictions["test_superfamily_labels"].tolist() == [0, 1, 2]
    assert config["metric"] == "accuracy"
    assert report.main(["--task", "fold", "--outputs-root", str(outputs)]) == 0
    table = (outputs / "downstream" / "fold" / "results.md").read_text()
    assert "test accuracy test_fold" in table and "GearNet-Edge-IEConv" in table and "48.3 %" in table
    rows = json.loads((outputs / "downstream" / "fold" / "results.json").read_text())["rows"]
    assert rows[0]["seeds"] == [0, 1] and rows[0]["test_accuracy_test_family"]["n"] == 2


def test_probe_needs_num_classes_and_unknown_kinds_raise():
    import torch

    x, y = torch.zeros(4, 2), torch.zeros(4, dtype=torch.long)
    with pytest.raises(ValueError, match="needs num_classes"):
        probe.train_probe(x, y, x, y, head_kind="linear", settings={}, seed=0, log=print, kind="multiclass")
    for call in (lambda: probe.build_loss("ordinal", {}), lambda: probe.validation_score("ordinal", y, y),
                 lambda: probe.predict(torch.nn.Linear(2, 2), x, kind="ordinal")):
        with pytest.raises(ValueError, match="unknown task kind 'ordinal'"):
            call()


def test_finetune_refuses_multiclass_tasks_loudly():
    for task in ("fold", "reaction"):
        with pytest.raises(SystemExit) as exc:
            finetune.main(["--task", task, "--model", "anything"])
        assert exc.value.code == 2


def test_split_roles_keep_three_split_tasks_unchanged():
    assert multiclass.split_roles({"splits": ["train", "valid", "test"]}) == ("train", "valid", ["test"])
    assert multiclass.split_roles(fold.load_task_config()) == ("train", "valid", ["test_fold", "test_family", "test_superfamily"])


# ---------------------------------------------------------------------------
# Leakage registry additions
# ---------------------------------------------------------------------------

def test_leakage_namespaces_for_scop_domains_and_dotted_pdb_chains():
    assert leakage.normalize_id("D1V4WB_", "scop_domain") == "d1v4wb_"
    assert leakage.normalize_id("d1aaab1", "scop_domain") == "d1aaab1"
    assert leakage.normalize_id("d1abc.1", "scop_domain") == "d1abc.1"
    for bad in ("1v4wb_", "d1v4w", "d1v4wb__x", "", "p12345"):
        with pytest.raises(leakage.LeakageError):
            leakage.normalize_id(bad, "scop_domain")
    assert leakage.normalize_id("4XI6.A", "pdb_chain") == "4xi6_A"
    assert leakage.normalize_id("1ABC-A", "pdb_chain") == "1abc_A" and leakage.normalize_id("6kmw_aL", "pdb_chain") == "6kmw_aL"
    for bad in ("4xi6.", "4xi6.A.B", "4xi6.A_B", "4xi6"):
        with pytest.raises(leakage.LeakageError):
            leakage.normalize_id(bad, "pdb_chain")


def test_task_queries_yaml_documents_namespaces_and_comparisons_fail_loud(tmp_path):
    queries = leakage.load_task_queries()
    assert queries["fold"]["namespace"] == "scop_domain" and queries["fold"]["compare_with"] == []
    assert queries["reaction"]["namespace"] == "pdb_chain" and queries["reaction"]["id_example"] == "4xi6.A"
    plan = tmp_path / "plan.csv"
    plan.write_text("id,split,chunk,position\nP12345,train,0,0\nO00123,val,0,1\n")
    reference = leakage.load_reference("protein_test", {
        "kind": "plan", "path": plan, "namespace": "uniprot", "id_column": "id", "split_column": "split"})
    with pytest.raises(leakage.NamespaceMismatchError, match="SCOPe domain ids.*no '0 leaked'"):
        leakage.compare("test_fold", ["d1v4wb_"], "scop_domain", reference)
    with pytest.raises(leakage.NamespaceMismatchError, match="refusing to report 0 leaked"):
        leakage.compare("test", ["4xi6.A"], "pdb_chain", reference)
    mapped = leakage.compare("test", ["4xi6.A", "4xi7.A"], "pdb_chain", reference, id_map={"4xi6_A": "P12345"})
    assert mapped.counts == {"seen_train": 1, "unmapped": 1} and mapped.method == "id_map"
    bad = tmp_path / "refs.yaml"
    bad.write_text(yaml.safe_dump({"references": {}, "queries": {"x": {
        "namespace": "uniprot", "id_example": "P12345", "id_source": "s", "compare_with": ["nope"]}}}))
    with pytest.raises(leakage.LeakageError, match="unknown references"):
        leakage.load_task_queries(bad)
    bad.write_text(yaml.safe_dump({"references": {}, "queries": {"x": {
        "namespace": "uniprot", "id_example": "not-an-id", "id_source": "s", "compare_with": []}}}))
    with pytest.raises(leakage.LeakageError, match="not a UniProt accession"):
        leakage.load_task_queries(bad)

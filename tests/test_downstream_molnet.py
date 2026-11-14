"""MoleculeNet downstream family (mn_*): Uni-Mol LMDB reading, processing, QC, kinds, probe and report.

Synthetic LMDBs are written inside the tests in the Uni-Mol layout (keys ``b"0", b"1", ...``; values are
pickled dicts ``atoms / coordinates / smi / scaffold / target / ori_index`` with 11 conformers per molecule,
the last one flat). No real data and no network are used.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess

import lmdb
import numpy as np
import pytest
import torch
import yaml

from core.pretraining_config import create_pretraining_config, load_config_yaml
from data_loading.atom_position_vocab import RARE_ATOMIC_NUMBER
from downstream import finetune, probe, report
from downstream.common import complex_graph as cg
from downstream.common.metrics import multitarget_regression_errors, multitask_prc_auc, multitask_roc_auc
from downstream.tasks import KINDS, TASKS, load_task
from downstream.tasks.molnet import common, prepare

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOLNET_TASKS = (
    "mn_bbbp", "mn_bace", "mn_clintox", "mn_tox21", "mn_toxcast", "mn_sider", "mn_hiv", "mn_pcba",
    "mn_muv", "mn_esol", "mn_freesolv", "mn_lipo", "mn_qm7", "mn_qm8",
)
CLASSIFICATION = ("mn_bbbp", "mn_bace", "mn_clintox", "mn_tox21", "mn_toxcast", "mn_sider", "mn_hiv", "mn_pcba", "mn_muv")
TARGETS = {"mn_clintox": 2, "mn_tox21": 12, "mn_toxcast": 617, "mn_sider": 27, "mn_pcba": 128, "mn_muv": 17, "mn_qm8": 12}


# ---------------------------------------------------------------------------
# Synthetic Uni-Mol LMDB
# ---------------------------------------------------------------------------

def make_molecule(atoms, target, *, ori_index, smi=None, flat_all=False, conformers=11, seed=0):
    """One LMDB record: ``conformers`` distinct 3-D conformers, the last flat unless ``flat_all`` (all flat)."""
    rng = np.random.default_rng(seed + ori_index)
    count = len(atoms)
    frames = []
    for k in range(conformers):
        xyz = rng.normal(0.0, 1.8, (count, 3)).astype(np.float32)
        if flat_all or k == conformers - 1:
            xyz[:, 2] = 0.0
        frames.append(xyz)
    return {"atoms": list(atoms), "coordinates": frames, "smi": smi or f"MOL{ori_index}", "scaffold": f"SCAF{ori_index % 3}",
            "target": tuple(target), "ori_index": ori_index}


def write_lmdb(path, records, *, skip_keys=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), subdir=False, map_size=1 << 30)
    with env.begin(write=True) as txn:
        for i, record in enumerate(records):
            if i in skip_keys:
                continue
            txn.put(str(i).encode(), pickle.dumps(record, protocol=4))
    env.close()


ESOL_ATOMS = {
    0: ["C", "C", "O", "H", "H", "H", "H", "H", "H"],          # ethanol with explicit H
    1: ["N", "C", "C", "C", "H", "H", "H"],
    2: ["Cl", "C", "C", "H", "H"],
    3: ["C", "C", "S", "H"],
    4: ["C", "O", "H", "H"],
    5: ["C", "C", "N", "H"],
    6: ["C", "O", "O", "H"],
    7: ["C", "C", "H"],
}


def esol_records(ids):
    return [make_molecule(ESOL_ATOMS[i], (-0.5 * i - 1.0,), ori_index=i) for i in ids]


def write_family(tmp_path, dataset="esol", *, train=None, valid=None, test=None, counts=None, edit_config=None):
    """Raw layout ``<source>/raw/extracted/molecular_property_prediction/<dataset>`` + a task config for it."""
    source = tmp_path / "unimol_molnet"
    raw = source / "raw" / "extracted" / "molecular_property_prediction" / dataset
    train = esol_records([0, 1, 2]) if train is None else train
    valid = esol_records([3]) if valid is None else valid
    test = esol_records([4, 5]) if test is None else test
    for name, records in (("train", train), ("valid", valid), ("test", test)):
        write_lmdb(raw / f"{name}.lmdb", records)
    config = common.load_task_config(data_root=source / common.TASK_OF_DATASET[dataset], dataset=dataset)
    config["expected_counts"] = counts or {"train": len(train), "valid": len(valid), "test": len(test)}
    if edit_config:
        edit_config(config)
    return config


def tiny_model_config():
    values = load_config_yaml(os.path.join(REPO, "core", "pretraining_config_multidomain_1m.yaml"))
    layers = len(values["layer_types"])
    values.update(hidden_dims=[32] * layers, num_heads=[4] * layers, graph_dim=32,
                  mlp_hidden_size=64, gaussian_kernels=16)
    return create_pretraining_config(**values)


def processed(config, split):
    return common.MolnetProcessedSplit(common.processed_dir(config) / f"{split}.pt")


# ---------------------------------------------------------------------------
# Registry and config
# ---------------------------------------------------------------------------

def test_registry_lists_the_14_molnet_tasks_and_keeps_the_old_ones_in_front():
    assert TASKS[:7] == ("ec", "lba", "go_mf", "go_bp", "go_cc", "fold", "reaction")
    assert TASKS[7:] == MOLNET_TASKS
    assert "multilabel_masked" in KINDS and "multitarget_regression" in KINDS
    assert KINDS[:3] == ("multilabel", "multiclass", "regression")


@pytest.mark.parametrize("name", MOLNET_TASKS)
def test_every_molnet_task_loads_with_the_registry_surface_and_the_right_kind(name):
    module = load_task(name)
    expected = "multilabel_masked" if name in CLASSIFICATION else "multitarget_regression"
    assert module.KIND == expected
    config = module.load_task_config(module.TASK_YAML)
    assert config["task"] == name and config["num_targets"] == TARGETS.get(name, 1)
    assert config["splits"] == ["train", "valid", "test"]
    assert config["conformer"]["policy"] == "first" and config["hydrogens"] == "drop"
    assert config["max_atoms_policy"] == "none"
    assert config["data_root"].endswith(f"/unimol_molnet/{name}")
    if expected == "multilabel_masked":
        assert config["metrics"] == ["roc_auc", "prc_auc"] and config["primary_metric"] == "roc_auc"
    else:
        assert config["metrics"] == ["rmse", "mae"]
        assert config["primary_metric"] == ("mae" if name in ("mn_qm7", "mn_qm8") else "rmse")
    assert sum(config["expected_counts"].values()) > 0


def test_task_yaml_covers_exactly_the_registry_names_and_excludes_qm9():
    values = yaml.safe_load(open(common.TASK_YAML))
    assert sorted(entry["task"] for entry in values["datasets"].values()) == sorted(MOLNET_TASKS)
    assert not any("qm9" in key for key in values["datasets"])
    with pytest.raises(ValueError, match="unknown task"):
        load_task("mn_qm9")


@pytest.mark.parametrize("edit,message", [
    (lambda c: c["conformer"].update(policy="last"), "conformer.policy"),
    (lambda c: c.update(hydrogens="keep"), "hydrogens"),
    (lambda c: c.update(max_atoms_policy="crop"), "max_atoms_policy"),
    (lambda c: c["conformer"].update(expected_per_molecule=0), "expected_per_molecule"),
    (lambda c: c.pop("probe"), "missing required keys"),
])
def test_config_loader_rejects_bad_decisions(tmp_path, edit, message):
    values = yaml.safe_load(open(common.TASK_YAML))
    edit(values)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match=message):
        common.load_task_config(path, dataset="esol")
    with pytest.raises(ValueError, match="unknown MoleculeNet dataset"):
        common.load_task_config(common.TASK_YAML, dataset="qm9dft")


# ---------------------------------------------------------------------------
# Processing: first conformer, hydrogens, flags, QC
# ---------------------------------------------------------------------------

def test_prepare_end_to_end_uses_first_conformer_drops_hydrogens_and_writes_qc(tmp_path):
    config = write_family(tmp_path)
    assert common.build_processed(config, splits=["train", "valid", "test"], config_path=common.TASK_YAML) == 0
    out = common.processed_dir(config)
    for split in ("train", "valid", "test"):
        assert (out / f"{split}.pt").is_file() and (out / f"qc_{split}.json").is_file() and (out / f"qc_{split}.tsv").is_file()
    split = processed(config, "train")
    assert split.ids == ["esol_0", "esol_1", "esol_2"] and len(split) == 3
    assert split.labels.shape == (3, 1) and split.labels.dtype == np.float32
    record = esol_records([0])[0]
    heavy = [i for i, a in enumerate(record["atoms"]) if a != "H"]
    ligand = split.ligand(0)
    assert list(ligand.elements) == ["C", "C", "O"]
    np.testing.assert_array_equal(ligand.positions, record["coordinates"][0][heavy])      # conformer index 0
    assert not np.array_equal(ligand.positions, record["coordinates"][10][heavy])           # not the flat last one
    assert split.counts["hydrogens_removed"].tolist() == [6, 3, 2]
    qc = json.loads((out / "qc_train.json").read_text())
    assert qc["molecules"] == 3 and qc["dropped"] == 0 and qc["conformer_policy"] == "first (index 0 of 11)"
    assert qc["hydrogens"] == {"molecules_with_h": 3, "h_atoms_removed": 11}
    rows = [line.split("\t") for line in (out / "qc_train.tsv").read_text().splitlines()]
    assert rows[0][:3] == ["id", "ori_index", "smiles"] and len(rows) == 4
    provenance = json.loads((common.task_paths(config)["provenance"]).read_text())
    run = provenance["runs"][-1]
    assert run["kind"] == "process" and run["exit_code"] == 0 and run["dataset"] == "esol"
    assert run["config_sha256"] and set(run["qc"]) == {"train", "valid", "test"}


def test_provenance_records_the_config_file_that_was_used(tmp_path, monkeypatch):
    import hashlib

    config = write_family(tmp_path)
    custom = tmp_path / "custom_task.yaml"
    custom.write_text(common.TASK_YAML.read_text() + "\n# a custom copy\n")
    loaded = []

    def load(path, **kwargs):
        loaded.append(path)
        return config

    monkeypatch.setattr(common, "load_task_config", load)
    assert prepare.main(["process", "--task", "mn_esol", "--config", str(custom)]) == 0
    assert loaded == [custom]
    run = json.loads(common.task_paths(config)["provenance"].read_text())["runs"][-1]
    assert run["config"] == str(custom)
    assert run["config_sha256"] == hashlib.sha256(custom.read_bytes()).hexdigest()


def test_ligand_graph_is_the_lba_separate_unimol_path(tmp_path):
    config = write_family(tmp_path)
    common.build_processed(config, splits=["train", "valid", "test"], config_path=common.TASK_YAML)
    split = processed(config, "train")
    model_config = tiny_model_config()
    sample = split.runtime(0, model_config, seed=0)
    expected = cg.ligand_runtime_sample(split.ligand(0), "esol_0", model_config, seed=0)
    assert sample.dataset_type == "unimol" == expected.dataset_type
    assert sample.z.tolist() == [6, 6, 8] and sample.num_nodes == 3          # no H, ids of the whitelist
    assert torch.equal(sample.edge_index, expected.edge_index)
    assert bool((sample.edge_attr <= 5.0 + 1e-4).all()) and not bool(sample.downstream_masked_block.any())
    assert [i for i, _ in split.iter_runtime(model_config, seed=0)] == [0, 1, 2]


def test_flat_conformers_rare_elements_single_atom_and_big_molecules_are_flagged_and_counted(tmp_path):
    big = make_molecule(["C"] * 300 + ["H"] * 10, (1.0,), ori_index=7)
    train = [
        make_molecule(["C", "C", "O", "H"], (0.1,), ori_index=0),                    # normal: only the last conformer is flat
        make_molecule(["C", "N", "H"], (0.2,), ori_index=1, flat_all=True),          # all 11 conformers flat
        make_molecule(["Si", "C", "Fe", "H"], (0.3,), ori_index=2),                  # rare elements
        make_molecule(["Na"], (0.4,), ori_index=3),                                  # single heavy atom (whitelist element)
        make_molecule(["H", "H"], (0.5,), ori_index=4),                              # nothing left after dropping H
        big,
    ]
    config = write_family(tmp_path, train=train, valid=esol_records([5]), test=esol_records([6]))
    assert common.build_processed(config, splits=["train", "valid", "test"], config_path=common.TASK_YAML) == 0
    split = processed(config, "train")
    flags = {name: values.tolist() for name, values in split.flags.items()}
    assert flags["all_conformers_flat"] == [False, True, False, False, False, False]
    assert flags["rare_element"] == [False, False, True, False, False, False]
    assert flags["single_heavy_atom"] == [False, False, False, True, False, False]
    assert flags["no_heavy_atoms"] == [False, False, False, False, True, False]
    assert flags["over_256_heavy_atoms"] == [False, False, False, False, False, True]
    assert split.embeddable.tolist() == [True, True, True, True, False, True]
    assert len(split.ligand(5).elements) == 300                                      # not cropped
    assert len(split.ligand(4).elements) == 0 and len(split.ligand(2).elements) == 3  # rare atoms stay
    assert cg.ligand_from_rows(split.ligand(2).elements, split.ligand(2).positions).model_ids().tolist() == [
        RARE_ATOMIC_NUMBER, 6, RARE_ATOMIC_NUMBER]
    qc = json.loads((common.processed_dir(config) / "qc_train.json").read_text())
    assert qc["flags"] == {"all_conformers_flat": 1, "first_conformer_flat": 1, "rare_element": 1,
                           "no_heavy_atoms": 1, "single_heavy_atom": 1, "over_256_heavy_atoms": 1}
    assert qc["rare_elements"]["molecules"] == 1
    assert qc["rare_elements"]["by_element_molecules"] == {"Fe": 1, "Si": 1}
    assert qc["rare_elements"]["atoms_mapped_to_119"] == 2
    assert qc["atoms"]["heavy_over_256"] == 1 and qc["atoms"]["heavy_max"] == 300 and qc["atoms"]["total_over_256"] == 1
    assert qc["dropped"] == 0 and qc["molecules"] == 6
    tsv = (common.processed_dir(config) / "qc_train.tsv").read_text().splitlines()
    assert "all_conformers_flat" in tsv[2] and "rare_element" in tsv[3] and "Fe,Si" in tsv[3]
    model_config = tiny_model_config()
    assert split.runtime(3, model_config, seed=0).num_nodes == 1                    # single atom builds a graph
    with pytest.raises(ValueError, match="no heavy atom.*esol_4"):
        split.runtime(4, model_config, seed=0)
    with pytest.raises(ValueError, match="no heavy atom.*esol_4"):
        list(split.iter_runtime(model_config, seed=0))


def test_test_subsets_read_the_qc_table_and_refuse_unlisted_molecules(tmp_path):
    test = [make_molecule(["C", "N", "H"], (0.2,), ori_index=10, flat_all=True),
            make_molecule(["C", "C", "O", "H"], (0.1,), ori_index=11),
            make_molecule(["Si", "C"], (0.3,), ori_index=12)]
    config = write_family(tmp_path, test=test)
    common.build_processed(config, splits=["train", "valid", "test"], config_path=common.TASK_YAML)
    subsets = common.test_subsets(config, ["esol_10", "esol_11", "esol_12"])
    assert subsets["all_conformers_flat"].tolist() == [True, False, False]
    assert subsets["rare_element"].tolist() == [False, False, True]
    assert subsets["no_flags"].tolist() == [False, True, False]
    assert "no_heavy_atoms" not in subsets                                           # empty subsets are not listed
    with pytest.raises(ValueError, match="not in qc_test.tsv"):
        common.test_subsets(config, ["esol_10", "esol_99"])


def test_classification_labels_become_nan_with_a_mask_and_qc_counts_them(tmp_path):
    def records(ids, targets):
        return [make_molecule(["C", "N", "O", "H"], t, ori_index=i) for i, t in zip(ids, targets)]
    config = write_family(
        tmp_path, "clintox", train=records([0, 1, 2], [(1, -1), (0, 1), (-1, 0)]),
        valid=records([3], [(1, 0)]), test=records([4], [(0, 0)]),
        edit_config=lambda c: None)
    assert config["num_targets"] == 2 and config["kind"] == "multilabel_masked"
    common.build_processed(config, splits=["train", "valid", "test"], config_path=common.TASK_YAML)
    split = processed(config, "train")
    assert np.isnan(split.labels).tolist() == [[False, True], [False, False], [True, False]]
    assert split.label_mask.tolist() == [[True, False], [True, True], [False, True]]
    assert split.labels[0, 0] == 1.0 and split.labels[1, 0] == 0.0
    qc = json.loads((common.processed_dir(config) / "qc_train.json").read_text())["labels"]
    assert qc["cells"] == 6 and qc["labelled"] == 4 and qc["missing"] == 2
    assert qc["positives_per_target"] == [1, 1] and qc["labelled_per_target"] == [2, 2]
    # The processed labels feed the ready-made masked metrics directly (NaN = missing, no extra conversion).
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    assert multitask_roc_auc(split.labels, scores)["roc_auc"] == pytest.approx(1.0)
    assert multitask_prc_auc(split.labels, scores)["num_tasks_skipped"] == 0
    assert multitarget_regression_errors(split.labels, scores)["num_labeled_per_target"] == [2, 2]


def test_regression_labels_keep_their_values_including_negatives_and_a_missing_value_is_masked(tmp_path):
    train = [make_molecule(["C", "O"], (-1.0, 2.5), ori_index=0), make_molecule(["C", "N"], (float("nan"), -3.0), ori_index=1)]
    config = write_family(tmp_path, "qm8dft", train=train, valid=[make_molecule(["C", "O"], (0.0, 0.0), ori_index=2)],
                          test=[make_molecule(["C", "O"], (1.0, 1.0), ori_index=3)],
                          edit_config=lambda c: c.update(num_targets=2))
    common.build_processed(config, splits=["train", "valid", "test"], config_path=common.TASK_YAML)
    split = processed(config, "train")
    assert split.labels[0].tolist() == [-1.0, 2.5] and np.isnan(split.labels[1, 0])
    assert split.label_mask.tolist() == [[True, True], [False, True]]


# ---------------------------------------------------------------------------
# Fail-loud paths
# ---------------------------------------------------------------------------

def test_a_missing_lmdb_key_is_refused(tmp_path):
    config = write_family(tmp_path)
    common.lmdb_path(config, "train").unlink()
    write_lmdb(common.lmdb_path(config, "train"), esol_records([0, 1, 2]), skip_keys=(1,))
    with pytest.raises(ValueError, match="molecules, expected 3"):
        common.build_processed(config, splits=["train"], config_path=common.TASK_YAML)


def test_lmdb_with_a_gap_in_the_keys_is_refused(tmp_path):
    config = write_family(tmp_path)
    path = common.lmdb_path(config, "train")
    path.unlink()
    env = lmdb.open(str(path), subdir=False, map_size=1 << 30)
    with env.begin(write=True) as txn:
        for key, record in ((b"0", esol_records([0])[0]), (b"5", esol_records([1])[0]), (b"2", esol_records([2])[0])):
            txn.put(key, pickle.dumps(record, protocol=4))
    env.close()
    with pytest.raises(KeyError, match="no key"):
        common.build_processed(config, splits=["train"], config_path=common.TASK_YAML)


@pytest.mark.parametrize("mutate,message", [
    (lambda r: r.update(atoms=[], coordinates=[np.zeros((0, 3), np.float32)] * 11), "no atoms"),
    (lambda r: r.pop("target"), "missing keys \\['target'\\]"),
    (lambda r: r.update(unexpected=1), "unexpected keys \\['unexpected'\\]"),
    (lambda r: r.update(coordinates=r["coordinates"][:5]), "5 conformers, expected 11"),
    (lambda r: r.update(coordinates=[c[:-1] for c in r["coordinates"]]), "coordinates"),
    (lambda r: r["coordinates"][3].__setitem__((0, 0), np.nan), "non-finite"),
    (lambda r: r.update(target=()), "targets, expected 1"),
    (lambda r: r.update(target=(float("inf"),)), "non-finite target"),
    (lambda r: r.update(smi=""), "smi"),
    (lambda r: r.update(atoms=["N", "C", "C", "C", "H", "H", "Zz9"]), "element"),
])
def test_corrupt_records_are_refused_with_their_position(tmp_path, mutate, message):
    records = esol_records([0, 1, 2])
    mutate(records[1])
    config = write_family(tmp_path, train=records)
    with pytest.raises((ValueError, KeyError), match=message):
        common.build_processed(config, splits=["train"], config_path=common.TASK_YAML)
    assert not (common.processed_dir(config) / "train.pt").exists()


def test_invalid_classification_codes_are_refused(tmp_path):
    records = [make_molecule(["C", "O"], (2, 0), ori_index=0)]
    config = write_family(tmp_path, "clintox", train=records, valid=records, test=records)
    with pytest.raises(ValueError, match="label code 2"):
        common.build_processed(config, splits=["train"], config_path=common.TASK_YAML)


def test_blocked_pickle_globals_count_mismatch_duplicates_and_overwrite_are_refused(tmp_path):
    config = write_family(tmp_path)
    path = common.lmdb_path(config, "valid")
    path.unlink()
    env = lmdb.open(str(path), subdir=False, map_size=1 << 30)
    with env.begin(write=True) as txn:
        txn.put(b"0", pickle.dumps({"atoms": __import__("collections").OrderedDict()}, protocol=4))
    env.close()
    with pytest.raises(pickle.UnpicklingError, match="blocked global"):
        common.build_processed(config, splits=["valid"], config_path=common.TASK_YAML)

    config = write_family(tmp_path / "count", counts={"train": 4, "valid": 1, "test": 2})
    with pytest.raises(ValueError, match="3 molecules, expected 4"):
        common.build_processed(config, splits=["train"], config_path=common.TASK_YAML)

    config = write_family(tmp_path / "dup", train=esol_records([0, 1, 2]), valid=esol_records([2]))
    with pytest.raises(ValueError, match="ori_index 2 .* both"):
        common.build_processed(config, splits=["train", "valid", "test"], config_path=common.TASK_YAML)

    config = write_family(tmp_path / "again")
    assert common.build_processed(config, splits=["train", "valid", "test"], config_path=common.TASK_YAML) == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        common.build_processed(config, splits=["train"], config_path=common.TASK_YAML)


def test_missing_raw_data_is_refused(tmp_path):
    config = common.load_task_config(data_root=tmp_path / "unimol_molnet" / "mn_esol", dataset="esol")
    with pytest.raises(FileNotFoundError, match="train.lmdb"):
        common.build_processed(config, splits=["train"], config_path=common.TASK_YAML)


def test_processed_split_rejects_corrupt_payloads(tmp_path):
    config = write_family(tmp_path)
    common.build_processed(config, splits=["train", "valid", "test"], config_path=common.TASK_YAML)
    payload = torch.load(common.processed_dir(config) / "valid.pt", weights_only=False)
    cases = {
        "lacks 'flags'": {k: v for k, v in payload.items() if k != "flags"},
        "labels": {**payload, "labels": payload["labels"][:, :0]},
        "label_mask": {**payload, "label_mask": ~payload["label_mask"]},
        "offsets": {**payload, "atom_offsets": payload["atom_offsets"][:-1]},
        "flag": {**payload, "flags": {**payload["flags"], "rare_element": payload["flags"]["rare_element"][:0]}},
        "non-finite": {**payload, "atom_positions": payload["atom_positions"] * float("nan")},
    }
    for message, broken in cases.items():
        target = tmp_path / "broken.pt"
        torch.save(broken, target)
        with pytest.raises(ValueError, match=message):
            common.MolnetProcessedSplit(target)


def test_prepare_cli_validates_without_writing(tmp_path, capsys):
    config = write_family(tmp_path)
    source = tmp_path / "unimol_molnet"
    args = ["--task", "mn_esol", "--data-root", str(source / "mn_esol")]
    # The checked-in expected counts (902 / 113 / 113) do not match the 3-molecule fixture: validation fails loudly.
    with pytest.raises(ValueError, match="expected 902"):
        prepare.main(["validate", *args])
    assert not (source / "mn_esol" / "processed").exists()
    assert config["task"] == "mn_esol"


# ---------------------------------------------------------------------------
# Probe, loss, metrics wiring
# ---------------------------------------------------------------------------

def test_masked_losses_ignore_missing_cells_and_an_all_missing_batch_has_no_gradient():
    out = torch.tensor([[2.0, -1.0], [0.5, 3.0]], requires_grad=True)
    target = torch.tensor([[1.0, float("nan")], [float("nan"), 0.0]])
    loss = probe.build_loss("multilabel_masked", {})(out, target)
    manual = torch.nn.functional.binary_cross_entropy_with_logits(
        out[[0, 1], [0, 1]], torch.tensor([1.0, 0.0]))
    assert float(loss) == pytest.approx(float(manual))
    loss.backward()
    assert out.grad[0, 1] == 0 and out.grad[1, 0] == 0
    empty = torch.full((2, 2), float("nan"))
    loss = probe.build_loss("multilabel_masked", {})(out, empty)
    assert float(loss) == 0.0
    mse = probe.build_loss("multitarget_regression", {})(torch.tensor([[1.0, 5.0]]), torch.tensor([[3.0, float("nan")]]))
    assert float(mse) == pytest.approx(4.0)


def test_old_kinds_keep_their_loss_prediction_and_score_behaviour():
    assert isinstance(probe.build_loss("multilabel", {}), torch.nn.BCEWithLogitsLoss)
    assert isinstance(probe.build_loss("regression", {}), torch.nn.MSELoss)
    assert isinstance(probe.build_loss("multiclass", {}), torch.nn.CrossEntropyLoss)
    targets, predictions = np.array([1.0, 2.0, 4.0]), np.array([1.0, 2.0, 3.0])
    assert probe.validation_score("regression", targets, predictions) == pytest.approx(-np.sqrt(1 / 3))
    head = torch.nn.Linear(2, 1)
    assert probe.predict(head, torch.zeros(3, 2), kind="regression").shape == (3,)
    with pytest.raises(ValueError, match="unknown task kind"):
        probe.build_loss("nonsense", {})
    with pytest.raises(ValueError, match="unknown task kind"):
        probe.validation_score("nonsense", targets, predictions)


def test_existing_kinds_keep_their_pinned_scores():
    """Pinned values for the multiclass/multilabel probes (same data, seed 3, linear head) must not drift."""
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(80, 10)).astype(np.float32))
    xv = torch.from_numpy(rng.normal(size=(30, 10)).astype(np.float32))
    settings = dict(mlp_hidden=16, mlp_dropout=0.1, lr=1e-2, weight_decay=1e-4, epochs=15, patience=5, batch_size=16)

    def multilabel(n):
        a = (rng.random((n, 5)) > 0.6).astype(np.float32)
        a[np.arange(n), rng.integers(0, 5, n)] = 1
        return torch.from_numpy(a)

    cases = {
        "multilabel": (multilabel(80), multilabel(30), {}, 0.6846325167037862, 1),
        "regression": (torch.from_numpy(rng.normal(size=(80, 1)).astype(np.float32)),
                       torch.from_numpy(rng.normal(size=(30, 1)).astype(np.float32)), {}, -1.1807437194436912, 7),
        "multiclass": (torch.from_numpy(rng.integers(0, 4, 80)), torch.from_numpy(rng.integers(0, 4, 30)),
                       {"num_classes": 4}, 0.2, 1),
    }
    for kind, (y, yv, extra, score, epoch) in cases.items():
        _, best = probe.train_probe(x, y, xv, yv, head_kind="linear", settings=settings, seed=3, log=lambda m: None,
                                    kind=kind, **extra)
        assert best["score"] == pytest.approx(score, abs=1e-5) and best["epoch"] == epoch, kind


def _write_embeddings(outputs, task, rng, *, num_targets, nan_fraction=0.0, positive=False, scale=1.0, shift=0.0):
    splits = {"train": 300, "valid": 60, "test": 60}
    ids = {name: [f"{task}_{name}_{i}" for i in range(n)] for name, n in splits.items()}
    for name, n in splits.items():
        features = rng.normal(0, 1, (n, 8)).astype(np.float32)
        base = features[:, :num_targets] if num_targets <= 8 else np.tile(features, (1, 2))[:, :num_targets]
        labels = (base > 0).astype(np.float32) if positive else (base * scale + shift).astype(np.float32)
        if nan_fraction:
            labels[rng.random(labels.shape) < nan_fraction] = np.nan
        path = outputs / "downstream" / task / "toy" / "embeddings" / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, chains=np.array(ids[name]), labels=labels, emb_mean_atoms=features)
        path.with_suffix(".json").write_text(json.dumps({"split": name}))
    return ids


def _write_qc(task, data_root, test_ids, flagged=()):
    config = common.load_task_config(data_root=data_root, dataset=common.DATASET_OF_TASK[task])
    qc_dir = common.processed_dir(config)
    qc_dir.mkdir(parents=True, exist_ok=True)
    rows = ["id\tori_index\tsmiles\tstatus\tflags\ttotal_atoms\theavy_atoms\thydrogens_removed\trare_elements"]
    for i, chain in enumerate(test_ids):
        flag = "rare_element" if i in flagged else ""
        rows.append(f"{chain}\t{i}\tX\t{'flagged' if flag else 'ok'}\t{flag}\t5\t5\t0\t{'Si' if flag else ''}")
    (qc_dir / "qc_test.tsv").write_text("\n".join(rows) + "\n")


def _run_probe(tmp_path, task, data_root):
    outputs = tmp_path / "outputs"
    args = ["--task", task, "--model-id", "toy", "--head", "linear", "--seeds", "0",
            "--data-root", str(data_root), "--outputs-root", str(outputs), "--threads", "1"]
    assert probe.main(args) == 0
    metrics = json.loads((outputs / "downstream" / task / "toy" / "linear_probe.mean_atoms" / "seed0" / "metrics.json").read_text())
    return outputs, metrics


def test_probe_runs_a_multilabel_masked_task_with_missing_labels_and_flagged_subsets(tmp_path):
    outputs = tmp_path / "outputs"
    rng = np.random.default_rng(0)
    data_root = tmp_path / "unimol_molnet" / "mn_tox21"
    ids = _write_embeddings(outputs, "mn_tox21", rng, num_targets=12, nan_fraction=0.2, positive=True)
    _write_qc("mn_tox21", data_root, ids["test"], flagged=(0, 5, 10, 15))
    _, metrics = _run_probe(tmp_path, "mn_tox21", data_root)
    assert metrics["kind"] == "multilabel_masked"
    assert 0.6 < metrics["test"]["all"]["roc_auc"] <= 1.0
    for key in ("roc_auc", "prc_auc", "num_tasks", "num_tasks_skipped", "per_task"):
        assert key in metrics["val"]
    assert metrics["test"]["all"]["num_tasks"] == 12
    assert "rare_element" in metrics["test"]["subsets"] and "no_flags" in metrics["test"]["subsets"]
    assert metrics["test"]["subsets"]["rare_element"]["molecules"] == 4


def test_probe_trains_multitarget_regression_on_standardized_targets_and_scores_in_raw_units(tmp_path):
    outputs = tmp_path / "outputs"
    rng = np.random.default_rng(1)
    data_root = tmp_path / "unimol_molnet" / "mn_qm8"
    ids = _write_embeddings(outputs, "mn_qm8", rng, num_targets=12, nan_fraction=0.1, scale=500.0, shift=-1500.0)
    _write_qc("mn_qm8", data_root, ids["test"])
    _, metrics = _run_probe(tmp_path, "mn_qm8", data_root)
    assert metrics["kind"] == "multitarget_regression"
    errors = metrics["test"]["all"]
    assert errors["num_targets"] == 12 and len(errors["rmse_per_target"]) == 12
    # Errors are in raw units (label std is 500): far above 1, and clearly below the 0.8 * std MAE of a constant predictor.
    assert 1.0 < errors["mae"] < 300.0 and 1.0 < metrics["val"]["mae"] < 300.0
    assert metrics["target_standardization"]["std"][0] == pytest.approx(500.0, rel=0.2)
    assert list(metrics["test"]["subsets"]) == ["no_flags"] and metrics["test"]["subsets"]["no_flags"]["molecules"] == 60


def test_probe_refuses_an_unlabelled_or_constant_regression_target(tmp_path):
    outputs = tmp_path / "outputs"
    rng = np.random.default_rng(2)
    ids = _write_embeddings(outputs, "mn_esol", rng, num_targets=1, scale=0.0, shift=3.0)
    _write_qc("mn_esol", tmp_path / "unimol_molnet" / "mn_esol", ids["test"])
    with pytest.raises(ValueError, match="constant"):
        _run_probe(tmp_path, "mn_esol", tmp_path / "unimol_molnet" / "mn_esol")


def test_finetune_refuses_the_new_kinds_until_they_are_implemented():
    for task in ("mn_esol", "mn_tox21"):
        with pytest.raises(SystemExit):
            finetune.main(["--task", task, "--model", "x", "--device", "cuda:0"])


def test_report_builds_tables_for_both_new_kinds(tmp_path):
    outputs = tmp_path / "outputs"
    rng = np.random.default_rng(0)
    for task, data in (("mn_tox21", dict(num_targets=12, nan_fraction=0.2, positive=True)),
                       ("mn_esol", dict(num_targets=1, scale=2.0, shift=-1.0))):
        ids = _write_embeddings(outputs, task, rng, **data)
        root = tmp_path / "unimol_molnet" / task
        _write_qc(task, root, ids["test"])
        _run_probe(tmp_path, task, root)
    clf = report.collect("mn_tox21", outputs)
    assert clf[0]["test_roc_auc"]["n"] == 1 and clf[0]["test_prc_auc"]["mean"] > 0
    assert "ROC-AUC" in report.to_markdown("mn_tox21", clf) and "tasks skipped" in report.to_markdown("mn_tox21", clf)
    reg = report.collect("mn_esol", outputs)
    assert reg[0]["test_rmse"]["n"] == 1 and reg[0]["test_mae"]["n"] == 1
    markdown = report.to_markdown("mn_esol", reg)
    assert "RMSE" in markdown and "MAE" in markdown and "literature" in markdown


# ---------------------------------------------------------------------------
# Make wiring
# ---------------------------------------------------------------------------

def _make_dry_run(*args):
    result = subprocess.run(["make", "-n", *args], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return " ".join(result.stdout.replace("\\\n", " ").split())


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
def test_make_ds_targets_route_molnet_tasks_to_the_molnet_prepare_with_pipefail():
    process = _make_dry_run("ds-process", "TASK=mn_esol")
    assert "downstream.tasks.molnet.prepare process --task mn_esol" in process and "set -o pipefail" in process
    assert "--workers" not in process
    assert "downstream.tasks.molnet.prepare validate --task mn_bbbp" in _make_dry_run("ds-validate", "TASK=mn_bbbp")
    other = _make_dry_run("ds-process", "TASK=ec", "WORKERS=2")
    assert "downstream.tasks.ec.prepare process --workers 2" in other and "--task" not in other
    assert "downstream.tasks.go.prepare validate" in _make_dry_run("ds-validate", "TASK=go_mf")

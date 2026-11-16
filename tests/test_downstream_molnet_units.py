"""MoleculeNet coordinate units: the declared unit is converted to Angstrom, implausible scales are refused, and
the QM7 revision (Bohr coordinates converted) is separate from the 13 unchanged datasets.

Synthetic LMDBs only (helpers of ``test_downstream_molnet``); no real data, no GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import yaml

from downstream import run_all
from downstream.common import registry
from downstream.tasks.molnet import common
from test_downstream_molnet import MOLNET_TASKS, ESOL_ATOMS, esol_records, make_molecule, write_family

BOHR = common.COORDINATE_UNIT_TO_ANGSTROM["bohr"]
SPLITS = ["train", "valid", "test"]


def scaled(records, factor):
    """The same records with every conformer multiplied by ``factor`` (1 / BOHR gives a Bohr-scale copy)."""
    return [{**r, "coordinates": [np.asarray(frame) * factor for frame in r["coordinates"]]} for r in records]


def qm7_records(ids, *, factor=1.0 / BOHR):
    atoms = {i: ESOL_ATOMS[i % len(ESOL_ATOMS)] for i in ids}
    return scaled([make_molecule(atoms[i], (0.1 * i,), ori_index=i) for i in ids], factor)


def qm7_family(tmp_path, *, factor=1.0 / BOHR, edit_config=None):
    return write_family(tmp_path, "qm7dft", train=qm7_records([0, 1, 2], factor=factor),
                        valid=qm7_records([3], factor=factor), test=qm7_records([4, 5], factor=factor),
                        edit_config=edit_config)


def process(config):
    return common.build_processed(config, splits=SPLITS, config_path=common.TASK_YAML)


def load(config, split):
    return common.MolnetProcessedSplit(common.processed_dir(config) / f"{split}.pt")


# -- declaration in task.yaml ----------------------------------------------------------------------------

def test_checked_in_yaml_declares_a_unit_per_dataset_and_only_qm7_is_bohr_and_v2():
    values = yaml.safe_load(open(common.TASK_YAML))
    assert values["processed_revision"] == "v1" and values["plausible_median_nn_angstrom"] == [1.0, 2.0]
    units = {entry["task"]: entry["coordinate_unit"] for entry in values["datasets"].values()}
    assert sorted(units) == sorted(MOLNET_TASKS)
    assert {task for task, unit in units.items() if unit != "angstrom"} == {"mn_qm7"} and units["mn_qm7"] == "bohr"
    revisions = {entry["task"]: str(entry.get("processed_revision", values["processed_revision"]))
                 for entry in values["datasets"].values()}
    assert revisions["mn_qm7"] == "v2"
    assert {revision for task, revision in revisions.items() if task != "mn_qm7"} == {"v1"}
    for dataset, entry in values["datasets"].items():
        config = common.load_task_config(dataset=dataset)
        assert config["coordinate_scale"] == (BOHR if entry["task"] == "mn_qm7" else 1.0)
        assert common.processed_dir(config).name == revisions[entry["task"]]


@pytest.mark.parametrize("edit,message", [
    (lambda d: d["datasets"]["qm7dft"].pop("coordinate_unit"), "lacks .*coordinate_unit"),
    (lambda d: d["datasets"]["qm7dft"].update(coordinate_unit="nm"), "coordinate_unit 'nm'"),
    (lambda d: d.pop("plausible_median_nn_angstrom"), "missing required keys.*plausible_median_nn_angstrom"),
    (lambda d: d.update(plausible_median_nn_angstrom=[2.0, 1.0]), "plausible_median_nn_angstrom"),
])
def test_config_loader_has_no_silent_default_for_the_unit(tmp_path, edit, message):
    values = yaml.safe_load(open(common.TASK_YAML))
    edit(values)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match=message):
        common.load_task_config(path, dataset="qm7dft")


# -- conversion ------------------------------------------------------------------------------------------

def test_qm7_bohr_coordinates_are_stored_in_angstrom_and_the_conversion_is_recorded(tmp_path):
    config = qm7_family(tmp_path)
    assert config["coordinate_unit"] == "bohr" and common.processed_dir(config).name == "v2"
    process(config)
    reference = make_molecule(ESOL_ATOMS[0], (0.0,), ori_index=0)           # the Angstrom original of molecule 0
    heavy = [i for i, a in enumerate(reference["atoms"]) if a != "H"]
    split = load(config, "train")
    np.testing.assert_allclose(split.ligand(0).positions, reference["coordinates"][0][heavy], atol=2e-5)
    assert split.schema["coordinate_source_unit"] == "bohr" and split.schema["coordinate_unit"] == "angstrom"
    assert split.schema["coordinate_scale"] == BOHR and split.schema["revision"] == "v2"
    qc = json.loads((common.processed_dir(config) / "qc_train.json").read_text())["coordinates"]
    assert qc["declared_unit"] == "bohr" and 1.2 < qc["median_nn_angstrom"] < 1.6 and qc["molecules_checked"] == 3


def test_angstrom_datasets_are_stored_unchanged_with_the_conversion_factor_one(tmp_path):
    config = write_family(tmp_path)
    process(config)
    record = esol_records([1])[0]
    heavy = [i for i, a in enumerate(record["atoms"]) if a != "H"]
    split = load(config, "train")
    np.testing.assert_array_equal(split.ligand(1).positions, record["coordinates"][0][heavy])
    assert split.schema["coordinate_source_unit"] == "angstrom" and split.schema["coordinate_scale"] == 1.0
    assert common.processed_dir(config).name == "v1"


# -- guard -----------------------------------------------------------------------------------------------

def test_guard_refuses_bohr_data_declared_as_angstrom_and_names_the_task(tmp_path):
    def declare_angstrom(config):
        config["coordinate_unit"], config["coordinate_scale"] = "angstrom", 1.0
    config = qm7_family(tmp_path, edit_config=declare_angstrom)
    with pytest.raises(ValueError, match=r"mn_qm7 train: median nearest-neighbour distance 2\.\d+ .*\[1\.0, 2\.0\]"):
        process(config)
    assert not (common.processed_dir(config) / "train.pt").exists()


def test_guard_refuses_a_set_that_is_too_small_for_angstrom_and_names_the_task(tmp_path):
    config = write_family(tmp_path, train=scaled(esol_records([0, 1, 2]), 0.1), valid=scaled(esol_records([3]), 0.1),
                          test=scaled(esol_records([4, 5]), 0.1))
    with pytest.raises(ValueError, match=r"mn_esol train: median nearest-neighbour distance 0\.1\d+ "):
        process(config)


def test_guard_passes_converted_bohr_and_plain_angstrom_data(tmp_path):
    for dataset, root in (("qm7dft", tmp_path / "a"), ("esol", tmp_path / "b")):
        config = qm7_family(root) if dataset == "qm7dft" else write_family(root)
        assert process(config) == 0


def test_guard_ignores_flat_conformers_and_single_atoms_and_refuses_a_split_with_nothing_to_check(tmp_path):
    flat = [make_molecule(["C", "C"], (1.0,), ori_index=0, flat_all=True)]
    flat[0]["coordinates"] = [np.asarray(f) * 40.0 for f in flat[0]["coordinates"]]      # would fail the range if used
    single = [make_molecule(["C"], (1.0,), ori_index=6)]
    config = write_family(tmp_path, train=flat + esol_records([1, 2]), valid=single + esol_records([3]),
                          test=esol_records([4, 5]))
    assert process(config) == 0
    nothing = write_family(tmp_path / "nothing", train=flat, valid=single + esol_records([3]), test=esol_records([4, 5]))
    with pytest.raises(ValueError, match="mn_esol train: no molecule with a 3-D conformer and two heavy atoms"):
        process(nothing)


# -- revision --------------------------------------------------------------------------------------------

def test_old_qm7_data_is_refused_with_a_message_naming_the_task_and_the_command(tmp_path):
    config = qm7_family(tmp_path)
    process(config)
    payload = torch.load(common.processed_dir(config) / "valid.pt", weights_only=False)
    old = {**payload, "schema": {k: v for k, v in payload["schema"].items() if not k.startswith("coordinate")}}
    old["schema"]["revision"] = "v1"
    target = tmp_path / "v1_valid.pt"
    torch.save(old, target)
    with pytest.raises(ValueError, match=r"revision 'v1' but task.yaml declares 'v2' for mn_qm7.*Bohr.*ds-process TASK=mn_qm7"):
        common.MolnetProcessedSplit(target)
    unconverted = {**payload, "schema": {k: v for k, v in payload["schema"].items() if not k.startswith("coordinate")}}
    torch.save(unconverted, target)
    with pytest.raises(ValueError, match="mn_qm7 coordinates are declared in bohr but the file does not record"):
        common.MolnetProcessedSplit(target)


def test_v1_files_of_the_other_datasets_still_load(tmp_path):
    """An existing v1 file has no coordinate keys in its schema; it stays valid for every Angstrom dataset."""
    config = write_family(tmp_path)
    process(config)
    payload = torch.load(common.processed_dir(config) / "train.pt", weights_only=False)
    payload["schema"] = {k: v for k, v in payload["schema"].items() if not k.startswith("coordinate")}
    target = tmp_path / "v1_train.pt"
    torch.save(payload, target)
    assert common.MolnetProcessedSplit(target).ids == ["esol_0", "esol_1", "esol_2"]


# -- ds-run keying ---------------------------------------------------------------------------------------

def test_a_stale_qm7_embedding_is_refused_while_the_embeddings_of_other_tasks_stay_complete(tmp_path):
    """Embedding paths are keyed by task and model: only mn_qm7 reads another processed directory now."""
    model = run_all.ModelPlan(name="m", model_id="m", sha256="x", domains=["unimol"], action="snapshot exists")
    info = {task: run_all.describe_task(run_all.Unit(task)) for task in ("mn_qm7", "mn_esol")}
    assert info["mn_qm7"]["processed"]["train"].parent.name == "v2"
    assert info["mn_esol"]["processed"]["train"].parent.name == "v1"
    for task in info:
        for split in SPLITS:
            path = registry.embeddings_path(task, "m", split, root=tmp_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"npz")
            path.with_suffix(".json").write_text(json.dumps({"processed": f"/data/{task}/processed/v1/{split}.pt"}))
    plan = run_all.plan_units([run_all.Unit("mn_qm7"), run_all.Unit("mn_esol")], True, model, heads=["linear"],
                              seeds=[0], outputs_root=tmp_path)
    assert [label for label, _ in plan["failed"]] == ["embed mn_qm7"]
    assert "built from processed directory 'v1'" in plan["failed"][0][1] and "reads 'v2'" in plan["failed"][0][1]
    assert [unit.task for unit in plan["embed_done"]] == ["mn_esol"] and plan["embed"] == []

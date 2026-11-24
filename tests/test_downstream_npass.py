"""NPASS downstream family (npass_*): config, split rule, CSV reading, conformers, processing, download, wiring.

Tiny synthetic CSV files with real SMILES in temporary directories; no real data, no network, no GPU. The conformer
generation is the real RDKit code (small molecules); one test replaces the embedding call to force failures.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from downstream import ledger, report
from downstream.baselines import ecfp
from downstream.tasks import KINDS, MOLNET_TASKS, NPASS_TASKS, SMILES_TASKS, TASKS, load_task
from downstream.tasks.ec import prepare as ec_prepare
from downstream.tasks.molnet import common as molnet
from downstream.tasks.npass import common, prepare

REPO = Path(__file__).resolve().parents[1]
PAPER_COUNTS = {"ptp1b": 612, "ache": 341, "cox2": 190, "hivrt": 186, "tyrosinase": 186, "cyp3a4": 178, "mrp4": 177,
                "cox1": 140}
COLUMNS = ["np_id", "SMILES", "target_id", "activity_value"]


def smiles_pool(count, chlorine_at=4):
    """Distinct small molecules: alcohols (even k) and amines (odd k) of growing chain length, one with chlorine."""
    pool = [f"{'C' * k}{'O' if k % 2 == 0 else 'N'}" for k in range(1, count + 1)]
    pool[chlorine_at] = "CCCCCl"          # the molecule whose embedding the failure test breaks
    assert len(set(pool)) == count
    return pool


def write_csv(path, rows, header=COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def fixture_rows(count=20, target="NPT178", chlorine_at=4):
    rows = [[f"NPC{1000 + i}", smi, target, f"{0.1 * i + 2.0:.4f}"]
            for i, smi in enumerate(smiles_pool(count, chlorine_at))]
    rows.insert(3, ["NPC9999", "CCCC", "NPT204", "9.9"])      # another target: never read for ptp1b
    return rows


def tiny_config(tmp_path, count=20, dataset="ptp1b"):
    """The real ptp1b config with the counts of ``count`` fixture molecules, rooted under ``tmp_path``."""
    config = common.load_task_config(data_root=tmp_path / "nafm_npass" / common.TASK_OF_DATASET[dataset], dataset=dataset)
    sizes = dict(zip(config["splits"], common.split_sizes(count, config["split"]["fractions"])))
    config.update(expected_molecules=count, paper_molecules=count, expected_counts=sizes)
    config["source_file"] = dict(config["source_file"], rows=count + 1)
    return config


def processed(config, split):
    return common.NpassProcessedSplit(common.processed_dir(config) / f"{split}.pt")


@pytest.fixture
def built(tmp_path):
    config = tiny_config(tmp_path)
    write_csv(common.csv_path(config), fixture_rows())
    assert common.build_processed(config, splits=list(config["splits"]), config_path=common.TASK_YAML) == 0
    return config


# ---------------------------------------------------------------------------
# Registry, task.yaml, Makefile and catalog wiring
# ---------------------------------------------------------------------------

def test_registry_lists_the_eight_npass_tasks_after_the_moleculenet_ones():
    assert NPASS_TASKS == tuple(f"npass_{name}" for name in PAPER_COUNTS)
    assert TASKS[-8:] == NPASS_TASKS and SMILES_TASKS == MOLNET_TASKS + NPASS_TASKS
    assert common.DATASET_OF_TASK == {f"npass_{name}": name for name in PAPER_COUNTS}
    for name in NPASS_TASKS:
        module = load_task(name)
        assert module.KIND == "multitarget_regression" and module.KIND in KINDS
        config = module.load_task_config(module.TASK_YAML)
        assert config["task"] == name and config["num_targets"] == 1 and config["primary_metric"] == "rmse"
        assert config["finetune_seconds_per_epoch"] >= 1


def test_task_yaml_pins_the_source_and_documents_every_count_difference():
    values = yaml.safe_load(common.TASK_YAML.read_text())
    pins = values["source_file"]
    assert re.fullmatch(r"[0-9a-f]{64}", pins["sha256"]) and re.fullmatch(r"[0-9a-f]{32}", pins["md5"])
    assert pins["bytes"] > 0 and pins["rows"] == 4928 and pins["url"].startswith("https://")
    different = {}
    for name, entry in values["datasets"].items():
        assert entry["paper_molecules"] == PAPER_COUNTS[name]
        config = common.load_task_config(dataset=name)
        assert sum(config["expected_counts"].values()) == entry["expected_molecules"]
        if entry["paper_molecules"] != entry["expected_molecules"]:
            different[name] = entry["count_note"]
    assert list(different) == ["cox2"] and "189" in different["cox2"] and "190" in different["cox2"]
    finetune = values["finetune"]
    assert finetune["lr_encoder"] == 1e-4 and finetune["seeds"] == [0, 1, 2] and values["probe"]["seeds"] == [0, 1, 2]
    assert values["split"]["seed"] == 0 and values["split"]["fractions"] == [0.8, 0.1, 0.1]


def test_the_config_loader_refuses_an_undocumented_difference_and_inconsistent_counts(tmp_path):
    def load(edit):
        values = yaml.safe_load(common.TASK_YAML.read_text())
        edit(values["datasets"]["cox1"])
        bad = tmp_path / "task.yaml"
        bad.write_text(yaml.safe_dump(values))
        return common.load_task_config(bad, dataset="cox1")

    with pytest.raises(ValueError, match="count_note"):
        load(lambda e: e.update(expected_molecules=139, expected_counts={"train": 111, "valid": 14, "test": 14}))
    with pytest.raises(ValueError, match="differ from the split rule"):
        load(lambda e: e.update(expected_counts={"train": 113, "valid": 14, "test": 13}))
    values = yaml.safe_load(common.TASK_YAML.read_text())
    del values["source_file"]["sha256"]
    bad = tmp_path / "pins.yaml"
    bad.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="sha256"):
        common.load_task_config(bad, dataset="cox1")
    values = yaml.safe_load(common.TASK_YAML.read_text())
    values["split"]["policy"] = "scaffold"
    bad.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="random_permutation"):
        common.load_task_config(bad, dataset="cox1")
    with pytest.raises(ValueError, match="unknown NPASS dataset"):
        common.load_task_config(dataset="ptp1c")


def test_makefile_alias_and_prepare_routing_and_help():
    import subprocess

    def make(*args):
        out = subprocess.run(["make", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout
        return " ".join(out.replace("\\\n", " ").split())

    tasks = ",".join(NPASS_TASKS)
    assert re.search(r"^NPASS_TASKS := (\S+)$", (REPO / "Makefile").read_text(), re.M).group(1) == tasks
    assert f"--tasks {tasks} " in make("-n", "ds-run", "MODEL=m", "GPU=0", "PLAN=1", "TASKS=npass")
    assert f"--tasks {tasks},ec " in make("-n", "ds-run", "MODEL=m", "GPU=0", "PLAN=1", "TASKS=npass,ec")
    assert "downstream.tasks.npass.prepare download --task npass_ptp1b" in make("-n", "ds-download", "TASK=npass_ptp1b")
    assert "downstream.tasks.npass.prepare process --task npass " in make("-n", "ds-process", "TASK=npass")
    assert "downstream.tasks.npass.prepare validate --task npass_cox1" in make("-n", "ds-validate", "TASK=npass_cox1")
    assert "TASKS=all|molnet|npass" in make("help") and "ds-process TASK=npass_" in make("help")


def test_catalog_requires_rmse_and_mae_for_every_npass_task_in_the_molecule_family():
    catalog = ledger.load_catalog()
    for name in NPASS_TASKS:
        spec = catalog["tasks"][name]
        assert spec["family"] == "molecule_property" and spec["kind"] == "multitarget_regression"
        assert {c["metric"] for c in spec["required"]} == {"rmse", "mae"} and spec["primary"]["metric"] == "rmse"
        assert spec["required_protocols"] == ["mlp_probe"]


# ---------------------------------------------------------------------------
# Split rule (the NaFM reference code)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("count, sizes", [(612, (490, 61, 61)), (341, (273, 34, 34)), (189, (151, 19, 19)),
                                          (186, (149, 19, 18)), (178, (142, 18, 18)), (177, (142, 18, 17)),
                                          (140, (112, 14, 14))])
def test_split_sizes_follow_the_reference_rounding(count, sizes):
    assert common.split_sizes(count, [0.8, 0.1, 0.1]) == sizes


def test_split_sizes_refuse_a_rule_that_would_drop_samples():
    with pytest.raises(ValueError, match="add up"):
        common.split_sizes(145, [0.8, 0.1, 0.1])        # 116 + 14 + 14 = 144: the reference code leaves one out silently


def test_split_indices_are_the_seeded_permutation_and_cover_every_row_once():
    split = common.split_indices(50, [0.8, 0.1, 0.1], 0)
    order = np.random.default_rng(0).permutation(np.arange(50, dtype=np.int64))
    assert split["train"].tolist() == order[:40].tolist() and split["valid"].tolist() == order[40:45].tolist()
    assert split["test"].tolist() == order[45:].tolist()
    assert sorted(np.concatenate(list(split.values())).tolist()) == list(range(50))
    other = common.split_indices(50, [0.8, 0.1, 0.1], 1)
    assert other["train"].tolist() != split["train"].tolist()


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def test_rows_of_other_targets_are_ignored_and_order_is_the_file_order(tmp_path):
    config = tiny_config(tmp_path)
    write_csv(common.csv_path(config), fixture_rows())
    rows = common.read_target_rows(config)
    assert [r["id"] for r in rows] == [f"NPC{1000 + i}" for i in range(20)]
    assert [r["row"] for r in rows][:4] == [0, 1, 2, 4]            # data row 3 belongs to NPT204
    assert rows[1]["value"] == pytest.approx(2.1) and rows[1]["smiles"] == smiles_pool(20)[1]


@pytest.mark.parametrize("edit, message", [
    (lambda rows: rows.pop(), "19 rows for NPT178 .*expected 20"),
    (lambda rows: rows.__setitem__(0, rows[0][:3] + ["abc"]), "not a number"),
    (lambda rows: rows.__setitem__(0, rows[0][:3] + ["nan"]), "non-finite"),
    (lambda rows: rows.__setitem__(1, [rows[0][0]] + rows[1][1:]), "occurs twice"),
    (lambda rows: rows.__setitem__(0, [rows[0][0], "", rows[0][2], rows[0][3]]), "empty id or SMILES"),
])
def test_corrupt_rows_and_count_differences_raise(tmp_path, edit, message):
    config = tiny_config(tmp_path)
    rows = fixture_rows()
    edit(rows)
    write_csv(common.csv_path(config), rows)
    with pytest.raises(ValueError, match=message):
        common.read_target_rows(config)


def test_a_missing_column_or_file_raises(tmp_path):
    config = tiny_config(tmp_path)
    with pytest.raises(FileNotFoundError, match="ds-download"):
        common.read_target_rows(config)
    write_csv(common.csv_path(config), [], header=["np_id", "SMILES", "target_id"])
    with pytest.raises(ValueError, match="lacks columns .*activity_value"):
        common.read_target_rows(config)


def test_the_count_error_names_the_paper_count_and_the_note(tmp_path):
    config = tiny_config(tmp_path)
    config.update(paper_molecules=21, count_note="the release differs from the paper")
    write_csv(common.csv_path(config), fixture_rows()[:-1])
    with pytest.raises(ValueError, match=r"paper: 21; the release differs from the paper"):
        common.read_target_rows(config)


# ---------------------------------------------------------------------------
# Conformers and processing
# ---------------------------------------------------------------------------

def test_conformer_is_3d_deterministic_and_keeps_the_heavy_atoms_in_smiles_order(tmp_path):
    config = tiny_config(tmp_path)
    first = common.generate_conformer("OC(=O)CCN", config=config, where="t")
    again = common.generate_conformer("OC(=O)CCN", config=config, where="t")
    symbols, positions, total, status, mmff = first
    assert symbols == ["O", "C", "O", "C", "C", "N"] and positions.shape == (6, 3) and total == 6 + 7
    assert status == "etkdg" and mmff in ("converged", "not_converged") and np.isfinite(positions).all()
    assert np.abs(positions[:, 2]).max() > 0
    assert np.array_equal(positions, again[1])                      # fixed seed: identical coordinates


def test_an_unparsable_smiles_raises_with_the_task_and_id(tmp_path):
    config = tiny_config(tmp_path)
    rows = fixture_rows()
    rows[1][1] = "C(C"
    write_csv(common.csv_path(config), rows)
    with pytest.raises(ValueError, match=r"npass_ptp1b NPC1001: RDKit cannot parse SMILES"):
        common.process_task(config)


def test_processing_writes_the_split_payloads_qc_and_provenance(built):
    config = built
    out = common.processed_dir(config)
    sizes = {s: len(processed(config, s)) for s in config["splits"]}
    assert sizes == config["expected_counts"] == {"train": 16, "valid": 2, "test": 2}
    rows = common.read_target_rows(config)
    split = common.split_indices(20, config["split"]["fractions"], config["split"]["seed"])
    for name in config["splits"]:
        loaded = processed(config, name)
        expected = [rows[i] for i in split[name]]
        assert loaded.ids == [f"ptp1b_{r['id']}" for r in expected] and loaded.smiles == [r["smiles"] for r in expected]
        assert np.allclose(loaded.labels[:, 0], [r["value"] for r in expected], atol=1e-5)
        assert loaded.labels.shape == (len(expected), 1) and loaded.label_mask.all()
        assert loaded.schema["format"] == "nafm_npass_v1" and loaded.schema["target_id"] == "NPT178"
        assert loaded.schema["split_seed"] == 0 and loaded.schema["rdkit"] and loaded.embeddable.all()
        summary = json.loads((out / f"qc_{name}.json").read_text())
        assert summary["molecules"] == summary["expected"] == len(expected) and summary["dropped"] == 0
        assert summary["conformer_generation"]["embedding"] == {"etkdg": len(expected), "etkdg_random_coords": 0,
                                                                "failed_2d_fallback": 0}
        assert sum(summary["conformer_generation"]["mmff"].values()) == len(expected)
        assert summary["split_rule"]["rows_of_target"] == 20 and summary["smiles_shared_with_other_split"] == 0
        assert summary["coordinates"]["declared_unit"] == "angstrom"
        assert len((out / f"qc_{name}.tsv").read_text().splitlines()) == len(expected) + 1
    ids = [i for s in config["splits"] for i in processed(config, s).ids]
    assert len(ids) == len(set(ids)) == 20                              # disjoint splits covering every molecule
    record = json.loads((common.task_paths(config)["provenance"]).read_text())["runs"][-1]
    assert record["kind"] == "process" and record["task"] == "npass_ptp1b" and record["policy"]["molecules_dropped"] == 0
    assert record["policy"]["split"]["seed"] == 0 and set(record["outputs_sha256"]) >= {"train.pt", "qc_test.tsv"}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        common.build_processed(config, splits=["train"], config_path=common.TASK_YAML)


def test_processing_twice_gives_identical_coordinates_and_validate_writes_nothing(tmp_path, built):
    other = tiny_config(tmp_path / "second")
    write_csv(common.csv_path(other), fixture_rows())
    assert common.build_processed(other, splits=["test"], config_path=common.TASK_YAML) == 0
    a, b = processed(built, "test"), processed(other, "test")
    assert a.ids == b.ids and np.array_equal(a._positions, b._positions)
    dry = tiny_config(tmp_path / "third")
    write_csv(common.csv_path(dry), fixture_rows())
    assert common.build_processed(dry, splits=["train"], config_path=common.TASK_YAML, dry_run=True) == 0
    assert not common.processed_dir(dry).exists()


def test_failed_embeddings_are_flagged_and_counted_never_dropped(tmp_path, monkeypatch):
    from rdkit.Chem import AllChem

    original = AllChem.EmbedMolecule

    def fail_for_chlorine(mol, *args, **kwargs):
        return -1 if any(a.GetSymbol() == "Cl" for a in mol.GetAtoms()) else original(mol, *args, **kwargs)

    monkeypatch.setattr(AllChem, "EmbedMolecule", fail_for_chlorine)
    config = tiny_config(tmp_path)
    test_row = int(common.split_indices(20, config["split"]["fractions"], config["split"]["seed"])["test"][0])
    write_csv(common.csv_path(config), fixture_rows(chlorine_at=test_row))      # the chlorine molecule is a test molecule
    assert common.build_processed(config, splits=list(config["splits"]), config_path=common.TASK_YAML) == 0
    failed_id = f"ptp1b_NPC{1000 + test_row}"
    loaded = processed(config, "test")
    index = loaded.ids.index(failed_id)
    assert sum(len(processed(config, s)) for s in config["splits"]) == 20           # nothing dropped
    assert loaded.flags["all_conformers_flat"][index] and loaded.flags["first_conformer_flat"][index]
    lo, hi = int(loaded._offsets[index]), int(loaded._offsets[index + 1])
    assert hi - lo == 5 and np.abs(loaded._positions[lo:hi, 2]).max() == 0 and np.isfinite(loaded._positions).all()
    summary = json.loads((common.processed_dir(config) / "qc_test.json").read_text())
    generation = summary["conformer_generation"]
    assert generation["embedding"]["failed_2d_fallback"] == 1 and generation["embedding_failed_ids"] == [failed_id]
    assert generation["mmff"]["skipped_2d_fallback"] == 1 and summary["flags"]["all_conformers_flat"] == 1
    tsv = (common.processed_dir(config) / "qc_test.tsv").read_text().splitlines()
    assert any(line.startswith(failed_id) and "all_conformers_flat" in line for line in tsv)
    subsets = molnet.test_subsets(config, loaded.chains)                         # reported next to the full test set
    assert subsets["all_conformers_flat"].tolist() == [i == index for i in range(len(loaded))]
    assert subsets["no_flags"].sum() == len(loaded) - 1


def test_a_retry_with_random_coordinates_is_counted(tmp_path, monkeypatch):
    from rdkit.Chem import AllChem

    original, calls = AllChem.EmbedMolecule, []

    def first_attempt_fails(mol, params):
        calls.append(bool(params.useRandomCoords))
        return -1 if not params.useRandomCoords else original(mol, params)

    monkeypatch.setattr(AllChem, "EmbedMolecule", first_attempt_fails)
    config = tiny_config(tmp_path)
    _, positions, _, status, mmff = common.generate_conformer("CCO", config=config, where="t")
    assert status == "etkdg_random_coords" and calls == [False, True] and np.abs(positions[:, 2]).max() > 0


# ---------------------------------------------------------------------------
# Processed split loader
# ---------------------------------------------------------------------------

def test_the_loader_refuses_a_file_of_another_target_revision_or_format(built):
    path = common.processed_dir(built) / "test.pt"
    payload = torch.load(path, weights_only=False)
    for key, value, message in (("target_id", "NPT204", "file holds target"), ("revision", "v0", "revision"),
                                ("format", "unimol_molnet_v1", "schema format")):
        broken = dict(payload, schema=dict(payload["schema"], **{key: value}))
        torch.save(broken, path)
        with pytest.raises(ValueError, match=message):
            common.NpassProcessedSplit(path)
    torch.save(payload, path)
    assert len(common.NpassProcessedSplit(path)) == 2


def test_a_clean_test_split_has_no_flagged_subset(built):
    subsets = molnet.test_subsets(built, processed(built, "test").chains)
    assert set(subsets) == {"no_flags"} and subsets["no_flags"].all()


# ---------------------------------------------------------------------------
# Download (pinned file, no network)
# ---------------------------------------------------------------------------

def _pin(config, data: bytes, rows: int):
    config["source_file"] = dict(config["source_file"], bytes=len(data), md5=hashlib.md5(data).hexdigest(),
                                 sha256=hashlib.sha256(data).hexdigest(), rows=rows)


def _fake_fetch(monkeypatch, data: bytes):
    calls = []

    def fetch(url, dest):
        calls.append(url)
        dest.write_bytes(data)
        return {"url": url, "path": str(dest), "sha256": hashlib.sha256(data).hexdigest(), "downloaded": True}

    monkeypatch.setattr(prepare, "_fetch", fetch)
    return calls


def test_download_verifies_size_checksums_and_rows(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    write_csv(tmp_path / "fixture.csv", fixture_rows())
    data = (tmp_path / "fixture.csv").read_bytes()
    _pin(config, data, rows=21)
    calls = _fake_fetch(monkeypatch, data)
    result = prepare.download(config)
    assert calls == [config["source_file"]["url"]] and result["rows"] == 21 and result["bytes"] == len(data)
    assert (common.task_paths(config)["raw"] / "regression_data.csv").read_bytes() == data


@pytest.mark.parametrize("field, value, message", [("bytes", 7, "bytes, pinned 7"), ("rows", 5, "data rows, pinned 5"),
                                                   ("md5", "0" * 32, "md5"), ("sha256", "0" * 64, "sha256")])
def test_download_refuses_any_pin_mismatch(tmp_path, monkeypatch, field, value, message):
    config = tiny_config(tmp_path)
    write_csv(tmp_path / "fixture.csv", fixture_rows())
    data = (tmp_path / "fixture.csv").read_bytes()
    _pin(config, data, rows=21)
    config["source_file"][field] = value
    _fake_fetch(monkeypatch, data)
    with pytest.raises(RuntimeError, match=message):
        prepare.download(config)


def test_download_never_fetches_a_file_that_is_already_in_place(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    write_csv(common.csv_path(config), fixture_rows())
    data = common.csv_path(config).read_bytes()
    _pin(config, data, rows=21)

    def no_network(*args, **kwargs):
        raise AssertionError("a file already in place must not be fetched")

    monkeypatch.setattr(ec_prepare.requests, "get", no_network)
    assert prepare.download(config)["file"]["downloaded"] is False


def test_prepare_cli_validates_without_writing_and_npass_alias_rejects_download(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MIND_DATA_ROOT", str(tmp_path))
    with pytest.raises(SystemExit):
        prepare.main(["download", "--task", "npass"])
    with pytest.raises(SystemExit):
        prepare.main(["process", "--task", "npass_unknown"])
    with pytest.raises(FileNotFoundError, match="ds-download"):
        prepare.main(["validate", "--task", "npass_cox1"])


# ---------------------------------------------------------------------------
# Fingerprint baseline and report on the family
# ---------------------------------------------------------------------------

def test_fingerprint_baseline_expands_npass_and_runs_end_to_end(tmp_path, monkeypatch):
    assert ecfp.expand_tasks("npass") == list(NPASS_TASKS) and ecfp.expand_tasks("molnet") == list(MOLNET_TASKS)
    assert ecfp.expand_tasks("all") == list(SMILES_TASKS)
    with pytest.raises(ValueError, match="MoleculeNet and NPASS tasks only"):
        ecfp.expand_tasks("npass_cox1,ec")
    data = tmp_path / "data"
    monkeypatch.setenv("MIND_DATA_ROOT", str(data))
    config = common.load_task_config(dataset="ptp1b")
    sizes = dict(zip(config["splits"], common.split_sizes(20, config["split"]["fractions"])))
    config.update(expected_molecules=20, paper_molecules=20, expected_counts=sizes)
    write_csv(common.csv_path(config), fixture_rows())
    assert common.build_processed(config, splits=list(config["splits"]), config_path=common.TASK_YAML) == 0
    assert ecfp.main(["--tasks", "npass_ptp1b", "--heads", "linear", "--seeds", "0", "--threads", "1", "--models", "probe",
                      "--outputs-root", str(tmp_path / "outputs"), "--results-root", str(tmp_path / "results"),
                      "--log-root", str(tmp_path / "logs"), "--no-ledger"]) == 0
    base = tmp_path / "outputs" / "downstream" / "npass_ptp1b" / "ecfp4"
    metrics = json.loads((base / "linear_probe.ecfp4_bits2048" / "seed0" / "metrics.json").read_text())
    assert metrics["task"] == "npass_ptp1b" and metrics["kind"] == "multitarget_regression"
    assert set(metrics["test"]["all"]) >= {"rmse", "mae"}
    markdown = report.to_markdown("npass_ptp1b", report.collect("npass_ptp1b", tmp_path / "outputs"))
    assert "NPASS (NaFM benchmark" in markdown and "MoleculeNet (Uni-Mol" not in markdown
    assert "MoleculeNet (Uni-Mol LMDB" in report.masked_markdown(report.collect("npass_ptp1b", tmp_path / "outputs"))

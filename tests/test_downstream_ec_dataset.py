"""EC task files, processing with QC, processed splits, and the probe/report CLIs on synthetic data."""

import gzip
import json

import numpy as np
import pytest

from downstream import probe, report
from downstream.tasks.ec import dataset
from downstream_helpers import chain_of, cif_text

TERMS = ["1.1.1.1", "2.7.11.1", "3.4.21.-"]
MASK = {"policy": "drop_residue_missing_backbone", "incomplete_sidechain": "mask"}


def _write_task(root, *, max_residues=20, length_policy="skip"):
    (root / "raw" / "structures").mkdir(parents=True)
    (root / "splits").mkdir()
    (root / "raw" / "annotations.tsv").write_text(
        "### EC-numbers\n" + "\t".join(TERMS) + "\n### PDB-chain\tEC-numbers\n"
        "1AAA-A\t1.1.1.1\n1AAA-B\t2.7.11.1,3.4.21.-\n1BBB-A\t1.1.1.1\n2AAA-A\t3.4.21.-\n"
        "3AAA-A\t1.1.1.1,2.7.11.1\n3AAA-B\t2.7.11.1\n")
    splits = {"train": ["1AAA-A", "1AAA-B", "1BBB-A"], "valid": ["2AAA-A"], "test": ["3AAA-A", "3AAA-B"]}
    for split, ids in splits.items():
        (root / "splits" / f"{split}.txt").write_text("\n".join(ids) + "\n")
    (root / "splits" / "test_cutoffs.csv").write_text(
        "PDB-chain,<30%,<40%,<50%,<70%,<95%\n3AAA-A,0,0,1,1,1\n3AAA-B,1,1,1,1,1\n")
    entries = {
        "1AAA": {"A": chain_of(["ALA", "SER", "GLY", "TRP"] * 3),
                 "B": chain_of(["LYS", "ALA"] * 5, start=20, drops={0: ("NZ",), 3: ("CA",)})},
        "2AAA": {"A": chain_of(["VAL", "GLU", "PHE"] * 4)},
        "3AAA": {"A": chain_of(["HIS", "CYS", "MET"] * 4), "B": chain_of(["ALA"] * 25, start=30)},
    }
    for entry, chains in entries.items():
        with gzip.open(root / "raw" / "structures" / f"{entry}.cif.gz", "wt") as handle:
            handle.write(cif_text(entry, chains))
    config = dataset.load_task_config(data_root=root)
    # The fixture writes the DeepFRI/RCSB layout, independent of the checked-in source.
    config.update(source="deepfri_rcsb", label_count=3, missing_atoms=MASK, max_residues=max_residues,
                  length_policy=length_policy)
    config["qc"]["max_failed_chain_fraction"] = 0.4
    return config


def test_annotation_split_and_cutoff_readers(tmp_path):
    config = _write_task(tmp_path)
    terms, labels = dataset.read_annotations(tmp_path / "raw" / "annotations.tsv", 3)
    assert terms == TERMS and labels["1AAA-B"] == [1, 2]
    with pytest.raises(ValueError, match="expected 538"):
        dataset.read_annotations(tmp_path / "raw" / "annotations.tsv", 538)
    subsets = dataset.read_test_cutoffs(tmp_path / "splits" / "test_cutoffs.csv", config["test_cutoffs"])
    assert subsets[30] == {"3AAA-B"} and subsets[95] == {"3AAA-A", "3AAA-B"}
    assert dataset.split_chain_id("6KMW-aL") == ("6KMW", "aL")
    with pytest.raises(ValueError):
        dataset.split_chain_id("6KMWaL")


def test_task_config_without_a_processed_revision_is_refused(tmp_path):
    import yaml

    config = yaml.safe_load(dataset.TASK_YAML.read_text())
    del config["processed_revision"]
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="processed_revision"):
        dataset.load_task_config(path)


def test_checked_in_yaml_marks_the_reprocessed_revision_and_names_the_exceptions():
    """v2 = chains with 2/3-character ids are read (46 were lost as parse_error); the old directory stays untouched."""
    config = dataset.load_task_config()
    assert config["processed_revision"] == "v2"
    assert dataset.processed_dir(config).name == "torchdrug_zip_dropbb_masksc_skip1024_v2"
    assert config["test_cutoff_unlisted_chains"] == ["7OA5-A"]


def test_build_processed_with_qc_report_and_failure_threshold(tmp_path):
    config = _write_task(tmp_path)
    assert dataset.build_processed(config, splits=["train", "valid", "test"], workers=1) == 1
    out = dataset.processed_dir(config)
    assert out.name == "deepfri_rcsb_dropbb_masksc_skip20_v2"
    # train: 1BBB file missing -> 1/3 failed (<= 0.4); test: 25-residue chain skipped -> 1/2 (> 0.4)
    assert (out / "train.pt").is_file() and (out / "valid.pt").is_file() and not (out / "test.pt").exists()
    qc_test = json.loads((out / "qc_test.json").read_text())
    assert qc_test["status"] == {"ok": 1, "dropped": 1} and qc_test["reasons"] == {"too_long": 1}
    rows = (out / "qc_train.tsv").read_text().splitlines()
    assert rows[0].split("\t")[:4] == ["chain", "split", "status", "reason"]
    assert any("1BBB-A\ttrain\tfailed\tstructure_missing" in row for row in rows)

    train = dataset.ProcessedSplit(out / "train.pt")
    assert train.chains == ["1AAA-A", "1AAA-B"]
    assert train.labels.tolist() == [[1, 0, 0], [0, 1, 1]]
    assert train.sequences[1] == "KAKKAKAKA"  # residue 3 (ALA, missing CA) dropped
    masked = train.payload["samples"][1].downstream_masked_block
    assert masked.numel() == 9 * 14 and bool(masked[:14].all()) and int(masked.sum()) == 14
    with pytest.raises(FileExistsError):
        dataset.build_processed(config, splits=["train"], workers=1)


def _cutoff_config(tmp_path, *, unlisted=()):
    config = _write_task(tmp_path)
    config["test_cutoff_unlisted_chains"] = list(unlisted)
    return config


def test_test_subsets_refuse_a_test_chain_that_the_cutoff_table_does_not_list(tmp_path):
    config = _cutoff_config(tmp_path)
    subsets = dataset.test_subsets(config, ["3AAA-A", "3AAA-B"])
    assert subsets[30].tolist() == [False, True] and subsets[95].tolist() == [True, True]
    with pytest.raises(ValueError, match=r"2 test chains are missing from test_cutoffs.csv: \['9ZZZ-A', '9ZZZ-B'\]"):
        dataset.test_subsets(config, ["3AAA-A", "9ZZZ-A", "9ZZZ-B"])


def test_documented_unlisted_chain_is_excluded_from_every_subset_with_a_warning(tmp_path):
    config = _cutoff_config(tmp_path, unlisted=["9ZZZ-A"])
    with pytest.warns(UserWarning, match=r"9ZZZ-A.*counted only in the full-test-set"):
        subsets = dataset.test_subsets(config, ["3AAA-A", "9ZZZ-A", "3AAA-B"])
    assert subsets[95].tolist() == [True, False, True] and subsets[30].tolist() == [False, False, True]
    with pytest.raises(ValueError, match=r"missing from test_cutoffs.csv: \['9ZZZ-B'\]"):
        dataset.test_subsets(config, ["9ZZZ-A", "9ZZZ-B"])  # only the documented chain is excused


def test_stale_exception_for_a_chain_the_table_now_lists_is_refused(tmp_path):
    config = _cutoff_config(tmp_path, unlisted=["3AAA-A"])
    with pytest.raises(ValueError, match=r"test_cutoff_unlisted_chains lists \['3AAA-A'\].*now in test_cutoffs.csv"):
        dataset.test_subsets(config, ["3AAA-A", "3AAA-B"])


def test_crop_windows_and_residue_weighted_chain_embeddings(tmp_path):
    config = _write_task(tmp_path, max_residues=10, length_policy="crop_windows")
    assert dataset.build_processed(config, splits=["test"], workers=1) == 0
    test = dataset.ProcessedSplit(dataset.processed_dir(config) / "test.pt")
    assert test.chains == ["3AAA-A", "3AAA-B"]
    assert test.sample_chain.tolist() == [0, 0, 1, 1, 1]
    assert test.sample_residues.tolist() == [6, 6, 8, 9, 8]
    embeddings = np.array([[1.0], [3.0], [0.0], [9.0], [0.0]], dtype=np.float32)
    np.testing.assert_allclose(test.chain_embeddings(embeddings), [[2.0], [81.0 / 25.0]], rtol=1e-6)


def test_probe_and_report_on_synthetic_embeddings(tmp_path):
    data_root = tmp_path / "data"
    _write_task(data_root)
    outputs = tmp_path / "outputs"
    rng = np.random.default_rng(0)
    chains = {"train": [f"T{i}" for i in range(40)], "valid": [f"V{i}" for i in range(10)], "test": ["3AAA-A", "3AAA-B"]}
    for split, ids in chains.items():
        labels = np.zeros((len(ids), 3), dtype=np.uint8)
        labels[np.arange(len(ids)), np.arange(len(ids)) % 3] = 1
        features = labels.astype(np.float32) * 2 + rng.normal(0, 0.3, labels.shape).astype(np.float32)
        path = outputs / "downstream" / "ec" / "toy" / "embeddings" / f"{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, chains=np.array(ids), labels=labels, emb_mean_atoms=features)
        path.with_suffix(".json").write_text(json.dumps({"split": split}))
    args = ["--task", "ec", "--model-id", "toy", "--head", "linear", "--seeds", "0", "1",
            "--data-root", str(data_root), "--outputs-root", str(outputs)]
    assert probe.main(args) == 0
    metrics = json.loads((outputs / "downstream" / "ec" / "toy" / "linear_probe.mean_atoms" / "seed0" / "metrics.json").read_text())
    assert metrics["val"]["fmax"] > 0.6  # constant scores give F = 0.5 (all three terms predicted)
    assert metrics["test"]["cutoff_30"]["num_proteins"] == 1 and metrics["test"]["all"]["num_proteins"] == 2
    with pytest.raises(FileExistsError):
        probe.main(args)
    rows = report.collect("ec", outputs)
    assert len(rows) == 1 and rows[0]["seeds"] == [0, 1] and rows[0]["test_fmax"]["n"] == 2
    markdown = report.to_markdown("ec", rows)
    assert "SaProt" in markdown and "literature, fine-tuned, different protocol" in markdown

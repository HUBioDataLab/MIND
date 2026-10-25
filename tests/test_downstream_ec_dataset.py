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
    config.update(label_count=3, missing_atoms=MASK, max_residues=max_residues, length_policy=length_policy)
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


def test_build_processed_with_qc_report_and_failure_threshold(tmp_path):
    config = _write_task(tmp_path)
    assert dataset.build_processed(config, splits=["train", "valid", "test"], workers=1) == 1
    out = dataset.processed_dir(config)
    assert out.name == "deepfri_rcsb_dropbb_masksc_skip20"
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

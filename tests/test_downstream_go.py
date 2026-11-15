"""GO task family (go_mf / go_bp / go_cc): label/split contracts, processing, branch views, CLI wiring.

Synthetic inputs only (no archive, no network): structures are built with ``downstream_helpers``
(production atom names and slot layout), annotation/split files follow the DeepFRI formats.
"""

import json

import numpy as np
import pytest

from downstream import probe
from downstream.common.metrics import protein_fmax
from downstream.tasks import TASKS, load_task
from downstream.tasks.go import bp, cc, common, mf
from downstream_helpers import chain_of, pdb_text

MF = [f"GO:000{i:04d}" for i in range(1, 4)]
BP = [f"GO:001{i:04d}" for i in range(1, 5)]
CC = [f"GO:002{i:04d}" for i in range(1, 3)]
COUNTS = {"mf": 3, "bp": 4, "cc": 2}


def _annotation_text(rows, *, mf_terms=MF, bp_terms=BP, cc_terms=CC):
    def block(ontology, terms):
        return (f"### GO-terms ({ontology})\n" + "\t".join(terms) + f"\n### GO-names ({ontology})\n"
                + "\t".join(f"name {t}" for t in terms) + "\n")
    return (block("molecular_function", mf_terms) + block("biological_process", bp_terms)
            + block("cellular_component", cc_terms)
            + "### PDB-chain\tmolecular_function\tbiological_process\tcellular_component\n" + rows)


ROWS = (
    "1AAA-A\tGO:0000001,GO:0000002\tGO:0010001\tGO:0020001\n"
    "1AAA-B\tGO:0000003\t\tGO:0020002\n"          # no BP term
    "1BBB-A\t\tGO:0010002,GO:0010004\t\n"          # no MF and no CC term
    "2AAA-A\tGO:0000001\tGO:0010003\tGO:0020001\n"
    "3AAA-A\tGO:0000002\tGO:0010001\n"             # trailing cell trimmed
    "3AAA-B\tGO:0000003\tGO:0010004\tGO:0020002\n"
)


def _write_family(tmp_path, *, rows=ROWS):
    """Fixture in the shared-source layout: <tmp>/deepfri/raw/GeneOntology and <tmp>/deepfri/go/splits."""
    source = tmp_path / "deepfri"
    raw = source / "raw" / "GeneOntology"
    root = source / "go"
    (root / "splits").mkdir(parents=True)
    raw.mkdir(parents=True)
    (raw / "nrPDB-GO_annot.tsv").write_text(_annotation_text(rows))
    splits = {"train": ["1AAA-A", "1AAA-B", "1BBB-A"], "valid": ["2AAA-A"], "test": ["3AAA-A", "3AAA-B"]}
    for split, ids in splits.items():
        (root / "splits" / f"{split}.txt").write_text("\n".join(ids) + "\n")
    (root / "splits" / "test_cutoffs.csv").write_text(
        "PDB-chain,<30%,<40%,<50%,<70%,<95%\n3AAA-A,0,0,1,1,1\n3AAA-B,1,1,1,1,1\n")
    structures = {
        "1AAA-A": chain_of(["ALA", "SER", "GLY", "TRP"] * 3),
        "1AAA-B": chain_of(["LYS", "ALA"] * 5, start=20, drops={0: ("NZ",), 3: ("CA",)}),
        "1BBB-A": chain_of(["VAL", "GLU", "PHE"] * 4),
        "2AAA-A": chain_of(["HIS", "CYS", "MET"] * 4),
        "3AAA-A": chain_of(["ALA", "GLY"] * 6),
        "3AAA-B": chain_of(["SER", "THR"] * 6),
    }
    index = {}
    for chain, residues in structures.items():
        path = raw / f"{chain}_x.pdb"
        path.write_text(pdb_text({"A": residues}))
        index[chain] = str(path)
    (source / "raw" / "GeneOntology_file_index.json").write_text(json.dumps(index))
    config = common.load_task_config(data_root=root, branch="mf")
    for branch, count in COUNTS.items():
        config["branches"][branch]["label_count"] = count
    config["expected_chain_counts"] = {"train": 3, "valid": 1, "test": 2}
    return config


# ---------------------------------------------------------------------------
# Checked-in config and registry
# ---------------------------------------------------------------------------

def test_checked_in_yaml_loads_for_three_branches_and_registry_exposes_them():
    expected = {"mf": ("go_mf", 489), "bp": ("go_bp", 1943), "cc": ("go_cc", 320)}
    for branch, (task, count) in expected.items():
        config = common.load_task_config(branch=branch)
        assert (config["task"], config["label_count"], config["branch"]) == (task, count, branch)
        assert config["missing_atoms"]["policy"] != "UNDECIDED"
        assert task in TASKS
        module = load_task(task)
        assert module.KIND == "multilabel" and module.BRANCH == branch
        assert module.load_task_config()["label_count"] == count
    assert config["sources"]["torchprotein_go_zip"]["md5"] == "376be1f088cd1fe720e1eaafb701b5cb"
    assert config["data_root"].endswith("/deepfri/go") and config["source_root"].endswith("/deepfri")


def test_config_loader_rejects_bad_decisions(tmp_path):
    import yaml

    base = yaml.safe_load(common.TASK_YAML.read_text())

    def write(**overrides):
        path = tmp_path / "task.yaml"
        path.write_text(yaml.safe_dump({**base, **overrides}))
        return path

    with pytest.raises(ValueError, match="unknown GO branch"):
        common.load_task_config(branch="xx")
    with pytest.raises(ValueError, match="keep_zero_rows is not implemented"):
        common.load_task_config(write(unannotated_policy="keep_zero_rows"), branch="mf")
    with pytest.raises(Exception, match="UNDECIDED|policy"):
        common.load_task_config(write(missing_atoms={"policy": "UNDECIDED"}), branch="mf")
    with pytest.raises(ValueError, match="test_cutoffs must include 30"):
        common.load_task_config(write(test_cutoffs=[40, 95]), branch="mf")
    broken = dict(base["branches"], cc={"task": "go_cc", "ontology": "biological_process", "label_count": 320})
    with pytest.raises(ValueError, match="cellular_component"):
        common.load_task_config(write(branches=broken), branch="mf")
    stripped = {k: v for k, v in base.items() if k != "missing_atoms"}
    path = tmp_path / "stripped.yaml"
    path.write_text(yaml.safe_dump(stripped))
    with pytest.raises(ValueError, match="missing required keys"):
        common.load_task_config(path, branch="mf")


# ---------------------------------------------------------------------------
# Annotation file contract
# ---------------------------------------------------------------------------

def test_annotation_reader_parses_branches_empty_cells_and_trimmed_rows(tmp_path):
    path = tmp_path / "annot.tsv"
    path.write_text(_annotation_text(ROWS))
    terms, labels = common.read_go_annotations(path, COUNTS)
    assert terms == {"mf": MF, "bp": BP, "cc": CC}
    assert labels["1AAA-A"] == {"mf": [0, 1], "bp": [0], "cc": [0]}
    assert labels["1AAA-B"]["bp"] == [] and labels["1BBB-A"]["mf"] == [] and labels["1BBB-A"]["cc"] == []
    assert labels["3AAA-A"] == {"mf": [1], "bp": [0], "cc": []}
    matrix = common.multi_hot_branch(labels, ["1AAA-A", "1BBB-A"], "bp", 4)
    assert matrix.tolist() == [[1, 0, 0, 0], [0, 1, 0, 1]]
    with pytest.raises(KeyError, match="no GO annotation row"):
        common.multi_hot_branch(labels, ["9ZZZ-A"], "bp", 4)


@pytest.mark.parametrize("mutate, message", [
    (lambda t: t.replace("### GO-terms (molecular_function)", "### GO (molecular_function)"), "expected '### GO-terms"),
    (lambda t: t.replace("### GO-names (biological_process)", "### names"), "expected '### GO-names"),
    (lambda t: t.replace("GO:0000001\tGO:0000002\tGO:0000003", "GO:0000001\tGO:0000002"), "has 2 terms, expected 3"),
    (lambda t: t.replace("GO:0000002\tGO:0000003\n###", "GO:0000002\tGO:000003\n###"), "malformed GO ids"),
    (lambda t: t.replace("1AAA-B\t", "1AAA-A\t", 1), "duplicate chain"),
    (lambda t: t.replace("1AAA-A\tGO:0000001,GO:0000002\tGO:0010001", "1AAA-A\tGO:0000001,GO:0010001\tGO:0010001"),
     "outside the molecular_function list"),
    (lambda t: t.replace("GO:0020002\n1BBB-A", "GO:0020002\textra\n1BBB-A"), "columns, expected at most 4"),
    (lambda t: t.replace("1BBB-A", "1BBBA"), "not <PDB>-<chain>"),
    (lambda t: t.split("### PDB-chain")[0], "'### PDB-chain' header"),
    (lambda t: t.split("### GO-terms (cellular_component)")[0], "ends inside the cellular_component block"),
])
def test_annotation_reader_fails_loud(tmp_path, mutate, message):
    text = mutate(_annotation_text(ROWS))
    path = tmp_path / "annot.tsv"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        common.read_go_annotations(path, COUNTS)


def test_annotation_reader_rejects_duplicate_terms_and_name_mismatch(tmp_path):
    path = tmp_path / "annot.tsv"
    path.write_text(_annotation_text(ROWS, bp_terms=BP[:3] + [BP[0]]))
    with pytest.raises(ValueError, match="duplicate GO terms"):
        common.read_go_annotations(path, COUNTS)
    text = _annotation_text(ROWS).replace("\tname GO:0000003", "", 1)
    path.write_text(text)
    with pytest.raises(ValueError, match="names"):
        common.read_go_annotations(path, COUNTS)


def test_validate_inputs_counts_overlaps_and_missing_rows(tmp_path):
    path = tmp_path / "annot.tsv"
    path.write_text(_annotation_text(ROWS))
    _, labels = common.read_go_annotations(path, COUNTS)
    config = {"expected_chain_counts": {"train": 3, "valid": 1, "test": 2}}
    splits = {"train": ["1AAA-A", "1AAA-B", "1BBB-A"], "valid": ["2AAA-A"], "test": ["3AAA-A", "3AAA-B"]}
    result = common.validate_inputs(config, splits, labels)
    assert result["chains_without_any_term"] == 0
    assert result["annotated_per_branch"]["bp"] == {"train": 2, "valid": 1, "test": 2}
    with pytest.raises(ValueError, match="differ from expected"):
        common.validate_inputs(config, {**splits, "valid": ["2AAA-A", "1AAA-A"]}, labels)
    with pytest.raises(ValueError, match="in both train and test"):
        common.validate_inputs(config, {**splits, "test": ["1AAA-A", "3AAA-B"]}, labels)
    with pytest.raises(ValueError, match="no annotation row"):
        common.validate_inputs(config, {**splits, "valid": ["9ZZZ-A"]}, labels)


# ---------------------------------------------------------------------------
# Processing, missing-atom policy, branch views
# ---------------------------------------------------------------------------

def test_processing_applies_missing_atom_policy_and_branch_views_exclude_unannotated(tmp_path):
    config = _write_family(tmp_path)
    assert common.build_processed(config, splits=["train", "valid", "test"], workers=1) == 0
    out = common.processed_dir(config)
    assert out.name == "torchprotein_go_zip_dropbb_masksc_skip1024_v3"
    qc = json.loads((out / "qc_train.json").read_text())
    assert qc["status"] == {"ok": 3} and qc["annotated_per_branch"] == {"mf": 2, "bp": 2, "cc": 2}
    assert qc["masked_residues"] == 1 and qc["residues_missing_backbone"] == 1

    # One shared file holds all three label matrices over the same chains.
    train_mf = mf.ProcessedSplit(out / "train.pt")
    shared = train_mf.payload  # view payload only; raw file checked below
    import torch
    raw_payload = torch.load(out / "train.pt", weights_only=False)
    assert sorted(raw_payload["labels"]) == ["bp", "cc", "mf"] and raw_payload["chains"] == ["1AAA-A", "1AAA-B", "1BBB-A"]
    assert raw_payload["labels"]["bp"].tolist() == [[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 1]]

    # Missing atoms never reach the V-shifting adapter: residue 3 of 1AAA-B (no CA) is dropped, residue 0 (no NZ)
    # keeps its backbone and its whole 14-slot block is masked.
    assert raw_payload["sequences"][1] == "KAKKAKAKA"
    masked = raw_payload["samples"][1].downstream_masked_block
    assert masked.numel() == 9 * 14 and bool(masked[:14].all()) and int(masked.sum()) == 14

    # Branch views drop chains without a term in that branch, and nothing else.
    assert train_mf.chains == ["1AAA-A", "1AAA-B"] and train_mf.report["excluded_unannotated_chains"] == 1
    train_bp, train_cc = bp.ProcessedSplit(out / "train.pt"), cc.ProcessedSplit(out / "train.pt")
    assert train_bp.chains == ["1AAA-A", "1BBB-A"] and train_cc.chains == ["1AAA-A", "1AAA-B"]
    assert train_bp.labels.tolist() == [[1, 0, 0, 0], [0, 1, 0, 1]]
    assert train_bp.labels.shape[1] == 4 and train_cc.labels.shape[1] == 2
    for view in (train_mf, train_bp, train_cc):
        assert (view.labels.sum(axis=1) > 0).all()  # CAFA Fmax contract
        assert len(view) == len(view.sample_chain) == len(view.sample_residues)
        assert int(view.sample_chain.max()) == len(view.chains) - 1
    # BP view sample 1 belongs to chain 1BBB-A (view row 1), not to the dropped 1AAA-B
    assert train_bp.sample_chain.tolist() == [0, 1]
    assert train_bp.sequences == [raw_payload["sequences"][0], raw_payload["sequences"][2]]
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(train_bp.chain_embeddings(embeddings), embeddings)
    with pytest.raises(ValueError, match="expected 2 sample embeddings"):
        train_bp.chain_embeddings(embeddings[:1])
    with pytest.raises(FileExistsError):
        common.build_processed(config, splits=["train"], workers=1)


def test_processing_fails_loud_on_missing_structure_or_annotation(tmp_path):
    config = _write_family(tmp_path)
    index_path = common.file_index_path(config)
    index = json.loads(index_path.read_text())
    del index["3AAA-B"]
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="no structure file, first \\['3AAA-B'\\]"):
        common.build_processed(config, splits=["test"], workers=1)
    config = _write_family(tmp_path / "second", rows=ROWS.replace("3AAA-B", "3AAA-Z"))
    with pytest.raises(ValueError, match="no annotation row"):
        common.build_processed(config, splits=["test"], workers=1)


def test_qc_threshold_blocks_the_processed_file(tmp_path):
    config = _write_family(tmp_path)
    config["qc"]["max_failed_chain_fraction"] = 0.2
    (tmp_path / "deepfri" / "raw" / "GeneOntology" / "3AAA-B_x.pdb").unlink()
    assert common.build_processed(config, splits=["test"], workers=1) == 1
    out = common.processed_dir(config)
    assert not (out / "test.pt").exists()
    assert json.loads((out / "qc_test.json").read_text())["reasons"] == {"structure_missing": 1}


def test_processed_split_rejects_corrupt_payloads(tmp_path):
    import torch

    config = _write_family(tmp_path)
    assert common.build_processed(config, splits=["valid"], workers=1) == 0
    path = common.processed_dir(config) / "valid.pt"
    payload = torch.load(path, weights_only=False)
    cases = {
        "lacks 'terms'": {k: v for k, v in payload.items() if k != "terms"},
        "labels must hold exactly": {**payload, "labels": {"mf": payload["labels"]["mf"]}},
        "label matrix": {**payload, "labels": {**payload["labels"], "bp": payload["labels"]["bp"][:, :2]}},
        "must be 0/1": {**payload, "labels": {**payload["labels"], "mf": payload["labels"]["mf"] * 2}},
        "inconsistent chain/sample": {**payload, "sample_chain": payload["sample_chain"][:0]},
    }
    for message, broken in cases.items():
        target = tmp_path / "broken.pt"
        torch.save(broken, target)
        with pytest.raises(ValueError, match=message):
            mf.ProcessedSplit(target)
    empty = {**payload, "labels": {b: m * 0 for b, m in payload["labels"].items()}}
    torch.save(empty, tmp_path / "empty.pt")
    with pytest.raises(ValueError, match="refusing an empty branch view"):
        cc.ProcessedSplit(tmp_path / "empty.pt")
    with pytest.raises(ValueError, match="unknown GO branch"):
        common.GoProcessedSplit(path, branch="zz")


# ---------------------------------------------------------------------------
# Test-cutoff subsets and metric contract
# ---------------------------------------------------------------------------

def test_test_subsets_cover_95_and_30_and_refuse_unlisted_chains(tmp_path):
    config = _write_family(tmp_path)
    subsets = common.test_subsets(config, ["3AAA-A", "3AAA-B"])
    assert subsets[95].tolist() == [True, True] and subsets[30].tolist() == [False, True]
    assert subsets[50].tolist() == [True, True]
    with pytest.raises(ValueError, match="missing from test_cutoffs.csv, first \\['9ZZZ-A'\\]"):
        common.test_subsets(config, ["3AAA-A", "9ZZZ-A"])
    (tmp_path / "deepfri" / "go" / "splits" / "test_cutoffs.csv").write_text("PDB-chain,<30%,<95%\n3AAA-A,0,1\n")
    with pytest.raises(ValueError, match="no columns for cutoffs"):
        common.test_subsets(config, ["3AAA-A"])


def test_fmax_rejects_unannotated_rows_so_views_must_exclude_them():
    labels = np.array([[1, 0], [0, 0]])
    with pytest.raises(ValueError, match="no true term"):
        protein_fmax(labels, np.array([[0.9, 0.1], [0.1, 0.2]]))
    perfect = protein_fmax(labels[:1], np.array([[0.9, 0.1]]))
    assert perfect["fmax"] == pytest.approx(1.0)


def test_generic_probe_cli_runs_a_go_branch_on_synthetic_embeddings(tmp_path):
    config = _write_family(tmp_path)
    data_root = tmp_path / "deepfri" / "go"
    outputs = tmp_path / "outputs"
    rng = np.random.default_rng(0)
    chains = {"train": [f"T{i}" for i in range(40)], "valid": [f"V{i}" for i in range(10)], "test": ["3AAA-A", "3AAA-B"]}
    for split, ids in chains.items():
        labels = np.zeros((len(ids), 3), dtype=np.uint8)
        labels[np.arange(len(ids)), np.arange(len(ids)) % 3] = 1
        features = labels.astype(np.float32) * 2 + rng.normal(0, 0.3, labels.shape).astype(np.float32)
        path = outputs / "downstream" / "go_mf" / "toy" / "embeddings" / f"{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, chains=np.array(ids), labels=labels, emb_mean_atoms=features)
        path.with_suffix(".json").write_text(json.dumps({"split": split}))
    args = ["--task", "go_mf", "--model-id", "toy", "--head", "linear", "--seeds", "0",
            "--data-root", str(data_root), "--outputs-root", str(outputs), "--threads", "1"]
    assert probe.main(args) == 0
    metrics = json.loads((outputs / "downstream" / "go_mf" / "toy" / "linear_probe.mean_atoms" / "seed0"
                          / "metrics.json").read_text())
    assert metrics["task"] == "go_mf" and metrics["test"]["all"]["num_proteins"] == 2
    assert metrics["test"]["cutoff_30"]["num_proteins"] == 1 and config["task"] == "go_mf"


def test_two_character_chain_ids_are_processed_not_dropped_as_parse_errors(tmp_path):
    """3J9M-AK-style files (2-character chain id, shifted columns) used to fail as parse_error (1,171 GO chains)."""
    config = _write_family(tmp_path)
    raw = tmp_path / "deepfri" / "raw" / "GeneOntology"
    (raw / "3AAA-B_x.pdb").write_text(pdb_text({"AK": chain_of(["SER", "THR"] * 6)}))
    (raw / "1AAA-B_x.pdb").write_text(pdb_text({"S2": chain_of(["LYS", "ALA"] * 5, start=20,
                                                               drops={0: ("NZ",), 3: ("CA",)})}))
    assert common.build_processed(config, splits=["train", "valid", "test"], workers=1) == 0
    out = common.processed_dir(config)
    for split, expected in (("train", {"ok": 3}), ("valid", {"ok": 1}), ("test", {"ok": 2})):
        assert json.loads((out / f"qc_{split}.json").read_text())["status"] == expected
    import torch
    train = torch.load(out / "train.pt", weights_only=False)
    test = torch.load(out / "test.pt", weights_only=False)
    assert train["chains"] == ["1AAA-A", "1AAA-B", "1BBB-A"] and test["chains"] == ["3AAA-A", "3AAA-B"]
    assert test["sequences"][1] == "ST" * 6
    assert train["sequences"][1] == "KAKKAKAKA"       # same policy result as the single-character fixture

"""Leakage registry contract: reference formats, namespaces, seen/unseen reports, provenance output."""

import csv
import json

import pytest

from downstream.common import leakage


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


@pytest.fixture
def protein_plan(tmp_path):
    path = _write_csv(tmp_path / "protein_plan.csv", ["id", "split", "chunk", "position"], [
        ("P12345", "train", 0, 0), ("Q9Y2X3", "train", 0, 1), ("A0A6D0ENQ7", "val", 1, 0), ("O00123", "test", 2, 0)])
    return leakage.load_reference("protein_test", {
        "kind": "plan", "path": path, "namespace": "uniprot", "id_column": "id", "split_column": "split"})


@pytest.fixture
def rna_plan(tmp_path):
    path = _write_csv(tmp_path / "rna_plan.csv", ["id", "split", "chunk", "position"], [
        ("1abc_A", "train", 0, 0), ("1abc_B", "train", 0, 1), ("2xyz_AA", "val", 1, 0), ("3def_R", "test", 2, 0)])
    return leakage.load_reference("rna_test", {
        "kind": "plan", "path": path, "namespace": "pdb_chain", "id_column": "id", "split_column": "split"})


def test_normalization_per_namespace():
    assert leakage.normalize_id(" p12345 ", "uniprot") == "P12345"
    assert leakage.normalize_id("AF-A0A6D0ENQ7-F1-model_v4", "uniprot") == "A0A6D0ENQ7"
    assert leakage.normalize_id("1ABC", "pdb_entry") == "1abc"
    assert leakage.normalize_id("1ABC-A", "pdb_chain") == "1abc_A"
    assert leakage.normalize_id("6kmw_aL", "pdb_chain") == "6kmw_aL"  # chain case is meaningful
    for raw, namespace in [("", "uniprot"), ("12", "uniprot"), ("1ab", "pdb_entry"), ("1abc", "pdb_chain"),
                           ("1abc_", "pdb_chain"), ("1abc_A_B", "pdb_chain"), (None, "uniprot")]:
        with pytest.raises(leakage.LeakageError):
            leakage.normalize_id(raw, namespace)
    with pytest.raises(leakage.LeakageError, match="unknown namespace"):
        leakage.normalize_id("P12345", "ensembl")


def test_reference_loading_is_strict(tmp_path):
    spec = {"kind": "plan", "namespace": "uniprot", "id_column": "id", "split_column": "split"}
    good = _write_csv(tmp_path / "ok.csv", ["id", "split"], [("P12345", "train"), ("Q9Y2X3", "val")])
    reference = leakage.load_reference("ok", {**spec, "path": good})
    assert reference.count() == 2 and reference.has_splits and len(reference.sha256) == 64
    cases = {
        "unknown pretraining split": [("P12345", "dev")],
        "duplicate id": [("P12345", "train"), ("P12345", "val")],
        "is not a UniProt accession": [("not-an-id", "train")],
        "reference is empty": [],
    }
    for message, rows in cases.items():
        path = _write_csv(tmp_path / "bad.csv", ["id", "split"], rows)
        with pytest.raises(leakage.LeakageError, match=message):
            leakage.load_reference("bad", {**spec, "path": path})
    path = _write_csv(tmp_path / "nocol.csv", ["name", "split"], [("P12345", "train")])
    with pytest.raises(leakage.LeakageError, match="missing columns"):
        leakage.load_reference("bad", {**spec, "path": path})
    with pytest.raises(FileNotFoundError):
        leakage.load_reference("bad", {**spec, "path": tmp_path / "absent.csv"})
    with pytest.raises(leakage.LeakageError, match="lacks 'kind'"):
        leakage.load_reference("bad", {"path": good, "namespace": "uniprot", "id_column": "id"})
    with pytest.raises(leakage.LeakageError, match="needs split_column"):
        leakage.load_reference("bad", {"kind": "plan", "path": good, "namespace": "uniprot", "id_column": "id"})
    manifest = leakage.load_reference("m", {"kind": "manifest", "path": good, "namespace": "uniprot", "id_column": "id"})
    assert not manifest.has_splits and manifest.ids_by_split["all"] == {"P12345", "Q9Y2X3"}


def test_uniprot_query_reports_seen_and_unseen_separately(protein_plan):
    report = leakage.compare("test", ["P12345", "a0a6d0enq7", "O00123", "P99999"], "uniprot", protein_plan)
    assert report.method == "direct" and report.total == 4
    assert report.status_by_id == {"P12345": "seen_train", "A0A6D0ENQ7": "seen_val",
                                   "O00123": "seen_test", "P99999": "unseen"}
    assert (report.seen, report.touched, report.counts["unseen"]) == (1, 2, 1)
    assert sum(report.counts.values()) == report.total


def test_manifest_without_splits_counts_members_as_seen_train(tmp_path):
    path = _write_csv(tmp_path / "m.csv", ["repId"], [("P12345",), ("Q9Y2X3",)])
    reference = leakage.load_reference("m", {"kind": "manifest", "path": path, "namespace": "uniprot", "id_column": "repId"})
    report = leakage.compare("valid", ["P12345", "O00123"], "uniprot", reference)
    assert report.counts == {"seen_train": 1, "unseen": 1}


def test_pdb_query_against_uniprot_reference_is_refused_not_reported_as_unseen(protein_plan):
    with pytest.raises(leakage.NamespaceMismatchError, match="refusing to report 0 leaked"):
        leakage.compare("test", ["1ABC-A"], "pdb_chain", protein_plan)
    with pytest.raises(leakage.NamespaceMismatchError, match="cannot compare"):
        leakage.compare("test", ["P12345"], "uniprot", leakage.ReferenceSet(
            "r", "pdb_chain", "p", "h", {"train": frozenset({"1abc_A"})}, True))


def test_id_map_bridges_pdb_to_uniprot_and_reports_unmapped(protein_plan, tmp_path):
    map_path = tmp_path / "sifts.tsv"
    map_path.write_text("PDB_CHAIN\tUNIPROT\n1ABC-A\tP12345\n2XYZ-B\tP99999\n")
    id_map = leakage.read_id_map(map_path)
    assert id_map == {"1abc_A": "P12345", "2xyz_B": "P99999"}
    report = leakage.compare("test", ["1ABC-A", "2XYZ-B", "3DEF-C"], "pdb_chain", protein_plan, id_map=id_map)
    assert report.method == "id_map"
    assert report.status_by_id == {"1abc_A": "seen_train", "2xyz_B": "unseen", "3def_C": "unmapped"}
    with pytest.raises(leakage.NamespaceMismatchError, match="per pdb_chain"):
        leakage.compare("test", ["1ABC"], "pdb_entry", protein_plan, id_map=id_map)
    map_path.write_text("1ABC-A\tP12345\n1ABC-A\tQ9Y2X3\n")
    with pytest.raises(leakage.LeakageError, match="maps to both"):
        leakage.read_id_map(map_path)
    map_path.write_text("")
    with pytest.raises(leakage.LeakageError, match="id map is empty"):
        leakage.read_id_map(map_path)


def test_rna_comparison_is_entry_level_with_chain_detail(rna_plan):
    # 1abc_C is a new chain of a seen entry; 2xyz_AA is in the pretraining val split; 9zzz is unseen.
    report = leakage.compare("test", ["1ABC-A", "1abc_C", "2xyz_AA", "9zzz_A"], "pdb_chain", rna_plan)
    assert report.method == "entry_level"
    assert report.status_by_id == {"1abc_A": "seen_train", "1abc_C": "seen_train",
                                   "2xyz_AA": "seen_val", "9zzz_A": "unseen"}
    assert report.chain_level_seen == 1  # only 1abc_A matches a pretraining chain exactly
    entries = leakage.compare("test", ["1ABC", "3def", "4ghi"], "pdb_entry", rna_plan)
    assert entries.counts == {"seen_train": 1, "seen_test": 1, "unseen": 1} and entries.chain_level_seen is None


def test_query_validation_is_loud(protein_plan, rna_plan):
    with pytest.raises(leakage.LeakageError, match="has no ids"):
        leakage.compare("empty", [], "uniprot", protein_plan)
    with pytest.raises(leakage.LeakageError, match="duplicate ids"):
        leakage.compare("dup", ["P12345", "p12345"], "uniprot", protein_plan)
    with pytest.raises(leakage.LeakageError, match="id #2"):
        leakage.compare("bad", ["P12345", "bogus"], "uniprot", protein_plan)
    with pytest.raises(leakage.LeakageError, match="unknown query namespace"):
        leakage.compare("bad", ["P12345"], "sra", protein_plan)
    with pytest.raises(leakage.NamespaceMismatchError):
        leakage.compare("rna", ["1abc_A"], "pdb_chain", leakage.ReferenceSet(
            "u", "uniprot", "p", "h", {"all": frozenset({"P12345"})}, False))
    with pytest.raises(leakage.NamespaceMismatchError):
        leakage.compare("prot", ["P12345"], "uniprot", rna_plan)


def test_write_report_appends_to_provenance_and_never_overwrites(rna_plan, tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    provenance = task_dir / "provenance.json"
    provenance.write_text(json.dumps({"runs": [{"kind": "download"}]}))
    test_ids = task_dir / "test.txt"
    test_ids.write_text("1ABC-A\n9zzz_A\n")
    valid_ids = task_dir / "valid.txt"
    valid_ids.write_text("2xyz_AA\n")
    reports = [leakage.compare("test", leakage.read_id_list(test_ids), "pdb_chain", rna_plan),
               leakage.compare("valid", leakage.read_id_list(valid_ids), "pdb_chain", rna_plan)]
    table = leakage.write_report("rna_site", reports, rna_plan, provenance,
                                 query_files={"test": test_ids, "valid": valid_ids})
    stored = json.loads(provenance.read_text())
    assert stored["runs"] == [{"kind": "download"}]  # existing keys preserved
    record = stored["leakage"][0]
    assert record["reference"]["sha256"] == rna_plan.sha256 and record["reference"]["ids"] == 4
    assert record["queries"]["test"]["path"] == str(test_ids) and len(record["queries"]["test"]["sha256"]) == 64
    by_query = {r["query"]: r for r in record["results"]}
    assert by_query["test"]["seen_train"] == 1 and by_query["test"]["unseen"] == 1
    assert by_query["valid"]["touched_by_train_or_val"] == 1 and by_query["valid"]["seen_train"] == 0
    assert "homologous" in record["caveat"]
    rows = [line.split("\t") for line in table.read_text().splitlines()]
    assert rows[0] == ["query", "id", "status"] and ["test", "9zzz_A", "unseen"] in rows
    with pytest.raises(FileExistsError):
        leakage.write_report("rna_site", reports, rna_plan, provenance, query_files={"test": test_ids})
    provenance.write_text("[1, 2]")
    with pytest.raises(leakage.LeakageError, match="expected a JSON object"):
        leakage.write_report("rna_site2", reports, rna_plan, provenance, query_files={"test": test_ids})


def test_cli_end_to_end_and_unknown_reference(tmp_path, capsys):
    plan = _write_csv(tmp_path / "plan.csv", ["id", "split", "chunk", "position"],
                      [("P12345", "train", 0, 0), ("Q9Y2X3", "val", 0, 1)])
    config = tmp_path / "refs.yaml"
    config.write_text(f"references:\n  toy:\n    kind: plan\n    path: {plan}\n    namespace: uniprot\n"
                      "    id_column: id\n    split_column: split\n")
    ids = tmp_path / "test.txt"
    ids.write_text("P12345\nO00123\n")
    provenance = tmp_path / "out" / "provenance.json"
    provenance.parent.mkdir()
    args = ["--task", "toy_task", "--reference", "toy", "--references-yaml", str(config), "--query", f"test={ids}",
            "--query-namespace", "uniprot", "--provenance", str(provenance)]
    assert leakage.main(args) == 0
    out = capsys.readouterr().out
    assert "seen_train 1" in out and "unseen 1" in out
    assert json.loads(provenance.read_text())["leakage"][0]["results"][0]["total"] == 2
    with pytest.raises(SystemExit):
        leakage.main(["--task", "t", "--reference", "nope", "--references-yaml", str(config), "--query", f"test={ids}",
                      "--query-namespace", "uniprot", "--provenance", str(provenance)])
    with pytest.raises(SystemExit):
        leakage.main(["--task", "t", "--reference", "toy", "--references-yaml", str(config), "--query", "bad",
                      "--query-namespace", "uniprot", "--provenance", str(provenance)])


def test_checked_in_reference_config_is_well_formed():
    specs = leakage.load_references_config()
    assert {"protein_1m", "protein_1m_manifest", "protein_40k_manifest", "rna_full"} <= set(specs)
    for name, spec in specs.items():
        assert spec["kind"] in ("plan", "manifest") and spec["namespace"] in leakage.NAMESPACES, name
        assert (spec["kind"] == "plan") == ("split_column" in spec), name

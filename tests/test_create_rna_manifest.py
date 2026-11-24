"""RNA manifest creation: RNA3DB filters, builder acceptance test, determinism, provenance."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from Bio.PDB import MMCIFIO, PDBParser

SPEC = importlib.util.spec_from_file_location(
    "create_rna_manifest", Path(__file__).resolve().parents[1] / "data" / "rna" / "create_rna_manifest.py")
cm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cm
SPEC.loader.exec_module(cm)

GOOD = "ACGU" * 10
BACKBONE = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
BASE = ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"]


def meta(sequence=GOOD, resolution=2.0, method="x-ray diffraction"):
    return {"release_date": "2021-05-01", "structure_method": method, "resolution": resolution,
            "length": len(sequence), "sequence": sequence}


def write_cif(path, residues):
    """Write a tiny RNA mmCIF; ``residues`` is a list of atom-name lists (one residue each)."""
    lines, serial = [], 0
    for seq, atoms in enumerate(residues, start=1):
        for k, atom in enumerate(atoms):
            serial += 1
            x, y, z = 3.3 * seq + 0.9 * (k % 5), 1.1 * (k // 5) + 0.3 * seq, 0.2 * k
            lines.append(f"ATOM  {serial:5d} {atom:<4s} A   A{seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                         f"  1.00  0.00          {atom[0]:>2s}")
    pdb = path.with_suffix(".pdb")
    pdb.write_text("\n".join(lines + ["END"]) + "\n")
    io = MMCIFIO()
    io.set_structure(PDBParser(QUIET=True).get_structure(path.stem, str(pdb)))
    io.save(str(path))
    pdb.unlink()


def prepared(tmp_path, chains, *, status="complete", write_files=True, cif_writer=None):
    """A directory shaped like the download tool's output; ``chains`` maps name -> metadata."""
    root = tmp_path / "prepared"
    (root / "raw_structures").mkdir(parents=True)
    (root / "rna3db-jsons").mkdir()
    names = sorted(chains)
    split = {
        "train_set": {"component_1": {names[0]: {n: chains[n] for n in names[:-1]}}},
        "valid_set": {},
        "test_set": {"component_2": {names[-1]: {names[-1]: chains[names[-1]]}}},
    }
    (root / "rna3db-jsons" / "split.json").write_text(json.dumps(split))
    if write_files:
        for name in names:
            path = root / "raw_structures" / f"{name}.cif"
            (cif_writer or (lambda p: p.write_text("data_x\n")))(path)
    (root / "download_provenance.json").write_text(json.dumps({
        "release": "t-1", "tag": "t-tag", "status": status, "split_json_chains": len(chains),
        "archives": {"mmcifs": {"sha256": "ab" * 32}},
    }))
    return root


def run(root, *, check=lambda paths: {}, name="manifest.csv", **kwargs):
    return cm.create_manifest(root, root / name, check=check, log=lambda message: None, **kwargs)


def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


FILTER_CHAINS = {
    "1abc_A": meta(),
    "1abc_a": meta(),                               # lowercase chain of the same entry
    "2xyz_B": meta("ACGUACGUAC"),                   # 10 nt
    "3def_C": meta(resolution=12.5),                # low resolution
    "4ghi_D": meta(resolution=float("inf"), method="solution nmr"),
    "5jkl_E": meta("A" * 35 + "CGUAC"),              # 87.5 % A
    "6mno_F": meta("ACGU" * 5 + "N" * 20),          # 50 % N
    "7pqr_G": meta("CGUA" * 10, resolution=9.0),    # boundary values are kept
    "8stu_H": meta("ACGU" * 8, resolution=2.5),     # length 32 is kept
}


def test_rna3db_filters_reject_with_counts_per_reason(tmp_path):
    root = prepared(tmp_path, FILTER_CHAINS)
    record = run(root)
    assert [r["name"] for r in rows(root / "manifest.csv")] == ["1abc_A", "1abc_a", "7pqr_G", "8stu_H"]
    counts = record["counts"]
    assert counts["release_chains"] == 9 and counts["manifest_chains"] == 4
    assert counts["rejected_by_first_reason"] == {
        "length_below_minimum": 1, "resolution_above_maximum": 2,
        "single_nucleotide_ratio": 1, "unknown_nucleotide_ratio": 1,
    }
    reasons = {r["name"]: r["reason"] for r in record["rejected"]}
    assert reasons["4ghi_D"] == "resolution_above_maximum" and reasons["2xyz_B"] == "length_below_minimum"
    assert counts["manifest_chains_per_rna3db_set"] == {"test_set": 1, "train_set": 3}
    assert record["filters"] == {
        "min_length": 32, "max_resolution": 9.0, "single_ratio_cutoff": 0.8, "max_unknown_ratio": 0.3,
        "mirror_anchors": ["C2'", "C3'", "C4'"], "mirror_nonpositive_fraction": 0.5,
    }
    saved = json.loads((root / "manifest_provenance.json").read_text())
    assert saved["manifest"]["sha256"] == record["manifest"]["sha256"]
    assert saved["release"] == "t-1" and saved["download_archives"]["mmcifs"]["sha256"] == "ab" * 32


def test_thresholds_are_configurable(tmp_path):
    root = prepared(tmp_path, FILTER_CHAINS)
    run(root, min_length=10, max_resolution=20.0, single_ratio_cutoff=1.0, max_unknown_ratio=1.0)
    names = {r["name"] for r in rows(root / "manifest.csv")}
    assert "2xyz_B" in names and "3def_C" in names and "4ghi_D" not in names  # inf stays rejected


def test_manifest_is_sorted_has_builder_column_and_is_deterministic(tmp_path):
    chains = {"9zzz_A": meta(), "1abc_B": meta(), "1abc_A": meta(), "5mmm_c": meta()}
    first = prepared(tmp_path / "one", chains)
    second = prepared(tmp_path / "two", dict(reversed(list(chains.items()))))
    run(first)
    run(second)
    text = (first / "manifest.csv").read_bytes()
    assert text == (second / "manifest.csv").read_bytes()
    table = rows(first / "manifest.csv")
    assert [r["name"] for r in table] == sorted(chains)
    assert list(table[0]) == list(cm.MANIFEST_COLUMNS)
    assert table[0]["pdb_id"] == "1abc" and table[0]["chain"] == "A" and table[0]["length"] == "40"
    assert b"\r" not in text


def test_builder_rejections_are_dropped_and_recorded(tmp_path):
    root = prepared(tmp_path, {"1abc_A": meta(), "2xyz_B": meta(), "3def_C": meta()})
    seen = []

    def check(paths):
        seen.extend(path.name for path in paths)
        return {"2xyz_B": "contains a P-only nucleotide block"}

    record = run(root, check=check)
    assert seen == ["1abc_A.cif", "2xyz_B.cif", "3def_C.cif"]
    assert [r["name"] for r in rows(root / "manifest.csv")] == ["1abc_A", "3def_C"]
    assert record["counts"]["rejected_by_first_reason"] == {"builder: contains a P-only nucleotide block": 1}


def test_filtered_chains_are_not_sent_to_the_builder_check(tmp_path):
    root = prepared(tmp_path, {"1abc_A": meta(), "2xyz_B": meta("ACGU")})
    seen = []
    run(root, check=lambda paths: seen.extend(p.stem for p in paths) or {})
    assert seen == ["1abc_A"]


def test_unexpected_conversion_error_is_fatal(tmp_path):
    root = prepared(tmp_path, {"1abc_A": meta(), "2xyz_B": meta()})
    with pytest.raises(cm.ManifestError, match="unexpected reason"):
        run(root, check=lambda paths: {"2xyz_B": "ValueError: cannot read structure"})
    assert not (root / "manifest.csv").exists() and not (root / "manifest_provenance.json").exists()


def test_everything_rejected_is_fatal(tmp_path):
    root = prepared(tmp_path, {"1abc_A": meta("ACGU"), "2xyz_B": meta("ACGU")})
    with pytest.raises(cm.ManifestError, match="no chain survived"):
        run(root)


def test_existing_outputs_are_never_overwritten(tmp_path):
    root = prepared(tmp_path, {"1abc_A": meta(), "2xyz_B": meta()})
    run(root)
    before = (root / "manifest.csv").read_bytes()
    with pytest.raises(FileExistsError):
        run(root)
    assert (root / "manifest.csv").read_bytes() == before
    (root / "other_provenance.json").write_text("{}")
    with pytest.raises(FileExistsError):
        run(root, name="other.csv")


def test_missing_or_extra_structure_files_are_fatal(tmp_path):
    root = prepared(tmp_path, {"1abc_A": meta(), "2xyz_B": meta()})
    (root / "raw_structures" / "2xyz_B.cif").unlink()
    (root / "raw_structures" / "9zzz_Z.cif").write_text("x")
    with pytest.raises(cm.ManifestError, match="1 chains without a file.*1 files not in split.json"):
        run(root)


def test_incomplete_or_foreign_download_is_refused(tmp_path):
    root = prepared(tmp_path / "a", {"1abc_A": meta(), "2xyz_B": meta()}, status="started")
    with pytest.raises(cm.ManifestError, match="not 'complete'"):
        run(root)
    root = prepared(tmp_path / "b", {"1abc_A": meta(), "2xyz_B": meta()})
    (root / "download_provenance.json").unlink()
    with pytest.raises(cm.ManifestError, match="download_rna3db"):
        run(root)
    root = prepared(tmp_path / "c", {"1abc_A": meta(), "2xyz_B": meta()})
    record = json.loads((root / "download_provenance.json").read_text())
    record["split_json_chains"] = 5
    (root / "download_provenance.json").write_text(json.dumps(record))
    with pytest.raises(cm.ManifestError, match="recorded 5"):
        run(root)


@pytest.mark.parametrize("damage, message", [
    (lambda c: c["1abc_A"].pop("resolution"), "lacks"),
    (lambda c: c["1abc_A"].update(resolution="2.0"), "non-numeric"),
    (lambda c: c["1abc_A"].update(sequence=""), "no sequence"),
])
def test_damaged_split_metadata_is_fatal(tmp_path, damage, message):
    chains = {"1abc_A": meta(), "2xyz_B": meta()}
    damage(chains)
    root = prepared(tmp_path, chains)
    with pytest.raises(cm.ManifestError, match=message):
        run(root)


def test_wrong_split_layout_is_fatal(tmp_path):
    root = prepared(tmp_path, {"1abc_A": meta(), "2xyz_B": meta()})
    (root / "rna3db-jsons" / "split.json").write_text(json.dumps({"train_set": ["1abc_A"]}))
    with pytest.raises(cm.ManifestError, match="layout"):
        run(root)


def test_check_must_only_name_checked_chains(tmp_path):
    root = prepared(tmp_path, {"1abc_A": meta(), "2xyz_B": meta()})
    with pytest.raises(cm.ManifestError, match="were not checked"):
        run(root, check=lambda paths: {"9zzz_Z": "no atoms"})


def test_production_worker_rejects_p_only_chain_and_keeps_full_nucleotides(tmp_path):
    """Integration with the real builder worker on tiny structures (including a lowercase chain)."""
    full = [BACKBONE + BASE, BACKBONE + BASE]
    structures = {"1abc_A": full, "1abc_a": full, "2xyz_B": [["P"], ["P"]]}

    def writer(path):
        write_cif(path, structures[path.stem])

    root = prepared(tmp_path, {name: meta() for name in structures}, cif_writer=writer)
    record = cm.create_manifest(root, root / "manifest.csv", workers=1, log=lambda message: None)
    assert [r["name"] for r in rows(root / "manifest.csv")] == ["1abc_A", "1abc_a"]
    assert record["counts"]["rejected_by_first_reason"] == {"builder: contains a P-only nucleotide block": 1}


def test_cli_requires_prepared_dir_and_writes_default_manifest(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        cm.main([])
    root = prepared(tmp_path, {"1abc_A": meta(), "2xyz_B": meta()})
    monkeypatch.setattr(cm, "builder_rejections", lambda paths, **kwargs: {})
    assert cm.main(["--prepared-dir", str(root), "--workers", "1"]) == 0
    assert (root / "manifest_rna.csv").exists() and (root / "manifest_rna_provenance.json").exists()


def test_manifest_and_raw_dir_feed_the_edge_free_builder(tmp_path):
    """Download-tool layout + manifest are accepted by the builder CLI, lowercase chains included."""
    from data_loading import build_edge_free_chunks as builder

    full = [BACKBONE + BASE, BACKBONE + BASE]
    structures = {"1abc_A": full, "1abc_a": full, "3def_B": full, "4ghi_c": full, "5jkl_P": [["P"], ["P"]]}
    root = prepared(tmp_path, {name: meta() for name in structures},
                    cif_writer=lambda path: write_cif(path, structures[path.stem]))
    cm.create_manifest(root, root / "manifest_rna.csv", workers=1, log=lambda message: None)
    prefix = tmp_path / "chunks" / "rna"
    assert builder.main([
        "--domain", "rna", "--manifest", str(root / "manifest_rna.csv"),
        "--raw-dir", str(root / "raw_structures"), "--dest-prefix", str(prefix),
        "--chunk-size", "2", "--workers", "1", "--split-group", "pdb-entry",
    ]) == 0
    with open(tmp_path / "chunks" / "rna_plan.csv", newline="") as handle:
        planned = {row["id"] for row in csv.DictReader(handle)}
    assert planned == {"1abc_A", "1abc_a", "3def_B", "4ghi_c"}  # P-only chain never reaches the builder
    metadata = list((tmp_path / "chunks").glob("rna_*_chunk_*/processed/*_metadata.json"))
    assert metadata
    assert sum(json.loads(path.read_text())["failures"] == [] for path in metadata) == len(metadata)


# ---------------------------------------------------------------------------
# Mirror-image (left-handed) chains
# ---------------------------------------------------------------------------

# Frame of the chirality criterion: origin C3', x toward C4', plane through C2'; a right-handed
# (D-ribose) nucleotide has O2' on the positive-z side, its mirror image on the negative side.
ANCHORS = {"C3'": (0.0, 0.0, 0.0), "C4'": (1.5, 0.0, 0.0), "C2'": (-0.5, 1.4, 0.0), "O2'": (-1.0, 1.5, 1.2)}


def write_handed_cif(path, handedness):
    """Three nucleotides; ``handedness`` gives the sign per residue (+1 right-handed, -1 mirrored)."""
    lines, serial = [], 0
    for seq, sign in enumerate(handedness, start=1):
        for k, atom in enumerate(BACKBONE + BASE):
            serial += 1
            x, y, z = ANCHORS.get(atom, (0.7 * (k % 6) - 1.0, 1.9 + 0.8 * (k // 6), 0.35 * k - 2.0))
            x += 12.0 * seq
            lines.append(f"ATOM  {serial:5d} {atom:<4s} A   A{seq:4d}    {x:8.3f}{y:8.3f}{sign * z:8.3f}"
                         f"  1.00  0.00          {atom[0]:>2s}")
    pdb = path.with_suffix(".pdb")
    pdb.write_text("\n".join(lines + ["END"]) + "\n")
    io = MMCIFIO()
    io.set_structure(PDBParser(QUIET=True).get_structure(path.stem, str(pdb)))
    io.save(str(path))
    pdb.unlink()


def test_mirror_criterion_separates_right_and_left_handed_fixtures(tmp_path):
    cm._init_check_worker()
    cases = {"1abc_A": [1, 1, 1], "2xyz_B": [-1, -1, -1], "3def_C": [1, 1, -1], "4ghi_D": [-1, -1, 1]}
    results = {}
    for name, handedness in cases.items():
        path = tmp_path / f"{name}.cif"
        write_handed_cif(path, handedness)
        results[name] = cm._check_one(str(path))
    assert results["1abc_A"] == ("1abc_A", None)
    assert results["2xyz_B"] == ("2xyz_B", cm.MIRROR_REASON)
    assert results["3def_C"][1] is None                      # minority of left-handed blocks: kept
    assert results["4ghi_D"][1] == cm.MIRROR_REASON          # majority left-handed: rejected


def test_mirror_chains_are_dropped_and_counted_in_the_provenance(tmp_path):
    structures = {"1abc_A": [1, 1, 1], "2xyz_B": [-1, -1, -1], "3def_C": [-1, -1, -1]}
    root = prepared(tmp_path, {name: meta() for name in structures},
                    cif_writer=lambda path: write_handed_cif(path, structures[path.stem]))
    record = cm.create_manifest(root, root / "manifest.csv", workers=1, log=lambda message: None)
    assert [r["name"] for r in rows(root / "manifest.csv")] == ["1abc_A"]
    assert record["counts"]["mirror_image_chains"] == 2
    assert record["counts"]["rejected_by_first_reason"] == {"mirror_image": 2}
    assert {r["name"]: r["reason"] for r in record["rejected"]} == {"2xyz_B": "mirror_image", "3def_C": "mirror_image"}


def test_p_only_is_reported_before_mirror(tmp_path):
    cm._init_check_worker()
    path = tmp_path / "1abc_A.cif"
    write_cif(path, [["P"], ["P"]])
    assert cm._check_one(str(path))[1] == "contains a P-only nucleotide block"


# ---------------------------------------------------------------------------
# Re-verification of a finished manifest (rna-prepare RESUME=1)
# ---------------------------------------------------------------------------

def finished(tmp_path):
    root = prepared(tmp_path, {"1abc_A": meta(), "2xyz_B": meta(), "3def_C": meta("ACGU")})
    run(root)
    return root


def verify(root, **thresholds):
    base = {"min_length": 32, "max_resolution": 9.0, "single_ratio_cutoff": 0.8, "max_unknown_ratio": 0.3}
    base.update(thresholds)
    return cm.verify_existing_manifest(root, root / "manifest.csv", base, log=lambda message: None)


def test_finished_manifest_is_verified_without_writing(tmp_path):
    root = finished(tmp_path)
    snapshot = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.glob("manifest*")}
    assert verify(root)["counts"]["manifest_chains"] == 2
    assert snapshot == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.glob("manifest*")}


@pytest.mark.parametrize("damage", ["manifest", "split", "rawfile", "filters", "release", "provenance_missing"])
def test_manifest_that_does_not_match_the_inputs_fails_loudly(tmp_path, damage):
    root = finished(tmp_path)
    thresholds = {}
    if damage == "manifest":
        path = root / "manifest.csv"
        path.write_text(path.read_text().replace("1abc_A", "1abc_Z"))
    elif damage == "split":
        path = root / "rna3db-jsons" / "split.json"
        data = json.loads(path.read_text())
        next(iter(next(iter(data["train_set"].values())).values()))["1abc_A"]["resolution"] = 1.0
        path.write_text(json.dumps(data))
    elif damage == "rawfile":
        (root / "raw_structures" / "9zzz_Z.cif").write_text("x")
    elif damage == "filters":
        thresholds = {"min_length": 20}
    elif damage == "release":
        record = json.loads((root / "download_provenance.json").read_text())
        record["release"] = "other"
        (root / "download_provenance.json").write_text(json.dumps(record))
    else:
        (root / "manifest_provenance.json").unlink()
    with pytest.raises(cm.ManifestError):
        verify(root, **thresholds)


def test_cli_verify_existing_is_idempotent_and_creates_when_absent(tmp_path, monkeypatch):
    root = prepared(tmp_path, {"1abc_A": meta(), "2xyz_B": meta()})
    monkeypatch.setattr(cm, "builder_rejections", lambda paths, **kwargs: {})
    args = ["--prepared-dir", str(root), "--workers", "1", "--verify-existing"]
    assert cm.main(args) == 0                       # nothing there yet: creates
    created = (root / "manifest_rna.csv").read_bytes()
    monkeypatch.setattr(cm, "builder_rejections", lambda paths, **kwargs: pytest.fail("must not re-run the check"))
    assert cm.main(args) == 0                       # finished: verifies only
    assert (root / "manifest_rna.csv").read_bytes() == created
    with pytest.raises(FileExistsError):
        cm.main(args[:-1])                          # without the flag it still refuses
    (root / "manifest_rna.csv").write_text("name\n1abc_A\n")
    with pytest.raises(cm.ManifestError, match="manifest sha256"):
        cm.main(args)

"""COCONUT builder: filters, molecule-level splits, reconciliation and completeness check."""

import json
import shutil
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from data_loading import build_coconut as bc
from data_loading.build_edge_free_chunks import chunk_directory

SALT = "test-salt"
FRACTIONS = (0.5, 0.25, 0.25)
CONFORMERS = 3


def _record(block: str, smiles: str, conformer: int) -> str:
    return f"{block}>  <SMILES>  (1) \n{smiles}\n\n>  <CONFORMER_ID>  (1) \n{conformer}\n\n$$$$\n"


def _embedded(smiles: str, seed: int = 7) -> list[str]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    conformer_ids = AllChem.EmbedMultipleConfs(mol, CONFORMERS, randomSeed=seed)
    assert len(conformer_ids) == CONFORMERS
    return [
        _record(Chem.MolToMolBlock(mol, confId=cid), smiles, index)
        for index, cid in enumerate(conformer_ids, start=1)
    ]


def _flat(smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    return [_record(Chem.MolToMolBlock(mol), smiles, index) for index in (1, 2)]


def _hypervalent(smiles: str) -> list[str]:
    """Carbon with six neighbours: passes the filters, fails the PS_300 tokenizer."""
    rw = Chem.RWMol(Chem.MolFromSmiles("CCCCCCCCCCC"))
    for _ in range(5):
        fluorine = rw.AddAtom(Chem.Atom(9))
        rw.AddBond(5, fluorine, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    rng = np.random.default_rng(0)
    records = []
    for index in range(1, CONFORMERS + 1):
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for atom in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(atom, tuple(float(v) for v in rng.normal(size=3) * 3))
        mol.RemoveAllConformers()
        mol.AddConformer(conformer)
        records.append(_record(Chem.MolToMolBlock(mol), smiles, index))
    return records


GOOD = [f"{'C' * length}OC(=O)c1ccccc1N" for length in range(3, 27)]
SMALL = "CCO"                              # 3 heavy atoms
DISALLOWED = "C[Si](C)(C)OCCCCCCCCC"       # silicon
FLAT = "CCCCCCCCCCOC(=O)c1ccccc1"          # 2D depiction
BROKEN = "C(F)(F)(F)(F)(F)CCCCCCCCCC"      # tokenizer failure


@pytest.fixture(scope="module")
def synthetic_sdf(tmp_path_factory):
    """Interleaved records (not grouped by molecule); returns (path, counts)."""
    groups = [_embedded(smiles) for smiles in GOOD]
    groups += [_embedded(SMALL), _embedded(DISALLOWED), _flat(FLAT), _hypervalent(BROKEN)]
    records = [text for index in range(CONFORMERS) for group in groups for text in group[index:index + 1]]
    records += [text for group in groups for text in group[CONFORMERS:]]  # flat molecules have 2 records
    path = tmp_path_factory.mktemp("raw") / "coconut.sdf"
    path.write_text("".join(records))
    return path, {
        "raw": len(records),
        "too_few_heavy_atoms": CONFORMERS,
        "disallowed_element": CONFORMERS,
        "flat_2d_coordinates": 2,
        "failed": CONFORMERS,
        "built": CONFORMERS * len(GOOD),
    }


def _args(tmp_path, source, **overrides):
    argv = [
        "--input", str(source), "--dest-prefix", str(tmp_path / "coco"), "--chunk-size", "7",
        "--workers", "2", "--split-salt", SALT, "--split-fractions", *map(str, FRACTIONS),
        "--scan-range-mb", "1", "--max-failure-fraction", "0.5",
    ]
    for key, value in overrides.items():
        argv = [a for i, a in enumerate(argv) if not (a == f"--{key.replace('_', '-')}" or
                                                      (i and argv[i - 1] == f"--{key.replace('_', '-')}"))]
        argv += [f"--{key.replace('_', '-')}"] + ([] if value is True else [str(value)])
    return bc.parse_args(argv)


@pytest.fixture(scope="module")
def built(synthetic_sdf, tmp_path_factory):
    path, counts = synthetic_sdf
    root = tmp_path_factory.mktemp("built")
    assert bc.run_build(_args(root, path)) == 0
    return root, counts


# --- record filters -----------------------------------------------------------

def _first_record(smiles_list):
    return _embedded(smiles_list)[0].encode()


def test_classify_record_reasons_in_documented_order(synthetic_sdf):
    options = dict(min_heavy_atoms=10, allowed_elements=bc.DEFAULT_ALLOWED_ELEMENTS, molecule_key="smiles")
    heavy, reason, key, conformer = bc.classify_record(_first_record(GOOD[0]), **options)
    assert (reason, key, conformer) == (bc.KEPT, GOOD[0].encode(), 1) and heavy == 3 + 1 + 1 + 1 + 6 + 1
    assert bc.classify_record(_first_record(SMALL), **options)[1] == 1
    assert bc.classify_record(_first_record(DISALLOWED), **options)[1] == 2
    assert bc.classify_record(_flat(FLAT)[0].encode(), **options)[1] == 3
    # Small and disallowed: the first rule in the order wins.
    assert bc.classify_record(_first_record("C[Si](C)C"), **options)[1] == 1


@pytest.mark.parametrize("damage", ["counts", "tag", "block"])
def test_malformed_records_raise(damage):
    options = dict(min_heavy_atoms=10, allowed_elements=bc.DEFAULT_ALLOWED_ELEMENTS, molecule_key="smiles")
    record = _embedded(GOOD[0])[0]
    if damage == "counts":
        record = record.replace("V2000", "V3000")
    elif damage == "tag":
        record = record.replace("<SMILES>", "<OTHER>")
    else:
        record = record[: record.index("M  END") - 200]
    with pytest.raises(ValueError):
        bc.classify_record(record.encode(), **options)


def test_unparsable_record_aborts_the_build(tmp_path, synthetic_sdf):
    path, _ = synthetic_sdf
    text = path.read_text().replace("V2000", "V3000", 1)
    broken = tmp_path / "broken.sdf"
    broken.write_text(text)
    with pytest.raises(ValueError, match="malformed SDF record at byte 0"):
        bc.run_build(_args(tmp_path, broken))


# --- scan and plan --------------------------------------------------------------

def test_scan_is_independent_of_range_size(synthetic_sdf):
    path, counts = synthetic_sdf
    options = dict(salt=SALT, fractions=FRACTIONS, molecule_key="smiles", min_heavy_atoms=10,
                   allowed_elements=bc.DEFAULT_ALLOWED_ELEMENTS, workers=2, max_records=None)
    whole = bc.scan_sdf(path, range_bytes=10**9, **options)
    pieces = bc.scan_sdf(path, range_bytes=20_000, **options)
    assert len(bc.plan_byte_ranges(path, 20_000)) > 3
    for name in ("offset", "length", "heavy", "reason", "molecule", "conformer", "split"):
        assert np.array_equal(getattr(whole, name), getattr(pieces, name)), name
    assert whole.num_records == counts["raw"]
    data = path.read_bytes()
    for offset, length in zip(whole.offset, whole.length):
        assert data[offset + length - 5:offset + length] == b"$$$$\n"
    assert bc.reason_counts(whole) == {k: counts[k] for k in bc.REASON_NAMES.values()}


def test_max_records_truncates_and_reports_scanned_bytes(synthetic_sdf):
    path, _ = synthetic_sdf
    scan = bc.scan_sdf(path, salt=SALT, fractions=FRACTIONS, molecule_key="smiles", min_heavy_atoms=10,
                       allowed_elements=bc.DEFAULT_ALLOWED_ELEMENTS, workers=2,
                       range_bytes=20_000, max_records=10)
    assert scan.num_records == 10
    assert scan.scanned_bytes == int(scan.offset[-1] + scan.length[-1])


def test_conformers_of_a_molecule_stay_in_one_split(synthetic_sdf):
    path, counts = synthetic_sdf
    scan = bc.scan_sdf(path, salt=SALT, fractions=FRACTIONS, molecule_key="smiles", min_heavy_atoms=10,
                       allowed_elements=bc.DEFAULT_ALLOWED_ELEMENTS, workers=2,
                       range_bytes=10**9, max_records=None)
    plan = bc.build_plan(scan, chunk_size=7, shuffle_seed=1)
    assert plan.groupby("molecule")["split"].nunique().eq(1).all()
    assert plan["molecule"].nunique() == len(GOOD) + 1  # 24 good molecules + the tokenizer failure
    assert set(plan["split"]) == {"train", "val", "test"}  # the salt exercises every split
    assert not plan["record"].duplicated().any()
    sizes = plan.groupby(["split", "chunk"]).size()
    for split in bc.SPLITS:  # balanced chunks, none above the bound, no tiny tail
        per_split = sizes[split]
        assert per_split.max() <= 7 and per_split.max() - per_split.min() <= 1
    assert (plan.groupby(["split", "chunk"])["position"].apply(lambda p: sorted(p) == list(range(len(p))))).all()
    # Chunks are shuffled across the size-ordered input, not cut by position.
    train = plan[plan["split"] == "train"].sort_values(["chunk", "position"])
    assert not train["record"].is_monotonic_increasing
    broken = pd.DataFrame({"molecule": ["a", "a"], "split": ["train", "val"]})
    with pytest.raises(bc.BuildVerificationError, match="more than one split"):
        bc.check_no_molecule_in_two_splits(broken)


def test_split_does_not_depend_on_record_position(synthetic_sdf, tmp_path):
    path, _ = synthetic_sdf
    reversed_path = tmp_path / "reversed.sdf"
    records = path.read_text().split("$$$$\n")[:-1]
    reversed_path.write_text("".join(f"{text}$$$$\n" for text in reversed(records)))
    common = dict(salt=SALT, fractions=FRACTIONS, molecule_key="smiles", min_heavy_atoms=10,
                  allowed_elements=bc.DEFAULT_ALLOWED_ELEMENTS, workers=2, range_bytes=10**9, max_records=None)
    forward = bc.scan_sdf(path, **common)
    backward = bc.scan_sdf(reversed_path, **common)
    by_molecule = lambda scan: dict(zip(scan.molecule.tolist(), scan.split.tolist()))  # noqa: E731
    assert by_molecule(forward) == by_molecule(backward)


# --- end to end ---------------------------------------------------------------

def test_build_reconciles_exactly(built):
    root, counts = built
    manifest = json.loads((root / "coco_build_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["filter_version"] == bc.FILTER_VERSION
    assert len(manifest["inputs"]["sdf_sha256"]) == 64 and manifest["tokenizer"]["name"] == "PS_300"
    report = manifest["build"]
    assert report["raw_records"] == counts["raw"]
    assert report["rejected_by_reason"] == {k: counts[k] for k in bc.REASON_NAMES.values()}
    assert report["failed"] == counts["failed"] and report["built"] == counts["built"]
    assert report["raw_records"] == report["rejected"] + report["failed"] + report["built"]
    assert report["chunks_built"] == report["chunks_planned"] > 3
    assert sum(split["built"] for split in report["per_split"].values()) == counts["built"]


def test_failures_are_recorded_with_ids_and_reasons(built):
    root, counts = built
    failures = pd.read_csv(root / "coco_failures.csv")
    assert len(failures) == counts["failed"]
    assert failures["reason"].str.startswith("tokenization error").all()
    plan = pd.read_csv(root / "coco_plan.csv.gz")
    assert set(failures["record"]) <= set(plan["record"])
    assert (failures["id"] == "coconut_" + failures["record"].astype(str)).all()
    rejected = pd.read_csv(root / "coco_rejected.csv")
    assert Counter(rejected["reason"]) == {k: counts[k] for k in bc.REASON_NAMES.values()}


def test_chunks_load_through_the_training_config_path(built, tmp_path, monkeypatch):
    """The metabolite domain reads the new layout through ``splits`` with null chunk ranges."""
    from types import SimpleNamespace

    from core.train_pretrain import load_universal_dataset, split_dataset

    root, counts = built
    monkeypatch.setenv("MIND_INDEX_MAP_CACHE_DIR", str(tmp_path / "maps"))
    config = SimpleNamespace(
        multi_domain={"enabled": True, "max_cache_chunks": 2, "datasets": {"metabolite": {
            "enabled": True,
            "splits": {split: {"chunk_dir": str(root / f"coco_{split}"), "chunk_range": None}
                       for split in bc.SPLITS},
        }}},
        pretraining_tasks=[], use_improved_sampler=True,
    )
    dataset = load_universal_dataset(config, "METABOLITE", "")
    assert len(dataset) == counts["built"] and dataset.metadata_loaded
    per_split = json.loads((root / "coco_build_manifest.json").read_text())["build"]["per_split"]
    train, val, test = split_dataset(dataset, seed=0)
    assert [len(train), len(val), len(test)] == [per_split[split]["built"] for split in bc.SPLITS]
    sample = dataset[0]
    assert sample.dataset_type == "coconut" and sample.mol_id.startswith("coconut_")
    assert sample.edge_index.shape[0] == 2 and sample.bond_edge_index.shape[0] == 2
    assert sample.pos.shape == (sample.z.numel(), 3) and len(sample.pos_code) == sample.z.numel()
    assert int(sample.z.max()) < 120 and sample.num_nodes == sample.z.numel()
    # No molecule is in two splits: split membership of every built record matches the plan.
    plan = pd.read_csv(root / "coco_plan.csv.gz").set_index("record")["split"]
    for name, subset in zip(bc.SPLITS, (train, val, test)):
        ids = {int(dataset[i].mol_id.split("_")[1]) for i in subset.indices}
        assert ids and set(plan.loc[sorted(ids)]) == {name}


def test_second_run_verifies_and_refuses_changed_parameters(built, synthetic_sdf):
    root, _ = built
    path, _ = synthetic_sdf
    assert bc.run_build(_args(root, path)) == 0
    with pytest.raises(ValueError, match="different"):
        bc.run_build(_args(root, path, chunk_size=9))
    assert bc.main(["--verify-only", "--dest-prefix", str(root / "coco")]) == 0


def _copy_build(built, tmp_path):
    root, _ = built
    target = tmp_path / "copy"
    shutil.copytree(root, target)
    return target


def test_missing_chunk_fails_the_completeness_check(built, tmp_path):
    target = _copy_build(built, tmp_path)
    victim = chunk_directory(target / "coco", "train", 0)
    shutil.rmtree(victim)
    with pytest.raises(bc.BuildVerificationError, match="chunk directory missing"):
        bc.verify_build(target / "coco")
    with pytest.raises(bc.BuildVerificationError, match="built chunks"):
        bc.verify_build(target / "coco")


def test_unplanned_and_unfinished_chunk_directories_fail(built, tmp_path):
    target = _copy_build(built, tmp_path)
    (target / "coco_train_chunk_999.partial").mkdir()
    with pytest.raises(bc.BuildVerificationError, match="unplanned or unfinished"):
        bc.verify_build(target / "coco")


def test_dropped_record_breaks_the_books(built, tmp_path):
    target = _copy_build(built, tmp_path)
    sidecar = next((chunk_directory(target / "coco", "val", 0) / "processed").glob("*_metadata.json"))
    metadata = json.loads(sidecar.read_text())
    metadata["samples"].pop()
    metadata["num_samples"] -= 1
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(bc.BuildVerificationError, match="do not equal the .* planned records"):
        bc.verify_build(target / "coco")


def test_altered_raw_count_breaks_the_books(built, tmp_path):
    target = _copy_build(built, tmp_path)
    manifest_path = target / "coco_build_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["counts"]["raw_records"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(bc.BuildVerificationError, match="raw"):
        bc.verify_build(target / "coco")


def test_interrupted_build_resumes_without_rewriting_chunks(built, synthetic_sdf, tmp_path):
    root, counts = built
    path, _ = synthetic_sdf
    target = tmp_path / "resume"
    shutil.copytree(root, target)
    manifest_path = target / "coco_build_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status="started")
    manifest_path.write_text(json.dumps(manifest))
    victim = chunk_directory(target / "coco", "test", 0)
    shutil.rmtree(victim)
    kept = chunk_directory(target / "coco", "train", 0) / "processed"
    stamp = {p.name: p.stat().st_mtime_ns for p in kept.iterdir()}
    assert bc.run_build(_args(target, path)) == 0
    assert victim.exists()
    assert stamp == {p.name: p.stat().st_mtime_ns for p in kept.iterdir()}
    assert json.loads(manifest_path.read_text())["build"]["built"] == counts["built"]


def test_resume_refuses_a_different_plan(built, synthetic_sdf, tmp_path):
    root, _ = built
    path, _ = synthetic_sdf
    target = tmp_path / "mixed"
    shutil.copytree(root, target)
    manifest_path = target / "coco_build_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status="started")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs"):
        bc.run_build(_args(target, path, shuffle_seed=43))


def test_failure_fraction_aborts_the_build(synthetic_sdf, tmp_path):
    path, _ = synthetic_sdf
    with pytest.raises(RuntimeError, match="failed conversion"):
        bc.run_build(_args(tmp_path, path, max_failure_fraction=0.0))


def test_zip_input_is_extracted_and_cleaned(synthetic_sdf, tmp_path, built):
    path, counts = synthetic_sdf
    archive = tmp_path / "coconut.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.write(path, "coconut_sdf_3d.sdf")
    assert bc.run_build(_args(tmp_path, archive)) == 0
    manifest = json.loads((tmp_path / "coco_build_manifest.json").read_text())
    assert manifest["inputs"]["archive_sha256"] and manifest["build"]["built"] == counts["built"]
    assert manifest["inputs"]["sdf_sha256"] == json.loads(
        (built[0] / "coco_build_manifest.json").read_text())["inputs"]["sdf_sha256"]
    assert not (tmp_path / "coco_work").exists()
    assert pd.read_csv(tmp_path / "coco_plan.csv.gz").equals(pd.read_csv(built[0] / "coco_plan.csv.gz"))


def test_ordered_bounded_map_preserves_order():
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(4) as executor:
        assert list(bc.ordered_bounded_map(executor, lambda x: x * x, range(20), window=3)) == [
            x * x for x in range(20)]

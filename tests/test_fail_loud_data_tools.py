"""Data acquisition tools: failures are recorded and end in a non-zero exit, never a quiet success."""

import gzip
import importlib.util
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load(relative):
    path = DATA_DIR / relative
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_") + "_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def metadata(tmp_path):
    path = tmp_path / "meta.tsv.gz"
    rows = [f"P{i}\t0\t1\t100\t100\t{90 - i}\t{90 - i}\t9606" for i in range(4)]
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(rows) + "\n")
    return path


def test_manifest_records_failed_url_lookups_and_fails_the_run(tmp_path, metadata):
    tool = _load("protein_pipeline/1_filter_and_create_manifest.py")

    def lookup(uniprot_id):
        if uniprot_id == "P1":
            raise ConnectionError("api down")
        if uniprot_id == "P2":
            return None, None
        return f"http://x/{uniprot_id}.pdb", f"http://x/{uniprot_id}.cif"

    tool.get_download_urls_from_api = lookup
    output = tmp_path / "manifest.csv"
    with pytest.raises(RuntimeError, match=r"2 of 4 proteins"):
        tool.create_manifest(metadata, output)
    assert sorted(line.split("\t")[0] for line in
                  Path(str(output) + ".failed.tsv").read_text().splitlines()) == ["P1", "P2"]
    assert "ConnectionError" in Path(str(output) + ".failed.tsv").read_text()
    assert "P0" in output.read_text() and "P1" not in output.read_text()


def test_manifest_with_all_urls_succeeds_without_a_failure_file(tmp_path, metadata):
    tool = _load("protein_pipeline/1_filter_and_create_manifest.py")
    tool.get_download_urls_from_api = lambda uid: (f"http://x/{uid}.pdb", f"http://x/{uid}.cif")
    output = tmp_path / "manifest.csv"
    tool.create_manifest(metadata, output)
    assert not Path(str(output) + ".failed.tsv").exists()


def test_unreadable_metadata_raises_instead_of_returning(tmp_path):
    tool = _load("protein_pipeline/1_filter_and_create_manifest.py")
    bad = tmp_path / "bad.tsv.gz"
    bad.write_text("not gzip")
    with pytest.raises(ValueError, match="Failed to read metadata"):
        tool.create_manifest(bad, tmp_path / "m.csv")
    with pytest.raises(ValueError, match="Failed to read metadata"):
        tool.analyze_metadata(bad)


def test_download_tool_raises_on_unreadable_manifest(tmp_path):
    tool = _load("protein_pipeline/2_download_pdbs_from_manifest.py")
    with pytest.raises(Exception):
        tool.parallel_download_from_manifest(tmp_path / "absent.csv", tmp_path / "out")


def test_download_tool_exits_nonzero_when_a_structure_failed(tmp_path, monkeypatch):
    tool = _load("protein_pipeline/2_download_pdbs_from_manifest.py")
    manifest = tmp_path / "m.csv"
    manifest.write_text("repId,pdb_url,cif_url\nA,http://x/a,http://x/a\nB,http://x/b,http://x/b\n")
    monkeypatch.setattr(tool, "download_text", lambda url, **kw: None if url.endswith("b") else "ATOM")
    monkeypatch.setattr("sys.argv", ["tool", "--manifest-file", str(manifest),
                                     "--structures-outdir", str(tmp_path / "out"), "--workers", "1"])
    with pytest.raises(SystemExit) as caught:
        tool.main()
    assert "1 of 2" in str(caught.value) and "B" in str(caught.value)


@pytest.mark.parametrize("script, attr", [("download_lba.py", "download_lba"),
                                           ("download_coconut.py", "download_coconut")])
def test_single_file_downloaders_exit_nonzero_on_failure(script, attr, monkeypatch):
    tool = _load(script)
    monkeypatch.setattr(tool, attr, lambda: None)
    with pytest.raises(SystemExit) as caught:
        tool.main()
    assert caught.value.code not in (0, None)


def test_rna_size_filter_fails_when_a_structure_cannot_be_read(tmp_path):
    tool = _load("rna/filter_rna_manifest.py")
    manifest = tmp_path / "in.csv"
    manifest.write_text("name\nmissing_one\n")
    output = tmp_path / "out.csv"
    with pytest.raises(SystemExit, match=r"1 of 1 .*missing_one"):
        tool.filter_rna_manifest_by_size(str(manifest), str(tmp_path), str(output))

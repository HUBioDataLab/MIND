"""RNA3DB download/extract: chain-id decoding, integrity checks, safe extraction, resume."""

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import urllib.error
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "download_rna3db", Path(__file__).resolve().parents[1] / "data" / "rna" / "download_rna3db.py")
dl = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dl  # dataclasses resolve annotations through sys.modules
SPEC.loader.exec_module(dl)

# Archive member stem (as RNA3DB writes it) -> decoded chain name.
CHAINS = {"1abc_A": "1abc_A", "1abc_-a-": "1abc_a", "2xyz_-m-R": "2xyz_mR", "3def_B": "3def_B"}
META = {"release_date": "2020-01-01", "structure_method": "x-ray diffraction", "resolution": 2.0,
        "length": 40, "sequence": "ACGU" * 10}


def _split(names):
    """split.json layout: set -> component -> representative -> chain -> metadata."""
    return {
        "train_set": {"component_1": {names[0]: {n: dict(META) for n in names[:-1]}}},
        "valid_set": {},
        "test_set": {"component_2": {names[-1]: {names[-1]: dict(META)}}},
    }


def _tar_bytes(members, mode):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=mode) as tar:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            if payload is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _mmcif_members(chains=CHAINS, extra=()):
    members = [("rna3db-mmcifs/", None)]
    for stem in chains:
        members.append((f"rna3db-mmcifs/train_set/component_1/{stem}/{stem}.cif", f"data_{stem}\n".encode()))
    members.extend(extra)
    return members


class Remote:
    """Local stand-in for the release host: serves byte ranges, can be told to fail."""

    def __init__(self, files):
        self.files = files
        self.requests = []
        self.fail_after = None  # bytes to serve before raising, once

    def open(self, url, offset):
        data = self.files[url.rsplit("/", 1)[1]]
        self.requests.append((url, offset))
        remote = self

        class Response:
            status = 206 if offset else 200

            def __init__(self):
                self.stream = io.BytesIO(data[offset:])

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, size=-1):
                if remote.fail_after is not None:
                    cut, remote.fail_after = remote.fail_after, None
                    chunk = self.stream.read(cut)
                    self.pending = chunk
                    raise urllib.error.URLError("connection reset")
                return self.stream.read(size)

        return Response()


def make_release(monkeypatch, tmp_path, *, chains=CHAINS, split_names=None, pinned=True,
                 mmcif_extra=(), corrupt_pin=False):
    split_names = list(split_names if split_names is not None else chains.values())
    mmcifs = _tar_bytes(_mmcif_members(chains, mmcif_extra), "w:xz")
    jsons = _tar_bytes([("rna3db-jsons/split.json", json.dumps(_split(split_names)).encode()),
                        ("rna3db-jsons/filter.json", b"{}")], "w:gz")
    digest = hashlib.sha256(mmcifs).hexdigest()
    spec = dl.ReleaseSpec(
        "t-1", "t-tag", "rna3db-mmcifs.tar.xz", len(mmcifs), ("0" * 64 if corrupt_pin else digest) if pinned else None,
        "rna3db-jsons.tar.gz", len(jsons), hashlib.sha256(jsons).hexdigest(), len(split_names))
    remote = Remote({"rna3db-mmcifs.tar.xz": mmcifs, "rna3db-jsons.tar.gz": jsons})
    monkeypatch.setattr(dl, "_open_remote", remote.open)
    monkeypatch.setattr(dl.time, "sleep", lambda seconds: None)
    return spec, remote, digest


def quiet(*args, **kwargs):
    return dl.run(*args, log=lambda message: None, **kwargs)


# ---------------------------------------------------------------------------

@pytest.mark.parametrize("encoded, decoded", [
    ("A", "A"), ("-m-", "m"), ("-hh-", "hh"), ("-m-R", "mR"), ("A-b-C", "AbC"), ("1", "1"), ("-a-1-b-", "a1b"),
])
def test_decode_chain_id_inverts_rna3db_encoding(encoded, decoded):
    assert dl.decode_chain_id(encoded) == decoded
    assert dl.encode_chain_id(decoded) == encoded


@pytest.mark.parametrize("bad", ["-m", "m-", "-M-", "a", "-a--b-", "A--"])
def test_malformed_encodings_are_rejected(bad):
    with pytest.raises(dl.Rna3dbError):
        dl.decode_chain_id(bad)


@pytest.mark.parametrize("stem", ["7taw", "7taw_", "XYZ_A", "7taw_A_B", "7taw_-m"])
def test_unexpected_file_names_are_rejected(stem):
    with pytest.raises(dl.Rna3dbError):
        dl.decode_member_stem(stem)


def test_release_aliases_and_tags_resolve_and_unknown_fails():
    assert dl.resolve_release("2025-10").tag == "2025-10-01-incremental-release"
    assert dl.resolve_release("2026-01-05-full-release").alias == "2026-01"
    with pytest.raises(dl.Rna3dbError, match="unknown release"):
        dl.resolve_release("latest")


def test_release_is_a_required_argument(tmp_path):
    with pytest.raises(SystemExit):
        dl.main(["--out-dir", str(tmp_path / "x")])


def test_happy_path_decodes_lowercase_chains_and_loses_none(monkeypatch, tmp_path):
    spec, _, digest = make_release(monkeypatch, tmp_path)
    out = tmp_path / "out"
    record = quiet(spec, out)
    names = sorted(path.name for path in (out / "raw_structures").iterdir())
    assert names == sorted(f"{decoded}.cif" for decoded in CHAINS.values())
    assert {"1abc_A.cif", "1abc_a.cif"} <= set(names)  # chains differing only by case coexist
    assert (out / "raw_structures" / "2xyz_mR.cif").read_text() == "data_2xyz_-m-R\n"
    assert record["status"] == "complete" and record["structures_on_disk"] == 4
    assert record["extraction"]["encoded_names"] == 2
    assert record["archives"]["mmcifs"]["sha256"] == digest
    assert record["archives"]["mmcifs"]["sha256_source"] == "pinned"
    assert record["split_json_chains_per_set"] == {"train_set": 3, "test_set": 1}
    assert not (out / "downloads").exists()
    assert not list(out.rglob("*.tmp")) and not list(out.rglob("*.part"))
    assert json.loads((out / dl.PROVENANCE_NAME).read_text())["status"] == "complete"
    assert (out / "rna3db-jsons" / "split.json").exists()


def test_keep_archives_keeps_the_tarballs(monkeypatch, tmp_path):
    spec, _, _ = make_release(monkeypatch, tmp_path)
    quiet(spec, tmp_path / "out", keep_archives=True)
    assert (tmp_path / "out" / "downloads" / spec.mmcifs_asset).exists()


def test_corrupted_archive_fails_on_pinned_sha_and_is_removed(monkeypatch, tmp_path):
    spec, _, _ = make_release(monkeypatch, tmp_path, corrupt_pin=True)
    out = tmp_path / "out"
    with pytest.raises(dl.Rna3dbError, match="SHA-256 mismatch"):
        quiet(spec, out)
    assert not (out / "downloads" / spec.mmcifs_asset).exists()
    assert not (out / "raw_structures").exists()


def test_unpinned_release_records_sha_and_refuses_a_later_mismatch(monkeypatch, tmp_path):
    spec, remote, digest = make_release(monkeypatch, tmp_path, pinned=False)
    out = tmp_path / "out"
    record = quiet(spec, out, keep_archives=True)
    assert record["archives"]["mmcifs"]["sha256"] == digest
    assert record["archives"]["mmcifs"]["sha256_source"] == "computed on first download"
    # Same release re-downloaded into the same directory with other bytes: refused.
    (out / dl.PROVENANCE_NAME).write_text(json.dumps(dict(record, status="started")))
    (out / "downloads" / spec.mmcifs_asset).unlink()
    other = _tar_bytes(_mmcif_members(extra=[("rna3db-mmcifs/x.txt", b"x")]), "w:xz")
    remote.files[spec.mmcifs_asset] = other
    spec_other = dl.ReleaseSpec(**{**spec.__dict__, "mmcifs_size": len(other)})
    with pytest.raises(dl.Rna3dbError, match="conflicting SHA-256 references|mismatch"):
        quiet(spec_other, out, resume=True)


def test_expected_sha_flag_must_match(monkeypatch, tmp_path):
    spec, _, digest = make_release(monkeypatch, tmp_path, pinned=False)
    with pytest.raises(dl.Rna3dbError, match="SHA-256 mismatch"):
        quiet(spec, tmp_path / "a", expected_sha256="1" * 64)
    quiet(spec, tmp_path / "b", expected_sha256=digest)


def test_chain_count_difference_is_fatal(monkeypatch, tmp_path):
    # split.json lists one chain more than the archive holds
    spec, _, _ = make_release(monkeypatch, tmp_path, split_names=list(CHAINS.values()) + ["4ghi_A"])
    with pytest.raises(dl.Rna3dbError, match="1 missing"):
        quiet(spec, tmp_path / "out")
    # the archive holds one chain split.json does not list
    spec, _, _ = make_release(monkeypatch, tmp_path, split_names=list(CHAINS.values())[:-1])
    with pytest.raises(dl.Rna3dbError, match="not in split.json"):
        quiet(spec, tmp_path / "out2")


def test_pinned_count_must_equal_split_json(monkeypatch, tmp_path):
    spec, _, _ = make_release(monkeypatch, tmp_path)
    wrong = dl.ReleaseSpec(**{**spec.__dict__, "chains": 5})
    with pytest.raises(dl.Rna3dbError, match="pinned count"):
        quiet(wrong, tmp_path / "out")


def test_duplicate_decoded_name_is_fatal(monkeypatch, tmp_path):
    members = _mmcif_members() + [("rna3db-mmcifs/other/1abc_-a-.cif", b"dup\n")]
    mmcifs = _tar_bytes(members, "w:xz")
    spec, remote, _ = make_release(monkeypatch, tmp_path)
    remote.files[spec.mmcifs_asset] = mmcifs
    spec = dl.ReleaseSpec(**{**spec.__dict__, "mmcifs_size": len(mmcifs), "mmcifs_sha256": None})
    with pytest.raises(dl.Rna3dbError, match="occurs twice"):
        quiet(spec, tmp_path / "out")


@pytest.mark.parametrize("member", [
    "rna3db-mmcifs/../../evil.cif", "/abs/evil.cif", "rna3db-mmcifs/a\\b.cif",
])
def test_unsafe_member_names_are_rejected(monkeypatch, tmp_path, member):
    spec, remote, _ = make_release(monkeypatch, tmp_path)
    mmcifs = _tar_bytes(_mmcif_members(extra=[(member, b"x\n")]), "w:xz")
    remote.files[spec.mmcifs_asset] = mmcifs
    spec = dl.ReleaseSpec(**{**spec.__dict__, "mmcifs_size": len(mmcifs), "mmcifs_sha256": None})
    out = tmp_path / "out"
    with pytest.raises(dl.Rna3dbError, match="unsafe tar member"):
        quiet(spec, out)
    assert not (tmp_path / "evil.cif").exists() and not Path("/abs/evil.cif").exists()


def test_symlink_and_foreign_members_are_rejected(monkeypatch, tmp_path):
    spec, remote, _ = make_release(monkeypatch, tmp_path)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as tar:
        link = tarfile.TarInfo("rna3db-mmcifs/1abc_A.cif")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)
    remote.files[spec.mmcifs_asset] = buffer.getvalue()
    spec = dl.ReleaseSpec(**{**spec.__dict__, "mmcifs_size": len(buffer.getvalue()), "mmcifs_sha256": None})
    with pytest.raises(dl.Rna3dbError, match="unsupported member type"):
        quiet(spec, tmp_path / "out")
    mmcifs = _tar_bytes(_mmcif_members(extra=[("rna3db-mmcifs/notes.txt", b"x")]), "w:xz")
    remote.files[spec.mmcifs_asset] = mmcifs
    spec = dl.ReleaseSpec(**{**spec.__dict__, "mmcifs_size": len(mmcifs)})
    with pytest.raises(dl.Rna3dbError, match="non-mmCIF"):
        quiet(spec, tmp_path / "out2")


def test_appledouble_files_are_skipped_and_counted(monkeypatch, tmp_path):
    spec, remote, _ = make_release(monkeypatch, tmp_path)
    mmcifs = _tar_bytes(_mmcif_members(extra=[("rna3db-mmcifs/._1abc_A.cif", b"junk")]), "w:xz")
    remote.files[spec.mmcifs_asset] = mmcifs
    spec = dl.ReleaseSpec(**{**spec.__dict__, "mmcifs_size": len(mmcifs), "mmcifs_sha256": None})
    record = quiet(spec, tmp_path / "out")
    assert record["extraction"]["appledouble_skipped"] == 1 and record["structures_on_disk"] == 4


def test_nonempty_directory_is_never_overwritten_without_resume(monkeypatch, tmp_path):
    spec, _, _ = make_release(monkeypatch, tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "precious.txt").write_text("keep")
    with pytest.raises(dl.Rna3dbError, match="refusing to overwrite"):
        quiet(spec, out)
    with pytest.raises(dl.Rna3dbError, match="cannot resume"):
        quiet(spec, out, resume=True)
    assert (out / "precious.txt").read_text() == "keep"
    quiet(spec, tmp_path / "fresh")
    with pytest.raises(dl.Rna3dbError, match="refusing to overwrite"):
        quiet(spec, tmp_path / "fresh")


def test_resume_of_other_release_is_refused(monkeypatch, tmp_path):
    spec, _, _ = make_release(monkeypatch, tmp_path)
    quiet(spec, tmp_path / "out")
    other = dl.ReleaseSpec(**{**spec.__dict__, "alias": "t-2"})
    with pytest.raises(dl.Rna3dbError, match="belongs to release"):
        quiet(other, tmp_path / "out", resume=True)


def test_interrupted_download_resumes_from_part_file(monkeypatch, tmp_path):
    spec, remote, digest = make_release(monkeypatch, tmp_path)
    out = tmp_path / "out"
    data = remote.files[spec.mmcifs_asset]
    (out / "downloads").mkdir(parents=True)
    (out / "downloads" / (spec.mmcifs_asset + ".part")).write_bytes(data[:50])
    (out / dl.PROVENANCE_NAME).write_text(json.dumps({"release": spec.alias, "status": "started"}))
    record = quiet(spec, out, resume=True)
    assert (f"https://github.com/marcellszi/rna3db/releases/download/t-tag/{spec.mmcifs_asset}", 50) in remote.requests
    assert record["archives"]["mmcifs"]["sha256"] == digest


def test_network_error_mid_stream_is_retried_with_a_range_request(monkeypatch, tmp_path):
    spec, remote, digest = make_release(monkeypatch, tmp_path)
    target = tmp_path / "file.bin"
    payload = remote.files[spec.mmcifs_asset]
    original_open = remote.open
    state = {"first": True}

    def flaky(url, offset):
        response = original_open(url, offset)
        if state["first"]:
            state["first"] = False
            half = response.stream.read(60)

            def read(size=-1, _half=[half]):
                if _half:
                    return _half.pop()
                raise urllib.error.URLError("reset")

            response.read = read
        return response

    monkeypatch.setattr(dl, "_open_remote", flaky)
    dl.download_file("http://host/rna3db-mmcifs.tar.xz", target, expected_size=len(payload), retries=3)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    assert not target.with_name(target.name + ".part").exists()


def test_download_gives_up_loudly_and_never_publishes_a_short_file(monkeypatch, tmp_path):
    def always_fails(url, offset):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(dl, "_open_remote", always_fails)
    monkeypatch.setattr(dl.time, "sleep", lambda seconds: None)
    target = tmp_path / "file.bin"
    with pytest.raises(dl.Rna3dbError, match="could not download"):
        dl.download_file("http://host/x", target, expected_size=10, retries=2)
    assert not target.exists()


def test_http_404_fails_immediately(monkeypatch, tmp_path):
    calls = []

    def missing(url, offset):
        calls.append(url)
        raise urllib.error.HTTPError(url, 404, "not found", None, None)

    monkeypatch.setattr(dl, "_open_remote", missing)
    with pytest.raises(dl.Rna3dbError, match="HTTP 404"):
        dl.download_file("http://host/x", tmp_path / "f", expected_size=10, retries=5)
    assert len(calls) == 1


def test_wrong_size_archive_is_refused(monkeypatch, tmp_path):
    spec, remote, _ = make_release(monkeypatch, tmp_path)
    remote.files[spec.mmcifs_asset] = remote.files[spec.mmcifs_asset] + b"extra"
    with pytest.raises(dl.Rna3dbError, match="pinned size"):
        quiet(spec, tmp_path / "out")
    assert not list((tmp_path / "out").rglob("*.xz*"))
    short = remote.files[spec.mmcifs_asset][:-10]
    remote.files[spec.mmcifs_asset] = short
    with pytest.raises(dl.Rna3dbError, match="could not download"):
        quiet(spec, tmp_path / "out2")
    assert not (tmp_path / "out2" / "downloads" / spec.mmcifs_asset).exists()


def test_finished_run_is_idempotent_with_resume(monkeypatch, tmp_path):
    spec, remote, _ = make_release(monkeypatch, tmp_path)
    out = tmp_path / "out"
    first = quiet(spec, out)
    before = {path.name: path.stat().st_mtime_ns for path in (out / "raw_structures").iterdir()}
    monkeypatch.setattr(dl, "_open_remote", lambda url, offset: pytest.fail("must not download again"))
    second = quiet(spec, out, resume=True)
    assert second == first
    assert before == {path.name: path.stat().st_mtime_ns for path in (out / "raw_structures").iterdir()}
    (out / "raw_structures" / "3def_B.cif").unlink()
    with pytest.raises(dl.Rna3dbError, match="missing"):
        quiet(spec, out, resume=True)


def test_resume_after_partial_extraction_keeps_finished_files(monkeypatch, tmp_path):
    spec, _, _ = make_release(monkeypatch, tmp_path)
    out = tmp_path / "out"
    quiet(spec, out, keep_archives=True)
    kept = out / "raw_structures" / "1abc_A.cif"
    marker = kept.stat().st_mtime_ns
    (out / "raw_structures" / "3def_B.cif").unlink()
    (out / "raw_structures" / "2xyz_mR.cif.tmp").write_bytes(b"torn")
    record = json.loads((out / dl.PROVENANCE_NAME).read_text())
    record["status"] = "started"
    (out / dl.PROVENANCE_NAME).write_text(json.dumps(record))
    again = quiet(spec, out, resume=True)
    assert kept.stat().st_mtime_ns == marker
    assert again["extraction"]["kept_existing"] == 3 and again["extraction"]["written"] == 1
    assert not list((out / "raw_structures").glob("*.tmp"))


def test_case_insensitive_file_system_is_detected(monkeypatch, tmp_path):
    probes = {}

    def write_text(self, text, *args, **kwargs):
        probes[self.name.lower()] = text

    def read_text(self, *args, **kwargs):
        return probes[self.name.lower()]

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(dl.Rna3dbError, match="case-insensitive"):
        dl.require_case_sensitive(tmp_path / "raw")
    assert not list((tmp_path / "raw").iterdir())

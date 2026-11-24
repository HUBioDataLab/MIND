#!/usr/bin/env python3
"""Download and unpack one RNA3DB release into single-chain mmCIF files.

RNA3DB (https://github.com/marcellszi/rna3db) publishes every release as a
``rna3db-mmcifs`` tarball (one mmCIF per chain) and a ``rna3db-jsons`` tarball
(``parse/filter/cluster/split.json``). This tool fetches both, verifies them,
and writes the layout that ``data/rna/create_rna_manifest.py`` and
``make build-edge-free DOMAIN=rna`` expect::

    <out-dir>/raw_structures/<pdb>_<chain>.cif
    <out-dir>/rna3db-jsons/{split,filter,parse,cluster}.json
    <out-dir>/download_provenance.json

Chain-id encoding. The tarball names a chain with lowercase letters by wrapping
each lowercase run in dashes (``7taw_-m-.cif`` is chain ``m``) so that archives
survive case-insensitive file systems. Files are written under their decoded
names (``7taw_m.cif``); an extraction therefore needs a case-sensitive file
system, because chains such as ``A`` and ``a`` of one entry coexist. This is
checked before any file is written.

Integrity. Each archive size is pinned per release. SHA-256 is pinned where the
release host published a digest; for the other releases the digest is computed
on the first download, recorded in the provenance file, and a later run that
sees another digest is refused. The extracted chain-name set must equal the
chain set of the release's ``split.json`` (fail loud on any difference).

The tool never prompts and never overwrites: ``--out-dir`` must be new or empty
unless ``--resume`` continues an earlier run of the same release.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

BASE_URL = "https://github.com/marcellszi/rna3db/releases/download"
PROVENANCE_NAME = "download_provenance.json"
JSON_NAMES = ("split.json", "filter.json", "parse.json", "cluster.json")
USER_AGENT = "mind-rna3db-downloader"
_COPY_BLOCK = 1 << 20


class Rna3dbError(RuntimeError):
    """Any integrity, layout or usage failure of the RNA3DB preparation."""


@dataclass(frozen=True)
class ReleaseSpec:
    """One RNA3DB release: asset names, pinned sizes/digests, expected chain count."""

    alias: str
    tag: str
    mmcifs_asset: str
    mmcifs_size: int
    mmcifs_sha256: Optional[str]
    jsons_asset: str
    jsons_size: int
    jsons_sha256: Optional[str]
    chains: int


# Sizes and digests are the release-asset values reported by the GitHub release API
# (the API publishes SHA-256 digests only for recent assets); chain counts are the
# chain counts of each release's split.json.
RELEASES: dict[str, ReleaseSpec] = {
    spec.alias: spec
    for spec in (
        ReleaseSpec("2024-01", "full-release", "rna3db-mmcifs.v2.tar.xz", 1788578744, None,
                    "rna3db-jsons.tar.gz", 5039350, None, 11176),
        ReleaseSpec("2024-04", "incremental-update", "rna3db-mmcifs.v2.tar.xz", 1646954288, None,
                    "rna3db-jsons.tar.gz", 5285023, None, 11622),
        ReleaseSpec("2024-12", "2024-12-04-full-release", "rna3db-mmcifs.tar.xz", 1808700240, None,
                    "rna3db-jsons.tar.gz", 5777021, None, 12892),
        ReleaseSpec("2025-10", "2025-10-01-incremental-release", "rna3db-mmcifs.tar.xz", 2063786736,
                    "a45624a136fcaf1b3016dfd76a86d67cf896905241f23bb05f31cc27c12f27f3",
                    "rna3db-jsons.tar.gz", 6522877,
                    "d4996a5f78ffa0453f9b817d49cde85de4f8c038655d8699f6a1eadb10270a01", 14774),
        ReleaseSpec("2026-01", "2026-01-05-full-release", "rna3db-mmcifs.tar.xz", 2146440832,
                    "4be4290ee40e1407e1356d804355ba5e4971da527baa5f381116bac327ec9067",
                    "rna3db-jsons.tar.gz", 6851100,
                    "820ba611042ef3e4ec1de7d00e739eefcbf1f78df1452252cc002e369babae8b", 15441),
    )
}


def resolve_release(name: str) -> ReleaseSpec:
    """Accept the short alias (``2025-10``) or the upstream tag."""
    if name in RELEASES:
        return RELEASES[name]
    for spec in RELEASES.values():
        if spec.tag == name:
            return spec
    known = ", ".join(f"{s.alias} ({s.tag})" for s in RELEASES.values())
    raise Rna3dbError(f"unknown release {name!r}; available: {known}")


# ---------------------------------------------------------------------------
# Chain-id decoding
# ---------------------------------------------------------------------------

_ENCODED_RUN = re.compile(r"-([a-z]+)-")
_LOWER_RUN = re.compile(r"([a-z]+)")
_ENTRY_ID = re.compile(r"[0-9][0-9a-z]{3}")
_CHAIN_ID = re.compile(r"[0-9A-Za-z]+")


def encode_chain_id(chain: str) -> str:
    """RNA3DB ``to_case_insensitive``: wrap every lowercase run in dashes."""
    return _LOWER_RUN.sub(r"-\1-", chain)


def decode_chain_id(encoded: str) -> str:
    """Inverse of :func:`encode_chain_id`; a name that does not round-trip is an error."""
    decoded = _ENCODED_RUN.sub(r"\1", encoded)
    if not _CHAIN_ID.fullmatch(decoded) or encode_chain_id(decoded) != encoded:
        raise Rna3dbError(f"chain id {encoded!r} is not a valid RNA3DB case-insensitive encoding")
    return decoded


def decode_member_stem(stem: str) -> str:
    """``7taw_-m-`` -> ``7taw_m``; validates the entry and chain syntax."""
    entry, separator, encoded_chain = stem.partition("_")
    if not separator or not _ENTRY_ID.fullmatch(entry):
        raise Rna3dbError(f"unexpected structure file name {stem!r}")
    return f"{entry}_{decode_chain_id(encoded_chain)}"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_COPY_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def check_member_name(name: str) -> str:
    """Reject absolute paths, parent references and backslashes in a tar member name."""
    parts = name.split("/")
    if name.startswith("/") or ".." in parts or "\\" in name or "\0" in name:
        raise Rna3dbError(f"unsafe tar member name {name!r}")
    return parts[-1]


def require_case_sensitive(directory: Path) -> None:
    """Chains ``A`` and ``a`` of one entry must be two files."""
    directory.mkdir(parents=True, exist_ok=True)
    upper = directory / ".case_probe_A"
    lower = directory / ".case_probe_a"
    try:
        upper.write_text("A")
        lower.write_text("a")
        if upper.read_text() != "A":
            raise Rna3dbError(
                f"{directory} is on a case-insensitive file system; RNA3DB chain ids that differ "
                "only by case (for example 1abc_A and 1abc_a) would overwrite each other"
            )
    finally:
        for probe in (upper, lower):
            if probe.exists():
                probe.unlink()


def chains_of_split(split: dict) -> dict[str, str]:
    """Map every chain name of a release ``split.json`` to its set name; names must be unique."""
    names: dict[str, str] = {}
    try:
        for set_name, components in split.items():
            for component in components.values():
                for representative in component.values():
                    for name in representative:
                        if name in names:
                            raise Rna3dbError(f"chain {name!r} occurs twice in split.json")
                        names[name] = set_name
    except (AttributeError, TypeError) as error:
        raise Rna3dbError(
            f"split.json does not have the set/component/representative/chain layout: {error}"
        ) from error
    return names


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _open_remote(url: str, offset: int):
    """Open ``url`` from byte ``offset``; replaced in tests."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    return urllib.request.urlopen(request, timeout=60)


def download_file(url: str, destination: Path, *, expected_size: int, retries: int) -> None:
    """Stream ``url`` into ``destination`` through ``<destination>.part``, resuming after errors.

    The file only appears under its final name once its size equals ``expected_size``.
    """
    if destination.exists():
        if destination.stat().st_size != expected_size:
            raise Rna3dbError(f"{destination} has {destination.stat().st_size} bytes, expected {expected_size}")
        return
    part = destination.with_name(destination.name + ".part")
    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        if offset > expected_size:
            part.unlink()
            raise Rna3dbError(f"{part} was larger than the pinned size {expected_size}; the file was removed")
        if offset < expected_size:
            try:
                with _open_remote(url, offset) as response:
                    status = getattr(response, "status", 206 if offset else 200)
                    if offset and status != 206:
                        offset = 0  # the server ignored the range request; restart
                    elif status not in (200, 206):
                        raise Rna3dbError(f"unexpected HTTP status {status} for {url}")
                    with open(part, "ab" if offset else "wb") as handle:
                        shutil.copyfileobj(response, handle, _COPY_BLOCK)
            except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code in (400, 401, 403, 404):
                    raise Rna3dbError(f"{url}: HTTP {error.code}") from error
                last_error = error
                print(f"  download interrupted ({type(error).__name__}: {error}); "
                      f"attempt {attempt + 1}/{retries + 1}", flush=True)
                time.sleep(min(2.0 * 2 ** attempt, 30.0))
                continue
        size = part.stat().st_size
        if size > expected_size:
            part.unlink()
            raise Rna3dbError(f"{url} served {size} bytes, the pinned size is {expected_size}; the file was removed")
        if size == expected_size:
            os.replace(part, destination)
            return
        last_error = Rna3dbError(f"{part} has {part.stat().st_size} bytes, expected {expected_size}")
    raise Rna3dbError(f"could not download {url}: {last_error}") from last_error


def verify_digest(
    path: Path,
    *,
    pinned: Optional[str],
    recorded: Optional[str],
    cli_expected: Optional[str] = None,
) -> str:
    """Return the SHA-256 of ``path`` after checking every available reference."""
    references = {"pinned": pinned, "recorded by an earlier run": recorded, "--expected-sha256": cli_expected}
    values = {value.lower() for value in references.values() if value}
    if len(values) > 1:
        raise Rna3dbError(f"conflicting SHA-256 references for {path.name}: {references}")
    actual = sha256_file(path)
    if values and actual not in values:
        path.unlink()
        raise Rna3dbError(
            f"SHA-256 mismatch for {path.name}: got {actual}, expected {next(iter(values))}; "
            "the corrupted file was removed"
        )
    return actual


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_jsons(archive: Path, destination: Path) -> dict[str, str]:
    """Copy the release JSON files (by base name) out of the archive; ``split.json`` is required."""
    destination.mkdir(parents=True, exist_ok=True)
    found: dict[str, str] = {}
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if member.isdir():
                continue
            base = check_member_name(member.name)
            if base.startswith("._"):
                continue
            if not member.isreg():
                raise Rna3dbError(f"{archive.name}: unsupported member type for {member.name!r}")
            if base not in JSON_NAMES:
                continue
            if base in found:
                raise Rna3dbError(f"{archive.name}: {base} occurs twice")
            target = destination / base
            tmp = target.with_name(base + ".tmp")
            with tar.extractfile(member) as source, open(tmp, "wb") as sink:
                shutil.copyfileobj(source, sink, _COPY_BLOCK)
            os.replace(tmp, target)
            found[base] = sha256_file(target)
    if "split.json" not in found:
        raise Rna3dbError(f"{archive.name} contains no split.json")
    return found


def extract_mmcifs(archive: Path, raw_dir: Path) -> dict:
    """Write every ``*.cif`` member to ``raw_dir/<decoded name>.cif``; return counters and the name set.

    A file that already exists with the member's size is kept (resume); files are written
    through a temporary name so an interrupted run never leaves a truncated structure.
    """
    require_case_sensitive(raw_dir)
    for leftover in raw_dir.glob("*.tmp"):
        leftover.unlink()
    names: set[str] = set()
    counters = {"members": 0, "cif_files": 0, "encoded_names": 0, "appledouble_skipped": 0,
                "kept_existing": 0, "written": 0, "bytes": 0}
    with tarfile.open(archive, "r|xz") as tar:
        for member in tar:
            counters["members"] += 1
            if member.isdir():
                continue
            base = check_member_name(member.name)
            if base.startswith("._"):
                counters["appledouble_skipped"] += 1
                continue
            if not member.isreg():
                raise Rna3dbError(f"{archive.name}: unsupported member type for {member.name!r}")
            if not base.endswith(".cif"):
                raise Rna3dbError(f"{archive.name}: unexpected non-mmCIF member {member.name!r}")
            if member.size == 0:
                raise Rna3dbError(f"{archive.name}: empty structure file {member.name!r}")
            stem = base[:-4]
            name = decode_member_stem(stem)
            if name in names:
                raise Rna3dbError(f"{archive.name}: chain {name!r} occurs twice")
            names.add(name)
            counters["cif_files"] += 1
            counters["encoded_names"] += int(stem != name)
            counters["bytes"] += member.size
            target = raw_dir / f"{name}.cif"
            if target.exists() and target.stat().st_size == member.size:
                counters["kept_existing"] += 1
                continue
            tmp = target.with_name(target.name + ".tmp")
            with tar.extractfile(member) as source, open(tmp, "wb") as sink:
                shutil.copyfileobj(source, sink, _COPY_BLOCK)
            os.replace(tmp, target)
            counters["written"] += 1
    counters["names"] = names
    return counters


def verify_raw_dir(raw_dir: Path, split_names: dict[str, str], expected_chains: Optional[int]) -> int:
    """Fail unless ``raw_dir`` holds exactly the chains of ``split.json`` (and the pinned count)."""
    on_disk = {path.name for path in raw_dir.iterdir()}
    bad = sorted(name for name in on_disk if not name.endswith(".cif"))
    if bad:
        raise Rna3dbError(f"{raw_dir} contains files that are not mmCIF structures: {bad[:5]}")
    stems = {name[:-4] for name in on_disk}
    compare_name_sets(stems, split_names, expected_chains, where=str(raw_dir))
    return len(stems)


def compare_name_sets(found: set[str], split_names: dict[str, str], expected_chains: Optional[int], *, where: str) -> None:
    missing = sorted(set(split_names) - found)
    extra = sorted(found - set(split_names))
    if missing or extra:
        raise Rna3dbError(
            f"{where}: {len(found)} structures vs {len(split_names)} chains in split.json; "
            f"{len(missing)} missing (first {missing[:5]}), {len(extra)} not in split.json (first {extra[:5]})"
        )
    if expected_chains is not None and len(split_names) != expected_chains:
        raise Rna3dbError(
            f"split.json lists {len(split_names)} chains, the pinned count of this release is {expected_chains}"
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _load_provenance(out_dir: Path) -> Optional[dict]:
    path = out_dir / PROVENANCE_NAME
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def prepare_out_dir(out_dir: Path, spec: ReleaseSpec, resume: bool) -> Optional[dict]:
    """Enforce the never-overwrite rule; return the provenance of an earlier run when resuming."""
    if out_dir.exists() and not out_dir.is_dir():
        raise Rna3dbError(f"{out_dir} exists and is not a directory")
    populated = out_dir.exists() and any(out_dir.iterdir())
    if not populated:
        out_dir.mkdir(parents=True, exist_ok=True)
        return None
    if not resume:
        raise Rna3dbError(
            f"{out_dir} is not empty; refusing to overwrite. Use a new directory, or --resume to "
            "continue an earlier run of the same release"
        )
    previous = _load_provenance(out_dir)
    if previous is None:
        raise Rna3dbError(f"{out_dir} is not empty and has no {PROVENANCE_NAME}; cannot resume into it")
    if previous.get("release") != spec.alias:
        raise Rna3dbError(
            f"{out_dir} belongs to release {previous.get('release')!r}, not {spec.alias!r}"
        )
    return previous


def run(
    spec: ReleaseSpec,
    out_dir: Path,
    *,
    resume: bool = False,
    keep_archives: bool = False,
    expected_sha256: Optional[str] = None,
    retries: int = 5,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
) -> dict:
    """Download, verify, decode and extract one release; return the provenance record."""
    previous = prepare_out_dir(out_dir, spec, resume)
    if expected_sha256 and spec.mmcifs_sha256 and expected_sha256.lower() != spec.mmcifs_sha256:
        raise Rna3dbError("--expected-sha256 differs from the SHA-256 pinned for this release")
    raw_dir = out_dir / "raw_structures"
    json_dir = out_dir / "rna3db-jsons"
    download_dir = out_dir / "downloads"

    if previous is not None and previous.get("status") == "complete":
        split_names = chains_of_split(json.loads((json_dir / "split.json").read_text()))
        count = verify_raw_dir(raw_dir, split_names, spec.chains)
        log(f"release {spec.alias} already complete: {count} structures verified")
        return previous

    record: dict = {
        "release": spec.alias,
        "tag": spec.tag,
        "status": "started",
        "mmcifs_url": f"{BASE_URL}/{spec.tag}/{spec.mmcifs_asset}",
        "jsons_url": f"{BASE_URL}/{spec.tag}/{spec.jsons_asset}",
        "pinned_chains": spec.chains,
        "archives": {},
    }
    if previous is not None:
        record["archives"] = dict(previous.get("archives", {}))
    download_dir.mkdir(exist_ok=True)

    # 1. archives
    for kind, asset, size, pinned in (
        ("jsons", spec.jsons_asset, spec.jsons_size, spec.jsons_sha256),
        ("mmcifs", spec.mmcifs_asset, spec.mmcifs_size, spec.mmcifs_sha256),
    ):
        archive = download_dir / asset
        url = f"{BASE_URL}/{spec.tag}/{asset}"
        log(f"[{kind}] {url} ({size / 1e6:.0f} MB)")
        download_file(url, archive, expected_size=size, retries=retries)
        recorded = record["archives"].get(kind, {}).get("sha256")
        digest = verify_digest(
            archive, pinned=pinned, recorded=recorded,
            cli_expected=expected_sha256 if kind == "mmcifs" else None,
        )
        record["archives"][kind] = {
            "asset": asset, "bytes": size, "sha256": digest,
            "sha256_source": "pinned" if pinned else "computed on first download",
        }
        write_json_atomic(out_dir / PROVENANCE_NAME, record)
        log(f"[{kind}] sha256 {digest} ({record['archives'][kind]['sha256_source']})")

    # 2. release metadata and the chain list the structures must match
    json_hashes = extract_jsons(download_dir / spec.jsons_asset, json_dir)
    split_names = chains_of_split(json.loads((json_dir / "split.json").read_text()))
    per_set: dict[str, int] = {}
    for set_name in split_names.values():
        per_set[set_name] = per_set.get(set_name, 0) + 1
    if len(split_names) != spec.chains:
        raise Rna3dbError(
            f"split.json lists {len(split_names)} chains, the pinned count of release {spec.alias} is {spec.chains}"
        )

    # 3. structures
    log(f"[mmcifs] extracting into {raw_dir}")
    counters = extract_mmcifs(download_dir / spec.mmcifs_asset, raw_dir)
    names = counters.pop("names")
    compare_name_sets(names, split_names, spec.chains, where="archive")
    on_disk = verify_raw_dir(raw_dir, split_names, spec.chains)

    record.update({
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_json_chains": len(split_names),
        "split_json_chains_per_set": per_set,
        "json_sha256": json_hashes,
        "extraction": counters,
        "structures_on_disk": on_disk,
    })
    write_json_atomic(out_dir / PROVENANCE_NAME, record)
    if not keep_archives:
        shutil.rmtree(download_dir)
    log(f"done: {on_disk} structures ({counters['encoded_names']} lowercase-chain names decoded) in {raw_dir}")
    return record


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--release", required=True,
        help="release alias or upstream tag: " + ", ".join(f"{s.alias} ({s.tag})" for s in RELEASES.values()),
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="new or empty directory")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run of the same release (also a no-op re-check of a finished one)")
    parser.add_argument("--keep-archives", action="store_true", help="keep the downloaded tarballs")
    parser.add_argument("--expected-sha256", default=None,
                        help="SHA-256 of the mmCIF tarball for releases without a pinned digest")
    parser.add_argument("--retries", type=int, default=5, help="network retries per archive")
    args = parser.parse_args(argv)
    run(
        resolve_release(args.release), args.out_dir, resume=args.resume,
        keep_archives=args.keep_archives, expected_sha256=args.expected_sha256, retries=args.retries,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

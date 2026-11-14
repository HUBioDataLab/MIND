"""Leakage registry: which downstream test/valid identities did pretraining see?

A downstream task lists its split identities (UniProt accessions, PDB entries or
PDB chains). A *reference set* lists what the pretraining run used (the chunk
build plans ``id,split,chunk,position`` or a plain manifest of ids). This module
compares the two, reports ``seen`` and ``unseen`` ids separately and writes the
result into the task's ``provenance.json`` (summary) plus a per-id TSV.

Rules (fail loud, no silent "0 leaked"):

* Ids are normalised per namespace (``uniprot``, ``pdb_entry``, ``pdb_chain``, ``scop_domain``);
  a malformed id raises with its position, it is never dropped.
* Different namespaces are NOT comparable by string. A PDB-derived task against
  the AFDB (UniProt) protein pretraining set raises
  ``NamespaceMismatchError`` unless a ``pdb_chain -> uniprot`` map (for example
  SIFTS) is supplied; ids missing from the map are reported as ``unmapped``,
  never as ``unseen``.
* PDB-derived ids against a PDB-derived reference (RNA) are compared at the PDB
  entry level (the RNA pretraining split is by PDB entry); exact chain matches
  are reported in addition.
* Pretraining ``train`` membership means seen by the weights; ``val`` was used
  for checkpoint selection; ``test`` is held out. ``unseen`` means absent from
  all three. Id matching cannot see homologs: every report carries the caveat
  that sequence-similar proteins are not detected.

CLI (CPU, read-only on the references)::

    python -m downstream.common.leakage --task go_mf --reference protein_1m \\
        --query valid=splits/valid.txt --query test=splits/test.txt \\
        --query-namespace pdb_chain --id-map sifts_pdb_chain_uniprot.tsv \\
        --provenance <MIND_DATA_ROOT>/downstream/deepfri/go/provenance.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from data_loading.data_root import expand_data_root
from downstream.common.registry import git_commit, sha256_file

REFERENCES_YAML = Path(__file__).resolve().parents[1] / "leakage_references.yaml"
NAMESPACES = ("uniprot", "pdb_entry", "pdb_chain", "scop_domain")
PRETRAIN_SPLITS = ("train", "val", "test")
CAVEAT = ("Identity match only: homologous or sequence-similar proteins/RNAs that are not the same id are "
          "not detected; AFDB pretraining used cluster representatives, not cluster members.")

_UNIPROT = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")
_AFDB_MODEL = re.compile(r"^AF-([A-Z0-9]+)-F\d+(?:-model_v\d+)?$", re.IGNORECASE)
_PDB_ENTRY = re.compile(r"^[0-9][a-z0-9]{3}$")
# SCOPe domain id (sid): 'd' + PDB entry + one chain character ('_' = no chain id, '.' = several chains) + domain character
_SCOP_DOMAIN = re.compile(r"^d[0-9][a-z0-9]{3}[a-z0-9_.][a-z0-9_]$")
_CHAIN_SEPARATORS = ("_", "-", ".")


class LeakageError(ValueError):
    """Contract violation in a leakage reference or query."""


class NamespaceMismatchError(LeakageError):
    """Query and reference use identity namespaces that cannot be compared by string."""


# ---------------------------------------------------------------------------
# Identity normalisation
# ---------------------------------------------------------------------------

def normalize_id(raw: str, namespace: str) -> str:
    """Canonical string of ``raw`` in ``namespace`` (uniprot upper, PDB lower; chain case kept)."""
    if namespace not in NAMESPACES:
        raise LeakageError(f"unknown namespace {namespace!r}; choose from {NAMESPACES}")
    if not isinstance(raw, str) or not raw.strip():
        raise LeakageError(f"empty {namespace} id {raw!r}")
    text = raw.strip()
    if namespace == "uniprot":
        match = _AFDB_MODEL.match(text)
        accession = (match.group(1) if match else text).upper()
        if not _UNIPROT.match(accession):
            raise LeakageError(f"{raw!r} is not a UniProt accession")
        return accession
    if namespace == "pdb_entry":
        entry = text.lower()
        if not _PDB_ENTRY.match(entry):
            raise LeakageError(f"{raw!r} is not a 4-character PDB entry id")
        return entry
    if namespace == "scop_domain":
        sid = text.lower()
        if not _SCOP_DOMAIN.match(sid):
            raise LeakageError(f"{raw!r} is not a SCOPe domain id such as d1v4wb_")
        return sid
    separator = next((candidate for candidate in _CHAIN_SEPARATORS if candidate in text), "-")
    entry, sep, chain = text.partition(separator)
    entry = entry.lower()
    if not sep or not _PDB_ENTRY.match(entry) or not chain or any(c in chain for c in _CHAIN_SEPARATORS):
        raise LeakageError(f"{raw!r} is not <pdb>_<chain>, <PDB>-<chain> or <pdb>.<chain>")
    return f"{entry}_{chain}"


def entry_of(chain_id: str) -> str:
    """PDB entry of a normalised ``pdb_chain`` id."""
    return chain_id.partition("_")[0]


# ---------------------------------------------------------------------------
# Reference sets
# ---------------------------------------------------------------------------

@dataclass
class ReferenceSet:
    """Pretraining identities grouped by pretraining split (``all`` for plain manifests)."""

    name: str
    namespace: str
    path: str
    sha256: str
    ids_by_split: Dict[str, frozenset]
    has_splits: bool
    description: str = ""

    def count(self) -> int:
        return sum(len(ids) for ids in self.ids_by_split.values())

    def entries_by_split(self) -> Dict[str, frozenset]:
        if self.namespace != "pdb_chain":
            raise LeakageError(f"reference {self.name!r} has namespace {self.namespace}, not pdb_chain")
        return {split: frozenset(entry_of(i) for i in ids) for split, ids in self.ids_by_split.items()}


def load_reference(name: str, spec: Mapping, *, hash_file: bool = True) -> ReferenceSet:
    """Read one reference from its yaml spec: ``{kind, path, namespace, id_column[, split_column]}``."""
    for key in ("kind", "path", "namespace", "id_column"):
        if key not in spec:
            raise LeakageError(f"reference {name!r} lacks {key!r}")
    kind, namespace, path = spec["kind"], spec["namespace"], Path(spec["path"])
    if kind not in ("plan", "manifest"):
        raise LeakageError(f"reference {name!r}: kind must be plan or manifest, got {kind!r}")
    if namespace not in NAMESPACES:
        raise LeakageError(f"reference {name!r}: unknown namespace {namespace!r}")
    if not path.is_file():
        raise FileNotFoundError(f"reference {name!r}: {path} does not exist")
    id_column = spec["id_column"]
    split_column = spec.get("split_column") if kind == "plan" else None
    if kind == "plan" and not split_column:
        raise LeakageError(f"reference {name!r}: a plan needs split_column")
    by_split: Dict[str, set] = {}
    seen_ids: Dict[str, str] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        needed = [id_column] + ([split_column] if split_column else [])
        missing = [c for c in needed if c not in (reader.fieldnames or [])]
        if missing:
            raise LeakageError(f"{path}: missing columns {missing}; header {reader.fieldnames}")
        for number, row in enumerate(reader, start=2):
            try:
                identity = normalize_id(row[id_column], namespace)
            except LeakageError as exc:
                raise LeakageError(f"{path}:{number}: {exc}") from None
            split = row[split_column].strip() if split_column else "all"
            if kind == "plan" and split not in PRETRAIN_SPLITS:
                raise LeakageError(f"{path}:{number}: unknown pretraining split {split!r}")
            if identity in seen_ids:
                raise LeakageError(f"{path}:{number}: duplicate id {identity} (also in split {seen_ids[identity]})")
            seen_ids[identity] = split
            by_split.setdefault(split, set()).add(identity)
    if not seen_ids:
        raise LeakageError(f"{path}: reference is empty")
    return ReferenceSet(
        name=name, namespace=namespace, path=str(path),
        sha256=sha256_file(path) if hash_file else "not_hashed",
        ids_by_split={s: frozenset(v) for s, v in by_split.items()}, has_splits=kind == "plan",
        description=str(spec.get("description", "")),
    )


def load_references_config(path: Path = REFERENCES_YAML) -> Dict[str, dict]:
    with open(path) as handle:
        config = expand_data_root(yaml.safe_load(handle))
    if not isinstance(config, dict) or "references" not in config:
        raise LeakageError(f"{path}: expected a top-level 'references' mapping")
    return config["references"]


def load_task_queries(path: Path = REFERENCES_YAML) -> Dict[str, dict]:
    """Per-task identity format and the references it may be compared with (the ``queries`` section); validated."""
    with open(path) as handle:
        config = expand_data_root(yaml.safe_load(handle))
    queries = (config or {}).get("queries")
    if not isinstance(queries, dict) or not queries:
        raise LeakageError(f"{path}: expected a non-empty top-level 'queries' mapping")
    references = config.get("references", {})
    for task, spec in queries.items():
        for key in ("namespace", "id_example", "id_source", "compare_with"):
            if key not in spec:
                raise LeakageError(f"{path}: queries.{task} lacks {key!r}")
        if spec["namespace"] not in NAMESPACES:
            raise LeakageError(f"{path}: queries.{task} has unknown namespace {spec['namespace']!r}")
        normalize_id(spec["id_example"], spec["namespace"])  # the documented example must itself be valid
        unknown = [name for name in spec["compare_with"] if name not in references]
        if unknown:
            raise LeakageError(f"{path}: queries.{task} names unknown references {unknown}")
    return queries


def read_id_map(path: Path, *, source_namespace: str = "pdb_chain", target_namespace: str = "uniprot") -> Dict[str, str]:
    """Two-column TSV ``query_id<TAB>reference_id`` (header optional, lines with ``#`` skipped)."""
    mapping: Dict[str, str] = {}
    with open(path) as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise LeakageError(f"{path}:{number}: expected 2 tab-separated columns")
            try:
                key, value = normalize_id(parts[0], source_namespace), normalize_id(parts[1], target_namespace)
            except LeakageError:
                if number == 1:  # header line
                    continue
                raise
            if key in mapping and mapping[key] != value:
                raise LeakageError(f"{path}:{number}: {key} maps to both {mapping[key]} and {value}")
            mapping[key] = value
    if not mapping:
        raise LeakageError(f"{path}: id map is empty")
    return mapping


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class QueryReport:
    query_name: str
    reference: str
    query_namespace: str
    reference_namespace: str
    method: str                      # direct | entry_level | id_map
    total: int
    status_by_id: Dict[str, str]     # id -> seen_train | seen_val | seen_test | unseen | unmapped
    counts: Dict[str, int]
    chain_level_seen: Optional[int] = None
    extra: dict = field(default_factory=dict)

    @property
    def seen(self) -> int:
        return self.counts.get("seen_train", 0)

    @property
    def touched(self) -> int:
        """Seen by the weights or used for checkpoint selection."""
        return self.counts.get("seen_train", 0) + self.counts.get("seen_val", 0)

    def summary(self) -> dict:
        return {
            "query": self.query_name, "reference": self.reference, "method": self.method,
            "query_namespace": self.query_namespace, "reference_namespace": self.reference_namespace,
            "total": self.total, "counts": dict(self.counts), "seen_train": self.seen,
            "touched_by_train_or_val": self.touched,
            "unseen": self.counts.get("unseen", 0), "unmapped": self.counts.get("unmapped", 0),
            "chain_level_seen_train": self.chain_level_seen, **self.extra,
        }


def _classify(identity: str, by_split: Mapping[str, frozenset], has_splits: bool) -> str:
    if not has_splits:
        return "seen_train" if identity in by_split["all"] else "unseen"
    for split in PRETRAIN_SPLITS:
        if identity in by_split.get(split, frozenset()):
            return f"seen_{split}"
    return "unseen"


def compare(query_name: str, raw_ids: Sequence[str], query_namespace: str, reference: ReferenceSet,
            *, id_map: Optional[Mapping[str, str]] = None) -> QueryReport:
    """Classify every id of one downstream split against ``reference``."""
    if query_namespace not in NAMESPACES:
        raise LeakageError(f"unknown query namespace {query_namespace!r}")
    if not raw_ids:
        raise LeakageError(f"query {query_name!r} has no ids")
    ids: List[str] = []
    for number, raw in enumerate(raw_ids, start=1):
        try:
            ids.append(normalize_id(raw, query_namespace))
        except LeakageError as exc:
            raise LeakageError(f"query {query_name!r} id #{number}: {exc}") from None
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise LeakageError(f"query {query_name!r} has duplicate ids, first {duplicates}")

    status: Dict[str, str] = {}
    chain_level = None
    if query_namespace in ("pdb_chain", "pdb_entry") and reference.namespace == "pdb_chain":
        method = "entry_level"
        by_split = reference.entries_by_split()
        lookup = {i: (entry_of(i) if query_namespace == "pdb_chain" else i) for i in ids}
        if query_namespace == "pdb_chain":
            chain_level = sum(
                _classify(i, reference.ids_by_split, reference.has_splits) == "seen_train" for i in ids)
    elif query_namespace == reference.namespace:
        method, lookup, by_split = "direct", {i: i for i in ids}, reference.ids_by_split
    elif query_namespace in ("pdb_chain", "pdb_entry") and reference.namespace == "uniprot":
        if id_map is None:
            raise NamespaceMismatchError(
                f"query {query_name!r} uses {query_namespace} ids but reference {reference.name!r} uses UniProt "
                "accessions: they cannot be compared by string. Supply --id-map (pdb_chain -> uniprot, e.g. "
                "SIFTS) or screen by sequence identity; refusing to report 0 leaked.")
        if query_namespace != "pdb_chain":
            raise NamespaceMismatchError("an id map is defined per pdb_chain, not per pdb_entry")
        method, by_split = "id_map", reference.ids_by_split
        lookup = {i: id_map.get(i) for i in ids}
    elif query_namespace == "scop_domain":
        raise NamespaceMismatchError(
            f"query {query_name!r} uses SCOPe domain ids (e.g. d1v4wb_) but reference {reference.name!r} uses "
            f"{reference.namespace} ids; no scop_domain -> {reference.namespace} map is implemented, so nothing is "
            "compared and no '0 leaked' is reported. Screen by sequence identity instead (processed splits store the "
            "sequences).")
    else:
        raise NamespaceMismatchError(
            f"cannot compare query namespace {query_namespace!r} with reference namespace {reference.namespace!r}")

    for identity in ids:
        target = lookup[identity]
        status[identity] = "unmapped" if target is None else _classify(target, by_split, reference.has_splits)
    counts: Dict[str, int] = {}
    for value in status.values():
        counts[value] = counts.get(value, 0) + 1
    return QueryReport(query_name, reference.name, query_namespace, reference.namespace, method,
                       len(ids), status, counts, chain_level)


# ---------------------------------------------------------------------------
# Provenance output
# ---------------------------------------------------------------------------

def write_report(task: str, reports: Sequence[QueryReport], reference: ReferenceSet, provenance: Path,
                 *, query_files: Mapping[str, Path], id_map_path: Optional[Path] = None) -> Path:
    """Append a ``leakage`` record to ``provenance`` and write the per-id TSV next to it.

    Existing TSVs are never overwritten; an existing provenance file must be valid JSON.
    """
    provenance = Path(provenance)
    table = provenance.parent / "leakage" / f"{task}__{reference.name}.tsv"
    if table.exists():
        raise FileExistsError(f"refusing to overwrite {table}")
    table.parent.mkdir(parents=True, exist_ok=True)
    partial = table.with_name(table.name + ".partial")
    with open(partial, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("query", "id", "status"))
        for report in reports:
            for identity, status in report.status_by_id.items():
                writer.writerow((report.query_name, identity, status))
    os.replace(partial, table)
    record = {
        "task": task, "created_utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "reference": {"name": reference.name, "namespace": reference.namespace, "path": reference.path,
                      "sha256": reference.sha256, "ids": reference.count(),
                      "has_pretrain_splits": reference.has_splits},
        "queries": {name: {"path": str(path), "sha256": sha256_file(Path(path))} for name, path in query_files.items()},
        "id_map": ({"path": str(id_map_path), "sha256": sha256_file(Path(id_map_path))} if id_map_path else None),
        "results": [report.summary() for report in reports],
        "per_id_table": str(table), "caveat": CAVEAT,
    }
    existing = {}
    if provenance.exists():
        existing = json.loads(provenance.read_text())
        if not isinstance(existing, dict):
            raise LeakageError(f"{provenance}: expected a JSON object")
    existing.setdefault("leakage", []).append(record)
    partial = provenance.with_name(provenance.name + ".partial")
    partial.write_text(json.dumps(existing, indent=1))
    os.replace(partial, provenance)
    return table


def format_report(report: QueryReport) -> str:
    c = report.counts
    return (f"[{report.query_name}] vs {report.reference} ({report.method}): total {report.total}, "
            f"seen_train {c.get('seen_train', 0)}, seen_val {c.get('seen_val', 0)}, "
            f"seen_test {c.get('seen_test', 0)}, unseen {c.get('unseen', 0)}, unmapped {c.get('unmapped', 0)}")


def read_id_list(path: Path) -> List[str]:
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--reference", required=True, help="name in leakage_references.yaml")
    parser.add_argument("--references-yaml", type=Path, default=REFERENCES_YAML)
    parser.add_argument("--query", action="append", required=True, metavar="NAME=PATH",
                        help="one id list per downstream split, e.g. test=splits/test.txt (repeatable)")
    parser.add_argument("--query-namespace", required=True, choices=NAMESPACES)
    parser.add_argument("--id-map", type=Path, default=None)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args(argv)
    specs = load_references_config(args.references_yaml)
    if args.reference not in specs:
        parser.error(f"unknown reference {args.reference!r}; choose from {sorted(specs)}")
    reference = load_reference(args.reference, specs[args.reference])
    id_map = read_id_map(args.id_map) if args.id_map else None
    queries: Dict[str, Path] = {}
    for item in args.query:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            parser.error(f"--query expects NAME=PATH, got {item!r}")
        queries[name] = Path(path)
    try:
        reports = [compare(name, read_id_list(path), args.query_namespace, reference, id_map=id_map)
                   for name, path in queries.items()]
        table = write_report(args.task, reports, reference, args.provenance, query_files=queries,
                             id_map_path=args.id_map)
    except LeakageError as exc:
        print(f"LEAKAGE CHECK FAILED: {exc}", file=sys.stderr)
        return 2
    for report in reports:
        print(format_report(report))
    print(f"per-id table: {table}\nprovenance: {args.provenance}\ncaveat: {CAVEAT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

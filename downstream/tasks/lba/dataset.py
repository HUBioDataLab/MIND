"""ATOM3D LBA: LMDB reading, pocket selection, complex -> graph processing with QC.

``build_processed`` turns one ATOM3D split LMDB into
``processed/<tag>/<policy_tag>/<split>.pt``: a compact edge-free pocket sample
plus the ligand's heavy atoms and the pK target, per complex. A split whose
failed/dropped fraction exceeds ``qc.max_failed_complex_fraction`` gets only its
QC report and makes the run exit non-zero.

The LMDB is read without the ``atom3d`` package. Each value is
``gzip.compress(json.dumps(item))`` where the DataFrame entries are
``DataFrame.to_dict(orient="split")`` dicts (``columns``/``index``/``data``).
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from downstream.common import complex_graph as cg
from downstream.common.graph_builder import build_node_sample
from downstream.common.missing_atoms import MissingAtomError, apply_policy, parse_policy
from downstream.common.structure_io import STANDARD_RESIDUES, ChainRecord, ResidueRecord, THREE_TO_ONE
from data_loading.pt_schema import encode_edge_free_sample

TASK_YAML = Path(__file__).with_name("task.yaml")
KIND = "regression"
TARGET_NAME = "neglog_aff"
HYDROGEN_ELEMENTS = frozenset({"H", "D"})
REQUIRED_KEYS = (
    "task", "modality", "data_root", "kind", "source", "sources", "split_tags",
    "primary_split_tag", "splits", "expected_counts", "metric", "representation",
    "representations", "pocket_cutoff", "pocket_mode", "max_atoms", "ligand", "structure",
    "missing_atoms", "min_residues", "min_ligand_atoms", "contact_atoms", "qc", "graph", "embedding",
    "probe", "finetune",
)
POS_CODE_INDEX = {code: index for index, code in enumerate(cg.COMPLEX_POS_CODE_VOCABULARY)}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_task_config(path: Path = TASK_YAML, *, data_root: Optional[Path] = None) -> dict:
    with open(path) as handle:
        config = yaml.safe_load(handle)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if config["source"] not in config["sources"]:
        raise ValueError(f"source {config['source']!r} is not configured under sources")
    if config["representation"] not in config["representations"]:
        raise ValueError(f"representation must be one of {config['representations']}")
    if config["contact_atoms"] not in ("all", "heavy"):
        raise ValueError("contact_atoms must be 'all' or 'heavy'")
    if config["pocket_mode"] != "whole_residues":
        raise ValueError("pocket_mode: only 'whole_residues' is implemented")
    if config["ligand"]["hydrogens"] != "drop":
        raise ValueError("ligand.hydrogens: only 'drop' is implemented")
    if config["ligand"]["bonds"] not in ("distance_proxy", "explicit"):
        raise ValueError("ligand.bonds must be distance_proxy or explicit")
    if config["ligand"]["bonds"] == "explicit":
        raise ValueError(
            "ligand.bonds=explicit sets has_explicit_bonds for the whole graph and makes "
            "_compute_chemical_coordination drop the <=1.85 A proxy for every pocket atom; "
            "it is refused until that interaction is handled per entity"
        )
    if config["primary_split_tag"] not in config["split_tags"]:
        raise ValueError("primary_split_tag is not in split_tags")
    if float(config["pocket_cutoff"]) <= 0:
        raise ValueError("pocket_cutoff must be positive")
    if int(config["max_atoms"]) < 1:
        raise ValueError("max_atoms must be positive")
    if data_root is not None:
        config["data_root"] = str(data_root)
    return config


def task_paths(config: dict) -> Dict[str, Path]:
    root = Path(config["data_root"])
    return {
        "root": root, "raw": root / "raw", "extracted": root / "raw" / "extracted",
        "processed": root / "processed", "provenance": root / "provenance.json",
    }


def processed_dir(config: dict, tag: Optional[str] = None) -> Path:
    policy = parse_policy(config["missing_atoms"])
    tag = tag or config["primary_split_tag"]
    if tag not in config["split_tags"]:
        raise ValueError(f"unknown split tag {tag!r}; expected one of {config['split_tags']}")
    return task_paths(config)["processed"] / tag / (
        f"{policy.tag}_cut{float(config['pocket_cutoff']):g}_max{int(config['max_atoms'])}"
    )


def lmdb_dir(config: dict, tag: str, split: str) -> Path:
    name = config["sources"][config["source"]]["files"][tag]["name"]
    stem = name[len("LBA-"):-len(".tar.gz")]  # split-by-sequence-identity-30
    return task_paths(config)["extracted"] / stem / "data" / split


# ---------------------------------------------------------------------------
# LMDB reading (no atom3d dependency)
# ---------------------------------------------------------------------------

def _lmdb_count(txn, path: Path) -> int:
    """``num_examples`` of an ATOM3D LMDB.

    ``stat()['entries']`` is three too large: the database also holds the
    ``id_to_idx``, ``num_examples`` and ``serialization_format`` metadata keys.
    """
    fmt = txn.get(b"serialization_format")
    if fmt is None or fmt.decode() != "json":
        raise ValueError(f"{path}: serialization_format is {fmt!r}, expected b'json'")
    raw = txn.get(b"num_examples")
    if raw is None:
        raise ValueError(f"{path}: LMDB has no num_examples key")
    count = int(raw.decode())
    entries = int(txn.stat()["entries"])
    if entries != count + 3:
        raise ValueError(f"{path}: {entries} LMDB keys for num_examples={count} (expected {count + 3})")
    return count


def lmdb_length(path: Path) -> int:
    import lmdb

    env = lmdb.open(str(path), readonly=True, lock=False, max_readers=1)
    try:
        with env.begin() as txn:
            return _lmdb_count(txn, path)
    finally:
        env.close()


def iter_lmdb(path: Path) -> Iterator[Tuple[int, dict]]:
    """Yield ``(index, item)`` for keys ``0 .. num_examples-1`` in numeric order."""
    import lmdb

    env = lmdb.open(str(path), readonly=True, lock=False, max_readers=1)
    try:
        with env.begin() as txn:
            count = _lmdb_count(txn, path)
            for index in range(count):
                raw = txn.get(str(index).encode())
                if raw is None:
                    raise KeyError(f"{path}: LMDB declares {count} examples but has no key {index!r}")
                yield index, json.loads(gzip.decompress(raw).decode())
    finally:
        env.close()


def frame(item: dict, name: str) -> Tuple[Dict[str, int], List[list]]:
    """``(column -> position, rows)`` of one ``to_dict(orient='split')`` DataFrame."""
    value = item.get(name)
    if not isinstance(value, dict) or "columns" not in value or "data" not in value:
        raise ValueError(f"entry {item.get('id', '?')}: {name!r} is not a split-oriented DataFrame")
    return {str(column): index for index, column in enumerate(value["columns"])}, value["data"]


# ---------------------------------------------------------------------------
# Pocket selection
# ---------------------------------------------------------------------------

@dataclass
class ResidueCandidate:
    chain: str
    chain_ordinal: int
    sequence_index: int      # position inside its own chain, in file order
    resseq: int
    icode: str
    source_resname: str
    resname: str
    atoms: "OrderedDict[str, Tuple[str, np.ndarray]]"
    # Every atom of the residue as read, hydrogens included. Only used for the
    # 6 A contact test: ATOM3D built `atoms_pocket` from the hydrogen-bearing
    # PDBbind frames, and including H here reproduces their residue set exactly.
    contact_positions: List[List[float]] = field(default_factory=list)
    min_ligand_distance: float = float("inf")

    @property
    def heavy_atoms(self) -> int:
        return len(self.atoms)

    @property
    def label(self) -> str:
        return f"{self.chain}:{self.source_resname}{self.resseq}{self.icode}"


@dataclass
class ComplexRecord:
    complex_id: str
    target: float
    residues: List[ResidueCandidate]
    ligand: cg.LigandAtoms
    report: dict = field(default_factory=dict)


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def read_ligand(item: dict) -> Tuple[cg.LigandAtoms, np.ndarray]:
    """``(heavy ligand atoms, every ligand atom position for the contact test)``."""
    columns, rows = frame(item, "atoms_ligand")
    elements, positions, contact = [], [], []
    for row in rows:
        element = _clean(row[columns["element"]]).upper()
        xyz = [float(row[columns[axis]]) for axis in ("x", "y", "z")]
        contact.append(xyz)
        if element in HYDROGEN_ELEMENTS:
            continue
        elements.append(element)
        positions.append(xyz)
    if not elements:
        raise ValueError("ligand has no heavy atom")
    return cg.ligand_from_rows(elements, positions), np.asarray(contact, dtype=np.float64)


def read_protein_residues(item: dict, *, mse_to_met: bool) -> Tuple[List[ResidueCandidate], Counter]:
    """Standard protein residues of ``atoms_protein`` in file order, heavy atoms only."""
    columns, rows = frame(item, "atoms_protein")
    counts: Counter = Counter()
    groups: "OrderedDict[tuple, ResidueCandidate]" = OrderedDict()
    chain_ordinals: Dict[str, int] = {}
    chain_lengths: Counter = Counter()
    for row in rows:
        hetero = _clean(row[columns["hetero"]])
        source = _clean(row[columns["resname"]]).upper()
        element = _clean(row[columns["element"]]).upper()
        altloc = _clean(row[columns["altloc"]])
        if altloc not in ("", "A"):
            counts["altloc_atoms_skipped"] += 1
            continue
        resname = "MET" if (mse_to_met and source == "MSE") else source
        if hetero not in ("", "H_MSE") or resname not in STANDARD_RESIDUES:
            counts["hetero_or_nonstandard_atoms_skipped"] += 1
            continue
        if hetero == "H_MSE" and not (mse_to_met and source == "MSE"):
            counts["hetero_or_nonstandard_atoms_skipped"] += 1
            continue
        chain = _clean(row[columns["chain"]])
        key = (chain, int(row[columns["residue"]]), _clean(row[columns["insertion_code"]]), source)
        candidate = groups.get(key)
        if candidate is None:
            if chain not in chain_ordinals:
                chain_ordinals[chain] = len(chain_ordinals)
            candidate = ResidueCandidate(
                chain=chain, chain_ordinal=chain_ordinals[chain],
                sequence_index=chain_lengths[chain], resseq=key[1], icode=key[2],
                source_resname=source, resname=resname, atoms=OrderedDict(),
            )
            chain_lengths[chain] += 1
            groups[key] = candidate
            if source == "MSE":
                counts["mse_converted"] += 1
        xyz = [float(row[columns[axis]]) for axis in ("x", "y", "z")]
        candidate.contact_positions.append(xyz)
        if element in HYDROGEN_ELEMENTS:
            counts["hydrogens_skipped"] += 1
            continue
        name = _clean(row[columns["name"]])
        if resname == "MET" and source == "MSE" and name == "SE":
            name, element = "SD", "S"
        if name in candidate.atoms:
            counts["duplicate_atom_names_skipped"] += 1
            continue
        candidate.atoms[name] = (element, np.array(xyz, dtype=np.float32))
    residues = [candidate for candidate in groups.values() if candidate.atoms]
    counts["residues_without_heavy_atoms"] = len(groups) - len(residues)
    return residues, counts


def select_pocket(
    residues: Sequence[ResidueCandidate], ligand: cg.LigandAtoms, *, cutoff: float, max_atoms: int,
    ligand_contact: Optional[np.ndarray] = None, contact_atoms: str = "all",
) -> Tuple[List[ResidueCandidate], dict]:
    """Whole residues within ``cutoff`` of a ligand atom, capped deterministically.

    ``contact_atoms="all"`` measures the distance with every atom as read,
    hydrogens included, which reproduces ATOM3D's own ``atoms_pocket`` residue
    set exactly (verified: 0 missing, 0 extra on 40 val complexes).
    ``contact_atoms="heavy"`` uses heavy atoms only and keeps ~10 % fewer residues.
    """
    if contact_atoms not in ("all", "heavy"):
        raise ValueError(f"contact_atoms must be 'all' or 'heavy', got {contact_atoms!r}")
    ligand_pos = (
        np.asarray(ligand.positions, dtype=np.float64) if contact_atoms == "heavy"
        else np.asarray(ligand_contact if ligand_contact is not None else ligand.positions,
                        dtype=np.float64)
    )
    kept: List[ResidueCandidate] = []
    for residue in residues:
        if contact_atoms == "heavy":
            atom_pos = np.stack([xyz for _, xyz in residue.atoms.values()]).astype(np.float64)
        else:
            atom_pos = np.asarray(residue.contact_positions, dtype=np.float64)
        distance = float(
            np.sqrt(((atom_pos[:, None, :] - ligand_pos[None, :, :]) ** 2).sum(-1)).min()
        )
        residue.min_ligand_distance = distance
        if distance <= float(cutoff):
            kept.append(residue)
    report = {
        "pocket_residues_at_cutoff": len(kept),
        "residues_dropped_by_cap": 0,
        "pocket_heavy_atoms_at_cutoff": sum(r.heavy_atoms for r in kept),
    }
    budget = int(max_atoms) - len(ligand.elements)
    if budget < 1:
        raise ValueError(
            f"ligand has {len(ligand.elements)} heavy atoms; max_atoms={max_atoms} leaves no pocket budget"
        )
    # Farthest-first, whole residues only; the ordering key is total so the cap
    # is reproducible for any input order.
    order = sorted(
        range(len(kept)),
        key=lambda i: (-kept[i].min_ligand_distance, -kept[i].chain_ordinal, -kept[i].sequence_index),
    )
    total = sum(r.heavy_atoms for r in kept)
    dropped = set()
    for index in order:
        if total <= budget:
            break
        dropped.add(index)
        total -= kept[index].heavy_atoms
    if dropped:
        report["residues_dropped_by_cap"] = len(dropped)
        kept = [r for i, r in enumerate(kept) if i not in dropped]
    kept.sort(key=lambda r: (r.chain_ordinal, r.sequence_index))
    return kept, report


# ---------------------------------------------------------------------------
# Complex -> stored sample
# ---------------------------------------------------------------------------

@dataclass
class ComplexResult:
    complex_id: str
    status: str        # ok | dropped | failed | strict_violation
    reason: str
    report: dict
    sample: object = None       # compact protein Data
    ligand: object = None       # dict with elements/positions
    target: float = float("nan")
    sequence: str = ""


def _pocket_chain_record(complex_id: str, residues: Sequence[ResidueCandidate]) -> ChainRecord:
    """Synthetic chain whose ``resseq`` is the pocket index, so policy output maps back."""
    records = [
        ResidueRecord(
            resname=residue.resname, source_resname=residue.source_resname,
            resseq=index, icode="", atoms=dict(residue.atoms),
        )
        for index, residue in enumerate(residues)
    ]
    return ChainRecord(complex_id, "pocket", records, {"structure_id": complex_id, "chain_id": "pocket"})


def process_item(item: dict, settings: dict) -> ComplexResult:
    complex_id = str(item.get("id", "?"))
    policy = parse_policy(settings["missing_atoms"])
    report: dict = {"complex": complex_id}
    try:
        target = float(item["scores"][TARGET_NAME])
        if not np.isfinite(target):
            raise ValueError(f"non-finite {TARGET_NAME}")
        ligand, ligand_contact = read_ligand(item)
        residues, counts = read_protein_residues(item, mse_to_met=settings["mse_to_met"])
        report.update({key: int(value) for key, value in counts.items()})
        report["protein_residues"] = len(residues)
        report["ligand_atoms"] = len(ligand.elements)
    except Exception as exc:
        return ComplexResult(complex_id, "failed", f"parse_error: {type(exc).__name__}: {exc}", report)
    if len(ligand.elements) < settings["min_ligand_atoms"]:
        return ComplexResult(complex_id, "dropped",
                             f"ligand_too_small: {len(ligand.elements)} heavy atoms", report)
    try:
        pocket, pocket_report = select_pocket(
            residues, ligand, cutoff=settings["pocket_cutoff"], max_atoms=settings["max_atoms"],
            ligand_contact=ligand_contact, contact_atoms=settings["contact_atoms"],
        )
    except Exception as exc:
        return ComplexResult(complex_id, "failed", f"pocket_error: {type(exc).__name__}: {exc}", report)
    report.update(pocket_report)
    if not pocket:
        return ComplexResult(complex_id, "dropped", "empty_pocket: no residue within the cutoff", report)

    try:
        outcome = apply_policy(_pocket_chain_record(complex_id, pocket), policy)
    except MissingAtomError as exc:
        return ComplexResult(complex_id, "strict_violation", f"strict_violation: {exc}", report)
    report.update({key: outcome.report[key] for key in (
        "residues_missing_backbone", "residues_incomplete_sidechain", "unexpected_atoms_removed",
        "incomplete_sidechain_fraction", "kept_residues", "masked_residues",
    )})
    if outcome.dropped_reason:
        return ComplexResult(complex_id, "dropped", f"policy_drop: {outcome.dropped_reason}", report)
    kept_residues, masked = outcome.residues, outcome.masked
    if len(kept_residues) < settings["min_residues"]:
        return ComplexResult(complex_id, "dropped",
                             f"too_few_residues: {len(kept_residues)} < min_residues", report)
    if kept_residues and sum(masked) == len(masked):
        return ComplexResult(complex_id, "dropped", "all_masked: every pocket residue is masked", report)

    sources = [pocket[residue.resseq] for residue in kept_residues]
    try:
        node = build_node_sample(kept_residues, masked, complex_id)
        node = cg.remap_protein_blocks(node, cg.protein_block_ids(
            [source.chain_ordinal for source in sources], [source.sequence_index for source in sources]
        ))
        compact = encode_edge_free_sample(node, "protein", POS_CODE_INDEX)
        compact.downstream_masked_block = node.downstream_masked_block.clone()
    except Exception as exc:
        return ComplexResult(complex_id, "failed", f"graph_error: {type(exc).__name__}: {exc}", report)

    report["pocket_chains"] = len({source.chain for source in sources})
    report["real_atoms"] = int(sum(len(r.atoms) for r in kept_residues)) + len(ligand.elements)
    report["stored_nodes"] = int(compact.z.numel()) + len(ligand.elements)
    report["min_ligand_distance"] = round(min(source.min_ligand_distance for source in sources), 3)
    sequence = "".join(THREE_TO_ONE[residue.resname] for residue in kept_residues)
    return ComplexResult(complex_id, "ok", "", report, compact, cg.stored_ligand(ligand), target, sequence)


QC_COLUMNS = (
    "complex", "split", "status", "reason", "protein_residues", "ligand_atoms",
    "pocket_residues_at_cutoff", "pocket_heavy_atoms_at_cutoff", "residues_dropped_by_cap",
    "kept_residues", "masked_residues", "pocket_chains", "real_atoms", "stored_nodes",
    "residues_missing_backbone", "residues_incomplete_sidechain", "unexpected_atoms_removed",
    "hydrogens_skipped", "altloc_atoms_skipped", "hetero_or_nonstandard_atoms_skipped",
    "residues_without_heavy_atoms",
    "duplicate_atom_names_skipped", "mse_converted", "min_ligand_distance",
)


def _percentiles(values: Sequence[float]) -> dict:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()), "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)), "p75": float(np.percentile(array, 75)),
        "max": float(array.max()), "mean": float(array.mean()),
    }


def build_processed(config: dict, *, tag: str, splits: Sequence[str]) -> int:
    """Process one split tag; return the exit code (0 ok, 1 QC failure)."""
    policy = parse_policy(config["missing_atoms"])
    out_dir = processed_dir(config, tag)
    schema = cg.complex_edge_free_schema(int(config["graph"]["sequence_neighbors_k"]), {
        "builder": "downstream.tasks.lba.dataset", "source": config["source"], "split_tag": tag,
        "policy": policy.tag, "pocket_cutoff": float(config["pocket_cutoff"]),
        "max_atoms": int(config["max_atoms"]),
    })
    settings = {
        "missing_atoms": config["missing_atoms"],
        "mse_to_met": bool(config["structure"]["mse_to_met"]),
        "pocket_cutoff": float(config["pocket_cutoff"]),
        "max_atoms": int(config["max_atoms"]),
        "min_residues": int(config["min_residues"]),
        "min_ligand_atoms": int(config["min_ligand_atoms"]),
        "contact_atoms": str(config["contact_atoms"]),
    }
    exit_code = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        target_path = out_dir / f"{split}.pt"
        if target_path.exists():
            raise FileExistsError(f"refusing to overwrite {target_path}")
        source = lmdb_dir(config, tag, split)
        if not source.is_dir():
            raise FileNotFoundError(f"{source} is not an extracted ATOM3D LMDB directory")
        expected = int(config["expected_counts"][tag][split])
        results: List[ComplexResult] = []
        for index, item in iter_lmdb(source):
            results.append(process_item(item, settings))
            if (index + 1) % 500 == 0:
                print(f"[{tag}/{split}] {index + 1} complexes", flush=True)
        if len(results) != expected:
            raise ValueError(
                f"{source}: {len(results)} entries, task.yaml expects {expected} for {tag}/{split}"
            )
        status = Counter(result.status for result in results)
        bad = len(results) - status["ok"]
        fraction = bad / max(len(results), 1)
        kept = [result for result in results if result.status == "ok"]
        summary = {
            "task": "lba", "split_tag": tag, "split": split, "complexes": len(results),
            "status": dict(status), "failed_or_dropped_fraction": round(fraction, 6),
            "limit": config["qc"]["max_failed_complex_fraction"],
            "reasons": dict(Counter(r.reason.split(":")[0] for r in results if r.status != "ok")),
            "pocket_residues": _percentiles([r.report.get("kept_residues", 0) for r in kept]),
            "ligand_atoms": _percentiles([r.report.get("ligand_atoms", 0) for r in kept]),
            "real_atoms": _percentiles([r.report.get("real_atoms", 0) for r in kept]),
            "stored_nodes": _percentiles([r.report.get("stored_nodes", 0) for r in kept]),
            "pocket_chains": dict(Counter(r.report.get("pocket_chains", 0) for r in kept)),
            "target_pk": _percentiles([r.target for r in kept]),
            "masked_residues_total": int(sum(r.report.get("masked_residues", 0) for r in kept)),
            "residues_missing_backbone_total": int(
                sum(r.report.get("residues_missing_backbone", 0) for r in kept)),
            "residues_dropped_by_cap_total": int(
                sum(r.report.get("residues_dropped_by_cap", 0) for r in kept)),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(out_dir / f"qc_{split}.tsv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=QC_COLUMNS, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for result in results:
                writer.writerow({"complex": result.complex_id, "split": split,
                                 "status": result.status, "reason": result.reason, **result.report})
        (out_dir / f"qc_{split}.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1), flush=True)
        if status["strict_violation"] or fraction > config["qc"]["max_failed_complex_fraction"]:
            print(f"[{tag}/{split}] QC FAILED: {bad}/{len(results)} complexes not usable; "
                  f"{target_path} not written", flush=True)
            exit_code = 1
            continue
        payload = {
            "schema": schema, "policy": policy.tag, "split": split, "split_tag": tag,
            "ids": [result.complex_id for result in kept],
            "targets": torch.tensor([result.target for result in kept], dtype=torch.float32),
            "samples": [result.sample for result in kept],
            "ligands": [result.ligand for result in kept],
            "sequences": [result.sequence for result in kept],
            "qc": summary,
        }
        partial = target_path.with_name(target_path.name + ".partial")
        torch.save(payload, partial)
        os.replace(partial, target_path)
        print(f"[{tag}/{split}] wrote {target_path} ({len(kept)} complexes)", flush=True)
    return exit_code


# ---------------------------------------------------------------------------
# Processed split access
# ---------------------------------------------------------------------------

class ProcessedSplit:
    """Complexes, pK targets and stored samples of one processed split.

    ``representation="joint"``   one runtime graph per complex (pocket + ligand).
    ``representation="separate"`` ``2N`` graphs: pocket graphs first, then ligand
    graphs; ``chain_embeddings`` concatenates the two pooled halves per complex.
    """

    def __init__(self, path: Path, *, representation: str = "joint"):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        for key in ("schema", "ids", "targets", "samples", "ligands", "sequences"):
            if key not in payload:
                raise ValueError(f"{self.path}: processed split lacks {key!r}")
        if representation not in ("joint", "separate"):
            raise ValueError(f"unknown representation {representation!r}")
        self.payload = payload
        self.representation = representation
        self.schema = payload["schema"]
        self.ids: List[str] = list(payload["ids"])
        self.chains = self.ids            # embed.py writes this column
        self.sequences: List[str] = list(payload["sequences"])
        self.targets: np.ndarray = payload["targets"].numpy().astype(np.float32)
        counts = {len(self.ids), len(payload["samples"]), len(payload["ligands"]),
                  len(self.sequences), int(self.targets.shape[0])}
        if len(counts) != 1:
            raise ValueError(f"{self.path}: inconsistent complex counts {counts}")
        self.labels = self.targets.reshape(-1, 1)
        self.sample_chain = torch.arange(len(self)) % len(self.ids)
        self.sample_residues = torch.ones(len(self), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.ids) * (2 if self.representation == "separate" else 1)

    def runtime(self, index: int, model_config, *, seed: int):
        count = len(self.ids)
        if self.representation == "joint":
            return cg.joint_runtime_sample(
                self.payload["samples"][index], cg.ligand_from_stored(self.payload["ligands"][index]),
                self.schema, model_config, seed=seed,
            )
        if index < count:
            return cg.pocket_runtime_sample(
                self.payload["samples"][index], self.schema, model_config, seed=seed)
        position = index - count
        return cg.ligand_runtime_sample(
            cg.ligand_from_stored(self.payload["ligands"][position]),
            f"{self.ids[position]}#ligand", model_config, seed=seed,
        )

    def iter_runtime(self, model_config, *, seed: int) -> Iterator:
        for index in range(len(self)):
            yield index, self.runtime(index, model_config, seed=seed)

    def chain_embeddings(self, sample_embeddings: np.ndarray) -> np.ndarray:
        count = len(self.ids)
        if sample_embeddings.shape[0] != len(self):
            raise ValueError(
                f"{self.path}: {sample_embeddings.shape[0]} sample embeddings for {len(self)} samples")
        if self.representation == "joint":
            return sample_embeddings.astype(np.float32)
        return np.concatenate(
            (sample_embeddings[:count], sample_embeddings[count:]), axis=1
        ).astype(np.float32)

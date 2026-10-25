"""Experimental PDB/mmCIF files -> per-chain residue records with an explicit report.

Only the first MODEL is read (NMR ensembles), hydrogens are skipped, and each
disordered atom keeps its highest-occupancy alternate location (ties: first
altloc label in sorted order). Residues outside the 20 standard amino acids are
never passed on: amino-acid-like HETATM residues are reported as nonstandard,
everything else as ligand or water. ``mse_to_met=True`` converts
selenomethionine to MET (SE -> SD, element S) and reports it.

Missing atoms are not judged here; ``missing_atoms`` applies the task policy.
"""

from __future__ import annotations

import gzip
import io
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

from data_loading.atom_position_vocab import PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE

STANDARD_RESIDUES = frozenset(PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE)
HYDROGEN_ELEMENTS = frozenset(("H", "D"))
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
assert set(THREE_TO_ONE) == STANDARD_RESIDUES


@dataclass
class ResidueRecord:
    """One standard residue; ``atoms`` maps atom name -> (element, xyz)."""

    resname: str
    source_resname: str
    resseq: int
    icode: str
    atoms: Dict[str, Tuple[str, np.ndarray]]


@dataclass
class ChainRecord:
    structure_id: str
    chain_id: str
    residues: List[ResidueRecord]
    report: Dict[str, object] = field(default_factory=dict)

    @property
    def sequence(self) -> str:
        return "".join(THREE_TO_ONE[residue.resname] for residue in self.residues)


def _open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _structure_format(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes if suffix.lower() != ".gz"]
    if suffixes and suffixes[-1] in (".pdb", ".ent"):
        return "pdb"
    if suffixes and suffixes[-1] in (".cif", ".mmcif"):
        return "cif"
    raise ValueError(f"unsupported structure file type: {path}")


def load_first_model(path: Path):
    """Return ``(first_model, num_models)`` for a .pdb/.ent/.cif file (optionally .gz)."""
    path = Path(path)
    parser = PDBParser(QUIET=True) if _structure_format(path) == "pdb" else MMCIFParser(QUIET=True)
    with _open_text(path) as handle:
        structure = parser.get_structure(path.name.split(".")[0], handle)
    models = list(structure)
    if not models:
        raise ValueError(f"{path}: structure contains no models")
    return models[0], len(models)


def parse_structure_text(text: str, fmt: str, structure_id: str):
    """Parse an in-memory PDB/mmCIF string (used by tests)."""
    parser = PDBParser(QUIET=True) if fmt == "pdb" else MMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, io.StringIO(text))
    models = list(structure)
    if not models:
        raise ValueError(f"{structure_id}: structure contains no models")
    return models[0], len(models)


def _select_altloc(atom):
    """Highest occupancy child of a disordered atom; ties keep the first altloc label."""
    if not atom.is_disordered():
        return atom, False
    children = sorted(atom.disordered_get_list(), key=lambda child: child.get_altloc())
    best = max(children, key=lambda child: (child.get_occupancy() or 0.0))
    return best, True


def _select_residue(residue):
    """Resolve point-mutation disorder (two residue names at one position) by occupancy."""
    if residue.is_disordered() != 2:
        return residue, False
    children = residue.disordered_get_list()

    def occupancy(child):
        return sum((_select_altloc(atom)[0].get_occupancy() or 0.0) for atom in child)

    return max(children, key=occupancy), True


def chain_records_from_model(
    model, structure_id: str, num_models: int, *, mse_to_met: bool
) -> Dict[str, ChainRecord]:
    """Convert every chain of one Biopython model into a ``ChainRecord``."""
    chains: Dict[str, ChainRecord] = {}
    for chain in model:
        counts = Counter()
        nonstandard = Counter()
        residues: List[ResidueRecord] = []
        for raw_residue in chain:
            residue, resolved = _select_residue(raw_residue)
            counts["disordered_residues_resolved"] += int(resolved)
            hetflag, resseq, icode = residue.get_id()
            source = residue.get_resname().strip().upper()
            if hetflag == "W" or source in ("HOH", "WAT", "DOD"):
                counts["waters_skipped"] += 1
                continue
            resname = "MET" if (mse_to_met and source == "MSE") else source
            atoms: Dict[str, Tuple[str, np.ndarray]] = {}
            for raw_atom in residue:
                atom, had_altloc = _select_altloc(raw_atom)
                counts["altloc_atoms_resolved"] += int(had_altloc)
                element = (atom.element or "").strip().upper()
                if element in HYDROGEN_ELEMENTS:
                    counts["hydrogens_skipped"] += 1
                    continue
                name = atom.get_name().strip()
                if resname == "MET" and source == "MSE" and name == "SE":
                    name, element = "SD", "S"
                if name in atoms:
                    raise ValueError(
                        f"{structure_id} chain {chain.id} residue {source}{resseq}{icode.strip()}: "
                        f"duplicate atom name {name} after altloc selection"
                    )
                atoms[name] = (element, np.asarray(atom.get_coord(), dtype=np.float32))
            if resname not in STANDARD_RESIDUES:
                if {"N", "CA", "C"} <= set(atoms):
                    nonstandard[source] += 1
                else:
                    counts["ligand_residues_skipped"] += 1
                continue
            if source == "MSE":
                counts["mse_converted"] += 1
            residues.append(ResidueRecord(resname, source, int(resseq), icode.strip(), atoms))
        report = {
            "structure_id": structure_id,
            "chain_id": chain.id,
            "num_models_in_file": num_models,
            "standard_residues": len(residues),
            "nonstandard_residues": int(sum(nonstandard.values())),
            "nonstandard_residue_names": dict(sorted(nonstandard.items())),
            "oxt_atoms": sum("OXT" in residue.atoms for residue in residues),
            **{key: int(counts[key]) for key in (
                "disordered_residues_resolved", "altloc_atoms_resolved", "hydrogens_skipped",
                "mse_converted", "waters_skipped", "ligand_residues_skipped",
            )},
        }
        chains[chain.id] = ChainRecord(structure_id, chain.id, residues, report)
    return chains


def read_chains(path: Path, *, mse_to_met: bool) -> Dict[str, ChainRecord]:
    """Parse a structure file's first model into per-chain records."""
    model, num_models = load_first_model(Path(path))
    return chain_records_from_model(model, Path(path).name.split(".")[0], num_models, mse_to_met=mse_to_met)

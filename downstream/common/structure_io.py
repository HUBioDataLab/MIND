"""Experimental PDB/mmCIF files -> per-chain residue records with an explicit report.

Only the first MODEL is read (NMR ensembles), hydrogens are skipped, and each
disordered atom keeps its highest-occupancy alternate location (ties: first
altloc label in sorted order). Residues outside the 20 standard amino acids are
never passed on: amino-acid-like HETATM residues are reported as nonstandard,
everything else as ligand or water. ``mse_to_met=True`` converts
selenomethionine to MET (SE -> SD, element S) and reports it.

Missing atoms are not judged here; ``missing_atoms`` applies the task policy.

``.hdf5`` files in the IEConv / GearNet-Fold3D layout (one domain or chain per file, written by Hermosilla et al.)
are read by ``read_hdf5_chain`` into the same ``ChainRecord`` so that the missing-atom policy, the graph builder and
the QC counters are shared with the PDB/mmCIF path. Atom elements come from the atom NAME (first letter), never from
``atom_types``: the archives' ``atom_types`` indexing (periodic-table index vs atomic number) is not verified.
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


def unshift_wide_chain_pdb(text: str) -> Tuple[str, str]:
    """Undo the column shift of PDB files that write a MULTI-character chain id (3J9M-AK, 4V7G-AA, 4UE3-SSS, ...).

    Those files (the DeepFRI / TorchProtein GeneOntology and EnzymeCommission per-chain files, mostly cryo-EM and
    ribosome complexes) put the chain id in columns 22-23 (or 22-24), so residue number, coordinates and element sit
    one (two) columns to the right: 81 (82) character ATOM/HETATM lines. Biopython reads the extra chain characters as
    part of the residue number and either raises (``invalid literal for int()``) or finds no coordinates. A line is
    "wide" by k columns (k = 1..3) when the decimal point of x is in column 35 + k and that of y in column 43 + k; the
    k extra chain columns are then removed so that the fixed-column parser sees column 22 as the chain. Returns
    ``(text, wide_chain_id)`` with ``wide_chain_id == ""`` when no line was shifted.

    One file may hold only one such chain id: dropping the extra characters of several ids would merge chains. A file
    that mixes wide and standard ATOM/HETATM lines is ambiguous and raises. Lines that match no layout are left
    untouched, so the parser still reports them.
    """
    lines = text.split("\n")
    wide_ids = set()
    standard = 0
    for number, line in enumerate(lines):
        if line[:6] not in ("ATOM  ", "HETATM") or len(line) < 46:
            continue
        if line[34] == "." and line[42] == ".":
            standard += 1
            continue
        extra = next((k for k in (1, 2, 3) if line[34 + k] == "." and line[42 + k] == "."), 0)
        if not extra:
            continue
        chain = line[21:22 + extra]
        if " " in chain:
            raise ValueError(f"PDB line {number + 1}: columns shifted by {extra} but chain field {chain!r} "
                             f"is not {extra + 1} characters")
        wide_ids.add(chain)
        lines[number] = line[:22] + line[22 + extra:]
    if not wide_ids:
        return text, ""
    if standard:
        raise ValueError(f"PDB mixes multi-character and standard chain-id columns ({standard} standard atom lines, "
                         f"wide chain ids {sorted(wide_ids)}); cannot unshift unambiguously")
    if len(wide_ids) > 1:
        raise ValueError(f"PDB holds more than one multi-character chain id {sorted(wide_ids)}; "
                         f"dropping the extra characters would merge chains")
    return "\n".join(lines), next(iter(wide_ids))


def parse_structure_text(text: str, fmt: str, structure_id: str):
    """Parse an in-memory PDB/mmCIF string; returns ``(first_model, num_models)``.

    PDB text with multi-character chain ids is unshifted first (``unshift_wide_chain_pdb``) and the chain keeps its
    real multi-character id.
    """
    wide_chain = ""
    if fmt == "pdb":
        text, wide_chain = unshift_wide_chain_pdb(text)
    parser = PDBParser(QUIET=True) if fmt == "pdb" else MMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, io.StringIO(text))
    models = list(structure)
    if not models:
        raise ValueError(f"{structure_id}: structure contains no models")
    if wide_chain:
        chains = list(models[0])
        if len(chains) != 1:
            raise ValueError(f"{structure_id}: expected one chain after unshifting {wide_chain!r}, found {[c.id for c in chains]}")
        chains[0].id = wide_chain
    return models[0], len(models)


def load_first_model(path: Path):
    """Return ``(first_model, num_models)`` for a .pdb/.ent/.cif file (optionally .gz)."""
    path = Path(path)
    fmt = _structure_format(path)
    with _open_text(path) as handle:
        text = handle.read()
    return parse_structure_text(text, fmt, path.name.split(".")[0])


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


HDF5_REQUIRED = ("atom_pos", "atom_names", "atom_residue_names", "atom_residue_id")


def _text_array(values, what: str, path) -> List[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str):
            raise ValueError(f"{path}: {what} holds a {type(value).__name__}, expected bytes or str")
        out.append(value.strip())
    return out


def _element_of_atom_name(name: str) -> str:
    letters = name.lstrip("0123456789")
    if not letters or not letters[0].isalpha():
        raise ValueError(f"atom name {name!r} has no element letter")
    return letters[0].upper()


def read_hdf5_chain(path: Path, *, mse_to_met: bool) -> Dict[str, ChainRecord]:
    """One ``.hdf5`` structure (IEConv / Fold3D layout) -> ``{stem: ChainRecord}`` holding ALL its residues.

    Required datasets: ``atom_pos`` (frames x atoms x 3, only frame 0 is used, or atoms x 3), ``atom_names``,
    ``atom_residue_names``, ``atom_residue_id``; ``atom_chain_names`` is used when present to separate residues.
    Residues are runs of atoms with the same (chain, residue id, residue name). Missing datasets, inconsistent
    lengths, non-finite coordinates and duplicate atom names inside a residue raise.
    """
    try:
        import h5py
    except ImportError as exc:  # fail loudly: no PDB fallback exists for these archives
        raise ImportError("reading .hdf5 structures needs h5py (not installed in this environment)") from exc
    path = Path(path)
    with h5py.File(path, "r") as handle:
        missing = [key for key in HDF5_REQUIRED if key not in handle]
        if missing:
            raise ValueError(f"{path}: hdf5 lacks datasets {missing}; has {sorted(handle.keys())}")
        positions = np.asarray(handle["atom_pos"][()], dtype=np.float32)
        names = _text_array(handle["atom_names"][()], "atom_names", path)
        resnames = _text_array(handle["atom_residue_names"][()], "atom_residue_names", path)
        residue_ids = np.asarray(handle["atom_residue_id"][()]).reshape(-1)
        chains = (_text_array(handle["atom_chain_names"][()], "atom_chain_names", path)
                  if "atom_chain_names" in handle else [""] * len(names))
    num_frames = 1
    if positions.ndim == 3:
        num_frames, positions = positions.shape[0], positions[0]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{path}: atom_pos has shape {positions.shape}, expected (frames, atoms, 3) or (atoms, 3)")
    count = len(names)
    if not (len(positions) == len(resnames) == len(residue_ids) == len(chains) == count) or count == 0:
        raise ValueError(f"{path}: inconsistent atom counts (pos {len(positions)}, names {count}, "
                         f"residue names {len(resnames)}, residue ids {len(residue_ids)}, chains {len(chains)})")
    if not np.isfinite(positions).all():
        raise ValueError(f"{path}: atom_pos contains NaN or inf")
    counts = Counter()
    nonstandard = Counter()
    residues: List[ResidueRecord] = []
    start = 0
    while start < count:
        key = (chains[start], int(residue_ids[start]), resnames[start])
        stop = start
        while stop < count and (chains[stop], int(residue_ids[stop]), resnames[stop]) == key:
            stop += 1
        source = key[2].upper()
        if source in ("HOH", "WAT", "DOD"):
            counts["waters_skipped"] += 1
            start = stop
            continue
        resname = "MET" if (mse_to_met and source == "MSE") else source
        atoms: Dict[str, Tuple[str, np.ndarray]] = {}
        for row in range(start, stop):
            name, element = names[row], _element_of_atom_name(names[row])
            if element in HYDROGEN_ELEMENTS:
                counts["hydrogens_skipped"] += 1
                continue
            if resname == "MET" and source == "MSE" and name == "SE":
                name, element = "SD", "S"
            if name in atoms:
                raise ValueError(f"{path} residue {source}{key[1]}: duplicate atom name {name}")
            atoms[name] = (element, positions[row].copy())
        start = stop
        if resname not in STANDARD_RESIDUES:
            if {"N", "CA", "C"} <= set(atoms):
                nonstandard[source] += 1
            else:
                counts["ligand_residues_skipped"] += 1
            continue
        if source == "MSE":
            counts["mse_converted"] += 1
        residues.append(ResidueRecord(resname, source, key[1], "", atoms))
    report = {
        "structure_id": path.name.split(".hdf5")[0], "chain_id": "", "num_models_in_file": num_frames,
        "standard_residues": len(residues), "nonstandard_residues": int(sum(nonstandard.values())),
        "nonstandard_residue_names": dict(sorted(nonstandard.items())),
        "oxt_atoms": sum("OXT" in residue.atoms for residue in residues),
        "disordered_residues_resolved": 0, "altloc_atoms_resolved": 0,
        **{key: int(counts[key]) for key in ("hydrogens_skipped", "mse_converted", "waters_skipped",
                                              "ligand_residues_skipped")},
    }
    stem = path.name.split(".hdf5")[0]
    return {stem: ChainRecord(stem, "", residues, report)}


def read_chains(path: Path, *, mse_to_met: bool) -> Dict[str, ChainRecord]:
    """Parse a structure file's first model into per-chain records."""
    if Path(path).suffix.lower() == ".hdf5":
        return read_hdf5_chain(Path(path), mse_to_met=mse_to_met)
    model, num_models = load_first_model(Path(path))
    return chain_records_from_model(model, Path(path).name.split(".")[0], num_models, mse_to_met=mse_to_met)

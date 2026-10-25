"""Synthetic PDB/mmCIF builders for the downstream tests (no network, no real data)."""

from __future__ import annotations

from data_loading.atom_position_vocab import PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE

ELEMENT = {"C": "C", "N": "N", "O": "O", "S": "S"}


def residue_atoms(resname: str, index: int, *, drop=(), extra=()):
    """Heavy atoms of one residue with a non-degenerate local geometry (CA spacing 3.8 A)."""
    x = 3.8 * index
    zig = 0.6 if index % 2 else -0.6
    atoms = [
        ("N", "N", (x - 1.20, 0.80 + zig, 0.10)),
        ("CA", "C", (x, zig, 0.0)),
        ("C", "C", (x + 1.25, 0.75 + zig, 0.20)),
        ("O", "O", (x + 1.30, 1.95 + zig, 0.40)),
    ]
    for k, name in enumerate(PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE[resname]):
        element = "SE" if name == "SE" else name[0]
        atoms.append((name, element, (x + 0.3 * (k % 2), zig - 1.5 - 1.3 * k, 0.5 + 0.4 * (k % 3))))
    atoms = [a for a in atoms if a[0] not in drop]
    return atoms + list(extra)


def pdb_line(serial, name, resname, chain, resseq, xyz, element, *, altloc=" ", occ=1.0, record="ATOM"):
    field = f" {name:<3}" if len(name) < 4 else name
    return (f"{record:<6}{serial:>5} {field}{altloc}{resname:>3} {chain}{resseq:>4}    "
            f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}{occ:>6.2f}{20.0:>6.2f}          {element:>2}")


def pdb_text(chains, *, models=1, model_shift=0.0):
    """``chains``: {chain_id: [(resname, atoms, record)]}; returns PDB text with ``models`` MODELs."""
    lines, serial = [], 1
    for model in range(models):
        if models > 1:
            lines.append(f"MODEL     {model + 1:>4}")
        for chain_id, residues in chains.items():
            for resseq, (resname, atoms, record) in enumerate(residues, start=1):
                for atom in atoms:
                    name, element, xyz = atom[:3]
                    altloc, occ = (atom[3], atom[4]) if len(atom) > 3 else (" ", 1.0)
                    shifted = (xyz[0] + model * model_shift, xyz[1], xyz[2])
                    lines.append(pdb_line(serial, name, resname, chain_id, resseq, shifted, element,
                                          altloc=altloc, occ=occ, record=record))
                    serial += 1
            lines.append("TER")
        if models > 1:
            lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines) + "\n"


def cif_text(entry, chains):
    """Minimal mmCIF ``_atom_site`` loop; chain ids may be multi-character (auth_asym_id)."""
    header = [
        f"data_{entry}", "loop_", "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
        "_atom_site.label_atom_id", "_atom_site.label_alt_id", "_atom_site.label_comp_id",
        "_atom_site.label_asym_id", "_atom_site.label_entity_id", "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
        "_atom_site.occupancy", "_atom_site.B_iso_or_equiv", "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id", "_atom_site.auth_asym_id", "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    rows, serial = [], 1
    for entity, (chain_id, residues) in enumerate(chains.items(), start=1):
        for resseq, (resname, atoms, record) in enumerate(residues, start=1):
            for name, element, xyz in atoms:
                rows.append(f"{record} {serial} {element} {name} . {resname} {chain_id} {entity} {resseq} ? "
                            f"{xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f} 1.00 20.00 {resseq} {resname} {chain_id} {name} 1")
                serial += 1
    return "\n".join(header + rows) + "\n#\n"


def chain_of(resnames, *, start=0, drops=None):
    drops = drops or {}
    return [(r, residue_atoms(r, start + i, drop=drops.get(i, ())), "ATOM") for i, r in enumerate(resnames)]

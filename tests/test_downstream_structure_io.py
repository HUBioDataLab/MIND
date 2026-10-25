"""structure_io on synthetic PDB/mmCIF text: models, chains, altlocs, MSE, OXT, nonstandard residues."""

import gzip

import numpy as np
import pytest

from downstream.common.structure_io import chain_records_from_model, parse_structure_text, read_chains
from downstream_helpers import chain_of, cif_text, pdb_text, residue_atoms


def _chains(text, fmt="pdb", mse_to_met=True):
    model, num_models = parse_structure_text(text, fmt, "syn")
    return chain_records_from_model(model, "syn", num_models, mse_to_met=mse_to_met)


def test_nmr_reads_only_the_first_model():
    text = pdb_text({"A": chain_of(["ALA", "GLY"])}, models=2, model_shift=50.0)
    chains = _chains(text)
    record = chains["A"]
    assert record.report["num_models_in_file"] == 2
    assert len(record.residues) == 2
    np.testing.assert_allclose(record.residues[0].atoms["CA"][1], residue_atoms("ALA", 0)[1][2], atol=1e-3)


def test_chains_are_split_and_keep_their_own_residues():
    chains = _chains(pdb_text({"A": chain_of(["ALA", "SER"]), "B": chain_of(["GLY"], start=5)}))
    assert set(chains) == {"A", "B"}
    assert chains["A"].sequence == "AS" and chains["B"].sequence == "G"


def test_altloc_keeps_highest_occupancy_and_ties_keep_first_label():
    atoms = residue_atoms("SER", 0)
    og = next(a for a in atoms if a[0] == "OG")
    ca = next(a for a in atoms if a[0] == "CA")
    rest = [a for a in atoms if a[0] not in ("OG", "CA")]
    alt = rest + [
        ("OG", "O", (9.0, 9.0, 9.0), "A", 0.3), ("OG", "O", og[2], "B", 0.7),
        ("CA", "C", ca[2], "A", 0.5), ("CA", "C", (7.0, 7.0, 7.0), "B", 0.5),
    ]
    record = _chains(pdb_text({"A": [("SER", alt, "ATOM")]}))["A"]
    np.testing.assert_allclose(record.residues[0].atoms["OG"][1], og[2], atol=1e-3)
    np.testing.assert_allclose(record.residues[0].atoms["CA"][1], ca[2], atol=1e-3)
    assert record.report["altloc_atoms_resolved"] == 2


def test_missing_atoms_are_preserved_as_absent_not_filled():
    residues = chain_of(["LYS", "ALA"], drops={0: ("CD",), 1: ("O",)})
    record = _chains(pdb_text({"A": residues}))["A"]
    assert "CD" not in record.residues[0].atoms and "CE" in record.residues[0].atoms
    assert "O" not in record.residues[1].atoms


def test_mse_conversion_is_explicit_and_reported():
    mse = [a if a[0] != "SD" else ("SE", "SE", a[2]) for a in residue_atoms("MET", 0)]
    text = pdb_text({"A": [("MSE", mse, "HETATM")] + chain_of(["ALA"], start=1)})
    converted = _chains(text, mse_to_met=True)["A"]
    assert converted.sequence == "MA" and converted.report["mse_converted"] == 1
    assert converted.residues[0].atoms["SD"][0] == "S"
    kept_out = _chains(text, mse_to_met=False)["A"]
    assert kept_out.sequence == "A"
    assert kept_out.report["nonstandard_residue_names"] == {"MSE": 1}


def test_oxt_is_reported_and_kept_for_the_adapter_to_drop():
    atoms = residue_atoms("GLY", 0, extra=[("OXT", "O", (2.0, 3.0, 0.5))])
    record = _chains(pdb_text({"A": [("GLY", atoms, "ATOM")]}))["A"]
    assert record.report["oxt_atoms"] == 1 and "OXT" in record.residues[0].atoms


def test_ligands_waters_and_hydrogens_are_skipped_and_counted():
    ligand = [("ZN", "ZN", (20.0, 0.0, 0.0))]
    water = [("O", "O", (25.0, 0.0, 0.0))]
    ala = residue_atoms("ALA", 0, extra=[("H", "H", (0.5, 0.5, 0.5))])
    text = pdb_text({"A": [("ALA", ala, "ATOM"), ("ZN", ligand, "HETATM"), ("HOH", water, "HETATM")]})
    report = _chains(text)["A"].report
    assert report["ligand_residues_skipped"] == 1
    assert report["waters_skipped"] == 1
    assert report["hydrogens_skipped"] == 1


def test_mmcif_gz_with_multi_character_chain(tmp_path):
    path = tmp_path / "1ABC.cif.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(cif_text("1ABC", {"aL": chain_of(["ALA", "TRP"]), "B": chain_of(["GLY"], start=4)}))
    chains = read_chains(path, mse_to_met=True)
    assert set(chains) == {"aL", "B"}
    assert chains["aL"].sequence == "AW"


def test_unsupported_suffix_fails_loudly(tmp_path):
    path = tmp_path / "x.xyz"
    path.write_text("")
    with pytest.raises(ValueError, match="unsupported structure file type"):
        read_chains(path, mse_to_met=True)

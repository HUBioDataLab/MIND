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


# ---------------------------------------------------------------------------
# Two-character PDB chain ids (DeepFRI / TorchProtein GeneOntology per-chain files, e.g. 3J9M-AK)
# ---------------------------------------------------------------------------

# Verbatim first atoms of GeneOntology/test/3J9M-AK_15808.pdb: the writer put a 2-character chain id in
# columns 22-23, so every field after the chain is shifted one column to the right (81-character lines).
REAL_WIDE_CHAIN_LINES = (
    "ATOM      1  N   HIS AK  28     177.137 124.931 209.744  1.00174.51           N  \n"
    "ATOM      2  CA  HIS AK  28     177.867 124.174 208.674  1.00173.48           C  \n"
    "ATOM      3  C   HIS AK  28     177.539 124.615 207.234  1.00168.32           C  \n"
    "ATOM      4  O   HIS AK  28     178.008 123.995 206.277  1.00171.27           O  \n"
    "TER       5      HIS AK  28\n"
    "END\n"
)


def test_two_character_pdb_chain_id_is_read_with_correct_residue_number_and_coordinates(tmp_path):
    path = tmp_path / "3J9M-AK_15808.pdb"
    path.write_text(REAL_WIDE_CHAIN_LINES)
    chains = read_chains(path, mse_to_met=True)
    assert set(chains) == {"AK"}                      # the real two-character id survives
    record = chains["AK"]
    assert record.sequence == "H" and record.residues[0].resseq == 28
    np.testing.assert_allclose(record.residues[0].atoms["CA"][1], (177.867, 124.174, 208.674), atol=1e-3)
    assert set(record.residues[0].atoms) == {"N", "CA", "C", "O"}


@pytest.mark.parametrize("chain_id", ["AK", "S2", "19", "aB", "SSS", "AAA", "A12"])
def test_two_character_pdb_chain_matches_the_single_character_reading(tmp_path, chain_id):
    residues = chain_of(["ALA", "SER", "TRP", "GLY"])
    wide = tmp_path / f"1XYZ-{chain_id}_0.pdb"
    wide.write_text(pdb_text({chain_id: residues}))
    narrow = tmp_path / "1XYZ-A_0.pdb"
    narrow.write_text(pdb_text({"A": residues}))
    got, want = read_chains(wide, mse_to_met=True), read_chains(narrow, mse_to_met=True)
    assert set(got) == {chain_id} and set(want) == {"A"}
    assert got[chain_id].sequence == want["A"].sequence == "ASWG"
    for left, right in zip(got[chain_id].residues, want["A"].residues):
        assert left.resseq == right.resseq and set(left.atoms) == set(right.atoms)
        for name in left.atoms:
            np.testing.assert_allclose(left.atoms[name][1], right.atoms[name][1], atol=1e-3)
    assert got[chain_id].report["chain_id"] == chain_id


def test_three_character_chain_id_real_line_layout(tmp_path):
    # verbatim layout of EnzymeCommission 4UE3-SSS (82-character lines, x starts one column later still)
    path = tmp_path / "4UE3-SSS_1.pdb"
    path.write_text("ATOM      1  N   LYS SSS   4       3.625 -29.105  -7.449  1.00 52.54           N  \n"
                    "ATOM      2  CA  LYS SSS   4       4.100 -28.000  -6.500  1.00 52.54           C  \n"
                    "ATOM      3  C   LYS SSS   4       5.100 -27.500  -7.500  1.00 52.54           C  \n"
                    "END\n")
    record = read_chains(path, mse_to_met=True)["SSS"]
    assert record.sequence == "K" and record.residues[0].resseq == 4
    np.testing.assert_allclose(record.residues[0].atoms["N"][1], (3.625, -29.105, -7.449), atol=1e-3)


def test_two_character_chain_hetatm_lines_are_unshifted_too(tmp_path):
    ligand = [("ZN", "ZN", (1.0, 2.0, 3.0))]
    path = tmp_path / "1XYZ-AK_0.pdb"
    path.write_text(pdb_text({"AK": chain_of(["ALA"]) + [("ZN", ligand, "HETATM")]}))
    record = read_chains(path, mse_to_met=True)["AK"]
    assert record.sequence == "A" and record.report["ligand_residues_skipped"] == 1


def test_two_character_chain_gz_is_read_like_plain_text(tmp_path):
    path = tmp_path / "3J9M-AK_15808.pdb.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(REAL_WIDE_CHAIN_LINES)
    assert read_chains(path, mse_to_met=True)["AK"].sequence == "H"


def test_wide_chain_ids_that_cannot_be_unshifted_unambiguously_fail_loudly(tmp_path):
    two_wide = tmp_path / "two.pdb"
    two_wide.write_text(pdb_text({"AK": chain_of(["ALA"]), "AL": chain_of(["GLY"])}))
    with pytest.raises(ValueError, match="more than one multi-character chain id"):
        read_chains(two_wide, mse_to_met=True)
    mixed = tmp_path / "mixed.pdb"
    mixed.write_text(pdb_text({"AK": chain_of(["ALA"]), "B": chain_of(["GLY"])}))
    with pytest.raises(ValueError, match="mixes multi-character and standard"):
        read_chains(mixed, mse_to_met=True)

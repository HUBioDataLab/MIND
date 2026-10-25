"""Missing-atom detection and the explicit policies (no silent default)."""

import pytest

from downstream.common.missing_atoms import (
    MissingAtomError, apply_policy, parse_policy, residue_completeness,
)
from downstream.common.structure_io import chain_records_from_model, parse_structure_text
from downstream.tasks.ec.dataset import load_task_config
from downstream_helpers import chain_of, pdb_text, residue_atoms

MASK = {"policy": "drop_residue_missing_backbone", "incomplete_sidechain": "mask"}


def _chain(residues):
    model, n = parse_structure_text(pdb_text({"A": residues}), "pdb", "syn")
    return chain_records_from_model(model, "syn", n, mse_to_met=True)["A"]


def _mixed_chain():
    # 0 ALA complete, 1 LYS missing NZ, 2 SER missing CA, 3 TRP complete
    return _chain(chain_of(["ALA", "LYS", "SER", "TRP"], drops={1: ("NZ",), 2: ("CA",)}))


def test_completeness_uses_canonical_heavy_atom_lists():
    chain = _mixed_chain()
    status = [residue_completeness(r) for r in chain.residues]
    assert status[0].complete and status[3].complete
    assert status[1].missing_sidechain == ("NZ",) and not status[1].missing_backbone
    assert status[2].missing_backbone == ("CA",)


def test_unexpected_atom_names_are_removed_and_counted():
    atoms = residue_atoms("ALA", 0, extra=[("XX1", "C", (9.0, 9.0, 9.0))])
    result = apply_policy(_chain([("ALA", atoms, "ATOM")]), parse_policy(MASK))
    assert result.report["unexpected_atoms_removed"] == 1
    assert "XX1" not in result.residues[0].atoms


def test_strict_policy_raises_listing_residues():
    with pytest.raises(MissingAtomError, match=r"LYS2.*NZ|SER3.*CA"):
        apply_policy(_mixed_chain(), parse_policy({"policy": "strict"}))
    complete = apply_policy(_chain(chain_of(["ALA", "GLY"])), parse_policy({"policy": "strict"}))
    assert complete.masked == [False, False]


def test_drop_backbone_and_mask_incomplete_sidechain():
    result = apply_policy(_mixed_chain(), parse_policy(MASK))
    assert [r.resname for r in result.residues] == ["ALA", "LYS", "TRP"]
    assert result.masked == [False, True, False]
    assert result.report["residues_missing_backbone"] == 1
    assert result.report["residues_incomplete_sidechain"] == 1
    assert result.report["masked_residues"] == 1
    assert result.dropped_reason is None


def test_drop_chain_if_incomplete_fraction_above_threshold():
    policy = parse_policy({**MASK, "incomplete_sidechain": "drop_chain_if_fraction_above",
                           "max_incomplete_sidechain_fraction": 0.2})
    dropped = apply_policy(_mixed_chain(), policy)  # 1 of 3 kept residues incomplete
    assert dropped.dropped_reason and dropped.residues == []
    lenient = parse_policy({**MASK, "incomplete_sidechain": "drop_chain_if_fraction_above",
                            "max_incomplete_sidechain_fraction": 0.5})
    kept = apply_policy(_mixed_chain(), lenient)
    assert kept.dropped_reason is None and kept.masked == [False, True, False]


@pytest.mark.parametrize("config, message", [
    ({}, "policy is required"),
    ({"policy": "UNDECIDED"}, "must be one of"),
    ({"policy": "drop_residue_missing_backbone"}, "incomplete_sidechain"),
    ({**MASK, "max_incomplete_sidechain_fraction": 0.1}, "only used with"),
    ({**MASK, "incomplete_sidechain": "drop_chain_if_fraction_above"}, r"in \[0, 1\)"),
    ({"policy": "strict", "incomplete_sidechain": "mask"}, "not used by policy"),
])
def test_policy_configuration_fails_loudly(config, message):
    with pytest.raises(ValueError, match=message):
        parse_policy(config)


def test_checked_in_task_yaml_names_an_explicit_policy():
    config = load_task_config()
    assert parse_policy(config["missing_atoms"]) == parse_policy(
        {"policy": "drop_residue_missing_backbone", "incomplete_sidechain": "mask"}
    )


def test_undecided_policy_is_refused():
    with pytest.raises(ValueError, match="must be one of"):
        parse_policy({"policy": "UNDECIDED"})

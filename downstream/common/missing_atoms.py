"""Missing-atom detection and the explicit per-task policy for experimental chains.

The training adapter (``ProteinAdapter._standardize_residue_to_14_atoms``) pads
any absent sidechain atom with V and packs the remaining atoms into earlier
slots, and keeps a residue with a missing backbone atom as a 13-atom block.
Experimental structures therefore must be checked here, against the canonical
heavy-atom lists in ``data_loading.atom_position_vocab``, before conversion.

Policies (``missing_atoms`` in task.yaml; every key is required, no defaults):

``policy: strict``
    Any missing backbone or sidechain atom raises ``MissingAtomError`` that
    lists the chain and residues.
``policy: drop_residue_missing_backbone``
    Residues missing any of N, CA, C, O are removed and counted. Residues with
    an incomplete sidechain are handled by ``incomplete_sidechain``:

    ``mask``
        keep the residue; its whole block is flagged ``masked`` so the encoder
        sees it exactly like a pretraining MLM block (all 14 slots selected,
        atom ids replaced by the mask token, sidechain slots removed from ESA
        and read out as latent slots).
    ``drop_chain_if_fraction_above``
        drop the chain when the incomplete-sidechain fraction of its kept
        residues exceeds ``max_incomplete_sidechain_fraction``; residues of a
        kept chain are masked as above (passing them through would place
        atoms into the wrong slots).

Atom names that are neither canonical nor ``OXT`` are always removed and
counted (``unexpected_atoms``); OXT is dropped later by the adapter itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS,
    PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE,
    PROTEIN_TERMINAL_ATOMS,
)
from downstream.common.structure_io import ChainRecord, ResidueRecord

POLICIES = ("strict", "drop_residue_missing_backbone")
SIDECHAIN_OPTIONS = ("mask", "drop_chain_if_fraction_above")


class MissingAtomError(ValueError):
    """Raised by the strict policy."""


@dataclass(frozen=True)
class ResidueCompleteness:
    missing_backbone: Tuple[str, ...]
    missing_sidechain: Tuple[str, ...]
    unexpected: Tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_backbone and not self.missing_sidechain


def residue_completeness(residue: ResidueRecord) -> ResidueCompleteness:
    expected_side = PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE[residue.resname]
    known = set(PROTEIN_BACKBONE_ATOMS) | set(expected_side) | set(PROTEIN_TERMINAL_ATOMS)
    return ResidueCompleteness(
        missing_backbone=tuple(a for a in PROTEIN_BACKBONE_ATOMS if a not in residue.atoms),
        missing_sidechain=tuple(a for a in expected_side if a not in residue.atoms),
        unexpected=tuple(sorted(name for name in residue.atoms if name not in known)),
    )


@dataclass(frozen=True)
class MissingAtomPolicy:
    policy: str
    incomplete_sidechain: Optional[str]
    max_incomplete_sidechain_fraction: Optional[float]

    @property
    def tag(self) -> str:
        if self.policy == "strict":
            return "strict"
        if self.incomplete_sidechain == "mask":
            return "dropbb_masksc"
        return f"dropbb_maxsc{self.max_incomplete_sidechain_fraction:g}"


def parse_policy(config: Mapping[str, object]) -> MissingAtomPolicy:
    """Validate the ``missing_atoms`` mapping; unknown or absent keys fail loudly."""
    if not isinstance(config, Mapping) or "policy" not in config:
        raise ValueError("missing_atoms.policy is required (strict | drop_residue_missing_backbone)")
    policy = config["policy"]
    if policy not in POLICIES:
        raise ValueError(f"missing_atoms.policy must be one of {POLICIES}, got {policy!r}")
    allowed = {"policy"} if policy == "strict" else {
        "policy", "incomplete_sidechain", "max_incomplete_sidechain_fraction"
    }
    extra = set(config) - allowed
    if extra:
        raise ValueError(f"missing_atoms has keys not used by policy {policy!r}: {sorted(extra)}")
    if policy == "strict":
        return MissingAtomPolicy("strict", None, None)
    sidechain = config.get("incomplete_sidechain")
    if sidechain not in SIDECHAIN_OPTIONS:
        raise ValueError(
            f"missing_atoms.incomplete_sidechain must be one of {SIDECHAIN_OPTIONS}, got {sidechain!r}"
        )
    fraction = config.get("max_incomplete_sidechain_fraction")
    if sidechain == "mask":
        if fraction is not None:
            raise ValueError("max_incomplete_sidechain_fraction is only used with drop_chain_if_fraction_above")
        return MissingAtomPolicy(policy, sidechain, None)
    if not isinstance(fraction, (int, float)) or not 0.0 <= float(fraction) < 1.0:
        raise ValueError(f"max_incomplete_sidechain_fraction must be in [0, 1), got {fraction!r}")
    return MissingAtomPolicy(policy, sidechain, float(fraction))


@dataclass
class PolicyResult:
    residues: List[ResidueRecord]
    masked: List[bool]
    report: Dict[str, object]
    dropped_reason: Optional[str]
    sequence_index: List[int]  # position of each kept residue in the parsed chain


def _residue_label(residue: ResidueRecord) -> str:
    return f"{residue.resname}{residue.resseq}{residue.icode}"


def apply_policy(chain: ChainRecord, policy: MissingAtomPolicy) -> PolicyResult:
    """Return the residues to convert, their mask flags, and the chain QC record."""
    kept: List[ResidueRecord] = []
    masked: List[bool] = []
    sequence_index: List[int] = []
    missing_backbone: List[str] = []
    incomplete: List[str] = []
    unexpected_atoms = 0
    for position, residue in enumerate(chain.residues):
        status = residue_completeness(residue)
        if status.unexpected:
            unexpected_atoms += len(status.unexpected)
            residue = ResidueRecord(
                residue.resname, residue.source_resname, residue.resseq, residue.icode,
                {k: v for k, v in residue.atoms.items() if k not in status.unexpected},
            )
        if status.missing_backbone:
            missing_backbone.append(f"{_residue_label(residue)}:-{','.join(status.missing_backbone)}")
            continue
        kept.append(residue)
        masked.append(bool(status.missing_sidechain))
        sequence_index.append(position)
        if status.missing_sidechain:
            incomplete.append(f"{_residue_label(residue)}:-{','.join(status.missing_sidechain)}")

    report = dict(chain.report)
    report.update({
        "policy": policy.tag,
        "residues_missing_backbone": len(missing_backbone),
        "residues_incomplete_sidechain": len(incomplete),
        "unexpected_atoms_removed": unexpected_atoms,
        "missing_backbone_examples": missing_backbone[:10],
        "incomplete_sidechain_examples": incomplete[:10],
    })
    if policy.policy == "strict" and (missing_backbone or incomplete):
        raise MissingAtomError(
            f"{chain.structure_id} chain {chain.chain_id}: {len(missing_backbone)} residues missing "
            f"backbone {missing_backbone[:20]}, {len(incomplete)} incomplete sidechains {incomplete[:20]}"
        )
    dropped = None
    fraction = len(incomplete) / len(kept) if kept else 0.0
    report["incomplete_sidechain_fraction"] = round(fraction, 6)
    if (
        policy.incomplete_sidechain == "drop_chain_if_fraction_above"
        and fraction > policy.max_incomplete_sidechain_fraction
    ):
        dropped = (
            f"incomplete_sidechain_fraction {fraction:.3f} > "
            f"{policy.max_incomplete_sidechain_fraction:g}"
        )
    report["kept_residues"] = 0 if dropped else len(kept)
    report["masked_residues"] = 0 if dropped else int(sum(masked))
    return PolicyResult([] if dropped else kept, [] if dropped else masked, report, dropped,
                        [] if dropped else sequence_index)

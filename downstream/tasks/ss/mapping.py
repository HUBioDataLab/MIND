"""Literature chain (sequence + per-residue labels) -> labels on the residues of a 3D structure.

The literature secondary-structure sets (TAPE / PEER release of the NetSurfP-2.0 data) are sequence plus label arrays
with PDB-like ids. The encoder needs coordinates, so each dataset chain is matched to the residues of its RCSB
structure by sequence alignment. Nothing is dropped silently: every dataset residue ends in exactly one bucket of
``RESIDUE_REASONS`` (``account_residues`` raises when the counts do not add up) and every structure residue that
receives no label carries a reason. Labels are the dataset's own (never recomputed with DSSP).

Label definitions (verified on the TAPE secondary-structure LMDB against ``mkdssp`` 4.6.1 on four PDB chains, see
``Docs`` of the task): ``ss8`` index 0..7 is the DSSP class ``GHIBESTL`` (L = coil, ``-`` of DSSP) and ``ss3`` index 0..2
is helix / strand / coil with the standard collapse H = {G, H, I}, E = {B, E}, C = {S, T, L}.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from Bio import Align

SS8_CLASSES = "GHIBESTL"
SS3_CLASSES = "HEC"
SS8_TO_SS3 = {"G": "H", "H": "H", "I": "H", "B": "E", "E": "E", "S": "C", "T": "C", "L": "C"}
SS8_INDEX_TO_SS3_INDEX = np.array([SS3_CLASSES.index(SS8_TO_SS3[c]) for c in SS8_CLASSES], dtype=np.int64)
COIL_INDEX = {"ss3": SS3_CLASSES.index("C"), "ss8": SS8_CLASSES.index("L")}
LABEL_SETS = {"ss3": SS3_CLASSES, "ss8": SS8_CLASSES}

# Every dataset residue is in exactly one bucket.
EVALUATED = "evaluated"
RESIDUE_REASONS = (
    EVALUATED,
    "not_in_structure_flagged_unresolved",   # no aligned structure residue; the dataset itself marks it unresolved
    "not_in_structure_unflagged",            # no aligned structure residue; the dataset does not mark it unresolved
    "sequence_mismatch",                     # aligned to a structure residue of another amino acid (incl. dataset 'X')
    "dropped_missing_backbone",              # aligned and identical, but the structure residue lacks a backbone atom
    "chain_not_processed",                   # the whole chain could not be mapped (reason at chain level)
)


def check_label_arrays(chain: str, sequence: str, ss3: np.ndarray, ss8: np.ndarray, observed: np.ndarray) -> None:
    """Dataset record sanity: equal lengths, class ranges and ss3 == collapse(ss8) at every residue."""
    n = len(sequence)
    if not (len(ss3) == len(ss8) == len(observed) == n) or n == 0:
        raise ValueError(f"{chain}: lengths sequence {n}, ss3 {len(ss3)}, ss8 {len(ss8)}, observed {len(observed)}")
    if int(ss8.min()) < 0 or int(ss8.max()) >= len(SS8_CLASSES) or int(ss3.min()) < 0 or int(ss3.max()) >= len(SS3_CLASSES):
        raise ValueError(f"{chain}: label outside ss3 [0,3) / ss8 [0,8)")
    if not np.array_equal(SS8_INDEX_TO_SS3_INDEX[ss8], ss3):
        bad = int((SS8_INDEX_TO_SS3_INDEX[ss8] != ss3).sum())
        raise ValueError(f"{chain}: ss3 differs from the standard collapse of ss8 at {bad} residues")


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def _substitution_matrix():
    """+2 identical, -1 different; 'X' (the dataset's unknown / nonstandard residue) scores 0 against everything."""
    from Bio.Align import substitution_matrices

    matrix = substitution_matrices.Array(alphabet=ALPHABET, dims=2)
    for a in ALPHABET:
        for b in ALPHABET:
            matrix[a, b] = 0.0 if "X" in (a, b) else (2.0 if a == b else -1.0)
    return matrix


_ALIGNER = Align.PairwiseAligner()
_ALIGNER.mode = "global"
_ALIGNER.substitution_matrix = _substitution_matrix()
_ALIGNER.open_gap_score = -4.0
_ALIGNER.extend_gap_score = -0.5
_ALIGNER.end_gap_score = 0.0   # both sequences may be longer at either end (unresolved termini, tags)


def align_structure_to_dataset(structure_seq: str, dataset_seq: str) -> np.ndarray:
    """``pair[i]`` = dataset index aligned to structure residue ``i``, or -1 (deterministic first optimal alignment)."""
    if not structure_seq or not dataset_seq:
        raise ValueError("cannot align an empty sequence")
    alignment = _ALIGNER.align(dataset_seq, structure_seq)[0]
    pair = np.full(len(structure_seq), -1, dtype=np.int64)
    for (d0, d1), (s0, s1) in zip(*alignment.aligned):
        pair[s0:s1] = np.arange(d0, d1)
    return pair


@dataclass
class ChainMapping:
    """Result of mapping one chain; ``labels_*`` are per STRUCTURE residue (-1 = no label) in the order given."""

    pair: np.ndarray
    labels: Dict[str, np.ndarray]
    structure_reasons: List[str]
    identical: int
    mismatched: int
    unaligned_structure: int

    @property
    def aligned(self) -> int:
        return self.identical + self.mismatched

    @property
    def identity(self) -> float:
        """Identical / aligned residue pairs (0 when nothing is aligned)."""
        return self.identical / self.aligned if self.aligned else 0.0

    def aligned_fraction(self, dataset_length: int) -> float:
        """Aligned pairs over the shorter of the two sequences: a dataset REGION of a longer chain (CB513 holds 513
        regions of 434 proteins) aligns completely, a random alignment of unrelated sequences does not."""
        return self.aligned / min(len(self.pair), dataset_length)


def map_labels(structure_seq: str, dataset_seq: str, ss: Dict[str, np.ndarray],
               selenomethionine: Optional[Sequence[bool]] = None) -> ChainMapping:
    """Align and build labels on the structure residues; a mismatching or unaligned residue gets -1.

    ``selenomethionine[i]`` marks a structure residue that was MSE and is read as MET: the dataset writes ``X`` for MSE, and an
    aligned ``X`` there is the same residue, so it keeps its label (reason ``ok_selenomethionine``, counted as identical).
    """
    pair = align_structure_to_dataset(structure_seq, dataset_seq)
    labels = {name: np.full(len(pair), -1, dtype=np.int64) for name in ss}
    reasons, identical, mismatched = [], 0, 0
    for i, j in enumerate(pair):
        if j >= 0 and selenomethionine is not None and selenomethionine[i] and dataset_seq[j] == "X":
            identical += 1
            reasons.append("ok_selenomethionine")
            for name, values in ss.items():
                labels[name][i] = values[j]
        elif j < 0:
            reasons.append("no_dataset_counterpart")
        elif structure_seq[i] != dataset_seq[j]:
            mismatched += 1
            reasons.append("mismatch_dataset_x" if dataset_seq[j] == "X" else "mismatch_other")
        else:
            identical += 1
            reasons.append("ok")
            for name, values in ss.items():
                labels[name][i] = values[j]
    return ChainMapping(pair, labels, reasons, identical, mismatched, int((pair < 0).sum()))


def account_residues(chain: str, dataset_length: int, buckets: Counter) -> Dict[str, int]:
    """The reason counts of one chain; raises unless every dataset residue is in exactly one known bucket."""
    unknown = set(buckets) - set(RESIDUE_REASONS)
    if unknown:
        raise ValueError(f"{chain}: unknown residue reasons {sorted(unknown)}")
    if sum(buckets.values()) != dataset_length:
        raise ValueError(f"{chain}: residue buckets {dict(buckets)} sum to {sum(buckets.values())}, "
                         f"dataset length {dataset_length}")
    return {reason: int(buckets.get(reason, 0)) for reason in RESIDUE_REASONS}


def dataset_residue_buckets(dataset_length: int, observed: np.ndarray, mapping: ChainMapping, structure_kept: Sequence[bool],
                            structure_seq: str, dataset_seq: str) -> Counter:  # noqa: C901
    """Bucket every dataset residue of a mapped chain.

    ``structure_kept[i]`` says whether structure residue ``i`` survived the missing-atom policy (and so carries its label).
    """
    buckets: Counter = Counter()
    claimed = np.zeros(dataset_length, dtype=bool)
    for i, j in enumerate(mapping.pair):
        if j < 0:
            continue
        claimed[j] = True
        if mapping.structure_reasons[i] in ("mismatch_other", "mismatch_dataset_x"):
            buckets["sequence_mismatch"] += 1
        elif structure_kept[i]:
            buckets[EVALUATED] += 1
        else:
            buckets["dropped_missing_backbone"] += 1
    for j in np.where(~claimed)[0]:
        buckets["not_in_structure_flagged_unresolved" if int(observed[j]) == 0 else "not_in_structure_unflagged"] += 1
    return buckets

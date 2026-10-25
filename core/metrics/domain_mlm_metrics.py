"""Domain-agnostic masked-atom (MLM) metrics, accumulated per domain.

The MLM head predicts one of 121 classes: atomic numbers 0..118, R=119 for
rare real elements and V=120 for the fixed-slot padding target. Every metric
below is computed over *masked* atoms only, split by the graph's domain
(``batch.dataset_type``: protein, rna, unimol, coconut, qm9).

Metric definitions (all counts are exact; nothing is sampled):

* ``top{1,3,5}_acc_{all,real,virtual}``: the target is among the k highest
  logits. ``real`` = target != V, ``virtual`` = target == V. Top-1 is the
  argmax, identical to ``mlm_preds_atomic_num``.
* ``top1_acc_real_noH``: real targets other than hydrogen (Z=1); for the
  small-molecule domains this is the heavy-atom element accuracy.
* per element (rows = classes present as target or prediction): support
  (#targets), predicted (#predictions), recall = hits/support, precision =
  hits/predicted. Precision is taken over the *predicted* element, as in the
  corrected ``per_atom_metrics.py``: of all masked atoms predicted as the
  element, the fraction whose true element is that element.
* confusion matrix over the classes present (rows = target, cols = top-1).
* ``ce_{real,virtual,all}``: mean of -log p(target) (natural log, float32
  log-softmax of the logits); ``ppl_*`` = exp(ce). The training MLM loss is
  ``ce_real + virtual_atom_loss_weight * ce_virtual`` per batch; these are the
  unweighted per-atom means.
* ``ece_{all,real}``: expected calibration error with 15 equal-width bins of
  the top-1 probability, bins ``((i-1)/15, i/15]`` (confidence 0 in bin 1):
  sum_b n_b/N * |accuracy_b - mean confidence_b|.
* protein residue recovery: a residue counts when all 10 sidechain slots are
  masked. The adapter sorts sidechain slots C, N, O, S and then V
  (``PROTEIN_SIDECHAIN_ATOM_PRIORITY``), so the slot pattern equals the
  element composition (V count included). The predicted composition is
  mapped back to a residue class; LEU/ILE (C4) and VAL/PRO (C3) share a
  composition, so there are 18 classes. A residue is correct iff its
  predicted composition maps to the true residue's class.
* RNA nucleotide identity: the same rule over the 11 base slots; A, G, C, U
  have distinct base compositions, so there are four classes.
  Residues/nucleotides are skipped, with a counted reason, when the symbol is
  non-standard, the slot count is not 10/11, the block is only partially
  masked, or its target composition differs from the canonical one.
* distance-head health (*visible pairs, diagnostic only*): pairs scored by the
  long-range (random within-graph pairs) and short-range (physical edges)
  losses, both restricted to encoder-visible atoms. Bin accuracy, within +-1
  bin accuracy, and expected-distance MAE in Angstrom using the bin centers
  of ``PretrainingTasks.distance_bin_centers`` (open-ended long-range last bin
  = its lower edge, so ``mae_bounded_A`` excludes pairs whose true bin is
  that last bin).
* embedding health on sampled encoder-visible, unmasked node states: mean/std
  of the L2 norm, effective rank exp(H(sigma / sum sigma)) of the uncentred
  sample matrix (RankMe), and mean pairwise cosine similarity over the first
  1024 sampled nodes.

Undefined values (e.g. no V target in a domain) are ``None``, never 0.0.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from core.masked_slot_mapping import flatten_pos_codes
from core.molecular_encoder import PROTEIN_LOCAL_SLOT_COUNT, RNA_LOCAL_SLOT_COUNT
from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS,
    PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE,
    PROTEIN_TERMINAL_ATOMS,
    RNA_BACKBONE_ATOMS,
    RNA_BASE_ATOMS_BY_NUCLEOTIDE,
    RNA_TERMINAL_ATOMS,
    VIRTUAL_ATOMIC_NUMBER,
    normalize_rna_position_code,
)

NUM_CLASSES = VIRTUAL_ATOMIC_NUMBER + 1
HYDROGEN = 1
TOP_K = (1, 3, 5)
ECE_BINS = 15
COSINE_SAMPLE = 1024
NO_MATCH = "NO_MATCH"

DOMAIN_ALIASES = {
    "pdb": "protein",
    "protein": "protein",
    "rna": "rna",
    "unimol": "unimol",
    "coconut": "coconut",
    "metabolite": "coconut",
    "qm9": "qm9",
}
DOMAIN_ORDER = ("protein", "rna", "unimol", "coconut", "qm9")
SMALL_MOLECULE_DOMAINS = ("unimol", "coconut", "qm9")

_PERIODIC = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
    "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()

# Element letter of a canonical protein/RNA position name (CB, NE2, OD1, SG ...).
_POSITION_ELEMENT = {"C": 6, "N": 7, "O": 8, "S": 16, "P": 15}


def canonical_domain(dataset_type) -> str:
    key = str(dataset_type).lower()
    if key not in DOMAIN_ALIASES:
        raise ValueError(
            f"unknown dataset_type {dataset_type!r} for domain MLM metrics "
            f"(known: {sorted(DOMAIN_ALIASES)})"
        )
    return DOMAIN_ALIASES[key]


def class_label(z: int) -> str:
    """Display name of an MLM class (V = padding target 120; vanadium is V23)."""
    z = int(z)
    if z == VIRTUAL_ATOMIC_NUMBER:
        return "V"
    if z == 119:
        return "R"
    if z == 0:
        return "MASK"
    if z == 23:
        return "V23"
    if 1 <= z <= len(_PERIODIC):
        return _PERIODIC[z - 1]
    return f"Z{z}"


def position_element(name: str) -> int:
    letter = str(name)[:1]
    if letter not in _POSITION_ELEMENT:
        raise ValueError(f"position name {name!r} has no known element letter")
    return _POSITION_ELEMENT[letter]


def composition_key(atomic_numbers: Iterable[int]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted(Counter(int(z) for z in atomic_numbers).items()))


def _slot_compositions(atoms_by_block: Dict[str, Sequence[str]], slots: int) -> Dict[str, tuple]:
    compositions = {}
    for name, atoms in atoms_by_block.items():
        if len(atoms) > slots:
            raise ValueError(f"{name}: {len(atoms)} identity atoms exceed {slots} slots")
        elements = [position_element(atom) for atom in atoms]
        compositions[name] = composition_key(elements + [VIRTUAL_ATOMIC_NUMBER] * (slots - len(atoms)))
    return compositions


def composition_classes(compositions: Dict[str, tuple]):
    """Merge names that share a composition: (composition -> class, name -> class)."""
    by_composition = defaultdict(list)
    for name, composition in compositions.items():
        by_composition[composition].append(name)
    composition_to_class = {
        composition: "|".join(sorted(names)) for composition, names in by_composition.items()
    }
    name_to_class = {name: composition_to_class[c] for name, c in compositions.items()}
    return composition_to_class, name_to_class


PROTEIN_SIDECHAIN_SLOTS = PROTEIN_LOCAL_SLOT_COUNT - len(PROTEIN_BACKBONE_ATOMS)
RNA_BASE_SLOTS = RNA_LOCAL_SLOT_COUNT - len(RNA_BACKBONE_ATOMS)
PROTEIN_SIDECHAIN_COMPOSITIONS = _slot_compositions(PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE, PROTEIN_SIDECHAIN_SLOTS)
RNA_BASE_COMPOSITIONS = _slot_compositions(RNA_BASE_ATOMS_BY_NUCLEOTIDE, RNA_BASE_SLOTS)

# Same aliases as the RNA per-atom callback; anything else is a modified base.
RNA_NUCLEOTIDE_ALIASES = {
    "A": "A", "RA": "A", "ADE": "A", "DA": "A",
    "G": "G", "RG": "G", "GUA": "G", "DG": "G",
    "C": "C", "RC": "C", "CYT": "C", "DC": "C",
    "U": "U", "RU": "U", "URA": "U", "DT": "U",
}


class _BlockSpec:
    """Which nodes of a block carry the identity, and how to name the block."""

    def __init__(self, domain, slots, compositions, non_identity_codes, normalize_code, normalize_symbol):
        self.domain = domain
        self.slots = slots
        self.compositions = compositions
        self.composition_to_class, self.name_to_class = composition_classes(compositions)
        self.non_identity_codes = frozenset(non_identity_codes)
        self.normalize_code = normalize_code
        self.normalize_symbol = normalize_symbol

    def identity_slot_mask(self, pos_codes: Sequence[str]) -> np.ndarray:
        return np.fromiter(
            (self.normalize_code(code) not in self.non_identity_codes for code in pos_codes),
            dtype=bool, count=len(pos_codes),
        )


BLOCK_SPECS = {
    "protein": _BlockSpec(
        "protein", PROTEIN_SIDECHAIN_SLOTS, PROTEIN_SIDECHAIN_COMPOSITIONS,
        PROTEIN_BACKBONE_ATOMS + PROTEIN_TERMINAL_ATOMS,
        lambda code: str(code),
        lambda symbol: str(symbol).upper().strip(),
    ),
    "rna": _BlockSpec(
        "rna", RNA_BASE_SLOTS, RNA_BASE_COMPOSITIONS,
        RNA_BACKBONE_ATOMS + RNA_TERMINAL_ATOMS,
        normalize_rna_position_code,
        lambda symbol: RNA_NUCLEOTIDE_ALIASES.get(str(symbol).upper().strip(), str(symbol).upper().strip()),
    ),
}


def _ratio(numerator, denominator) -> Optional[float]:
    return float(numerator) / float(denominator) if denominator else None


def _as_numpy(value, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()  # numpy has no bfloat16; float32 is exact
        value = value.numpy()
    array = np.asarray(value)
    return array.astype(dtype) if dtype is not None else array


# ---------------------------------------------------------------------------
# Per-domain accumulators
# ---------------------------------------------------------------------------


class MaskedAtomStats:
    """Top-k, element, cross-entropy and calibration counts for one domain."""

    GROUPS = ("all", "real", "virtual", "real_noH")

    def __init__(self):
        self.count = Counter()
        self.topk_hits = {k: Counter() for k in TOP_K}
        self.ce_sum = Counter()
        self.confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        self.ece = {group: np.zeros((3, ECE_BINS), dtype=np.float64) for group in ("all", "real")}

    def update(self, targets, topk, target_log_prob, top1_prob) -> None:
        targets = _as_numpy(targets, np.int64)
        topk = _as_numpy(topk, np.int64)
        target_log_prob = _as_numpy(target_log_prob, np.float64)
        top1_prob = _as_numpy(top1_prob, np.float64)
        n = targets.shape[0]
        if targets.ndim != 1 or topk.ndim != 2 or topk.shape[0] != n:
            raise ValueError(f"targets {targets.shape} and topk {topk.shape} do not align")
        if topk.shape[1] < max(TOP_K):
            raise ValueError(f"topk has {topk.shape[1]} columns; {max(TOP_K)} are required")
        if target_log_prob.shape != (n,) or top1_prob.shape != (n,):
            raise ValueError("target_log_prob/top1_prob must have one value per masked atom")
        for name, values in (("targets", targets), ("topk", topk)):
            if values.size and (values.min() < 0 or values.max() >= NUM_CLASSES):
                raise ValueError(f"{name} outside the {NUM_CLASSES} MLM classes")
        if not np.all(np.isfinite(target_log_prob)) or np.any(target_log_prob > 1e-6):
            raise ValueError("target_log_prob must be finite log-probabilities (<= 0)")
        if np.any(~np.isfinite(top1_prob)) or np.any(top1_prob < 0) or np.any(top1_prob > 1 + 1e-6):
            raise ValueError("top1_prob must lie in [0, 1]")
        if n == 0:
            return

        real = targets != VIRTUAL_ATOMIC_NUMBER
        groups = {
            "all": np.ones(n, dtype=bool),
            "real": real,
            "virtual": ~real,
            "real_noH": real & (targets != HYDROGEN),
        }
        hits = {k: (topk[:, :k] == targets[:, None]).any(axis=1) for k in TOP_K}
        for group, mask in groups.items():
            self.count[group] += int(mask.sum())
            self.ce_sum[group] += float(-target_log_prob[mask].sum())
            for k in TOP_K:
                self.topk_hits[k][group] += int((hits[k] & mask).sum())
        self.count["hydrogen"] += int((targets == HYDROGEN).sum())
        np.add.at(self.confusion, (targets, topk[:, 0]), 1)

        bins = np.clip(np.ceil(top1_prob * ECE_BINS).astype(np.int64) - 1, 0, ECE_BINS - 1)
        for group in ("all", "real"):
            mask = groups[group]
            table = self.ece[group]
            np.add.at(table[0], bins[mask], 1.0)
            np.add.at(table[1], bins[mask], top1_prob[mask])
            np.add.at(table[2], bins[mask], hits[1][mask].astype(np.float64))

    @staticmethod
    def expected_calibration_error(table: np.ndarray) -> Optional[float]:
        total = table[0].sum()
        if total == 0:
            return None
        filled = table[0] > 0
        gaps = np.abs(table[2][filled] - table[1][filled]) / table[0][filled]
        return float((table[0][filled] / total * gaps).sum())

    def summary(self) -> dict:
        out = {
            "n_masked": self.count["all"],
            "n_masked_real": self.count["real"],
            "n_masked_virtual": self.count["virtual"],
            "n_masked_hydrogen": self.count["hydrogen"],
        }
        for k in TOP_K:
            for group in ("all", "real", "virtual"):
                out[f"top{k}_acc_{group}"] = _ratio(self.topk_hits[k][group], self.count[group])
        out["top1_acc_real_noH"] = _ratio(self.topk_hits[1]["real_noH"], self.count["real_noH"])
        for group in ("real", "virtual", "all"):
            ce = _ratio(self.ce_sum[group], self.count[group])
            out[f"ce_{group}"] = ce
            out[f"ppl_{group}"] = math.exp(ce) if ce is not None else None
        for group in ("all", "real"):
            out[f"ece_{group}"] = self.expected_calibration_error(self.ece[group])
        return out

    def element_table(self) -> List[dict]:
        support = self.confusion.sum(axis=1)
        predicted = self.confusion.sum(axis=0)
        rows = []
        for z in np.where((support > 0) | (predicted > 0))[0]:
            hits = int(self.confusion[z, z])
            rows.append({
                "class": int(z),
                "element": class_label(z),
                "support": int(support[z]),
                "predicted": int(predicted[z]),
                "recall": _ratio(hits, support[z]),
                "precision": _ratio(hits, predicted[z]),
            })
        return rows

    def confusion_table(self) -> dict:
        present = np.where((self.confusion.sum(axis=1) > 0) | (self.confusion.sum(axis=0) > 0))[0]
        return {
            "classes": [int(z) for z in present],
            "labels": [class_label(z) for z in present],
            "matrix": self.confusion[np.ix_(present, present)].tolist(),
        }

    def calibration_table(self) -> List[dict]:
        rows = []
        for group in ("all", "real"):
            table = self.ece[group]
            for b in range(ECE_BINS):
                n = int(table[0, b])
                rows.append({
                    "group": group,
                    "bin": b + 1,
                    "lower": b / ECE_BINS,
                    "upper": (b + 1) / ECE_BINS,
                    "count": n,
                    "mean_confidence": _ratio(table[1, b], n),
                    "accuracy": _ratio(table[2, b], n),
                })
        return rows


class BlockIdentityStats:
    """Residue (protein) or nucleotide (RNA) identity from slot compositions."""

    def __init__(self, spec: _BlockSpec):
        self.spec = spec
        self.total = Counter()
        self.correct = Counter()
        self.class_confusion = Counter()
        self.skipped = Counter()

    def update_block(self, symbol: str, target_slots: Sequence[int], predicted_slots: Sequence[int]) -> None:
        """One fully masked block: its symbol and the target/top-1 class of each slot."""
        spec = self.spec
        name = spec.normalize_symbol(symbol)
        if name not in spec.name_to_class:
            self.skipped["nonstandard_symbol"] += 1
            return
        if composition_key(target_slots) != spec.compositions[name]:
            self.skipped["target_composition_mismatch"] += 1
            return
        true_class = spec.name_to_class[name]
        predicted_class = spec.composition_to_class.get(composition_key(predicted_slots), NO_MATCH)
        self.total[name] += 1
        self.correct[name] += int(predicted_class == true_class)
        self.class_confusion[(true_class, predicted_class)] += 1

    def summary(self) -> dict:
        total = sum(self.total.values())
        out = {
            "n_blocks": total,
            "accuracy": _ratio(sum(self.correct.values()), total),
            "n_classes": len(set(self.spec.name_to_class.values())),
            "skipped": {reason: count for reason, count in sorted(self.skipped.items()) if count},
        }
        class_total, class_correct = Counter(), Counter()
        for (true_class, predicted_class), count in self.class_confusion.items():
            class_total[true_class] += count
            class_correct[true_class] += count * int(true_class == predicted_class)
        out["per_name"] = {
            name: {"support": self.total[name], "recall": _ratio(self.correct[name], self.total[name])}
            for name in sorted(self.total)
        }
        out["per_class"] = {
            cls: {"support": class_total[cls], "recall": _ratio(class_correct[cls], class_total[cls])}
            for cls in sorted(class_total)
        }
        out["class_confusion"] = [
            {"true_class": t, "predicted_class": p, "count": c}
            for (t, p), c in sorted(self.class_confusion.items())
        ]
        return out


class DistanceHeadStats:
    """Visible-pair distance-head diagnostic for one head and domain."""

    def __init__(self):
        self.count = Counter()
        self.abs_error = Counter()
        self.num_bins = None

    def update(self, pred_bin, true_bin, expected_distance, true_distance, num_bins: int) -> None:
        pred_bin = _as_numpy(pred_bin, np.int64)
        true_bin = _as_numpy(true_bin, np.int64)
        expected_distance = _as_numpy(expected_distance, np.float64)
        true_distance = _as_numpy(true_distance, np.float64)
        n = pred_bin.shape[0]
        if not (true_bin.shape == expected_distance.shape == true_distance.shape == (n,)):
            raise ValueError("distance diagnostic arrays must have one value per pair")
        if self.num_bins is not None and self.num_bins != int(num_bins):
            raise ValueError(f"distance head bin count changed: {self.num_bins} -> {num_bins}")
        self.num_bins = int(num_bins)
        if n and (true_bin.min() < 0 or true_bin.max() >= num_bins or pred_bin.min() < 0 or pred_bin.max() >= num_bins):
            raise ValueError(f"distance bins outside 0..{num_bins - 1}")
        if not (np.all(np.isfinite(expected_distance)) and np.all(np.isfinite(true_distance))):
            raise ValueError("distance diagnostics contain non-finite distances")
        error = np.abs(expected_distance - true_distance)
        bounded = true_bin < num_bins - 1
        self.count["pairs"] += n
        self.count["bin_hits"] += int((pred_bin == true_bin).sum())
        self.count["within1"] += int((np.abs(pred_bin - true_bin) <= 1).sum())
        self.count["bounded_pairs"] += int(bounded.sum())
        self.abs_error["all"] += float(error.sum())
        self.abs_error["bounded"] += float(error[bounded].sum())

    def summary(self) -> dict:
        return {
            "n_pairs": self.count["pairs"],
            "bin_acc": _ratio(self.count["bin_hits"], self.count["pairs"]),
            "within1_bin_acc": _ratio(self.count["within1"], self.count["pairs"]),
            "mae_A": _ratio(self.abs_error["all"], self.count["pairs"]),
            "n_bounded_pairs": self.count["bounded_pairs"],
            "mae_bounded_A": _ratio(self.abs_error["bounded"], self.count["bounded_pairs"]),
        }


class EmbeddingStats:
    """Norm, effective rank and mean pairwise cosine of sampled node states."""

    def __init__(self, max_nodes: int):
        self.max_nodes = int(max_nodes)
        self.parts: List[np.ndarray] = []
        self.size = 0

    @property
    def full(self) -> bool:
        return self.size >= self.max_nodes

    def update(self, embeddings) -> None:
        embeddings = _as_numpy(embeddings, np.float64)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be [nodes, dim], got {embeddings.shape}")
        if self.parts and embeddings.shape[1] != self.parts[0].shape[1]:
            raise ValueError("embedding width changed between batches")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("node embeddings contain non-finite values")
        take = embeddings[: max(self.max_nodes - self.size, 0)]
        if take.shape[0]:
            self.parts.append(take)
            self.size += take.shape[0]

    def summary(self) -> dict:
        if not self.parts:
            return {"n_nodes": 0}
        matrix = np.concatenate(self.parts)
        norms = np.linalg.norm(matrix, axis=1)
        singular = np.linalg.svd(matrix, compute_uv=False)
        effective_rank = None
        if singular.sum() > 0:
            p = singular / singular.sum()
            p = p[p > 0]
            effective_rank = float(np.exp(-(p * np.log(p)).sum()))
        sample = matrix[:COSINE_SAMPLE]
        sample_norms = norms[:COSINE_SAMPLE]
        cosine = None
        if sample.shape[0] >= 2:
            if np.any(sample_norms == 0):
                raise ValueError("a sampled node embedding has zero norm; cosine is undefined")
            unit = sample / sample_norms[:, None]
            gram = unit @ unit.T
            m = gram.shape[0]
            cosine = float((gram.sum() - np.trace(gram)) / (m * (m - 1)))
        return {
            "n_nodes": int(matrix.shape[0]),
            "dim": int(matrix.shape[1]),
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
            "effective_rank": effective_rank,
            "mean_pairwise_cosine": cosine,
        }


# ---------------------------------------------------------------------------
# Batch-level accumulator
# ---------------------------------------------------------------------------


def graph_domains(batch) -> List[str]:
    raw = getattr(batch, "dataset_type", None)
    if raw is None:
        raise ValueError("domain MLM metrics require batch.dataset_type")
    num_graphs = int(getattr(batch, "num_graphs", 1) or 1)
    values = list(raw) if isinstance(raw, (list, tuple)) else [raw] * num_graphs
    if len(values) != num_graphs:
        raise ValueError(f"dataset_type has {len(values)} entries for {num_graphs} graphs")
    return [canonical_domain(value) for value in values]


def _per_graph_block_symbols(raw) -> List[List[str]]:
    if isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple)):
        return [list(symbols) for symbols in raw]
    return [list(raw)]


class DomainMLMMetrics:
    """Accumulate every domain metric over validation/test batches."""

    def __init__(self, embedding_max_nodes_per_domain: int = 4096):
        self.embedding_max_nodes_per_domain = int(embedding_max_nodes_per_domain)
        self.reset()

    def reset(self) -> None:
        self.atoms: Dict[str, MaskedAtomStats] = defaultdict(MaskedAtomStats)
        self.blocks: Dict[str, BlockIdentityStats] = {
            domain: BlockIdentityStats(spec) for domain, spec in BLOCK_SPECS.items()
        }
        self.distance: Dict[Tuple[str, str], DistanceHeadStats] = defaultdict(DistanceHeadStats)
        self.embeddings: Dict[str, EmbeddingStats] = defaultdict(
            lambda: EmbeddingStats(self.embedding_max_nodes_per_domain)
        )
        self.batches = Counter()

    @property
    def seen_domains(self) -> List[str]:
        return [d for d in DOMAIN_ORDER if self.atoms.get(d) is not None and self.atoms[d].count["all"] > 0]

    # -- pure entry points (also used by tests) ---------------------------

    def update_masked_atoms(self, domain, targets, topk, target_log_prob, top1_prob) -> None:
        self.atoms[canonical_domain(domain)].update(targets, topk, target_log_prob, top1_prob)

    def update_block(self, domain, symbol, target_slots, predicted_slots) -> None:
        self.blocks[canonical_domain(domain)].update_block(symbol, target_slots, predicted_slots)

    def update_distance(self, domain, head, pred_bin, true_bin, expected_distance, true_distance, num_bins) -> None:
        self.distance[(canonical_domain(domain), head)].update(
            pred_bin, true_bin, expected_distance, true_distance, num_bins
        )

    def update_embeddings(self, domain, embeddings) -> None:
        self.embeddings[canonical_domain(domain)].update(embeddings)

    # -- batch adapter ----------------------------------------------------

    def update_from_batch(self, batch, outputs: dict) -> None:
        if not isinstance(outputs, dict) or outputs.get("mlm_eval") is None:
            raise ValueError("domain MLM metrics require validation/test outputs with 'mlm_eval'")
        domains = graph_domains(batch)
        graph_of_node = _as_numpy(batch.batch, np.int64)
        node_domain = np.asarray(domains, dtype=object)[graph_of_node]
        for domain in set(domains):
            self.batches[domain] += 1

        record = outputs["mlm_eval"]
        node_index = _as_numpy(record["node_index"], np.int64)
        mlm_mask = _as_numpy(batch.mlm_mask, bool)
        if not np.array_equal(node_index, np.where(mlm_mask)[0]):
            raise ValueError("mlm_eval.node_index does not match batch.mlm_mask")
        targets = _as_numpy(record["target"], np.int64)
        original = _as_numpy(batch.original_types, np.int64)
        if not np.array_equal(targets, original[node_index]):
            raise ValueError("mlm_eval.target does not match batch.original_types")
        topk = _as_numpy(record["topk"], np.int64)
        log_prob = _as_numpy(record["target_log_prob"], np.float64)
        top1_prob = _as_numpy(record["top1_prob"], np.float64)
        masked_domain = node_domain[node_index]
        for domain in sorted(set(masked_domain.tolist())):
            sel = masked_domain == domain
            self.atoms[domain].update(targets[sel], topk[sel], log_prob[sel], top1_prob[sel])

        block_domains = [d for d in BLOCK_SPECS if d in domains]
        if block_domains and node_index.size:
            top1 = np.full(original.shape[0], -1, dtype=np.int64)
            top1[node_index] = topk[:, 0]
            self._update_blocks_from_batch(batch, block_domains, node_domain, graph_of_node,
                                           mlm_mask, original, top1)

        for head, rec in (outputs.get("distance_eval") or {}).items():
            pair_domain = node_domain[_as_numpy(rec["node_index"], np.int64)]
            for domain in sorted(set(pair_domain.tolist())):
                sel = pair_domain == domain
                self.distance[(domain, head)].update(
                    _as_numpy(rec["pred_bin"])[sel], _as_numpy(rec["true_bin"])[sel],
                    _as_numpy(rec["expected_distance"])[sel], _as_numpy(rec["true_distance"])[sel],
                    rec["num_bins"],
                )

        sample = outputs.get("embedding_sample")
        if sample is not None:
            sample_domain = node_domain[_as_numpy(sample["node_index"], np.int64)]
            embedding = _as_numpy(sample["embedding"], np.float64)
            for domain in sorted(set(sample_domain.tolist())):
                if not self.embeddings[domain].full:
                    self.embeddings[domain].update(embedding[sample_domain == domain])

    def _update_blocks_from_batch(self, batch, block_domains, node_domain, graph_of_node,
                                  mlm_mask, original, top1) -> None:
        missing = [name for name in ("pos_code", "block_idx", "block_symbols") if getattr(batch, name, None) is None]
        if missing:
            raise ValueError(f"block identity metrics require batch attributes: {', '.join(missing)}")
        num_nodes = original.shape[0]
        pos_codes = flatten_pos_codes(batch.pos_code, num_nodes)
        block_ids = _as_numpy(batch.block_idx, np.int64)
        symbols = _per_graph_block_symbols(batch.block_symbols)
        if block_ids.shape[0] != num_nodes:
            raise ValueError("block_idx must have one value per node")
        for domain in block_domains:
            spec = BLOCK_SPECS[domain]
            stats = self.blocks[domain]
            selected = np.where((node_domain == domain) & spec.identity_slot_mask(pos_codes))[0]
            if selected.size == 0:
                continue
            keys = np.stack((graph_of_node[selected], block_ids[selected]), axis=1)
            unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
            inverse = inverse.reshape(-1)
            masked_counts = np.bincount(inverse, weights=mlm_mask[selected], minlength=len(unique)).astype(np.int64)
            touched = masked_counts > 0
            irregular = touched & (counts != spec.slots)
            partial = touched & (counts == spec.slots) & (masked_counts < counts)
            stats.skipped["irregular_slot_count"] += int(irregular.sum())
            stats.skipped["partially_masked"] += int(partial.sum())
            full = np.where((counts == spec.slots) & (masked_counts == counts))[0]
            if full.size == 0:
                continue
            order = np.argsort(inverse, kind="stable")
            starts = np.concatenate(([0], np.cumsum(counts)))
            for block in full:
                nodes = selected[order[starts[block]:starts[block + 1]]]
                graph, block_id = int(unique[block, 0]), int(unique[block, 1])
                if graph >= len(symbols) or block_id >= len(symbols[graph]):
                    raise ValueError(f"graph {graph} block {block_id} has no block_symbols entry")
                stats.update_block(symbols[graph][block_id], original[nodes], top1[nodes])

    # -- results ----------------------------------------------------------

    def compute(self) -> Dict[str, dict]:
        results = {}
        for domain in DOMAIN_ORDER:
            atoms = self.atoms.get(domain)
            if atoms is None or atoms.count["all"] == 0:
                continue
            result = {
                "batches": self.batches[domain],
                "summary": atoms.summary(),
                "elements": atoms.element_table(),
                "confusion": atoms.confusion_table(),
                "calibration": atoms.calibration_table(),
            }
            if domain in self.blocks and (
                sum(self.blocks[domain].total.values()) or self.blocks[domain].skipped
            ):
                result["block_identity"] = self.blocks[domain].summary()
            distance = {
                head: stats.summary() for (d, head), stats in sorted(self.distance.items()) if d == domain
            }
            if distance:
                result["distance_visible_pairs_diagnostic"] = distance
            if domain in self.embeddings and self.embeddings[domain].size:
                result["embedding"] = self.embeddings[domain].summary()
            results[domain] = result
        return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def flatten_scalar_metrics(results: Dict[str, dict], stage: str) -> Dict[str, float]:
    """W&B keys ``<stage>_<domain>_mlm/<metric>``; undefined values are omitted."""
    flat = {}
    for domain, result in results.items():
        prefix = f"{stage}_{domain}_mlm"
        for name, value in result["summary"].items():
            if value is not None:
                flat[f"{prefix}/{name}"] = float(value)
        for row in result["elements"]:
            for name in ("recall", "precision", "support"):
                if row[name] is not None:
                    flat[f"{prefix}/element_{row['element']}/{name}"] = float(row[name])
        identity = result.get("block_identity")
        if identity is not None:
            kind = "residue" if domain == "protein" else "nucleotide"
            if identity["accuracy"] is not None:
                flat[f"{prefix}/{kind}_identity_acc"] = identity["accuracy"]
            flat[f"{prefix}/{kind}_identity_n"] = float(identity["n_blocks"])
            for name, row in identity["per_name"].items():
                if row["recall"] is not None:
                    flat[f"{prefix}/{kind}_{name}/recall"] = row["recall"]
        for head, stats in (result.get("distance_visible_pairs_diagnostic") or {}).items():
            for name, value in stats.items():
                if value is not None:
                    flat[f"{prefix}/distance_visible_pairs_diagnostic/{head}/{name}"] = float(value)
        for name, value in (result.get("embedding") or {}).items():
            if value is not None:
                flat[f"{prefix}/embedding/{name}"] = float(value)
    return flat


def _fmt(value, digits=4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{value:.{digits}f}"


def format_text_tables(results: Dict[str, dict], header: str) -> str:
    lines = ["=" * 96, header, "=" * 96]
    if not results:
        lines.append("(no masked atoms in any domain)")
    for domain, result in results.items():
        s = result["summary"]
        lines.append(f"\n[{domain}] masked={_fmt(s['n_masked'])} real={_fmt(s['n_masked_real'])} "
                     f"V={_fmt(s['n_masked_virtual'])} H={_fmt(s['n_masked_hydrogen'])} batches={result['batches']}")
        lines.append("  top-k acc       all      real     V")
        for k in TOP_K:
            lines.append(f"    top{k}       {_fmt(s[f'top{k}_acc_all']):>8} {_fmt(s[f'top{k}_acc_real']):>8} "
                         f"{_fmt(s[f'top{k}_acc_virtual']):>8}")
        lines.append(f"  real top1 without H: {_fmt(s['top1_acc_real_noH'])}")
        lines.append(f"  CE real/V/all: {_fmt(s['ce_real'])} / {_fmt(s['ce_virtual'])} / {_fmt(s['ce_all'])}   "
                     f"PPL real/V/all: {_fmt(s['ppl_real'], 3)} / {_fmt(s['ppl_virtual'], 3)} / {_fmt(s['ppl_all'], 3)}")
        lines.append(f"  ECE(15 bins) all/real: {_fmt(s['ece_all'])} / {_fmt(s['ece_real'])}")
        lines.append("  element   support  predicted   recall  precision")
        for row in result["elements"]:
            lines.append(f"    {row['element']:<6} {row['support']:>9,} {row['predicted']:>10,} "
                         f"{_fmt(row['recall']):>8} {_fmt(row['precision']):>10}")
        identity = result.get("block_identity")
        if identity is not None:
            kind = "residue (18-class, LEU|ILE and PRO|VAL merged)" if domain == "protein" else "nucleotide"
            lines.append(f"  {kind} identity from slot composition: acc={_fmt(identity['accuracy'])} "
                         f"n={_fmt(identity['n_blocks'])} skipped={identity['skipped']}")
            for name, row in identity["per_name"].items():
                lines.append(f"    {name:<5} support={row['support']:>8,} recall={_fmt(row['recall'])}")
        distance = result.get("distance_visible_pairs_diagnostic")
        if distance:
            lines.append("  distance heads (visible pairs, diagnostic only):")
            for head, d in distance.items():
                lines.append(f"    {head:<11} pairs={_fmt(d['n_pairs'])} bin_acc={_fmt(d['bin_acc'])} "
                             f"within1={_fmt(d['within1_bin_acc'])} MAE={_fmt(d['mae_A'], 3)}A "
                             f"MAE(bounded)={_fmt(d['mae_bounded_A'], 3)}A")
        embedding = result.get("embedding")
        if embedding and embedding.get("n_nodes"):
            lines.append(f"  embeddings: n={embedding['n_nodes']} dim={embedding['dim']} "
                         f"norm={_fmt(embedding['norm_mean'], 3)}+-{_fmt(embedding['norm_std'], 3)} "
                         f"eff_rank={_fmt(embedding['effective_rank'], 2)} "
                         f"mean_cos={_fmt(embedding['mean_pairwise_cosine'])}")
    lines.append("=" * 96)
    return "\n".join(lines)

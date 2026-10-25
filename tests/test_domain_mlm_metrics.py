"""Hand-computed checks for core/metrics/domain_mlm_metrics.py."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.metrics.domain_mlm_metrics import (
    BLOCK_SPECS,
    NO_MATCH,
    PROTEIN_SIDECHAIN_COMPOSITIONS,
    RNA_BASE_COMPOSITIONS,
    DomainMLMMetrics,
    MaskedAtomStats,
    composition_classes,
    flatten_scalar_metrics,
    format_text_tables,
)
from core.metrics.domain_mlm_metrics_callback import enabled_config_domains
from core.pretraining_config import create_pretraining_config
from core.pretraining_model import PretrainingESAModel
from core.pretraining_tasks import PretrainingTasks
from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS,
    PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE,
    RNA_BACKBONE_ATOMS,
    RNA_BASE_ATOMS_BY_NUCLEOTIDE,
)

V = 120


def _topk(*rows):
    return np.array(rows, dtype=np.int64)


def test_topk_ce_ece_and_element_tables_by_hand():
    stats = MaskedAtomStats()
    stats.update(
        targets=[6, 7, V, 1],
        topk=_topk([6, 7, 8, 16, V], [6, 8, 7, 16, V], [6, 7, 8, 16, V], [6, 8, 7, 1, 16]),
        target_log_prob=[-0.1, -1.0, -2.0, -3.0],
        top1_prob=[0.9, 0.5, 0.6, 0.25],
    )
    s = stats.summary()
    assert (s["n_masked"], s["n_masked_real"], s["n_masked_virtual"], s["n_masked_hydrogen"]) == (4, 3, 1, 1)
    assert s["top1_acc_all"] == pytest.approx(1 / 4)
    assert s["top3_acc_all"] == pytest.approx(2 / 4)
    assert s["top5_acc_all"] == pytest.approx(1.0)
    assert s["top1_acc_real"] == pytest.approx(1 / 3)
    assert s["top3_acc_real"] == pytest.approx(2 / 3)
    assert s["top5_acc_real"] == pytest.approx(1.0)
    assert (s["top1_acc_virtual"], s["top3_acc_virtual"], s["top5_acc_virtual"]) == (0.0, 0.0, 1.0)
    assert s["top1_acc_real_noH"] == pytest.approx(1 / 2)
    assert s["ce_real"] == pytest.approx((0.1 + 1.0 + 3.0) / 3)
    assert s["ce_virtual"] == pytest.approx(2.0)
    assert s["ce_all"] == pytest.approx(6.1 / 4)
    assert s["ppl_real"] == pytest.approx(math.exp((0.1 + 1.0 + 3.0) / 3))
    assert s["ppl_virtual"] == pytest.approx(math.exp(2.0))
    # Every atom sits alone in its confidence bin: ECE = mean |correct - confidence|.
    assert s["ece_all"] == pytest.approx((0.1 + 0.5 + 0.6 + 0.25) / 4)
    assert s["ece_real"] == pytest.approx((0.1 + 0.5 + 0.25) / 3)

    rows = {row["element"]: row for row in stats.element_table()}
    assert set(rows) == {"H", "C", "N", "V"}
    assert rows["C"] == {"class": 6, "element": "C", "support": 1, "predicted": 4, "recall": 1.0, "precision": 0.25}
    assert rows["N"]["recall"] == 0.0 and rows["N"]["precision"] is None
    assert rows["V"]["support"] == 1 and rows["V"]["predicted"] == 0
    confusion = stats.confusion_table()
    assert confusion["labels"] == ["H", "C", "N", "V"]
    assert confusion["matrix"] == [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]


def test_precision_is_over_the_predicted_element():
    stats = MaskedAtomStats()
    top1 = [6, 7, 7, 7]
    stats.update([6, 6, 7, 8], _topk(*[[p, 0, 0, 0, 0] for p in top1]), [-0.5] * 4, [0.5] * 4)
    rows = {row["element"]: row for row in stats.element_table()}
    assert (rows["C"]["recall"], rows["C"]["precision"]) == (0.5, 1.0)
    assert rows["N"]["recall"] == 1.0
    assert rows["N"]["precision"] == pytest.approx(1 / 3)
    assert rows["O"]["recall"] == 0.0 and rows["O"]["precision"] is None


def test_ece_shared_bin_and_bin_table():
    stats = MaskedAtomStats()
    stats.update([6, 6], _topk([6, 0, 0, 0, 0], [7, 0, 0, 0, 0]), [-0.3, -2.0], [0.70, 0.72])
    s = stats.summary()
    assert s["ece_all"] == pytest.approx(abs(0.5 - 0.71))
    filled = [row for row in stats.calibration_table() if row["group"] == "all" and row["count"]]
    assert len(filled) == 1
    assert filled[0]["bin"] == 11 and filled[0]["count"] == 2
    assert filled[0]["mean_confidence"] == pytest.approx(0.71)
    assert filled[0]["accuracy"] == pytest.approx(0.5)


def test_undefined_groups_are_none_and_bad_inputs_fail():
    stats = MaskedAtomStats()
    stats.update([6], _topk([6, 0, 0, 0, 0]), [-0.1], [0.9])
    s = stats.summary()
    assert s["top1_acc_virtual"] is None and s["ce_virtual"] is None and s["ppl_virtual"] is None
    with pytest.raises(ValueError, match="columns"):
        stats.update([6], _topk([6, 0, 0]), [-0.1], [0.9])
    with pytest.raises(ValueError, match="outside"):
        stats.update([121], _topk([6, 0, 0, 0, 0]), [-0.1], [0.9])
    with pytest.raises(ValueError, match="log-probabilities"):
        stats.update([6], _topk([6, 0, 0, 0, 0]), [0.5], [0.9])
    with pytest.raises(ValueError, match="top1_prob"):
        stats.update([6], _topk([6, 0, 0, 0, 0]), [-0.1], [1.5])


def test_composition_tables_have_exactly_the_two_protein_collisions():
    to_class, name_to_class = composition_classes(PROTEIN_SIDECHAIN_COMPOSITIONS)
    merged = sorted(cls for cls in set(name_to_class.values()) if "|" in cls)
    assert merged == ["ILE|LEU", "PRO|VAL"]
    assert len(set(name_to_class.values())) == 18
    assert set(PROTEIN_SIDECHAIN_COMPOSITIONS) == set(PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE)
    assert PROTEIN_SIDECHAIN_COMPOSITIONS["ALA"] == ((6, 1), (V, 9))
    assert PROTEIN_SIDECHAIN_COMPOSITIONS["GLY"] == ((V, 10),)
    assert PROTEIN_SIDECHAIN_COMPOSITIONS["TRP"] == ((6, 9), (7, 1))
    assert PROTEIN_SIDECHAIN_COMPOSITIONS["CYS"] == ((6, 1), (16, 1), (V, 8))
    for name, composition in PROTEIN_SIDECHAIN_COMPOSITIONS.items():
        assert sum(count for _, count in composition) == 10, name


def test_rna_base_compositions_are_distinct():
    _, name_to_class = composition_classes(RNA_BASE_COMPOSITIONS)
    assert sorted(name_to_class.values()) == ["A", "C", "G", "U"]
    assert RNA_BASE_COMPOSITIONS["A"] == ((6, 5), (7, 5), (V, 1))
    assert RNA_BASE_COMPOSITIONS["G"] == ((6, 5), (7, 5), (8, 1))
    assert RNA_BASE_COMPOSITIONS["C"] == ((6, 4), (7, 3), (8, 1), (V, 3))
    assert RNA_BASE_COMPOSITIONS["U"] == ((6, 4), (7, 2), (8, 2), (V, 3))
    assert set(RNA_BASE_COMPOSITIONS) == set(RNA_BASE_ATOMS_BY_NUCLEOTIDE)


def _slots(residue):
    return [int(dict(C=6, N=7, O=8, S=16)[a[0]]) for a in PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE[residue]] + \
        [V] * (10 - len(PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE[residue]))


def test_residue_identity_merges_collisions_and_counts_skips():
    metrics = DomainMLMMetrics()
    metrics.update_block("protein", "LEU", _slots("LEU"), _slots("ILE"))   # merged class: correct
    metrics.update_block("protein", "VAL", _slots("VAL"), _slots("ALA"))   # wrong class
    metrics.update_block("protein", "SER", _slots("SER"), [1] + _slots("SER")[1:])  # no residue: NO_MATCH
    metrics.update_block("protein", "ala", _slots("ALA"), list(reversed(_slots("ALA"))))  # order-free
    metrics.update_block("protein", "MSE", _slots("MET"), _slots("MET"))   # non-standard symbol
    metrics.update_block("protein", "GLY", _slots("ALA"), _slots("GLY"))   # target != canonical
    summary = metrics.blocks["protein"].summary()
    assert summary["n_blocks"] == 4
    assert summary["accuracy"] == pytest.approx(2 / 4)
    assert summary["n_classes"] == 18
    assert summary["skipped"] == {"nonstandard_symbol": 1, "target_composition_mismatch": 1}
    assert summary["per_name"]["LEU"] == {"support": 1, "recall": 1.0}
    assert summary["per_name"]["VAL"]["recall"] == 0.0
    assert summary["per_class"]["ILE|LEU"]["recall"] == 1.0
    confusion = {(r["true_class"], r["predicted_class"]): r["count"] for r in summary["class_confusion"]}
    assert confusion[("SER", NO_MATCH)] == 1 and confusion[("PRO|VAL", "ALA")] == 1


def test_rna_nucleotide_identity_with_aliases():
    metrics = DomainMLMMetrics()
    a = [6] * 5 + [7] * 5 + [V]
    g = [6] * 5 + [7] * 5 + [8]
    metrics.update_block("rna", "RA", a, a)
    metrics.update_block("rna", "G", g, a)
    metrics.update_block("rna", "PSU", a, a)
    summary = metrics.blocks["rna"].summary()
    assert summary["n_blocks"] == 2 and summary["accuracy"] == 0.5
    assert summary["skipped"] == {"nonstandard_symbol": 1}


def _protein_block(residue):
    codes = list(PROTEIN_BACKBONE_ATOMS) + list(PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE[residue])
    codes += [f"V{i}" for i in range(len(codes) + 1, 15)]
    z = [7, 6, 6, 8] + _slots(residue)
    return codes, z


def _batch_and_outputs():
    """Graph 0: protein ALA (fully masked), LEU (fully masked), SER (backbone only). Graph 1: unimol."""
    codes, z, blocks = [], [], []
    for block, residue in enumerate(("ALA", "LEU", "SER")):
        c, zz = _protein_block(residue)
        codes += c
        z += zz
        blocks += [block] * 14
    unimol_z = [6, 8, 1, 1]
    original = torch.tensor(z + unimol_z)
    mask = torch.zeros(original.numel(), dtype=torch.bool)
    mask[:28] = True                      # ALA and LEU blocks
    mask[28:32] = True                    # SER backbone only: no sidechain slot masked, not scored
    mask[42 + 0] = True                   # unimol C
    mask[42 + 2] = True                   # unimol H
    batch = SimpleNamespace(
        dataset_type=["protein", "unimol"], num_graphs=2,
        batch=torch.tensor([0] * 42 + [1] * 4), mlm_mask=mask, original_types=original,
        pos_code=[codes, ["C", "O", "H", "H"]], block_idx=torch.tensor(blocks + [0] * 4),
        block_symbols=[["ALA", "LEU", "SER"], ["mol"]],
    )
    node_index = torch.where(mask)[0]
    targets = original[node_index]
    top1 = targets.clone()
    top1[14 + 4] = V                      # LEU first sidechain C predicted V -> LEU composition broken
    topk = torch.stack([top1, *(torch.zeros_like(top1) for _ in range(4))], dim=1)
    outputs = {"mlm_eval": {
        "node_index": node_index, "target": targets, "topk": topk,
        "target_log_prob": torch.full((node_index.numel(),), -0.2),
        "top1_prob": torch.full((node_index.numel(),), 0.8),
    }}
    return batch, outputs


def test_update_from_batch_splits_domains_and_scores_masked_residues():
    metrics = DomainMLMMetrics()
    batch, outputs = _batch_and_outputs()
    metrics.update_from_batch(batch, outputs)
    results = metrics.compute()
    assert list(results) == ["protein", "unimol"]
    assert results["protein"]["summary"]["n_masked"] == 32
    assert results["protein"]["summary"]["top1_acc_all"] == pytest.approx(31 / 32)
    assert results["unimol"]["summary"]["n_masked"] == 2
    assert results["unimol"]["summary"]["n_masked_hydrogen"] == 1
    identity = results["protein"]["block_identity"]
    assert identity["n_blocks"] == 2
    assert identity["per_name"]["ALA"]["recall"] == 1.0
    assert identity["per_name"]["LEU"]["recall"] == 0.0
    assert "block_identity" not in results["unimol"]
    flat = flatten_scalar_metrics(results, "val")
    assert flat["val_protein_mlm/residue_identity_acc"] == 0.5
    assert flat["val_unimol_mlm/top1_acc_real_noH"] == 1.0
    assert "val_unimol_mlm/top1_acc_virtual" not in flat  # undefined -> omitted
    text = format_text_tables(results, "T")
    assert "[protein]" in text and "residue (18-class" in text and "[unimol]" in text


def test_update_from_batch_fails_loud_on_contract_violations():
    batch, outputs = _batch_and_outputs()
    metrics = DomainMLMMetrics()
    with pytest.raises(ValueError, match="mlm_eval"):
        metrics.update_from_batch(batch, {"loss": 0.0})
    bad = dict(outputs["mlm_eval"], node_index=outputs["mlm_eval"]["node_index"][1:])
    with pytest.raises(ValueError, match="node_index"):
        metrics.update_from_batch(batch, {"mlm_eval": bad})
    batch.dataset_type = ["protein", "dna"]
    with pytest.raises(ValueError, match="unknown dataset_type"):
        metrics.update_from_batch(batch, outputs)
    batch, outputs = _batch_and_outputs()
    del batch.block_symbols
    with pytest.raises(ValueError, match="block_symbols"):
        metrics.update_from_batch(batch, outputs)


def test_distance_head_stats_by_hand():
    metrics = DomainMLMMetrics()
    metrics.update_distance("rna", "long_range", [0, 1, 3], [0, 2, 3], [0.5, 2.0, 7.0], [0.4, 2.6, 9.0], 4)
    d = metrics.distance[("rna", "long_range")].summary()
    assert d["n_pairs"] == 3
    assert d["bin_acc"] == pytest.approx(2 / 3)
    assert d["within1_bin_acc"] == 1.0
    assert d["mae_A"] == pytest.approx((0.1 + 0.6 + 2.0) / 3)
    assert d["n_bounded_pairs"] == 2 and d["mae_bounded_A"] == pytest.approx(0.35)
    with pytest.raises(ValueError, match="bin count"):
        metrics.update_distance("rna", "long_range", [0], [0], [0.5], [0.5], 5)


def test_embedding_health_by_hand():
    metrics = DomainMLMMetrics(embedding_max_nodes_per_domain=3)
    metrics.update_embeddings("coconut", [[1.0, 0.0], [0.0, 1.0]])
    metrics.update_embeddings("coconut", [[2.0, 0.0], [5.0, 5.0]])  # only one more node fits
    e = metrics.embeddings["coconut"].summary()
    assert e["n_nodes"] == 3 and e["dim"] == 2
    assert e["norm_mean"] == pytest.approx(4 / 3)
    assert e["norm_std"] == pytest.approx(np.std([1.0, 1.0, 2.0]))
    p = np.array([math.sqrt(5), 1.0]) / (math.sqrt(5) + 1)
    assert e["effective_rank"] == pytest.approx(math.exp(-(p * np.log(p)).sum()))
    assert e["mean_pairwise_cosine"] == pytest.approx(1 / 3)


def _tasks(**overrides):
    values = dict(graph_dim=16, distance_bins=4, distance_bin_boundaries=[1.0, 2.0, 4.0], max_distance=8.0,
                  short_range_distance_bins=3, short_range_bin_boundaries=[1.0, 2.0], short_range_max_distance=3.0)
    values.update(overrides)
    return PretrainingTasks(create_pretraining_config(**values))


def test_distance_bin_centers():
    tasks = _tasks()
    assert tasks.distance_bin_centers("long_range").tolist() == [0.5, 1.5, 3.0, 4.0]   # open last bin: lower edge
    assert tasks.distance_bin_centers("short_range").tolist() == [0.5, 1.5, 2.5]       # closed at 3.0
    uniform = _tasks(distance_bin_boundaries=None, short_range_bin_boundaries=None)
    assert uniform.distance_bin_centers("long_range").tolist() == [1.0, 3.0, 5.0, 6.0]
    assert uniform.distance_bin_centers("short_range").tolist() == [0.5, 1.5, 2.5]


def test_distance_capture_is_eval_only_and_leaves_losses_unchanged():
    tasks = _tasks()
    emb = torch.randn(6, 16)
    data = SimpleNamespace(pos=torch.randn(6, 3) * 2, batch=torch.tensor([0, 0, 0, 1, 1, 1]))
    edges = torch.tensor([[0, 1, 3], [1, 2, 4]])
    distances = torch.tensor([0.5, 1.7, 2.9])
    mask = torch.ones(3, dtype=torch.bool)

    tasks.eval()
    reference = (tasks.long_range_distance_loss(emb, data), tasks.short_range_distance_loss(emb, edges, distances, mask))
    assert tasks.pop_distance_diagnostics() == {}
    tasks.capture_distance_diagnostics = True
    captured = (tasks.long_range_distance_loss(emb, data), tasks.short_range_distance_loss(emb, edges, distances, mask))
    assert all(torch.equal(a, b) for a, b in zip(reference, captured))
    records = tasks.pop_distance_diagnostics()
    assert set(records) == {"long_range", "short_range"}
    assert records["long_range"]["node_index"].numel() == 6        # 3 + 3 within-graph pairs
    assert records["short_range"]["true_bin"].tolist() == [0, 1, 2]
    assert records["short_range"]["num_bins"] == 3
    tasks.short_range_distance_loss(emb, edges, distances, mask, scope="protein")
    assert tasks.pop_distance_diagnostics() == {}                  # per-type calls are not recorded
    tasks.train()
    tasks.short_range_distance_loss(emb, edges, distances, mask)
    assert tasks.pop_distance_diagnostics() == {}                  # never in training mode
    tasks.eval()
    tasks.short_range_distance_loss(emb, edges, distances, mask)
    with pytest.raises(RuntimeError, match="twice"):
        tasks.short_range_distance_loss(emb, edges, distances, mask)


def test_masked_eval_outputs_put_the_argmax_first_on_ties():
    logits = torch.zeros(3, 121)
    logits[0, 7] = logits[0, 6] = 5.0          # tie: argmax returns 6
    logits[1, 120] = 3.0
    logits[2, 8] = 1.0
    batch = SimpleNamespace(mlm_mask=torch.tensor([True, False, True]), original_types=torch.tensor([7, 6, 120]))
    out = PretrainingESAModel._masked_mlm_eval_outputs(batch, logits)
    assert out["node_index"].tolist() == [0, 2]
    assert out["target"].tolist() == [7, 120]
    assert out["topk"][:, 0].tolist() == [int(logits[0].argmax()), 8]
    assert sorted(out["topk"][0, :2].tolist()) == [6, 7]
    assert len(set(out["topk"][0].tolist())) == 5
    log_probs = torch.log_softmax(logits, dim=-1)
    assert torch.allclose(out["target_log_prob"], torch.stack([log_probs[0, 7], log_probs[2, 120]]))
    assert torch.allclose(out["top1_prob"], torch.stack([log_probs[0].max(), log_probs[2].max()]).exp())


def test_enabled_config_domains_maps_config_keys():
    config = SimpleNamespace(multi_domain={"enabled": True, "datasets": {
        "pdb": {"enabled": True}, "rna": {"enabled": False}, "unimol": {"enabled": True},
        "metabolite": {"enabled": True}}})
    assert enabled_config_domains(config) == ["protein", "unimol", "coconut"]
    assert BLOCK_SPECS["protein"].slots == 10 and BLOCK_SPECS["rna"].slots == 11
    assert len(RNA_BACKBONE_ATOMS) == 12

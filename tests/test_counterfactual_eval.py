"""Contract tests of the offline counterfactual evaluation modes.

The modes corrupt what the encoder/ESA see of the masked UniMol atoms, only in
evaluation, to measure where the masked-atom accuracy falls once geometry or
neighbourhood information is removed. They never touch protein/RNA graphs, the
clean ``batch.pos`` that the distance losses use, or the training path.

Small synthetic inputs; the end-to-end tests reuse the tiny four-domain CPU
split of ``test_evaluate_pretrain``. The module under test is imported inside
each test so that a missing implementation fails every test individually.
"""

import copy
import json
import math
import os

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

import test_evaluate_pretrain as tev
from core import evaluate_pretrain as ev
from core.pretraining_model import PretrainingESAModel

ALL_MODES = (
    "masked_pos_noise",
    "masked_pos_random",
    "masked_dist_shuffle",
    "esa_edges_remove",
    "esa_edges_shuffle",
    "masked_edges_remove",
)


def cfe():
    import core.counterfactual_eval as module
    return module


# ---------------------------------------------------------------------------
# Synthetic inputs (unit level)
# ---------------------------------------------------------------------------


def _radius_edges(pos, graph_of, cutoff=3.0):
    src, dst = torch.meshgrid(torch.arange(pos.size(0)), torch.arange(pos.size(0)), indexing="ij")
    keep = (src != dst) & (graph_of[src] == graph_of[dst]) & ((pos[src] - pos[dst]).norm(dim=-1) < cutoff)
    return torch.stack((src[keep], dst[keep]))


def _inputs(sizes=(9, 12, 10), domains=("unimol", "unimol", "protein"), seed=0, masked=((1, 5), (2,), (0,))):
    """Packed nodes of several graphs; ``masked`` lists local masked indices per graph."""
    gen = torch.Generator().manual_seed(seed)
    batch_mapping = torch.cat([torch.full((n,), g) for g, n in enumerate(sizes)])
    pos = torch.randn(batch_mapping.numel(), 3, generator=gen) * 1.6
    edge_index = _radius_edges(pos, batch_mapping)
    mlm_mask = torch.zeros(batch_mapping.numel(), dtype=torch.bool)
    starts = torch.cumsum(torch.tensor([0, *sizes]), 0)
    for g, local in enumerate(masked):
        for i in local:
            mlm_mask[starts[g] + i] = True
    return {
        "pos": pos, "edge_index": edge_index, "attention_edge_index": edge_index.clone(),
        "batch_mapping": batch_mapping, "mlm_mask": mlm_mask,
        "graph_domains": list(domains), "eval_mask_seeds": [101 + g for g in range(len(sizes))],
    }


def _apply(mode, sigma=None, **overrides):
    inputs = _inputs()
    inputs.update(overrides)
    settings = cfe().build_counterfactual_settings(mode, sigma)
    return settings, inputs, cfe().apply_counterfactual(settings, **inputs)


def _pairs(edge_index):
    return {(int(a), int(b)) for a, b in edge_index.t().tolist()}


def _incident(edge_index, nodes):
    keep = nodes[edge_index[0]] | nodes[edge_index[1]]
    return edge_index[:, keep]


def _unimol_node_mask(inputs):
    return torch.tensor([d == "unimol" for d in inputs["graph_domains"]])[inputs["batch_mapping"]]


# ---------------------------------------------------------------------------
# Mode names and settings fail loud
# ---------------------------------------------------------------------------


def test_mode_catalog_matches_the_documented_modes():
    assert tuple(cfe().COUNTERFACTUAL_MODES) == ("none", *ALL_MODES)


@pytest.mark.parametrize("name", ["bogus", "", "Masked_Pos_Noise", "masked-pos-noise", "anchor_remove"])
def test_unknown_mode_name_fails_loudly(name):
    with pytest.raises(ValueError, match="unknown counterfactual mode") as info:
        cfe().build_counterfactual_settings(name, None)
    assert "masked_pos_noise" in str(info.value)  # the message lists the valid modes


def test_none_means_no_settings_and_rejects_a_sigma():
    assert cfe().build_counterfactual_settings("none", None) is None
    with pytest.raises(ValueError, match="sigma"):
        cfe().build_counterfactual_settings("none", 1.0)


def test_sigma_contract():
    with pytest.raises(ValueError, match="sigma"):
        cfe().build_counterfactual_settings("masked_pos_noise", None)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="sigma"):
            cfe().build_counterfactual_settings("masked_pos_noise", bad)
    for mode in ALL_MODES[1:]:
        with pytest.raises(ValueError, match="sigma"):
            cfe().build_counterfactual_settings(mode, 1.0)
    assert cfe().build_counterfactual_settings("masked_pos_noise", 0.5).sigma == 0.5


def test_only_unimol_is_a_supported_target_domain():
    for domains in (("protein",), ("rna",), ("coconut",), ("unimol", "protein"), ()):
        with pytest.raises(ValueError, match="domain"):
            cfe().build_counterfactual_settings("esa_edges_remove", None, domains=domains)


# ---------------------------------------------------------------------------
# Input validation of apply_counterfactual
# ---------------------------------------------------------------------------


def test_apply_rejects_malformed_inputs():
    settings = cfe().build_counterfactual_settings("esa_edges_remove", None)
    good = _inputs()

    def call(**changes):
        inputs = dict(good)
        inputs.update(changes)
        return cfe().apply_counterfactual(settings, **inputs)

    call()  # the unmodified inputs are valid
    with pytest.raises(ValueError, match="eval_mask_seeds"):
        call(eval_mask_seeds=[1, 2])
    with pytest.raises(ValueError, match="graph_domains"):
        call(graph_domains=["unimol", "unimol"])
    with pytest.raises(ValueError, match="sorted"):
        call(batch_mapping=good["batch_mapping"].flip(0))
    with pytest.raises(ValueError, match="mlm_mask"):
        call(mlm_mask=good["mlm_mask"].long())
    with pytest.raises(ValueError, match="mlm_mask"):
        call(mlm_mask=good["mlm_mask"][:-1])
    with pytest.raises(ValueError, match="edge"):
        call(attention_edge_index=torch.tensor([[0], [good["pos"].size(0)]]))
    cross = torch.tensor([[0], [int(good["batch_mapping"].numel()) - 1]])
    with pytest.raises(ValueError, match="cross-graph"):
        call(attention_edge_index=cross)
    with pytest.raises(ValueError, match="pos"):
        call(pos=good["pos"][:, :2])


# ---------------------------------------------------------------------------
# Position modes
# ---------------------------------------------------------------------------


def test_pos_noise_moves_only_masked_unimol_atoms_and_leaves_edges_alone():
    sigma = 1.5
    settings, inputs, out = _apply("masked_pos_noise", sigma)
    clean = inputs["pos"]
    moved = (out.pos != clean).any(dim=1)
    target = inputs["mlm_mask"] & _unimol_node_mask(inputs)
    assert torch.equal(moved, target)
    assert torch.equal(out.edge_index, inputs["edge_index"])
    assert torch.equal(out.attention_edge_index, inputs["attention_edge_index"])
    assert out.edge_distances is None
    assert not clean.requires_grad and torch.equal(clean, _inputs()["pos"])  # input not mutated


def test_pos_noise_has_the_requested_scale():
    sizes = (200,) * 6
    inputs = _inputs(sizes=sizes, domains=("unimol",) * 6, masked=(tuple(range(150)),) * 6)
    sigma = 0.7
    out = cfe().apply_counterfactual(cfe().build_counterfactual_settings("masked_pos_noise", sigma), **inputs)
    delta = (out.pos - inputs["pos"])[inputs["mlm_mask"]]
    assert delta.numel() == 6 * 150 * 3
    assert abs(float(delta.mean())) < 0.05
    assert float(delta.std()) == pytest.approx(sigma, rel=0.08)


def test_pos_random_replaces_masked_unimol_atoms_only():
    settings, inputs, out = _apply("masked_pos_random")
    moved = (out.pos != inputs["pos"]).any(dim=1)
    target = inputs["mlm_mask"] & _unimol_node_mask(inputs)
    assert torch.equal(moved, target)
    assert torch.equal(out.edge_index, inputs["edge_index"])
    assert torch.isfinite(out.pos).all()
    # the replacement does not reuse the clean position of the atom it replaces
    assert (out.pos[target] - inputs["pos"][target]).norm(dim=1).min() > 0


@pytest.mark.parametrize("mode,sigma", [("masked_pos_noise", 1.0), ("masked_pos_random", None)])
def test_protein_graph_is_untouched_by_position_modes(mode, sigma):
    _, inputs, out = _apply(mode, sigma)
    protein = inputs["batch_mapping"] == 2
    assert inputs["mlm_mask"][protein].any()  # the protein graph does carry masked nodes
    assert torch.equal(out.pos[protein], inputs["pos"][protein])


# ---------------------------------------------------------------------------
# Distance/RBF mode
# ---------------------------------------------------------------------------


def test_dist_shuffle_permutes_distances_of_masked_atoms_symmetrically():
    inputs = _inputs(masked=((1,), (2,), (0,)))
    settings = cfe().build_counterfactual_settings("masked_dist_shuffle", None)
    out = cfe().apply_counterfactual(settings, **inputs)
    edge_index = inputs["edge_index"]
    clean = (inputs["pos"][edge_index[0]] - inputs["pos"][edge_index[1]]).norm(dim=1)
    assert torch.equal(out.edge_index, edge_index)
    assert torch.equal(out.attention_edge_index, inputs["attention_edge_index"])
    assert out.pos is inputs["pos"] or torch.equal(out.pos, inputs["pos"])
    assert out.edge_distances.shape == clean.shape and torch.isfinite(out.edge_distances).all()

    unimol_masked = inputs["mlm_mask"] & _unimol_node_mask(inputs)
    incident = unimol_masked[edge_index[0]] | unimol_masked[edge_index[1]]
    # every distance away from the masked atoms is the clean one, including the protein graph
    assert torch.equal(out.edge_distances[~incident], clean[~incident])
    assert incident.any()
    # symmetric: an edge and its reverse carry the same value
    lookup = {(int(a), int(b)): float(d) for (a, b), d in zip(edge_index.t().tolist(), out.edge_distances)}
    for (a, b), value in lookup.items():
        assert lookup[(b, a)] == value
    # each masked atom keeps the multiset of its incident distances, in another order
    changed = 0
    for node in torch.where(unimol_masked)[0].tolist():
        own = (edge_index[0] == node)
        assert torch.equal(torch.sort(out.edge_distances[own]).values, torch.sort(clean[own]).values)
        changed += int((out.edge_distances[own] != clean[own]).any())
    assert changed >= 1


def _assert_dist_shuffle_contract(inputs, out):
    edge_index = inputs["edge_index"]
    clean = (inputs["pos"][edge_index[0]] - inputs["pos"][edge_index[1]]).norm(dim=1)
    masked = inputs["mlm_mask"] & _unimol_node_mask(inputs)
    for node in torch.where(masked)[0].tolist():
        own = edge_index[0] == node
        assert torch.equal(torch.sort(out.edge_distances[own]).values, torch.sort(clean[own]).values), node
    lookup = {(int(a), int(b)): float(d) for (a, b), d in zip(edge_index.t().tolist(), out.edge_distances)}
    for a, b in lookup:
        if not bool(masked[a]) and bool(masked[b]):
            assert lookup[(a, b)] == lookup[(b, a)]  # a visible atom sees the masked atom's drawn value
    untouched = ~(masked[edge_index[0]] | masked[edge_index[1]])
    assert torch.equal(out.edge_distances[untouched], clean[untouched])


def test_dist_shuffle_keeps_the_distance_multiset_of_two_adjacent_masked_atoms():
    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [3., 0., 0.]])
    edges = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 1, 3], [1, 0, 2, 0, 3, 0, 2, 1, 3, 1]])
    inputs = {"pos": pos, "edge_index": edges, "attention_edge_index": edges.clone(),
              "batch_mapping": torch.zeros(4, dtype=torch.long),
              "mlm_mask": torch.tensor([True, True, False, False]), "graph_domains": ["unimol"]}
    changed = torch.zeros(2, dtype=torch.bool)
    clean = (pos[edges[0]] - pos[edges[1]]).norm(dim=1)
    for seed in range(1, 33):
        inputs["eval_mask_seeds"] = [seed]
        out = cfe().apply_counterfactual(cfe().build_counterfactual_settings("masked_dist_shuffle", None), **inputs)
        _assert_dist_shuffle_contract(inputs, out)
        for node in (0, 1):
            own = edges[0] == node
            changed[node] |= bool((out.edge_distances[own] != clean[own]).any())
    assert changed.all()


def test_dist_shuffle_contract_holds_under_dense_masking():
    inputs = _inputs(masked=((0, 1, 2, 3, 5, 7), (2, 3, 4, 8, 9), (0,)))
    out = cfe().apply_counterfactual(cfe().build_counterfactual_settings("masked_dist_shuffle", None), **inputs)
    masked = inputs["mlm_mask"] & _unimol_node_mask(inputs)
    edge_index = inputs["edge_index"]
    assert (masked[edge_index[0]] & masked[edge_index[1]]).any()  # masked atoms share edges
    _assert_dist_shuffle_contract(inputs, out)


def test_dist_shuffle_has_no_effect_on_protein_graphs():
    inputs = _inputs()
    out = cfe().apply_counterfactual(cfe().build_counterfactual_settings("masked_dist_shuffle", None), **inputs)
    protein_edges = inputs["batch_mapping"][inputs["edge_index"][0]] == 2
    clean = (inputs["pos"][inputs["edge_index"][0]] - inputs["pos"][inputs["edge_index"][1]]).norm(dim=1)
    assert torch.equal(out.edge_distances[protein_edges], clean[protein_edges])


# ---------------------------------------------------------------------------
# ESA edge / anchor modes
# ---------------------------------------------------------------------------


def test_esa_edges_remove_drops_every_attention_edge_of_masked_unimol_atoms():
    settings, inputs, out = _apply("esa_edges_remove")
    target = inputs["mlm_mask"] & _unimol_node_mask(inputs)
    attention = inputs["attention_edge_index"]
    assert _incident(attention, target).size(1) > 0
    assert _incident(out.attention_edge_index, target).size(1) == 0
    kept = _pairs(attention) - _pairs(_incident(attention, target))
    assert _pairs(out.attention_edge_index) == kept
    # the geometry the encoder sees is untouched
    assert torch.equal(out.edge_index, inputs["edge_index"]) and torch.equal(out.pos, inputs["pos"])
    assert out.edge_distances is None
    # the protein graph keeps even its edges at masked nodes
    protein = inputs["batch_mapping"] == 2
    protein_edges = {p for p in _pairs(attention) if protein[p[0]]}
    assert protein_edges <= _pairs(out.attention_edge_index)


def test_esa_edges_shuffle_rewires_masked_atoms_inside_their_graph():
    settings, inputs, out = _apply("esa_edges_shuffle")
    target = inputs["mlm_mask"] & _unimol_node_mask(inputs)
    attention = inputs["attention_edge_index"]
    new = out.attention_edge_index
    graph_of = inputs["batch_mapping"]
    assert bool((graph_of[new[0]] == graph_of[new[1]]).all())
    assert bool((new[0] != new[1]).all())
    # symmetric result
    assert {(b, a) for a, b in _pairs(new)} == _pairs(new)
    # edges that touch no masked unimol atom are unchanged
    untouched = _pairs(attention) - _pairs(_incident(attention, target))
    assert untouched <= _pairs(new)
    # the masked atoms were rewired, not merely kept
    assert _pairs(_incident(new, target)) != _pairs(_incident(attention, target))
    # degree of a masked atom is preserved (at least as many partners as before)
    for node in torch.where(target)[0].tolist():
        assert int((new[0] == node).sum()) >= int((attention[0] == node).sum())
    assert torch.equal(out.edge_index, inputs["edge_index"]) and torch.equal(out.pos, inputs["pos"])
    protein = graph_of == 2
    assert {p for p in _pairs(attention) if protein[p[0]]} == {p for p in _pairs(new) if protein[p[0]]}


def test_masked_edges_remove_cuts_encoder_and_attention_edges():
    settings, inputs, out = _apply("masked_edges_remove")
    target = inputs["mlm_mask"] & _unimol_node_mask(inputs)
    assert _incident(out.edge_index, target).size(1) == 0
    assert _incident(out.attention_edge_index, target).size(1) == 0
    assert _pairs(out.edge_index) == _pairs(inputs["edge_index"]) - _pairs(_incident(inputs["edge_index"], target))
    assert torch.equal(out.pos, inputs["pos"])


def test_attention_and_encoder_edges_are_edited_independently():
    # the graph-view path passes a different tensor for the ESA edges than for the encoder
    inputs = _inputs()
    inputs["attention_edge_index"] = inputs["edge_index"][:, ::2].contiguous()
    out = cfe().apply_counterfactual(cfe().build_counterfactual_settings("masked_edges_remove", None), **inputs)
    target = inputs["mlm_mask"] & _unimol_node_mask(inputs)
    assert _pairs(out.attention_edge_index) == (
        _pairs(inputs["attention_edge_index"]) - _pairs(_incident(inputs["attention_edge_index"], target)))
    assert _pairs(out.edge_index) == _pairs(inputs["edge_index"]) - _pairs(_incident(inputs["edge_index"], target))


# ---------------------------------------------------------------------------
# Reproducibility and independence from batch composition
# ---------------------------------------------------------------------------


def _graph_slice(inputs, g):
    """Single-graph inputs (local node ids) of graph ``g``."""
    node = inputs["batch_mapping"] == g
    first = int(torch.where(node)[0][0])
    out = {
        "pos": inputs["pos"][node], "batch_mapping": torch.zeros(int(node.sum()), dtype=torch.long),
        "mlm_mask": inputs["mlm_mask"][node], "graph_domains": [inputs["graph_domains"][g]],
        "eval_mask_seeds": [inputs["eval_mask_seeds"][g]],
    }
    for key in ("edge_index", "attention_edge_index"):
        edges = inputs[key]
        keep = node[edges[0]]
        out[key] = edges[:, keep] - first
    return out


@pytest.mark.parametrize("mode,sigma", [(m, 1.0 if m == "masked_pos_noise" else None) for m in ALL_MODES])
def test_a_graph_gets_the_same_corruption_alone_or_in_a_batch(mode, sigma):
    inputs = _inputs()
    settings = cfe().build_counterfactual_settings(mode, sigma)
    batched = cfe().apply_counterfactual(settings, **inputs)
    for g in (0, 1):
        alone = cfe().apply_counterfactual(cfe().build_counterfactual_settings(mode, sigma), **_graph_slice(inputs, g))
        node = inputs["batch_mapping"] == g
        first = int(torch.where(node)[0][0])
        assert torch.equal(alone.pos, batched.pos[node])
        for name in ("edge_index", "attention_edge_index"):
            edges = getattr(batched, name)
            expected = edges[:, node[edges[0]]] - first
            assert _pairs(getattr(alone, name)) == _pairs(expected)
        if mode == "masked_dist_shuffle":
            edge_index = inputs["edge_index"]
            sel = node[edge_index[0]]
            assert torch.equal(alone.edge_distances, batched.edge_distances[sel])


@pytest.mark.parametrize("mode,sigma", [(m, 1.0 if m == "masked_pos_noise" else None) for m in ALL_MODES])
def test_corruption_is_reproducible_and_depends_on_the_eval_seed(mode, sigma):
    inputs = _inputs()
    first = cfe().apply_counterfactual(cfe().build_counterfactual_settings(mode, sigma), **inputs)
    second = cfe().apply_counterfactual(cfe().build_counterfactual_settings(mode, sigma), **inputs)
    for name in ("pos", "edge_index", "attention_edge_index"):
        assert torch.equal(getattr(first, name), getattr(second, name))
    if first.edge_distances is not None:
        assert torch.equal(first.edge_distances, second.edge_distances)
    if mode in ("esa_edges_remove", "masked_edges_remove"):
        return  # deterministic by construction, no random draw
    reseeded = dict(inputs, eval_mask_seeds=[s + 1000 for s in inputs["eval_mask_seeds"]])
    third = cfe().apply_counterfactual(cfe().build_counterfactual_settings(mode, sigma), **reseeded)
    differs = (not torch.equal(first.pos, third.pos)
               or _pairs(first.attention_edge_index) != _pairs(third.attention_edge_index)
               or (first.edge_distances is not None and not torch.equal(first.edge_distances, third.edge_distances)))
    assert differs


def test_corruption_does_not_depend_on_the_global_rng():
    inputs = _inputs()
    torch.manual_seed(1)
    first = cfe().apply_counterfactual(cfe().build_counterfactual_settings("masked_pos_noise", 1.0), **inputs)
    torch.manual_seed(2)
    second = cfe().apply_counterfactual(cfe().build_counterfactual_settings("masked_pos_noise", 1.0), **inputs)
    assert torch.equal(first.pos, second.pos)


def test_stats_count_what_was_corrupted():
    inputs = _inputs()
    settings = cfe().build_counterfactual_settings("esa_edges_remove", None)
    out = cfe().apply_counterfactual(settings, **inputs)
    target = inputs["mlm_mask"] & _unimol_node_mask(inputs)
    assert settings.stats["masked_atoms"] == int(target.sum()) == 3
    assert settings.stats["graphs"] == 2
    assert settings.stats["batches"] == 1
    removed = inputs["attention_edge_index"].size(1) - out.attention_edge_index.size(1)
    assert settings.stats["attention_edges_removed"] == removed > 0
    cfe().apply_counterfactual(settings, **inputs)
    assert settings.stats["batches"] == 2 and settings.stats["masked_atoms"] == 6


# ---------------------------------------------------------------------------
# Encoder hook: edge-distance override
# ---------------------------------------------------------------------------


def test_encoder_edge_distance_override_is_used_and_validated(tmp_path, monkeypatch, cpu_flash):
    setup = tev._make_setup(tmp_path, monkeypatch)
    checkpoint = torch.load(setup["a"], map_location="cpu", weights_only=False)
    model = PretrainingESAModel(config=dict(checkpoint["hyper_parameters"]["config"]))
    model.eval()
    encoder = model.encoder
    batch = Batch.from_data_list([tev._small("unimol", 8, "u", 3)])
    x = batch.z.clone()
    pos = batch.pos
    with torch.no_grad():
        clean = encoder(x, pos, batch)
        again = encoder(x, pos, batch)
        assert torch.equal(clean, again)
        edge = batch.edge_index
        true_d = (pos[edge[0]] - pos[edge[1]]).norm(dim=1)
        batch.counterfactual_edge_distances = true_d.clone()
        assert torch.equal(encoder(x, pos, batch), clean)  # the true distances change nothing
        batch.counterfactual_edge_distances = true_d.flip(0).contiguous()
        assert not torch.equal(encoder(x, pos, batch), clean)  # other distances change the features
        batch.counterfactual_edge_distances = true_d[:-1]
        with pytest.raises(ValueError, match="counterfactual_edge_distances"):
            encoder(x, pos, batch)


# ---------------------------------------------------------------------------
# Model forward and evaluator
# ---------------------------------------------------------------------------


def _spread_small(domain, size, mol_id, seed):
    """Like the evaluation test's small molecules, but spread so that the 5 A graph is not complete."""
    gen = torch.Generator().manual_seed(seed)
    size = size + 6
    z = [6, 8, 7, 6, 16, 9] + [6] * (size - 6)
    pos = torch.randn(size, 3, generator=gen) * 2.2
    bonds = None
    if domain == "coconut":
        chain = torch.arange(size - 1)
        bonds = torch.cat([torch.stack((chain, chain + 1)), torch.stack((chain + 1, chain))], dim=1)
    return tev._graph(["sm"] * size, z, pos, [0] * size, ["mol"], domain, mol_id, bonds=bonds)


@pytest.fixture
def spread_setup(tmp_path, monkeypatch, cpu_flash):
    monkeypatch.setattr(tev, "_small", _spread_small)
    return tev._make_setup(tmp_path, monkeypatch)


@pytest.fixture
def tiny(spread_setup):
    setup = spread_setup
    checkpoint = torch.load(setup["a"], map_location="cpu", weights_only=False)
    model = PretrainingESAModel(config=dict(checkpoint["hyper_parameters"]["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.config.seed = 7
    return model, setup


def _unimol_batch(model, seeds=(11, 12, 13)):
    batch = Batch.from_data_list([_spread_small("unimol", 4 + i, f"u{i}", 40 + i) for i in range(3)])
    batch.eval_mask_seed = torch.tensor(seeds, dtype=torch.long)
    model._ensure_mlm_attributes(batch, batch_idx=0, is_validation=True)
    return batch


def _logits(model, batch):
    batch = copy.copy(batch)
    # the trainer runs bf16-mixed (autocast) when the checkpoint config asks for it
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(model.config.use_bfloat16)):
        _, nodes = model.forward(batch)
        return model.pretraining_tasks.mlm_head(nodes), batch


def test_model_without_settings_is_unchanged_and_none_leaves_no_trace(tiny):
    model, _ = tiny
    batch = _unimol_batch(model)
    assert getattr(model, "counterfactual_eval", None) is None
    clean, _ = _logits(model, batch)
    again, _ = _logits(model, batch)
    assert torch.equal(clean, again)


@pytest.mark.parametrize("mode,sigma", [(m, 1.0 if m == "masked_pos_noise" else None) for m in ALL_MODES])
def test_model_forward_changes_masked_unimol_logits_but_not_the_clean_targets(tiny, mode, sigma):
    model, _ = tiny
    batch = _unimol_batch(model)
    clean, clean_batch = _logits(model, batch)
    model.counterfactual_eval = cfe().build_counterfactual_settings(mode, sigma)
    corrupted, cf_batch = _logits(model, batch)
    assert corrupted.shape == clean.shape and torch.isfinite(corrupted).all()
    assert not torch.equal(corrupted, clean)
    # the distance-loss frame and the mask/targets of the batch stay clean
    assert torch.equal(cf_batch.pos, clean_batch.pos)
    assert torch.equal(cf_batch.original_types, batch.original_types)
    assert torch.equal(cf_batch.mlm_mask, batch.mlm_mask)
    assert model.counterfactual_eval.stats["masked_atoms"] == int(batch.mlm_mask.sum())
    # the same input twice gives identical corrupted logits
    again, _ = _logits(model, batch)
    assert torch.equal(corrupted, again)


def test_model_refuses_the_counterfactual_in_training_mode(tiny):
    model, _ = tiny
    batch = _unimol_batch(model)
    model.counterfactual_eval = cfe().build_counterfactual_settings("esa_edges_remove", None)
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(model.config.use_bfloat16)):
        with pytest.raises(RuntimeError, match="training"):
            model.forward(copy.copy(batch))


def test_model_requires_per_graph_eval_seeds(tiny):
    model, _ = tiny
    batch = _unimol_batch(model)
    del batch.eval_mask_seed
    model.counterfactual_eval = cfe().build_counterfactual_settings("masked_pos_noise", 1.0)
    with pytest.raises(ValueError, match="eval_mask_seed"):
        model.forward(copy.copy(batch))


def test_cli_accepts_modes_and_rejects_unknown_ones(capsys):
    base = ["--ckpt", "x.ckpt", "--split", "val", "--seed", "0", "--output-dir", "/nonexistent/out"]
    args = ev.parse_args(base)
    assert args.counterfactual == "none" and args.counterfactual_sigma is None
    args = ev.parse_args(base + ["--counterfactual", "masked_pos_noise", "--counterfactual-sigma", "2.0"])
    assert args.counterfactual == "masked_pos_noise" and args.counterfactual_sigma == 2.0
    with pytest.raises(SystemExit):
        ev.parse_args(base + ["--counterfactual", "bogus"])
    assert "unknown counterfactual mode" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        ev.parse_args(base + ["--counterfactual", "esa_edges_remove", "--counterfactual-sigma", "1.0"])
    assert "sigma" in capsys.readouterr().err


def _numbers(metrics):
    return {"domain_mlm": metrics["domain_mlm"], "losses": metrics["losses"],
            "mask_digest": metrics["mask_digest"], "protein_per_atom": metrics["protein_per_atom"],
            "rna_per_atom": metrics["rna_per_atom"]}


def test_none_mode_keeps_the_evaluation_identical(plain_setup):
    setup = plain_setup
    tmp = setup["tmp"]
    plain = ev.evaluate(ev.parse_args(["--ckpt", str(setup["a"]), "--split", "val", "--seed", "5",
                                       "--device", "cpu", "--output-dir", str(tmp / "plain")]))
    explicit = ev.evaluate(ev.parse_args(["--ckpt", str(setup["a"]), "--split", "val", "--seed", "5",
                                          "--device", "cpu", "--output-dir", str(tmp / "explicit"),
                                          "--counterfactual", "none"]))
    assert _numbers(plain) == _numbers(explicit)
    assert plain["counterfactual"]["mode"] == "none"
    assert plain["counterfactual"]["applied"] == {}
    assert plain["counterfactual"]["unimol"]["top1_acc_real"] == plain["domain_mlm"]["unimol"]["summary"]["top1_acc_real"]
    assert plain["provenance"]["arguments"]["counterfactual"] == "none"
    # the tiny model has the same numbers as the evaluation before the modes existed
    reference = os.environ.get("MIND_CLEAN_EVAL_REFERENCE")
    if reference:
        expected = json.load(open(reference))
        got = json.loads(json.dumps(_numbers(plain), sort_keys=True, default=str))
        assert got == expected


@pytest.fixture
def plain_setup(tmp_path, monkeypatch, cpu_flash):
    return tev._make_setup(tmp_path, monkeypatch)


@pytest.mark.parametrize("mode,sigma", [(m, "1.0" if m == "masked_pos_noise" else None) for m in ALL_MODES])
def test_mode_changes_only_unimol_and_records_its_fields(spread_setup, mode, sigma):
    setup = spread_setup
    tmp = setup["tmp"]
    common = ["--ckpt", str(setup["a"]), "--split", "val", "--seed", "5", "--device", "cpu"]
    clean = ev.evaluate(ev.parse_args(common + ["--output-dir", str(tmp / "clean")]))
    extra = ["--counterfactual", mode] + (["--counterfactual-sigma", sigma] if sigma else [])
    corrupted = ev.evaluate(ev.parse_args(common + ["--output-dir", str(tmp / mode)] + extra))

    # protein, RNA and COCONUT: identical numbers and masks; UniMol: different
    for domain in ("protein", "rna", "coconut"):
        assert corrupted["domain_mlm"][domain] == clean["domain_mlm"][domain], domain
        assert (corrupted["losses"]["per_domain_batch_means"][domain]
                == clean["losses"]["per_domain_batch_means"][domain]), domain
    assert corrupted["protein_per_atom"] == clean["protein_per_atom"]
    assert corrupted["rna_per_atom"] == clean["rna_per_atom"]
    assert corrupted["mask_digest"] == clean["mask_digest"]
    assert corrupted["domain_mlm"]["unimol"] != clean["domain_mlm"]["unimol"]
    assert (corrupted["losses"]["per_domain_batch_means"]["unimol"]["mlm"]
            != clean["losses"]["per_domain_batch_means"]["unimol"]["mlm"])

    block = corrupted["counterfactual"]
    assert block["mode"] == mode
    assert block["domains"] == ["unimol"]
    assert block["sigma"] == (float(sigma) if sigma else None)
    assert block["applied"]["masked_atoms"] == corrupted["domain_mlm"]["unimol"]["summary"]["n_masked"] > 0
    assert block["applied"]["batches"] == corrupted["plan"]["unimol"]["evaluated_batches"]
    summary = corrupted["domain_mlm"]["unimol"]["summary"]
    assert block["unimol"]["top1_acc_real"] == summary["top1_acc_real"]
    assert block["unimol"]["ce_real"] == summary["ce_real"]
    losses = corrupted["losses"]["per_domain_batch_means"]["unimol"]
    assert block["unimol"]["losses"] == {k: v for k, v in losses.items() if k != "n_batches"}
    support = block["unimol"]["element_support"]
    assert "C" in support and sum(support.values()) == block["unimol"]["n_masked"]
    assert set(block["unimol"]["element_recall"]) == set(support)
    assert math.isfinite(block["unimol"]["losses"]["total"])

    saved = json.loads((tmp / mode / "metrics.json").read_text())
    assert saved["counterfactual"]["mode"] == mode
    assert saved["provenance"]["counterfactual"]["mode"] == mode
    # a redefined mode records its revision, so old and new results get different protocol hashes
    expected_revision = 2 if mode == "masked_dist_shuffle" else None
    assert saved["provenance"]["counterfactual"].get("mode_revision") == expected_revision
    assert block.get("mode_revision") == expected_revision
    assert saved["provenance"]["arguments"]["counterfactual"] == mode
    # the checkpoint's own model config is unchanged by the evaluation modes
    assert saved["provenance"]["model_config"] == clean["provenance"]["model_config"]


def test_evaluator_refuses_a_mode_that_would_touch_nothing(spread_setup):
    setup = spread_setup
    args = ev.parse_args(["--ckpt", str(setup["a"]), "--split", "val", "--seed", "5", "--device", "cpu",
                          "--output-dir", str(setup["tmp"] / "bad"), "--domains", "protein,rna",
                          "--counterfactual", "esa_edges_remove"])
    with pytest.raises(ValueError, match="unimol"):
        ev.evaluate(args)


def test_all_four_domains_still_evaluate_with_a_mode(spread_setup):
    setup = spread_setup
    args = ev.parse_args(["--ckpt", str(setup["a"]), "--split", "val", "--seed", "5", "--device", "cpu",
                          "--output-dir", str(setup["tmp"] / "four"), "--counterfactual", "masked_pos_random"])
    metrics = ev.evaluate(args)
    assert set(metrics["domain_mlm"]) == {"protein", "rna", "unimol", "coconut"}

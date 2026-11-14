"""Contract tests of the UniMol masked-atom geometry hiding (G1, "candidate C").

UniMol masks single atoms but leaves the masked atom's position, its distances and its
distance-derived features visible, so the element can be read from the geometry (99.9 %
accuracy; 34 % once the position is moved by 1 A). With ``unimol_masked_geometry_hiding``
the encoder and ESA see, of a masked UniMol atom, only the binary bond topology
(``0.5 < d <= bond_threshold_angstrom`` on the clean geometry); no position, no distance,
no RBF/neighbour-count feature, no radius edge.

Synthetic inputs use the production field names (``z``, ``pos``, ``edge_index``,
``dataset_type``, ``block_idx``, ``mol_id``); the model-level tests reuse the tiny
four-domain CPU split of ``test_evaluate_pretrain``; the false-bond test reads real UniMol
molecules (skipped, loudly, when the data or RDKit is missing).
"""

import copy
import glob
import inspect
import math
import os
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data.batch import Batch

import test_evaluate_pretrain as tev
from core import evaluate_pretrain as ev
from core.pretraining_config import create_pretraining_config
from data_loading.data_root import data_root
from core.pretraining_model import PretrainingESAModel

THRESHOLD = 2.1


def hiding():
    import core.unimol_geometry_hiding as module
    return module


def normalize(value):
    from core.pretraining_config import normalize_unimol_masked_geometry_hiding_config
    return normalize_unimol_masked_geometry_hiding_config(value)


# ---------------------------------------------------------------------------
# Config contract (fail loud)
# ---------------------------------------------------------------------------


def test_default_is_disabled_and_needs_no_threshold():
    for value in (None, {}, {"enabled": False}):
        assert normalize(value)["enabled"] is False
    assert normalize(None)["bond_threshold_angstrom"] is None


def test_enabled_requires_an_explicit_threshold():
    with pytest.raises(ValueError, match="bond_threshold_angstrom"):
        normalize({"enabled": True})
    with pytest.raises(ValueError, match="bond_threshold_angstrom"):
        normalize({"enabled": True, "bond_threshold_angstrom": None})
    assert normalize({"enabled": True, "bond_threshold_angstrom": 2.2}) == {
        "enabled": True, "bond_threshold_angstrom": 2.2}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0, 0.5, 3.01, True, "2.2", [2.2]])
def test_invalid_threshold_values_fail_loudly(bad):
    for enabled in (True, False):
        with pytest.raises((ValueError, TypeError), match="bond_threshold_angstrom"):
            normalize({"enabled": enabled, "bond_threshold_angstrom": bad})


def test_integer_threshold_is_accepted_as_a_float():
    assert normalize({"enabled": True, "bond_threshold_angstrom": 2}) ["bond_threshold_angstrom"] == 2.0


@pytest.mark.parametrize("bad", [
    "yes", 1, [True], {"enabled": 1, "bond_threshold_angstrom": 2.2},
    {"enabled": "true", "bond_threshold_angstrom": 2.2},
    {"enabled": True, "bond_threshold_angstrom": 2.2, "unimol": True},
    {"enable": True, "bond_threshold_angstrom": 2.2},
])
def test_malformed_config_fails_loudly(bad):
    with pytest.raises((ValueError, TypeError)):
        normalize(bad)


def _base_config_kwargs():
    values = tev.load_config_yaml(os.path.join(tev.REPO, "core", "pretraining_config_multidomain_1m_coconut.yaml"))
    return values


def test_create_config_normalizes_and_rejects_incompatible_settings():
    values = _base_config_kwargs()
    good = {"enabled": True, "bond_threshold_angstrom": 2.2}
    config = create_pretraining_config(**{**values, "unimol_masked_geometry_hiding": good})
    assert config.unimol_masked_geometry_hiding == good
    assert create_pretraining_config(**values).unimol_masked_geometry_hiding["enabled"] is False
    for key, value, text in (
        ("apply_attention_on", "edge", "node attention"),
        ("use_3d_coordinates", False, "use_3d_coordinates"),
        ("use_distance_bias", True, "distance"),
        ("use_equivariant_features", True, "equivariant"),
        ("pretraining_tasks", [t for t in values["pretraining_tasks"] if t != "mlm"], "mlm"),
        ("pretraining_tasks", [*values["pretraining_tasks"], "coordinate_denoising"], "coordinate_denoising"),
    ):
        with pytest.raises(ValueError, match=text):
            create_pretraining_config(**{**values, "unimol_masked_geometry_hiding": good, key: value})


def test_shipped_configs_keep_the_feature_off_but_carry_the_key():
    """Candidate C ships switched off; turning it on is a separate, explicit decision."""
    import yaml
    for name in ("pretraining_config_multidomain_1m.yaml", "pretraining_config_multidomain_1m_coconut.yaml"):
        with open(os.path.join(tev.REPO, "core", name)) as handle:
            raw = yaml.safe_load(handle)
        assert raw["unimol_masked_geometry_hiding"] == {"enabled": False, "bond_threshold_angstrom": 2.1}, name
    # derived single-domain configs inherit the key through base_config
    for name in ("pretraining_config_unimol_only.yaml", "pretraining_config_protein_only.yaml"):
        values = tev.load_config_yaml(os.path.join(tev.REPO, "core", name))
        assert create_pretraining_config(**values).unimol_masked_geometry_hiding == {
            "enabled": False, "bond_threshold_angstrom": 2.1}, name


# ---------------------------------------------------------------------------
# Synthetic molecules (view level)
# ---------------------------------------------------------------------------


def _chain(n, bond=1.45, angle_deg=112.0, seed=0, x0=0.0, extra=None):
    """Zig-zag chain with bond length ``bond``; 1,3 distance about 2.4 A. ``extra``: {index: bond length}."""
    half = math.radians(angle_deg) / 2
    pos = []
    for i in range(n):
        pos.append([x0 + i * bond * math.sin(half), (i % 2) * bond * math.cos(half), 0.0])
    pos = torch.tensor(pos, dtype=torch.float)
    return pos


def _edges(pos, graph_of, cutoff=5.0):
    n = pos.size(0)
    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    keep = (src != dst) & (graph_of[src] == graph_of[dst]) & ((pos[src] - pos[dst]).norm(dim=-1) < cutoff)
    return torch.stack((src[keep], dst[keep]))


def _batch_inputs(masked=((2,), (1, 3)), domains=("unimol", "unimol"), sizes=(6, 7)):
    graph_of = torch.cat([torch.full((n,), g) for g, n in enumerate(sizes)])
    pos = torch.cat([_chain(n, bond=1.45 + 0.02 * g, x0=10.0 * g) for g, n in enumerate(sizes)])
    pos = pos + 0.05 * torch.randn(pos.shape, generator=torch.Generator().manual_seed(3))
    edge_index = _edges(pos, graph_of)
    mlm = torch.zeros(pos.size(0), dtype=torch.bool)
    starts = torch.cumsum(torch.tensor([0, *sizes]), 0)
    for g, local in enumerate(masked):
        for i in local:
            mlm[starts[g] + i] = True
    return dict(
        pos=pos, edge_index=edge_index, attention_edge_index=edge_index.clone(), batch_mapping=graph_of,
        mlm_mask=mlm, graph_domains=list(domains),
        bond_edge_index=torch.empty((2, 0), dtype=torch.long),
        has_explicit_bonds=torch.zeros(len(sizes), dtype=torch.bool),
        bond_threshold_angstrom=THRESHOLD, mol_ids=[f"mol{g}" for g in range(len(sizes))],
    )


def _pairs(edge_index):
    return {(int(a), int(b)) for a, b in edge_index.t().tolist()}


def _touching(edge_index, nodes):
    return nodes[edge_index[0]] | nodes[edge_index[1]]


def _reference_bonds(pos, edge_index, graph_unimol_node, threshold=THRESHOLD):
    d = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=1)
    keep = (d > 0.5) & (d <= threshold) & graph_unimol_node[edge_index[0]]
    return _pairs(edge_index[:, keep])


def test_encoder_view_has_no_radius_edge_at_a_masked_atom_and_keeps_the_rest():
    inputs = _batch_inputs()
    view = hiding().hide_masked_unimol_geometry(**inputs)
    masked = inputs["mlm_mask"]
    assert not bool(_touching(view.edge_index, masked).any())
    original = _pairs(inputs["edge_index"])
    visible_visible = {p for p in original if not (masked[p[0]] or masked[p[1]])}
    assert _pairs(view.edge_index) == visible_visible  # same edges, same order of content
    assert torch.equal(view.hidden_nodes, masked)
    assert view.hidden_graphs.tolist() == [True, True]


def test_encoder_position_of_a_masked_atom_is_a_constant_and_visible_atoms_keep_their_distances():
    inputs = _batch_inputs()
    view = hiding().hide_masked_unimol_geometry(**inputs)
    masked = inputs["mlm_mask"]
    assert torch.equal(view.pos[masked], torch.zeros_like(view.pos[masked]))  # one constant, not the clean positions
    pos, new = inputs["pos"], view.pos
    vis_edges = view.edge_index
    clean_d = (pos[vis_edges[0]] - pos[vis_edges[1]]).norm(dim=1)
    new_d = (new[vis_edges[0]] - new[vis_edges[1]]).norm(dim=1)
    assert torch.allclose(clean_d, new_d, atol=1e-5)
    # centered on the visible atoms of each graph (never on the masked ones)
    for g in range(2):
        nodes = (inputs["batch_mapping"] == g) & ~masked
        assert torch.allclose(new[nodes].mean(0), torch.zeros(3), atol=1e-5)


def test_attention_edges_at_a_masked_atom_are_exactly_its_binary_bonds():
    inputs = _batch_inputs()
    view = hiding().hide_masked_unimol_geometry(**inputs)
    masked = inputs["mlm_mask"]
    all_unimol = torch.ones_like(masked)
    bonds = _reference_bonds(inputs["pos"], inputs["edge_index"], all_unimol)
    assert bonds, "the synthetic chain must contain bonds"
    at_masked = _pairs(view.attention_edge_index[:, _touching(view.attention_edge_index, masked)])
    expected = {p for p in bonds if masked[p[0]] or masked[p[1]]}
    assert at_masked == expected
    # the masked atom's neighbours are reachable over a bond edge in both directions
    for node in torch.where(masked)[0].tolist():
        neighbours = {b for a, b in bonds if a == node}
        assert neighbours
        assert {b for a, b in at_masked if a == node} == neighbours
        assert {a for a, b in at_masked if b == node} == neighbours
    # visible-visible attention edges are untouched
    original = _pairs(inputs["attention_edge_index"])
    assert {p for p in _pairs(view.attention_edge_index) if not (masked[p[0]] or masked[p[1]])} == {
        p for p in original if not (masked[p[0]] or masked[p[1]])}


def test_bond_topology_is_symmetric_binary_and_handed_to_the_encoder_as_explicit_bonds():
    inputs = _batch_inputs()
    view = hiding().hide_masked_unimol_geometry(**inputs)
    bonds = _pairs(view.bond_edge_index)
    assert bonds == {(b, a) for a, b in bonds}
    assert bonds == _reference_bonds(inputs["pos"], inputs["edge_index"], torch.ones_like(inputs["mlm_mask"]))
    assert view.has_explicit_bonds.tolist() == [True, True]
    assert view.bond_edge_index.dtype == torch.long and view.bond_edge_index.shape[0] == 2


def test_threshold_decides_halogen_bonds_and_never_looks_at_the_element():
    # C-Br 1.94 A and C-I 2.14 A: common chemistry values, not checked against these molecules
    pos = torch.tensor([[0.0, 0, 0], [1.94, 0, 0], [0.0, 5.0, 0], [2.14, 5.0, 0]])
    graph_of = torch.zeros(4, dtype=torch.long)
    edges = _edges(pos, graph_of, cutoff=6.0)
    mlm = torch.tensor([False, True, False, True])

    def run(threshold):
        return hiding().hide_masked_unimol_geometry(
            pos=pos, edge_index=edges, attention_edge_index=edges.clone(), batch_mapping=graph_of, mlm_mask=mlm,
            graph_domains=["unimol"], bond_edge_index=torch.empty((2, 0), dtype=torch.long),
            has_explicit_bonds=torch.zeros(1, dtype=torch.bool), bond_threshold_angstrom=threshold, mol_ids=["m"])

    assert _pairs(run(2.2).bond_edge_index) == {(0, 1), (1, 0), (2, 3), (3, 2)}
    assert _pairs(run(2.1).bond_edge_index) == {(0, 1), (1, 0)}  # C-I at 2.14 A is just outside 2.1 A
    assert _pairs(run(1.85).bond_edge_index) == set()  # the legacy threshold would leave both halogens isolated
    assert _pairs(run(2.0).bond_edge_index) == {(0, 1), (1, 0)}
    # the function has no argument that could carry an element
    parameters = set(inspect.signature(hiding().hide_masked_unimol_geometry).parameters)
    assert not parameters & {"z", "atom_types", "x", "original_types", "elements"}


def test_masked_atom_position_is_invisible_unless_the_bond_set_changes():
    inputs = _batch_inputs()
    base = hiding().hide_masked_unimol_geometry(**inputs)
    masked = inputs["mlm_mask"]
    moved = dict(inputs)
    pos = inputs["pos"].clone()
    direction = torch.randn(pos.shape, generator=torch.Generator().manual_seed(9))
    pos[masked] += 0.03 * direction[masked]  # small move: no distance crosses the threshold or the 0.5 floor
    moved["pos"] = pos
    ref = torch.ones_like(masked)
    assert _reference_bonds(pos, inputs["edge_index"], ref) == _reference_bonds(inputs["pos"], inputs["edge_index"], ref), (
        "precondition: the perturbation must keep the bond set")
    again = hiding().hide_masked_unimol_geometry(**moved)
    for name in ("pos", "edge_index", "attention_edge_index", "bond_edge_index", "has_explicit_bonds"):
        assert torch.equal(getattr(base, name), getattr(again, name)), name

    # a large move changes the bond set (the only channel): encoder-level position/edges stay the same
    far = pos.clone()
    far[masked] += 50.0
    changed = hiding().hide_masked_unimol_geometry(**{**inputs, "pos": far})
    assert torch.equal(changed.pos, base.pos) and torch.equal(changed.edge_index, base.edge_index)
    assert not torch.equal(changed.bond_edge_index, base.bond_edge_index)


def test_other_domains_pass_through_and_keep_their_own_bond_source():
    inputs = _batch_inputs(domains=("unimol", "rna"), masked=((2,), (1, 3)))
    # the second graph carries an explicit bond list of its own (COCONUT-style) or none (RNA-style: proxy later)
    view = hiding().hide_masked_unimol_geometry(**inputs)
    other = inputs["batch_mapping"] == 1
    assert view.hidden_graphs.tolist() == [True, False]
    assert torch.equal(view.pos[other], inputs["pos"][other])
    assert torch.equal(view.hidden_nodes, inputs["mlm_mask"] & ~other)
    kept = _pairs(inputs["edge_index"][:, other[inputs["edge_index"][0]]])
    assert {p for p in _pairs(view.edge_index) if other[p[0]]} == kept
    assert {p for p in _pairs(view.attention_edge_index) if other[p[0]]} == kept
    assert view.has_explicit_bonds.tolist() == [True, False]  # the RNA-style graph keeps the 1.85 A proxy path
    assert not bool(other[view.bond_edge_index[0]].any())


def test_unimol_graph_without_masked_atoms_is_unchanged_apart_from_the_bond_source():
    inputs = _batch_inputs(masked=((), (1,)))
    view = hiding().hide_masked_unimol_geometry(**inputs)
    first = inputs["batch_mapping"] == 0
    assert view.hidden_graphs.tolist() == [True, True]
    assert {p for p in _pairs(view.edge_index) if first[p[0]]} == {p for p in _pairs(inputs["edge_index"]) if first[p[0]]}


def test_view_fails_loudly_on_malformed_inputs_naming_the_molecule():
    good = _batch_inputs()

    def call(**changes):
        return hiding().hide_masked_unimol_geometry(**{**good, **changes})

    explicit = torch.tensor([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="explicit bond") as info:
        call(bond_edge_index=explicit, has_explicit_bonds=torch.tensor([True, False]))
    assert "mol0" in str(info.value) and "unimol" in str(info.value)
    with pytest.raises(ValueError, match="mlm_mask"):
        call(mlm_mask=good["mlm_mask"].long())
    with pytest.raises(ValueError, match="mlm_mask"):
        call(mlm_mask=good["mlm_mask"][:-1])
    with pytest.raises(ValueError, match="graph_domains"):
        call(graph_domains=["unimol"])
    with pytest.raises(ValueError, match="mol_ids"):
        call(mol_ids=["a"])
    with pytest.raises(ValueError, match="bond_threshold_angstrom"):
        call(bond_threshold_angstrom=float("nan"))
    with pytest.raises(ValueError, match="bond_threshold_angstrom"):
        call(bond_threshold_angstrom=-2.2)
    with pytest.raises(ValueError, match="edge"):
        call(edge_index=torch.tensor([[0, 99], [99, 0]]))
    cross = torch.tensor([[0, 8], [8, 0]])
    with pytest.raises(ValueError, match="cross-graph"):
        call(edge_index=cross, attention_edge_index=cross)
    with pytest.raises(ValueError, match="pos"):
        call(pos=good["pos"][:, :2])


def test_protein_rna_threshold_stays_1_85_while_unimol_uses_the_configured_one():
    """Encoder level: the explicit bonds of the hidden UniMol graph and the 1.85 A proxy of the other graph coexist."""
    # one UniMol-like pair at 1.95 A (bonded under 2.1) and one RNA-like pair at 1.95 A (not bonded under 1.85)
    pos = torch.tensor([[0.0, 0, 0], [1.95, 0, 0], [0.0, 9, 0], [1.95, 9, 0]])
    graph_of = torch.tensor([0, 0, 1, 1])
    edges = _edges(pos, graph_of, cutoff=6.0)
    view = hiding().hide_masked_unimol_geometry(
        pos=pos, edge_index=edges, attention_edge_index=edges.clone(), batch_mapping=graph_of,
        mlm_mask=torch.zeros(4, dtype=torch.bool), graph_domains=["unimol", "rna"],
        bond_edge_index=torch.empty((2, 0), dtype=torch.long), has_explicit_bonds=torch.zeros(2, dtype=torch.bool),
        bond_threshold_angstrom=2.1, mol_ids=["u", "r"])
    encoder_batch = SimpleNamespace(batch=graph_of, edge_index=view.edge_index, bond_edge_index=view.bond_edge_index,
                                    has_explicit_bonds=view.has_explicit_bonds)
    distances = (pos[view.edge_index[0]] - pos[view.edge_index[1]]).norm(dim=1)
    values = _base_config_kwargs()
    layers = len(values["layer_types"])
    config = create_pretraining_config(**{**values, "hidden_dims": [32] * layers, "num_heads": [4] * layers,
                                          "graph_dim": 32, "mlp_hidden_size": 64, "gaussian_kernels": 16})
    encoder = PretrainingESAModel(config).encoder
    coordination = encoder._compute_chemical_coordination(torch.ones(2, 2, dtype=torch.long), encoder_batch,
                                                          pos=pos, edge_distances=distances)
    assert coordination.tolist() == [[1.0, 1.0], [0.0, 0.0]]


# ---------------------------------------------------------------------------
# Model level (tiny CPU model, production forward)
# ---------------------------------------------------------------------------


def _spread_small(domain, size, mol_id, seed):
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


def _model(setup, enabled, threshold=THRESHOLD):
    checkpoint = torch.load(setup["a"], map_location="cpu", weights_only=False)
    config = dict(checkpoint["hyper_parameters"]["config"])
    config["unimol_masked_geometry_hiding"] = {"enabled": enabled, "bond_threshold_angstrom": threshold}
    model = PretrainingESAModel(config=config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.config.seed = 7
    return model


def _unimol_batch(model, domain="unimol", count=3, seeds=(11, 12, 13)):
    graphs = [_spread_small(domain, 4 + i, f"{domain[0]}{i}", 40 + i) for i in range(count)]
    batch = Batch.from_data_list(graphs)
    batch.eval_mask_seed = torch.tensor(seeds[:count], dtype=torch.long)
    model._ensure_mlm_attributes(batch, batch_idx=0, is_validation=True)
    return batch


def _logits(model, batch):
    batch = copy.copy(batch)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(model.config.use_bfloat16)):
        _, nodes = model.forward(batch)
        return model.pretraining_tasks.mlm_head(nodes), batch


def _spy_encoder(model, monkeypatch):
    seen = {}
    original = model.encoder.forward

    def spy(x, pos, batch_obj):
        seen.update(x=x.clone(), pos=pos.clone(), edge_index=batch_obj.edge_index.clone(),
                    bonds=batch_obj.bond_edge_index.clone() if getattr(batch_obj, "bond_edge_index", None) is not None else None,
                    explicit=getattr(batch_obj, "has_explicit_bonds", None),
                    batch=batch_obj.batch.clone())
        return original(x, pos, batch_obj)

    monkeypatch.setattr(model.encoder, "forward", spy)
    return seen


def _bond_set(batch, threshold=THRESHOLD):
    return _reference_bonds(batch.pos, batch.edge_index, torch.ones(batch.pos.size(0), dtype=torch.bool), threshold)


def test_flag_off_matches_a_model_without_the_key(spread_setup):
    off = _model(spread_setup, enabled=False)
    plain = _model(spread_setup, enabled=False)
    plain.config.unimol_masked_geometry_hiding = None  # an old checkpoint / config without the key
    batch = _unimol_batch(off)
    a, _ = _logits(off, batch)
    b, _ = _logits(plain, batch)
    assert torch.equal(a, b)


def test_encoder_input_of_a_masked_unimol_atom_has_no_clean_geometry(spread_setup, monkeypatch):
    model = _model(spread_setup, enabled=True)
    batch = _unimol_batch(model)
    seen = _spy_encoder(model, monkeypatch)
    _logits(model, batch)
    masked = batch.mlm_mask
    assert int(masked.sum()) > 0
    assert torch.equal(seen["x"][masked], torch.zeros_like(seen["x"][masked]))  # the mask token
    assert torch.equal(seen["pos"][masked], torch.zeros_like(seen["pos"][masked]))
    assert not bool(_touching(seen["edge_index"], masked).any())  # no distance edge at a masked atom
    assert bool(seen["explicit"].all())  # UniMol bonds are handed over as an explicit binary list
    # topology of the masked atoms survives: every proxy bond at a masked atom is in the bond list
    bonds = _pairs(seen["bonds"])
    assert {p for p in _bond_set(batch) if masked[p[0]] or masked[p[1]]} <= bonds
    assert any(masked[a] for a, _ in bonds), "the synthetic masked atoms must have bonds for this test to mean anything"

    # control: without the feature the encoder does see edges (and so distances) at masked atoms
    plain = _model(spread_setup, enabled=False)
    seen_plain = _spy_encoder(plain, monkeypatch)
    _logits(plain, batch)
    assert bool(_touching(seen_plain["edge_index"], masked).any())
    assert not torch.equal(seen_plain["pos"][masked], torch.zeros_like(seen_plain["pos"][masked]))


def test_moving_a_masked_atom_changes_nothing_with_hiding_and_something_without(spread_setup):
    hidden, plain = _model(spread_setup, enabled=True), _model(spread_setup, enabled=False)
    batch = _unimol_batch(hidden)
    moved = copy.copy(batch)
    pos = batch.pos.clone()
    delta = torch.randn(pos.shape, generator=torch.Generator().manual_seed(5))
    pos[batch.mlm_mask] += 0.04 * delta[batch.mlm_mask]
    moved.pos = pos
    moved.edge_attr = batch.edge_attr.clone()
    assert _bond_set(moved) == _bond_set(batch), "precondition: the move must keep the bond set"
    clean, _ = _logits(hidden, batch)
    shifted, _ = _logits(hidden, moved)
    assert torch.equal(clean, shifted)  # no leak: the encoder output does not depend on the masked position
    leaky, _ = _logits(plain, batch)
    leaky_shifted, _ = _logits(plain, moved)
    assert not torch.equal(leaky, leaky_shifted)  # the same probe detects the leak when the feature is off


def test_hiding_changes_the_masked_logits_and_keeps_the_clean_targets_and_loss_frame(spread_setup):
    hidden, plain = _model(spread_setup, enabled=True), _model(spread_setup, enabled=False)
    batch = _unimol_batch(hidden)
    on, on_batch = _logits(hidden, batch)
    off, off_batch = _logits(plain, batch)
    assert on.shape == off.shape and torch.isfinite(on).all()
    assert not torch.equal(on, off)
    assert torch.equal(on_batch.pos, off_batch.pos)  # the distance-loss frame keeps the clean positions
    assert torch.equal(on_batch.original_types, batch.original_types)
    assert torch.equal(on_batch.mlm_mask, batch.mlm_mask)
    again, _ = _logits(hidden, batch)
    assert torch.equal(on, again)


def test_a_graph_gets_the_same_logits_alone_or_in_a_mixed_batch(spread_setup):
    model = _model(spread_setup, enabled=True)
    mixed_graphs = [_spread_small("unimol", 5, "u0", 40), _spread_small("coconut", 6, "c0", 41),
                    _spread_small("unimol", 7, "u1", 42)]
    mixed = Batch.from_data_list(mixed_graphs)
    mixed.eval_mask_seed = torch.tensor([21, 22, 23])
    model._ensure_mlm_attributes(mixed, batch_idx=0, is_validation=True)
    alone = Batch.from_data_list([mixed_graphs[0]])
    alone.eval_mask_seed = torch.tensor([21])
    model._ensure_mlm_attributes(alone, batch_idx=0, is_validation=True)
    mixed_logits, _ = _logits(model, mixed)
    alone_logits, _ = _logits(model, alone)
    first = mixed.batch == 0
    assert torch.allclose(mixed_logits[first].float(), alone_logits.float(), atol=5e-2)

    # the COCONUT graph of the mixed batch is unaffected by the UniMol feature
    plain = _model(spread_setup, enabled=False)
    plain_logits, _ = _logits(plain, mixed)
    coconut = mixed.batch == 1
    assert torch.equal(mixed_logits[coconut], plain_logits[coconut])
    assert not torch.equal(mixed_logits[first], plain_logits[first])


def test_model_fails_loudly_when_a_unimol_graph_has_explicit_bonds(spread_setup):
    model = _model(spread_setup, enabled=True)
    batch = _unimol_batch(model)
    batch.bond_edge_index = torch.tensor([[0, 1], [1, 0]])
    batch.has_explicit_bonds = torch.ones(3, dtype=torch.bool)
    with pytest.raises(ValueError, match="explicit bond"):
        _logits(model, batch)


def test_enabled_model_refuses_masking_free_training_batches(spread_setup):
    model = _model(spread_setup, enabled=True)
    # the geometry-neutral protein/RNA view has its own mlm_mask requirement; switch it off to reach ours
    model.config.geometry_neutral_virtual_slots = {"enabled": False, "protein": False, "rna": False}
    batch = Batch.from_data_list([_spread_small("unimol", 4, "u0", 40)])
    model.train()
    with pytest.raises(ValueError, match="mlm_mask"):
        model.forward(copy.copy(batch))
    model.eval()
    # evaluation / downstream embedding without any mask hides nothing and does not fail
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(model.config.use_bfloat16)):
        _, nodes = model.forward(copy.copy(batch))
    assert torch.isfinite(nodes).all()


# ---------------------------------------------------------------------------
# Counterfactual tool on the hidden view
# ---------------------------------------------------------------------------


def _cf(mode, sigma=None):
    import core.counterfactual_eval as module
    return module.build_counterfactual_settings(mode, sigma)


@pytest.mark.parametrize("mode,sigma", [("masked_pos_noise", 1.0), ("masked_pos_random", None),
                                        ("masked_dist_shuffle", None)])
def test_position_and_distance_counterfactuals_are_inert_when_the_geometry_is_hidden(spread_setup, mode, sigma):
    model = _model(spread_setup, enabled=True)
    batch = _unimol_batch(model)
    clean, _ = _logits(model, batch)
    model.counterfactual_eval = _cf(mode, sigma)
    corrupted, _ = _logits(model, batch)
    assert torch.equal(clean, corrupted)
    assert model.counterfactual_eval.stats["masked_atoms"] == int(batch.mlm_mask.sum())


@pytest.mark.parametrize("mode", ["esa_edges_remove", "esa_edges_shuffle", "masked_edges_remove"])
def test_topology_counterfactuals_change_the_hidden_model_logits(spread_setup, mode):
    model = _model(spread_setup, enabled=True)
    batch = _unimol_batch(model)
    clean, _ = _logits(model, batch)
    model.counterfactual_eval = _cf(mode)
    corrupted, _ = _logits(model, batch)
    assert torch.isfinite(corrupted).all() and not torch.equal(clean, corrupted)


def test_masked_edges_remove_also_drops_the_bond_list_of_masked_atoms(spread_setup, monkeypatch):
    model = _model(spread_setup, enabled=True)
    batch = _unimol_batch(model)
    model.counterfactual_eval = _cf("masked_edges_remove")
    seen = _spy_encoder(model, monkeypatch)
    _logits(model, batch)
    assert not bool(_touching(seen["bonds"], batch.mlm_mask).any())
    assert int(seen["bonds"].size(1)) > 0  # visible-visible bonds remain


# ---------------------------------------------------------------------------
# Four-domain evaluation: only UniMol changes
# ---------------------------------------------------------------------------


def test_four_domain_evaluation_changes_only_unimol(spread_setup):
    setup = spread_setup
    config = tev._config(setup["tmp"] / "data")
    config.unimol_masked_geometry_hiding = {"enabled": True, "bond_threshold_angstrom": THRESHOLD}
    hidden_ckpt = tev._checkpoint(setup["tmp"] / "ckpts" / "hidden.ckpt", config, 0)
    base = tev._run(setup["a"], setup["tmp"] / "eval_base")
    hid = tev._run(hidden_ckpt, setup["tmp"] / "eval_hidden")
    assert base["mask_digest"] == hid["mask_digest"]
    for domain in ("protein", "rna", "coconut"):
        assert base["domain_mlm"][domain] == hid["domain_mlm"][domain], domain
        assert base["losses"]["per_domain_batch_means"][domain] == hid["losses"]["per_domain_batch_means"][domain]
    assert base["domain_mlm"]["unimol"] != hid["domain_mlm"]["unimol"]


# ---------------------------------------------------------------------------
# Real UniMol molecules: false-bond rate of the distance proxy (mandatory check)
# ---------------------------------------------------------------------------

UNIMOL_ROOT = str(data_root() / "small_molecules" / "unimol")
REAL_CHUNK = "universal_unimol_chunk_199"


def _real_sample(count_random=300, count_halogen=150):
    rdkit = pytest.importorskip("rdkit", reason="RDKit is needed for the covalent-radius reference")
    paths = glob.glob(os.path.join(UNIMOL_ROOT, REAL_CHUNK, "processed", "optimized_*.pt"))
    if not paths:
        pytest.skip(f"real UniMol chunk not found under {UNIMOL_ROOT}/{REAL_CHUNK}")
    data, slices = torch.load(paths[0], map_location="cpu", weights_only=False, mmap=True)[:2]
    n_mol = slices["z"].numel() - 1
    gen = torch.Generator().manual_seed(0)
    chosen = torch.randperm(n_mol, generator=gen)[:count_random].tolist()
    z_all, z_slice = data.z, slices["z"]
    halogen = []
    for m in torch.randperm(n_mol, generator=gen)[:6000].tolist():
        zz = z_all[z_slice[m]:z_slice[m + 1]]
        if bool(((zz == 35) | (zz == 53)).any()):
            halogen.append(m)
            if len(halogen) >= count_halogen:
                break
    molecules = []
    e_slice = slices["edge_index"]
    for m in sorted(set(chosen) | set(halogen)):
        a, b = int(z_slice[m]), int(z_slice[m + 1])
        edges = data.edge_index[:, int(e_slice[m]):int(e_slice[m + 1])]
        molecules.append((m in halogen, data.z[a:b].clone(), data.pos[a:b].clone(), edges.clone()))
    return rdkit, molecules


def _false_bond_table(rdkit, molecules, threshold, factor):
    from rdkit import Chem
    table = Chem.GetPeriodicTable()
    radius = torch.tensor([0.0] + [table.GetRcovalent(i) for i in range(1, 100)])
    rows = {}
    wrong_atoms = atoms = 0
    for _, z, pos, edges in molecules:
        if bool(((z < 1) | (z > 99)).any()) or z.numel() < 2:
            continue
        n = z.numel()
        graph_of = torch.zeros(n, dtype=torch.long)
        view = hiding().hide_masked_unimol_geometry(
            pos=pos, edge_index=edges, attention_edge_index=edges.clone(), batch_mapping=graph_of,
            mlm_mask=torch.zeros(n, dtype=torch.bool), graph_domains=["unimol"],
            bond_edge_index=torch.empty((2, 0), dtype=torch.long), has_explicit_bonds=torch.zeros(1, dtype=torch.bool),
            bond_threshold_angstrom=threshold, mol_ids=["real"])
        proxy = {(min(a, b), max(a, b)) for a, b in view.bond_edge_index.t().tolist()}
        d = (pos[edges[0]] - pos[edges[1]]).norm(dim=1)
        ref_mask = (d > 0.5) & (d <= factor * (radius[z[edges[0]]] + radius[z[edges[1]]]))
        ref = {(min(a, b), max(a, b)) for a, b in edges[:, ref_mask].t().tolist()}
        for a, b in proxy | ref:
            names = "-".join(sorted((table.GetElementSymbol(int(z[a])), table.GetElementSymbol(int(z[b])))))
            row = rows.setdefault(names, {"proxy": 0, "false": 0, "ref": 0, "missed": 0})
            row["proxy"] += (a, b) in proxy
            row["false"] += (a, b) in proxy and (a, b) not in ref
            row["ref"] += (a, b) in ref
            row["missed"] += (a, b) in ref and (a, b) not in proxy
        bad = {x for pair in proxy ^ ref for x in pair}
        wrong_atoms += len(bad)
        atoms += n
    return rows, wrong_atoms, atoms


@pytest.mark.parametrize("factor", [1.15, 1.30])
def test_false_bond_rate_of_the_distance_proxy_on_real_unimol_molecules(factor, capsys):
    """Mandatory: how often does ``d <= 2.1 A`` call a non-bonded pair a bond, per element pair.

    Reference: covalent radii (Cordero, RDKit periodic table), bonded when d <= factor * (r_i + r_j);
    the UniMol chunks store no bond list. Radii and factors are common chemistry values, not
    verified against these molecules. Measured on 1800 random molecules of three chunks (outside
    this test): false share of proxy bonds 0.24-0.56 %, atoms with a wrong bond set 0.5-1.1 %.
    Measured on this test's sample (449 molecules of chunk 199): false share
    0.19-0.38 %, wrong bond set 0.47-0.82 %, missed 6 of ~13000 reference bonds, all of them C-I
    (6 of 27). On 1196 iodine/bromine-rich molecules 35 of 189 C-I bonds are missed (18.5 %), no C-Br.
    """
    rdkit, molecules = _real_sample()
    rows, wrong_atoms, atoms = _false_bond_table(rdkit, molecules, THRESHOLD, factor)
    total = {k: sum(row[k] for row in rows.values()) for k in ("proxy", "false", "ref", "missed")}
    with capsys.disabled():
        print(f"\nfalse-bond table, thr={THRESHOLD} A, reference factor {factor}, {len(molecules)} molecules, {atoms} atoms")
        for name, row in sorted(rows.items(), key=lambda item: -item[1]["proxy"])[:14]:
            print(f"  {name:6s} proxy {row['proxy']:6d} false {row['false']:5d} ref {row['ref']:6d} missed {row['missed']:4d}")
        print(f"  total proxy {total['proxy']} false {total['false']} ({total['false'] / total['proxy']:.2%}); "
              f"missed {total['missed']}/{total['ref']}; atoms with a wrong bond set {wrong_atoms / atoms:.2%}")
    assert total["proxy"] > 5000 and total["ref"] > 5000, "sample too small to mean anything"
    # bromine keeps its bond at 2.1 A (masked atoms are treated like any other element)
    if "Br-C" in rows and rows["Br-C"]["ref"] >= 10:
        assert rows["Br-C"]["missed"] / rows["Br-C"]["ref"] < 0.01
    # known cost of 2.1 A: part of the long C-I bonds (~2.1-2.2 A) is lost; alarm, not a quality claim
    if "C-I" in rows and rows["C-I"]["ref"] >= 10:
        assert rows["C-I"]["missed"] / rows["C-I"]["ref"] < 0.30
    # same-element pairs that are never bonded (F-F, O-O sit at ~2.1-2.2 A): the 2.2 A threshold turned
    # every one of them into a bond, an element-dependent cue; 2.1 A must not
    for name in ("F-F", "O-O"):
        if name in rows:
            assert rows[name]["false"] == 0, name
    # bounds just above the measured cost of 2.1 A (measured 0.19-0.38 % / 0.47-0.82 % / 0.046 %)
    assert total["false"] / total["proxy"] < 0.006
    assert wrong_atoms / atoms < 0.012
    assert total["missed"] / total["ref"] < 0.001


# ---------------------------------------------------------------------------
# Paths that need their own check: legacy (no geometry-neutral view) and a protein+UniMol batch
# ---------------------------------------------------------------------------


def test_hiding_also_works_without_the_geometry_neutral_view(spread_setup, monkeypatch):
    model = _model(spread_setup, enabled=True)
    model.config.geometry_neutral_virtual_slots = {"enabled": False, "protein": False, "rna": False}
    batch = _unimol_batch(model)
    seen = _spy_encoder(model, monkeypatch)
    clean, _ = _logits(model, batch)
    assert not bool(_touching(seen["edge_index"], batch.mlm_mask).any())
    assert torch.equal(seen["pos"][batch.mlm_mask], torch.zeros_like(seen["pos"][batch.mlm_mask]))
    moved = copy.copy(batch)
    pos = batch.pos.clone()
    pos[batch.mlm_mask] += 0.04 * torch.randn(pos.shape, generator=torch.Generator().manual_seed(5))[batch.mlm_mask]
    moved.pos = pos
    assert _bond_set(moved) == _bond_set(batch)
    shifted, _ = _logits(model, moved)
    assert torch.equal(clean, shifted)


def test_a_protein_and_a_unimol_graph_in_one_batch_keep_independent_logits(spread_setup):
    """Compact indexing: protein nodes leave the encoder graph, UniMol nodes shift; the UniMol logits must not care."""
    model = _model(spread_setup, enabled=True)
    config = model.config
    dataset, indices, _ = ev.load_split_dataset(config, "val", ["protein", "unimol"])
    entries = ev.sample_entries(dataset, indices, ["protein", "unimol"])
    protein_item = dataset[entries["protein"][0][0]]
    unimol_item = dataset[entries["unimol"][0][0]]

    def prepared(items, seeds):
        batch = Batch.from_data_list(items)
        batch.eval_mask_seed = torch.tensor(seeds)
        model._ensure_mlm_attributes(batch, batch_idx=0, is_validation=True)
        return batch

    mixed = prepared([protein_item, unimol_item], [31, 32])
    alone = prepared([unimol_item], [32])
    mixed_logits, _ = _logits(model, mixed)
    alone_logits, _ = _logits(model, alone)
    unimol_nodes = mixed.batch == 1
    assert torch.allclose(mixed_logits[unimol_nodes].float(), alone_logits.float(), atol=5e-2)
    plain = _model(spread_setup, enabled=False)
    plain_logits, _ = _logits(plain, prepared([protein_item, unimol_item], [31, 32]))
    assert torch.equal(mixed_logits[~unimol_nodes], plain_logits[~unimol_nodes])  # protein graph unaffected


# ---------------------------------------------------------------------------
# Startup log line (the flag was invisible in run logs)
# ---------------------------------------------------------------------------


class _StopAfterBanner(Exception):
    pass


def _startup_output(monkeypatch, capsys, hiding, ckpt_path=None):
    """Run train_universal_pretraining up to the dataset load and return what it printed."""
    from core import train_pretrain

    def stop(*args, **kwargs):
        raise _StopAfterBanner

    monkeypatch.setattr(train_pretrain, "load_universal_dataset", stop)
    values = _base_config_kwargs()
    values["unimol_masked_geometry_hiding"] = hiding
    config = create_pretraining_config(**values)
    with pytest.raises(_StopAfterBanner):
        train_pretrain.train_universal_pretraining(
            config, "d", "/nonexistent", "/nonexistent", "p", "r", ckpt_path=ckpt_path
        )
    return capsys.readouterr().out


@pytest.mark.parametrize("ckpt_path", [None, "/nonexistent/last.ckpt"])
def test_startup_logs_g1_enabled_with_threshold(monkeypatch, capsys, ckpt_path):
    out = _startup_output(
        monkeypatch, capsys, {"enabled": True, "bond_threshold_angstrom": 2.2}, ckpt_path
    )
    lines = [line for line in out.splitlines() if "G1 masked-geometry hiding" in line]
    assert lines == ["G1 masked-geometry hiding: ENABLED (bond_threshold_angstrom=2.2)"]


@pytest.mark.parametrize("ckpt_path", [None, "/nonexistent/last.ckpt"])
def test_startup_logs_g1_disabled(monkeypatch, capsys, ckpt_path):
    out = _startup_output(
        monkeypatch, capsys, {"enabled": False, "bond_threshold_angstrom": 2.1}, ckpt_path
    )
    lines = [line for line in out.splitlines() if "G1 masked-geometry hiding" in line]
    assert lines == ["G1 masked-geometry hiding: DISABLED"]


# ---------------------------------------------------------------------------
# Evaluation override: one test condition for checkpoints trained with and without hiding
# ---------------------------------------------------------------------------

OFF = {"enabled": False, "bond_threshold_angstrom": None}
ON = {"enabled": True, "bond_threshold_angstrom": THRESHOLD}


def _checkpoint_without_the_key(src, dst):
    """A checkpoint written before the flag existed: its config has no hiding key at all."""
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    del checkpoint["hyper_parameters"]["config"]["unimol_masked_geometry_hiding"]
    torch.save(checkpoint, dst)
    return dst


def _hidden_checkpoint(setup):
    config = tev._config(setup["tmp"] / "data")
    config.unimol_masked_geometry_hiding = dict(ON)
    return tev._checkpoint(setup["tmp"] / "ckpts" / "hidden.ckpt", config, 0)  # same weights as "a" (seed 0)


def test_eval_switches_hiding_on_for_a_checkpoint_without_the_key(spread_setup):
    tmp = spread_setup["tmp"]
    old = _checkpoint_without_the_key(spread_setup["a"], tmp / "ckpts" / "old.ckpt")
    hidden = _hidden_checkpoint(spread_setup)
    plain = tev._run(old, tmp / "plain", "--domains", "unimol")
    forced = tev._run(old, tmp / "forced", "--domains", "unimol",
                      "--unimol-geometry-hiding", "on", "--unimol-bond-threshold", str(THRESHOLD))
    trained = tev._run(hidden, tmp / "trained", "--domains", "unimol")
    assert forced["mask_digest"] == plain["mask_digest"] == trained["mask_digest"]
    # same weights + forced hiding = the model whose checkpoint carries the flag
    assert forced["domain_mlm"]["unimol"] == trained["domain_mlm"]["unimol"]
    assert forced["domain_mlm"]["unimol"] != plain["domain_mlm"]["unimol"]
    assert forced["provenance"]["unimol_masked_geometry_hiding"] == {"source": "on", "checkpoint": OFF, "effective": ON}
    assert plain["provenance"]["unimol_masked_geometry_hiding"] == {"source": "checkpoint", "checkpoint": OFF,
                                                                    "effective": OFF}
    assert trained["provenance"]["unimol_masked_geometry_hiding"] == {"source": "checkpoint", "checkpoint": ON,
                                                                      "effective": ON}


def test_eval_switches_hiding_off_for_a_hidden_checkpoint(spread_setup):
    tmp = spread_setup["tmp"]
    hidden = _hidden_checkpoint(spread_setup)
    off = tev._run(hidden, tmp / "off", "--domains", "unimol", "--unimol-geometry-hiding", "off")
    plain = tev._run(spread_setup["a"], tmp / "plain", "--domains", "unimol")
    assert off["domain_mlm"]["unimol"] == plain["domain_mlm"]["unimol"]
    assert off["provenance"]["unimol_masked_geometry_hiding"] == {"source": "off", "checkpoint": ON, "effective": OFF}


@pytest.mark.parametrize("extra", [
    ["--unimol-geometry-hiding", "on"],                                     # on needs an explicit threshold
    ["--unimol-bond-threshold", "2.1"],                                     # a threshold without on
    ["--unimol-geometry-hiding", "off", "--unimol-bond-threshold", "2.1"],
    ["--unimol-geometry-hiding", "maybe"],
    ["--unimol-geometry-hiding", "on", "--unimol-bond-threshold", "-1"],
    ["--unimol-geometry-hiding", "on", "--unimol-bond-threshold", "nan"],
])
def test_eval_hiding_arguments_fail_loudly(extra, tmp_path):
    with pytest.raises(SystemExit):
        ev.parse_args(["--ckpt", "x.ckpt", "--split", "test", "--seed", "0", "--output-dir", str(tmp_path / "o"), *extra])


def test_yaml_key_absent_from_the_checkpoint_is_reported_not_dropped():
    merged, ignored = ev.merge_eval_config({"mlm_mask_ratio": 0.15}, {"unimol_masked_geometry_hiding": dict(ON)})
    assert "unimol_masked_geometry_hiding" not in merged
    assert ignored == {"unimol_masked_geometry_hiding": {"checkpoint": ev.ABSENT_FROM_CHECKPOINT, "yaml": ON}}


@pytest.mark.parametrize("target, out", [("eval-pretrain", "OUT=/nonexistent/o"),
                                         ("eval-counterfactual", "OUT_ROOT=/nonexistent/o")])
def test_eval_make_targets_pass_the_hiding_override(target, out):
    import subprocess

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def dry_run(*extra):
        result = subprocess.run(["make", "--no-print-directory", "-n", target, "CKPT=/c.ckpt", "SPLIT=test", "GPU=0",
                                 out, *extra], cwd=repo, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return " ".join(result.stdout.replace("\\\n", " ").split())

    assert "--unimol-geometry-hiding on --unimol-bond-threshold 2.1" in dry_run("HIDING=on", "BOND_THRESHOLD=2.1")
    assert "--unimol-geometry-hiding off" in dry_run("HIDING=off")
    assert "--unimol-geometry-hiding" not in dry_run()

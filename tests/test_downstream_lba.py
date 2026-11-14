"""ATOM3D LBA downstream task: pocket selection, complex graph contract and metrics.

Everything runs on CPU with synthetic complexes. The forward pass uses the
production flash_varlen layers through the ``cpu_flash`` SDPA stand-in
(tests/conftest.py) on a tiny model.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from core.pretraining_config import create_pretraining_config, load_config_yaml
from core.pretraining_model import PretrainingESAModel
from data_loading.atom_position_vocab import VIRTUAL_ATOMIC_NUMBER
from downstream.common import complex_graph as cg
from downstream.common import encoder as enc
from downstream.common import graph_builder as gb
from downstream.common.metrics import evaluate_regression, pearson_r, rmse, spearman_rho
from downstream.common.missing_atoms import apply_policy, parse_policy
from downstream.common.structure_io import ChainRecord, ResidueRecord
from downstream.tasks.lba import dataset as ds
from downstream_helpers import residue_atoms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY = {"policy": "drop_residue_missing_backbone", "incomplete_sidechain": "mask"}
ATOM_COLUMNS = [
    "ensemble", "subunit", "structure", "model", "chain", "hetero", "insertion_code", "residue",
    "segid", "resname", "altloc", "occupancy", "bfactor", "x", "y", "z", "element", "name",
    "fullname", "serial_number",
]
SETTINGS = {
    "missing_atoms": POLICY, "mse_to_met": True, "pocket_cutoff": 6.0, "max_atoms": 600,
    "min_residues": 2, "min_ligand_atoms": 1, "contact_atoms": "all",
}


# ---------------------------------------------------------------------------
# Synthetic ATOM3D entries
# ---------------------------------------------------------------------------

def _frame(rows):
    return {"columns": list(ATOM_COLUMNS), "index": list(range(len(rows))), "data": rows}


def _protein_rows(chains):
    """``chains``: {chain_id: [(resname, drop_atom_names)]}; residues sit 3.8 A apart along x."""
    rows, index = [], 0
    for chain_id, residues in chains.items():
        for position, (resname, drop) in enumerate(residues):
            for name, element, xyz in residue_atoms(resname, index, drop=drop):
                rows.append([None, 0, None, 1, chain_id, " ", " ", position + 1, " ", resname, " ",
                             1.0, 20.0, float(xyz[0]), float(xyz[1]), float(xyz[2]), element, name,
                             name, None])
            index += 1
    return rows


def _ligand_rows(atoms, complex_id):
    return [
        [f"{complex_id}_ligand", 0, f"{complex_id}_ligand", f"{complex_id}_ligand", "LIG", "", "",
         0, "", "LIG", "", 1, 0, float(x), float(y), float(z), element, f"{element}{i + 1}",
         f"{element}{i + 1}", 0]
        for i, (element, (x, y, z)) in enumerate(atoms)
    ]


def lba_item(complex_id, chains, ligand_atoms, target):
    return {
        "id": complex_id,
        "scores": {"neglog_aff": float(target)},
        "atoms_protein": _frame(_protein_rows(chains)),
        "atoms_ligand": _frame(_ligand_rows(ligand_atoms, complex_id)),
        "atoms_pocket": _frame([]),
        "seq": [], "smiles": "C",
    }


def simple_item(complex_id="test", target=6.5, ligand=None, resnames=("ALA", "SER", "LEU", "VAL", "ALA")):
    """Five residues along x; the ligand sits above residue index 2."""
    ligand = ligand or [("C", (3.8 * 2, 0.0, 3.0)), ("N", (3.8 * 2 + 1.4, 0.0, 3.0))]
    return lba_item(complex_id, {"A": [(name, ()) for name in resnames]}, ligand, target)


def brute_force_pocket(item, cutoff):
    """Independent 6 A whole-residue selection, straight from the raw rows."""
    columns = {name: i for i, name in enumerate(ATOM_COLUMNS)}
    ligand = np.array([[row[columns[a]] for a in "xyz"] for row in item["atoms_ligand"]["data"]])
    groups = {}
    for row in item["atoms_protein"]["data"]:
        key = (row[columns["chain"]], row[columns["residue"]])
        groups.setdefault(key, []).append([row[columns[a]] for a in "xyz"])
    kept = set()
    for key, positions in groups.items():
        best = min(
            math.dist(p, q) for p in positions for q in ligand
        )
        if best <= cutoff:
            kept.add(key)
    return kept


# ---------------------------------------------------------------------------
# Pocket selection
# ---------------------------------------------------------------------------

def test_pocket_is_whole_residues_within_six_angstrom():
    item = simple_item()
    ligand, contact = ds.read_ligand(item)
    residues, _ = ds.read_protein_residues(item, mse_to_met=True)
    kept, report = ds.select_pocket(residues, ligand, cutoff=6.0, max_atoms=600,
                                    ligand_contact=contact, contact_atoms="all")
    selected = {(r.chain, r.resseq) for r in kept}
    assert selected == brute_force_pocket(item, 6.0)
    assert 0 < len(selected) < 5, "the synthetic ligand must not touch every residue"
    # whole residues: every atom of a selected residue is kept
    columns = {name: i for i, name in enumerate(ATOM_COLUMNS)}
    totals = {}
    for row in item["atoms_protein"]["data"]:
        totals[(row[columns["chain"]], row[columns["residue"]])] = totals.get(
            (row[columns["chain"]], row[columns["residue"]]), 0) + 1
    for residue in kept:
        assert residue.heavy_atoms == totals[(residue.chain, residue.resseq)]
    assert report["residues_dropped_by_cap"] == 0
    assert report["pocket_residues_at_cutoff"] == len(kept)


def test_pocket_cutoff_is_honoured():
    item = simple_item()
    ligand, contact = ds.read_ligand(item)
    residues, _ = ds.read_protein_residues(item, mse_to_met=True)
    narrow, _ = ds.select_pocket(residues, ligand, cutoff=3.5, max_atoms=600,
                                 ligand_contact=contact, contact_atoms="all")
    wide, _ = ds.select_pocket(residues, ligand, cutoff=12.0, max_atoms=600,
                               ligand_contact=contact, contact_atoms="all")
    assert {(r.chain, r.resseq) for r in narrow} <= {(r.chain, r.resseq) for r in wide}
    assert len(narrow) < len(wide) == 5


def test_hydrogens_count_for_contact_but_never_become_nodes():
    """ATOM3D built atoms_pocket from hydrogen-bearing frames; contact_atoms=all matches it."""
    item = simple_item()
    # One hydrogen sticking out of the last residue, and a ligand atom 2.5 A from
    # it but ~10 A from every heavy atom of the complex.
    hydrogen = (3.8 * 4, -0.6, 7.5)
    item["atoms_protein"]["data"].append(
        [None, 0, None, 1, "A", " ", " ", 5, " ", "ALA", " ", 1.0, 20.0, *hydrogen, "H", "HB1",
         "HB1", None])
    item["atoms_ligand"] = _frame(_ligand_rows([("C", (3.8 * 4, -0.6, 10.0))], "test"))
    atoms, contact = ds.read_ligand(item)
    residues, counts = ds.read_protein_residues(item, mse_to_met=True)
    assert counts["hydrogens_skipped"] == 1
    assert all("HB1" not in r.atoms for r in residues)
    with_h, _ = ds.select_pocket(residues, atoms, cutoff=3.0, max_atoms=600,
                                 ligand_contact=contact, contact_atoms="all")
    heavy_only, _ = ds.select_pocket(residues, atoms, cutoff=3.0, max_atoms=600,
                                     ligand_contact=contact, contact_atoms="heavy")
    assert {(r.chain, r.resseq) for r in with_h} == {("A", 5)}
    assert heavy_only == []


def test_atom_cap_drops_whole_residues_farthest_first_and_is_deterministic():
    item = simple_item()
    ligand, contact = ds.read_ligand(item)
    residues, _ = ds.read_protein_residues(item, mse_to_met=True)
    full, _ = ds.select_pocket(residues, ligand, cutoff=20.0, max_atoms=600,
                               ligand_contact=contact, contact_atoms="all")
    assert len(full) == 5
    budget = sum(r.heavy_atoms for r in full[:-1]) + len(ligand.elements)
    capped, report = ds.select_pocket(residues, ligand, cutoff=20.0, max_atoms=budget,
                                      ligand_contact=contact, contact_atoms="all")
    assert report["residues_dropped_by_cap"] >= 1
    assert sum(r.heavy_atoms for r in capped) + len(ligand.elements) <= budget
    farthest = max(full, key=lambda r: r.min_ligand_distance)
    assert (farthest.chain, farthest.resseq) not in {(r.chain, r.resseq) for r in capped}
    # deterministic for any input order, and residues are never split
    shuffled, _ = ds.read_protein_residues(item, mse_to_met=True)
    shuffled = list(reversed(shuffled))
    again, _ = ds.select_pocket(shuffled, ligand, cutoff=20.0, max_atoms=budget,
                                ligand_contact=contact, contact_atoms="all")
    assert [(r.chain, r.resseq) for r in again] == [(r.chain, r.resseq) for r in capped]
    for residue in capped:
        assert residue.heavy_atoms == len(residue.atoms)


# ---------------------------------------------------------------------------
# Graph contract
# ---------------------------------------------------------------------------

def _tiny_config():
    values = load_config_yaml(os.path.join(REPO, "core", "pretraining_config_multidomain_1m.yaml"))
    layers = len(values["layer_types"])
    values.update(hidden_dims=[32] * layers, num_heads=[4] * layers, graph_dim=32,
                  mlp_hidden_size=64, gaussian_kernels=16)
    return create_pretraining_config(**values)


def _built(item=None):
    item = item or simple_item()
    result = ds.process_item(item, SETTINGS)
    assert result.status == "ok", result.reason
    return result


def _schema():
    return cg.complex_edge_free_schema(3, {"builder": "tests"})


def test_protein_part_matches_the_ec_graph_builder_for_the_same_residues():
    item = simple_item()
    ligand, contact = ds.read_ligand(item)
    residues, _ = ds.read_protein_residues(item, mse_to_met=True)
    pocket, _ = ds.select_pocket(residues, ligand, cutoff=6.0, max_atoms=600,
                                 ligand_contact=contact, contact_atoms="all")
    records = [ResidueRecord(r.resname, r.source_resname, i, "", dict(r.atoms))
               for i, r in enumerate(pocket)]
    outcome = apply_policy(ChainRecord("test", "pocket", records, {}), parse_policy(POLICY))
    reference = gb.build_node_sample(outcome.residues, outcome.masked, "test", sequence_index=None)

    result = _built(item)
    config = _tiny_config()
    sample = cg.joint_runtime_sample(result.sample, cg.ligand_from_stored(result.ligand),
                                     _schema(), config, seed=0)
    protein = sample.entity_idx.long() == cg.PROTEIN_ENTITY
    assert int(protein.sum()) == int(reference.z.numel())
    assert torch.equal(sample.z[protein], reference.z)
    assert torch.allclose(sample.pos[protein], reference.pos)
    assert list(sample.pos_code)[:int(protein.sum())] == list(reference.pos_code)
    # 14 slots per residue, V id 120 exactly on V position codes
    assert int(protein.sum()) % gb.SLOTS_PER_RESIDUE == 0
    virtual = torch.tensor([code.startswith("V") and code[1:].isdigit()
                            for code in list(sample.pos_code)[:int(protein.sum())]])
    assert torch.equal(virtual, sample.z[protein] == VIRTUAL_ATOMIC_NUMBER)


def test_ligand_nodes_carry_their_elements_and_claim_no_protein_slot():
    from core.molecular_encoder import build_domain_local_slot_ids
    from torch_geometric.data import Batch

    item = simple_item(ligand=[("C", (7.6, 0.0, 3.0)), ("N", (9.0, 0.0, 3.0)),
                               ("Cl", (7.6, 1.4, 3.0))])
    result = _built(item)
    config = _tiny_config()
    sample = cg.joint_runtime_sample(result.sample, cg.ligand_from_stored(result.ligand),
                                     _schema(), config, seed=0)
    ligand = sample.entity_idx.long() == cg.LIGAND_ENTITY
    assert int(ligand.sum()) == 3
    assert sample.z[ligand].tolist() == [6, 7, 17]
    assert [code for code, flag in zip(sample.pos_code, ligand.tolist()) if flag] == ["sm"] * 3
    # one block per ligand atom, far above every protein block id
    assert len(set(sample.block_idx[ligand].tolist())) == 3
    assert int(sample.block_idx[ligand].min()) >= cg.LIGAND_BLOCK_OFFSET
    # the domain-local slot embedding must not treat ligand atoms as protein slots
    batch = Batch.from_data_list([sample])
    slot_ids = build_domain_local_slot_ids(batch, batch.z.numel(), batch.z.device)
    assert int(slot_ids[ligand].abs().sum()) == 0
    assert set(slot_ids[~ligand].tolist()) == set(range(1, gb.SLOTS_PER_RESIDUE + 1))
    # ligand atoms are never CA anchors, so they cannot create Tier-1 or random edges
    assert int((sample.template_anchor_order[ligand] >= 0).sum()) == 0


def test_block_ids_are_sequence_faithful_across_chains():
    item = lba_item(
        "two", {"A": [("ALA", ()), ("SER", ())], "B": [("LEU", ()), ("VAL", ())]},
        [("C", (0.0, 0.0, 3.0))], 7.0)
    residues, _ = ds.read_protein_residues(item, mse_to_met=True)
    ordinals = [r.chain_ordinal for r in residues]
    indices = [r.sequence_index for r in residues]
    assert ordinals == [0, 0, 1, 1] and indices == [0, 1, 0, 1]
    ids = cg.protein_block_ids(ordinals, indices).tolist()
    assert ids == [0, 1, cg.PROTEIN_CHAIN_STRIDE, cg.PROTEIN_CHAIN_STRIDE + 1]
    # two residues in different chains are never sequence neighbours
    assert abs(ids[1] - ids[2]) > 3


def test_cross_entity_edges_exist_in_the_runtime_graph():
    result = _built()
    config = _tiny_config()
    sample = cg.joint_runtime_sample(result.sample, cg.ligand_from_stored(result.ligand),
                                     _schema(), config, seed=0)
    total, cross = cg.measured_physical_edges(sample, config)
    assert total > 0
    assert cross > 0, "a joint complex graph without protein-ligand edges is meaningless"
    # The stored edges are CA-only (Tier-1 sequence plus random long-range), so
    # every interface edge has to come from the runtime radius rebuild.
    assert cg.cross_entity_edge_count(sample, sample.edge_index) == 0
    # Every ligand atom must actually be connected to the pocket.
    from torch_geometric.data import Batch

    from core.virtual_slot_graph import build_geometry_neutral_graph_view

    batch = Batch.from_data_list([sample])
    batch.original_types = batch.z.clone()
    batch.mlm_mask = batch.downstream_masked_block.bool().clone()
    edges = build_geometry_neutral_graph_view(batch, config).physical_edge_index
    entity = batch.entity_idx.long()
    crossing = edges[:, entity[edges[0]] != entity[edges[1]]]
    touched = set(crossing[0].tolist()) | set(crossing[1].tolist())
    ligand_nodes = set(torch.where(entity == cg.LIGAND_ENTITY)[0].tolist())
    assert ligand_nodes <= touched, "some ligand atom has no edge to the pocket"


def test_runtime_sample_is_deterministic_for_one_seed():
    result = _built()
    config = _tiny_config()
    first = cg.joint_runtime_sample(result.sample, cg.ligand_from_stored(result.ligand),
                                    _schema(), config, seed=0)
    second = cg.joint_runtime_sample(result.sample, cg.ligand_from_stored(result.ligand),
                                     _schema(), config, seed=0)
    assert torch.equal(first.edge_index, second.edge_index)
    assert torch.equal(first.z, second.z)
    assert torch.allclose(first.pos, second.pos)


def test_separate_representation_builds_two_graphs():
    result = _built()
    config = _tiny_config()
    pocket = cg.pocket_runtime_sample(result.sample, _schema(), config, seed=0)
    ligand = cg.ligand_runtime_sample(cg.ligand_from_stored(result.ligand), "test#ligand",
                                      config, seed=0)
    assert str(pocket.dataset_type) == "protein"
    assert str(ligand.dataset_type) == cg.LIGAND_DATASET_TYPE
    assert int(ligand.z.numel()) == len(result.ligand["elements"])
    assert cg.cross_entity_edge_count(pocket, pocket.edge_index) == 0
    joint = cg.joint_runtime_sample(result.sample, cg.ligand_from_stored(result.ligand),
                                    _schema(), config, seed=0)
    protein = joint.entity_idx.long() == cg.PROTEIN_ENTITY
    assert torch.equal(pocket.z, joint.z[protein])


def test_no_explicit_bonds_are_attached():
    """Explicit bonds would zero the pocket's chemical-coordination feature."""
    result = _built()
    sample = cg.joint_runtime_sample(result.sample, cg.ligand_from_stored(result.ligand),
                                     _schema(), _tiny_config(), seed=0)
    assert int(sample.bond_edge_index.numel()) == 0
    assert not bool(sample.has_explicit_bonds.any())


def test_explicit_ligand_bonds_are_refused_by_the_config_loader(tmp_path):
    import yaml

    config = yaml.safe_load(ds.TASK_YAML.read_text())
    config["ligand"]["bonds"] = "explicit"
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="has_explicit_bonds"):
        ds.load_task_config(path)


# ---------------------------------------------------------------------------
# Processing and the stored split
# ---------------------------------------------------------------------------

def test_process_item_reports_and_drops_loudly():
    assert ds.process_item(simple_item(), SETTINGS).status == "ok"
    far = simple_item(ligand=[("C", (0.0, 0.0, 500.0))])
    assert ds.process_item(far, SETTINGS).status == "dropped"
    broken = simple_item()
    del broken["scores"]
    result = ds.process_item(broken, SETTINGS)
    assert result.status == "failed" and result.reason.startswith("parse_error")


def test_missing_backbone_residue_is_dropped_and_recorded():
    item = lba_item("drop", {"A": [("ALA", ()), ("SER", ("CA",)), ("LEU", ()), ("VAL", ())]},
                    [("C", (3.8, 0.0, 3.0))], 5.0)
    result = ds.process_item(item, SETTINGS)
    assert result.status == "ok"
    assert result.report["residues_missing_backbone"] == 1


def test_incomplete_sidechain_is_masked_like_a_pretraining_block():
    item = lba_item("mask", {"A": [("ALA", ()), ("LEU", ("CD1",)), ("VAL", ()), ("SER", ())]},
                    [("C", (3.8, 0.0, 3.0))], 5.0)
    result = ds.process_item(item, SETTINGS)
    assert result.status == "ok"
    assert result.report["masked_residues"] == 1
    flags = result.sample.downstream_masked_block.bool()
    assert int(flags.sum()) == gb.SLOTS_PER_RESIDUE


def test_processed_split_round_trip(tmp_path):
    import copy

    config = ds.load_task_config(data_root=tmp_path)
    model_config = _tiny_config()
    results = [_built(simple_item(f"c{i}", target=5.0 + i)) for i in range(3)]
    payload = {
        "schema": _schema(), "policy": "dropbb_masksc", "split": "val", "split_tag": "identity30",
        "ids": [r.complex_id for r in results],
        "targets": torch.tensor([r.target for r in results], dtype=torch.float32),
        "samples": [r.sample for r in results], "ligands": [r.ligand for r in results],
        "sequences": [r.sequence for r in results], "qc": {},
    }
    path = tmp_path / "val.pt"
    torch.save(payload, path)

    joint = ds.ProcessedSplit(path)
    assert len(joint) == 3 and joint.ids == ["c0", "c1", "c2"]
    assert np.allclose(joint.targets, [5.0, 6.0, 7.0])
    assert joint.labels.shape == (3, 1)
    graphs = [sample for _, sample in joint.iter_runtime(model_config, seed=0)]
    assert len(graphs) == 3
    assert joint.chain_embeddings(np.zeros((3, 4), dtype=np.float32)).shape == (3, 4)

    separate = ds.ProcessedSplit(path, representation="separate")
    assert len(separate) == 6
    kinds = [str(separate.runtime(i, model_config, seed=0).dataset_type) for i in range(6)]
    assert kinds == ["protein"] * 3 + [cg.LIGAND_DATASET_TYPE] * 3
    stacked = np.arange(12, dtype=np.float32).reshape(6, 2)
    merged = separate.chain_embeddings(stacked)
    assert merged.shape == (3, 4)
    assert np.allclose(merged[0], [0, 1, 6, 7])
    assert copy.copy(separate.representation) == "separate"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_regression_metrics_against_hand_computed_values():
    targets = np.array([1.0, 2.0, 3.0, 4.0])
    predictions = np.array([1.5, 1.5, 3.5, 5.0])
    assert rmse(targets, predictions) == pytest.approx(math.sqrt((0.25 + 0.25 + 0.25 + 1.0) / 4))
    assert evaluate_regression(targets, predictions)["mae"] == pytest.approx((0.5 + 0.5 + 0.5 + 1.0) / 4)
    # Pearson of these two vectors, computed by hand
    tx, px = targets - targets.mean(), predictions - predictions.mean()
    expected = float((tx * px).sum() / math.sqrt((tx ** 2).sum() * (px ** 2).sum()))
    assert pearson_r(targets, predictions) == pytest.approx(expected)
    # ranks 1,2,3,4 vs 1.5,1.5,3,4 -> Spearman below 1 but high
    assert 0.9 < spearman_rho(targets, predictions) < 1.0
    assert spearman_rho(targets, targets) == pytest.approx(1.0)
    assert rmse(targets, targets) == pytest.approx(0.0)


def test_regression_metrics_fail_loud():
    with pytest.raises(ValueError, match="NaN"):
        rmse(np.array([1.0, 2.0]), np.array([1.0, float("nan")]))
    with pytest.raises(ValueError, match="differ"):
        rmse(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="constant"):
        pearson_r(np.array([1.0, 2.0]), np.array([3.0, 3.0]))
    with pytest.raises(ValueError, match="at least one sample"):
        rmse(np.array([]), np.array([]))


def test_validation_score_direction_differs_per_task():
    from downstream.probe import validation_score

    targets = np.array([1.0, 2.0, 3.0])
    good = validation_score("regression", targets, np.array([1.0, 2.0, 3.1]))
    bad = validation_score("regression", targets, np.array([5.0, 9.0, 0.0]))
    assert good > bad, "early stopping must prefer the lower RMSE"
    assert good == pytest.approx(-rmse(targets, np.array([1.0, 2.0, 3.1])))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def test_complex_poolings_on_hand_built_tensors():
    from types import SimpleNamespace

    batch = SimpleNamespace(
        num_graphs=1, z=torch.zeros(5, dtype=torch.long), batch=torch.zeros(5, dtype=torch.long),
        block_idx=torch.tensor([0, 0, 1, 2, 3]),
        template_anchor_order=torch.tensor([1, -1, -1, -1, -1]),
        entity_idx=torch.tensor([0, 0, 0, 1, 1]),
    )
    states = torch.tensor([[1.0], [3.0], [5.0], [7.0], [11.0]])
    mask = torch.ones(5, dtype=torch.bool)
    pooled = enc.pool(states, batch, mask, enc.COMPLEX_POOLINGS)
    assert pooled["mean_pocket"].item() == pytest.approx(3.0)
    assert pooled["mean_ligand"].item() == pytest.approx(9.0)
    assert pooled["concat_pocket_ligand"].tolist() == [[3.0, 9.0]]
    with pytest.raises(ValueError, match="no unmasked real atom"):
        enc.pool(states, batch, torch.tensor([True, True, True, False, False]), ["mean_ligand"])
    stripped = SimpleNamespace(**{k: v for k, v in vars(batch).items() if k != "entity_idx"})
    with pytest.raises(ValueError, match="entity_idx"):
        enc.pool(states, stripped, mask, ["mean_pocket"])


def _tiny_checkpoint(path, seed=0):
    torch.manual_seed(seed)
    model = PretrainingESAModel(_tiny_config())
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": pl.__version__, "epoch": 0, "global_step": 0}, path)
    return model


def test_complex_graph_survives_forward_and_embeds_deterministically(cpu_flash, tmp_path):
    _tiny_checkpoint(tmp_path / "tiny.ckpt")
    model = enc.load_mind_encoder(str(tmp_path / "tiny.ckpt"), "cpu")
    samples = []
    for index in range(2):
        result = _built(simple_item(f"c{index}", target=5.0 + index))
        samples.append(cg.joint_runtime_sample(result.sample, cg.ligand_from_stored(result.ligand),
                                               _schema(), model.config, seed=0))
    modes = list(enc.POOLINGS) + list(enc.COMPLEX_POOLINGS)
    first = enc.embed_batch(model, samples, "cpu", modes)
    second = enc.embed_batch(model, samples, "cpu", modes)
    width = int(model.config.hidden_dims[-1])
    for mode in modes:
        assert first[mode].shape[0] == 2
        assert torch.equal(first[mode], second[mode]), mode
        assert torch.isfinite(first[mode]).all(), mode
    assert first["concat_pocket_ligand"].shape[1] == 2 * first["mean_pocket"].shape[1]
    assert first["mean_atoms"].shape[1] == width


def test_forward_uses_every_ligand_atom(cpu_flash, tmp_path):
    """The pooled complex embedding must change when the ligand changes."""
    _tiny_checkpoint(tmp_path / "tiny.ckpt")
    model = enc.load_mind_encoder(str(tmp_path / "tiny.ckpt"), "cpu")
    base = _built(simple_item("a"))
    moved = _built(simple_item("a", ligand=[("C", (3.8 * 2, 0.0, 3.4)), ("O", (3.8 * 2 + 1.4, 0.0, 3.0))]))
    graphs = [
        cg.joint_runtime_sample(result.sample, cg.ligand_from_stored(result.ligand),
                                _schema(), model.config, seed=0)
        for result in (base, moved)
    ]
    pooled = [enc.embed_batch(model, [graph], "cpu", ["mean_pocket"])["mean_pocket"] for graph in graphs]
    assert not torch.allclose(pooled[0], pooled[1]), \
        "the pocket embedding ignores the ligand; the interface edges are not reaching the encoder"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_report_renders_the_regression_table(tmp_path):
    import json

    from downstream import report

    base = tmp_path / "downstream" / "lba" / "snap" / "mlp_probe.identity30.joint.concat_pocket_ligand"
    for seed, value in ((0, 1.40), (1, 1.50)):
        directory = base / f"seed{seed}"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps({
            "task": "lba", "kind": "regression", "seed": seed,
            "val": {"rmse": value + 0.1},
            "test": {"all": {"rmse": value, "mae": value - 0.2, "pearson_r": 0.6,
                             "spearman_rho": 0.55}},
        }))
    rows = report.collect("lba", tmp_path)
    assert len(rows) == 1
    assert rows[0]["test_rmse"]["mean"] == pytest.approx(1.45)
    assert rows[0]["seeds"] == [0, 1]
    markdown = report.to_markdown("lba", rows)
    assert "test RMSE" in markdown and "Spearman rho" in markdown
    assert "1.450" in markdown
    assert "ATOM3D 3DCNN" in markdown and "1.416" in markdown
    assert "Fmax" not in markdown


# ---------------------------------------------------------------------------
# Results layout: LBA qualifies the protocol with split tag and representation
# ---------------------------------------------------------------------------

def test_results_dir_accepts_qualified_protocols_and_rejects_bad_ones(tmp_path):
    from downstream.common.registry import results_dir

    path = results_dir("lba", "m", "mlp_probe.identity30.joint", 0,
                       pooling="concat_pocket_ligand", root=tmp_path)
    assert path.parent.name == "mlp_probe.identity30.joint.concat_pocket_ligand"
    assert path.name == "seed0"
    # EC's unqualified form still works.
    assert results_dir("ec", "m", "mlp_probe", 1, pooling="mean_atoms",
                       root=tmp_path).parent.name == "mlp_probe.mean_atoms"
    # The leading component must still name a real protocol.
    with pytest.raises(ValueError, match="must start with"):
        results_dir("lba", "m", "bogus.identity30.joint", 0, pooling="p", root=tmp_path)
    # Qualifiers are name-checked, so a path separator cannot slip in.
    with pytest.raises(ValueError):
        results_dir("lba", "m", "mlp_probe.../etc", 0, pooling="p", root=tmp_path)


def test_lba_probe_metrics_name_the_split_tag_and_representation_even_when_defaulted(tmp_path):
    import json

    from downstream import probe

    rng = np.random.default_rng(0)
    outputs = tmp_path / "out"
    for split, count in {"train": 30, "val": 10, "test": 10}.items():
        features = rng.normal(size=(count, 4)).astype(np.float32)
        labels = (features[:, :1] * 2 + 6.0).astype(np.float32)
        path = outputs / "downstream" / "lba" / "toy" / "embeddings" / f"identity30.joint.{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, chains=np.array([f"{split}{i}" for i in range(count)]), labels=labels,
                 emb_concat_pocket_ligand=features)
        path.with_suffix(".json").write_text(json.dumps({"split": split}))
    args = ["--task", "lba", "--model-id", "toy", "--head", "linear", "--seeds", "0", "--pooling",
            "concat_pocket_ligand", "--data-root", str(tmp_path / "data"), "--outputs-root", str(outputs)]
    assert probe.main(args) == 0
    metrics = json.loads((outputs / "downstream" / "lba" / "toy" /
                          "linear_probe.identity30.joint.concat_pocket_ligand" / "seed0" / "metrics.json").read_text())
    assert metrics["split_tag"] == "identity30" and metrics["representation"] == "joint"

"""graph_builder output contract against the training path, and the incomplete-residue guard."""

import contextlib
import io
import os

import pytest
import torch

from core.pretraining_config import create_pretraining_config, load_config_yaml
from core.virtual_slot_graph import build_geometry_neutral_graph_view
from data_loading.adapters.protein_adapter import ProteinAdapter
from data_loading.atom_position_vocab import (
    PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE, VIRTUAL_ATOMIC_NUMBER,
)
from data_loading.data_types import UniversalAtom
from downstream.common import graph_builder as gb
from downstream.common.encoder import collate, prepare_encoder_batch
from downstream.common.missing_atoms import apply_policy, parse_policy
from downstream.common.structure_io import read_chains
from downstream.tasks.ec.dataset import windows
from downstream_helpers import chain_of, pdb_text, residue_atoms

MASK = parse_policy({"policy": "drop_residue_missing_backbone", "incomplete_sidechain": "mask"})
ALL_RESIDUES = sorted(PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def model_config():
    return create_pretraining_config(**load_config_yaml(os.path.join(REPO, "core", "pretraining_config_multidomain_1m.yaml")))


def _write(tmp_path, name, residues):
    path = tmp_path / f"{name}.pdb"
    path.write_text(pdb_text({"A": residues}))
    return path


def _training_node_sample(path):
    sample, reason = _training_build(path)
    assert sample is not None, reason
    return sample


def _training_build(path):
    """``build_edge_free_chunks.build_node_sample`` without leaking its CUDA_VISIBLE_DEVICES side effect."""
    saved_env, saved_threads = os.environ.get("CUDA_VISIBLE_DEVICES"), torch.get_num_threads()
    try:
        from data_loading import build_edge_free_chunks as builder

        builder._init_worker("protein")
        _, sample, reason = builder.build_node_sample(str(path))
    finally:
        if saved_env is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_env
        torch.set_num_threads(saved_threads)
    return sample, reason


def _downstream_node(path):
    chain = read_chains(path, mse_to_met=True)["A"]
    result = apply_policy(chain, MASK)
    return gb.build_node_sample(result.residues, result.masked, path.stem), result


def test_complete_residues_match_the_training_builder_exactly(tmp_path):
    path = _write(tmp_path, "all20", chain_of(ALL_RESIDUES))
    expected = _training_node_sample(path)
    node, _ = _downstream_node(path)
    assert node.pos_code == expected.pos_code
    assert node.block_symbols == expected.block_symbols
    for name in ("z", "pos", "block_idx", "entity_idx"):
        assert torch.equal(getattr(node, name), getattr(expected, name)), name
    assert node.dataset_type == expected.dataset_type == "protein"
    assert not bool(node.downstream_masked_block.any())


def test_fixed_slot_contract_14_slots_and_v_id(tmp_path):
    node, _ = _downstream_node(_write(tmp_path, "c", chain_of(["GLY", "ALA", "TRP"])))
    assert node.z.numel() == 3 * 14
    assert node.pos_code[:4] == ["N", "CA", "C", "O"] and node.pos_code[4:14] == [f"V{i}" for i in range(1, 11)]
    assert int((node.z[4:14] == VIRTUAL_ATOMIC_NUMBER).sum()) == 10  # GLY: all sidechain slots V
    assert int((node.z[28:42] == VIRTUAL_ATOMIC_NUMBER).sum()) == 0   # TRP fills all ten


def test_adapter_silent_shift_is_caught_by_the_downstream_path(tmp_path):
    path = _write(tmp_path, "lys", chain_of(["LYS"], drops={0: ("CD",)}))
    with contextlib.redirect_stdout(io.StringIO()):
        adapter = ProteinAdapter()
    # The raw-file entry (pretraining builders) refuses the residue ...
    with pytest.raises(ValueError, match=r"missing \['CD'\]"):
        adapter.convert_to_universal(path)
    # ... and the pretraining builder records it as a failed structure instead of a sample.
    sample, reason = _training_build(path)
    assert sample is None and "missing ['CD']" in reason

    # The 14-slot routine itself still packs CE into the CD slot and pads with V;
    # the downstream path calls it only for residues the policy has masked.
    atoms = [UniversalAtom(element=e, position=xyz, pos_code=n, block_idx=0, atom_idx_in_block=i, entity_idx=0)
             for i, (n, e, xyz) in enumerate(residue_atoms("LYS", 0, drop=("CD",)))]
    codes = [atom.pos_code for atom in adapter._standardize_residue_to_14_atoms(atoms, "LYS", 0)]
    assert len(codes) == 14 and codes[6] == "CE" and codes[8] == "V5"

    chain = read_chains(path, mse_to_met=True)["A"]
    with pytest.raises(ValueError, match="incomplete sidechain .* is not masked"):
        gb.build_node_sample(chain.residues, [False], "lys")
    node, result = _downstream_node(path)
    assert result.masked == [True]
    assert bool(node.downstream_masked_block.all())


def test_missing_backbone_never_reaches_the_builder(tmp_path):
    chain = read_chains(_write(tmp_path, "bb", chain_of(["ALA", "SER"], drops={1: ("O",)})), mse_to_met=True)["A"]
    with pytest.raises(ValueError, match="missing backbone"):
        gb.build_node_sample(chain.residues, [False, False], "bb")


def test_runtime_sample_is_deterministic_and_leaves_global_rng_alone(tmp_path, model_config):
    node, _ = _downstream_node(_write(tmp_path, "long", chain_of(["ALA"] * 30)))
    compact = gb.compact_sample(node)
    schema = gb.edge_free_schema(3, {})
    state = torch.random.get_rng_state()
    first = gb.runtime_sample(compact, schema, model_config, seed=0)
    second = gb.runtime_sample(compact, schema, model_config, seed=0)
    other = gb.runtime_sample(compact, schema, model_config, seed=1)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(first.edge_index, second.edge_index)
    assert not torch.equal(first.edge_index, other.edge_index)
    assert torch.equal(first.z, node.z) and first.pos_code == node.pos_code
    assert int(first.template_reason_bits.abs().sum()) == 0


def test_schema_mismatch_with_model_config_fails_loudly(tmp_path, model_config):
    node, _ = _downstream_node(_write(tmp_path, "k", chain_of(["ALA", "GLY"])))
    with pytest.raises(ValueError, match="sequence_neighbors_k"):
        gb.runtime_sample(gb.compact_sample(node), gb.edge_free_schema(5, {}), model_config, seed=0)


def test_masked_residue_is_a_latent_block_in_the_geometry_neutral_view(tmp_path, model_config):
    node_a, _ = _downstream_node(_write(tmp_path, "a", chain_of(["ALA", "LYS", "TRP"], drops={1: ("NZ",)})))
    node_b, _ = _downstream_node(_write(tmp_path, "b", chain_of(["GLY", "SER"])))
    schema = gb.edge_free_schema(3, {})
    samples = [gb.runtime_sample(gb.compact_sample(n), schema, model_config, seed=0) for n in (node_a, node_b)]
    batch = prepare_encoder_batch(collate(samples))
    view = build_geometry_neutral_graph_view(batch, model_config)

    lys = slice(14, 28)
    assert bool(batch.mlm_mask[lys].all()) and int(batch.mlm_mask.sum()) == 14
    assert bool((batch.z[lys] == gb.MASK_TOKEN).all())
    assert bool(view.target_slot_mask[18:28].all()) and int(view.target_slot_mask.sum()) == 10
    assert bool(view.active_node_mask[14:18].all())            # masked backbone stays in ESA
    visible_v = (batch.original_types == VIRTUAL_ATOMIC_NUMBER) & ~batch.mlm_mask
    assert not bool(view.active_node_mask[visible_v].any())
    pool_mask = view.active_node_mask & ~batch.mlm_mask
    assert int(pool_mask[lys].sum()) == 0
    assert int(pool_mask.sum()) == int((batch.original_types != VIRTUAL_ATOMIC_NUMBER).sum()) - 8  # LYS without NZ: 8 real atoms


def test_windows_cover_the_chain_contiguously():
    spans = windows(2500, 1024)
    assert spans[0][0] == 0 and spans[-1][1] == 2500 and len(spans) == 3
    assert all(b - a <= 1024 for a, b in spans) and all(spans[i][1] == spans[i + 1][0] for i in range(2))

"""Encoder loading, pooling and embedding determinism.

CPU tests run the production flash_varlen layers through the ``cpu_flash``
SDPA stand-in (tests/conftest.py) on a tiny model; the masked-residue forward
(latent slot readout under CUDA bf16 autocast) needs a GPU and is skipped
without one.
"""

import os
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch

from core.deterministic import enable_deterministic_algorithms
from core.pretraining_config import create_pretraining_config, load_config_yaml
from core.pretraining_model import PretrainingESAModel
from downstream.common import encoder as enc
from downstream.common import graph_builder as gb
from downstream.common.missing_atoms import apply_policy, parse_policy
from downstream.common.structure_io import read_chains
from downstream_helpers import chain_of, pdb_text

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK = parse_policy({"policy": "drop_residue_missing_backbone", "incomplete_sidechain": "mask"})


def _tiny_config():
    values = load_config_yaml(os.path.join(REPO, "core", "pretraining_config_multidomain_1m.yaml"))
    layers = len(values["layer_types"])
    values.update(hidden_dims=[32] * layers, num_heads=[4] * layers, graph_dim=32,
                  mlp_hidden_size=64, gaussian_kernels=16)
    return create_pretraining_config(**values)


def _tiny_checkpoint(path, seed=0):
    torch.manual_seed(seed)
    model = PretrainingESAModel(_tiny_config())
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": pl.__version__, "epoch": 0, "global_step": 0}, path)
    return model


def _samples(tmp_path, config, chains):
    out = []
    for name, (resnames, drops) in chains.items():
        path = tmp_path / f"{name}.pdb"
        path.write_text(pdb_text({"A": chain_of(resnames, drops=drops)}))
        result = apply_policy(read_chains(path, mse_to_met=True)["A"], MASK)
        node = gb.build_node_sample(result.residues, result.masked, name)
        out.append(gb.runtime_sample(gb.compact_sample(node), gb.edge_free_schema(3, {}), config, seed=0))
    return out


def test_pooling_modes_on_hand_built_tensors():
    # one graph, two residues of three nodes; node 2 masked out of pooling, node 0 and 3 are CA
    batch = SimpleNamespace(num_graphs=1, batch=torch.zeros(6, dtype=torch.long),
                            block_idx=torch.tensor([0, 0, 0, 1, 1, 1]),
                            template_anchor_order=torch.tensor([1, 0, -1, 1, 0, -1]))
    states = torch.tensor([[1.0], [3.0], [100.0], [5.0], [7.0], [9.0]])
    mask = torch.tensor([True, True, False, True, True, True])
    pooled = enc.pool(states, batch, mask, enc.POOLINGS)
    assert pooled["mean_atoms"].item() == pytest.approx((1 + 3 + 5 + 7 + 9) / 5)
    assert pooled["ca"].item() == pytest.approx(3.0)
    assert pooled["mean_residues"].item() == pytest.approx((2.0 + 7.0) / 2)
    with pytest.raises(ValueError, match="no unmasked real atom"):
        enc.pool(states, batch, torch.zeros(6, dtype=torch.bool), ["mean_atoms"])


def test_load_strict_and_random_init_and_deterministic_embeddings(cpu_flash, tmp_path):
    reference = _tiny_checkpoint(tmp_path / "tiny.ckpt", seed=0)
    loaded = enc.load_mind_encoder(str(tmp_path / "tiny.ckpt"), "cpu")
    for key, value in reference.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key
    random_a = enc.load_mind_encoder(str(tmp_path / "tiny.ckpt"), "cpu", random_init=True, init_seed=1)
    random_b = enc.load_mind_encoder(str(tmp_path / "tiny.ckpt"), "cpu", random_init=True, init_seed=1)
    key = next(k for k in reference.state_dict() if k.endswith("weight") and reference.state_dict()[k].ndim == 2)
    assert torch.equal(random_a.state_dict()[key], random_b.state_dict()[key])
    assert not torch.equal(random_a.state_dict()[key], loaded.state_dict()[key])

    chains = {"c1": (["ALA"] * 16, {}), "c2": (["GLY", "TRP", "SER", "LYS"] * 3, {})}
    first = enc.embed_batch(loaded, _samples(tmp_path, loaded.config, chains), "cpu")
    second = enc.embed_batch(loaded, _samples(tmp_path, loaded.config, chains), "cpu")
    for mode in enc.POOLINGS:
        assert first[mode].shape == (2, 32)
        assert torch.isfinite(first[mode]).all()
        assert torch.equal(first[mode], second[mode]), mode
    untrained = enc.embed_batch(random_a, _samples(tmp_path, loaded.config, chains), "cpu")
    assert not torch.equal(untrained["mean_atoms"], first["mean_atoms"])


def test_masked_residue_forward_on_cpu_is_deterministic(cpu_flash, tmp_path):
    _tiny_checkpoint(tmp_path / "tiny.ckpt")
    model = enc.load_mind_encoder(str(tmp_path / "tiny.ckpt"), "cpu")
    chains = {"m": (["ALA", "LYS", "TRP", "SER"] * 4, {1: ("NZ",)})}
    first = enc.embed_batch(model, _samples(tmp_path, model.config, chains), "cpu")
    second = enc.embed_batch(model, _samples(tmp_path, model.config, chains), "cpu")
    for mode in enc.POOLINGS:
        assert torch.isfinite(first[mode]).all(), mode
        assert torch.equal(first[mode], second[mode]), mode


@pytest.fixture
def deterministic_cuda():
    enable_deterministic_algorithms()
    yield
    torch.use_deterministic_algorithms(False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real flash_varlen kernels need CUDA")
def test_masked_residue_forward_on_gpu_is_deterministic(deterministic_cuda, tmp_path):
    _tiny_checkpoint(tmp_path / "tiny.ckpt")
    model = enc.load_mind_encoder(str(tmp_path / "tiny.ckpt"), "cuda")
    chains = {"m": (["ALA", "LYS", "TRP", "SER"] * 4, {1: ("NZ",)})}
    first = enc.embed_batch(model, _samples(tmp_path, model.config, chains), "cuda")
    second = enc.embed_batch(model, _samples(tmp_path, model.config, chains), "cuda")
    for mode in enc.POOLINGS:
        assert torch.equal(first[mode], second[mode]), mode

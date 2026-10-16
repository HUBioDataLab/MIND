"""The flash_varlen path no longer builds the unused dense [B, 1, N, N] adjacency mask.

The real flash kernel is CUDA-only, so these CPU tests replace ``flash_attn_varlen_func`` with
a per-sequence SDPA stand-in; everything else (layers, unpadding, edge-list attention) is the
production code.
"""

import pytest
import torch

import esa.masked_layers as masked_layers
from esa.masked_layers import ESA
from esa.mha_flash_varlen import MABFlashVarlen

DIM = 16


def _esa(**overrides):
    torch.manual_seed(0)
    kwargs = dict(
        num_outputs=1,
        dim_output=DIM,
        dim_hidden=[DIM] * 5,
        num_heads=[2] * 5,
        layer_types=["M", "S", "M", "S", "P"],
        node_or_edge="node",
        xformers_or_torch_attn="flash_varlen",
        pre_or_post="pre",
        norm_type="LN",
        use_mlps=True,
        mlp_hidden_size=32,
        num_mlp_layers=2,
        mlp_type="standard",
        use_bfloat16=True,
    )
    kwargs.update(overrides)
    return ESA(**kwargs).to(torch.bfloat16).eval()


def _batch(seed=0, sizes=(9, 5, 12, 3)):
    """Random production-shaped batch: packed nodes, graph-local edges, every node has an edge."""
    generator = torch.Generator().manual_seed(seed)
    batch_mapping = torch.cat([torch.full((s,), g) for g, s in enumerate(sizes)])
    offsets = torch.cumsum(torch.tensor([0, *sizes[:-1]]), 0)
    sources, targets = [], []
    for g, size in enumerate(sizes):
        for i in range(size):
            sources.append(offsets[g] + i)
            targets.append(offsets[g] + i)  # self loop: no isolated node
            j = int(torch.randint(0, size, (1,), generator=generator))
            sources += [offsets[g] + i, offsets[g] + j]
            targets += [offsets[g] + j, offsets[g] + i]
    edge_index = torch.stack([torch.stack(sources), torch.stack(targets)])
    max_items = ((max(sizes) + 1 + 7) // 8) * 8
    dense = torch.randn(len(sizes), max_items, DIM, generator=generator).to(torch.bfloat16)
    pos = torch.randn(batch_mapping.numel(), 3, generator=generator)
    return dense, edge_index, batch_mapping, max_items, pos


def _run(model, batch, **kwargs):
    dense, edge_index, batch_mapping, max_items, pos = batch
    with torch.no_grad():
        return model(dense, edge_index, batch_mapping, max_items, **kwargs)


def _forbid(monkeypatch):
    def boom(**_):
        raise AssertionError("dense adjacency mask was built on the flash_varlen path")

    monkeypatch.setattr(masked_layers, "get_adj_mask_from_edge_index_node", boom)
    monkeypatch.setattr(masked_layers, "get_adj_mask_from_edge_index_edge", boom)


def test_flash_forward_never_builds_the_dense_mask(monkeypatch, cpu_flash):
    _forbid(monkeypatch)

    graph_embeddings, node_embeddings = _run(_esa(), _batch())

    assert torch.isfinite(graph_embeddings.float()).all()
    assert torch.isfinite(node_embeddings.float()).all()


def test_flash_output_is_unchanged_without_the_unused_mask(monkeypatch, cpu_flash):
    """Compare against the previous behaviour: build the mask on CPU and pass it along."""
    model, batch = _esa(), _batch(seed=3)
    new_graph, new_nodes = _run(model, batch)

    monkeypatch.setattr(ESA, "_flash_varlen_ignores_dense_mask", lambda self: False)
    old_graph, old_nodes = _run(model, batch)

    torch.testing.assert_close(new_nodes, old_nodes, atol=1e-5, rtol=0)
    torch.testing.assert_close(new_graph, old_graph, atol=1e-5, rtol=0)


def test_distance_bias_branch_still_works_without_a_mask(monkeypatch, cpu_flash):
    """use_distance_bias with adj_mask=None takes the existing ``adj_mask is None`` branch."""
    model, batch = _esa(), _batch(seed=5)
    model.use_distance_bias = True
    pos = batch[4]
    new_graph, new_nodes = _run(model, batch, pos=pos)

    monkeypatch.setattr(ESA, "_flash_varlen_ignores_dense_mask", lambda self: False)
    old_graph, old_nodes = _run(model, batch, pos=pos)

    torch.testing.assert_close(new_nodes, old_nodes, atol=1e-5, rtol=0)
    torch.testing.assert_close(new_graph, old_graph, atol=1e-5, rtol=0)


def test_equivariant_flag_keeps_building_the_mask_on_the_input_device(monkeypatch, cpu_flash):
    model, batch = _esa(), _batch(seed=7)
    model.use_equivariant = True  # the equivariant branch still reads the dense mask
    seen = []
    original = masked_layers.get_adj_mask_from_edge_index_node

    def spy(**kwargs):
        seen.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(masked_layers, "get_adj_mask_from_edge_index_node", spy)

    _run(model, batch)

    assert len(seen) == 1
    assert seen[0]["device"] == batch[0].device


def test_mask_helpers_require_a_device_and_follow_it():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    batch_mapping = torch.zeros(3, dtype=torch.long)
    common = dict(edge_index=edge_index, batch_size=1, max_items=4, batch_mapping=batch_mapping)

    with pytest.raises(TypeError, match="device"):
        masked_layers.get_adj_mask_from_edge_index_node(**common)
    with pytest.raises(TypeError, match="device"):
        masked_layers.get_adj_mask_from_edge_index_edge(**common)
    with pytest.raises(TypeError, match="device"):
        masked_layers.compute_distance_bias(torch.zeros(3, 3), batch_mapping, 4)

    node = masked_layers.get_adj_mask_from_edge_index_node(
        **common, device="cpu", use_bfloat16=False
    )
    edge = masked_layers.get_adj_mask_from_edge_index_edge(
        **common, device="cpu", use_bfloat16=False
    )
    bias = masked_layers.compute_distance_bias(
        torch.randn(3, 3), batch_mapping, 4, device="cpu", use_bfloat16=False
    )
    for tensor in (node, edge, bias):
        assert tensor.device.type == "cpu"
        assert tensor.shape == (1, 1, 4, 4)


def test_index_helpers_follow_their_input_device():
    """They used to pin ``cuda:0``, so they could not run without a GPU."""
    first = masked_layers.get_first_unique_index(torch.tensor([0, 0, 1, 1, 1, 2]))
    assert first.tolist() == [0, 2, 5]

    consecutive = masked_layers.generate_consecutive_tensor(torch.tensor([0, 2, 5]), 6)
    assert consecutive.tolist() == [0, 1, 0, 1, 2, 0]
    assert first.device.type == consecutive.device.type == "cpu"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dense_mask_and_sparse_edge_list_attention_agree(seed):
    """Random weights and graphs; every node has >= 1 edge (isolated nodes differ by design)."""
    torch.manual_seed(seed)
    sizes = [7, 4, 10]
    attention = MABFlashVarlen(DIM, DIM, DIM, num_heads=2, dropout_p=0.0).eval()
    dense_x, edge_index, batch_mapping, max_items, _ = _batch(seed, tuple(sizes))
    dense_x = dense_x.float()
    offsets = torch.cumsum(torch.tensor([0, *sizes[:-1]]), 0)
    local = torch.arange(batch_mapping.numel()) - offsets[batch_mapping]
    mask = torch.full((len(sizes), max_items, max_items), -99999.0)
    mask[batch_mapping[edge_index[0]], local[edge_index[0]], local[edge_index[1]]] = 0.0
    # Garbage in the padding rows must not leak into real rows through either path.
    for g, size in enumerate(sizes):
        dense_x[g, size:] = torch.randn(max_items - size, DIM) * 5

    with torch.no_grad():
        dense_out = attention._forward_padded(dense_x, dense_x, mask.unsqueeze(1))
        sparse_out = attention.forward_graph(dense_x, edge_index, batch_mapping)

    real = torch.zeros(len(sizes), max_items, dtype=torch.bool)
    for g, size in enumerate(sizes):
        real[g, :size] = True
    torch.testing.assert_close(dense_out[real], sparse_out[real], atol=1e-5, rtol=0)


def test_removed_attention_backends_are_refused_loudly():
    from core.pretraining_config import create_pretraining_config
    from esa.masked_layers import PMAComplete, SABComplete

    for backend in ("torch", "xformers", "flash"):
        with pytest.raises(ValueError, match="not supported"):
            _esa(xformers_or_torch_attn=backend)
        with pytest.raises(ValueError, match="not supported"):
            SABComplete(8, 8, 2, 0.0, idx=0, norm_type="LN", xformers_or_torch_attn=backend)
        with pytest.raises(ValueError, match="not supported"):
            PMAComplete(8, 2, 1, "LN", xformers_or_torch_attn=backend)
        with pytest.raises(ValueError, match="not supported"):
            create_pretraining_config(xformers_or_torch_attn=backend)


def test_esa_mha_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        __import__("esa.mha")

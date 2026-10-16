"""ESA on packed rows ([T, D], one row per node) gives the dense-path result on the real rows.

The dense path pads every graph to the longest one and runs the row-wise layers (norm, MLP,
projections) on the padding too. The packed path keeps only the real rows, in node order.
Padding is filled with loud garbage so any leak into a real row would show up.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

from core.pretraining_config import create_pretraining_config
from core.pretraining_model import PretrainingESAModel
from esa.masked_layers import ESA, PackedLayout
from esa.mha_flash_varlen import MABFlashVarlen

from test_geometry_neutral_virtual_slots import _settings, _two_alanine_batch

DIM = 16


def _esa(layer_types=("M", "S", "M", "S", "P"), dim=DIM, **overrides):
    torch.manual_seed(0)
    kwargs = dict(
        num_outputs=1,
        dim_output=dim,
        dim_hidden=[dim] * len(layer_types),
        num_heads=[2] * len(layer_types),
        layer_types=list(layer_types),
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


def _graphs(sizes, seed=0, dim=DIM):
    """Packed rows, graph-local edges (every node has an edge), padded copy with garbage."""
    generator = torch.Generator().manual_seed(seed)
    batch_mapping = torch.cat([torch.full((s,), g) for g, s in enumerate(sizes)])
    offsets = torch.cumsum(torch.tensor([0, *sizes[:-1]]), 0)
    sources, targets = [], []
    for g, size in enumerate(sizes):
        for i in range(size):
            j = int(torch.randint(0, size, (1,), generator=generator))
            sources += [offsets[g] + i, offsets[g] + i, offsets[g] + j]
            targets += [offsets[g] + i, offsets[g] + j, offsets[g] + i]
    edge_index = torch.stack([torch.stack(sources), torch.stack(targets)])
    rows = torch.randn(batch_mapping.numel(), dim, generator=generator).to(torch.bfloat16)
    max_items = ((max(sizes) + 1 + 7) // 8) * 8
    dense, valid = to_dense_batch(rows, batch_mapping, fill_value=0, max_num_nodes=max_items)
    garbage = (torch.randn_like(dense.float()) * 50).to(torch.bfloat16)
    dense = torch.where(valid.unsqueeze(-1), dense, garbage)
    return rows, dense, valid, edge_index, batch_mapping, max_items


@pytest.mark.parametrize("sizes", [(9, 5, 12, 3), (7,), (1, 1, 1), (2, 30, 1, 4)])
@pytest.mark.parametrize("layer_types", [("M", "S", "M", "S", "P"), ("M", "S")])
def test_esa_node_embeddings_match_on_real_rows(cpu_flash, sizes, layer_types):
    model = _esa(layer_types)
    rows, dense, valid, edge_index, batch_mapping, max_items = _graphs(sizes, seed=sum(sizes))
    with torch.no_grad():
        dense_graph, dense_nodes = model(dense, edge_index, batch_mapping, max_items)
        packed_graph, packed_nodes = model(rows, edge_index, batch_mapping, max_items)

    assert packed_nodes.shape == (rows.size(0), DIM)
    torch.testing.assert_close(
        packed_nodes.float(), dense_nodes[valid].float(), atol=2e-2, rtol=2e-2
    )
    assert torch.isfinite(packed_graph.float()).all()
    if "P" in layer_types:
        # Pooling attends over the padded layout: dense padding is garbage, packed padding is zero,
        # so only the shape is comparable (no loss reads the pooled graph embedding).
        assert packed_graph.shape == dense_graph.shape
    else:
        # Without pooling the "graph embedding" is per row.
        torch.testing.assert_close(
            packed_graph.float(), dense_graph[valid].float(), atol=2e-2, rtol=2e-2
        )


def test_packed_run_builds_no_dense_tensor_before_the_pooling_layer(cpu_flash, monkeypatch):
    """Only the pooling (P) stage pads; with no P layer nothing is padded at all."""
    calls = []
    original = ESA._pad_rows
    monkeypatch.setattr(
        ESA, "_pad_rows", staticmethod(lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    )
    rows, _, _, edge_index, batch_mapping, max_items = _graphs((9, 5, 12))
    with torch.no_grad():
        _esa(("M", "S"))(rows, edge_index, batch_mapping, max_items)
        assert not calls
        _esa(("M", "S", "P"))(rows, edge_index, batch_mapping, max_items)
    assert len(calls) == 1


def test_packed_graph_attention_equals_dense_graph_attention():
    torch.manual_seed(1)
    attention = MABFlashVarlen(DIM, DIM, DIM, num_heads=2, dropout_p=0.0).eval()
    rows, dense, valid, edge_index, batch_mapping, _ = _graphs((7, 4, 10), dim=DIM)
    rows, dense = rows.float(), dense.float()
    with torch.no_grad():
        expected = attention.forward_graph(dense, edge_index, batch_mapping)[valid]
        packed = attention.forward_graph_packed(rows, edge_index, batch_mapping)
    torch.testing.assert_close(packed, expected, atol=1e-5, rtol=0)


def test_packed_flash_self_attention_equals_the_padded_path(cpu_flash):
    torch.manual_seed(2)
    attention = MABFlashVarlen(DIM, DIM, DIM, num_heads=2, dropout_p=0.0).to(torch.bfloat16).eval()
    sizes = (6, 3, 9)
    rows, dense, valid, _, batch_mapping, _ = _graphs(sizes)
    counts = torch.tensor(sizes, dtype=torch.int32)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(counts, 0).to(torch.int32)])
    with torch.no_grad():
        expected = attention(dense, dense, seq_lens=counts)[valid]
        packed = attention.forward_packed_varlen(rows, cu, max(sizes))
    torch.testing.assert_close(packed.float(), expected.float(), atol=2e-2, rtol=2e-2)


def test_packed_inputs_are_validated_loudly(cpu_flash):
    model = _esa()
    rows, dense, _, edge_index, batch_mapping, max_items = _graphs((4, 3))

    with pytest.raises(ValueError, match="rows but batch_mapping"):
        model(rows[:-1], edge_index, batch_mapping, max_items)
    with pytest.raises(ValueError, match="smaller than the longest graph"):
        model(rows, edge_index, batch_mapping, 2)
    with pytest.raises(ValueError, match="grouped by graph"):
        model(rows, edge_index, batch_mapping.flip(0), max_items)

    attention = MABFlashVarlen(DIM, DIM, DIM, num_heads=2, dropout_p=0.0).to(torch.bfloat16)
    bad_cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    with pytest.raises(ValueError, match="cu_seqlens ends"):
        attention.forward_packed_varlen(rows, bad_cu, 3)

    # A padded tensor must not be mixed with a layout, nor packed rows with the dense route.
    layer = model.encoder[0]
    layout = PackedLayout(torch.tensor([0, 4, 7], dtype=torch.int32), 4, 2)
    with pytest.raises(ValueError, match="packed rows need X"):
        layer((dense, edge_index, batch_mapping, max_items, layout))
    with pytest.raises(ValueError, match="packed rows need X"):
        layer((rows, edge_index, batch_mapping, max_items, None))


def test_packed_rows_refuse_unsupported_features():
    rows, _, _, edge_index, batch_mapping, max_items = _graphs((4, 3))
    model = _esa()
    model.use_equivariant = True
    with pytest.raises(ValueError, match="equivariant"):
        model(rows, edge_index, batch_mapping, max_items)
    model.use_equivariant = False
    model.use_distance_bias = True
    with pytest.raises(ValueError, match="distance bias"):
        model(rows, edge_index, batch_mapping, max_items)

    with pytest.raises(ValueError, match="esa_packed_rows"):
        create_pretraining_config(esa_packed_rows=True, use_equivariant_features=True)
    with pytest.raises(ValueError, match="esa_packed_rows"):
        create_pretraining_config(esa_packed_rows=True, use_distance_bias=True)
    assert create_pretraining_config().esa_packed_rows is False


def test_model_forward_gives_the_same_node_embeddings_with_packed_rows(cpu_flash):
    """Whole PretrainingESAModel.forward path (compaction, node_mlp, ESA, scatter back)."""

    class Encoder(nn.Module):
        def forward(self, x, pos, batch):
            coordination = torch.bincount(batch.edge_index[0], minlength=x.numel()).to(pos.dtype)
            features = torch.stack((x.to(pos.dtype), pos[:, 0], pos[:, 1], coordination), dim=1)
            return features.repeat(1, 2)

    class ToBF16(nn.Module):
        def forward(self, x):
            return x.to(torch.bfloat16)

    dummy = nn.Module()
    dummy.config = SimpleNamespace(
        geometry_neutral_virtual_slots=_settings(),
        cutoff_distance=2.1,
        max_spatial_neighbors=48,
        apply_attention_on="node",
        set_max_items=0,
        esa_packed_rows=False,
    )
    dummy.encoder = Encoder()
    dummy.encoder.domain_local_slot_embedding = nn.Embedding(38, 8).to(torch.bfloat16)
    dummy.node_mlp = ToBF16()
    dummy.mlp_norm = nn.Identity()
    dummy.esa_backbone = _esa(("M", "S", "P"), dim=8)

    seen_dims = []
    dummy.esa_backbone.register_forward_pre_hook(lambda _m, args: seen_dims.append(args[0].dim()))

    batch = _two_alanine_batch()
    with torch.no_grad():
        dense_graph, dense_nodes = PretrainingESAModel.forward(dummy, batch)
        dummy.config.esa_packed_rows = True
        packed_graph, packed_nodes = PretrainingESAModel.forward(dummy, batch)

    assert seen_dims == [3, 2]  # padded [B, N, D] first, packed [T, D] second

    assert packed_nodes.shape == dense_nodes.shape
    torch.testing.assert_close(packed_nodes.float(), dense_nodes.float(), atol=2e-2, rtol=2e-2)
    assert packed_graph.shape == dense_graph.shape

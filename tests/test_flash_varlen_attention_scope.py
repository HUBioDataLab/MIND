from unittest.mock import Mock

import pytest
import torch

import esa.mha_flash_varlen as flash_module
from esa.masked_layers import SABComplete
from esa.mha_flash_varlen import MABFlashVarlen


def _flash_layer(idx: int) -> SABComplete:
    return SABComplete(
        dim_in=4,
        dim_out=4,
        num_heads=2,
        dropout=0.0,
        idx=idx,
        norm_type="LN",
        node_or_edge="node",
        xformers_or_torch_attn="flash_varlen",
        set_max_items=3,
        pre_or_post="pre",
    )


@pytest.mark.parametrize("arity", [4, 5])
def test_unpad_input_compat_accepts_supported_api_arities(monkeypatch, arity):
    values = tuple(object() for _ in range(arity))
    monkeypatch.setattr(flash_module, "unpad_input", lambda *_args: values)

    result = flash_module._unpad_input_compat(object(), object())

    assert result == values[:4]


def test_unpad_input_compat_rejects_unknown_api_arity(monkeypatch):
    monkeypatch.setattr(
        flash_module, "unpad_input", lambda *_args: tuple(object() for _ in range(6))
    )

    with pytest.raises(ValueError, match="returned 6 values; expected 4 or 5"):
        flash_module._unpad_input_compat(object(), object())


def test_flash_backend_routes_m_layer_through_sparse_graph_attention():
    layer = _flash_layer(idx=1)
    inputs = torch.randn(2, 3, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch_mapping = torch.tensor([0, 0, 1], dtype=torch.long)
    adjacency = torch.zeros(2, 1, 3, 3)
    layer.sab.forward_graph = Mock(return_value=torch.zeros_like(inputs))

    layer((inputs, edge_index, batch_mapping, 3, adjacency))

    call = layer.sab.forward_graph.call_args
    torch.testing.assert_close(call.kwargs["edge_index"], edge_index)
    torch.testing.assert_close(call.kwargs["batch_mapping"], batch_mapping)


def test_flash_backend_keeps_s_layer_global_and_padding_free():
    layer = _flash_layer(idx=0)
    inputs = torch.randn(2, 3, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch_mapping = torch.tensor([0, 0, 1], dtype=torch.long)
    adjacency = torch.zeros(2, 1, 3, 3)
    layer.sab.forward = Mock(return_value=torch.zeros_like(inputs))

    layer((inputs, edge_index, batch_mapping, 3, adjacency))

    call = layer.sab.forward.call_args
    torch.testing.assert_close(
        call.kwargs["seq_lens"], torch.tensor([2, 1], dtype=torch.int32)
    )
    assert call.kwargs["adj_mask"] is None


def test_padded_attention_obeys_the_graph_mask():
    attention = MABFlashVarlen(2, 2, 2, num_heads=1, dropout_p=0.0)
    with torch.no_grad():
        attention.fc_q.weight.zero_()
        attention.fc_q.bias.zero_()
        attention.fc_k.weight.zero_()
        attention.fc_k.bias.zero_()
        attention.fc_v.weight.copy_(torch.eye(2))
        attention.fc_v.bias.zero_()
        attention.fc_o.weight.zero_()
        attention.fc_o.bias.zero_()

    inputs = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]])
    adjacency = torch.full((1, 1, 3, 3), float("-inf"))
    adjacency[0, 0, 0, 1] = 0.0
    adjacency[0, 0, 1, 0] = 0.0
    adjacency[0, 0, 2, 2] = 0.0

    output = attention(inputs, inputs, seq_lens=None, adj_mask=adjacency)

    expected = torch.tensor([[[0.0, 2.0], [1.0, 0.0], [3.0, 4.0]]])
    torch.testing.assert_close(output, expected)


def test_sparse_graph_attention_obeys_edges_and_coalesces_duplicates():
    attention = MABFlashVarlen(2, 2, 2, num_heads=1, dropout_p=0.0)
    with torch.no_grad():
        attention.fc_q.weight.zero_()
        attention.fc_q.bias.zero_()
        attention.fc_k.weight.zero_()
        attention.fc_k.bias.zero_()
        attention.fc_v.weight.copy_(torch.eye(2))
        attention.fc_v.bias.zero_()
        attention.fc_o.weight.zero_()
        attention.fc_o.bias.zero_()

    inputs = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]])
    edge_index = torch.tensor(
        [[0, 0, 1, 2], [1, 1, 0, 2]], dtype=torch.long
    )
    batch_mapping = torch.zeros(3, dtype=torch.long)

    output = attention.forward_graph(inputs, edge_index, batch_mapping)

    expected = torch.tensor([[[0.0, 2.0], [1.0, 0.0], [3.0, 4.0]]])
    torch.testing.assert_close(output, expected)


def test_global_identical_queries_stay_identical_but_local_owners_separate_them():
    attention = MABFlashVarlen(2, 2, 2, num_heads=1, dropout_p=0.0)
    with torch.no_grad():
        attention.fc_q.weight.zero_()
        attention.fc_q.bias.zero_()
        attention.fc_k.weight.zero_()
        attention.fc_k.bias.zero_()
        attention.fc_v.weight.copy_(torch.eye(2))
        attention.fc_v.bias.zero_()
        attention.fc_o.weight.zero_()
        attention.fc_o.bias.zero_()

    # Queries 0/1 are intentionally identical. Anchors 2/3 carry different
    # residue context.
    inputs = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]])
    global_output = attention(inputs, inputs, seq_lens=None, adj_mask=None)
    torch.testing.assert_close(global_output[:, 0], global_output[:, 1])

    local_output = attention.forward_graph(
        inputs,
        torch.tensor([[0, 1, 2, 3], [2, 3, 2, 3]], dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
    )
    assert not torch.allclose(local_output[:, 0], local_output[:, 1])


def test_sparse_graph_attention_is_finite_for_an_isolated_node_and_backward():
    attention = MABFlashVarlen(4, 4, 4, num_heads=2, dropout_p=0.0)
    inputs = torch.randn(1, 3, 4, requires_grad=True)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch_mapping = torch.zeros(3, dtype=torch.long)

    output = attention.forward_graph(inputs, edge_index, batch_mapping)

    assert bool(torch.isfinite(output).all())
    output.sum().backward()
    assert inputs.grad is not None
    assert bool(torch.isfinite(inputs.grad).all())


def test_sparse_graph_attention_supports_bfloat16_autocast_and_backward():
    attention = MABFlashVarlen(4, 4, 4, num_heads=2, dropout_p=0.0)
    inputs = torch.randn(1, 3, 4, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.long)
    batch_mapping = torch.zeros(3, dtype=torch.long)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = attention.forward_graph(inputs, edge_index, batch_mapping)
        loss = output.float().square().mean()

    assert bool(torch.isfinite(output).all())
    loss.backward()
    assert inputs.grad is not None
    assert bool(torch.isfinite(inputs.grad).all())


def test_sparse_graph_attention_rejects_cross_graph_edges():
    attention = MABFlashVarlen(2, 2, 2, num_heads=1, dropout_p=0.0)
    inputs = torch.randn(2, 1, 2)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    batch_mapping = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="cross-graph edge"):
        attention.forward_graph(inputs, edge_index, batch_mapping)


def test_varlen_path_rejects_a_silently_ignored_adjacency_mask():
    attention = MABFlashVarlen(2, 2, 2, num_heads=1, dropout_p=0.0)
    inputs = torch.zeros(1, 1, 2)

    with pytest.raises(ValueError, match="does not support arbitrary adjacency"):
        attention._forward_varlen(
            inputs,
            inputs,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            adj_mask=torch.zeros(1, 1, 1, 1),
        )


def test_varlen_value_projection_reads_the_original_key_value_input(monkeypatch):
    attention = MABFlashVarlen(2, 2, 2, num_heads=1, dropout_p=0.0).half()
    inputs = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float16
    )
    captured = {}

    def capture_value_input(_module, args):
        captured["value_input"] = args[0].detach().clone()

    def fake_unpad(tensor, _mask):
        flat = tensor.reshape(-1, *tensor.shape[2:])
        indices = torch.arange(flat.size(0))
        cumulative = torch.tensor([0, flat.size(0)], dtype=torch.int32)
        return flat, indices, cumulative, tensor.size(1)

    def fake_pad(tensor, _indices, batch_size, max_seq_len):
        return tensor.reshape(batch_size, max_seq_len, *tensor.shape[1:])

    monkeypatch.setattr(flash_module, "unpad_input", fake_unpad)
    monkeypatch.setattr(flash_module, "pad_input", fake_pad)
    monkeypatch.setattr(
        flash_module,
        "flash_attn_varlen_func",
        lambda _q, _k, value, *_args, **_kwargs: value,
    )
    attention.fc_v.register_forward_pre_hook(capture_value_input)

    attention._forward_varlen(
        inputs,
        inputs,
        seq_lens=torch.tensor([2], dtype=torch.int32),
    )

    torch.testing.assert_close(captured["value_input"], inputs)

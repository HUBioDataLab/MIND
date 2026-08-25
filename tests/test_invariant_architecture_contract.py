from contextlib import nullcontext
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import yaml

import esa.masked_layers as masked_layers
import esa.mha as mha
import esa.equivariant_sab as equivariant_sab
from core.pretraining_config import create_pretraining_config
from core.pretraining_model import (
    PretrainingESAModel,
    _migrate_legacy_invariant_sab_state_dict,
)
from esa.masked_layers import ESA


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIGS = (
    REPO_ROOT / "core" / "pretraining_config_fast.yaml",
    REPO_ROOT / "core" / "pretraining_config_multidomain.yaml",
)


def _build_esa(
    layer_types: list[str], *, use_equivariant: bool = False
) -> ESA:
    width = 8
    return ESA(
        num_outputs=2,
        dim_output=width,
        dim_hidden=[width] * len(layer_types),
        num_heads=[2] * len(layer_types),
        layer_types=layer_types,
        node_or_edge="node",
        xformers_or_torch_attn="torch",
        pre_or_post="pre",
        norm_type="LN",
        sab_dropout=0.0,
        mab_dropout=0.0,
        pma_dropout=0.0,
        residual_dropout=0.0,
        pma_residual_dropout=0.0,
        use_mlps=False,
        set_max_items=4,
        use_bfloat16=False,
        use_equivariant=use_equivariant,
    )


def _state_schema(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(tensor.shape)
        for name, tensor in module.state_dict().items()
    }


def test_duplicate_p_and_single_p_have_the_same_registered_architecture():
    legacy = _build_esa(["M", "S", "P", "P"])
    single = _build_esa(["M", "S", "P"])

    assert len(legacy.encoder) == len(single.encoder) == 2
    assert len(legacy.decoder) == len(single.decoder) == 1
    assert _state_schema(legacy) == _state_schema(single)
    assert sum(parameter.numel() for parameter in legacy.parameters()) == sum(
        parameter.numel() for parameter in single.parameters()
    )

    incompatible = single.load_state_dict(legacy.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_duplicate_p_and_single_p_match_after_strict_weight_load_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(7)
    legacy = _build_esa(["M", "S", "P", "P"])
    single = _build_esa(["M", "S", "P"])
    single.load_state_dict(legacy.state_dict(), strict=True)
    legacy.eval()
    single.eval()

    # ESA's production mask helper currently defaults to a CUDA device, while
    # this contract test deliberately exercises the torch attention backend on
    # CPU. A None mask is valid and gives both models identical global attention.
    monkeypatch.setattr(
        masked_layers,
        "get_adj_mask_from_edge_index_node",
        lambda **_: None,
    )
    # The torch MAB pins SDPA to the CUDA-efficient kernel. Let PyTorch select
    # its CPU math implementation without changing the model computation.
    monkeypatch.setattr(mha, "sdpa_kernel", lambda *_args, **_kwargs: nullcontext())

    inputs = torch.randn(1, 4, 8)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
        dtype=torch.long,
    )
    batch_mapping = torch.zeros(4, dtype=torch.long)

    with torch.no_grad():
        legacy_graph, legacy_nodes = legacy(
            inputs.clone(), edge_index, batch_mapping, num_max_items=4
        )
        single_graph, single_nodes = single(
            inputs.clone(), edge_index, batch_mapping, num_max_items=4
        )

    torch.testing.assert_close(single_graph, legacy_graph, rtol=0.0, atol=0.0)
    torch.testing.assert_close(single_nodes, legacy_nodes, rtol=0.0, atol=0.0)


def test_requested_equivariance_fails_when_dependencies_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(equivariant_sab, "NEQUIP_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="silent invariant fallback is disabled"):
        _build_esa(["S"], use_equivariant=True)
    output = capsys.readouterr().out
    assert "EquivariantSAB enabled" not in output


def test_legacy_wrapper_keys_migrate_to_direct_invariant_keys():
    state_dict = OrderedDict(
        [
            (
                "esa_backbone.encoder.0.sab.invariant_sab.mab.fc_q.weight",
                torch.randn(2, 2),
            ),
            ("encoder.atom_embedding.weight", torch.randn(3, 2)),
        ]
    )
    state_dict._metadata = OrderedDict(
        [
            ("esa_backbone.encoder.0.sab.invariant_sab", {"version": 1}),
        ]
    )
    expected = {
        "esa_backbone.encoder.0.sab.mab.fc_q.weight",
        "encoder.atom_embedding.weight",
    }

    migrated, count = _migrate_legacy_invariant_sab_state_dict(
        state_dict, expected
    )

    assert count == 1
    assert set(migrated) == expected
    assert (
        "esa_backbone.encoder.0.sab" in migrated._metadata
    )
    torch.testing.assert_close(
        migrated["esa_backbone.encoder.0.sab.mab.fc_q.weight"],
        state_dict[
            "esa_backbone.encoder.0.sab.invariant_sab.mab.fc_q.weight"
        ],
    )


def test_real_equivariant_checkpoint_is_not_silently_downgraded():
    state_dict = OrderedDict(
        [
            (
                "esa_backbone.encoder.0.sab.equivariant_interaction.weight",
                torch.randn(2, 2),
            )
        ]
    )

    with pytest.raises(RuntimeError, match="real equivariant branch weights"):
        _migrate_legacy_invariant_sab_state_dict(state_dict, state_dict.keys())


def test_nested_and_legacy_flat_hparams_reconstruct_invariant_models():
    config = create_pretraining_config(
        graph_dim=16,
        hidden_dims=[16],
        num_heads=[4],
        layer_types=["S"],
        xformers_or_torch_attn="torch",
        set_max_items=4,
        gaussian_kernels=4,
        use_3d_coordinates=True,
        use_equivariant_features=False,
    )
    model = PretrainingESAModel(config)

    assert set(model.hparams) == {"config"}
    nested = PretrainingESAModel(config=dict(model.hparams["config"]))
    nested.load_state_dict(model.state_dict(), strict=True)
    assert nested.equivariant_effective is False

    legacy_hparams = dict(vars(config))
    legacy_hparams["use_equivariant_features"] = True
    legacy = PretrainingESAModel(**legacy_hparams)
    assert legacy.config.use_equivariant_features is False
    assert legacy.equivariant_effective is False


@pytest.mark.parametrize("config_path", PRODUCTION_CONFIGS, ids=lambda path: path.stem)
def test_production_pretraining_config_has_one_terminal_p(config_path: Path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    hidden_dims = config["hidden_dims"]
    num_heads = config["num_heads"]
    layer_types = config["layer_types"]

    assert len(hidden_dims) == len(num_heads) == len(layer_types)
    assert layer_types.count("P") == 1
    assert layer_types[-1] == "P"
    assert config["use_3d_coordinates"] is True
    assert config["use_equivariant_features"] is False

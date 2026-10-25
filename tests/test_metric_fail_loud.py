"""Fail-loud contracts for the per-atom metric callbacks and the pretraining losses."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from core.metrics.per_atom_metrics import PerAtomMetricsCalculator
from core.metrics.protein_atom_types import ALL_PROTEIN_ATOMS
from core.metrics.protein_per_atom_metrics_callback import ProteinPerAtomMetricsCallback
from core.metrics.rna_atom_types import ALL_RNA_ATOMS
from core.metrics.rna_per_atom_metrics_callback import PerAtomMetricsCallback
from core.metrics.virtual_slot_diagnostics_callback import VirtualSlotDiagnosticsCallback
from core.pretraining_model import PretrainingESAModel
from core.pretraining_tasks import PretrainingTasks
from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS,
    RNA_BACKBONE_ATOMS,
    RNA_BASE_ATOMS_BY_NUCLEOTIDE,
    VIRTUAL_ATOMIC_NUMBER,
)

ELEMENT_Z = {"C": 6, "N": 7, "O": 8, "P": 15, "S": 16}


def _sample(domain, blocks=2):
    if domain == "protein":
        codes = list(PROTEIN_BACKBONE_ATOMS) + ["CB"] + [f"V{i}" for i in range(2, 11)]
        symbol = "ALA"
    elif domain == "rna":
        base = list(RNA_BASE_ATOMS_BY_NUCLEOTIDE["C"])
        codes = list(RNA_BACKBONE_ATOMS) + base + [f"V{i}" for i in range(len(base) + 1, 12)]
        symbol = "C"
    else:
        codes = ["sm"] * 4
        symbol = "MOL"
    per_block = len(codes)
    codes = codes * blocks
    z = torch.tensor([
        VIRTUAL_ATOMIC_NUMBER if code.startswith("V") else ELEMENT_Z.get(code[0], 6)
        for code in codes
    ])
    return Data(
        z=z,
        pos=torch.randn(len(codes), 3),
        block_idx=torch.arange(blocks).repeat_interleave(per_block),
        pos_code=codes,
        block_symbols=[symbol] * blocks,
        dataset_type=domain,
        num_nodes=len(codes),
    )


def _batch(domains, mask_all=True):
    batch = Batch.from_data_list([_sample(domain) for domain in domains])
    batch.original_types = batch.z.clone()
    batch.mlm_mask = torch.full_like(batch.z, mask_all, dtype=torch.bool)
    batch.masked_types = batch.z.clone()
    return batch


def _module(use_wandb=False, tasks=("mlm",)):
    return SimpleNamespace(config=SimpleNamespace(pretraining_tasks=list(tasks), use_wandb=use_wandb))


def _outputs(predictions):
    return {"mlm_preds_atomic_num": torch.as_tensor(predictions)}


# --------------------------------------------------------------------------- #
# Precision is computed over the predicted element                             #
# --------------------------------------------------------------------------- #

def test_element_aware_precision_counts_wrong_predictions_as_false_positives():
    atoms = ["C1'", "N3", "V"]
    calc = PerAtomMetricsCalculator(atom_types=atoms, compute_groups=False)
    y_true = np.array([0, 0, 1, 1, 2, 2])
    true_elements = np.array(["C", "C", "N", "N", "V", "V"])
    # One N atom predicted as C, one C atom predicted as V; V atoms correct.
    predicted_elements = np.array(["C", "V", "C", "N", "V", "V"])
    out_of_vocab = len(atoms)
    y_pred = np.where(predicted_elements == true_elements, y_true, out_of_vocab)
    calc.update(y_pred, y_true, predicted_elements=predicted_elements,
                target_elements=true_elements)
    metrics = calc.compute()["per_atom"]

    assert metrics["C1'"]["Recall"] == pytest.approx(0.5)
    assert metrics["C1'"]["Precision"] == pytest.approx(0.5)  # 1 of 2 C predictions
    assert metrics["N3"]["Precision"] == pytest.approx(1.0)
    assert metrics["V"]["Precision"] == pytest.approx(2 / 3)  # real atom predicted V
    assert metrics["V"]["F1"] == pytest.approx(2 * (2 / 3) * 1.0 / (2 / 3 + 1.0))


def test_overall_macro_recall_is_not_diluted_by_the_out_of_vocab_label():
    atoms = ["C1'", "N3"]
    calc = PerAtomMetricsCalculator(atom_types=atoms, compute_groups=False)
    y_true = np.array([0, 0, 1, 1])
    true_elements = np.array(["C", "C", "N", "N"])
    predicted_elements = np.array(["C", "N", "N", "N"])
    y_pred = np.where(predicted_elements == true_elements, y_true, len(atoms))
    calc.update(y_pred, y_true, predicted_elements=predicted_elements,
                target_elements=true_elements)
    overall = calc.compute()["overall"]
    assert overall["Accuracy"] == pytest.approx(0.75)
    assert overall["Recall_Macro"] == pytest.approx((0.5 + 1.0) / 2)
    assert overall["Recall_Weighted"] == pytest.approx(0.75)
    assert overall["Precision_Macro"] == pytest.approx((1.0 + 2 / 3) / 2)


def test_rna_callback_reports_precision_below_one_when_elements_are_confused():
    batch = _batch(["rna"])
    predictions = batch.original_types.clone()
    real = predictions != VIRTUAL_ATOMIC_NUMBER
    predictions[real.nonzero()[:3, 0]] = VIRTUAL_ATOMIC_NUMBER  # real -> V confusions
    predictions[predictions == 8] = 7                           # O -> N confusions
    callback = PerAtomMetricsCallback(enabled=True)
    callback.on_validation_batch_end(None, _module(), _outputs(predictions), batch, 0)
    per_atom = callback.val_metrics.compute()["per_atom"]
    assert per_atom["V"]["Precision"] < 1.0
    assert per_atom["N3"]["Precision"] < 1.0
    assert per_atom["P"]["Precision"] == pytest.approx(1.0)


def test_protein_callback_reports_precision_below_one_when_elements_are_confused():
    batch = _batch(["protein"])
    predictions = batch.original_types.clone()
    predictions[predictions == VIRTUAL_ATOMIC_NUMBER] = 6  # V predicted as carbon
    callback = ProteinPerAtomMetricsCallback(enabled=True)
    callback.on_validation_batch_end(None, _module(), _outputs(predictions), batch, 0)
    per_atom = callback.val_metrics.compute()["per_atom"]
    assert per_atom["CA"]["Recall"] == pytest.approx(1.0)
    assert per_atom["CA"]["Precision"] < 1.0
    assert per_atom["V"]["Recall"] == pytest.approx(0.0)
    assert set(per_atom) <= set(ALL_PROTEIN_ATOMS)


# --------------------------------------------------------------------------- #
# Callbacks: domain gating and fail-loud contracts                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("callback_cls", [PerAtomMetricsCallback, ProteinPerAtomMetricsCallback])
def test_batch_without_dataset_type_raises(callback_cls):
    batch = _batch(["rna"])
    del batch.dataset_type
    callback = callback_cls(enabled=True)
    with pytest.raises(ValueError, match="dataset_type"):
        callback.on_validation_batch_end(
            None, _module(), _outputs(batch.original_types), batch, 0
        )


@pytest.mark.parametrize("callback_cls", [PerAtomMetricsCallback, ProteinPerAtomMetricsCallback])
def test_non_target_domain_batch_is_skipped_without_error(callback_cls):
    batch = _batch(["unimol", "coconut"])
    callback = callback_cls(enabled=True)
    callback.on_validation_batch_end(None, _module(), _outputs(batch.original_types), batch, 0)
    assert callback.val_metrics.all_predictions == []


def test_rna_metrics_exclude_other_domains_of_a_mixed_batch():
    batch = _batch(["unimol", "rna", "protein"])
    callback = PerAtomMetricsCallback(enabled=True)
    callback.on_validation_batch_end(None, _module(), _outputs(batch.original_types), batch, 0)
    targets = np.concatenate(callback.val_metrics.all_targets)
    assert len(targets) == len(_sample("rna").pos_code)
    assert callback._val_nt_block_stats["C"]["blocks_seen"] == 2


@pytest.mark.parametrize("callback_cls", [PerAtomMetricsCallback, ProteinPerAtomMetricsCallback])
def test_metric_errors_propagate_on_every_batch(callback_cls):
    domain = "rna" if callback_cls is PerAtomMetricsCallback else "protein"
    batch = _batch([domain])
    callback = callback_cls(enabled=True)
    bad = _outputs(batch.original_types[:-1])
    for _ in range(2):
        with pytest.raises(ValueError, match="length mismatch"):
            callback.on_validation_batch_end(None, _module(), bad, batch, 0)


@pytest.mark.parametrize("callback_cls", [PerAtomMetricsCallback, ProteinPerAtomMetricsCallback])
def test_missing_block_symbols_raises(callback_cls):
    domain = "rna" if callback_cls is PerAtomMetricsCallback else "protein"
    batch = _batch([domain])
    del batch.block_symbols
    callback = callback_cls(enabled=True)
    with pytest.raises((ValueError, RuntimeError), match="block_symbols"):
        callback.on_validation_batch_end(None, _module(), _outputs(batch.original_types), batch, 0)


@pytest.mark.parametrize("callback_cls", [PerAtomMetricsCallback, ProteinPerAtomMetricsCallback])
def test_block_index_beyond_block_symbols_raises(callback_cls):
    domain = "rna" if callback_cls is PerAtomMetricsCallback else "protein"
    batch = _batch([domain])
    batch.block_symbols = [batch.block_symbols[0][:1]]
    callback = callback_cls(enabled=True)
    with pytest.raises((ValueError, RuntimeError), match="exceeds"):
        callback.on_validation_batch_end(None, _module(), _outputs(batch.original_types), batch, 0)


@pytest.mark.parametrize("callback_cls", [PerAtomMetricsCallback, ProteinPerAtomMetricsCallback])
def test_wandb_enabled_without_logger_raises_and_disabled_is_announced(callback_cls, capsys):
    domain = "rna" if callback_cls is PerAtomMetricsCallback else "protein"
    batch = _batch([domain])
    trainer = SimpleNamespace(loggers=[], current_epoch=1, global_step=0)

    callback = callback_cls(enabled=True)
    callback.on_validation_batch_end(None, _module(), _outputs(batch.original_types), batch, 0)
    with pytest.raises(RuntimeError, match="WandbLogger"):
        callback.on_validation_epoch_end(trainer, _module(use_wandb=True))

    callback = callback_cls(enabled=True)
    for _ in range(2):
        callback.on_validation_batch_end(None, _module(), _outputs(batch.original_types), batch, 0)
        callback.on_validation_epoch_end(trainer, _module(use_wandb=False))
    assert capsys.readouterr().out.count("no W&B logger") == 1


@pytest.mark.parametrize("callback_cls", [PerAtomMetricsCallback, ProteinPerAtomMetricsCallback])
def test_empty_validation_epoch_is_announced(callback_cls, capsys):
    callback = callback_cls(enabled=True)
    callback.on_validation_epoch_end(SimpleNamespace(loggers=[], current_epoch=0), _module())
    assert "nothing to log" in capsys.readouterr().out


def test_virtual_slot_diagnostics_require_predictions_and_slot_mask():
    callback = VirtualSlotDiagnosticsCallback(enabled=True)
    callback.setup(None, _module())
    batch = _batch(["protein"])
    with pytest.raises(ValueError, match="mlm_preds_atomic_num"):
        callback.on_validation_batch_end(None, None, {"loss": 0.0}, batch, 0)
    with pytest.raises(ValueError, match="target_slot_mask"):
        callback.on_validation_batch_end(None, None, _outputs(batch.original_types), batch, 0)


def test_virtual_slot_diagnostics_switch_off_visibly_without_mlm(capsys):
    callback = VirtualSlotDiagnosticsCallback(enabled=True)
    callback.setup(None, _module(tasks=("long_range_distance",)))
    assert not callback.enabled
    assert "disabled" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Losses: contract violations raise, legitimate empty cases are counted        #
# --------------------------------------------------------------------------- #

def _tasks():
    config = SimpleNamespace(
        graph_dim=4, distance_bins=4, max_distance=20.0, distance_bin_boundaries=None,
        short_range_distance_bins=4, short_range_max_distance=5.0,
        short_range_bin_boundaries=None, atom_types=121, virtual_atom_loss_weight=1.0,
    )
    return PretrainingTasks(config)


def test_long_range_without_positions_raises():
    with pytest.raises(ValueError, match="pos"):
        _tasks().long_range_distance_loss(torch.randn(2, 4), SimpleNamespace(pos=None))


def test_mlm_loss_requires_mask_attributes():
    data = SimpleNamespace(mlm_mask=None, original_types=torch.tensor([6]), masked_types=None)
    with pytest.raises(ValueError, match="mlm_mask, masked_types"):
        _tasks().mlm_loss(torch.randn(1, 4), data)


def test_empty_mlm_mask_returns_connected_zero_and_is_counted():
    tasks = _tasks()
    embeddings = torch.randn(3, 4, requires_grad=True)
    data = SimpleNamespace(
        mlm_mask=torch.zeros(3, dtype=torch.bool),
        original_types=torch.tensor([6, 7, 8]),
        masked_types=torch.tensor([6, 7, 8]),
    )
    loss = tasks.mlm_loss(embeddings, data, scope="rna")
    loss.backward()
    assert float(loss) == 0.0 and embeddings.grad is not None
    assert tasks.pop_loss_skips() == {"rna_mlm_no_masked_atoms": 1}
    assert tasks.pop_loss_skips() == {}


def test_short_range_without_valid_edges_is_counted():
    tasks = _tasks()
    loss = tasks.short_range_distance_loss(
        torch.randn(2, 4, requires_grad=True),
        torch.tensor([[0], [1]]),
        torch.tensor([9.0]),
        torch.ones(1, dtype=torch.bool),
    )
    assert float(loss) == 0.0
    assert tasks.pop_loss_skips() == {"batch_short_range_distance_no_valid_edges": 1}


def test_coordinate_denoising_requires_clean_positions_and_mask():
    tasks = _tasks()
    data = SimpleNamespace(pos=torch.zeros(2, 3), clean_pos=None, coord_mask=None)
    with pytest.raises(ValueError, match="clean_pos, coord_mask"):
        tasks.coordinate_denoising_loss(torch.randn(2, 4), data)
    data = SimpleNamespace(pos=torch.zeros(2, 3), clean_pos=torch.zeros(2, 3),
                           coord_mask=torch.zeros(2, dtype=torch.bool))
    assert float(tasks.coordinate_denoising_loss(torch.randn(2, 4), data)) == 0.0
    assert tasks.pop_loss_skips() == {"batch_coordinate_denoising_no_masked_atoms": 1}


def _harness(tasks=("mlm",)):
    config = SimpleNamespace(
        pretraining_tasks=list(tasks), task_weights={"mlm": 1.0, "short_range_distance": 1.0},
        protein_block_mlm=True, rna_block_mlm=True,
    )
    return SimpleNamespace(config=config, pretraining_tasks=_tasks())


def test_per_type_mlm_skip_of_unmasked_domain_is_counted_not_logged_as_zero():
    batch = _batch(["rna", "unimol"])
    batch.mlm_mask = torch.tensor([g == 0 for g in batch.batch.tolist()])
    harness = _harness()
    losses = PretrainingESAModel._compute_per_type_losses(
        harness, batch, None, torch.randn(batch.num_nodes, 4), is_training=False
    )
    assert "block_mlm" in losses["rna"]
    assert set(losses["unimol"]) == {"total"}
    assert harness.pretraining_tasks.pop_loss_skips() == {"unimol_mlm_no_masked_atoms": 1}


def test_per_type_mlm_errors_are_not_swallowed():
    batch = _batch(["rna"])
    del batch.masked_types
    with pytest.raises(ValueError, match="masked_types"):
        PretrainingESAModel._compute_per_type_losses(
            _harness(), batch, None, torch.randn(batch.num_nodes, 4), is_training=False
        )


def test_per_type_losses_require_dataset_type():
    batch = _batch(["rna"])
    del batch.dataset_type
    with pytest.raises(ValueError, match="dataset_type"):
        PretrainingESAModel._compute_per_type_losses(
            _harness(), batch, None, torch.randn(batch.num_nodes, 4)
        )


def test_short_range_task_without_edges_raises():
    batch = _batch(["rna"])
    with pytest.raises(ValueError, match="edge_index"):
        PretrainingESAModel._compute_pretraining_losses(
            _harness(tasks=("short_range_distance",)), batch, None,
            torch.randn(batch.num_nodes, 4), is_training=False,
        )


def test_loss_skips_are_logged_cumulatively_for_train_and_summed_for_val(capsys):
    logged = []
    harness = SimpleNamespace(
        pretraining_tasks=_tasks(),
        _loss_skip_totals=__import__("collections").Counter(),
        log=lambda name, value, **kwargs: logged.append((name, value, kwargs)),
    )
    for _ in range(2):
        harness.pretraining_tasks.record_loss_skip("mlm", "no_masked_atoms", scope="rna")
        PretrainingESAModel._log_loss_skips(harness, "train")
    harness.pretraining_tasks.record_loss_skip("mlm", "no_masked_atoms", scope="rna")
    PretrainingESAModel._log_loss_skips(harness, "val")
    names_values = [(name, value) for name, value, _ in logged]
    assert names_values == [
        ("loss_skipped/train_rna_mlm_no_masked_atoms", 1.0),
        ("loss_skipped/train_rna_mlm_no_masked_atoms", 2.0),
        ("loss_skipped/val_rna_mlm_no_masked_atoms", 1.0),
    ]
    assert logged[-1][2]["reduce_fx"] == "sum"
    assert capsys.readouterr().out.count("legitimate empty loss case") == 2

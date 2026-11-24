"""Learning-rate schedules of the downstream fine-tune loop: values per step on toy optimizers, the plateau direction
for a higher-is-better and a lower-is-better task metric, default path unchanged, invalid settings refused."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from downstream import probe
from downstream.finetune_settings import resolve_settings, same_settings, schedule_of
from downstream.lr_schedule import (
    COSINE_FINAL_FRACTION, PLATEAU_FACTOR, PLATEAU_MIN_LR, PLATEAU_PATIENCE, build_schedule, count_batches,
    current_lrs, lr_factor,
)

BLOCK = {"pooling": "mean_atoms", "head": "mlp", "seeds": [0], "epochs": 10, "patience": None,
         "max_atoms_per_batch": 40, "max_samples_per_batch": None, "lr_encoder": 1e-4,
         "lr_encoder_random_init": 5e-3, "lr_head": 1e-3, "weight_decay": 0.0, "grad_clip_norm": 0.5,
         "mlp_hidden": 16, "mlp_dropout": 0.0}


def toy_optimizer(peak_encoder=1e-4, peak_head=1e-3):
    encoder, head = nn.Linear(2, 2), nn.Linear(2, 1)
    return torch.optim.AdamW([{"params": encoder.parameters(), "lr": peak_encoder},
                              {"params": head.parameters(), "lr": peak_head}])


def settings_for(**overrides):
    return resolve_settings(BLOCK, random_init=False, overrides=overrides)


def per_step_lrs(optimizer, schedule, steps):
    """The lr pair each optimizer step runs with (read before the step, schedule advanced after it)."""
    seen = []
    for _ in range(steps):
        seen.append(current_lrs(optimizer))
        optimizer.step()
        schedule.step_batch()
    return np.array(seen)


def test_warmup_ramps_linearly_to_the_peak_for_both_groups_then_holds():
    optimizer = toy_optimizer()
    schedule = build_schedule(optimizer, settings_for(warmup_ratio=0.2), steps_per_epoch=lambda: 10)  # 100 steps
    assert schedule.total_steps == 100 and schedule.warmup_steps == 20
    lrs = per_step_lrs(optimizer, schedule, 100)
    expected = np.minimum((np.arange(100) + 1) / 20, 1.0)
    np.testing.assert_allclose(lrs[:, 0], 1e-4 * expected, rtol=1e-9)
    np.testing.assert_allclose(lrs[:, 1], 1e-3 * expected, rtol=1e-9)  # the head ramps to its own peak
    assert lrs[0, 0] == pytest.approx(1e-4 / 20) and lrs[19, 0] == pytest.approx(1e-4) and lrs[99, 0] == pytest.approx(1e-4)


def test_cosine_warms_up_then_decays_to_a_tenth_of_the_peak_at_the_last_step():
    optimizer = toy_optimizer()
    schedule = build_schedule(optimizer, settings_for(lr_schedule="cosine", warmup_ratio=0.1),
                              steps_per_epoch=lambda: 10)  # 100 steps, 10 of warmup
    lrs = per_step_lrs(optimizer, schedule, 130)  # steps beyond the plan keep the final value
    encoder = lrs[:, 0]
    assert encoder[0] == pytest.approx(1e-5) and encoder[9] == pytest.approx(1e-4) and encoder[10] == pytest.approx(1e-4)
    assert encoder[99] == pytest.approx(1e-4 * COSINE_FINAL_FRACTION)
    assert (encoder[100:] == pytest.approx(1e-4 * COSINE_FINAL_FRACTION))
    assert (np.diff(encoder[10:100]) < 0).all()  # strictly decaying after warmup
    # half-way through the decay (89 decay steps after 11 warmup steps) the factor is the mean of 1 and the floor
    assert lr_factor(11 + 44, schedule="cosine", total_steps=100, warmup_steps=11) == pytest.approx(
        (1 + COSINE_FINAL_FRACTION) / 2)
    np.testing.assert_allclose(lrs[:, 1] / 1e-3, encoder / 1e-4, rtol=1e-9)  # one factor for both groups


def test_cosine_without_warmup_starts_at_the_peak():
    assert lr_factor(0, schedule="cosine", total_steps=50, warmup_steps=0) == 1.0
    assert lr_factor(49, schedule="cosine", total_steps=50, warmup_steps=0) == pytest.approx(COSINE_FINAL_FRACTION)
    assert lr_factor(0, schedule="cosine", total_steps=1, warmup_steps=0) == 1.0  # one planned step: no division by 0
    assert lr_factor(7, schedule="constant", total_steps=10, warmup_steps=0) == 1.0


def test_lr_factor_refuses_bad_plans():
    for kwargs in ({"total_steps": 0, "warmup_steps": 0}, {"total_steps": 10, "warmup_steps": 10},
                   {"total_steps": 10, "warmup_steps": -1}):
        with pytest.raises(ValueError, match="warmup_steps < total_steps"):
            lr_factor(0, schedule="cosine", **kwargs)
    with pytest.raises(ValueError, match="unknown lr schedule"):
        lr_factor(0, schedule="linear", total_steps=10, warmup_steps=0)
    with pytest.raises(ValueError, match="step must be >= 0"):
        lr_factor(-1, schedule="cosine", total_steps=10, warmup_steps=0)


def test_default_settings_build_no_scheduler_and_keep_the_recorded_settings_unchanged():
    plain = resolve_settings(BLOCK, random_init=False)
    assert "lr_schedule" not in plain and "warmup_ratio" not in plain  # old config.json files still match
    assert same_settings(plain, resolve_settings(BLOCK, random_init=False, overrides={
        "lr_schedule": "constant", "warmup_ratio": 0.0}))
    assert schedule_of(plain) == ("constant", 0.0)
    optimizer = toy_optimizer()

    def planned():
        raise AssertionError("the default path must not plan steps")
    assert build_schedule(optimizer, plain, steps_per_epoch=planned) is None
    assert current_lrs(optimizer) == [1e-4, 1e-3]


def test_schedule_settings_are_recorded_and_make_a_run_use_other_settings():
    plain = resolve_settings(BLOCK, random_init=False)
    cosine = settings_for(lr_schedule="cosine", warmup_ratio=0.1)
    assert (cosine["lr_schedule"], cosine["warmup_ratio"]) == ("cosine", 0.1)
    assert not same_settings(cosine, plain)
    assert not same_settings(settings_for(warmup_ratio=0.1), plain)  # warmup alone is a schedule too
    assert not same_settings(settings_for(lr_schedule="plateau"), plain)
    assert not same_settings(cosine, settings_for(lr_schedule="cosine", warmup_ratio=0.2))


@pytest.mark.parametrize("overrides, text", [
    ({"lr_schedule": "linear"}, "lr_schedule must be one of"),
    ({"warmup_ratio": -0.1}, "warmup_ratio must be in"),
    ({"warmup_ratio": 1.0}, "warmup_ratio must be in"),
    ({"warmup_ratio": True}, "warmup_ratio must be in"),
    ({"lr_schedule": "plateau", "warmup_ratio": 0.1}, "plateau"),
])
def test_invalid_schedule_settings_are_refused(overrides, text):
    with pytest.raises(ValueError, match=text):
        settings_for(**overrides)


def test_a_warmup_shorter_than_one_step_is_refused():
    with pytest.raises(ValueError, match="less than one step"):
        build_schedule(toy_optimizer(), settings_for(warmup_ratio=0.01), steps_per_epoch=lambda: 5)  # 50 steps
    with pytest.raises(ValueError, match="steps_per_epoch must be positive"):
        build_schedule(toy_optimizer(), settings_for(lr_schedule="cosine"), steps_per_epoch=lambda: 0)


def run_plateau(scores):
    optimizer = toy_optimizer()
    schedule = build_schedule(optimizer, settings_for(lr_schedule="plateau"), steps_per_epoch=lambda: 4)
    history = []
    for score in scores:
        schedule.step_batch()  # per-batch stepping is a no-op for plateau
        schedule.step_epoch(score)
        history.append(current_lrs(optimizer))
    return np.array(history)


def test_plateau_halves_after_the_patience_and_only_then():
    history = run_plateau([0.80, 0.81, 0.82] + [0.82] * 8)  # best at epoch 2, then no improvement
    assert PLATEAU_PATIENCE == 3 and PLATEAU_FACTOR == 0.5
    reduced = np.flatnonzero(history[:, 0] < 1e-4)
    assert reduced[0] == 2 + PLATEAU_PATIENCE + 1  # the fourth epoch without improvement
    np.testing.assert_allclose(history[reduced[0], :], [5e-5, 5e-4])  # both groups halve
    assert (history[:reduced[0]] == [1e-4, 1e-3]).all()
    np.testing.assert_allclose(history[-1], [2.5e-5, 2.5e-4])  # a second reduction needs its own patience window
    assert (run_plateau(np.linspace(0.5, 0.9, 12)) == [1e-4, 1e-3]).all()  # steady improvement never reduces


def test_plateau_never_goes_below_the_minimum_lr():
    history = run_plateau([0.9] + [0.1] * 200)
    np.testing.assert_allclose(history[-1], [PLATEAU_MIN_LR, PLATEAU_MIN_LR])
    assert (history >= PLATEAU_MIN_LR).all()


def test_plateau_threshold_is_absolute_so_negated_errors_do_not_count_as_improving():
    # worse by less than a relative 1e-4 each epoch; a relative threshold on negative scores would call these better
    scores = [-1.0, -1.00005, -1.0001, -1.00015, -1.0002]
    assert (run_plateau(scores)[-1] == [5e-5, 5e-4]).all()


def _auc_scores(signal_levels):
    labels = (np.arange(60) % 2).astype(np.float32)[:, None]
    shared_noise = np.random.default_rng(0).normal(0, 1.0, labels.shape)
    return [probe.validation_score("multilabel_masked", labels, (labels * 2 - 1) * signal + shared_noise)
            for signal in signal_levels]  # the same labels and noise; a stronger signal gives a higher AUC


def _rmse_scores(error_levels):
    rng = np.random.default_rng(1)
    labels = rng.normal(0, 1, (50, 1)).astype(np.float32)
    noise = rng.normal(0, 1, labels.shape).astype(np.float32)
    return [probe.validation_score("multitarget_regression", labels, labels + level * noise, "rmse")
            for level in error_levels]


@pytest.mark.parametrize("kind, scorer", [("ROC-AUC (higher is better)", _auc_scores),
                                          ("RMSE (lower is better)", _rmse_scores)])
def test_plateau_follows_the_direction_of_the_task_metric(kind, scorer):
    # error level (RMSE) or signal strength (AUC): quality improves for five epochs, then degrades steadily (overfitting)
    levels = [1.0, 0.8, 0.6, 0.4, 0.2] + [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    if scorer is _auc_scores:
        levels = [0.2, 0.4, 0.6, 0.8, 1.0] + [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4]
    scores = scorer(levels)
    history = run_plateau(scores)
    best_epoch = int(np.argmax(scores))
    assert best_epoch == 4, kind
    assert (history[:best_epoch + PLATEAU_PATIENCE + 1] == [1e-4, 1e-3]).all(), kind  # improving: untouched
    assert history[best_epoch + PLATEAU_PATIENCE + 1, 0] == pytest.approx(5e-5), kind  # degrading: halved
    # the same metric read in the wrong direction would have reduced during the improvement instead
    wrong = run_plateau([-s for s in scores])
    assert wrong[best_epoch, 0] < 1e-4


def test_count_batches_mirrors_the_batching_caps():
    assert count_batches([], max_atoms=10, max_samples=None) == 0
    assert count_batches([3] * 10, max_atoms=15000, max_samples=4) == 3
    assert count_batches([3] * 10, max_atoms=15000, max_samples=None) == 1
    assert count_batches([4, 4, 4, 4, 4], max_atoms=9, max_samples=None) == 3  # [4,4] [4,4] [4]
    assert count_batches([20, 3, 3], max_atoms=7, max_samples=None) == 2  # an oversized sample is its own batch
    assert math.isclose(count_batches([1] * 7, max_atoms=100, max_samples=2), 4)

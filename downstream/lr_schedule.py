"""Learning-rate schedules of the downstream fine-tune loop (``--lr-schedule``, ``--warmup-ratio``).

One schedule scales the encoder and the head learning rate together (each group keeps its own peak value):

``constant``  the peak lr for the whole run (default; with ``warmup_ratio`` 0 no scheduler object exists at all).
``cosine``    linear warmup, then a half-cosine decay of the lr factor from 1 to ``COSINE_FINAL_FRACTION`` (0.1) of
              the peak at the last planned optimizer step.
``plateau``   ``torch.optim.lr_scheduler.ReduceLROnPlateau`` on the validation score that selects the best epoch
              (``downstream.probe.validation_score``: higher is better for every task kind, RMSE / MAE enter negated,
              so the mode is always ``max``): the lr is multiplied by ``PLATEAU_FACTOR`` after ``PLATEAU_PATIENCE``
              epochs without a validation improvement above ``PLATEAU_THRESHOLD`` (absolute, so negated errors
              behave like scores), never below ``PLATEAU_MIN_LR``. Stepped once per epoch, after validation.

Warmup is linear from 0 to the peak over ``warmup_ratio`` of the planned optimizer steps and is stepwise (one lr
update per optimizer step). The planned step count is ``epochs x steps per epoch``, where the steps per epoch are
counted before training with the batching rule of ``downstream.finetune.train_epoch`` on the epoch-0 sample order
(the node budget makes later epochs differ by a step at most); an early stop shortens the run but not the plan, and
steps beyond the plan keep the final factor. Warmup together with ``plateau`` is refused: both would write the lr.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from downstream.finetune_settings import LR_SCHEDULES, schedule_of

COSINE_FINAL_FRACTION = 0.1
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 3
PLATEAU_MIN_LR = 1e-6
PLATEAU_THRESHOLD = 1e-4


def lr_factor(step: int, *, schedule: str, total_steps: int, warmup_steps: int,
              final_fraction: float = COSINE_FINAL_FRACTION) -> float:
    """Multiplier of the peak lr used by optimizer step ``step`` (0-based) of a plan of ``total_steps`` steps.

    Warmup: ``(step + 1) / warmup_steps`` (reaches 1 at step ``warmup_steps - 1``). Cosine: 1 at the first step after
    warmup, ``final_fraction`` at step ``total_steps - 1`` (a plan with one decay step stays at 1); later steps keep
    the last value."""
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"unknown lr schedule {schedule!r}; choose from {LR_SCHEDULES}")
    if total_steps < 1 or not 0 <= warmup_steps < total_steps:
        raise ValueError(f"need 0 <= warmup_steps < total_steps, got {warmup_steps} of {total_steps}")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule != "cosine":
        return 1.0
    decay_steps = total_steps - warmup_steps
    progress = 0.0 if decay_steps == 1 else min((step - warmup_steps) / (decay_steps - 1), 1.0)
    return final_fraction + (1.0 - final_fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))


class LearningRateSchedule:
    """The scheduler(s) of one fit: ``step_batch()`` after every optimizer step, ``step_epoch(score)`` after validation."""

    def __init__(self, optimizer: Optimizer, settings: dict, *, steps_per_epoch: int):
        schedule, warmup_ratio = schedule_of(settings)
        if int(steps_per_epoch) < 1:
            raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
        self.schedule = schedule
        self.total_steps = int(settings["epochs"]) * int(steps_per_epoch)
        self.warmup_steps = int(warmup_ratio * self.total_steps)
        if warmup_ratio > 0 and self.warmup_steps < 1:
            raise ValueError(f"warmup_ratio {warmup_ratio} of {self.total_steps} planned steps is less than one step")
        self.batch: Optional[LambdaLR] = None
        self.plateau: Optional[ReduceLROnPlateau] = None
        if schedule == "plateau":
            self.plateau = ReduceLROnPlateau(optimizer, mode="max", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE,
                                             threshold=PLATEAU_THRESHOLD, threshold_mode="abs",
                                             min_lr=PLATEAU_MIN_LR)
        else:
            self.batch = LambdaLR(optimizer, lambda step: lr_factor(
                step, schedule=schedule, total_steps=self.total_steps, warmup_steps=self.warmup_steps))

    def step_batch(self) -> None:
        if self.batch is not None:
            self.batch.step()

    def step_epoch(self, score: float) -> None:
        """``score`` is the validation score that selects the best epoch (higher is better for every task kind)."""
        if self.plateau is not None:
            self.plateau.step(score)


def build_schedule(optimizer: Optimizer, settings: dict, *, steps_per_epoch: Callable[[], int]
                   ) -> Optional[LearningRateSchedule]:
    """None for the default (constant, no warmup): the optimizer then keeps its lr untouched. ``steps_per_epoch`` is
    called only when a schedule exists (counting it builds every epoch-0 training sample once)."""
    schedule, warmup_ratio = schedule_of(settings)
    if schedule == "constant" and warmup_ratio == 0:
        return None
    return LearningRateSchedule(optimizer, settings, steps_per_epoch=steps_per_epoch())


def current_lrs(optimizer: Optimizer) -> List[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def count_batches(node_counts: Sequence[int], *, max_atoms: int, max_samples: Optional[int]) -> int:
    """Optimizer steps of one epoch for samples of ``node_counts`` nodes in training order: the batching rule of
    ``downstream.finetune.train_epoch`` (a batch closes before the sample that would exceed a cap)."""
    steps, count, atoms = 0, 0, 0
    for nodes in node_counts:
        full = max_samples is not None and count >= int(max_samples)
        if count and (full or atoms + nodes > max_atoms):
            steps += 1
            count, atoms = 0, 0
        count += 1
        atoms += nodes
    return steps + (1 if count else 0)

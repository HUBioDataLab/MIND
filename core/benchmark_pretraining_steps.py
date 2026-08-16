#!/usr/bin/env python3
"""Run a bounded MIND training benchmark without changing trainer production code."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import yaml
from pytorch_lightning.callbacks import Callback

from core.pretraining_config import create_pretraining_config
from core import train_pretrain


class StepTimingCallback(Callback):
    """Measure synchronized compute time and end-to-end time between steps."""

    def __init__(self, output_path: Path, requested_steps: int, warmup_steps: int):
        super().__init__()
        self.output_path = output_path
        self.requested_steps = requested_steps
        self.warmup_steps = warmup_steps
        self.compute_seconds: list[float] = []
        self.inter_step_seconds: list[float] = []
        self.edge_rebuild_ms: list[float] = []
        self._step_start: float | None = None
        self._previous_end: float | None = None
        self._fit_start: float | None = None

    @staticmethod
    def _synchronize() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_fit_start(self, trainer, pl_module) -> None:
        self._synchronize()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._fit_start = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._synchronize()
        self._step_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._synchronize()
        now = time.perf_counter()
        if self._step_start is not None:
            self.compute_seconds.append(now - self._step_start)
        if self._previous_end is not None:
            self.inter_step_seconds.append(now - self._previous_end)
        template_stats = getattr(batch, "virtual_template_stats", None) or {}
        if "edge_rebuild_ms" in template_stats:
            self.edge_rebuild_ms.append(float(template_stats["edge_rebuild_ms"]))
        self._previous_end = now

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
        return {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    def on_fit_end(self, trainer, pl_module) -> None:
        self._synchronize()
        fit_seconds = (
            time.perf_counter() - self._fit_start if self._fit_start is not None else None
        )
        compute_after_warmup = self.compute_seconds[self.warmup_steps :]
        # inter_step[i] ends step i+1, so discard warmup_steps-1 intervals.
        interval_offset = max(0, self.warmup_steps - 1)
        inter_after_warmup = self.inter_step_seconds[interval_offset:]
        report = {
            "requested_steps": self.requested_steps,
            "completed_steps": int(trainer.global_step),
            "warmup_steps_excluded": self.warmup_steps,
            "fit_seconds_including_initialization_after_fit_start": fit_seconds,
            "compute_step_seconds_all": self._summary(self.compute_seconds),
            "compute_step_seconds_after_warmup": self._summary(compute_after_warmup),
            "end_to_end_inter_step_seconds_after_warmup": self._summary(inter_after_warmup),
            "edge_rebuild_ms_after_warmup": self._summary(
                self.edge_rebuild_ms[self.warmup_steps :]
            ),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_peak_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            ),
            "cuda_peak_memory_reserved_bytes": (
                torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
            ),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("STAGE0_TIMING_JSON=" + json.dumps(report, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("--warmup-steps must be in [0, steps)")

    os.environ.setdefault("WANDB_MODE", "offline")
    config_dict = yaml.safe_load(args.config.read_text())
    config = create_pretraining_config(**config_dict)
    timing_callback = StepTimingCallback(args.report, args.steps, args.warmup_steps)

    original_trainer = train_pretrain.pl.Trainer

    def bounded_trainer(*trainer_args, **trainer_kwargs):
        trainer_kwargs["max_steps"] = args.steps
        trainer_kwargs["callbacks"] = [
            *trainer_kwargs.get("callbacks", []),
            timing_callback,
        ]
        return original_trainer(*trainer_args, **trainer_kwargs)

    train_pretrain.pl.Trainer = bounded_trainer
    try:
        train_pretrain.train_universal_pretraining(
            config=config,
            dataset_name=config.dataset,
            dataset_dir=config.dataset_download_dir,
            output_dir=str(args.output_dir),
            wandb_project=config.wandb_project_name,
            wandb_run_name=config.wandb_run_name + "-stage0-100step",
        )
    finally:
        train_pretrain.pl.Trainer = original_trainer


if __name__ == "__main__":
    main()

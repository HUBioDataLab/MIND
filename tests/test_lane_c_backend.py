"""Focused Lane C tests for telemetry cadence and backend/logger containment."""

from __future__ import annotations

import io
import inspect
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from core.batch_statistics_callback import BatchStatisticsCallback
from core.train_pretrain import create_experiment_logger


class LaneCBackendTest(unittest.TestCase):
    def _batch(self):
        return SimpleNamespace(
            num_nodes=7,
            num_graphs=2,
            batch=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
            dataset_type=["protein", "rna"],
            edge_index=torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]),
            virtual_template_stats={},
        )

    def test_telemetry_emits_on_cadence_but_retains_every_batch(self):
        callback = BatchStatisticsCallback(log_every_n_batches=2)
        trainer = SimpleNamespace(current_epoch=0, train_dataloader=SimpleNamespace(dataset=None))
        output = io.StringIO()
        with redirect_stdout(output):
            for batch_idx in (0, 1):
                batch = self._batch()
                callback.on_train_batch_start(trainer, None, batch, batch_idx)
                callback.on_train_batch_end(trainer, None, None, batch, batch_idx)

        rendered = output.getvalue()
        self.assertIn("TELEMETRY epoch=0 batch=0", rendered)
        self.assertNotIn("TELEMETRY epoch=0 batch=1", rendered)
        self.assertEqual(len(callback.batch_stats), 2)
        self.assertEqual(callback.batch_stats[1]["domains"]["protein"], {
            "samples": 1,
            "atoms": 3,
        })
        self.assertEqual(callback.batch_stats[1]["domains"]["rna"], {
            "samples": 1,
            "atoms": 4,
        })

    def test_domain_aggregation_has_no_per_graph_item_sync_loop(self):
        source = inspect.getsource(BatchStatisticsCallback._compute_batch_stats)
        self.assertNotIn(".item(", source)
        self.assertIn("detach().cpu().tolist()", source)

    def test_invalid_telemetry_cadence_fails_fast(self):
        with self.assertRaises(ValueError):
            BatchStatisticsCallback(log_every_n_batches=0)

    def test_wandb_mode_is_explicit(self):
        disabled = SimpleNamespace(use_wandb=False)
        output = io.StringIO()
        with redirect_stdout(output):
            logger = create_experiment_logger(
                disabled,
                project="p",
                run_name="r",
                output_dir="/tmp/out",
            )
        self.assertFalse(logger)
        self.assertIn("use_wandb=false", output.getvalue())

        enabled = SimpleNamespace(use_wandb=True)
        with patch("core.train_pretrain.WandbLogger", return_value="wandb") as factory:
            logger = create_experiment_logger(
                enabled,
                project="p",
                run_name="r",
                output_dir="/tmp/out",
            )
        self.assertEqual(logger, "wandb")
        factory.assert_called_once()

    def test_approved_configs_use_safe_cadence_and_explicit_wandb(self):
        for path in ("core/pretraining_config_fast.yaml", "core/pretraining_config_multidomain.yaml"):
            config = yaml.safe_load(Path(path).read_text())
            self.assertTrue(config["use_wandb"], path)
            self.assertGreaterEqual(config["batch_stats_log_frequency"], 10, path)
            self.assertGreaterEqual(config["log_every_n_steps"], 10, path)

    def test_unsafe_triton_modules_are_absent_and_unimported(self):
        self.assertFalse(Path("esa/triton_flash_mha.py").exists())
        self.assertFalse(Path("esa/triton_graph_gather_scatter.py").exists())
        for path in Path("esa").glob("*.py"):
            source = path.read_text()
            self.assertNotIn("triton_flash_mha", source, str(path))
            self.assertNotIn("triton_graph_gather_scatter", source, str(path))


if __name__ == "__main__":
    unittest.main()

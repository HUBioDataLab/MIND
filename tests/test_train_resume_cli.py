import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core import train_pretrain


class TrainResumeCliTest(unittest.TestCase):
    def test_main_forwards_resolved_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "epoch99.ckpt"
            checkpoint.write_bytes(b"checkpoint-placeholder")
            output_dir = root / "output"

            argv = [
                "train_pretrain",
                "--config-yaml-path",
                "core/pretraining_config_fast.yaml",
                "--ckpt-path",
                str(checkpoint),
                "--max-atoms-per-batch",
                "18000",
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                train_pretrain, "train_universal_pretraining"
            ) as training:
                training.return_value = (mock.sentinel.model, mock.sentinel.trainer)
                train_pretrain.main()

            self.assertEqual(
                training.call_args.kwargs["ckpt_path"], str(checkpoint.resolve())
            )
            self.assertEqual(
                training.call_args.kwargs["config"].max_atoms_per_batch, 18000
            )


    def test_main_rejects_nonpositive_atom_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "train_pretrain",
                "--config-yaml-path",
                "core/pretraining_config_fast.yaml",
                "--max-atoms-per-batch",
                "0",
                "--output-dir",
                str(Path(tmp) / "output"),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                train_pretrain, "train_universal_pretraining"
            ) as training, self.assertRaises(SystemExit) as raised:
                train_pretrain.main()

            self.assertEqual(raised.exception.code, 2)
            training.assert_not_called()

    def test_main_rejects_missing_checkpoint_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.ckpt"
            argv = [
                "train_pretrain",
                "--config-yaml-path",
                "core/pretraining_config_fast.yaml",
                "--ckpt-path",
                str(missing),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                train_pretrain, "train_universal_pretraining"
            ) as training, self.assertRaises(SystemExit) as raised:
                train_pretrain.main()

            self.assertEqual(raised.exception.code, 2)
            training.assert_not_called()


if __name__ == "__main__":
    unittest.main()

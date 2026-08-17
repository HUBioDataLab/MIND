import io
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch

from core.batch_statistics_callback import BatchStatisticsCallback


class _CacheDataset:
    def get_cache_telemetry(self):
        return {
            "hits": 7,
            "misses": 3,
            "loads": 3,
            "load_seconds": 0.25,
            "hit_rate": 0.7,
        }


class BatchStatisticsCallbackTest(unittest.TestCase):
    def _trainer(self):
        return SimpleNamespace(
            current_epoch=0,
            train_dataloader=SimpleNamespace(dataset=_CacheDataset()),
        )

    def test_aggregates_exact_domain_denominators_and_measured_telemetry(self):
        batch = SimpleNamespace(
            num_nodes=7,
            num_graphs=2,
            batch=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
            dataset_type=["protein", "rna"],
            edge_index=torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]),
            virtual_template_stats={
                "template_total_ms": 1.5,
                "edge_rebuild_ms": 0.5,
            },
        )
        callback = BatchStatisticsCallback(log_every_n_batches=1)
        output = io.StringIO()
        with redirect_stdout(output):
            callback.on_train_batch_start(self._trainer(), None, batch, 0)
            time.sleep(0.001)
            callback.on_train_batch_end(self._trainer(), None, None, batch, 0)
            callback.on_train_epoch_end(self._trainer(), None)

        self.assertEqual(callback.batch_stats[0]["domains"]["protein"], {
            "samples": 1,
            "atoms": 3,
        })
        self.assertEqual(callback.batch_stats[0]["domains"]["rna"], {
            "samples": 1,
            "atoms": 4,
        })
        self.assertGreater(callback.batch_stats[0]["atoms_per_second"], 0)
        rendered = output.getvalue()
        self.assertIn("TELEMETRY epoch=0 batch=0", rendered)
        self.assertIn("PROTEIN", rendered)
        self.assertIn("RNA", rendered)
        self.assertIn("CACHE hits=7 misses=3 hit_rate=70.00%", rendered)
        self.assertIn("GRAPH NODES p50=3 p90=4 max=4", rendered)
        self.assertIn("BATCH NODES p50=7 p90=7 max=7", rendered)

    def test_callback_failure_is_actionable(self):
        callback = BatchStatisticsCallback(log_every_n_batches=1)
        with self.assertRaisesRegex(RuntimeError, "epoch=0, batch=4"):
            callback.on_train_batch_start(
                self._trainer(), None, SimpleNamespace(), 4
            )


if __name__ == "__main__":
    unittest.main()

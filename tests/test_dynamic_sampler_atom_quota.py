import unittest

from data_loading.improved_dynamic_sampler import ImprovedDynamicBatchSampler
from tests.test_dynamic_sampler_epoch_plan import _metadata_dataset, _sampler


class DynamicSamplerAtomQuotaTest(unittest.TestCase):
    def test_auto_batch_count_tracks_the_atom_cap_without_changing_selection(self):
        dataset = _metadata_dataset([
            ("protein", [100] * 10),
            ("rna", [50] * 10),
            ("unimol", [10] * 100),
        ])
        common_options = {
            "shuffle_chunks": False,
            "shuffle_within_chunk": False,
            "epoch_atom_budget": 1000,
            "epoch_num_batches": "auto",
            "domain_atom_weights": {"protein": 30, "rna": 25, "unimol": 45},
        }
        small_cap_sampler = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            max_atoms_per_batch=200,
            **common_options,
        )
        large_cap_sampler = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            max_atoms_per_batch=500,
            **common_options,
        )

        small_cap_batches = list(small_cap_sampler)
        large_cap_batches = list(large_cap_sampler)

        self.assertGreater(len(small_cap_batches), len(large_cap_batches))
        for sampler, batches, atom_cap in (
            (small_cap_sampler, small_cap_batches, 200),
            (large_cap_sampler, large_cap_batches, 500),
        ):
            self.assertEqual(sampler.plan_stats["natural_batches"], len(batches))
            self.assertEqual(sampler.plan_stats["planned_batches"], len(batches))
            self.assertEqual(sampler.plan_stats["actual_total_atoms"], 1000)
            self.assertTrue(all(
                sum(dataset.sample_metadata_index[idx]["num_atoms"] for idx in batch)
                <= atom_cap
                for batch in batches
            ))

        self.assertEqual(
            small_cap_sampler.plan_stats["domains"],
            large_cap_sampler.plan_stats["domains"],
        )

    def test_atom_quota_sampling_uses_whole_graphs_and_exact_atom_targets(self):
        dataset = _metadata_dataset([
            ("protein", [100] * 10),
            ("rna", [50] * 10),
            ("unimol", [10] * 100),
        ])
        sampler = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            max_atoms_per_batch=500,
            shuffle_chunks=False,
            shuffle_within_chunk=False,
            epoch_atom_budget=1000,
            epoch_num_batches=5,
            domain_atom_weights={"protein": 30, "rna": 25, "unimol": 45},
        )

        batches = list(sampler)
        flattened = [idx for batch in batches for idx in batch]
        stats = sampler.plan_stats

        self.assertEqual(stats["requested_total_atoms"], 1000)
        self.assertEqual(stats["actual_total_atoms"], 1000)
        self.assertEqual(stats["planned_batches"], 5)
        self.assertEqual(len(batches), 5)
        self.assertEqual(
            {
                dtype: domain_stats["actual_atoms"]
                for dtype, domain_stats in stats["domains"].items()
            },
            {"protein": 300, "rna": 250, "unimol": 450},
        )
        self.assertTrue(
            all(
                domain_stats["repeated_samples"] == 0
                for domain_stats in stats["domains"].values()
            )
        )
        self.assertEqual(len(flattened), len(set(flattened)))
        for batch in batches:
            batch_atoms = sum(
                dataset.sample_metadata_index[idx]["num_atoms"] for idx in batch
            )
            self.assertLessEqual(batch_atoms, 500)

    def test_atom_quota_cycle_repeats_only_the_shortfall(self):
        dataset = _metadata_dataset([
            ("protein", [100] * 10),
            ("rna", [50] * 4),
            ("unimol", [10] * 100),
        ])
        sampler = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            max_atoms_per_batch=500,
            shuffle_chunks=False,
            shuffle_within_chunk=False,
            epoch_atom_budget=1000,
            epoch_num_batches=5,
            domain_atom_weights={"protein": 30, "rna": 25, "unimol": 45},
        )

        list(sampler)
        rna_stats = sampler.plan_stats["domains"]["rna"]

        self.assertEqual(rna_stats["target_atoms"], 250)
        self.assertEqual(rna_stats["actual_atoms"], 250)
        self.assertEqual(rna_stats["available_samples"], 4)
        self.assertEqual(rna_stats["unique_samples"], 4)
        self.assertEqual(rna_stats["repeated_samples"], 1)

    def test_atom_quota_without_cycle_rejects_an_unreachable_target(self):
        dataset = _metadata_dataset([
            ("protein", [100] * 10),
            ("rna", [50] * 4),
            ("unimol", [10] * 100),
        ])
        sampler = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            shuffle_chunks=False,
            shuffle_within_chunk=False,
            cycle_short_domains=False,
            epoch_atom_budget=1000,
            epoch_num_batches=5,
            domain_atom_weights={"protein": 30, "rna": 25, "unimol": 45},
        )

        with self.assertRaisesRegex(ValueError, "Atom quotas cannot be met"):
            list(sampler)

    def test_atom_quota_requires_a_weight_for_every_active_domain(self):
        dataset = _metadata_dataset([
            ("protein", [100]),
            ("rna", [50]),
        ])
        sampler = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            epoch_atom_budget=1000,
            epoch_num_batches=5,
            domain_atom_weights={"protein": 30},
        )

        with self.assertRaisesRegex(ValueError, "Missing atom weights"):
            list(sampler)


if __name__ == "__main__":
    unittest.main()

import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import pytorch_lightning as pl
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

from core.train_pretrain import _build_train_batch_sampler, set_data_loader_epoch
from data_loading.dynamic_chunk_sampler import DynamicChunkAwareBatchSampler
from data_loading.improved_dynamic_sampler import ImprovedDynamicBatchSampler
from data_loading.lazy_universal_dataset import LazyUniversalDataset


SAMPLER_CLASSES = (
    ImprovedDynamicBatchSampler,
    DynamicChunkAwareBatchSampler,
)


class _MetadataDataset(LazyUniversalDataset):
    def __getitem__(self, idx):
        return Data(
            x=torch.ones((1, 1)),
            pos=torch.zeros((1, 3)),
            sample_index=torch.tensor([idx]),
        )


def _metadata_dataset(chunks):
    dataset = _MetadataDataset.__new__(_MetadataDataset)
    dataset.chunk_pt_files = []
    dataset.file_ranges = []
    dataset.chunk_metadata = []
    dataset.sample_metadata_index = []
    dataset.metadata_loaded = True

    start = 0
    for chunk_index, (dataset_type, atom_counts) in enumerate(chunks):
        path = f"/virtual/{dataset_type}_chunk_{chunk_index}/processed/data.pt"
        end = start + len(atom_counts)
        dataset.chunk_pt_files.append(path)
        dataset.file_ranges.append((start, end, path))
        dataset.chunk_metadata.append({"dataset_type": dataset_type})
        for local_index, atom_count in enumerate(atom_counts):
            dataset.sample_metadata_index.append({
                "chunk_idx": chunk_index,
                "local_idx": local_index,
                "num_atoms": atom_count,
                "num_edges": 0,
                "bucket": "test",
                "dataset_type": dataset_type,
            })
        start = end

    dataset.total_samples = start
    return dataset


def _sampler(sampler_class, dataset, **overrides):
    options = {
        "max_atoms_per_batch": 1000,
        "shuffle_chunks": True,
        "shuffle_within_chunk": True,
        "seed": 17,
        "enable_cross_modal_batches": True,
        "cycle_short_domains": True,
    }
    options.update(overrides)
    return sampler_class(dataset, **options)


def _write_chunk(path, num_nodes):
    path.parent.mkdir(parents=True)
    graph = Data(pos=torch.zeros((num_nodes, 3)), z=torch.full((num_nodes,), 6))
    data, slices = InMemoryDataset.collate([graph, graph.clone()])
    torch.save((data, slices), path)


class DynamicSamplerEpochPlanTest(unittest.TestCase):
    def test_len_and_iter_share_one_exact_plan_for_train_val_and_test(self):
        dataset = _metadata_dataset([
            ("unimol", [180, 220, 260, 300, 340, 380]),
            ("protein", [450, 550, 650]),
            ("rna", [700, 800]),
        ])

        for sampler_class in SAMPLER_CLASSES:
            loaders = {
                "train": _sampler(
                    sampler_class,
                    Subset(dataset, list(range(len(dataset)))),
                ),
                "val": _sampler(
                    sampler_class,
                    Subset(dataset, [0, 2, 6, 9]),
                    shuffle_chunks=False,
                    shuffle_within_chunk=False,
                    cycle_short_domains=False,
                ),
                "test": _sampler(
                    sampler_class,
                    Subset(dataset, [1, 4, 7, 10]),
                    shuffle_chunks=False,
                    shuffle_within_chunk=False,
                    cycle_short_domains=False,
                ),
            }

            for split, sampler in loaders.items():
                with self.subTest(sampler=sampler_class.__name__, split=split):
                    expected_length = len(sampler)
                    first_plan = list(sampler)
                    second_plan = list(sampler)
                    self.assertEqual(expected_length, len(first_plan))
                    self.assertEqual(first_plan, second_plan)

    def test_empty_single_sample_and_single_domain_subsets(self):
        dataset = _metadata_dataset([("protein", [400, 400, 400])])

        for sampler_class in SAMPLER_CLASSES:
            for indices, expected_batches in (([], 0), ([0], 1), ([0, 1, 2], 2)):
                with self.subTest(
                    sampler=sampler_class.__name__, indices=indices
                ):
                    sampler = _sampler(
                        sampler_class,
                        Subset(dataset, indices),
                        max_atoms_per_batch=800,
                        enable_cross_modal_batches=False,
                    )
                    self.assertEqual(len(sampler), expected_batches)
                    self.assertEqual(len(list(sampler)), expected_batches)

    def test_plan_is_seeded_and_changes_only_after_epoch_update(self):
        dataset = _metadata_dataset([("protein", [400] * 12)])
        kwargs = {
            "max_atoms_per_batch": 800,
            "enable_cross_modal_batches": False,
            "seed": 29,
        }

        for sampler_class in SAMPLER_CLASSES:
            with self.subTest(sampler=sampler_class.__name__):
                sampler_a = _sampler(sampler_class, dataset, **kwargs)
                sampler_b = _sampler(sampler_class, dataset, **kwargs)

                python_rng_state = random.getstate()
                torch_rng_state = torch.random.get_rng_state().clone()
                self.assertEqual(len(sampler_a), len(list(sampler_a)))
                cached_plan = sampler_a._batch_plan
                sampler_a.set_epoch(0)
                self.assertIs(sampler_a._batch_plan, cached_plan)
                self.assertEqual(random.getstate(), python_rng_state)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_rng_state))

                epoch_zero = list(sampler_a)
                self.assertEqual(epoch_zero, list(sampler_b))

                sampler_a.set_epoch(1)
                epoch_one = list(sampler_a)
                self.assertNotEqual(epoch_one, epoch_zero)
                self.assertEqual(epoch_one, list(sampler_a))

    def test_actual_loader_batch_sampler_receives_epoch(self):
        dataset = _metadata_dataset([("protein", [400, 400])])

        for sampler_class in SAMPLER_CLASSES:
            with self.subTest(sampler=sampler_class.__name__):
                batch_sampler = _sampler(
                    sampler_class,
                    dataset,
                    enable_cross_modal_batches=False,
                )
                loader = GeometricDataLoader(
                    dataset, batch_sampler=batch_sampler, num_workers=0
                )

                updated = set_data_loader_epoch(loader, 3)

                self.assertEqual(updated, ["batch_sampler"])
                self.assertIs(loader.batch_sampler, batch_sampler)
                self.assertEqual(batch_sampler.epoch, 3)
                self.assertEqual(len(loader), len(list(batch_sampler)))

    def test_lightning_consumes_every_batch_in_the_reported_plan(self):
        class CountingModule(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.steps_seen = 0

            def training_step(self, batch, batch_idx):
                self.steps_seen += 1
                return self.weight.square()

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        dataset = _metadata_dataset([("protein", [1000] * 4)])
        sampler = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            max_atoms_per_batch=1200,
            shuffle_chunks=False,
            shuffle_within_chunk=False,
            enable_cross_modal_batches=False,
        )
        loader = GeometricDataLoader(dataset, batch_sampler=sampler, num_workers=0)
        model = CountingModule()
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )

        trainer.fit(model, train_dataloaders=loader)

        self.assertEqual(len(loader), 4)
        self.assertEqual(model.steps_seen, 4)
        self.assertEqual(trainer.global_step, 4)

    def test_index_map_cache_is_selection_keyed_and_outside_data_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_dir = temp_path / "runtime_cache"
            data_root = temp_path / "production_data"
            files = []
            for chunk_index, num_nodes in enumerate((2, 3, 4)):
                path = (
                    data_root
                    / f"protein_chunk_{chunk_index}"
                    / "processed"
                    / "data.pt"
                )
                _write_chunk(path, num_nodes)
                files.append(str(path))

            selection_a = files[:2]
            selection_b = [files[0], files[2]]
            with mock.patch.dict(
                os.environ,
                {"MIND_INDEX_MAP_CACHE_DIR": str(cache_dir)},
            ):
                dataset_a = LazyUniversalDataset(selection_a, verbose=False)
                dataset_b = LazyUniversalDataset(selection_b, verbose=False)

                path_a = LazyUniversalDataset._index_map_cache_path(selection_a)
                path_b = LazyUniversalDataset._index_map_cache_path(selection_b)
                self.assertNotEqual(path_a, path_b)
                self.assertEqual(path_a.parent, cache_dir)
                self.assertEqual(path_b.parent, cache_dir)
                self.assertEqual(
                    json.loads(path_a.read_text())["chunk_pt_files"],
                    [str(Path(path).resolve()) for path in selection_a],
                )
                self.assertEqual(dataset_a.total_samples, 4)
                self.assertEqual(dataset_b.total_samples, 4)
                self.assertFalse((data_root / "protein_index_map.json").exists())

                with mock.patch(
                    "data_loading.lazy_universal_dataset.torch.load",
                    side_effect=AssertionError(
                        "matching selection should use its cache"
                    ),
                ):
                    cached_dataset = LazyUniversalDataset(selection_a, verbose=False)
                self.assertEqual(cached_dataset.total_samples, dataset_a.total_samples)


    def test_full_pass_cross_modal_consumes_all_samples_exactly_once(self):
        dataset = _metadata_dataset([
            ("protein", [400, 500, 600, 700, 800]),
            ("rna", [300, 450, 600]),
            ("unimol", [50, 60, 70, 80, 90, 100, 110, 120, 130, 140]),
        ])
        total_samples = len(dataset)
        total_atoms = sum(
            dataset.sample_metadata_index[idx]["num_atoms"]
            for idx in range(total_samples)
        )

        sampler = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            max_atoms_per_batch=1000,
            enable_cross_modal_batches=True,
            cycle_short_domains=False,
            epoch_atom_budget=None,
        )

        batches = list(sampler)
        self.assertEqual(len(sampler), len(batches))
        flattened = [idx for batch in batches for idx in batch]

        self.assertEqual(len(flattened), total_samples)
        self.assertEqual(len(set(flattened)), total_samples)
        self.assertEqual(set(flattened), set(range(total_samples)))

        stats = sampler.plan_stats
        self.assertIsNotNone(stats)
        self.assertEqual(stats["mode"], "full_pass")
        self.assertIsNone(stats["requested_total_atoms"])
        self.assertEqual(stats["actual_total_atoms"], total_atoms)
        self.assertEqual(stats["natural_batches"], len(batches))
        self.assertEqual(stats["planned_batches"], len(batches))

        for dtype, domain_stats in stats["domains"].items():
            self.assertEqual(
                domain_stats["selected_samples"], domain_stats["available_samples"]
            )
            self.assertEqual(
                domain_stats["unique_samples"], domain_stats["available_samples"]
            )
            self.assertEqual(domain_stats["repeated_samples"], 0)
            self.assertGreater(domain_stats["actual_atoms"], 0)

    def test_full_pass_proportional_interleaving_distribution(self):
        dataset = _metadata_dataset([
            ("protein", [500] * 20),
            ("rna", [400] * 10),
            ("unimol", [50] * 100),
        ])
        sampler = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            max_atoms_per_batch=1000,
            enable_cross_modal_batches=True,
            cycle_short_domains=False,
            epoch_atom_budget=None,
        )

        batches = list(sampler)
        flattened = [idx for batch in batches for idx in batch]
        self.assertEqual(len(flattened), 130)

        # Check that RNA and Protein are not front-loaded (both present in first half and second half)
        midpoint = len(flattened) // 2
        first_half = flattened[:midpoint]
        second_half = flattened[midpoint:]

        first_half_types = [
            dataset.sample_metadata_index[idx]["dataset_type"] for idx in first_half
        ]
        second_half_types = [
            dataset.sample_metadata_index[idx]["dataset_type"] for idx in second_half
        ]

        self.assertGreater(first_half_types.count("protein"), 0)
        self.assertGreater(second_half_types.count("protein"), 0)
        self.assertGreater(first_half_types.count("rna"), 0)
        self.assertGreater(second_half_types.count("rna"), 0)
        self.assertGreater(first_half_types.count("unimol"), 0)
        self.assertGreater(second_half_types.count("unimol"), 0)

        # Proportions in both halves should be close to natural proportions (20/130, 10/130, 100/130)
        self.assertTrue(8 <= first_half_types.count("protein") <= 12)
        self.assertTrue(8 <= second_half_types.count("protein") <= 12)
        self.assertTrue(3 <= first_half_types.count("rna") <= 7)
        self.assertTrue(3 <= second_half_types.count("rna") <= 7)

    def test_full_pass_multi_epoch_reshuffle_preserves_membership(self):
        dataset = _metadata_dataset([
            ("protein", [400] * 10),
            ("rna", [300] * 5),
            ("unimol", [50] * 30),
        ])
        sampler = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            max_atoms_per_batch=1000,
            enable_cross_modal_batches=True,
            cycle_short_domains=False,
            epoch_atom_budget=None,
        )

        sampler.set_epoch(0)
        epoch_0_batches = list(sampler)
        epoch_0_flattened = [idx for batch in epoch_0_batches for idx in batch]

        sampler.set_epoch(1)
        epoch_1_batches = list(sampler)
        epoch_1_flattened = [idx for batch in epoch_1_batches for idx in batch]

        self.assertEqual(len(epoch_0_flattened), len(dataset))
        self.assertEqual(len(epoch_1_flattened), len(dataset))
        self.assertEqual(set(epoch_0_flattened), set(range(len(dataset))))
        self.assertEqual(set(epoch_1_flattened), set(range(len(dataset))))
        self.assertNotEqual(epoch_0_flattened, epoch_1_flattened)

    def test_full_pass_ddp_rank_partitioning(self):
        dataset = _metadata_dataset([
            ("protein", [400] * 8),
            ("rna", [300] * 4),
            ("unimol", [50] * 20),
        ])
        common_kwargs = {
            "max_atoms_per_batch": 1000,
            "enable_cross_modal_batches": True,
            "cycle_short_domains": False,
            "epoch_atom_budget": None,
            "shuffle_chunks": False,
            "shuffle_within_chunk": False,
            "num_replicas": 2,
        }
        sampler_rank0 = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            rank=0,
            **common_kwargs,
        )
        sampler_rank1 = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            rank=1,
            **common_kwargs,
        )

        batches_0 = list(sampler_rank0)
        batches_1 = list(sampler_rank1)

        self.assertEqual(len(sampler_rank0), len(batches_0))
        self.assertEqual(len(sampler_rank1), len(batches_1))

        # Combined batches should equal the full plan
        full_plan = sampler_rank0._get_batch_plan()
        self.assertEqual(len(batches_0) + len(batches_1), len(full_plan))
        self.assertEqual(batches_0, full_plan[0::2])
        self.assertEqual(batches_1, full_plan[1::2])


    def test_train_batch_sampler_builder_handles_quota_and_full_pass(self):
        dataset = _metadata_dataset([
            ("protein", [400, 500]),
            ("rna", [300, 450]),
        ])

        import types

        config_full_pass = types.SimpleNamespace(
            seed=42,
            multi_domain={
                "enabled": True,
                "atom_sampling": {
                    "enabled": False,
                    "epoch_atom_budget": None,
                    "epoch_num_batches": None,
                    "cycle_to_quota": False,
                },
            },
            use_improved_sampler=True,
        )

        sampler_fp = _build_train_batch_sampler(
            train_dataset=dataset,
            config=config_full_pass,
            max_atoms_per_batch=1000,
            is_multi_domain=True,
            use_improved_sampler=True,
        )
        self.assertIsInstance(sampler_fp, ImprovedDynamicBatchSampler)
        self.assertIsNone(sampler_fp.epoch_atom_budget)
        self.assertFalse(sampler_fp.cycle_short_domains)
        self.assertTrue(sampler_fp.enable_cross_modal_batches)

        config_quota = types.SimpleNamespace(
            seed=42,
            multi_domain={
                "enabled": True,
                "atom_sampling": {
                    "enabled": True,
                    "epoch_atom_budget": 5000,
                    "epoch_num_batches": "auto",
                    "cycle_to_quota": True,
                    "domain_weights": {"protein": 50, "rna": 50},
                },
            },
            use_improved_sampler=True,
        )

        sampler_quota = _build_train_batch_sampler(
            train_dataset=dataset,
            config=config_quota,
            max_atoms_per_batch=1000,
            is_multi_domain=True,
            use_improved_sampler=True,
        )
        self.assertIsInstance(sampler_quota, ImprovedDynamicBatchSampler)
        self.assertEqual(sampler_quota.epoch_atom_budget, 5000)
        self.assertTrue(sampler_quota.cycle_short_domains)

    def test_ddp_partition_equalizes_odd_batches_by_splitting(self):
        # 3 batches naturally with splittable samples
        dataset = _metadata_dataset([
            ("protein", [300, 300, 300, 300, 300, 300]),
        ])
        common_kwargs = {
            "max_atoms_per_batch": 600,
            "enable_cross_modal_batches": False,
            "shuffle_chunks": False,
            "shuffle_within_chunk": False,
            "num_replicas": 2,
        }
        sampler_rank0 = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            rank=0,
            **common_kwargs,
        )
        sampler_rank1 = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            rank=1,
            **common_kwargs,
        )

        batches_0 = list(sampler_rank0)
        batches_1 = list(sampler_rank1)

        # 3 natural batches should be equalized to 4 batches (2 per rank)
        self.assertEqual(len(batches_0), 2)
        self.assertEqual(len(batches_1), 2)
        self.assertEqual(len(sampler_rank0), 2)
        self.assertEqual(len(sampler_rank1), 2)

        # Combined samples must contain all dataset samples exactly once
        flattened = [idx for b in (batches_0 + batches_1) for idx in b]
        self.assertEqual(len(flattened), len(dataset))
        self.assertEqual(set(flattened), set(range(len(dataset))))

    def test_ddp_partition_rebuilds_plan_cached_before_process_group_init(self):
        dataset = _metadata_dataset([
            ("protein", [300, 300, 300, 300, 300, 300]),
        ])
        sampler = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            max_atoms_per_batch=600,
            enable_cross_modal_batches=False,
            shuffle_chunks=False,
            shuffle_within_chunk=False,
        )

        self.assertEqual(len(sampler), 3)
        self.assertEqual(sampler._batch_plan_num_replicas, 1)

        with mock.patch.multiple(
            "torch.distributed",
            is_available=mock.Mock(return_value=True),
            is_initialized=mock.Mock(return_value=True),
            get_world_size=mock.Mock(return_value=2),
            get_rank=mock.Mock(return_value=0),
        ):
            rank_batches = list(sampler)
            self.assertEqual(sampler._batch_plan_num_replicas, 2)
            self.assertEqual(len(sampler._get_batch_plan()), 4)

        self.assertEqual(len(rank_batches), 2)

    def test_ddp_partition_uses_all_split_capacity_in_one_batch(self):
        dataset = _metadata_dataset([
            ("protein", [300, 300, 300]),
        ])
        samplers = [
            _sampler(
                ImprovedDynamicBatchSampler,
                dataset,
                max_atoms_per_batch=900,
                enable_cross_modal_batches=False,
                shuffle_chunks=False,
                shuffle_within_chunk=False,
                num_replicas=3,
                rank=rank,
            )
            for rank in range(3)
        ]

        rank_batches = [list(sampler) for sampler in samplers]
        self.assertEqual([len(batches) for batches in rank_batches], [1, 1, 1])
        flattened = [
            index
            for batches in rank_batches
            for batch in batches
            for index in batch
        ]
        self.assertEqual(len(flattened), len(dataset))
        self.assertEqual(set(flattened), set(range(len(dataset))))

    def test_ddp_partition_fails_fast_when_unsplittable_odd_batches(self):
        dataset = _metadata_dataset([
            ("protein", [600, 600, 600]),
        ])
        sampler_rank0 = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            max_atoms_per_batch=600,
            enable_cross_modal_batches=False,
            shuffle_chunks=False,
            shuffle_within_chunk=False,
            num_replicas=2,
            rank=0,
        )

        with self.assertRaisesRegex(RuntimeError, "DDP partition error"):
            list(sampler_rank0)

    def test_ddp_partition_fails_fast_when_world_size_exceeds_batch_count_and_unsplittable(self):
        dataset = _metadata_dataset([
            ("protein", [600, 600]),
        ])
        sampler_rank0 = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            max_atoms_per_batch=600,
            enable_cross_modal_batches=False,
            shuffle_chunks=False,
            shuffle_within_chunk=False,
            num_replicas=4,
            rank=0,
        )

        with self.assertRaisesRegex(RuntimeError, "DDP partition error"):
            list(sampler_rank0)

    def test_telemetry_mode_label_full_pass_and_quota(self):
        dataset = _metadata_dataset([
            ("protein", [100] * 10),
            ("rna", [50] * 10),
            ("unimol", [10] * 50),
        ])

        sampler_fp = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            max_atoms_per_batch=500,
            enable_cross_modal_batches=True,
            cycle_short_domains=False,
            epoch_atom_budget=None,
        )
        list(sampler_fp)
        self.assertEqual(sampler_fp.plan_stats["mode"], "full_pass")

        sampler_qa = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            max_atoms_per_batch=500,
            enable_cross_modal_batches=True,
            epoch_atom_budget=500,
            epoch_num_batches="auto",
            domain_atom_weights={"protein": 30, "rna": 25, "unimol": 45},
        )
        list(sampler_qa)
        self.assertEqual(sampler_qa.plan_stats["mode"], "auto")

        sampler_qf = _sampler(
            ImprovedDynamicBatchSampler,
            dataset,
            max_atoms_per_batch=500,
            enable_cross_modal_batches=True,
            epoch_atom_budget=500,
            epoch_num_batches=4,
            domain_atom_weights={"protein": 30, "rna": 25, "unimol": 45},
        )
        list(sampler_qf)
        self.assertEqual(sampler_qf.plan_stats["mode"], "fixed")


if __name__ == "__main__":
    unittest.main()

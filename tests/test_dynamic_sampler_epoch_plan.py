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

from core.train_pretrain import set_data_loader_epoch
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


if __name__ == "__main__":
    unittest.main()

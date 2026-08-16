#!/usr/bin/env python3
"""
Improved Dynamic Chunk-Aware Batch Sampler

This is a simplified, more efficient version of DynamicChunkAwareBatchSampler
that fixes critical performance issues:

1. REQUIRES metadata (no slow fallback to dataset access)
2. Simpler proportional sampling (no complex credit system)
3. No batch reordering overhead (yields batches on-the-fly)
4. Cache-friendly chunk ordering

Key Improvements:
- 10-100x faster batch creation (no disk access for atom counts)
- Lower memory overhead (no pre-computation of all batches)
- Simpler code (easier to debug and maintain)
- Better cache efficiency (chunk-aware sample ordering)
"""

import torch
import random
from torch.utils.data import BatchSampler, SequentialSampler
from typing import Iterator, List
from collections import defaultdict
from data_loading.lazy_universal_dataset import LazyUniversalDataset


class ImprovedDynamicBatchSampler(BatchSampler):
    """
    Improved batch sampler with atom-aware dynamic batching.

    Key features:
    - Requires metadata (no slow fallbacks)
    - Simple proportional sampling
    - Chunk-aware ordering for cache efficiency
    - Variable batch size based on atom count
    """

    def __init__(
        self,
        dataset,
        max_atoms_per_batch: int,
        shuffle_chunks: bool = True,
        shuffle_within_chunk: bool = True,
        seed: int = 42,
        enable_cross_modal_batches: bool = False,
        cycle_short_domains: bool = True,
        epoch_atom_budget: int | None = None,
        domain_atom_weights: dict[str, float] | None = None,
        epoch_num_batches: int | None = None,
    ):
        """
        Initialize ImprovedDynamicBatchSampler

        Args:
            dataset: LazyUniversalDataset instance (or Subset wrapping one)
            max_atoms_per_batch: Maximum atoms per batch
            shuffle_chunks: Whether to shuffle chunk order each epoch
            shuffle_within_chunk: Whether to shuffle samples within chunks
            seed: Random seed for reproducibility
            enable_cross_modal_batches: Mix different dataset types in batches
            cycle_short_domains: Reuse shorter domain samples in cross-modal mode
            epoch_atom_budget: Total train atoms requested per epoch. When set,
                cross-modal sampling is driven by per-domain atom quotas.
            domain_atom_weights: Positive per-domain weights used to divide the
                epoch atom budget. Keys must match metadata dataset types.
            epoch_num_batches: Fixed train batch-plan length. Natural greedy
                batches are split, never merged or padded with repeated graphs.

        Raises:
            ValueError: If metadata is not loaded
        """
        # Handle Subset from random_split
        from torch.utils.data import Subset
        if isinstance(dataset, Subset):
            self.subset_indices = dataset.indices
            self.original_dataset = dataset
            self.dataset = dataset.dataset
        else:
            self.subset_indices = None
            self.original_dataset = dataset
            self.dataset = dataset

        if not isinstance(self.dataset, LazyUniversalDataset):
            raise ValueError(f"Expected LazyUniversalDataset, got {type(self.dataset)}")

        # CRITICAL: Require metadata (no slow fallbacks!)
        if not hasattr(self.dataset, 'metadata_loaded') or not self.dataset.metadata_loaded:
            raise ValueError(
                "Metadata is required for atom-aware batching!\n"
                "Please run: python data_loading/create_chunk_metadata.py --chunk-dir <your_chunk_dir>\n"
                "Or initialize dataset with load_metadata=True if metadata files exist."
            )

        self.max_atoms_per_batch = max_atoms_per_batch
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_chunk = shuffle_within_chunk
        self.enable_cross_modal_batches = enable_cross_modal_batches
        self.cycle_short_domains = cycle_short_domains
        self.epoch_atom_budget = epoch_atom_budget
        self.domain_atom_weights = domain_atom_weights
        self.epoch_num_batches = epoch_num_batches
        self.epoch = 0
        self.seed = seed
        self._batch_plan = None
        self._batch_plan_epoch = None
        self.plan_stats = None

        if self.epoch_atom_budget is not None:
            if not self.enable_cross_modal_batches:
                raise ValueError("epoch_atom_budget requires cross-modal batching")
            if self.epoch_atom_budget <= 0:
                raise ValueError("epoch_atom_budget must be positive")
            if not self.domain_atom_weights:
                raise ValueError("domain_atom_weights are required with epoch_atom_budget")
            invalid_weights = {
                dtype: weight
                for dtype, weight in self.domain_atom_weights.items()
                if float(weight) <= 0
            }
            if invalid_weights:
                raise ValueError(f"Domain atom weights must be positive: {invalid_weights}")
            if self.epoch_num_batches is None or self.epoch_num_batches <= 0:
                raise ValueError(
                    "A positive epoch_num_batches is required with epoch_atom_budget"
                )

        # Validate dataset
        if not hasattr(self.dataset, 'chunk_pt_files'):
            raise ValueError("Dataset must be LazyUniversalDataset with chunk_pt_files")
        if not hasattr(self.dataset, 'file_ranges'):
            raise ValueError("Dataset must have file_ranges attribute")

        # Initialize base BatchSampler
        dummy_sampler = SequentialSampler(())
        super().__init__(dummy_sampler, batch_size=1, drop_last=False)

    def _get_atom_count_fast(self, global_idx: int) -> int:
        """Get atom count from metadata (FAST - no disk access)"""
        try:
            metadata = self.dataset.get_sample_metadata(global_idx)
            if metadata and 'num_atoms' in metadata:
                return metadata['num_atoms']
        except Exception as e:
            raise RuntimeError(
                f"Failed to get atom count for sample {global_idx}. "
                f"Metadata may be corrupted. Error: {e}"
            )

        # Fallback to conservative estimate (should rarely happen)
        return 500

    def _get_chunk_for_sample(self, global_idx: int) -> int:
        """Get which chunk a sample belongs to"""
        for chunk_idx, (start_idx, end_idx, _) in enumerate(self.dataset.file_ranges):
            if start_idx <= global_idx < end_idx:
                return chunk_idx
        return -1

    def _get_dataset_type(self, chunk_idx: int) -> str:
        """Get dataset type for a chunk"""
        if hasattr(self.dataset, 'chunk_metadata') and chunk_idx < len(self.dataset.chunk_metadata):
            metadata = self.dataset.chunk_metadata[chunk_idx]
            if isinstance(metadata, dict) and 'dataset_type' in metadata:
                return metadata['dataset_type']

        # Fallback: infer from filename
        if chunk_idx < len(self.dataset.file_ranges):
            _, _, pt_file = self.dataset.file_ranges[chunk_idx]
            filename = pt_file.lower()
            if 'pdb' in filename or 'protein' in filename:
                return 'protein'
            elif 'qm9' in filename:
                return 'qm9'
            elif 'metabolite' in filename:
                return 'metabolite'
            elif 'rna' in filename:
                return 'rna'

        return 'unknown'

    def _organize_samples_by_chunk(self) -> dict:
        """
        Organize samples by chunk with optional cross-modal grouping.

        Returns:
            Dict mapping chunk_idx -> list of sample indices
        """
        num_chunks = len(self.dataset.chunk_pt_files)

        # Setup subset mapping
        if self.subset_indices is not None:
            subset_set = set(self.subset_indices)
            global_to_subset_pos = {global_idx: pos for pos, global_idx in enumerate(self.subset_indices)}
        else:
            subset_set = None
            global_to_subset_pos = None

        # Group samples by chunk
        samples_by_chunk = defaultdict(list)

        for chunk_idx in range(num_chunks):
            start_idx, end_idx, _ = self.dataset.file_ranges[chunk_idx]
            chunk_indices = list(range(start_idx, end_idx))

            # Filter by subset
            if subset_set is not None:
                chunk_indices = [idx for idx in chunk_indices if idx in subset_set]

            if not chunk_indices:
                continue

            # Shuffle within chunk if requested
            if self.shuffle_within_chunk:
                rng = random.Random(self.seed + self.epoch * 10000 + chunk_idx)
                rng.shuffle(chunk_indices)

            # Convert to subset positions if needed
            if global_to_subset_pos is not None:
                chunk_indices = [global_to_subset_pos[idx] for idx in chunk_indices if idx in global_to_subset_pos]

            if chunk_indices:
                samples_by_chunk[chunk_idx] = chunk_indices

        return samples_by_chunk

    def _create_batches_standard(self, samples_by_chunk: dict) -> List[List[int]]:
        """
        Create batches in standard mode (single-domain).
        Processes chunks sequentially for optimal cache efficiency.
        """
        # Determine chunk order
        chunk_order = list(samples_by_chunk.keys())
        if self.shuffle_chunks:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            chunk_order = torch.tensor(chunk_order)[torch.randperm(len(chunk_order), generator=g)].tolist()

        # Create batches chunk-by-chunk
        batches = []

        for chunk_idx in chunk_order:
            samples = samples_by_chunk[chunk_idx]

            current_batch = []
            current_atoms = 0

            for subset_idx in samples:
                global_idx = self.subset_indices[subset_idx] if self.subset_indices is not None else subset_idx
                atom_count = self._get_atom_count_fast(global_idx)

                # Check if adding this sample exceeds limit
                if current_atoms + atom_count > self.max_atoms_per_batch and current_batch:
                    # Batch is full, save it and start new one
                    batches.append(current_batch)
                    current_batch = [subset_idx]
                    current_atoms = atom_count
                else:
                    # Add to current batch
                    current_batch.append(subset_idx)
                    current_atoms += atom_count

            # Add remaining batch
            if current_batch:
                batches.append(current_batch)

        return batches

    @staticmethod
    def _smooth_weighted_type_order(type_weights: dict) -> List[str]:
        """Create a deterministic smooth weighted order for one interleave round."""
        ordered_types = sorted(type_weights.keys())
        current_weights = {dtype: 0 for dtype in ordered_types}
        total_weight = sum(type_weights.values())
        type_order = []

        for _ in range(total_weight):
            for dtype in ordered_types:
                current_weights[dtype] += type_weights[dtype]

            selected_type = max(ordered_types, key=lambda dtype: current_weights[dtype])
            current_weights[selected_type] -= total_weight
            type_order.append(selected_type)

        return type_order

    def _create_single_pass_mixed_stream(self, samples_by_type: dict) -> List[int]:
        """Interleave all samples once without cycling any domain."""
        total_samples = sum(len(samples) for samples in samples_by_type.values())
        type_proportions = {
            dtype: len(samples) / total_samples
            for dtype, samples in samples_by_type.items()
        }
        type_positions = {dtype: 0 for dtype in samples_by_type}
        type_credits = type_proportions.copy()
        mixed_samples = []

        while any(type_positions[dtype] < len(samples_by_type[dtype]) for dtype in samples_by_type):
            available_types = [
                dtype for dtype in samples_by_type
                if type_positions[dtype] < len(samples_by_type[dtype])
            ]
            selected_type = max(available_types, key=lambda dtype: type_credits[dtype])
            mixed_samples.append(samples_by_type[selected_type][type_positions[selected_type]])
            type_positions[selected_type] += 1

            type_credits[selected_type] -= 1.0
            for dtype in samples_by_type:
                type_credits[dtype] += type_proportions[dtype]

        return mixed_samples

    def _create_cycled_mixed_stream(self, samples_by_type: dict) -> List[int]:
        """Interleave domains and wrap shorter streams when they run out."""
        total_samples = sum(len(samples) for samples in samples_by_type.values())
        type_proportions = {
            dtype: len(samples) / total_samples
            for dtype, samples in samples_by_type.items()
        }

        # Build a repeating weighted schedule. The round size intentionally stays
        # small so short domains re-enter the stream before a UniMol-only tail can form.
        interleave_window = 10
        samples_per_round = {
            dtype: max(1, int(type_proportions[dtype] * interleave_window))
            for dtype in samples_by_type
        }
        rounds = max(
            (len(samples) + samples_per_round[dtype] - 1) // samples_per_round[dtype]
            for dtype, samples in samples_by_type.items()
        )
        type_order = self._smooth_weighted_type_order(samples_per_round)

        mixed_samples = []
        type_positions = {dtype: 0 for dtype in samples_by_type}

        for _ in range(rounds):
            for dtype in type_order:
                samples = samples_by_type[dtype]
                sample_pos = type_positions[dtype] % len(samples)
                mixed_samples.append(samples[sample_pos])
                type_positions[dtype] += 1

        return mixed_samples

    def _resolve_atom_targets(self, dataset_types: set[str]) -> dict[str, int]:
        """Divide the exact epoch budget across active domains by configured weight."""
        missing = sorted(dataset_types - set(self.domain_atom_weights))
        if missing:
            raise ValueError(f"Missing atom weights for active domains: {missing}")

        weights = {
            dtype: float(self.domain_atom_weights[dtype])
            for dtype in sorted(dataset_types)
        }
        total_weight = sum(weights.values())
        raw_targets = {
            dtype: self.epoch_atom_budget * weight / total_weight
            for dtype, weight in weights.items()
        }
        targets = {dtype: int(raw_targets[dtype]) for dtype in weights}
        remainder = self.epoch_atom_budget - sum(targets.values())
        remainder_order = sorted(
            weights,
            key=lambda dtype: (raw_targets[dtype] - targets[dtype], dtype),
            reverse=True,
        )
        for dtype in remainder_order[:remainder]:
            targets[dtype] += 1
        return targets

    def _create_atom_quota_stream(self, samples_by_type: dict) -> List[int]:
        """Select whole graphs until every configured per-domain atom quota is met."""
        targets = self._resolve_atom_targets(set(samples_by_type))
        selected_atoms = {dtype: 0 for dtype in samples_by_type}
        selected_samples = {dtype: 0 for dtype in samples_by_type}
        positions = {dtype: 0 for dtype in samples_by_type}
        mixed_samples = []

        while True:
            eligible_types = [
                dtype
                for dtype in sorted(samples_by_type)
                if selected_atoms[dtype] < targets[dtype]
                and (
                    self.cycle_short_domains
                    or positions[dtype] < len(samples_by_type[dtype])
                )
            ]
            if not eligible_types:
                break

            selected_type = max(
                eligible_types,
                key=lambda dtype: (
                    (targets[dtype] - selected_atoms[dtype]) / targets[dtype]
                ),
            )
            domain_samples = samples_by_type[selected_type]
            sample_position = positions[selected_type]
            subset_idx = domain_samples[sample_position % len(domain_samples)]
            global_idx = (
                self.subset_indices[subset_idx]
                if self.subset_indices is not None
                else subset_idx
            )
            atom_count = int(self._get_atom_count_fast(global_idx))
            if atom_count <= 0:
                raise ValueError(
                    f"Invalid atom count for sample {global_idx}: {atom_count}"
                )

            mixed_samples.append(subset_idx)
            positions[selected_type] += 1
            selected_samples[selected_type] += 1
            selected_atoms[selected_type] += atom_count

        unmet = {
            dtype: targets[dtype] - selected_atoms[dtype]
            for dtype in targets
            if selected_atoms[dtype] < targets[dtype]
        }
        if unmet:
            raise ValueError(
                "Atom quotas cannot be met without cycling; "
                f"enable cycle_short_domains or lower the budget: {unmet}"
            )

        actual_total = sum(selected_atoms.values())
        self.plan_stats = {
            "epoch": self.epoch,
            "requested_total_atoms": self.epoch_atom_budget,
            "actual_total_atoms": actual_total,
            "domains": {
                dtype: {
                    "weight": float(self.domain_atom_weights[dtype]),
                    "target_atoms": targets[dtype],
                    "actual_atoms": selected_atoms[dtype],
                    "actual_atom_ratio": selected_atoms[dtype] / actual_total,
                    "selected_samples": selected_samples[dtype],
                    "unique_samples": min(
                        selected_samples[dtype], len(samples_by_type[dtype])
                    ),
                    "repeated_samples": max(
                        0, selected_samples[dtype] - len(samples_by_type[dtype])
                    ),
                    "available_samples": len(samples_by_type[dtype]),
                }
                for dtype in sorted(samples_by_type)
            },
        }
        return mixed_samples

    def _print_atom_quota_stats(self) -> None:
        if not self.plan_stats:
            return
        print(
            f"   Atom quota: requested={self.plan_stats['requested_total_atoms']:,}, "
            f"actual={self.plan_stats['actual_total_atoms']:,}"
        )
        for dtype, stats in self.plan_stats["domains"].items():
            print(
                f"   {dtype.upper():10s} target={stats['target_atoms']:,} "
                f"actual={stats['actual_atoms']:,} "
                f"ratio={100.0 * stats['actual_atom_ratio']:.2f}% "
                f"samples={stats['selected_samples']:,} "
                f"unique={stats['unique_samples']:,} "
                f"repeated={stats['repeated_samples']:,}"
            )

    def _atom_count_for_subset_index(self, subset_idx: int) -> int:
        global_idx = (
            self.subset_indices[subset_idx]
            if self.subset_indices is not None
            else subset_idx
        )
        return int(self._get_atom_count_fast(global_idx))

    def _split_batches_to_fixed_count(
        self, batches: List[List[int]]
    ) -> List[List[int]]:
        """Reach a stable epoch length by splitting batches without changing samples."""
        natural_batch_count = len(batches)
        if natural_batch_count > self.epoch_num_batches:
            raise ValueError(
                f"Natural atom-quota plan needs {natural_batch_count} batches, above "
                f"configured epoch_num_batches={self.epoch_num_batches}; raise the "
                "fixed count or lower the epoch atom budget"
            )

        batches = [list(batch) for batch in batches]
        while len(batches) < self.epoch_num_batches:
            splittable = [
                (index, batch)
                for index, batch in enumerate(batches)
                if len(batch) > 1
            ]
            if not splittable:
                raise ValueError(
                    f"Cannot split {len(batches)} non-empty batches to requested "
                    f"epoch_num_batches={self.epoch_num_batches}"
                )

            batch_index, batch = max(
                splittable,
                key=lambda item: (
                    sum(self._atom_count_for_subset_index(idx) for idx in item[1]),
                    len(item[1]),
                    -item[0],
                ),
            )
            atom_counts = [
                self._atom_count_for_subset_index(idx) for idx in batch
            ]
            total_atoms = sum(atom_counts)
            prefix_atoms = 0
            best_cut = 1
            best_difference = total_atoms
            for cut, atom_count in enumerate(atom_counts[:-1], start=1):
                prefix_atoms += atom_count
                difference = abs(total_atoms - 2 * prefix_atoms)
                if difference < best_difference:
                    best_cut = cut
                    best_difference = difference

            batches[batch_index:batch_index + 1] = [
                batch[:best_cut], batch[best_cut:]
            ]

        self.plan_stats["natural_batches"] = natural_batch_count
        self.plan_stats["planned_batches"] = len(batches)
        return batches

    def _create_batches_cross_modal(self, samples_by_chunk: dict) -> List[List[int]]:
        """
        Create batches in cross-modal mode (multi-domain).
        Optionally cycles shorter domains instead of dropping them after first pass.
        """
        # Group chunks by dataset type
        chunks_by_type = defaultdict(list)
        for chunk_idx in samples_by_chunk.keys():
            dtype = self._get_dataset_type(chunk_idx)
            chunks_by_type[dtype].append(chunk_idx)

        # Shuffle chunk order within each type
        if self.shuffle_chunks:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            for dtype in chunks_by_type:
                chunk_list = chunks_by_type[dtype]
                shuffled = torch.tensor(chunk_list)[torch.randperm(len(chunk_list), generator=g)].tolist()
                chunks_by_type[dtype] = shuffled

        samples_by_type = defaultdict(list)
        for dtype, chunk_list in chunks_by_type.items():
            for chunk_idx in chunk_list:
                samples_by_type[dtype].extend(samples_by_chunk[chunk_idx])

        samples_by_type = {
            dtype: samples
            for dtype, samples in samples_by_type.items()
            if samples
        }
        total_samples = sum(len(samples) for samples in samples_by_type.values())
        if total_samples == 0:
            return []

        if self.epoch_atom_budget is not None:
            mixed_samples = self._create_atom_quota_stream(samples_by_type)
        elif self.cycle_short_domains and len(samples_by_type) > 1:
            mixed_samples = self._create_cycled_mixed_stream(samples_by_type)
        else:
            mixed_samples = self._create_single_pass_mixed_stream(samples_by_type)

        # REMOVED: Global shuffle destroys chunk locality and causes cache thrashing
        # mixed_samples are already well-mixed from proportional interleaving
        # Keeping chunk locality is more important for performance

        # Create batches from mixed stream
        batches = []
        current_batch = []
        current_atoms = 0

        for subset_idx in mixed_samples:
            global_idx = self.subset_indices[subset_idx] if self.subset_indices is not None else subset_idx
            atom_count = self._get_atom_count_fast(global_idx)

            if current_atoms + atom_count > self.max_atoms_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = [subset_idx]
                current_atoms = atom_count
            else:
                current_batch.append(subset_idx)
                current_atoms += atom_count

        if current_batch:
            batches.append(current_batch)

        if self.epoch_atom_budget is not None:
            batches = self._split_batches_to_fixed_count(batches)

        return batches

    def _get_batch_plan(self) -> List[List[int]]:
        """Build one authoritative batch plan for the current epoch."""
        if self._batch_plan_epoch == self.epoch and self._batch_plan is not None:
            return self._batch_plan

        # Step 1: Organize samples by chunk
        print(f"\r⚡ Organizing samples by chunk...", end='', flush=True)
        samples_by_chunk = self._organize_samples_by_chunk()

        # Step 2: Create batches (mode-dependent)
        print(f"\r⚡ Creating batches (cross-modal={self.enable_cross_modal_batches})...", end='', flush=True)
        if self.enable_cross_modal_batches:
            batches = self._create_batches_cross_modal(samples_by_chunk)
        else:
            batches = self._create_batches_standard(samples_by_chunk)

        self._batch_plan = batches
        self._batch_plan_epoch = self.epoch
        print(f"\r✅ Ready! {len(batches)} batches created.                    ")
        self._print_atom_quota_stats()
        return self._batch_plan

    def __iter__(self) -> Iterator[List[int]]:
        """Yield the same authoritative plan reported by ``__len__``."""
        batches = self._get_batch_plan()

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        """Return the exact number of batches in the current epoch plan."""
        return len(self._get_batch_plan())

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling"""
        if epoch == self.epoch:
            return
        self.epoch = epoch
        self._batch_plan = None
        self._batch_plan_epoch = None
        self.plan_stats = None

#!/usr/bin/env python3
"""
Protein Per-Atom and Per-Amino-Acid Metrics Evaluation Callback

PyTorch Lightning callback that mirrors the RNA per-atom metrics callback for
the protein domain. Differences from the RNA version:

1. Filters to protein batches only (dataset_type in {'pdb', 'protein'}).
2. Uses the 35-atom protein vocabulary from core.metrics.protein_atom_types.
3. Logs under a distinct W&B namespace (val_protein_atom_metrics/* and
   val_protein_aa_metrics/*) so runs that include both domains don't collide.
4. Adds a per-amino-acid accuracy metric alongside the per-atom metrics.

Per-atom fix (same design as the RNA callback):
- Autocast wrapper around pl_module.forward so bf16 mixed-precision training
  doesn't trip a silent dtype mismatch.
- Element-aware y_pred mapping: the MLM head predicts atomic number, so for
  each position we compare predicted element to the ground-truth element
  (derived from pos_code); the y_pred vector is set to y_true when the
  elements match and to an out-of-vocab index when they differ. This keeps
  Recall meaningful as per-atom element accuracy without fuzzy heuristics.

Per-amino-acid metric:
- For every residue that has at least one masked atom in the batch, we
  classify the residue as "correctly predicted" iff ALL masked atoms in that
  residue have the correct element prediction. We then aggregate per amino
  acid type (ALA, ARG, ..., VAL). This is inflated under the current %15
  atom-level MLM mask (most of the residue is visible) but still provides a
  relative signal. Replacing the mask with full-residue masking in a future
  iteration would make the metric a true AA recognition signal.
"""

import torch
import numpy as np
from itertools import chain
from typing import Dict, List, Optional
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

from core.metrics.per_atom_metrics import PerAtomMetricsCalculator
from core.metrics.protein_atom_types import (
    ALL_PROTEIN_ATOMS,
    PROTEIN_ATOM_GROUPS,
    ATOMIC_NUM_TO_ELEMENT,
    STANDARD_AMINO_ACIDS,
    pos_code_to_element,
    pos_code_to_atom_class,
    is_backbone_atom,
    is_sidechain_slot,
    is_virtual_atom,
)


# Dataset type tags that identify protein batches (case-insensitive match).
PROTEIN_DATASET_TYPES = {'pdb', 'protein'}

# W&B namespaces — kept as module-level constants so they can be referenced
# from tests or external tooling without instantiating the callback.
WANDB_NAMESPACE_ATOM = 'protein_atom_metrics'
WANDB_NAMESPACE_AA = 'protein_aa_metrics'
WANDB_NAMESPACE_AA_BLOCK = 'protein_aa_block'


class ProteinPerAtomMetricsCallback(Callback):
    """PyTorch Lightning callback for protein per-atom and per-AA metrics."""

    def __init__(
        self,
        enabled: bool = True,
        only_protein: bool = True,
        compute_groups: bool = True,
        compute_per_aa: bool = True,
        compute_per_aa_block: bool = True,
        log_every_n_epochs: int = 1,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.only_protein = only_protein
        self.compute_groups = compute_groups
        self.compute_per_aa = compute_per_aa
        self.compute_per_aa_block = compute_per_aa_block
        self.log_every_n_epochs = log_every_n_epochs

        # Per-atom calculators (reuse the generic calculator from rna callback).
        self.val_metrics = PerAtomMetricsCalculator(
            atom_types=ALL_PROTEIN_ATOMS,
            compute_groups=compute_groups,
            group_type='backbone_vs_sidechain',
        )
        # Override groupings to use protein-specific ones.
        self.val_metrics.groups = PROTEIN_ATOM_GROUPS['backbone_vs_sidechain']
        self.test_metrics = PerAtomMetricsCalculator(
            atom_types=ALL_PROTEIN_ATOMS,
            compute_groups=compute_groups,
            group_type='backbone_vs_sidechain',
        )
        self.test_metrics.groups = PROTEIN_ATOM_GROUPS['backbone_vs_sidechain']

        # Per-AA statistics — {aa_name: {'correct': int, 'total': int}}
        # Binary 1/0 per residue: residue counts as "correct" iff ALL masked
        # atoms in it got the right element. Preserved as-is.
        self._val_aa_stats = self._fresh_aa_stats()
        self._test_aa_stats = self._fresh_aa_stats()

        # Per-AA block statistics — atom-level accuracy split by
        # backbone vs sidechain slots. Per residue we count how many masked
        # atoms were correct out of how many were masked. Works for both
        # atom-level random MLM masks (sparse residue coverage) and future
        # block-level masks (full residue coverage).
        self._val_aa_block_stats = self._fresh_aa_block_stats()
        self._test_aa_block_stats = self._fresh_aa_block_stats()

        # Snapshot of the last validation metrics (used for on_train_end print).
        self._last_snapshot: Optional[Dict] = None

        if enabled:
            print("✅ ProteinPerAtomMetricsCallback enabled")
            print(f"   - Atom types:        {len(ALL_PROTEIN_ATOMS)} (4 backbone + 30 sidechain + 1 virtual)")
            print(f"   - Group metrics:     {compute_groups}")
            print(f"   - Per-AA binary:     {compute_per_aa}")
            print(f"   - Per-AA block (bb/sc): {compute_per_aa_block}")
            print(f"   - Protein-only:      {only_protein}")
            print(f"   - Log frequency:     every {log_every_n_epochs} epoch(s)")

    # ------------------------------------------------------------------ #
    # Batch hooks                                                         #
    # ------------------------------------------------------------------ #

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        if not self.enabled:
            return
        if self.only_protein and not self._is_protein_batch(batch):
            return
        try:
            self._process_batch_metrics(
                batch, pl_module, self.val_metrics,
                self._val_aa_stats, self._val_aa_block_stats,
            )
        except Exception:
            # Silent — don't break training if metric computation hits a
            # transient issue. Replace with a one-shot logger warning if
            # you need to diagnose.
            pass

    def on_test_batch_end(self, trainer, pl_module, outputs, batch,
                          batch_idx, dataloader_idx=0):
        if not self.enabled:
            return
        if self.only_protein and not self._is_protein_batch(batch):
            return
        try:
            self._process_batch_metrics(
                batch, pl_module, self.test_metrics,
                self._test_aa_stats, self._test_aa_block_stats,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Core metric computation                                             #
    # ------------------------------------------------------------------ #

    def _process_batch_metrics(self, batch, pl_module, metrics_calculator,
                                aa_stats, aa_block_stats):
        # Guard: MLM task must be active and pos_code must exist on the batch.
        if "mlm" not in pl_module.config.pretraining_tasks:
            return
        if not hasattr(batch, 'pos_code') or batch.pos_code is None:
            return

        # 1. Forward pass under autocast so bf16 training doesn't dtype-mismatch.
        use_bf16 = getattr(pl_module.config, 'use_bfloat16', False)
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad(), torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            _, node_emb = pl_module.forward(batch)
            mlm_logits = pl_module.pretraining_tasks.mlm_head(node_emb)
            preds_atomic_num = mlm_logits.argmax(dim=1).cpu().numpy()

        # 2. Flatten pos_code — PyG stores batch.pos_code as list-of-lists when
        #    multiple molecules are collated.
        raw_pos_codes = batch.pos_code
        if (isinstance(raw_pos_codes, list)
                and len(raw_pos_codes) > 0
                and isinstance(raw_pos_codes[0], (list, tuple))):
            flattened_pos_codes = list(chain.from_iterable(raw_pos_codes))
        else:
            flattened_pos_codes = raw_pos_codes

        if len(flattened_pos_codes) != len(preds_atomic_num):
            # Defensive: size mismatch means something upstream is off — skip.
            return

        # 3. Build y_true (protein atom class indices) and element arrays.
        y_true_indices = np.array(
            [ALL_PROTEIN_ATOMS.index(pos_code_to_atom_class(c)) for c in flattened_pos_codes]
        )
        pred_elements = np.array(
            [ATOMIC_NUM_TO_ELEMENT.get(int(n), 'V') for n in preds_atomic_num]
        )
        true_elements = np.array(
            [pos_code_to_element(c) for c in flattened_pos_codes]
        )
        is_correct_element = (pred_elements == true_elements)

        # 4. Element-aware y_pred: set to y_true when element matches, else to
        #    an out-of-vocab idx so one-vs-rest precision stays clean.
        out_of_vocab = len(ALL_PROTEIN_ATOMS)  # 35 — outside idx range [0, 34]
        y_pred_indices = np.where(is_correct_element, y_true_indices, out_of_vocab)

        # 5. Apply MLM mask if present; only masked atoms are MLM-supervised.
        mlm_mask_np: Optional[np.ndarray] = None
        if hasattr(batch, 'mlm_mask') and batch.mlm_mask is not None:
            mlm_mask_np = batch.mlm_mask.cpu().numpy().astype(bool)
            if len(mlm_mask_np) == len(y_true_indices):
                y_true_masked = y_true_indices[mlm_mask_np]
                y_pred_masked = y_pred_indices[mlm_mask_np]
            else:
                # Size mismatch → fall back to no masking
                y_true_masked = y_true_indices
                y_pred_masked = y_pred_indices
                mlm_mask_np = None
        else:
            y_true_masked = y_true_indices
            y_pred_masked = y_pred_indices

        if len(y_true_masked) > 0:
            metrics_calculator.update(y_pred_masked, y_true_masked)

        # 6. Per-AA metric: for each residue that has ≥1 masked atom, check
        #    whether ALL masked atoms in that residue got correct element.
        if self.compute_per_aa and mlm_mask_np is not None:
            self._accumulate_aa_stats(
                batch=batch,
                mlm_mask_np=mlm_mask_np,
                is_correct_element=is_correct_element,
                aa_stats=aa_stats,
            )

        # 7. Per-AA block metric: same residue grouping, but count atoms
        #    instead of whole residues. Backbone (N/CA/C/O), sidechain slot
        #    (real sidechain OR virtual padding), and sidechain-real-only are
        #    tracked separately so we can compare MIND's slot-recognition vs
        #    Salad-style real-atom recognition.
        if self.compute_per_aa_block and mlm_mask_np is not None:
            self._accumulate_aa_block_stats(
                batch=batch,
                flattened_pos_codes=flattened_pos_codes,
                mlm_mask_np=mlm_mask_np,
                is_correct_element=is_correct_element,
                aa_block_stats=aa_block_stats,
            )

    def _accumulate_aa_stats(self, batch, mlm_mask_np, is_correct_element, aa_stats):
        """Aggregate per-residue correctness into per-AA stats."""
        if not hasattr(batch, 'block_idx') or not hasattr(batch, 'block_symbols'):
            return
        if not hasattr(batch, 'batch') or batch.batch is None:
            # Can't group atoms by molecule without PyG's batch index.
            return

        mol_idx_per_atom = batch.batch.cpu().numpy()
        block_idx_per_atom = batch.block_idx.cpu().numpy()

        # block_symbols is stored per molecule as a list; PyG collates a list
        # attribute by concatenation, so we expect list-of-lists here.
        raw_block_symbols = batch.block_symbols
        per_mol_symbols: List[List[str]]
        if (isinstance(raw_block_symbols, list)
                and len(raw_block_symbols) > 0
                and isinstance(raw_block_symbols[0], list)):
            per_mol_symbols = raw_block_symbols
        else:
            # Flat fallback — assume single molecule or already-flat list.
            per_mol_symbols = [list(raw_block_symbols)]

        # Unique (molecule, block) pairs that contain at least one masked atom.
        n_atoms = len(mol_idx_per_atom)
        # Encode the pair as a single int for np.unique grouping.
        # Assumption: block_idx per molecule < 100_000 (protein residues cap).
        residue_key = mol_idx_per_atom.astype(np.int64) * 100_000 + block_idx_per_atom.astype(np.int64)
        masked_keys = residue_key[mlm_mask_np] if mlm_mask_np is not None else residue_key
        unique_masked_residues = np.unique(masked_keys)

        # Per-residue check: are all masked atoms in this residue correct?
        for rkey in unique_masked_residues:
            mol_i = int(rkey // 100_000)
            block_i = int(rkey % 100_000)

            # Mask for atoms that belong to this residue AND are masked.
            residue_mask = (residue_key == rkey)
            residue_masked_atoms = residue_mask & mlm_mask_np

            if not residue_masked_atoms.any():
                continue

            all_correct = bool(is_correct_element[residue_masked_atoms].all())

            # Resolve true AA symbol. Bounds-check both indices.
            if mol_i >= len(per_mol_symbols):
                continue
            mol_syms = per_mol_symbols[mol_i]
            if block_i >= len(mol_syms):
                continue
            true_aa = str(mol_syms[block_i]).upper().strip()

            bucket = true_aa if true_aa in STANDARD_AMINO_ACIDS else 'UNKNOWN'
            aa_stats[bucket]['total'] += 1
            if all_correct:
                aa_stats[bucket]['correct'] += 1

    def _accumulate_aa_block_stats(self, batch, flattened_pos_codes,
                                    mlm_mask_np, is_correct_element, aa_block_stats):
        """Aggregate atom-level backbone/sidechain accuracy per AA type.

        For each residue that has ≥1 masked atom:
        - Count masked backbone atoms (N/CA/C/O) → bb_total; correct ones → bb_correct.
        - Count masked sidechain-slot atoms (real sidechain OR virtual) → sc_total;
          correct ones → sc_correct.
        - Same for real sidechain atoms only (exclude virtual) → sc_real_*.
        - Increment residues_seen once per residue that contributed any masked atom.

        Works regardless of mask pattern:
        - Atom-level random mask (current): residue contributes ~1-2 atoms per residue;
          counts accumulate across the epoch.
        - Block mask (future): residue contributes its full 4 backbone + 10 sidechain
          slot atoms, giving clean per-residue "5/14" style numbers.
        """
        if not hasattr(batch, 'block_idx') or not hasattr(batch, 'block_symbols'):
            return
        if not hasattr(batch, 'batch') or batch.batch is None:
            return

        mol_idx_per_atom = batch.batch.cpu().numpy()
        block_idx_per_atom = batch.block_idx.cpu().numpy()

        raw_block_symbols = batch.block_symbols
        if (isinstance(raw_block_symbols, list)
                and len(raw_block_symbols) > 0
                and isinstance(raw_block_symbols[0], list)):
            per_mol_symbols = raw_block_symbols
        else:
            per_mol_symbols = [list(raw_block_symbols)]

        # Pre-classify every atom's pos_code → 'bb' / 'sc_slot' (= 'sc_virtual' or 'sc_real').
        # Stored as parallel numpy arrays for masked-AND filtering.
        n_atoms = len(flattened_pos_codes)
        is_bb = np.zeros(n_atoms, dtype=bool)
        is_sc_slot = np.zeros(n_atoms, dtype=bool)
        is_sc_real = np.zeros(n_atoms, dtype=bool)
        for i, code in enumerate(flattened_pos_codes):
            if is_backbone_atom(code):
                is_bb[i] = True
            elif is_sidechain_slot(code):
                is_sc_slot[i] = True
                if not is_virtual_atom(code):
                    is_sc_real[i] = True

        # Group atoms by (molecule, residue).
        residue_key = (mol_idx_per_atom.astype(np.int64) * 100_000
                       + block_idx_per_atom.astype(np.int64))
        masked_keys = residue_key[mlm_mask_np]
        unique_masked_residues = np.unique(masked_keys)

        for rkey in unique_masked_residues:
            mol_i = int(rkey // 100_000)
            block_i = int(rkey % 100_000)

            residue_mask = (residue_key == rkey)
            masked_in_residue = residue_mask & mlm_mask_np

            if not masked_in_residue.any():
                continue

            if mol_i >= len(per_mol_symbols):
                continue
            mol_syms = per_mol_symbols[mol_i]
            if block_i >= len(mol_syms):
                continue
            true_aa = str(mol_syms[block_i]).upper().strip()
            bucket = true_aa if true_aa in STANDARD_AMINO_ACIDS else 'UNKNOWN'

            # Backbone masked atoms in this residue.
            bb_mask = masked_in_residue & is_bb
            bb_total = int(bb_mask.sum())
            bb_correct = int(is_correct_element[bb_mask].sum())

            # Sidechain-slot masked atoms (includes virtual padding).
            sc_mask = masked_in_residue & is_sc_slot
            sc_total = int(sc_mask.sum())
            sc_correct = int(is_correct_element[sc_mask].sum())

            # Sidechain real-only masked atoms (excludes virtual).
            sc_real_mask = masked_in_residue & is_sc_real
            sc_real_total = int(sc_real_mask.sum())
            sc_real_correct = int(is_correct_element[sc_real_mask].sum())

            stats = aa_block_stats[bucket]
            stats['bb_correct'] += bb_correct
            stats['bb_total'] += bb_total
            stats['sc_correct'] += sc_correct
            stats['sc_total'] += sc_total
            stats['sc_real_correct'] += sc_real_correct
            stats['sc_real_total'] += sc_real_total
            stats['residues_seen'] += 1

    # ------------------------------------------------------------------ #
    # Epoch-end hooks                                                     #
    # ------------------------------------------------------------------ #

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.enabled:
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        if len(self.val_metrics.all_predictions) == 0:
            return

        metrics = self.val_metrics.compute()
        aa_accuracies = self._finalize_aa_stats(self._val_aa_stats)
        aa_block_accuracies = self._finalize_aa_block_stats(self._val_aa_block_stats)

        self._log_metrics(trainer, metrics, aa_accuracies, aa_block_accuracies, stage='val')

        # Snapshot for on_train_end. Deep-copy the dicts so subsequent resets
        # don't overwrite it.
        self._last_snapshot = {
            'epoch': trainer.current_epoch,
            'per_atom': dict(self.val_metrics.per_atom_metrics),
            'per_group': dict(self.val_metrics.per_group_metrics),
            'overall': dict(self.val_metrics.overall_metrics),
            'aa': {k: dict(v) for k, v in aa_accuracies.items()},
            'aa_block': {k: dict(v) for k, v in aa_block_accuracies.items()},
        }

        # Periodic printable summary (same cadence as the RNA callback).
        if trainer.current_epoch % (self.log_every_n_epochs * 5) == 0:
            self._print_summary(
                header=f"PROTEIN VALIDATION METRICS — Epoch {trainer.current_epoch}",
                calculator=self.val_metrics,
                aa_accuracies=aa_accuracies,
                aa_block_accuracies=aa_block_accuracies,
            )

        self.val_metrics.reset()
        self._val_aa_stats = self._fresh_aa_stats()
        self._val_aa_block_stats = self._fresh_aa_block_stats()

    def on_test_epoch_end(self, trainer, pl_module):
        if not self.enabled:
            return
        if len(self.test_metrics.all_predictions) == 0:
            return

        metrics = self.test_metrics.compute()
        aa_accuracies = self._finalize_aa_stats(self._test_aa_stats)
        aa_block_accuracies = self._finalize_aa_block_stats(self._test_aa_block_stats)

        self._log_metrics(trainer, metrics, aa_accuracies, aa_block_accuracies, stage='test')
        self._print_summary(
            header=f"PROTEIN TEST METRICS — Epoch {trainer.current_epoch}",
            calculator=self.test_metrics,
            aa_accuracies=aa_accuracies,
            aa_block_accuracies=aa_block_accuracies,
        )

        self.test_metrics.reset()
        self._test_aa_stats = self._fresh_aa_stats()
        self._test_aa_block_stats = self._fresh_aa_block_stats()

    def on_train_end(self, trainer, pl_module):
        """Print the final validation snapshot so it's easy to grep in the log."""
        if not self.enabled or self._last_snapshot is None:
            return
        print("\n" + "=" * 100)
        print(
            f"🏁 FINAL PROTEIN PER-ATOM METRICS "
            f"(last validation, epoch {self._last_snapshot['epoch']})"
        )
        print("=" * 100)
        # Restore the calculator state from the snapshot and reuse its printer.
        self.val_metrics.per_atom_metrics = self._last_snapshot['per_atom']
        self.val_metrics.per_group_metrics = self._last_snapshot['per_group']
        self.val_metrics.overall_metrics = self._last_snapshot['overall']
        self.val_metrics.print_summary()
        self._print_per_aa_table(self._last_snapshot['aa'])
        if self.compute_per_aa_block:
            self._print_per_aa_block_table(self._last_snapshot.get('aa_block', {}))

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fresh_aa_stats() -> Dict[str, Dict[str, int]]:
        stats = {aa: {'correct': 0, 'total': 0} for aa in STANDARD_AMINO_ACIDS}
        stats['UNKNOWN'] = {'correct': 0, 'total': 0}
        return stats

    @staticmethod
    def _finalize_aa_stats(aa_stats: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        """Compute accuracies from {correct, total} counts."""
        result: Dict[str, Dict[str, float]] = {}
        for aa, counts in aa_stats.items():
            total = counts['total']
            if total == 0:
                continue  # Skip amino acids not seen in this epoch's masks.
            result[aa] = {
                'accuracy': counts['correct'] / total,
                'correct': counts['correct'],
                'total': total,
            }
        return result

    @staticmethod
    def _fresh_aa_block_stats() -> Dict[str, Dict[str, int]]:
        keys = ['bb_correct', 'bb_total',
                'sc_correct', 'sc_total',
                'sc_real_correct', 'sc_real_total',
                'residues_seen']
        stats = {aa: {k: 0 for k in keys} for aa in STANDARD_AMINO_ACIDS}
        stats['UNKNOWN'] = {k: 0 for k in keys}
        return stats

    @staticmethod
    def _finalize_aa_block_stats(aa_block_stats: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        """Compute backbone / sidechain / sidechain-real accuracies per AA."""
        result: Dict[str, Dict[str, float]] = {}
        for aa, counts in aa_block_stats.items():
            if counts['residues_seen'] == 0:
                continue
            bb_total = counts['bb_total']
            sc_total = counts['sc_total']
            sc_real_total = counts['sc_real_total']
            result[aa] = {
                'backbone_accuracy': (counts['bb_correct'] / bb_total) if bb_total > 0 else 0.0,
                'sidechain_accuracy': (counts['sc_correct'] / sc_total) if sc_total > 0 else 0.0,
                'sidechain_real_accuracy': (counts['sc_real_correct'] / sc_real_total) if sc_real_total > 0 else 0.0,
                'bb_correct': counts['bb_correct'],
                'bb_total': bb_total,
                'sc_correct': counts['sc_correct'],
                'sc_total': sc_total,
                'sc_real_correct': counts['sc_real_correct'],
                'sc_real_total': sc_real_total,
                'residues_seen': counts['residues_seen'],
            }
        return result

    @staticmethod
    def _is_protein_batch(batch) -> bool:
        if not hasattr(batch, 'dataset_type') or batch.dataset_type is None:
            return False
        dtypes = batch.dataset_type
        if isinstance(dtypes, (list, tuple)):
            return any(str(dt).lower() in PROTEIN_DATASET_TYPES for dt in dtypes)
        return str(dtypes).lower() in PROTEIN_DATASET_TYPES

    def _log_metrics(self, trainer, metrics, aa_accuracies, aa_block_accuracies, stage: str):
        """Push per-atom + per-AA (binary) + per-AA-block (atom-level) metrics to W&B."""
        wandb_logger = next(
            (lg for lg in trainer.loggers if isinstance(lg, WandbLogger)), None
        )
        if wandb_logger is None:
            return

        prefix = f"{stage}_"

        # Per-atom metrics — reuse the generic calculator's wandb formatter but
        # point it at the protein namespace.
        calc = self.val_metrics if stage == 'val' else self.test_metrics
        flat = calc.get_wandb_logs(prefix=prefix, namespace=WANDB_NAMESPACE_ATOM)

        # Per-AA binary metric — one entry per AA + an overall aggregate.
        if self.compute_per_aa and aa_accuracies:
            total_correct = sum(v['correct'] for v in aa_accuracies.values())
            total_count = sum(v['total'] for v in aa_accuracies.values())
            overall = total_correct / total_count if total_count > 0 else 0.0
            flat[f"{prefix}{WANDB_NAMESPACE_AA}/overall_accuracy"] = overall
            flat[f"{prefix}{WANDB_NAMESPACE_AA}/total_residues"] = total_count

            for aa, vals in aa_accuracies.items():
                flat[f"{prefix}{WANDB_NAMESPACE_AA}/AA_{aa}/accuracy"] = vals['accuracy']
                # Support value kept as separate key so users can spot sparse AAs.
                flat[f"{prefix}{WANDB_NAMESPACE_AA}/AA_{aa}/support"] = vals['total']

        # Per-AA block metric — atom-level backbone / sidechain accuracy per AA.
        if self.compute_per_aa_block and aa_block_accuracies:
            bb_correct_total = sum(v['bb_correct'] for v in aa_block_accuracies.values())
            bb_total_total = sum(v['bb_total'] for v in aa_block_accuracies.values())
            sc_correct_total = sum(v['sc_correct'] for v in aa_block_accuracies.values())
            sc_total_total = sum(v['sc_total'] for v in aa_block_accuracies.values())
            sc_real_correct_total = sum(v['sc_real_correct'] for v in aa_block_accuracies.values())
            sc_real_total_total = sum(v['sc_real_total'] for v in aa_block_accuracies.values())
            residues_seen_total = sum(v['residues_seen'] for v in aa_block_accuracies.values())

            ns = WANDB_NAMESPACE_AA_BLOCK
            flat[f"{prefix}{ns}/overall_backbone_accuracy"] = (
                bb_correct_total / bb_total_total if bb_total_total > 0 else 0.0
            )
            flat[f"{prefix}{ns}/overall_sidechain_accuracy"] = (
                sc_correct_total / sc_total_total if sc_total_total > 0 else 0.0
            )
            flat[f"{prefix}{ns}/overall_sidechain_real_accuracy"] = (
                sc_real_correct_total / sc_real_total_total if sc_real_total_total > 0 else 0.0
            )
            flat[f"{prefix}{ns}/total_residues"] = residues_seen_total
            flat[f"{prefix}{ns}/total_bb_atoms"] = bb_total_total
            flat[f"{prefix}{ns}/total_sc_atoms"] = sc_total_total
            flat[f"{prefix}{ns}/total_sc_real_atoms"] = sc_real_total_total

            for aa, vals in aa_block_accuracies.items():
                flat[f"{prefix}{ns}/AA_{aa}/backbone_accuracy"] = vals['backbone_accuracy']
                flat[f"{prefix}{ns}/AA_{aa}/sidechain_accuracy"] = vals['sidechain_accuracy']
                flat[f"{prefix}{ns}/AA_{aa}/sidechain_real_accuracy"] = vals['sidechain_real_accuracy']
                flat[f"{prefix}{ns}/AA_{aa}/bb_total"] = vals['bb_total']
                flat[f"{prefix}{ns}/AA_{aa}/sc_total"] = vals['sc_total']
                flat[f"{prefix}{ns}/AA_{aa}/sc_real_total"] = vals['sc_real_total']
                flat[f"{prefix}{ns}/AA_{aa}/residues_seen"] = vals['residues_seen']

        for key, value in flat.items():
            wandb_logger.log_metrics({key: value}, step=trainer.global_step)

    def _print_summary(self, header: str, calculator: PerAtomMetricsCalculator,
                       aa_accuracies, aa_block_accuracies=None):
        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)
        calculator.print_summary()
        if self.compute_per_aa:
            self._print_per_aa_table(aa_accuracies)
        if self.compute_per_aa_block and aa_block_accuracies:
            self._print_per_aa_block_table(aa_block_accuracies)

    @staticmethod
    def _print_per_aa_table(aa_accuracies: Dict[str, Dict[str, float]]) -> None:
        if not aa_accuracies:
            print("\n(no residues with masked atoms observed — per-AA table empty)\n")
            return
        try:
            from tabulate import tabulate
            rows = [
                [aa, f"{vals['accuracy']:.4f}", int(vals['correct']), int(vals['total'])]
                for aa, vals in sorted(aa_accuracies.items())
            ]
            table = tabulate(
                rows,
                headers=['AA', 'Accuracy', 'Correct', 'Total'],
                tablefmt='grid',
            )
        except ImportError:
            table_lines = ['AA  | Accuracy | Correct | Total']
            for aa, vals in sorted(aa_accuracies.items()):
                table_lines.append(
                    f"{aa:4s} | {vals['accuracy']:.4f}   | "
                    f"{int(vals['correct']):7d} | {int(vals['total']):5d}"
                )
            table = "\n".join(table_lines)

        total_correct = sum(v['correct'] for v in aa_accuracies.values())
        total_count = sum(v['total'] for v in aa_accuracies.values())
        overall = total_correct / total_count if total_count > 0 else 0.0

        print("\n📊 PER-AMINO-ACID METRICS — binary 1/0 (residue fully correct on masked atoms):")
        print(table)
        print(f"\n   Overall residue accuracy: {overall:.4f}  "
              f"({total_correct} / {total_count})")

    @staticmethod
    def _print_per_aa_block_table(aa_block_accuracies: Dict[str, Dict[str, float]]) -> None:
        """Print atom-level backbone / sidechain accuracy per AA.

        Columns:
          BB acc:  bb_correct / bb_total  (N/CA/C/O atoms only)
          SC acc:  sc_correct / sc_total  (10-slot sidechain incl. virtual padding)
          SCr acc: sc_real_correct / sc_real_total  (real sidechain atoms only)
          Res:     number of residues that contributed any masked atom
        """
        if not aa_block_accuracies:
            print("\n(no residues with masked atoms observed — per-AA-block table empty)\n")
            return
        try:
            from tabulate import tabulate
            rows = []
            for aa, v in sorted(aa_block_accuracies.items()):
                rows.append([
                    aa,
                    f"{v['backbone_accuracy']:.4f}",
                    f"{int(v['bb_correct'])}/{int(v['bb_total'])}",
                    f"{v['sidechain_accuracy']:.4f}",
                    f"{int(v['sc_correct'])}/{int(v['sc_total'])}",
                    f"{v['sidechain_real_accuracy']:.4f}",
                    f"{int(v['sc_real_correct'])}/{int(v['sc_real_total'])}",
                    int(v['residues_seen']),
                ])
            table = tabulate(
                rows,
                headers=['AA', 'BB acc', 'BB c/t', 'SC acc', 'SC c/t',
                         'SCr acc', 'SCr c/t', 'Res'],
                tablefmt='grid',
            )
        except ImportError:
            lines = ['AA  | BB acc | BB c/t | SC acc | SC c/t | SCr acc | SCr c/t | Res']
            for aa, v in sorted(aa_block_accuracies.items()):
                lines.append(
                    f"{aa:4s} | {v['backbone_accuracy']:.4f} | "
                    f"{int(v['bb_correct'])}/{int(v['bb_total'])} | "
                    f"{v['sidechain_accuracy']:.4f} | "
                    f"{int(v['sc_correct'])}/{int(v['sc_total'])} | "
                    f"{v['sidechain_real_accuracy']:.4f} | "
                    f"{int(v['sc_real_correct'])}/{int(v['sc_real_total'])} | "
                    f"{int(v['residues_seen']):5d}"
                )
            table = "\n".join(lines)

        bb_c = sum(v['bb_correct'] for v in aa_block_accuracies.values())
        bb_t = sum(v['bb_total'] for v in aa_block_accuracies.values())
        sc_c = sum(v['sc_correct'] for v in aa_block_accuracies.values())
        sc_t = sum(v['sc_total'] for v in aa_block_accuracies.values())
        scr_c = sum(v['sc_real_correct'] for v in aa_block_accuracies.values())
        scr_t = sum(v['sc_real_total'] for v in aa_block_accuracies.values())
        res = sum(v['residues_seen'] for v in aa_block_accuracies.values())

        print("\n📊 PER-AMINO-ACID BLOCK METRICS — atom-level (BB / SC / SC-real):")
        print(table)
        print(
            f"\n   Overall BB: {(bb_c/bb_t if bb_t else 0):.4f} ({bb_c}/{bb_t})  |  "
            f"SC: {(sc_c/sc_t if sc_t else 0):.4f} ({sc_c}/{sc_t})  |  "
            f"SCr: {(scr_c/scr_t if scr_t else 0):.4f} ({scr_c}/{scr_t})  |  "
            f"Residues seen: {res}"
        )


if __name__ == "__main__":
    # Smoke test — just instantiate and print its banner.
    cb = ProteinPerAtomMetricsCallback(
        enabled=True,
        only_protein=True,
        compute_groups=True,
        compute_per_aa=True,
        compute_per_aa_block=True,
    )
    print("\nCallback instantiated successfully.")

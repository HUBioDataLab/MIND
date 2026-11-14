#!/usr/bin/env python3
import torch
import numpy as np
from itertools import chain
from pytorch_lightning.callbacks import Callback
from typing import Optional, Dict, List

from core.metrics.per_atom_metrics import (
    PerAtomMetricsCalculator,
    require_wandb_logger_or_announce,
)
from core.metrics.rna_atom_types import (
    ALL_RNA_ATOMS,
    ATOMIC_NUM_TO_ELEMENT,
    RNA_UNKNOWN_ATOM,
    RNA_VIRTUAL_ATOM,
    is_backbone_atom,
    pos_code_to_atom_class,
    pos_code_to_element,
)
from data_loading.atom_position_vocab import is_virtual_position_code


RNA_DATASET_TYPES = {'rna'}
STANDARD_RNA_NUCLEOTIDES = ['A', 'U', 'G', 'C']
WANDB_NAMESPACE_ATOM = 'rna_atom_metrics'
WANDB_NAMESPACE_NUCLEOTIDE = 'rna_nucleotide_metrics'
WANDB_NAMESPACE_NUCLEOTIDE_BLOCK = 'rna_nucleotide_block'

NUCLEOTIDE_ALIASES = {
    'A': 'A',
    'RA': 'A',
    'ADE': 'A',
    'DA': 'A',
    'G': 'G',
    'RG': 'G',
    'GUA': 'G',
    'DG': 'G',
    'C': 'C',
    'RC': 'C',
    'CYT': 'C',
    'DC': 'C',
    'U': 'U',
    'RU': 'U',
    'URA': 'U',
    'DT': 'U',
}


def _is_virtual_atom(pos_code: str) -> bool:
    return is_virtual_position_code(pos_code)


def _is_backbone_atom(pos_code: str) -> bool:
    return is_backbone_atom(pos_code)


def _is_base_slot(pos_code: str) -> bool:
    return not _is_backbone_atom(pos_code)


def _is_real_base_atom(pos_code: str) -> bool:
    # Unknown modified-base labels are still real atoms for block-level element
    # accuracy even when they are grouped under the dedicated UNK metric row.
    return _is_base_slot(pos_code) and not _is_virtual_atom(pos_code)


def _normalize_nucleotide_symbol(symbol: str) -> str:
    key = str(symbol).upper().strip()
    return NUCLEOTIDE_ALIASES.get(key, 'MODIFIED')


class PerAtomMetricsCallback(Callback):
    """
    Callback to compute and log per-atom classification metrics during validation/testing.
    
    Computes metrics for each atom type in the RNA dataset using MLM predictions.
    Logs to WandB with hierarchical naming for easy visualization.
    """
    
    def __init__(self,
                 enabled: bool = True,
                 only_rna: bool = True,
                 atom_types: Optional[list] = None,
                 compute_groups: bool = True,
                 compute_per_nucleotide: bool = True,
                 compute_per_nucleotide_block: bool = True,
                 log_every_n_epochs: int = 1):
        """
        Initialize callback.
        
        Args:
            enabled: Whether callback is active
            only_rna: Only compute metrics for RNA datasets
            atom_types: List of atom types to track (default: ALL_RNA_ATOMS)
            compute_groups: Compute grouped metrics (Backbone/SideChain)
            log_every_n_epochs: How often to log metrics
        """
        super().__init__()
        self.enabled = enabled
        self.only_rna = only_rna
        self.atom_types = atom_types or ALL_RNA_ATOMS
        self.compute_groups = compute_groups
        self.compute_per_nucleotide = compute_per_nucleotide
        self.compute_per_nucleotide_block = compute_per_nucleotide_block
        self.log_every_n_epochs = log_every_n_epochs
        
        # Metrics calculators for different stages
        self.val_metrics = PerAtomMetricsCalculator(
            atom_types=self.atom_types,
            compute_groups=compute_groups
        )
        self.test_metrics = PerAtomMetricsCalculator(
            atom_types=self.atom_types,
            compute_groups=compute_groups
        )

        self._val_nt_stats = self._fresh_nt_stats()
        self._test_nt_stats = self._fresh_nt_stats()
        self._val_nt_block_stats = self._fresh_nt_block_stats()
        self._test_nt_block_stats = self._fresh_nt_block_stats()
        self._last_val_metrics_snapshot: Optional[Dict] = None
        self._no_wandb_announced = False
        
        if enabled:
            print(f"✅ PerAtomMetricsCallback enabled")
            print(f"   - Atom types: {len(self.atom_types)}")
            print(f"   - Group metrics: {compute_groups}")
            print(f"   - Per-nucleotide binary: {compute_per_nucleotide}")
            print(f"   - Per-nucleotide block (backbone/base): {compute_per_nucleotide_block}")
            print(f"   - Log frequency: every {log_every_n_epochs} epochs")
            print(f"   - RNA-only mode: {only_rna}")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Process validation batch."""
        if not self.enabled:
            return
        
        if self.only_rna and not self._is_rna_batch(batch):
            return
        
        self._process_batch_metrics(
            batch,
            pl_module,
            self.val_metrics,
            self._val_nt_stats,
            self._val_nt_block_stats,
            outputs=outputs,
        )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Process test batch."""
        if not self.enabled:
            return
        
        if self.only_rna and not self._is_rna_batch(batch):
            return
        
        self._process_batch_metrics(
            batch,
            pl_module,
            self.test_metrics,
            self._test_nt_stats,
            self._test_nt_block_stats,
            outputs=outputs,
        )
    
    def _process_batch_metrics(self, batch, pl_module, metrics_calculator,
                               nt_stats, nt_block_stats, outputs=None):
        if "mlm" not in pl_module.config.pretraining_tasks:
            return

        if not hasattr(batch, 'pos_code') or batch.pos_code is None:
            raise ValueError("batch.pos_code is missing")

        # 1. Reuse predictions from validation_step/test_step instead of
        # running the full ESA backbone again for this callback.
        preds_atomic_num = self._get_atomic_predictions(
            batch=batch,
            pl_module=pl_module,
            outputs=outputs,
        )

        # 2. Flatten the pos_code list: with batch_size > 1 PyG collates it as a
        # list of per-graph lists ([['P', 'C1'], ['P', 'C2']]); one flat list is needed.
        raw_pos_codes = batch.pos_code
        if isinstance(raw_pos_codes, list) and len(raw_pos_codes) > 0 and isinstance(raw_pos_codes[0], (list, tuple)):
            flattened_pos_codes = list(chain.from_iterable(raw_pos_codes))
        else:
            flattened_pos_codes = raw_pos_codes

        if len(flattened_pos_codes) != len(preds_atomic_num):
            raise ValueError(
                f"pos_code/prediction length mismatch "
                f"({len(flattened_pos_codes)} != {len(preds_atomic_num)})"
            )

        # 3. Alignment. The MLM head predicts an element, so an atom counts as
        # correct when the predicted element matches the element of its
        # pos_code. y_pred keeps the true class for correct atoms and an
        # out-of-vocabulary label otherwise (Recall = element accuracy); the
        # predicted and true elements go to the calculator so Precision is
        # computed over the predicted element instead of being ~1 by
        # construction.
        try:
            y_true_indices = np.array([
                ALL_RNA_ATOMS.index(pos_code_to_atom_class(c))
                for c in flattened_pos_codes
            ])
            pred_elements = np.array([
                ATOMIC_NUM_TO_ELEMENT.get(int(n), RNA_UNKNOWN_ATOM)
                for n in preds_atomic_num
            ])
            true_elements = np.array([pos_code_to_element(c) for c in flattened_pos_codes])
            is_correct_element = (pred_elements == true_elements)
            OUT_OF_VOCAB = len(ALL_RNA_ATOMS)
            y_pred_indices = np.where(is_correct_element, y_true_indices, OUT_OF_VOCAB)

            if not hasattr(batch, 'mlm_mask') or batch.mlm_mask is None:
                raise ValueError("batch.mlm_mask is missing")
            mlm_mask_np: np.ndarray = batch.mlm_mask.cpu().numpy().astype(bool)
            if len(mlm_mask_np) != len(y_true_indices):
                raise ValueError(
                    f"mlm_mask/target length mismatch "
                    f"({len(mlm_mask_np)} != {len(y_true_indices)})"
                )

            # Mixed batches can contain RNA + protein + UniMol. The batch-level
            # RNA gate only says "there is RNA somewhere"; this per-atom mask
            # keeps non-RNA graph atoms out of RNA metrics.
            is_rna_per_atom = self._rna_atom_mask(batch, len(y_true_indices))

            effective_mask = mlm_mask_np & is_rna_per_atom
            y_true = y_true_indices[effective_mask]
            y_pred = y_pred_indices[effective_mask]

            # 4. Update the calculator
            if len(y_true) > 0:
                metrics_calculator.update(
                    y_pred,
                    y_true,
                    predicted_elements=pred_elements[effective_mask],
                    target_elements=true_elements[effective_mask],
                )

            if self.compute_per_nucleotide:
                self._accumulate_nt_stats(
                    batch=batch,
                    flattened_pos_codes=flattened_pos_codes,
                    mlm_mask_np=mlm_mask_np,
                    is_correct_element=is_correct_element,
                    nt_stats=nt_stats,
                )

            if self.compute_per_nucleotide_block:
                self._accumulate_nt_block_stats(
                    batch=batch,
                    flattened_pos_codes=flattened_pos_codes,
                    mlm_mask_np=mlm_mask_np,
                    is_correct_element=is_correct_element,
                    nt_block_stats=nt_block_stats,
                )

        except Exception as exc:
            raise RuntimeError(f"RNA metric alignment failed: {exc}") from exc

    @classmethod
    def _rna_atom_mask(cls, batch, n_atoms: int) -> np.ndarray:
        graph_dtypes = cls._graph_dtype_list(batch)
        mol_idx = cls._mol_idx_per_atom(batch, n_atoms)
        is_rna_graph = np.array(
            [str(dt).lower() in RNA_DATASET_TYPES for dt in graph_dtypes],
            dtype=bool,
        )
        return is_rna_graph[mol_idx]

    @staticmethod
    def _mol_idx_per_atom(batch, n_atoms: int) -> np.ndarray:
        if getattr(batch, 'batch', None) is None:
            raise ValueError("batch.batch (graph index per atom) is missing")
        mol_idx = batch.batch.cpu().numpy()
        if len(mol_idx) != n_atoms:
            raise ValueError(
                f"batch.batch/atom length mismatch ({len(mol_idx)} != {n_atoms})"
            )
        num_graphs = len(PerAtomMetricsCallback._graph_dtype_list(batch))
        if len(mol_idx) and int(mol_idx.max()) >= num_graphs:
            raise ValueError(
                f"graph index {int(mol_idx.max())} out of range for "
                f"{num_graphs} dataset_type entries"
            )
        return mol_idx

    @staticmethod
    def _block_inputs(batch, flattened_pos_codes, mlm_mask_np, is_correct_element):
        """Validate and return (mol_idx, block_idx, per-molecule block symbols)."""
        missing = [
            name for name in ('block_idx', 'block_symbols')
            if getattr(batch, name, None) is None
        ]
        if missing:
            raise ValueError(
                f"RNA per-nucleotide metrics require batch attributes: {', '.join(missing)}"
            )
        n_atoms = len(flattened_pos_codes)
        mol_idx_per_atom = PerAtomMetricsCallback._mol_idx_per_atom(batch, n_atoms)
        block_idx_per_atom = batch.block_idx.cpu().numpy()
        if not (len(block_idx_per_atom) == len(mlm_mask_np)
                == len(is_correct_element) == n_atoms):
            raise ValueError(
                "block_idx/mlm_mask/prediction/pos_code length mismatch "
                f"({len(block_idx_per_atom)}, {len(mlm_mask_np)}, "
                f"{len(is_correct_element)}, {n_atoms})"
            )
        if n_atoms and int(block_idx_per_atom.max()) >= 100_000:
            raise ValueError("block_idx >= 100000 breaks the (graph, block) key")
        per_mol_symbols = PerAtomMetricsCallback._per_mol_block_symbols(batch.block_symbols)
        return mol_idx_per_atom, block_idx_per_atom, per_mol_symbols

    @staticmethod
    def _block_symbol(per_mol_symbols, mol_i: int, block_i: int) -> str:
        if mol_i >= len(per_mol_symbols):
            raise ValueError(
                f"graph {mol_i} has no block_symbols entry ({len(per_mol_symbols)} graphs)"
            )
        mol_symbols = per_mol_symbols[mol_i]
        if block_i >= len(mol_symbols):
            raise ValueError(
                f"graph {mol_i} block_idx {block_i} exceeds its "
                f"{len(mol_symbols)} block_symbols"
            )
        return mol_symbols[block_i]

    @staticmethod
    def _get_atomic_predictions(batch, pl_module, outputs=None) -> np.ndarray:
        if isinstance(outputs, dict):
            cached = outputs.get('mlm_preds_atomic_num')
            if cached is not None:
                if isinstance(cached, torch.Tensor):
                    return cached.detach().cpu().numpy()
                return np.asarray(cached)

        use_bf16 = getattr(pl_module.config, 'use_bfloat16', False)
        device_type = batch.z.device.type if hasattr(batch, 'z') else next(
            pl_module.parameters()
        ).device.type
        with torch.no_grad(), torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            _, node_emb = pl_module.forward(batch)
            mlm_logits = pl_module.pretraining_tasks.mlm_head(node_emb)
            return mlm_logits.argmax(dim=1).detach().cpu().numpy()

    def _accumulate_nt_stats(self, batch, flattened_pos_codes, mlm_mask_np,
                             is_correct_element, nt_stats):
        graph_dtypes = self._graph_dtype_list(batch)
        mol_idx_per_atom, block_idx_per_atom, per_mol_symbols = self._block_inputs(
            batch, flattened_pos_codes, mlm_mask_np, is_correct_element
        )
        is_real_per_atom = np.array(
            [not _is_virtual_atom(c) for c in flattened_pos_codes],
            dtype=bool,
        )

        block_key = (
            mol_idx_per_atom.astype(np.int64) * 100_000
            + block_idx_per_atom.astype(np.int64)
        )
        unique_masked_blocks = np.unique(block_key[mlm_mask_np])

        for bkey in unique_masked_blocks:
            mol_i = int(bkey // 100_000)
            block_i = int(bkey % 100_000)

            # Domain filter: other graphs of a mixed batch are not RNA.
            if str(graph_dtypes[mol_i]).lower() not in RNA_DATASET_TYPES:
                continue

            block_mask = block_key == bkey
            masked_atoms = block_mask & mlm_mask_np
            bucket = _normalize_nucleotide_symbol(
                self._block_symbol(per_mol_symbols, mol_i, block_i)
            )
            all_correct = bool(is_correct_element[masked_atoms].all())

            masked_real = masked_atoms & is_real_per_atom
            has_real_masked = bool(masked_real.any())
            real_all_correct = (
                bool(is_correct_element[masked_real].all())
                if has_real_masked else False
            )

            nt_stats[bucket]['total'] += 1
            if all_correct:
                nt_stats[bucket]['correct'] += 1
            if has_real_masked:
                nt_stats[bucket]['real_total'] += 1
                if real_all_correct:
                    nt_stats[bucket]['real_correct'] += 1

    def _accumulate_nt_block_stats(self, batch, flattened_pos_codes,
                                   mlm_mask_np, is_correct_element,
                                   nt_block_stats):
        graph_dtypes = self._graph_dtype_list(batch)
        mol_idx_per_atom, block_idx_per_atom, per_mol_symbols = self._block_inputs(
            batch, flattened_pos_codes, mlm_mask_np, is_correct_element
        )

        n_atoms = len(flattened_pos_codes)
        is_backbone = np.zeros(n_atoms, dtype=bool)
        is_base_slot = np.zeros(n_atoms, dtype=bool)
        is_base_real = np.zeros(n_atoms, dtype=bool)
        for i, code in enumerate(flattened_pos_codes):
            if _is_backbone_atom(code):
                is_backbone[i] = True
            elif _is_base_slot(code):
                is_base_slot[i] = True
                if _is_real_base_atom(code):
                    is_base_real[i] = True

        block_key = (
            mol_idx_per_atom.astype(np.int64) * 100_000
            + block_idx_per_atom.astype(np.int64)
        )
        unique_masked_blocks = np.unique(block_key[mlm_mask_np])

        for bkey in unique_masked_blocks:
            mol_i = int(bkey // 100_000)
            block_i = int(bkey % 100_000)

            if str(graph_dtypes[mol_i]).lower() not in RNA_DATASET_TYPES:
                continue

            block_mask = block_key == bkey
            masked_in_block = block_mask & mlm_mask_np
            bucket = _normalize_nucleotide_symbol(
                self._block_symbol(per_mol_symbols, mol_i, block_i)
            )

            bb_mask = masked_in_block & is_backbone
            base_mask = masked_in_block & is_base_slot
            base_real_mask = masked_in_block & is_base_real

            stats = nt_block_stats[bucket]
            stats['bb_correct'] += int(is_correct_element[bb_mask].sum())
            stats['bb_total'] += int(bb_mask.sum())
            stats['base_correct'] += int(is_correct_element[base_mask].sum())
            stats['base_total'] += int(base_mask.sum())
            stats['base_real_correct'] += int(is_correct_element[base_real_mask].sum())
            stats['base_real_total'] += int(base_real_mask.sum())
            stats['blocks_seen'] += 1
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation metrics at epoch end."""
        if not self.enabled:
            return

        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        if len(self.val_metrics.all_predictions) == 0:
            self._announce_empty_epoch('validation', pl_module)
            return

        metrics = self.val_metrics.compute()
        nt_accuracies = self._finalize_nt_stats(self._val_nt_stats)
        nt_block_accuracies = self._finalize_nt_block_stats(self._val_nt_block_stats)

        self._log_metrics(
            trainer,
            metrics,
            nt_accuracies,
            nt_block_accuracies,
            stage='val',
            pl_module=pl_module,
        )

        # Keep the last validation snapshot; on_train_end prints the final table from it.
        self._last_val_metrics_snapshot = {
            'per_atom': dict(self.val_metrics.per_atom_metrics),
            'per_group': dict(self.val_metrics.per_group_metrics),
            'overall': dict(self.val_metrics.overall_metrics),
            'nt': {k: dict(v) for k, v in nt_accuracies.items()},
            'nt_block': {k: dict(v) for k, v in nt_block_accuracies.items()},
            'epoch': trainer.current_epoch,
        }

        # Print summary (periyodik; eskiden sadece epoch 0'da tetikleniyordu)
        if trainer.current_epoch % (self.log_every_n_epochs * 5) == 0:
            print("\n" + "="*80)
            print(f"RNA VALIDATION METRICS - Epoch {trainer.current_epoch}")
            print("="*80)
            self.val_metrics.print_summary()
            if self.compute_per_nucleotide:
                self._print_per_nt_table(nt_accuracies)
            if self.compute_per_nucleotide_block:
                self._print_per_nt_block_table(nt_block_accuracies)

        # Reset for next epoch
        self.val_metrics.reset()
        self._val_nt_stats = self._fresh_nt_stats()
        self._val_nt_block_stats = self._fresh_nt_block_stats()

    def on_train_end(self, trainer, pl_module):
        """Print the final metric table at the very end of the log after training."""
        if not self.enabled:
            return
        snap = getattr(self, '_last_val_metrics_snapshot', None)
        if snap is None:
            print("ℹ️  RNA per-atom metrics: no validation epoch with RNA MLM atoms; no final table")
            return
        print("\n" + "="*100)
        print(f"🏁 FINAL RNA METRICS (last validation, epoch {snap['epoch']})")
        print("="*100)
        # Restore the snapshot into the calculator and reuse its print_summary API.
        self.val_metrics.per_atom_metrics = snap['per_atom']
        self.val_metrics.per_group_metrics = snap['per_group']
        self.val_metrics.overall_metrics = snap['overall']
        self.val_metrics.print_summary()
        if self.compute_per_nucleotide:
            self._print_per_nt_table(snap.get('nt', {}))
        if self.compute_per_nucleotide_block:
            self._print_per_nt_block_table(snap.get('nt_block', {}))
    
    def on_test_epoch_end(self, trainer, pl_module):
        """Log test metrics at epoch end."""
        if not self.enabled:
            return
        
        if len(self.test_metrics.all_predictions) == 0:
            self._announce_empty_epoch('test', pl_module)
            return
        
        metrics = self.test_metrics.compute()
        nt_accuracies = self._finalize_nt_stats(self._test_nt_stats)
        nt_block_accuracies = self._finalize_nt_block_stats(self._test_nt_block_stats)
        
        self._log_metrics(
            trainer,
            metrics,
            nt_accuracies,
            nt_block_accuracies,
            stage='test',
            pl_module=pl_module,
        )
        
        # Print summary
        print("\n" + "="*80)
        print(f"RNA TEST METRICS - Epoch {trainer.current_epoch}")
        print("="*80)
        self.test_metrics.print_summary()
        if self.compute_per_nucleotide:
            self._print_per_nt_table(nt_accuracies)
        if self.compute_per_nucleotide_block:
            self._print_per_nt_block_table(nt_block_accuracies)
        
        # Reset
        self.test_metrics.reset()
        self._test_nt_stats = self._fresh_nt_stats()
        self._test_nt_block_stats = self._fresh_nt_block_stats()
    
    @staticmethod
    def _announce_empty_epoch(stage: str, pl_module) -> None:
        # Legitimate when the run has no RNA domain or MLM is not trained.
        reason = (
            "MLM is not a pretraining task"
            if "mlm" not in pl_module.config.pretraining_tasks
            else "no RNA graph with masked atoms was seen"
        )
        print(f"ℹ️  RNA per-atom metrics: nothing to log for this {stage} epoch ({reason})")

    def _log_metrics(self, trainer, metrics: Dict[str, Dict],
                     nt_accuracies, nt_block_accuracies,
                     stage: str = 'val', pl_module=None):
        """
        Log metrics to WandB.
        
        Args:
            trainer: PyTorch Lightning trainer
            metrics: Dict with 'per_atom', 'per_group', 'overall' keys
            stage: 'val' or 'test'
        """
        wandb_logger = require_wandb_logger_or_announce(
            self, trainer, pl_module, 'RNA per-atom metrics'
        )
        if wandb_logger is None:
            return
        
        prefix = f"{stage}_"
        calc = self.val_metrics if stage == 'val' else self.test_metrics
        flat_metrics = calc.get_wandb_logs(
            prefix=prefix,
            namespace=WANDB_NAMESPACE_ATOM,
        )

        if self.compute_per_nucleotide and nt_accuracies:
            total_correct = sum(v['correct'] for v in nt_accuracies.values())
            total_count = sum(v['total'] for v in nt_accuracies.values())
            real_correct = sum(v['real_correct'] for v in nt_accuracies.values())
            real_count = sum(v['real_total'] for v in nt_accuracies.values())
            flat_metrics[f"{prefix}{WANDB_NAMESPACE_NUCLEOTIDE}/overall_accuracy"] = (
                total_correct / total_count if total_count > 0 else 0.0
            )
            flat_metrics[f"{prefix}{WANDB_NAMESPACE_NUCLEOTIDE}/overall_real_accuracy"] = (
                real_correct / real_count if real_count > 0 else 0.0
            )
            flat_metrics[f"{prefix}{WANDB_NAMESPACE_NUCLEOTIDE}/total_nucleotides"] = total_count
            flat_metrics[f"{prefix}{WANDB_NAMESPACE_NUCLEOTIDE}/total_real_nucleotides"] = real_count

            for nt, vals in nt_accuracies.items():
                base = f"{prefix}{WANDB_NAMESPACE_NUCLEOTIDE}/NT_{nt}"
                flat_metrics[f"{base}/accuracy"] = vals['accuracy']
                flat_metrics[f"{base}/real_accuracy"] = vals['real_accuracy']
                flat_metrics[f"{base}/support"] = vals['total']
                flat_metrics[f"{base}/real_support"] = vals['real_total']

        if self.compute_per_nucleotide_block and nt_block_accuracies:
            bb_c = sum(v['bb_correct'] for v in nt_block_accuracies.values())
            bb_t = sum(v['bb_total'] for v in nt_block_accuracies.values())
            base_c = sum(v['base_correct'] for v in nt_block_accuracies.values())
            base_t = sum(v['base_total'] for v in nt_block_accuracies.values())
            base_real_c = sum(v['base_real_correct'] for v in nt_block_accuracies.values())
            base_real_t = sum(v['base_real_total'] for v in nt_block_accuracies.values())
            blocks = sum(v['blocks_seen'] for v in nt_block_accuracies.values())
            ns = WANDB_NAMESPACE_NUCLEOTIDE_BLOCK

            flat_metrics[f"{prefix}{ns}/overall_backbone_accuracy"] = (
                bb_c / bb_t if bb_t > 0 else 0.0
            )
            flat_metrics[f"{prefix}{ns}/overall_base_accuracy"] = (
                base_c / base_t if base_t > 0 else 0.0
            )
            flat_metrics[f"{prefix}{ns}/overall_base_real_accuracy"] = (
                base_real_c / base_real_t if base_real_t > 0 else 0.0
            )
            flat_metrics[f"{prefix}{ns}/total_nucleotides"] = blocks
            flat_metrics[f"{prefix}{ns}/total_backbone_atoms"] = bb_t
            flat_metrics[f"{prefix}{ns}/total_base_atoms"] = base_t
            flat_metrics[f"{prefix}{ns}/total_base_real_atoms"] = base_real_t

            for nt, vals in nt_block_accuracies.items():
                base = f"{prefix}{ns}/NT_{nt}"
                flat_metrics[f"{base}/backbone_accuracy"] = vals['backbone_accuracy']
                flat_metrics[f"{base}/base_accuracy"] = vals['base_accuracy']
                flat_metrics[f"{base}/base_real_accuracy"] = vals['base_real_accuracy']
                flat_metrics[f"{base}/bb_total"] = vals['bb_total']
                flat_metrics[f"{base}/base_total"] = vals['base_total']
                flat_metrics[f"{base}/base_real_total"] = vals['base_real_total']
                flat_metrics[f"{base}/nucleotides_seen"] = vals['blocks_seen']
        
        # Log to wandb
        for metric_name, value in flat_metrics.items():
            wandb_logger.log_metrics({metric_name: value}, step=trainer.global_step)

    @staticmethod
    def _fresh_nt_stats() -> Dict[str, Dict[str, int]]:
        keys = ['correct', 'total', 'real_correct', 'real_total']
        stats = {nt: {k: 0 for k in keys} for nt in STANDARD_RNA_NUCLEOTIDES}
        stats['MODIFIED'] = {k: 0 for k in keys}
        return stats

    @staticmethod
    def _fresh_nt_block_stats() -> Dict[str, Dict[str, int]]:
        keys = [
            'bb_correct',
            'bb_total',
            'base_correct',
            'base_total',
            'base_real_correct',
            'base_real_total',
            'blocks_seen',
        ]
        stats = {nt: {k: 0 for k in keys} for nt in STANDARD_RNA_NUCLEOTIDES}
        stats['MODIFIED'] = {k: 0 for k in keys}
        return stats

    @staticmethod
    def _finalize_nt_stats(nt_stats) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for nt, counts in nt_stats.items():
            total = counts['total']
            if total == 0:
                continue
            real_total = counts['real_total']
            result[nt] = {
                'accuracy': counts['correct'] / total,
                'correct': counts['correct'],
                'total': total,
                'real_accuracy': (
                    counts['real_correct'] / real_total
                    if real_total > 0 else 0.0
                ),
                'real_correct': counts['real_correct'],
                'real_total': real_total,
            }
        return result

    @staticmethod
    def _finalize_nt_block_stats(nt_block_stats) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for nt, counts in nt_block_stats.items():
            if counts['blocks_seen'] == 0:
                continue
            bb_total = counts['bb_total']
            base_total = counts['base_total']
            base_real_total = counts['base_real_total']
            result[nt] = {
                'backbone_accuracy': (
                    counts['bb_correct'] / bb_total if bb_total > 0 else 0.0
                ),
                'base_accuracy': (
                    counts['base_correct'] / base_total
                    if base_total > 0 else 0.0
                ),
                'base_real_accuracy': (
                    counts['base_real_correct'] / base_real_total
                    if base_real_total > 0 else 0.0
                ),
                'bb_correct': counts['bb_correct'],
                'bb_total': bb_total,
                'base_correct': counts['base_correct'],
                'base_total': base_total,
                'base_real_correct': counts['base_real_correct'],
                'base_real_total': base_real_total,
                'blocks_seen': counts['blocks_seen'],
            }
        return result

    @staticmethod
    def _is_rna_batch(batch) -> bool:
        if getattr(batch, 'dataset_type', None) is None:
            raise ValueError("RNA metrics require batch.dataset_type to select RNA graphs")
        dtypes = batch.dataset_type
        if isinstance(dtypes, (list, tuple)):
            return any(str(dt).lower() in RNA_DATASET_TYPES for dt in dtypes)
        return str(dtypes).lower() in RNA_DATASET_TYPES

    @staticmethod
    def _graph_dtype_list(batch):
        if getattr(batch, 'dataset_type', None) is None:
            raise ValueError("RNA metrics require batch.dataset_type to select RNA graphs")
        dt = batch.dataset_type
        if isinstance(dt, (list, tuple)):
            return list(dt)
        n = getattr(batch, 'num_graphs', 1) or 1
        return [dt] * n

    @staticmethod
    def _per_mol_block_symbols(raw_block_symbols) -> List[List[str]]:
        if (isinstance(raw_block_symbols, list)
                and len(raw_block_symbols) > 0
                and isinstance(raw_block_symbols[0], list)):
            return raw_block_symbols
        return [list(raw_block_symbols)]

    @staticmethod
    def _print_per_nt_table(nt_accuracies: Dict[str, Dict[str, float]]) -> None:
        if not nt_accuracies:
            print("\n(no nucleotides with masked atoms observed — per-NT table empty)\n")
            return
        try:
            from tabulate import tabulate
            rows = [
                [
                    nt,
                    f"{vals['accuracy']:.4f}",
                    f"{int(vals['correct'])}/{int(vals['total'])}",
                    f"{vals['real_accuracy']:.4f}",
                    f"{int(vals['real_correct'])}/{int(vals['real_total'])}",
                ]
                for nt, vals in sorted(nt_accuracies.items())
            ]
            table = tabulate(
                rows,
                headers=['NT', 'Accuracy', 'c/t', 'Real Acc', 'Real c/t'],
                tablefmt='grid',
            )
        except ImportError:
            lines = ['NT | Accuracy | c/t | Real Acc | Real c/t']
            for nt, vals in sorted(nt_accuracies.items()):
                lines.append(
                    f"{nt:8s} | {vals['accuracy']:.4f} | "
                    f"{int(vals['correct'])}/{int(vals['total'])} | "
                    f"{vals['real_accuracy']:.4f} | "
                    f"{int(vals['real_correct'])}/{int(vals['real_total'])}"
                )
            table = "\n".join(lines)

        total_correct = sum(v['correct'] for v in nt_accuracies.values())
        total_count = sum(v['total'] for v in nt_accuracies.values())
        real_correct = sum(v['real_correct'] for v in nt_accuracies.values())
        real_count = sum(v['real_total'] for v in nt_accuracies.values())
        overall = total_correct / total_count if total_count > 0 else 0.0
        real_overall = real_correct / real_count if real_count > 0 else 0.0

        print("\n📊 PER-NUCLEOTIDE METRICS — block-level all-or-nothing (1 iff EVERY masked atom in the nucleotide got the right element):")
        print("    Accuracy = all masked atoms incl. virtual base padding. Real Acc excludes virtual atoms and tracks real nucleotide recognition.")
        print(table)
        print(
            f"\n   Overall: {overall:.4f}  ({total_correct} / {total_count})  |  "
            f"Real-only: {real_overall:.4f}  ({real_correct} / {real_count})"
        )

    @staticmethod
    def _print_per_nt_block_table(nt_block_accuracies: Dict[str, Dict[str, float]]) -> None:
        if not nt_block_accuracies:
            print("\n(no nucleotides with masked atoms observed — per-NT-block table empty)\n")
            return
        try:
            from tabulate import tabulate
            rows = []
            for nt, v in sorted(nt_block_accuracies.items()):
                rows.append([
                    nt,
                    f"{v['backbone_accuracy']:.4f}",
                    f"{int(v['bb_correct'])}/{int(v['bb_total'])}",
                    f"{v['base_accuracy']:.4f}",
                    f"{int(v['base_correct'])}/{int(v['base_total'])}",
                    f"{v['base_real_accuracy']:.4f}",
                    f"{int(v['base_real_correct'])}/{int(v['base_real_total'])}",
                    int(v['blocks_seen']),
                ])
            table = tabulate(
                rows,
                headers=['NT', 'BB acc', 'BB c/t', 'Base acc', 'Base c/t',
                         'BaseR acc', 'BaseR c/t', 'NTs'],
                tablefmt='grid',
            )
        except ImportError:
            lines = ['NT | BB acc | BB c/t | Base acc | Base c/t | BaseR acc | BaseR c/t | NTs']
            for nt, v in sorted(nt_block_accuracies.items()):
                lines.append(
                    f"{nt:8s} | {v['backbone_accuracy']:.4f} | "
                    f"{int(v['bb_correct'])}/{int(v['bb_total'])} | "
                    f"{v['base_accuracy']:.4f} | "
                    f"{int(v['base_correct'])}/{int(v['base_total'])} | "
                    f"{v['base_real_accuracy']:.4f} | "
                    f"{int(v['base_real_correct'])}/{int(v['base_real_total'])} | "
                    f"{int(v['blocks_seen'])}"
                )
            table = "\n".join(lines)

        bb_c = sum(v['bb_correct'] for v in nt_block_accuracies.values())
        bb_t = sum(v['bb_total'] for v in nt_block_accuracies.values())
        base_c = sum(v['base_correct'] for v in nt_block_accuracies.values())
        base_t = sum(v['base_total'] for v in nt_block_accuracies.values())
        base_real_c = sum(v['base_real_correct'] for v in nt_block_accuracies.values())
        base_real_t = sum(v['base_real_total'] for v in nt_block_accuracies.values())
        blocks = sum(v['blocks_seen'] for v in nt_block_accuracies.values())

        print("\n📊 PER-NUCLEOTIDE BLOCK METRICS — atom-level fraction correct (backbone / base-slot / base-real):")
        print("    BB = phosphate+sugar backbone. Base = 11 base slots incl. virtual padding. BaseR = real base atoms only.")
        print(table)
        print(
            f"\n   Overall BB: {(bb_c/bb_t if bb_t else 0):.4f} ({bb_c}/{bb_t})  |  "
            f"Base: {(base_c/base_t if base_t else 0):.4f} ({base_c}/{base_t})  |  "
            f"BaseR: {(base_real_c/base_real_t if base_real_t else 0):.4f} ({base_real_c}/{base_real_t})  |  "
            f"Nucleotides seen: {blocks}"
        )
    
    def print_metrics_summary(self, stage: str = 'val'):
        """Print metrics summary."""
        if stage == 'val':
            self.val_metrics.print_summary()
        else:
            self.test_metrics.print_summary()


if __name__ == "__main__":
    # Test the callback initialization
    callback = PerAtomMetricsCallback(
        enabled=True,
        only_rna=True,
        compute_groups=True
    )
    print("✅ PerAtomMetricsCallback initialized successfully")

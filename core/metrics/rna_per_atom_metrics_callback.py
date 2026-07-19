#!/usr/bin/env python3
import torch
import numpy as np
from itertools import chain
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from typing import Optional, Dict, List

from core.metrics.per_atom_metrics import PerAtomMetricsCalculator
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
        
        # Extract atom predictions and targets from MLM task
        try:
            self._process_batch_metrics(
                batch,
                pl_module,
                self.val_metrics,
                self._val_nt_stats,
                self._val_nt_block_stats,
            )
        except Exception as e:
            # Silently skip if we can't compute metrics
            pass

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Process test batch."""
        if not self.enabled:
            return
        
        if self.only_rna and not self._is_rna_batch(batch):
            return
        
        try:
            self._process_batch_metrics(
                batch,
                pl_module,
                self.test_metrics,
                self._test_nt_stats,
                self._test_nt_block_stats,
            )
        except Exception as e:
            pass
    
    def _process_batch_metrics(self, batch, pl_module, metrics_calculator,
                               nt_stats, nt_block_stats):
        if "mlm" not in pl_module.config.pretraining_tasks:
            return

        if not hasattr(batch, 'pos_code') or batch.pos_code is None:
            return

        # 1. Model Tahminlerini Al (Tensör olduğu için zaten düzdür)
        # Lightning'in autocast context'i callback dışında kaldığı için burada manuel kuruyoruz.
        # bf16 training'de dtype mismatch'i önler; fp32 run'larda enabled=False → no-op (davranış değişmez).
        use_bf16 = getattr(pl_module.config, 'use_bfloat16', False)
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16):
            _, node_emb = pl_module.forward(batch)
            mlm_logits = pl_module.pretraining_tasks.mlm_head(node_emb)
            preds_atomic_num = mlm_logits.argmax(dim=1).cpu().numpy()

        # 2. KRİTİK DÜZELTME: pos_code listesini düzleştir
        raw_pos_codes = batch.pos_code
        # Eğer batch_size > 1 ise PyG bunu [['P', 'C1'], ['P', 'C2']] gibi getirir.
        # Bunu tek bir düz liste haline getiriyoruz:
        if isinstance(raw_pos_codes, list) and len(raw_pos_codes) > 0 and isinstance(raw_pos_codes[0], (list, tuple)):
            flattened_pos_codes = list(chain.from_iterable(raw_pos_codes))
        else:
            flattened_pos_codes = raw_pos_codes

        if len(flattened_pos_codes) != len(preds_atomic_num):
            return

        # 3. Hizalama (Alignment)
        try:
            # Gerçek değerleri RNA indekslerine çevir
            y_true_indices = np.array([
                ALL_RNA_ATOMS.index(pos_code_to_atom_class(c))
                for c in flattened_pos_codes
            ])

            # FIX: "first-matching-element" lock-in bug'ı yerine element-aware eşleşme.
            # Her pozisyon için: modelin tahmin ettiği element (C/N/O/P/...) pos_code'un
            # beklenen elementi ile uyuşuyorsa y_pred = y_true (yani "o pozisyondaki atomu
            # doğru bildi" sayılır); uyuşmuyorsa vocabulary dışı bir idx verilir.
            # Bu sayede:
            #   Recall (per-atom) = doğru-element-tahmin-edilen / toplam = element accuracy
            #   Precision by construction ≈ 1.0 (artefakt), anlamlı kolon Recall'dır
            #   F1 = 2R/(1+R) Recall ile monotonik — anlamlı
            pred_elements = np.array([
                ATOMIC_NUM_TO_ELEMENT.get(int(n), RNA_UNKNOWN_ATOM)
                for n in preds_atomic_num
            ])
            true_elements = np.array([pos_code_to_element(c) for c in flattened_pos_codes])
            is_correct_element = (pred_elements == true_elements)
            OUT_OF_VOCAB = len(ALL_RNA_ATOMS)
            y_pred_indices = np.where(is_correct_element, y_true_indices, OUT_OF_VOCAB)
            
            # Boyut Kontrolü (Hata almamak için savunma hattı)
            if len(y_true_indices) != len(y_pred_indices):
                # Eğer hala uyuşmazlık varsa (nadir bir PyG durumu) logla ve geç
                if self.trainer.global_step % 100 == 0:
                    print(f"⚠️ Boyut Uyuşmazlığı: True({len(y_true_indices)}) vs Pred({len(y_pred_indices)})")
                return

            mlm_mask_np: Optional[np.ndarray] = None
            if hasattr(batch, 'mlm_mask') and batch.mlm_mask is not None:
                mlm_mask_np = batch.mlm_mask.cpu().numpy().astype(bool)
                if len(mlm_mask_np) != len(y_true_indices):
                    mlm_mask_np = None

            # Mixed batches can contain RNA + protein + UniMol. The batch-level
            # RNA gate only says "there is RNA somewhere"; this per-atom mask
            # keeps non-RNA graph atoms out of RNA metrics.
            is_rna_per_atom = np.ones(len(y_true_indices), dtype=bool)
            graph_dtypes = self._graph_dtype_list(batch)
            if (graph_dtypes is not None
                    and hasattr(batch, 'batch')
                    and batch.batch is not None):
                mol_idx = batch.batch.cpu().numpy()
                if len(mol_idx) == len(y_true_indices):
                    is_rna_per_atom = np.array(
                        [
                            (g < len(graph_dtypes)
                             and str(graph_dtypes[g]).lower() in RNA_DATASET_TYPES)
                            for g in mol_idx
                        ],
                        dtype=bool,
                    )

            if mlm_mask_np is not None:
                effective_mask = mlm_mask_np & is_rna_per_atom
                y_true = y_true_indices[effective_mask]
                y_pred = y_pred_indices[effective_mask]
            else:
                y_true = y_true_indices[is_rna_per_atom]
                y_pred = y_pred_indices[is_rna_per_atom]

            # 4. Hesaplayıcıyı Güncelle
            if len(y_true) > 0:
                metrics_calculator.update(y_pred, y_true)

            if self.compute_per_nucleotide and mlm_mask_np is not None:
                self._accumulate_nt_stats(
                    batch=batch,
                    flattened_pos_codes=flattened_pos_codes,
                    mlm_mask_np=mlm_mask_np,
                    is_correct_element=is_correct_element,
                    nt_stats=nt_stats,
                )

            if self.compute_per_nucleotide_block and mlm_mask_np is not None:
                self._accumulate_nt_block_stats(
                    batch=batch,
                    flattened_pos_codes=flattened_pos_codes,
                    mlm_mask_np=mlm_mask_np,
                    is_correct_element=is_correct_element,
                    nt_block_stats=nt_block_stats,
                )
                
        except Exception as e:
            print(f"⚠️ PerAtomMetrics alignment error: {e}")

    def _accumulate_nt_stats(self, batch, flattened_pos_codes, mlm_mask_np,
                             is_correct_element, nt_stats):
        if not hasattr(batch, 'block_idx') or not hasattr(batch, 'block_symbols'):
            return
        if not hasattr(batch, 'batch') or batch.batch is None:
            return

        graph_dtypes = self._graph_dtype_list(batch)
        mol_idx_per_atom = batch.batch.cpu().numpy()
        block_idx_per_atom = batch.block_idx.cpu().numpy()
        if (
            len(mol_idx_per_atom) != len(block_idx_per_atom)
            or len(mlm_mask_np) != len(mol_idx_per_atom)
            or len(is_correct_element) != len(mol_idx_per_atom)
            or len(flattened_pos_codes) != len(mol_idx_per_atom)
        ):
            return

        per_mol_symbols = self._per_mol_block_symbols(batch.block_symbols)
        n_atoms = len(mol_idx_per_atom)
        if len(flattened_pos_codes) == n_atoms:
            is_real_per_atom = np.array(
                [not _is_virtual_atom(c) for c in flattened_pos_codes],
                dtype=bool,
            )
        else:
            is_real_per_atom = np.ones(n_atoms, dtype=bool)

        block_key = (
            mol_idx_per_atom.astype(np.int64) * 100_000
            + block_idx_per_atom.astype(np.int64)
        )
        unique_masked_blocks = np.unique(block_key[mlm_mask_np])

        for bkey in unique_masked_blocks:
            mol_i = int(bkey // 100_000)
            block_i = int(bkey % 100_000)

            if graph_dtypes is not None and mol_i < len(graph_dtypes):
                if str(graph_dtypes[mol_i]).lower() not in RNA_DATASET_TYPES:
                    continue

            block_mask = block_key == bkey
            masked_atoms = block_mask & mlm_mask_np
            if not masked_atoms.any():
                continue

            if mol_i >= len(per_mol_symbols):
                continue
            mol_symbols = per_mol_symbols[mol_i]
            if block_i >= len(mol_symbols):
                continue

            bucket = _normalize_nucleotide_symbol(mol_symbols[block_i])
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
        if not hasattr(batch, 'block_idx') or not hasattr(batch, 'block_symbols'):
            return
        if not hasattr(batch, 'batch') or batch.batch is None:
            return

        graph_dtypes = self._graph_dtype_list(batch)
        mol_idx_per_atom = batch.batch.cpu().numpy()
        block_idx_per_atom = batch.block_idx.cpu().numpy()
        if (
            len(mol_idx_per_atom) != len(block_idx_per_atom)
            or len(mlm_mask_np) != len(mol_idx_per_atom)
            or len(is_correct_element) != len(mol_idx_per_atom)
            or len(flattened_pos_codes) != len(mol_idx_per_atom)
        ):
            return
        per_mol_symbols = self._per_mol_block_symbols(batch.block_symbols)

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

            if graph_dtypes is not None and mol_i < len(graph_dtypes):
                if str(graph_dtypes[mol_i]).lower() not in RNA_DATASET_TYPES:
                    continue

            block_mask = block_key == bkey
            masked_in_block = block_mask & mlm_mask_np
            if not masked_in_block.any():
                continue

            if mol_i >= len(per_mol_symbols):
                continue
            mol_symbols = per_mol_symbols[mol_i]
            if block_i >= len(mol_symbols):
                continue
            bucket = _normalize_nucleotide_symbol(mol_symbols[block_i])

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
        )

        # Son validation snapshot'ını sakla — on_train_end'de final tablo için kullanılacak.
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
        """Final metric tablosu — eğitim bittikten sonra log'un en sonunda göster."""
        if not self.enabled:
            return
        snap = getattr(self, '_last_val_metrics_snapshot', None)
        if snap is None:
            return
        print("\n" + "="*100)
        print(f"🏁 FINAL RNA METRICS (last validation, epoch {snap['epoch']})")
        print("="*100)
        # Snapshot'ı calculator'a geri yükle ve basit print_summary API'sini kullan
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
    
    def _log_metrics(self, trainer, metrics: Dict[str, Dict],
                     nt_accuracies, nt_block_accuracies,
                     stage: str = 'val'):
        """
        Log metrics to WandB.
        
        Args:
            trainer: PyTorch Lightning trainer
            metrics: Dict with 'per_atom', 'per_group', 'overall' keys
            stage: 'val' or 'test'
        """
        # Get WandB logger
        wandb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
                break
        
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
        if not hasattr(batch, 'dataset_type') or batch.dataset_type is None:
            # Preserve the old single-domain behavior: when no dtype metadata
            # exists, assume the callback was attached to an RNA-only loader.
            return True
        dtypes = batch.dataset_type
        if isinstance(dtypes, (list, tuple)):
            return any(str(dt).lower() in RNA_DATASET_TYPES for dt in dtypes)
        return str(dtypes).lower() in RNA_DATASET_TYPES

    @staticmethod
    def _graph_dtype_list(batch):
        if not hasattr(batch, 'dataset_type') or batch.dataset_type is None:
            return None
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


class SimplePerAtomMetricsCallback(Callback):
    """
    Simplified version that only tracks atom prediction accuracy without detailed metrics.
    
    Useful for quick experiments where detailed per-atom metrics are not needed.
    """
    
    def __init__(self, enabled: bool = False):
        super().__init__()
        self.enabled = enabled
        self.val_atom_preds = []
        self.val_atom_targets = []
        self.test_atom_preds = []
        self.test_atom_targets = []
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.enabled or "mlm" not in pl_module.config.pretraining_tasks:
            return
        
        try:
            if hasattr(batch, 'original_types') and hasattr(batch, 'mlm_mask'):
                with torch.no_grad():
                    _, node_emb = pl_module.forward(batch)
                    mlm_logits = pl_module.pretraining_tasks.mlm_head(node_emb)
                    preds = mlm_logits.argmax(dim=1)
                    
                    mask = batch.mlm_mask
                    self.val_atom_preds.append(preds[mask].cpu())
                    self.val_atom_targets.append(batch.original_types[mask].cpu())
        except:
            pass
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.enabled or len(self.val_atom_preds) == 0:
            return
        
        preds = torch.cat(self.val_atom_preds)
        targets = torch.cat(self.val_atom_targets)
        
        acc = (preds == targets).float().mean()
        trainer.logger.log_metrics({"val_atom_accuracy": acc.item()}, step=trainer.global_step)
        
        self.val_atom_preds.clear()
        self.val_atom_targets.clear()
    
    def on_test_epoch_end(self, trainer, pl_module):
        if not self.enabled or len(self.test_atom_preds) == 0:
            return
        
        preds = torch.cat(self.test_atom_preds)
        targets = torch.cat(self.test_atom_targets)
        
        acc = (preds == targets).float().mean()
        trainer.logger.log_metrics({"test_atom_accuracy": acc.item()}, step=trainer.global_step)
        
        self.test_atom_preds.clear()
        self.test_atom_targets.clear()


if __name__ == "__main__":
    # Test the callback initialization
    callback = PerAtomMetricsCallback(
        enabled=True,
        only_rna=True,
        compute_groups=True
    )
    print("✅ PerAtomMetricsCallback initialized successfully")

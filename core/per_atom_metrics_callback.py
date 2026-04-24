#!/usr/bin/env python3
"""
Per-Atom Metrics Evaluation Callback

PyTorch Lightning callback that computes and logs per-atom classification metrics.
Integrated with WandB for visualization.

Tracks:
- Per-atom metrics (Accuracy, Precision, Recall, F1) for each of 11+12 atom types
- Group metrics (Backbone vs SideChain)
- Overall metrics (macro/weighted averages)
"""

import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from collections import defaultdict
from typing import Optional, Dict, Any

from core.per_atom_metrics import PerAtomMetricsCalculator
from core.rna_atom_types import ALL_RNA_ATOMS, ATOMIC_NUM_TO_ELEMENT


def _pos_code_to_element(pos_code: str) -> str:
    """
    RNA pos_code → element sembolü. İlk karakter elementi verir (P, O*, C*, N*).
    Virtual atomlar ('V', 'V1', ...) → 'V'.
    """
    if not pos_code:
        return 'V'
    if pos_code == 'V' or pos_code.startswith('V'):
        return 'V'
    return pos_code[0]


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
        
        if enabled:
            print(f"✅ PerAtomMetricsCallback enabled")
            print(f"   - Atom types: {len(self.atom_types)}")
            print(f"   - Group metrics: {compute_groups}")
            print(f"   - Log frequency: every {log_every_n_epochs} epochs")
            print(f"   - RNA-only mode: {only_rna}")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Process validation batch."""
        if not self.enabled:
            return
        
        # Skip non-RNA batches if only_rna is True
        if self.only_rna and hasattr(batch, 'dataset_type'):
            dataset_types = batch.dataset_type
            if isinstance(dataset_types, (list, tuple)):
                is_rna = any(dt.lower() == 'rna' for dt in dataset_types)
            else:
                is_rna = str(dataset_types).lower() == 'rna'
            if not is_rna:
                return
        
        # Extract atom predictions and targets from MLM task
        try:
            self._process_batch_metrics(batch, pl_module, self.val_metrics)
        except Exception as e:
            # Silently skip if we can't compute metrics
            pass

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Process test batch."""
        if not self.enabled:
            return
        
        # Skip non-RNA batches if only_rna is True
        if self.only_rna and hasattr(batch, 'dataset_type'):
            dataset_types = batch.dataset_type
            if isinstance(dataset_types, (list, tuple)):
                is_rna = any(dt.lower() == 'rna' for dt in dataset_types)
            else:
                is_rna = str(dataset_types).lower() == 'rna'
            if not is_rna:
                return
        
        try:
            self._process_batch_metrics(batch, pl_module, self.test_metrics)
        except Exception as e:
            pass
    
    def _process_batch_metrics(self, batch, pl_module, metrics_calculator):
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
            from itertools import chain
            flattened_pos_codes = list(chain.from_iterable(raw_pos_codes))
        else:
            flattened_pos_codes = raw_pos_codes

        # 3. Hizalama (Alignment)
        try:
            # Gerçek değerleri RNA indekslerine çevir
            y_true_indices = np.array([ALL_RNA_ATOMS.index(c) if c in ALL_RNA_ATOMS else ALL_RNA_ATOMS.index('V')
                                      for c in flattened_pos_codes])

            # FIX: "first-matching-element" lock-in bug'ı yerine element-aware eşleşme.
            # Her pozisyon için: modelin tahmin ettiği element (C/N/O/P/...) pos_code'un
            # beklenen elementi ile uyuşuyorsa y_pred = y_true (yani "o pozisyondaki atomu
            # doğru bildi" sayılır); uyuşmuyorsa vocabulary dışı bir idx verilir.
            # Bu sayede:
            #   Recall (per-atom) = doğru-element-tahmin-edilen / toplam = element accuracy
            #   Precision by construction ≈ 1.0 (artefakt), anlamlı kolon Recall'dır
            #   F1 = 2R/(1+R) Recall ile monotonik — anlamlı
            pred_elements = np.array([ATOMIC_NUM_TO_ELEMENT.get(int(n), 'V') for n in preds_atomic_num])
            true_elements = np.array([_pos_code_to_element(c) for c in flattened_pos_codes])
            is_correct_element = (pred_elements == true_elements)
            OUT_OF_VOCAB = len(ALL_RNA_ATOMS)  # 24 — ALL_RNA_ATOMS idx space dışında
            y_pred_indices = np.where(is_correct_element, y_true_indices, OUT_OF_VOCAB)
            
            # Boyut Kontrolü (Hata almamak için savunma hattı)
            if len(y_true_indices) != len(y_pred_indices):
                # Eğer hala uyuşmazlık varsa (nadir bir PyG durumu) logla ve geç
                if self.trainer.global_step % 100 == 0:
                    print(f"⚠️ Boyut Uyuşmazlığı: True({len(y_true_indices)}) vs Pred({len(y_pred_indices)})")
                return

            # Maskeleme
            if hasattr(batch, 'mlm_mask') and batch.mlm_mask is not None:
                mask = batch.mlm_mask.cpu().numpy().astype(bool)
                # Maske boyutu ile veriyi doğrula
                if len(mask) == len(y_true_indices):
                    y_true = y_true_indices[mask]
                    y_pred = y_pred_indices[mask]
                else:
                    # Maske bazen sadece belirli atomları kapsayabilir, 
                    # bu durumda en güvenli yol maskelenmiş tahminleri almaktır
                    y_true = y_true_indices[:len(mask)][mask]
                    y_pred = y_pred_indices[:len(mask)][mask]
            else:
                y_true = y_true_indices
                y_pred = y_pred_indices

            # 4. Hesaplayıcıyı Güncelle
            if len(y_true) > 0:
                metrics_calculator.update(y_pred, y_true)
                
        except Exception as e:
            print(f"⚠️ PerAtomMetrics alignment error: {e}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation metrics at epoch end."""
        if not self.enabled:
            return

        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        if len(self.val_metrics.all_predictions) == 0:
            return

        # Compute metrics
        metrics = self.val_metrics.compute()

        # Log to wandb
        self._log_metrics(trainer, metrics, stage='val')

        # Son validation snapshot'ını sakla — on_train_end'de final tablo için kullanılacak.
        self._last_val_metrics_snapshot = {
            'per_atom': dict(self.val_metrics.per_atom_metrics),
            'per_group': dict(self.val_metrics.per_group_metrics),
            'overall': dict(self.val_metrics.overall_metrics),
            'epoch': trainer.current_epoch,
        }

        # Print summary (periyodik; eskiden sadece epoch 0'da tetikleniyordu)
        if trainer.current_epoch % (self.log_every_n_epochs * 5) == 0:
            print("\n" + "="*80)
            print(f"VALIDATION METRICS - Epoch {trainer.current_epoch}")
            print("="*80)
            self.val_metrics.print_summary()

        # Reset for next epoch
        self.val_metrics.reset()

    def on_train_end(self, trainer, pl_module):
        """Final metric tablosu — eğitim bittikten sonra log'un en sonunda göster."""
        if not self.enabled:
            return
        snap = getattr(self, '_last_val_metrics_snapshot', None)
        if snap is None:
            return
        print("\n" + "="*100)
        print(f"🏁 FINAL PER-ATOM METRICS (last validation, epoch {snap['epoch']})")
        print("="*100)
        # Snapshot'ı calculator'a geri yükle ve basit print_summary API'sini kullan
        self.val_metrics.per_atom_metrics = snap['per_atom']
        self.val_metrics.per_group_metrics = snap['per_group']
        self.val_metrics.overall_metrics = snap['overall']
        self.val_metrics.print_summary()
    
    def on_test_epoch_end(self, trainer, pl_module):
        """Log test metrics at epoch end."""
        if not self.enabled:
            return
        
        if len(self.test_metrics.all_predictions) == 0:
            return
        
        # Compute metrics
        metrics = self.test_metrics.compute()
        
        # Log to wandb
        self._log_metrics(trainer, metrics, stage='test')
        
        # Print summary
        print("\n" + "="*80)
        print(f"TEST METRICS - Epoch {trainer.current_epoch}")
        print("="*80)
        self.test_metrics.print_summary()
        
        # Reset
        self.test_metrics.reset()
    
    def _log_metrics(self, trainer, metrics: Dict[str, Dict], stage: str = 'val'):
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
        
        # Get flat metrics dict for logging
        prefix = f"{stage}_"
        # Domain-specific namespace — RNA-only callback → val_rna_atom_metrics/*
        # (protein callback will use 'protein_atom_metrics' with a different callback class)
        namespace = 'rna_atom_metrics'
        flat_metrics = (self.val_metrics.get_wandb_logs(prefix=prefix, namespace=namespace)
                        if stage == 'val'
                        else self.test_metrics.get_wandb_logs(prefix=prefix, namespace=namespace))
        
        # Log to wandb
        for metric_name, value in flat_metrics.items():
            wandb_logger.log_metrics({metric_name: value}, step=trainer.global_step)
    
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

#!/usr/bin/env python3
"""
Per-Atom Metrics Calculator for RNA Classification

Calculates per-atom classification metrics:
- Accuracy, Precision, Recall, F1-score for each atom type
- Grouped metrics for Backbone vs SideChain
- WandB logging support for visualization

Uses scikit-learn for metric computation.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict

from core.metrics.rna_atom_types import (
    ALL_RNA_ATOMS,
    RNA_BACKBONE_ATOMS,
    RNA_SIDECHAIN_ATOMS,
    RNA_VIRTUAL_ATOM,
    RNA_ATOM_GROUPS,
    get_atom_type_vocabulary,
    is_backbone_atom,
    is_sidechain_atom,
)


class PerAtomMetricsCalculator:
    """
    Calculates per-atom classification metrics for RNA atoms.
    
    Supports:
    - Per-position metrics (Acc, P, R, F1) for the configured vocabulary
    - Group metrics for Backbone and SideChain
    - Accumulated metrics across batches
    - WandB-compatible logging
    """
    
    def __init__(self, 
                 atom_types: Optional[List[str]] = None,
                 compute_groups: bool = True,
                 group_type: str = 'backbone_vs_sidechain',
                 zero_division: int = 0):
        """
        Initialize metrics calculator.
        
        Args:
            atom_types: List of atom types to track. Defaults to ALL_RNA_ATOMS
            compute_groups: Whether to compute grouped metrics
            group_type: Type of grouping ('backbone_vs_sidechain', 'backbone_by_function', 'base_by_type')
            zero_division: How to handle zero division in precision/recall (0 or 1)
        """
        self.atom_types = atom_types or ALL_RNA_ATOMS
        self.atom_to_idx = {atom: i for i, atom in enumerate(self.atom_types)}
        
        self.compute_groups = compute_groups
        self.group_type = group_type
        self.groups = RNA_ATOM_GROUPS.get(group_type, {})
        self.zero_division = zero_division
        
        # Storage for predictions and targets
        self.all_predictions: List[np.ndarray] = []
        self.all_targets: List[np.ndarray] = []
        self.all_atoms: List[str] = []
        self.all_predicted_elements: List[np.ndarray] = []
        self.all_target_elements: List[np.ndarray] = []
        
        # Aggregated metrics
        self.per_atom_metrics: Dict[str, Dict[str, float]] = {}
        self.per_group_metrics: Dict[str, Dict[str, float]] = {}
        self.overall_metrics: Dict[str, float] = {}
    
    def reset(self):
        """Reset accumulated metrics."""
        self.all_predictions = []
        self.all_targets = []
        self.all_atoms = []
        self.all_predicted_elements = []
        self.all_target_elements = []
        self.per_atom_metrics = {}
        self.per_group_metrics = {}
        self.overall_metrics = {}
    
    def update(self,
               predictions: Union[torch.Tensor, np.ndarray],
               targets: Union[torch.Tensor, np.ndarray],
               atom_names: Optional[Union[List[str], np.ndarray]] = None,
               predicted_elements: Optional[np.ndarray] = None,
               target_elements: Optional[np.ndarray] = None):
        """
        Update metrics with new batch predictions.
        
        Args:
            predictions: Model predictions (logits or class indices). Shape: (batch_size,) or (batch_size, num_classes)
            targets: Ground truth atom types. Shape: (batch_size,)
            atom_names: Optional list of atom names for filtering/analysis
            predicted_elements / target_elements: element-aware mode. The MLM
                head predicts an element, not a position class, so a wrong
                prediction has no position class. When given, Precision for a
                row is computed over the predicted element: of all atoms
                predicted as the row's element, the fraction whose true element
                is that element. Recall and Accuracy still use the class labels.
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:
                # If logits, take argmax
                predictions = predictions.argmax(dim=1).cpu().numpy()
            else:
                predictions = predictions.cpu().numpy()
        
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        if (predicted_elements is None) != (target_elements is None):
            raise ValueError(
                "predicted_elements and target_elements must be given together"
            )
        element_mode = predicted_elements is not None
        if self.all_predictions and element_mode != bool(self.all_predicted_elements):
            raise ValueError(
                "cannot mix element-aware and label-only updates in one calculator"
            )
        if element_mode:
            predicted_elements = np.asarray(predicted_elements)
            target_elements = np.asarray(target_elements)
            if not (len(predicted_elements) == len(target_elements)
                    == len(predictions) == len(targets)):
                raise ValueError(
                    "predictions, targets and element arrays must have equal length "
                    f"({len(predictions)}, {len(targets)}, "
                    f"{len(predicted_elements)}, {len(target_elements)})"
                )
            self.all_predicted_elements.append(predicted_elements)
            self.all_target_elements.append(target_elements)

        # Store
        self.all_predictions.append(predictions.astype(int))
        self.all_targets.append(targets.astype(int))
        
        if atom_names is not None:
            if isinstance(atom_names, torch.Tensor):
                atom_names = atom_names.cpu().numpy()
            if isinstance(atom_names, np.ndarray):
                atom_names = atom_names.tolist()
            self.all_atoms.extend(atom_names)
    
    def compute(self) -> Dict[str, Dict[str, float]]:
        """
        Compute all metrics.
        
        Returns:
            Dict with keys: 'per_atom', 'per_group', 'overall'
        """
        if len(self.all_predictions) == 0:
            return {'per_atom': {}, 'per_group': {}, 'overall': {}}
        
        # Concatenate all batches
        y_pred = np.concatenate(self.all_predictions)
        y_true = np.concatenate(self.all_targets)
        if self.all_predicted_elements:
            elements = (
                np.concatenate(self.all_predicted_elements),
                np.concatenate(self.all_target_elements),
            )
        else:
            elements = None
        
        # Compute per-atom metrics
        self.per_atom_metrics = self._compute_per_atom_metrics(y_pred, y_true, elements)
        
        # Compute grouped metrics if enabled
        if self.compute_groups:
            self.per_group_metrics = self._compute_group_metrics(y_pred, y_true, elements)
        
        # Compute overall metrics
        if elements is None:
            self.overall_metrics = self._compute_overall_metrics(y_pred, y_true)
        else:
            self.overall_metrics = self._compute_overall_metrics_from_rows(
                y_pred, y_true, self.per_atom_metrics
            )
        
        return {
            'per_atom': self.per_atom_metrics,
            'per_group': self.per_group_metrics,
            'overall': self.overall_metrics,
        }
    
    @staticmethod
    def _element_precision(predicted_elements: np.ndarray,
                           target_elements: np.ndarray,
                           element_set) -> float:
        predicted_in_set = np.isin(predicted_elements, list(element_set))
        n_predicted = int(predicted_in_set.sum())
        if n_predicted == 0:
            return 0.0
        hits = predicted_in_set & np.isin(target_elements, list(element_set))
        return float(hits.sum()) / n_predicted

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def _compute_per_atom_metrics(self, y_pred: np.ndarray, y_true: np.ndarray,
                                  elements=None) -> Dict[str, Dict[str, float]]:
        metrics = {}
        for idx, atom in enumerate(self.atom_types):
            y_pred_binary = (y_pred == idx).astype(int)
            y_true_binary = (y_true == idx).astype(int)
            
            support = int(y_true_binary.sum())
            
            if support == 0:
                # Skip atoms with zero support so they do not add noise to W&B.
                continue 
            
            acc = accuracy_score(y_true_binary, y_pred_binary)
            rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            if elements is None:
                prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            else:
                predicted_elements, target_elements = elements
                row_elements = set(target_elements[y_true_binary.astype(bool)].tolist())
                prec = self._element_precision(
                    predicted_elements, target_elements, row_elements
                )
                f1 = self._f1(prec, rec)
            
            metrics[atom] = {
                'Accuracy': float(acc),
                'Precision': float(prec),
                'Recall': float(rec),
                'F1': float(f1),
                'Support': support,
            }
        return metrics
    
    def _compute_group_metrics(self, y_pred: np.ndarray, y_true: np.ndarray,
                               elements=None) -> Dict[str, Dict[str, float]]:
        """Compute metrics for atom groups (Backbone, SideChain, etc.)."""
        metrics = {}
        
        for group_name, atom_list in self.groups.items():
            # Get indices for atoms in this group
            group_indices = [self.atom_to_idx[atom] for atom in atom_list if atom in self.atom_to_idx]
            
            if not group_indices:
                continue
            
            # Create binary classification: in group vs. outside group
            y_pred_binary = np.isin(y_pred, group_indices).astype(int)
            y_true_binary = np.isin(y_true, group_indices).astype(int)
            
            if y_true_binary.sum() == 0:
                metrics[group_name] = {
                    'Accuracy': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'F1': 0.0,
                    'Support': 0,
                }
                continue
            
            acc = accuracy_score(y_true_binary, y_pred_binary)
            rec = recall_score(y_true_binary, y_pred_binary, zero_division=self.zero_division)
            if elements is None:
                prec = precision_score(y_true_binary, y_pred_binary, zero_division=self.zero_division)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=self.zero_division)
            else:
                predicted_elements, target_elements = elements
                group_elements = set(target_elements[y_true_binary.astype(bool)].tolist())
                prec = self._element_precision(
                    predicted_elements, target_elements, group_elements
                )
                f1 = self._f1(prec, rec)
            support = y_true_binary.sum()
            
            metrics[group_name] = {
                'Accuracy': float(acc),
                'Precision': float(prec),
                'Recall': float(rec),
                'F1': float(f1),
                'Support': int(support),
            }
        
        return metrics
    
    def _compute_overall_metrics(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Compute overall classification metrics."""
        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=self.zero_division)
        rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=self.zero_division)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=self.zero_division)
        
        prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=self.zero_division)
        rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=self.zero_division)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=self.zero_division)
        
        return {
            'Accuracy': float(acc),
            'Precision_Macro': float(prec_macro),
            'Recall_Macro': float(rec_macro),
            'F1_Macro': float(f1_macro),
            'Precision_Weighted': float(prec_weighted),
            'Recall_Weighted': float(rec_weighted),
            'F1_Weighted': float(f1_weighted),
            'Total_Samples': int(len(y_true)),
        }
    
    @staticmethod
    def _compute_overall_metrics_from_rows(y_pred: np.ndarray, y_true: np.ndarray,
                                           per_atom_metrics) -> Dict[str, float]:
        """Macro/weighted averages over the vocabulary rows with support.

        Element-aware predictions use an out-of-vocabulary label for wrong
        elements; averaging with sklearn would add that label as an extra
        zero-recall class, so the averages are taken over the real rows.
        """
        rows = list(per_atom_metrics.values())
        supports = np.array([row['Support'] for row in rows], dtype=float)
        total = supports.sum()

        def macro(name):
            return float(np.mean([row[name] for row in rows])) if rows else 0.0

        def weighted(name):
            if total == 0:
                return 0.0
            return float(sum(row[name] * row['Support'] for row in rows) / total)

        return {
            'Accuracy': float(accuracy_score(y_true, y_pred)),
            'Precision_Macro': macro('Precision'),
            'Recall_Macro': macro('Recall'),
            'F1_Macro': macro('F1'),
            'Precision_Weighted': weighted('Precision'),
            'Recall_Weighted': weighted('Recall'),
            'F1_Weighted': weighted('F1'),
            'Total_Samples': int(len(y_true)),
        }

    def get_wandb_logs(self, prefix: str = '', namespace: str = 'atom_metrics') -> Dict[str, float]:
        """
        Format metrics for WandB logging.

        Args:
            prefix: Prefix for metric names (e.g., 'val_', 'test_')
            namespace: Top-level W&B section name (e.g., 'rna_atom_metrics', 'protein_atom_metrics').
                       Used to separate domain-specific metrics in the W&B UI.

        Returns:
            Dict with flat keys suitable for wandb.log()
        """
        logs = {}

        # Overall metrics
        for metric_name, value in self.overall_metrics.items():
            key = f"{prefix}{namespace}/{metric_name}"
            logs[key] = value

        # Per-atom metrics
        for atom, atom_metrics in self.per_atom_metrics.items():
            for metric_name, value in atom_metrics.items():
                key = f"{prefix}{namespace}/atom_{atom}/{metric_name}"
                # Skip support in wandb for cleaner interface
                if metric_name != 'Support':
                    logs[key] = value

        # Per-group metrics
        for group, group_metrics in self.per_group_metrics.items():
            for metric_name, value in group_metrics.items():
                key = f"{prefix}{namespace}/group_{group}/{metric_name}"
                if metric_name != 'Support':
                    logs[key] = value
        
        return logs
    
    def get_summary_table(self) -> str:
        """
        Get a formatted summary table of per-atom metrics.
        
        Returns:
            String with formatted table
        """
        try:
            from tabulate import tabulate
            use_tabulate = True
        except ImportError:
            use_tabulate = False
        
        # Per-atom table
        rows = []
        for atom in self.atom_types:
            if atom in self.per_atom_metrics:
                m = self.per_atom_metrics[atom]
                rows.append([
                    atom,
                    f"{m['Accuracy']:.4f}",
                    f"{m['Precision']:.4f}",
                    f"{m['Recall']:.4f}",
                    f"{m['F1']:.4f}",
                    m['Support'],
                ])
        
        if use_tabulate:
            table = tabulate(
                rows,
                headers=['Atom', 'Accuracy', 'Precision', 'Recall', 'F1', 'Support'],
                tablefmt='grid'
            )
        else:
            # Fallback: simple text formatting without tabulate
            headers = ['Atom', 'Accuracy', 'Precision', 'Recall', 'F1', 'Support']
            col_widths = [max(len(str(h)), max((len(str(row[i])) for row in rows), default=0)) for i, h in enumerate(headers)]
            
            # Header
            header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
            separator = "-" * len(header_line)
            
            # Rows
            table_lines = [separator, header_line, separator]
            for row in rows:
                row_line = " | ".join(str(val).ljust(w) for val, w in zip(row, col_widths))
                table_lines.append(row_line)
            table_lines.append(separator)
            
            table = "\n".join(table_lines)
        
        return table
    
    def print_summary(self):
        """Print comprehensive metrics summary."""
        print("\n" + "="*100)
        print("PER-ATOM CLASSIFICATION METRICS")
        print("="*100)
        
        # Overall metrics
        print("\n📊 OVERALL METRICS:")
        for metric_name, value in self.overall_metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name:25s}: {value:.4f}")
            else:
                print(f"  {metric_name:25s}: {value}")
        
        # Per-atom metrics
        print("\n🔷 PER-ATOM METRICS:")
        print(self.get_summary_table())
        
        # Per-group metrics
        if self.per_group_metrics:
            print("\n🔗 GROUP METRICS (", self.group_type, "):")
            rows = []
            for group in self.groups.keys():
                if group in self.per_group_metrics:
                    m = self.per_group_metrics[group]
                    rows.append([
                        group,
                        f"{m['Accuracy']:.4f}",
                        f"{m['Precision']:.4f}",
                        f"{m['Recall']:.4f}",
                        f"{m['F1']:.4f}",
                        m['Support'],
                    ])
            
            from tabulate import tabulate
            table = tabulate(
                rows,
                headers=['Group', 'Accuracy', 'Precision', 'Recall', 'F1', 'Support'],
                tablefmt='grid'
            )
            print(table)
        
        print("\n" + "="*100 + "\n")


class BatchPerAtomMetricsCalculator:
    """
    Real-time per-atom metrics calculator that computes metrics per batch.
    
    Useful for monitoring metrics during training/validation without storing all predictions.
    """
    
    def __init__(self, atom_types: Optional[List[str]] = None):
        """Initialize."""
        self.atom_types = atom_types or ALL_RNA_ATOMS
        self.atom_to_idx = {atom: i for i, atom in enumerate(self.atom_types)}
    
    def compute_batch_metrics(self,
                             predictions: Union[torch.Tensor, np.ndarray],
                             targets: Union[torch.Tensor, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for a single batch.
        
        Args:
            predictions: Model predictions. Shape: (batch_size,) or (batch_size, num_classes)
            targets: Ground truth. Shape: (batch_size,)
        
        Returns:
            Dict with 'per_atom', 'per_group', 'overall' keys
        """
        calc = PerAtomMetricsCalculator(atom_types=self.atom_types)
        calc.update(predictions, targets)
        return calc.compute()


def require_wandb_logger_or_announce(callback, trainer, pl_module, label: str):
    """Return the WandbLogger, raise if W&B is configured but absent.

    Without W&B (use_wandb=false) the tables are printed only; this is said
    once instead of skipping silently.
    """
    wandb_logger = next(
        (lg for lg in trainer.loggers if isinstance(lg, WandbLogger)), None
    )
    if wandb_logger is not None:
        return wandb_logger
    if pl_module is not None and getattr(pl_module.config, 'use_wandb', False):
        raise RuntimeError(f"{label}: use_wandb=true but the trainer has no WandbLogger")
    if not callback._no_wandb_announced:
        callback._no_wandb_announced = True
        print(f"ℹ️  {label}: no W&B logger (use_wandb=false); metrics are printed only")
    return None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_atom_confusion_matrix(y_pred: np.ndarray, 
                                 y_true: np.ndarray,
                                 atom_types: Optional[List[str]] = None) -> np.ndarray:
    """
    Create confusion matrix for atom classification.
    
    Args:
        y_pred: Predictions
        y_true: Targets
        atom_types: Atom type names
    
    Returns:
        Confusion matrix (num_atoms, num_atoms)
    """
    atom_types = atom_types or ALL_RNA_ATOMS
    num_atoms = len(atom_types)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_atoms)))
    return cm


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test metrics calculator
    calc = PerAtomMetricsCalculator()
    
    # Create dummy predictions and targets
    np.random.seed(42)
    n_samples = 1000
    n_atoms = len(ALL_RNA_ATOMS)
    
    y_true = np.random.randint(0, n_atoms, n_samples)
    y_pred = y_true.copy()
    # Add some noise
    noise_idx = np.random.choice(n_samples, n_samples // 10, replace=False)
    y_pred[noise_idx] = np.random.randint(0, n_atoms, len(noise_idx))
    
    # Update and compute
    calc.update(y_pred, y_true)
    metrics = calc.compute()
    
    # Print summary
    calc.print_summary()
    
    # Get WandB logs
    wandb_logs = calc.get_wandb_logs(prefix='val_')
    print("WandB log keys:", len(wandb_logs))
    print("Sample keys:", list(wandb_logs.keys())[:5])

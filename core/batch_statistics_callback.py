"""
Batch Statistics Callback for Multi-Domain Training

Tracks and logs statistics about batch composition:
- Number of samples from each dataset (PDB, QM9, etc.)
- Number of atoms from each dataset
- Dataset distribution per batch
"""

import torch
from pytorch_lightning.callbacks import Callback
from collections import defaultdict


class BatchStatisticsCallback(Callback):
    """
    Callback to track and log batch composition in multi-domain training.
    
    Logs every N batches:
    - Samples per dataset
    - Atoms per dataset
    - Dataset ratios
    """
    
    def __init__(self, log_every_n_batches=10, enabled=True):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.enabled = enabled
        self.batch_stats = []
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Track batch composition at the start of each training batch."""
        if not self.enabled:
            return
        
        # Only log every N batches to avoid spam
        if batch_idx % self.log_every_n_batches != 0:
            return
        
        # Extract batch information
        try:
            # Get dataset_type if available (added by LazyUniversalDataset)
            if hasattr(batch, 'dataset_type'):
                dataset_types = batch.dataset_type
                num_nodes = batch.num_nodes if hasattr(batch, 'num_nodes') else batch.x.size(0)
                
                # Count samples per dataset type
                stats = self._compute_batch_stats(dataset_types, batch)
                
                # Print statistics
                self._print_batch_stats(batch_idx, stats)
                
                # Store for later analysis
                self.batch_stats.append({
                    'batch_idx': batch_idx,
                    'epoch': trainer.current_epoch,
                    **stats
                })
        except Exception as e:
            # Silently skip if batch doesn't have required attributes
            pass
    
    def _compute_batch_stats(self, dataset_types, batch):
        """Compute statistics about batch composition."""
        stats = defaultdict(lambda: {'samples': 0, 'atoms': 0})
        
        # If dataset_types is a list/tensor
        if isinstance(dataset_types, (list, tuple)):
            for dt in dataset_types:
                stats[dt]['samples'] += 1
        elif isinstance(dataset_types, torch.Tensor):
            unique, counts = torch.unique(dataset_types, return_counts=True)
            for dt, count in zip(unique.tolist(), counts.tolist()):
                stats[dt]['samples'] = count
        elif isinstance(dataset_types, str):
            # Single dataset type for whole batch
            stats[dataset_types]['samples'] = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        
        # Count atoms per dataset if batch_mapping available
        if hasattr(batch, 'batch') and hasattr(batch, 'dataset_type'):
            batch_mapping = batch.batch
            
            # If dataset_type is per-graph
            if isinstance(dataset_types, (list, tuple, torch.Tensor)):
                for graph_idx in range(batch.num_graphs):
                    # Count atoms for this graph
                    atoms_in_graph = (batch_mapping == graph_idx).sum().item()
                    
                    # Get dataset type for this graph
                    if isinstance(dataset_types, torch.Tensor):
                        dt = dataset_types[graph_idx].item() if dataset_types.dim() > 0 else dataset_types.item()
                    else:
                        dt = dataset_types[graph_idx] if graph_idx < len(dataset_types) else dataset_types[0]
                    
                    stats[dt]['atoms'] += atoms_in_graph
        
        return dict(stats)
    
    def _print_batch_stats(self, batch_idx, stats):
        """Print formatted batch statistics."""
        print(f"\n{'='*80}")
        print(f"📊 Batch {batch_idx} Composition:")
        print(f"{'='*80}")
        
        total_samples = sum(s['samples'] for s in stats.values())
        total_atoms = sum(s['atoms'] for s in stats.values())
        
        for dataset_type, data in sorted(stats.items()):
            samples = data['samples']
            atoms = data['atoms']
            
            sample_pct = (samples / total_samples * 100) if total_samples > 0 else 0
            atom_pct = (atoms / total_atoms * 100) if total_atoms > 0 else 0
            
            print(f"  {dataset_type.upper():12s}: "
                  f"{samples:4d} samples ({sample_pct:5.1f}%) | "
                  f"{atoms:6d} atoms ({atom_pct:5.1f}%)")
        
        print(f"  {'TOTAL':12s}: {total_samples:4d} samples        | {total_atoms:6d} atoms")
        print(f"{'='*80}\n")
    
    def on_train_end(self, trainer, pl_module):
        """Print summary statistics at the end of training."""
        if not self.enabled or not self.batch_stats:
            return
        
        print(f"\n{'='*80}")
        print(f"📊 TRAINING BATCH STATISTICS SUMMARY")
        print(f"{'='*80}")
        print(f"Total batches tracked: {len(self.batch_stats)}")
        
        # Aggregate statistics
        dataset_totals = defaultdict(lambda: {'samples': 0, 'atoms': 0, 'batches': 0})
        
        for batch_stat in self.batch_stats:
            for dt in ['pdb', 'qm9', 'lba', 'metabolite', 'rna', 'dna']:
                if dt in batch_stat:
                    dataset_totals[dt]['samples'] += batch_stat[dt]['samples']
                    dataset_totals[dt]['atoms'] += batch_stat[dt]['atoms']
                    dataset_totals[dt]['batches'] += 1
        
        total_samples = sum(d['samples'] for d in dataset_totals.values())
        total_atoms = sum(d['atoms'] for d in dataset_totals.values())
        
        for dt, data in sorted(dataset_totals.items()):
            if data['batches'] > 0:
                sample_pct = (data['samples'] / total_samples * 100) if total_samples > 0 else 0
                atom_pct = (data['atoms'] / total_atoms * 100) if total_atoms > 0 else 0
                
                print(f"  {dt.upper():12s}: "
                      f"{data['samples']:6d} samples ({sample_pct:5.1f}%) | "
                      f"{data['atoms']:8d} atoms ({atom_pct:5.1f}%) | "
                      f"{data['batches']:3d} batches")
        
        print(f"{'='*80}\n")


# Simple function to add dataset_type to batch (if not already present)
def add_dataset_type_to_batch(batch, chunk_file_path):
    """
    Add dataset_type attribute to batch based on chunk file name.
    
    Args:
        batch: PyG batch object
        chunk_file_path: Path to the .pt file this batch came from
    
    Returns:
        batch with dataset_type attribute
    """
    import os
    filename = os.path.basename(chunk_file_path)
    
    # Detect dataset type from filename
    if 'pdb' in filename.lower() or 'protein' in filename.lower():
        dataset_type = 'pdb'
    elif 'qm9' in filename.lower():
        dataset_type = 'qm9'
    elif 'lba' in filename.lower():
        dataset_type = 'lba'
    elif 'metabolite' in filename.lower():
        dataset_type = 'metabolite'
    elif 'rna' in filename.lower():
        dataset_type = 'rna'
    elif 'dna' in filename.lower():
        dataset_type = 'dna'
    else:
        dataset_type = 'unknown'
    
    # Add as attribute
    if not hasattr(batch, 'dataset_type'):
        batch.dataset_type = dataset_type
    
    return batch


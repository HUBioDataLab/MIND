"""
Batch Statistics Callback for Multi-Domain Training

Tracks and logs statistics about batch composition:
- Number of samples from each dataset (PDB, QM9, etc.)
- Number of atoms from each dataset
- Dataset distribution per batch
"""

import math
import torch
import time
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
        if int(log_every_n_batches) <= 0:
            raise ValueError("log_every_n_batches must be positive")
        self.log_every_n_batches = log_every_n_batches
        self.enabled = enabled
        self.batch_stats = []
        self._active_batch_stat = None
        self._previous_batch_end = None
        
        if self.enabled:
            print(f"✅ BatchStatisticsCallback enabled (logging every {log_every_n_batches} batches)")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Track batch composition at the start of each training batch."""
        if not self.enabled:
            return
        
        try:
            started_at = time.perf_counter()
            num_nodes = int(batch.num_nodes if hasattr(batch, 'num_nodes') else batch.x.size(0))
            num_graphs = int(batch.num_graphs if hasattr(batch, 'num_graphs') else (batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1))
            num_edges = int(batch.edge_index.shape[1]) if hasattr(batch, 'edge_index') else 0
            graph_sizes = (
                torch.bincount(batch.batch, minlength=num_graphs).detach().cpu().tolist()
                if hasattr(batch, 'batch') else [num_nodes]
            )
            stats = {}
            
            # Get dataset_type if available (added by LazyUniversalDataset or multi-domain)
            if hasattr(batch, 'dataset_type'):
                dataset_types = batch.dataset_type
                # Count samples per dataset type
                stats = self._compute_batch_stats(dataset_types, batch, graph_sizes)
                if batch_idx % self.log_every_n_batches == 0:
                    self._print_batch_stats(batch_idx, stats, num_nodes, num_graphs)
            else:
                if batch_idx % self.log_every_n_batches == 0:
                    self._print_single_domain_stats(batch_idx, batch, num_nodes, num_graphs)

            self._active_batch_stat = {
                'batch_idx': batch_idx,
                'epoch': trainer.current_epoch,
                'num_nodes': num_nodes,
                'num_graphs': num_graphs,
                'num_edges': num_edges,
                'graph_sizes': graph_sizes,
                'domains': stats,
                'emit_telemetry': batch_idx % self.log_every_n_batches == 0,
                'data_wait_seconds': (
                    started_at - self._previous_batch_end
                    if self._previous_batch_end is not None else None
                ),
                '_started_at': started_at,
            }
        except Exception as e:
            raise RuntimeError(
                f"Batch statistics failed at epoch={trainer.current_epoch}, "
                f"batch={batch_idx}: {e}"
            ) from e

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Finish timing and emit one compact measured telemetry line per batch."""
        if not self.enabled or self._active_batch_stat is None:
            return
        ended_at = time.perf_counter()
        record = self._active_batch_stat
        step_seconds = ended_at - record.pop('_started_at')
        record['step_seconds'] = step_seconds
        record['atoms_per_second'] = record['num_nodes'] / step_seconds
        record['edges_per_second'] = record['num_edges'] / step_seconds
        record['graphs_per_second'] = record['num_graphs'] / step_seconds
        template_stats = getattr(batch, 'virtual_template_stats', None) or {}
        record['template_total_ms'] = template_stats.get('template_total_ms')
        record['edge_rebuild_ms'] = template_stats.get('edge_rebuild_ms')
        # Process-local allocator peak since start (other users' GPU memory excluded).
        record['peak_gpu_mb'] = (
            torch.cuda.max_memory_allocated() / 2**20
            if torch.cuda.is_available() and torch.cuda.is_initialized() else None
        )
        self.batch_stats.append(record)
        wait_ms = 1000.0 * (record['data_wait_seconds'] or 0.0)
        template_ms = (
            f"{record['template_total_ms']:.2f}"
            if record['template_total_ms'] is not None else "na"
        )
        edge_rebuild_ms = (
            f"{record['edge_rebuild_ms']:.2f}"
            if record['edge_rebuild_ms'] is not None else "na"
        )
        peak_gpu = (
            f"{record['peak_gpu_mb']:.0f}" if record['peak_gpu_mb'] is not None else "na"
        )
        if record['emit_telemetry']:
            print(
                f"TELEMETRY epoch={record['epoch']} batch={batch_idx} "
                f"graphs={record['num_graphs']} atoms={record['num_nodes']} "
                f"edges={record['num_edges']} step_ms={1000.0 * step_seconds:.2f} "
                f"data_wait_ms={wait_ms:.2f} atoms_s={record['atoms_per_second']:.1f} "
                f"edges_s={record['edges_per_second']:.1f} "
                f"graphs_s={record['graphs_per_second']:.2f} "
                f"template_ms={template_ms} edge_rebuild_ms={edge_rebuild_ms} "
                f"peak_gpu_mb={peak_gpu}"
            )
        self._previous_batch_end = ended_at
        self._active_batch_stat = None
    
    def _compute_batch_stats(self, dataset_types, batch, graph_sizes=None):
        """Compute statistics with one vector transfer and CPU aggregation."""
        stats = defaultdict(lambda: {'samples': 0, 'atoms': 0})

        num_graphs = int(batch.num_graphs if hasattr(batch, 'num_graphs') else 1)
        if isinstance(dataset_types, str):
            graph_domains = [dataset_types] * num_graphs
        elif isinstance(dataset_types, torch.Tensor):
            graph_domains = dataset_types.detach().cpu().tolist()
        else:
            graph_domains = list(dataset_types)
        if len(graph_domains) != num_graphs:
            raise ValueError(
                f"dataset_type count {len(graph_domains)} != graph count {num_graphs}"
            )

        if graph_sizes is None:
            if not hasattr(batch, 'batch'):
                graph_sizes = [int(batch.num_nodes)]
            else:
                graph_sizes = torch.bincount(
                    batch.batch, minlength=num_graphs
                ).detach().cpu().tolist()
        if len(graph_sizes) != num_graphs:
            raise ValueError(
                f"graph size count {len(graph_sizes)} != graph count {num_graphs}"
            )

        # The only device-to-host transfer is the graph-size vector above.
        # All per-graph aggregation happens on ordinary Python values.
        for dt, atoms in zip(graph_domains, graph_sizes):
            stats[dt]['samples'] += 1
            stats[dt]['atoms'] += int(atoms)
        return dict(stats)
    
    def _print_batch_stats(self, batch_idx, stats, num_nodes, num_graphs):
        """Print formatted batch statistics for multi-domain training."""
        print(f"\n{'='*80}")
        print(f"📊 Batch {batch_idx} Composition (Multi-Domain):")
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
        print(f"  {'SUMMARY':12s}: {num_graphs:4d} graphs, {num_nodes:6d} total nodes")
        print(f"{'='*80}\n")
    
    def _print_single_domain_stats(self, batch_idx, batch, num_nodes, num_graphs):
        """Print batch statistics for single-domain training."""
        # Calculate atoms per graph if batch_mapping is available
        atoms_per_graph = []
        if hasattr(batch, 'batch'):
            batch_mapping = batch.batch
            for graph_idx in range(num_graphs):
                atoms_in_graph = (batch_mapping == graph_idx).sum().item()
                atoms_per_graph.append(atoms_in_graph)
        
        avg_atoms = sum(atoms_per_graph) / len(atoms_per_graph) if atoms_per_graph else num_nodes / num_graphs
        min_atoms = min(atoms_per_graph) if atoms_per_graph else 0
        max_atoms = max(atoms_per_graph) if atoms_per_graph else 0
        
        print(f"\n{'='*80}")
        print(f"📊 Batch {batch_idx} Statistics:")
        print(f"{'='*80}")
        print(f"  Graphs:        {num_graphs:4d} {'⚠️ Fixed batch size!' if num_graphs == 32 else '✅ Dynamic batching'}")
        print(f"  Total Atoms:   {num_nodes:6d}")
        print(f"  Avg Atoms/Graph: {avg_atoms:6.1f}")
        if atoms_per_graph:
            print(f"  Min Atoms:     {min_atoms:6d}")
            print(f"  Max Atoms:     {max_atoms:6d}")
        print(f"{'='*80}\n")
    
    @staticmethod
    def _percentile(values, fraction):
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
        return float(ordered[index])

    def on_train_epoch_end(self, trainer, pl_module):
        """Print exact domain totals and graph/batch/timing quantiles for the epoch."""
        epoch = trainer.current_epoch
        records = [item for item in self.batch_stats if item['epoch'] == epoch]
        if not self.enabled or not records:
            return
        
        print(f"\n{'='*80}")
        print(f"📊 TRAINING BATCH STATISTICS SUMMARY — EPOCH {epoch}")
        print(f"{'='*80}")
        print(f"Total batches tracked: {len(records)}")
        
        # Aggregate statistics
        dataset_totals = defaultdict(lambda: {'samples': 0, 'atoms': 0, 'batches': 0})
        
        for batch_stat in records:
            for dt, domain_stat in batch_stat['domains'].items():
                dataset_totals[dt]['samples'] += domain_stat['samples']
                dataset_totals[dt]['atoms'] += domain_stat['atoms']
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

        graph_sizes = [size for item in records for size in item['graph_sizes']]
        batch_sizes = [item['num_nodes'] for item in records]
        step_ms = [1000.0 * item['step_seconds'] for item in records]
        wait_ms = [
            1000.0 * item['data_wait_seconds']
            for item in records if item['data_wait_seconds'] is not None
        ]
        total_seconds = sum(item['step_seconds'] for item in records)
        print(
            f"  GRAPH NODES p50={self._percentile(graph_sizes, 0.50):.0f} "
            f"p90={self._percentile(graph_sizes, 0.90):.0f} max={max(graph_sizes):.0f}"
        )
        print(
            f"  BATCH NODES p50={self._percentile(batch_sizes, 0.50):.0f} "
            f"p90={self._percentile(batch_sizes, 0.90):.0f} max={max(batch_sizes):.0f}"
        )
        print(
            f"  STEP ms p50={self._percentile(step_ms, 0.50):.2f} "
            f"p90={self._percentile(step_ms, 0.90):.2f}; "
            f"DATA WAIT ms p50={self._percentile(wait_ms, 0.50):.2f} "
            f"p90={self._percentile(wait_ms, 0.90):.2f}"
        )
        peaks = [item['peak_gpu_mb'] for item in records if item.get('peak_gpu_mb') is not None]
        if peaks:
            print(f"  PEAK GPU allocated MB={max(peaks):.0f} (process-local)")
        print(
            f"  THROUGHPUT atoms/s={sum(batch_sizes) / total_seconds:.1f} "
            f"edges/s={sum(item['num_edges'] for item in records) / total_seconds:.1f} "
            f"graphs/s={sum(item['num_graphs'] for item in records) / total_seconds:.2f}"
        )
        template_ms = [
            item['template_total_ms']
            for item in records if item['template_total_ms'] is not None
        ]
        edge_rebuild_ms = [
            item['edge_rebuild_ms']
            for item in records if item['edge_rebuild_ms'] is not None
        ]
        if template_ms:
            print(
                f"  TEMPLATE ms p50={self._percentile(template_ms, 0.50):.2f} "
                f"p90={self._percentile(template_ms, 0.90):.2f}; "
                f"EDGE REBUILD ms p50={self._percentile(edge_rebuild_ms, 0.50):.2f} "
                f"p90={self._percentile(edge_rebuild_ms, 0.90):.2f}"
            )

        dataset = getattr(trainer.train_dataloader, 'dataset', None)
        while dataset is not None and not hasattr(dataset, 'get_cache_telemetry'):
            dataset = getattr(dataset, 'dataset', None)
        if dataset is not None:
            cache = dataset.get_cache_telemetry()
            print(
                f"  CACHE hits={cache['hits']} misses={cache['misses']} "
                f"hit_rate={100.0 * cache['hit_rate']:.2f}% "
                f"loads={cache['loads']} load_seconds={cache['load_seconds']:.3f}; "
                f"template_metadata_hits={cache.get('template_metadata_hits', 0)} "
                f"misses={cache.get('template_metadata_misses', 0)}"
            )
        
        print(f"{'='*80}\n")

    def on_train_end(self, trainer, pl_module):
        """The exact summaries are emitted at each epoch boundary."""
        return


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

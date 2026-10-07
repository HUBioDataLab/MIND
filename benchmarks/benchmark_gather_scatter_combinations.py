#!/usr/bin/env python3
"""
==============================================================================
MIND Project - Gather-Scatter & Combinatorial Ablation Benchmark
==============================================================================
This benchmark:
1. Directly benchmarks PyG C++ scatter_add vs OpenAI Triton Fused Gather-Scatter
   across different feature dimensions (D=1, 2, 8, 32, 64, 128, 256) in float32.
2. Benchmarks full end-to-end model training performance with different combinations
   (with vs without Triton Gather-Scatter, and best-performing combinations).
==============================================================================
"""

import os
import sys
import time
import json
import yaml
import copy
import gc
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch_scatter import scatter_add

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pretraining_config import PretrainingConfig, create_pretraining_config
from core.pretraining_model import PretrainingESAModel
from core.train_pretrain import load_universal_dataset, create_data_loaders
from esa.triton_graph_gather_scatter import triton_fused_gather_scatter


def benchmark_kernel_level(num_nodes=5000, num_edges=50000, dims=[1, 2, 8, 32, 64, 128, 256], iters=1000):
    """Directly compares PyG scatter_add vs Triton fused gather-scatter at pure kernel level."""
    print("\n" + "="*95)
    print(f"🔬 PURE KERNEL BENCHMARK (Float32): PyG scatter_add vs Triton Gather-Scatter")
    print(f"   (Nodes: {num_nodes:,}, Edges: {num_edges:,}, Iterations: {iters:,})")
    print("="*95)
    print(f"{'Feature Dim (D)':<16} | {'PyG scatter_add (ms)':<22} | {'Triton Fused (ms)':<20} | {'Kernel Speedup':<16} | {'Winner':<10}")
    print("-" * 95)

    device = torch.device("cuda:0")
    src = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_index = torch.stack([src, dst], dim=0)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    kernel_results = []

    for d in dims:
        x = torch.randn((num_edges, d), device=device, dtype=torch.float32, requires_grad=True)
        grad_out = torch.randn((num_nodes, d), device=device, dtype=torch.float32)

        # Warmup PyG
        for _ in range(50):
            if d == 1:
                out = scatter_add(x.squeeze(-1), src, dim=0, dim_size=num_nodes)
            else:
                out = scatter_add(x, src, dim=0, dim_size=num_nodes)
            out.backward(grad_out.squeeze(-1) if d == 1 else grad_out, retain_graph=True)
            x.grad = None

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(iters):
            if d == 1:
                out = scatter_add(x.squeeze(-1), src, dim=0, dim_size=num_nodes)
            else:
                out = scatter_add(x, src, dim=0, dim_size=num_nodes)
            out.backward(grad_out.squeeze(-1) if d == 1 else grad_out, retain_graph=True)
            x.grad = None
        end_event.record()
        torch.cuda.synchronize()
        pyg_time = start_event.elapsed_time(end_event) / iters

        # Warmup Triton
        for _ in range(50):
            out_triton = triton_fused_gather_scatter(x, edge_index, num_nodes=num_nodes)
            out_triton.backward(grad_out, retain_graph=True)
            x.grad = None

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(iters):
            out_triton = triton_fused_gather_scatter(x, edge_index, num_nodes=num_nodes)
            out_triton.backward(grad_out, retain_graph=True)
            x.grad = None
        end_event.record()
        torch.cuda.synchronize()
        triton_time = start_event.elapsed_time(end_event) / iters

        speedup = pyg_time / triton_time
        if triton_time < pyg_time:
            winner = "Triton 🚀"
            speedup_str = f"+{(speedup - 1)*100:5.1f}% ({speedup:.2f}x)"
        else:
            winner = "PyG ⚡"
            speedup_str = f"-{(1 - 1/speedup)*100:5.1f}% ({speedup:.2f}x)"

        row = f"D = {d:<12} | {pyg_time:18.4f} ms | {triton_time:16.4f} ms | {speedup_str:<16} | {winner:<10}"
        print(row)
        kernel_results.append({
            "dim": d,
            "pyg_ms": pyg_time,
            "triton_ms": triton_time,
            "speedup": speedup,
            "winner": "Triton" if triton_time < pyg_time else "PyG"
        })

    print("="*95 + "\n")
    return kernel_results


def run_model_benchmark(exp_name: str, config: PretrainingConfig, dataset, device, warmup=5, steps=30):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = PretrainingESAModel(config).to(device)
    model.log = lambda *args, **kwargs: None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.optimiser_weight_decay)
    train_loader, _, _ = create_data_loaders(dataset, config)
    model.train()

    start_event = torch.cuda.Event(enable_timing=True)
    fwd_end_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    step_times = []
    fwd_times = []
    bwd_times = []
    total_atoms = 0
    total_samples = 0

    bench_start_wall = None
    step_idx = 0

    for batch_idx, batch in enumerate(train_loader):
        if step_idx >= warmup + steps:
            break

        batch = batch.to(device)
        num_atoms = batch.num_nodes if hasattr(batch, 'num_nodes') else batch.x.shape[0]
        num_samples = batch.num_graphs if hasattr(batch, 'num_graphs') else 1

        optimizer.zero_grad(set_to_none=True)

        is_bench = (step_idx >= warmup)
        if is_bench and bench_start_wall is None:
            torch.cuda.synchronize()
            bench_start_wall = time.perf_counter()

        start_event.record()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss_dict = model.training_step(batch, batch_idx)
            loss = loss_dict['loss'] if isinstance(loss_dict, dict) else loss_dict
        fwd_end_event.record()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
        optimizer.step()
        end_event.record()
        torch.cuda.synchronize()

        fwd_ms = start_event.elapsed_time(fwd_end_event)
        step_ms = start_event.elapsed_time(end_event)
        bwd_ms = step_ms - fwd_ms

        if is_bench:
            step_times.append(step_ms)
            fwd_times.append(fwd_ms)
            bwd_times.append(bwd_ms)
            total_atoms += num_atoms
            total_samples += num_samples

        step_idx += 1

    torch.cuda.synchronize()
    bench_end_wall = time.perf_counter()
    total_wall = bench_end_wall - bench_start_wall if bench_start_wall else 1.0

    alloc_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    res_vram = torch.cuda.max_memory_reserved() / (1024 ** 3)
    avg_step = sum(step_times) / len(step_times) if step_times else 0.0
    avg_fwd = sum(fwd_times) / len(fwd_times) if fwd_times else 0.0
    avg_bwd = sum(bwd_times) / len(bwd_times) if bwd_times else 0.0
    samples_sec = total_samples / total_wall if total_wall > 0 else 0.0
    atoms_sec = total_atoms / total_wall if total_wall > 0 else 0.0

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": exp_name,
        "avg_step_ms": avg_step,
        "avg_fwd_ms": avg_fwd,
        "avg_bwd_ms": avg_bwd,
        "samples_per_sec": samples_sec,
        "atoms_per_sec": atoms_sec,
        "peak_vram_gb": alloc_vram
    }


def main():
    # Step 1: Pure Kernel Benchmark
    benchmark_kernel_level()

    # Step 2: End-to-End Combinations Benchmark on Protein Dataset
    print("\n" + "="*115)
    print("🚀 END-TO-END MODEL TRAINING: COMBINATIONS BENCHMARK (Protein 400 Samples)")
    print("="*115)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_path = "core/pretraining_config_protein_optimized.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config_dict['max_samples'] = 400
    config_dict['use_bfloat16'] = True
    config_dict['precision'] = "bfloat16"
    config_dict['num_workers'] = 0

    base_config = create_pretraining_config(**config_dict)
    dataset = load_universal_dataset(base_config, base_config.dataset, getattr(base_config, 'dataset_download_dir', './data_loading/protein_processed'))

    combos = []

    # 1. Baseline SDPA (No optimizations)
    c1 = copy.deepcopy(base_config)
    c1.use_vectorized_geometric_features = False
    c1.use_vectorized_mlm_masking = False
    c1.use_vectorized_edge_masking = False
    c1.use_triton_gather_scatter = False
    c1.use_triton_flash_attention = False
    c1.xformers_or_torch_attn = "torch"
    combos.append(("1. Baseline (PyTorch SDPA + Standard PyG)", c1))

    # 2. Triton FlashAttention ONLY
    c2 = copy.deepcopy(base_config)
    c2.use_vectorized_geometric_features = False
    c2.use_vectorized_mlm_masking = False
    c2.use_vectorized_edge_masking = False
    c2.use_triton_gather_scatter = False
    c2.use_triton_flash_attention = True
    c2.use_triton_varlen_flash_attention = True
    c2.xformers_or_torch_attn = "triton"
    combos.append(("2. Triton FlashAttn ONLY (Standard PyG)", c2))

    # 3. Best Combo: Triton FlashAttn + Vectorized MLM/Edge/Geo (NO Triton Gather-Scatter)
    c3 = copy.deepcopy(base_config)
    c3.use_vectorized_geometric_features = True
    c3.use_vectorized_mlm_masking = True
    c3.use_vectorized_edge_masking = True
    c3.use_triton_gather_scatter = False   # <- PyG scatter_add
    c3.use_triton_flash_attention = True
    c3.use_triton_varlen_flash_attention = True
    c3.xformers_or_torch_attn = "triton"
    combos.append(("3. ⭐ Optimal Suite: Triton FlashAttn + Vectorized (PyG Scatter)", c3))

    # 4. Full Suite: Triton FlashAttn + Vectorized + Triton Gather-Scatter
    c4 = copy.deepcopy(base_config)
    c4.use_vectorized_geometric_features = True
    c4.use_vectorized_mlm_masking = True
    c4.use_vectorized_edge_masking = True
    c4.use_triton_gather_scatter = True    # <- Triton Gather-Scatter
    c4.use_triton_flash_attention = True
    c4.use_triton_varlen_flash_attention = True
    c4.xformers_or_torch_attn = "triton"
    combos.append(("4. Full Suite: Triton FlashAttn + Vectorized + Triton Gather-Scatter", c4))

    # 5. SDPA + Vectorized Suite (PyG Scatter)
    c5 = copy.deepcopy(base_config)
    c5.use_vectorized_geometric_features = True
    c5.use_vectorized_mlm_masking = True
    c5.use_vectorized_edge_masking = True
    c5.use_triton_gather_scatter = False
    c5.use_triton_flash_attention = False
    c5.xformers_or_torch_attn = "torch"
    combos.append(("5. SDPA + All Vectorized Optimizations (PyG Scatter)", c5))

    results = []
    for name, cfg in combos:
        print(f"🔄 Benchmarking: {name}...")
        r = run_model_benchmark(name, cfg, dataset, device, warmup=5, steps=25)
        results.append(r)
        print(f"   ➔ Step: {r['avg_step_ms']:.2f} ms | Throughput: {r['samples_per_sec']:.2f} spl/s | VRAM: {r['peak_vram_gb']:.2f} GB")

    base_ms = results[0]["avg_step_ms"]

    print("\n" + "="*115)
    print("🏆 FINAL COMBINATIONS COMPARISON TABLE")
    print("="*115)
    print(f"{'Configuration':<60} | {'Step Time':<11} | {'Throughput':<14} | {'Speedup':<14} | {'VRAM':<9}")
    print("-" * 115)

    for r in results:
        speedup = ((base_ms - r['avg_step_ms']) / base_ms * 100) if base_ms > 0 else 0.0
        factor = (base_ms / r['avg_step_ms']) if r['avg_step_ms'] > 0 else 1.0
        speedup_str = f"+{speedup:5.1f}% ({factor:.2f}x)" if r != results[0] else "Baseline (1.0x)"
        row = f"{r['name']:<60} | {r['avg_step_ms']:7.2f} ms  | {r['samples_per_sec']:6.2f} spl/s  | {speedup_str:<14} | {r['peak_vram_gb']:5.2f} GB"
        print(row)

    print("="*115 + "\n")


if __name__ == "__main__":
    main()

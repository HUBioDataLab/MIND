#!/usr/bin/env python3
"""
==============================================================================
MIND Project - Pretraining Optimization Ablation Study & Benchmark
==============================================================================
This script isolates and benchmarks each optimization component independently
against the baseline implementation.

Ablation Configurations Tested:
  [1] Baseline (SDPA, No Triton, No Vectorization)
  [2] + Opt 1: Vectorized Geometric Features & COM Centering
  [3] + Opt 2: Vectorized MLM GPU Masking
  [4] + Opt 3: Vectorized Edge/Pair Masking
  [5] + Opt 4: OpenAI Triton Fused Gather-Scatter Graph Kernel
  [6] + Opt 5 & 8: OpenAI Triton Block-Sparse & Varlen FlashAttention
  [7] Full Optimization Suite (ALL Optimizations Enabled)

Hardware & Precision:
  - Precision: BFloat16 AMP across all experiments
  - Metrics: Throughput (samples/s, atoms/s), Step Time (ms), VRAM (GB), Loss
==============================================================================
"""

import os
import sys
import time
import json
import yaml
import argparse
import copy
import gc
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pretraining_config import PretrainingConfig, create_pretraining_config
from core.pretraining_model import PretrainingESAModel
from core.train_pretrain import load_universal_dataset, create_data_loaders


def get_gpu_memory_gb():
    """Returns current allocated and reserved GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    return allocated, reserved


def run_benchmark_experiment(
    exp_name: str,
    exp_config: PretrainingConfig,
    dataset,
    device: torch.device,
    num_warmup_steps: int = 5,
    max_steps: int = 40
) -> Dict[str, Any]:
    """Runs a single ablation benchmark experiment with precise CUDA event timing."""
    print(f"\n{'='*80}")
    print(f"🚀 Running Ablation: {exp_name}")
    print(f"{'='*80}")

    # Force garbage collection and reset CUDA memory stats
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create model
    model = PretrainingESAModel(exp_config).to(device)
    model.log = lambda *args, **kwargs: None  # Mock Lightning logger for direct execution
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_config.lr, weight_decay=exp_config.optimiser_weight_decay)

    # Create dataloader
    train_loader, _, _ = create_data_loaders(dataset, exp_config)

    model.train()
    step_times_ms = []
    forward_times_ms = []
    backward_times_ms = []
    total_atoms_processed = 0
    total_samples_processed = 0
    losses = []

    start_event = torch.cuda.Event(enable_timing=True)
    fwd_end_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"⏱️  Warmup ({num_warmup_steps} steps) + Benchmark ({max_steps} steps)...")

    step_idx = 0
    bench_start_wall = None
    bench_samples = 0
    bench_atoms = 0

    for batch_idx, batch in enumerate(train_loader):
        if step_idx >= num_warmup_steps + max_steps:
            break

        batch = batch.to(device)
        num_atoms = batch.num_nodes if hasattr(batch, 'num_nodes') else batch.x.shape[0]
        num_samples = batch.num_graphs if hasattr(batch, 'num_graphs') else 1

        optimizer.zero_grad(set_to_none=True)

        is_benchmark_step = (step_idx >= num_warmup_steps)
        if is_benchmark_step and bench_start_wall is None:
            torch.cuda.synchronize()
            bench_start_wall = time.perf_counter()

        start_event.record()

        # Forward pass in bfloat16
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss_dict = model.training_step(batch, batch_idx)
            loss = loss_dict['loss'] if isinstance(loss_dict, dict) else loss_dict

        fwd_end_event.record()

        # Backward pass & optimizer step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), exp_config.gradient_clip_val)
        optimizer.step()

        end_event.record()
        torch.cuda.synchronize()

        fwd_ms = start_event.elapsed_time(fwd_end_event)
        step_ms = start_event.elapsed_time(end_event)
        bwd_ms = step_ms - fwd_ms

        losses.append(loss.item())

        if is_benchmark_step:
            step_times_ms.append(step_ms)
            forward_times_ms.append(fwd_ms)
            backward_times_ms.append(bwd_ms)
            bench_atoms += num_atoms
            bench_samples += num_samples

        if step_idx % 10 == 0:
            print(f"  Step {step_idx:3d} | Loss: {loss.item():.4f} | Step: {step_ms:6.2f} ms (Fwd: {fwd_ms:5.2f} ms, Bwd: {bwd_ms:5.2f} ms)")

        step_idx += 1

    torch.cuda.synchronize()
    bench_end_wall = time.perf_counter()
    total_wall_time = bench_end_wall - bench_start_wall if bench_start_wall else 1.0

    alloc_vram, res_vram = get_gpu_memory_gb()
    avg_step_ms = sum(step_times_ms) / len(step_times_ms) if step_times_ms else 0.0
    avg_fwd_ms = sum(forward_times_ms) / len(forward_times_ms) if forward_times_ms else 0.0
    avg_bwd_ms = sum(backward_times_ms) / len(backward_times_ms) if backward_times_ms else 0.0
    samples_per_sec = bench_samples / total_wall_time if total_wall_time > 0 else 0.0
    atoms_per_sec = bench_atoms / total_wall_time if total_wall_time > 0 else 0.0

    print(f"\n📊 Results for [{exp_name}]:")
    print(f"   • Avg Step Time: {avg_step_ms:.2f} ms (Fwd: {avg_fwd_ms:.2f} ms, Bwd: {avg_bwd_ms:.2f} ms)")
    print(f"   • Throughput:    {samples_per_sec:.2f} samples/s | {atoms_per_sec:,.1f} atoms/s")
    print(f"   • Peak VRAM:     Allocated: {alloc_vram:.2f} GB | Reserved: {res_vram:.2f} GB")
    print(f"   • Loss:          Initial: {losses[0]:.4f} ➔ Final: {losses[-1]:.4f}")

    # Clean up model
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": exp_name,
        "avg_step_ms": avg_step_ms,
        "avg_fwd_ms": avg_fwd_ms,
        "avg_bwd_ms": avg_bwd_ms,
        "samples_per_sec": samples_per_sec,
        "atoms_per_sec": atoms_per_sec,
        "peak_vram_allocated_gb": alloc_vram,
        "peak_vram_reserved_gb": res_vram,
        "initial_loss": losses[0] if losses else 0.0,
        "final_loss": losses[-1] if losses else 0.0,
        "total_steps": len(step_times_ms)
    }


def main():
    parser = argparse.ArgumentParser(description="MIND Pretraining Optimization Ablation Study")
    parser.add_argument("--config", type=str, default="core/pretraining_config_protein_optimized.yaml", help="Base config YAML")
    parser.add_argument("--samples", type=int, default=400, help="Number of dataset samples to use (default: 400)")
    parser.add_argument("--max-steps", type=int, default=30, help="Benchmark steps per experiment (default: 30)")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps (default: 5)")
    parser.add_argument("--out-dir", type=str, default="benchmarks/ablation_results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load Base Config from YAML
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    config_dict['max_samples'] = args.samples
    config_dict['use_bfloat16'] = True
    config_dict['precision'] = "bfloat16"
    config_dict['num_workers'] = 0

    base_config = create_pretraining_config(**config_dict)

    # Load dataset once for all experiments
    print(f"📦 Loading dataset with {args.samples} samples...")
    dataset = load_universal_dataset(base_config, base_config.dataset, getattr(base_config, 'dataset_download_dir', './data_loading/protein_processed'))
    print(f"✅ Loaded dataset: {len(dataset)} samples")

    # Define the Ablation Matrix
    ablation_experiments = []

    # [1] Baseline: All custom optimizations disabled, PyTorch Native SDPA
    cfg_baseline = copy.deepcopy(base_config)
    cfg_baseline.use_vectorized_geometric_features = False
    cfg_baseline.use_vectorized_mlm_masking = False
    cfg_baseline.use_vectorized_edge_masking = False
    cfg_baseline.use_triton_gather_scatter = False
    cfg_baseline.use_triton_flash_attention = False
    cfg_baseline.use_triton_varlen_flash_attention = False
    cfg_baseline.xformers_or_torch_attn = "torch"
    ablation_experiments.append(("Baseline (SDPA + Standard PyG)", cfg_baseline))

    # [2] + Opt 1: Vectorized Geometric Features & COM
    cfg_opt1 = copy.deepcopy(cfg_baseline)
    cfg_opt1.use_vectorized_geometric_features = True
    ablation_experiments.append(("+ Opt 1: Vectorized Geometric & COM", cfg_opt1))

    # [3] + Opt 2: Vectorized MLM Masking
    cfg_opt2 = copy.deepcopy(cfg_baseline)
    cfg_opt2.use_vectorized_mlm_masking = True
    ablation_experiments.append(("+ Opt 2: Vectorized MLM GPU Masking", cfg_opt2))

    # [4] + Opt 3: Vectorized Edge Masking
    cfg_opt3 = copy.deepcopy(cfg_baseline)
    cfg_opt3.use_vectorized_edge_masking = True
    ablation_experiments.append(("+ Opt 3: Vectorized Edge/Pair Masking", cfg_opt3))

    # [5] + Opt 4: Triton Fused Gather-Scatter
    cfg_opt4 = copy.deepcopy(cfg_baseline)
    cfg_opt4.use_triton_gather_scatter = True
    ablation_experiments.append(("+ Opt 4: Triton Gather-Scatter Kernel", cfg_opt4))

    # [6] + Opt 5 & 8: Triton FlashAttention + Varlen Tile
    cfg_opt5 = copy.deepcopy(cfg_baseline)
    cfg_opt5.use_triton_flash_attention = True
    cfg_opt5.use_triton_varlen_flash_attention = True
    cfg_opt5.xformers_or_torch_attn = "triton"
    ablation_experiments.append(("+ Opt 5&8: Triton FlashAttn + Varlen", cfg_opt5))

    # [7] Full Optimized Suite: ALL Optimizations Enabled
    cfg_full = copy.deepcopy(base_config)
    cfg_full.use_vectorized_geometric_features = True
    cfg_full.use_vectorized_mlm_masking = True
    cfg_full.use_vectorized_edge_masking = True
    cfg_full.use_triton_gather_scatter = True
    cfg_full.use_triton_flash_attention = True
    cfg_full.use_triton_varlen_flash_attention = True
    cfg_full.xformers_or_torch_attn = "triton"
    ablation_experiments.append(("Full Optimization Suite (ALL ON)", cfg_full))

    results = []
    for exp_name, exp_cfg in ablation_experiments:
        try:
            res = run_benchmark_experiment(
                exp_name=exp_name,
                exp_config=exp_cfg,
                dataset=dataset,
                device=device,
                num_warmup_steps=args.warmup_steps,
                max_steps=args.max_steps
            )
            results.append(res)
        except Exception as e:
            print(f"❌ Experiment [{exp_name}] failed with error: {e}")
            import traceback
            traceback.print_exc()

    # Summarize and print table
    if not results:
        print("❌ No results gathered.")
        return

    baseline_res = results[0]
    base_step_ms = baseline_res["avg_step_ms"]
    base_vram = baseline_res["peak_vram_allocated_gb"]

    print("\n" + "="*120)
    print("🏆 FINAL ABLATION STUDY COMPARISON TABLE (BFloat16 AMP)")
    print("="*120)
    header = f"{'Experiment / Optimization':<42} | {'Step (ms)':<10} | {'Fwd (ms)':<9} | {'Bwd (ms)':<9} | {'Throughput':<14} | {'Speedup':<14} | {'Peak VRAM':<10}"
    print(header)
    print("-" * 120)

    md_lines = [
        "# MIND Pretraining Optimization Ablation Study",
        f"**Dataset:** {base_config.dataset} ({args.samples} samples) | **Precision:** BFloat16 AMP | **Hardware:** {torch.cuda.get_device_name(0)}",
        "",
        "| Experiment / Configuration | Step Time (ms) | Forward (ms) | Backward (ms) | Throughput (samples/s) | Speedup vs Base | Peak VRAM (GB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for r in results:
        speedup_pct = ((base_step_ms - r['avg_step_ms']) / base_step_ms * 100) if base_step_ms > 0 else 0.0
        speedup_factor = (base_step_ms / r['avg_step_ms']) if r['avg_step_ms'] > 0 else 1.0
        speedup_str = f"+{speedup_pct:5.1f}% ({speedup_factor:.2f}x)" if r != baseline_res else "Baseline (1.0x)"
        throughput_str = f"{r['samples_per_sec']:.2f} spl/s"
        vram_str = f"{r['peak_vram_allocated_gb']:.2f} GB"

        row = f"{r['name']:<42} | {r['avg_step_ms']:<10.2f} | {r['avg_fwd_ms']:<9.2f} | {r['avg_bwd_ms']:<9.2f} | {throughput_str:<14} | {speedup_str:<14} | {vram_str:<10}"
        print(row)

        md_row = f"| **{r['name']}** | {r['avg_step_ms']:.2f} ms | {r['avg_fwd_ms']:.2f} ms | {r['avg_bwd_ms']:.2f} ms | {r['samples_per_sec']:.2f} spl/s | **{speedup_str}** | {vram_str} |"
        md_lines.append(md_row)

    print("="*120 + "\n")

    # Save to Markdown & JSON
    md_path = os.path.join(args.out_dir, "ablation_results.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"📝 Markdown report saved to: {md_path}")

    json_path = os.path.join(args.out_dir, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 JSON report saved to: {json_path}")


if __name__ == "__main__":
    main()

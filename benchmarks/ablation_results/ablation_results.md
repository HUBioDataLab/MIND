# MIND Pretraining Optimization Ablation Study
**Dataset:** PDB (400 samples) | **Precision:** BFloat16 AMP | **Hardware:** NVIDIA RTX 5000 Ada Generation

| Experiment / Configuration | Step Time (ms) | Forward (ms) | Backward (ms) | Throughput (samples/s) | Speedup vs Base | Peak VRAM (GB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (SDPA + Standard PyG)** | 182.19 ms | 93.71 ms | 88.48 ms | 18.30 spl/s | **Baseline (1.0x)** | 17.80 GB |
| **+ Opt 1: Vectorized Geometric & COM** | 160.35 ms | 80.63 ms | 79.72 ms | 20.22 spl/s | **+ 12.0% (1.14x)** | 15.74 GB |
| **+ Opt 2: Vectorized MLM GPU Masking** | 165.72 ms | 83.22 ms | 82.49 ms | 20.07 spl/s | **+  9.0% (1.10x)** | 17.73 GB |
| **+ Opt 3: Vectorized Edge/Pair Masking** | 155.41 ms | 75.27 ms | 80.14 ms | 21.16 spl/s | **+ 14.7% (1.17x)** | 15.79 GB |
| **+ Opt 4: Triton Gather-Scatter Kernel** | 175.12 ms | 91.66 ms | 83.46 ms | 19.02 spl/s | **+  3.9% (1.04x)** | 15.93 GB |
| **+ Opt 5&8: Triton FlashAttn + Varlen** | 117.52 ms | 58.76 ms | 58.76 ms | 26.44 spl/s | **+ 35.5% (1.55x)** | 3.38 GB |
| **Full Optimization Suite (ALL ON)** | 123.41 ms | 59.46 ms | 63.95 ms | 25.03 spl/s | **+ 32.3% (1.48x)** | 4.43 GB |

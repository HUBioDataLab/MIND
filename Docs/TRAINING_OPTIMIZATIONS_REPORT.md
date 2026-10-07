# MIND GNN Pretraining Optimizasyonu ve Takım Çalışması Raporu

## 📌 Genel Bakış (Executive Summary)

Bu rapor, **MIND (Multimodal Integration for Biomolecular Neural Dynamics)** GNN ön eğitim (pretraining) hattında gerçekleştirilen eğitim optimizasyonlarını, takım üyelerinin yaptığı katkıları, teknik detayları ve deneysel ablasyon (ablation) sonuçlarını içermektedir.

Eğitim hattında yapılan **donanım (bfloat16 AMP)**, **makro altyapı (Dynamic Batching, Multi-GPU DDP)**, **özel GPU Kernel'leri (OpenAI Triton)** ve **C++/Tensor seviyesinde vektörizasyon** optimizasyonları sonucunda sistem performansında devasa bir artış elde edilmiştir.

### 🚀 Ana Başarı Metrikleri (NVIDIA RTX A5000 / A6000 Benchmark)

- **Throughput (İşlem Hızı):** **2.67 it/s ➔ 4.73 it/s (+%76.8 Hızlanma) 🏆**
- **Epoch Süresi:** **3 dakika 38 saniye ➔ 2 dakika 03 saniye (-%43.6 Zaman Tasarrufu)**
- **Zirve VRAM Kullanımı:** **20.58 GB ➔ 12.46 GB (-8.12 GB / -%39.5 Bellek Tasarrufu) 📉**
- **Multi-GPU DDP Desteği:** Çoklu GPU üzerinde DDP (Distributed Data Parallel) ile sorunsuz ölçeklenme sağlandı.

---

## 👥 Takım Üyeleri ve Katkıları (Team Contributions)

| Takım Üyesi | Katkı Alanı | Anahtar Taahhütler (Commits / PRs) | Teknik Detay |
| :--- | :--- | :--- | :--- |
| **Arin (`arindalahmetoglu`)** | Özel GPU Kernels, Vektörizasyon, Optimizer & DDP Altyapısı | `b2afdcb`, `fa5bd83`, `46e7208`, `9b22a16`, `63a02ad`, `2d2f712`, `f3bb836`, `16165e5` | Triton FlashAttention (Block-sparse & Varlen), Fused Triton Gather-Scatter, Vektörize Edge/MLM Masking, Fused AdamW & Custom Grad Clip, Multi-GPU DDP (PR #20 & PR #21) |
| **Yusuf Demir (`yusufdemrr`)** | Metrik Callback Optimizasyonu, Veri İşleme, Protein & RNA MLM/Metrics | `7482ad3`, `983e66b`, `7f8f262`, `fca65a2`, `dbfbc1b`, `ad755f1`, `97ba174`, `65057d9` | Validation tahminlerinin callback'lerde yeniden kullanımı (çift model geçişini önleme), `.pkl`➔`.pt` veri dönüştürme 5-7x hızlandırma, Protein Block-MLM, Per-Atom/Residue Metrikleri, Log & Skill Altyapısı |
| **Arya Zeynep Mete (`aryazeynep`)** | RNA Metrik Altyapısı, Callback Yönetimi | `c0dd7f6`, `734184d`, `af476b0`, `9084fd2`, `416178c` | RNA per-atom metrik callback entegrasyonu, LearningRateMonitor eklenmesi, Veri seti temizliği ve PR birleştirmeleri |
| **Ege Kabapınar (`egekabapinarr`)** | COCONUT Maskeleme & Metabolit Altyapısı | `4612822` | COCONUT veri seti maskeleme parametreleri (`random_walk`), dinamik chunk sampler güncellemeleri ve bond edge indexing entegrasyonu |

---

## 🛠️ Detaylı Optimizasyon İncelemesi (Technical Deep-Dive)

### 1. ⚡ Özel GPU Kernel'leri (OpenAI Triton CUDA Optimization)

#### A. Triton Variable-Length & Block-Sparse FlashAttention (Opt 5 & Opt 8)
- **Konum:** `esa/triton_flash_mha.py`, `esa/mha.py`, `esa/masked_layers.py`
- **Geliştirici:** Arin (`arindalahmetoglu`)
- **Önceki Durum:** PyTorch varsayılan Multi-Head Attention mekanizması, VRAM üzerinde $(B, H, N, N)$ boyutunda devasa matrisler allocate ediyor ve Softmax işlemlerini VRAM üzerinden yürütüyordu. Ayrıca padding içeren token'lar gereksiz yere GPU döngülerinde işleniyordu.
- **Yapılan Optimizasyon:**
  1. **Block-Sparse SRAM Fusion (Opt 5):** $Q \cdot K^T$ matris çarpımı, 3D blok-seyrek grafik adjacency maskesi, Online Softmax ve $Softmax \cdot V$ birleştirmesi tamamen GPU SRAM register'ları içerisinde birleştirildi (fused). VRAM'e $(B, H, N, N)$ matris yazma ihtiyacı kaldırıldı.
  2. **Variable-Length Padding-Free Masking (Opt 8):** Batch içerisindeki gerçek atom sayıları (`seq_lens`) `bincount` ile hesaplanarak SRAM tile döngüleri grafik bazında sınırlandırıldı. Zero-padding jetonları tamamen atlandı.
- **Sonuç:**
  - Attention Kernel Hızlanması: **1.03 ms ➔ 0.22 ms (4.69x Kernel Hızlanması 🚀)**
  - Uçtan Uca Model Hızlanması: **+%30.2 (Opt 5)** ve ilave **+%15.0 (Opt 8)**
  - VRAM Tasarrufu: **-618 MB VRAM** düşüşü.

#### B. Triton Fused Gather-Scatter Graph Edge Kernel (Opt 4)
- **Konum:** `esa/triton_graph_gather_scatter.py`, `core/molecular_encoder.py`
- **Geliştirici:** Arin (`arindalahmetoglu`)
- **Önceki Durum:** Standard PyTorch Geometric `scatter_add` kernel'ı VRAM üzerinde geçici edge feature matrisleri oluşturuyor ve 4x VRAM bellek turu (Read node, Write edge, Read edge, Write node) yapıyordu.
- **Yapılan Optimizasyon:** Custom OpenAI Triton fused kernel (`_fused_gather_scatter_kernel`) yazıldı. Düğüm (node) özellikleri doğrudan SRAM register'larında `tl.atomic_add` ile biriktirilerek VRAM bellek turları 4x'ten 1x'e indirildi.
- **Sonuç:** Bellek darboğazı engellendi ve tepe (peak) VRAM bellek kullanımı önemli ölçüde düşürüldü.

---

### 2. 🔄 Tensor Vektörizasyonu ve Python Döngülerinin Temizlenmesi

#### A. Vectorized Edge Masking & Pair Sampling (Opt 3)
- **Konum:** `core/pretraining_tasks.py` (`long_range_distance_loss`, `short_range_distance_loss`)
- **Geliştirici:** Arin (`arindalahmetoglu`)
- **Önceki Durum:**
  1. `long_range_distance_loss`: Her adımda graf bazında Python `while len(pairs) < 1000` döngüsü çalıştırılıyor, binlerce CPU `.tolist()` çağrısı ve CPU set işlemleri yapılıyordu (CPU-GPU Senkronizasyon Darboğazı).
  2. `short_range_distance_loss`: Geçersiz kenarlar filtrelenmeden ÖNCE 100.000+ düğüm embedding'i VRAM'de toplanıyordu.
- **Yapılan Optimizasyon:**
  1. Long-range çift örneklemesi tamamen GPU üzerinde sıfır Python döngüsü ile `torch.randint(0, num_atoms, (2000,), device=gpu)` olarak vektörize edildi.
  2. Short-range kenar indeksleri VRAM'de embedding toplamadan ÖNCE filtrelendi (`edge_index[:, valid_mask]`).
- **Sonuç:** **2.68 it/s ➔ 3.24 it/s (+%21.2 SPEEDUP! 🚀)** — En büyük tekil mikro-kod hızlanması!

#### B. Vectorized MLM Masking (Opt 2)
- **Konum:** `core/pretraining_model.py` (`_apply_dynamic_mlm_masking`)
- **Geliştirici:** Arin (`arindalahmetoglu`)
- **Önceki Durum:** Python ile graf graf döngü (`for g in graph_idx.unique()`) kurulup her graf için ayrı `torch.randperm` çağrılıyordu.
- **Yapılan Optimizasyon:** Tüm batch (~50.000+ atom) boyunca tek hamlede GPU üzerinde vektörize rasgele maskeleme (`torch.rand(num_nodes) < ratio`) uygulandı.
- **Sonuç:** **2.63 it/s ➔ 2.70 it/s (+%2.7 Hız artışı)** ve GPU utilization dalgalanmaları engellendi.

#### C. Vectorized Geometric Feature Extraction (Opt 1) & COM Centering
- **Konum:** `core/molecular_encoder.py`, `core/pretraining_model.py`
- **Geliştirici:** Arin (`arindalahmetoglu`)
- **Yapılan Optimizasyon:** Python döngüleri ve `aten::nonzero` çağrıları yerini C++ PyTorch Geometric kernel'lerine (`to_dense_batch`, `scatter_min`, `scatter_add`) ve `scatter_mean` ile kütle merkezi (COM) merkezlemesine bıraktı.

---

### 3. ⚙️ Optimizer & Gradient Clipping ve DDP Altyapısı

#### A. Fused AdamW & Custom Gradient Clipping Override
- **Konum:** `core/pretraining_model.py`
- **Geliştirici:** Arin (`arindalahmetoglu`)
- **Yapılan Optimizasyon:** PyTorch Lightning'in varsayılan AMP (Automatic Mixed Precision) ve optimizer sarmalayıcısı ile `torch.optim.AdamW(fused=True)` arasındaki unscaling çakışmasını gidermek amacıyla özel `configure_gradient_clipping` override'ı yazıldı.
- **Sonuç:** CUDA Fused AdamW optimizer'ı tam performans ile PyTorch Lightning bfloat16 AMP altında stabil hale getirildi.

#### B. Multi-GPU Distributed Data Parallel (DDP) Desteği (PR #20)
- **Konum:** `core/train_pretrain.py`, `core/pretraining_model.py`
- **Geliştirici:** Arin (`arindalahmetoglu`)
- **Yapılan Optimizasyon:** Çoklu GPU üzerinde DDP eğitim desteği eklendi.

---

### 4. 📊 Validation & Metric Callback Optimizasyonları

#### Validation Predictions Reuse (Çift Evaluation Engelleme)
- **Konum:** `core/train_pretrain.py`, `core/metrics/`
- **Geliştirici:** Yusuf Demir (`yusufdemrr`)
- **Önceki Durum:** Validation adımında model tahminleri hem Lightning adımlarında hem de Metric Callback'lerde (Protein/RNA per-atom metrics) bağımsız olarak iki kez hesaplanıyordu.
- **Yapılan Optimizasyon:** Forward geçişi sonucundaki logits ve tahminler callback yapılarına aktarılarak validation süresindeki **çift forward geçişi maliyeti ortadan kaldırıldı**.

---

### 5. 🚀 Veri Dönüştürme ve Ön İşleme Optimizasyonu

#### Faster `.pkl` ➔ `.pt` Cache Conversion (5-7x Speedup)
- **Konum:** `data_loading/cache_to_pyg.py`
- **Geliştirici:** Yusuf Demir (`yusufdemrr`)
- **Yapılan Optimizasyon:** Ham veri setlerinin PyTorch Geometric `.pt` formatına dönüştürülmesinde kenar inşası (edge construction) vektörize edildi ve molekül işleme adımları çoklu çekirdeğe (multiprocessing) parallelize edildi.
- **Sonuç:** Veri dönüştürme süresi **5-7 kat hızlandı**.

---

## 📊 Ablasyon Çalışması ve Performans Özeti (Ablation Benchmark Table)

Aşağıdaki tablo, **NVIDIA RTX A5000 GPU** üzerinde gerçekleştirilen adım adım ablasyon deneylerini özetlemektedir:

| Opt ID | Optimizasyon Adı / Bayrağı | Baseline (Kapalı) | Optimized (Açık) | Etki / Performans Kazanımı | Durum |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Makro Donanım** | `use_bfloat16` | FP32 | BFloat16-AMP | **~2.1x GPU FLOPS Hızlanması** | ✅ Aktif |
| **Makro Batching** | `use_dynamic_batching` | Static Batch 4 | 12k Atoms/Batch | **-%25 Adım Sayısı (Padding Önleme)** | ✅ Aktif |
| **Opt 1** | `use_vectorized_geometric_features` | Python Loops | Vectorized C++ | Stable & Zero Sync | ✅ Aktif |
| **Opt 2** | `use_vectorized_mlm_masking` | Graph-by-graph loop | Batch GPU `rand` | **+%2.7 Hızlanma (2.63 ➔ 2.70 it/s)** | ✅ Aktif |
| **Opt 3** | `use_vectorized_edge_masking` | Python `while` loop | GPU Vectorized Tensor | **+%21.2 Hızlanma (2.68 ➔ 3.24 it/s) 🚀** | ✅ Aktif |
| **Opt 4** | `use_triton_gather_scatter` | PyG `scatter_add` | Fused Triton Kernel | **Zirve VRAM Bellek Tasarrufu** | ✅ Aktif |
| **Opt 5** | `use_triton_flash_attention` | Standard PyTorch MHA | 3D Triton FlashAttn | **+%30.2 Hızlanma & -618MB VRAM 🚀** | ✅ Aktif |
| **Opt 6** | `use_bfloat16_loss_head` | Lightning AMP | Explicit Autocast | **+%0.8 Hızlanma (Tensor Core Uyum)** | ✅ Aktif |
| **Opt 8** | `use_triton_varlen_flash_attention` | Padded Attention | Fused Varlen Triton | **4.69x Attention & +%15.0 Model Hızlanması 🏆** | ✅ Aktif |
| **FULL SUITE** | **Tüm Aktif Optimizasyonlar (1+2+3+4+5+6+8)** | Baseline | FULL OPTIMIZED | **+%76.8 HIZLANMA (2.67 ➔ 4.73 it/s) \| -8.12 GB VRAM (-%39.5) 🏆🔥** | ✅ **ÜRETİMDE** |

---

## ❌ Başarısız / Reddedilen Optimizasyon Çalışması (Negative Results)

### ❌ Opt 7: PyTorch `torch.compile(mode="reduce-overhead")` JIT Derlemesi

- **Bayrak:** `use_torch_compile`
- **Konum:** `core/pretraining_model.py`
- **Gözlenen Etki:** **Ciddi Yavaşlama (-%40.5 Performans Kaybı)**
  - *Opt 1-6 (Tüm Opt'lar Açık, compile Kapalı):* **4.72 it/s** (~211 ms/step)
  - *Opt 1-7 (compile Açık):* **2.81 it/s** (~355 ms/step)
- **Karar:** ❌ **REDDEDİLDİ / DEVRE DIŞI BIRAKILDI (`use_torch_compile: false`)**

#### 🔍 Kök Neden Analizi (Neden Yavaşladı?):
1. **Dinamik Graf Yapısı (Dynamic Shape Recapture Overhead):** Protein ve moleküler veri setlerinde atom ve kenar sayıları her adımda değişmektedir (`ImprovedDynamicBatchSampler`). `torch.compile` her yeni batch boyutunda yeni bir CUDA Graph kaydetmeye çalışmış (loglarda *51 distinct sizes* uyarısı), bu da sürekli CPU-GPU pipeline stall'larına ve yeniden derleme yüküne neden olmuştur.
2. **Küçük Modül Sarmalama Yükü:** Derlenen `UniversalMolecularEncoder` modülü sadece ~684K parametreli küçük bir katmandır. JIT derlemesinin getirdiği Dynamo tracing guard yükü, sağladığı mikro-saniye kazançtan kat kat fazla kalmıştır.
3. **Özel Triton Kernelleri ile Çakışma:** Opt 4 ve Opt 5 zaten Triton seviyesinde yazılmış GPU kernel'leridir. `torch.compile` bu kernel'leri sarmalarken ekstra kontrol katmanları ekleyerek gecikmeye yol açmıştır.

---

## 🏁 Sonuç

Yapılan kolektif çalışma ile MIND GNN modelinin eğitim hattı **dönüştürülmüş ve optimize edilmiştir**:
1. Eğitim süresi neredeyse **yarı yarıya indirilmiş** (4.73 it/s),
2. VRAM bellek kullanımı **8.12 GB azaltılarak** daha büyük batch boyutlarına olanak tanınmış,
3. Multi-GPU DDP altyapısı ve gelişmiş per-atom/residue metrik takibi entegre edilmiştir.

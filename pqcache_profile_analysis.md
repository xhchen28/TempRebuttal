# PQCache nsys Profiling Analysis

**Model**: LLaMA 3.1 8B Instruct (fp16)
**Data**: LongBench-v2 easy+long, 1 sample truncated to 128K tokens
**GPU**: A100 (80GB)
**Config**: `compress_ratio=0.2`, `sink=32`, `local=256`, `topk_ratio=10%`, `n_subvec=2`, `n_subbits=6`, `global_cache_size=4096`, `cache_block_size=128`

---

## 1. Decode Latency Summary

| Seq Len | TTFT (s) | TT2T (s) | Decode 28 tokens (s) | Per Token (ms) | Overhead vs 16K |
|---------|----------|----------|----------------------|----------------|-----------------|
| 16K     | 1.73     | 1.77     | 1.20                 | 42.9           | baseline        |
| 32K     | 4.21     | 4.29     | 1.98                 | 70.9           | +65%            |
| 64K     | 11.61    | 11.76    | 3.85                 | 137.5          | +220%           |
| 96K     | 22.32    | 22.44    | 5.65                 | 201.8          | +370%           |
| 128K    | 36.42    | 36.59    | 7.35                 | 262.5          | +512%           |

Per-token decode latency scales roughly **linearly** with context length: ~2ms per additional 1K tokens.

---

## 2. CUDA API Breakdown

| API                        | Time (s)  | %       | Calls  | Avg (ms)  | Median (μs) |
|----------------------------|-----------|---------|--------|-----------|--------------|
| **cudaStreamSynchronize**  | **160.2** | **65.2%** | 585    | 273.8     | 7.9          |
| cudaLaunchKernel           | 82.4      | 33.5%   | 245,529 | 0.34      | 4.6          |
| cudaMemsetAsync            | 3.2       | 1.3%    | 2,496  | 1.28      | 8.9          |
| cudaMemcpyAsync            | 0.02      | 0.0%    | 960    | 0.024     | 10.2         |

**GPU idle 65% of the time**, blocked on `cudaStreamSynchronize` — waiting for CPU-side work.

---

## 3. GPU Kernel Time (Top Kernels)

| Kernel | Time (s) | % | Instances | Avg (ms) | Notes |
|--------|----------|---|-----------|----------|-------|
| FlashAttention (prefill, `flash_fwd_kernel`) | 142.8 | 57.1% | 480 | 297.5 | Prefill-dominated |
| GEMM (`s16816gemm 128x128`) | 49.1 | 19.6% | 1,344 | 36.5 | Q/K/V/O projections + MLP |
| elementwise (mul/add etc.) | 13.0 | 5.2% | 10,560 | 1.23 | PQ scoring ops |
| GEMM + ReLU (cutlass) | 10.7 | 4.3% | 1,152 | 9.3 | MLP layers |
| GEMM (`s16816gemm 256x128`) | 6.3 | 2.5% | 768 | 8.3 | MLP layers |
| CatArrayBatchedCopy | 4.3 | 1.7% | 9,600 | 0.45 | KV buffer assembly |
| **FlashAttention splitkv (decode)** | **3.4** | **1.3%** | **4,800** | **0.70** | **Actual decode attention** |
| GEMM (`128x64`, decode proj) | 0.78 | 0.3% | 9,600 | 0.08 | Decode Q/K/V/O projections |

Decode-specific GPU work (splitkv attention + decode GEMM): **~4.2s out of ~250s total** — GPU is barely doing useful decode work.

---

## 4. GPU Memory Operations

| Direction | Time (ms) | Count | Avg (μs) |
|-----------|-----------|-------|----------|
| CUDA memset | 7.0 | 2,496 | 2.8 |
| Device-to-Host | 0.87 | 540 | 1.6 |
| Device-to-Device | 0.81 | 375 | 2.2 |
| **Host-to-Device** | **0.05** | **45** | **1.1** |

Total GPU-visible data transfer: **< 2ms** for the entire profiled iteration. PCIe bandwidth is NOT the bottleneck.

---

## 5. OS Runtime Summary

| Syscall | Time (s) | % | Calls | Notes |
|---------|----------|---|-------|-------|
| `sem_timedwait` | 820.0 | 58.1% | 82 | K-Means worker process synchronization |
| `poll` | 561.7 | 39.8% | 5,609 | Multiprocessing queue polling |
| `select` | 30.0 | 2.1% | 10 | `time.sleep(3)` between measurements |

---

## 6. Root Cause Analysis

### The Decode Pipeline Bottleneck

For each decode token, per layer, PQCache follows this critical path:

```
GPU top-k → indices.cpu() [SYNC!] → CPU cache lookup → CPU gather → pin→GPU → GPU scatter → FlashAttention
            ↑                       ↑                   ↑
            Pipeline drain          Python LFU loop     CPU memory-bound
            (~0.1-0.3ms)           (~0.1-0.3ms)        (~0.2-0.5ms)
```

### Three Bottleneck Categories

**1. GPU Pipeline Stalls (dominant — 65% of CUDA API time)**

`fetch_and_concat_kv_w_cache` (cache_manager.py:310-439) contains synchronous `.cpu()` calls:

```python
to_fetch_idx = (to_fetch_idx[0].cpu(), to_fetch_idx[1].cpu())  # ← implicit cudaDeviceSynchronize
miss_cnt = miss_cnt.cpu()
hit_cnt = hit_cnt.cpu()
block2token_times, qualified_block_idx = qualified_block_result.values.cpu().tolist(), qualified_block_result.indices.cpu()
```

Each `.cpu()` forces a full GPU pipeline drain — 32 layers × N tokens × 5 seqlens = thousands of sync points.

**2. CPU-side Gather (invisible to GPU profiler)**

```python
self.fetch_k_pin_buffer[layer_idx][:fetched_token_cnt,:] = self.cpu_key_buffers[layer_idx][(0, to_fetch_idx[1], to_fetch_idx[0])]
self.fetch_v_pin_buffer[layer_idx][:fetched_token_cnt,:] = self.cpu_value_buffer[layer_idx][(0, to_fetch_idx[1], to_fetch_idx[0])]
```

Advanced indexing on large CPU pinned tensors (128K × 8 heads × 128 dim) — memory-bandwidth limited on CPU, completely invisible in GPU profiler.

**3. Python-level LFU Cache Update**

```python
for i in range(len(selected_block_indices)):
    if old_gpu_pos == -1 and new_gpu_pos >= 0:
        self.global_key_cache[...].copy_(self.cpu_key_buffers[...], non_blocking=True)
        self.global_value_cache[...].copy_(self.cpu_value_buffer[...], non_blocking=True)
```

Python for-loop with per-block conditional H2D copies — serial overhead that doesn't vectorize.

### Why H2D Memcpy is So Small

Only **45 H2D operations** in the entire run despite 4,800 decode layer passes. This is because:
- The `global_cache_size=4096` (32 blocks) caches frequently-accessed blocks
- LFU promotion only triggers on cache misses for "qualified" blocks
- The actual token fetch uses `.to(device, non_blocking=True)` on pre-staged pinned buffers, which may coalesce into fewer transfers

The data volume per transfer is small; the cost is the **pipeline stall to determine what to transfer**.

---

## 7. Comparison Context

| Method | Per-Token @ 128K | Primary Bottleneck |
|--------|------------------|--------------------|
| **PQCache** | **262.5 ms** | CPU-GPU sync barriers + CPU gather |
| ParisKV (Polar ANN) | TBD (nsys done) | GPU-side ANN search |
| MagicPIG | TBD | TBD |

PQCache's architectural limitation: the retrieval decision (top-k on GPU) must cross PCIe to CPU for the actual KV gather, creating unavoidable serial sync points. ParisKV avoids this by keeping both the index and the retrieval path on GPU.

---

## 8. Profile Artifacts

- nsys report: `pqcache_profile.nsys-rep`
- SQLite export: `pqcache_profile.sqlite`
- Profiled iteration: iter 2 (after 2 warmup iterations)
- Capture: `cudaProfilerApi` gated, covering all 5 sequence lengths × (TTFT + TT2T + 30-token decode)

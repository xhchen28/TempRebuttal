# Latency Breakdown and Kernel Analysis

## Reviewer Concern

> The source of the latency gap between ParisKV and MagicPIG / PQCache is not fully explained. Is it due to needing to run the retrieval lookup CPU-side? Are some of the speedups due to other system/kernel improvements? It would be useful to get a breakdown of the runtime for this, given that the overhead is so large.

We thank the reviewer for this insightful question.  
We agree that the source of the latency gap should be clearly explained.

Our analysis shows that the performance gap mainly comes from two factors:

1. **ParisKV avoids CPU-side retrieval and synchronization**, while MagicPIG and PQCache both incur substantial CPU-side overhead.
2. **ParisKV also benefits from highly optimized GPU kernels** across all retrieval stages, so the gains are not only architectural, but also implementation-level.

---

## 1. End-to-end runtime breakdown

Table 1 reports the per-layer decode latency breakdown at **BS=1, 128K context**.

### Table 1. Per-Layer Decode Latency Breakdown (BS=1, 128K context, per layer)

| Method   | Stage                         | Time (ms) | Total (ms) | Reason |
|----------|-------------------------------|----------:|-----------:|--------|
| ParisKV  | kv_update                     | 0.022 | 0.734 | Append-only: 1-byte codebook entry + 4-byte 4-bit packed per token, no index rebuild |
|          | retrieval_algo (GPU)          | 0.549 |  | End-to-end GPU pipeline: bitset collision scan O(n) → bucket top-k O(n) → 4-bit rerank (64 B/token memory IO) |
|          | H2D (UVA sparse gather)       | 0.097 |  | UVA-based sparse gather of only final top-k value vectors; no bulk CPU→GPU transfer |
|          | Attention                     | 0.066 |  | FlashAttention over final top-k selected tokens only |
| MagicPIG | query_hash                    | 0.378 | 6.491 | GPU-side LSH hash computation |
|          | kv_update                     | 0.062 |  | GPU kernel not pipelined with subsequent stages; idle gap |
|          | gpu_attention (sink+local)    | 0.058 |  | Small attention over sink and local window tokens |
|          | cpu_retrieval & cpu_attention | 5.881 |  | **Bottleneck:** CPU-side hash table traversal + attention; requires maintaining dynamic per-bucket inverted lists on CPU |
|          | attention_merge               | 0.112 |  | Merging GPU (sink+local) and CPU (retrieved) attention outputs |
| PQCache  | codebook_transfer & lookup top-k | 0.687 | 9.555 | PQ codebook CPU→GPU transfer + approximate distance computation |
|          | kv_fetch_cpu2gpu             | 7.770 |  | **Bottleneck:** bulk transfer of full-precision KV vectors from CPU to GPU for selected candidates, bounded by PCIe bandwidth |
|          | attention                     | 1.050 |  | Full attention over fetched candidate tokens |
|          | eviction_pq_predict           | 0.048 |  | PQ-based prediction for eviction policy |


| Aspect | ParisKV | MagicPIG | PQCache |
|--------|--------|----------|---------|
| **Retrieval location** | Fully on GPU | CPU + GPU | CPU + GPU |
| **CPU involvement** | None (only storage) | CPU hash traversal + attention | CPU lookup + transfer |
| **Dominant bottleneck** | GPU retrieval (~0.55 ms) | CPU retrieval (5.88 / 6.49 ms) | CPU→GPU transfer (7.77 / 9.55 ms) |
| **Data movement** | Sparse UVA fetch (top-$k$ only) | CPU-side KV access + sync | Bulk KV transfer (PCIe-bound) |
| **Index structure** | Bitset + scan (O($n$)) | LSH + inverted lists | IVF-style clustering |
| **GPU efficiency** | Fully pipelined | Partial (CPU–GPU sync gaps) | Partial (transfer-bound) |


### Main observations

- **MagicPIG** is dominated by **CPU retrieval + CPU sparse attention**:  
  `5.881 ms / 6.491 ms`
- **PQCache** is dominated by **CPU→GPU KV transfer**:  
  `7.770 ms / 9.555 ms`
- **ParisKV** keeps the retrieval path on GPU and only performs a small **UVA sparse gather** for the final top-k KV:
  - retrieval on GPU: `0.549 ms`
  - sparse KV fetch: `0.097 ms`
  - attention: `0.066 ms`

Thus, the latency gap is not explained by a single kernel alone. It is mainly due to:

- removing CPU-side retrieval / synchronization,
- avoiding bulk CPU→GPU transfer,
- and using a GPU-native retrieval pipeline.

---

## 2. Why ParisKV is faster

### 2.1 Avoiding CPU-side retrieval and synchronization

A key difference is that **ParisKV keeps the entire retrieval pipeline on GPU**, while both MagicPIG and PQCache involve substantial CPU-side operations.

- **MagicPIG** performs hash-table traversal and sparse attention on CPU, which becomes the dominant bottleneck.
- **PQCache** performs CPU-side KV lookup and bulk CPU→GPU transfer, which is inherently memory-bandwidth bound and incurs synchronization overhead.

In contrast, ParisKV:

- avoids CPU-side retrieval entirely,
- uses CPU memory (DRAM) only as an extended storage pool,
- and performs **GPU-side sparse gathering via UVA**, transferring only the final top-k KV vectors (e.g., `k=100`).

This significantly reduces both **data movement** and **CPU–GPU synchronization overhead**.

---

### 2.2 GPU-native retrieval design

ParisKV uses a **collision-based coarse candidate selection algorithm** that is more GPU-friendly than conventional PQ- or LSH-style retrieval pipelines under the same recall target.

Importantly, ParisKV does **not** require explicit CPU-side traversal of dynamic cluster-to-token mappings during decoding. Instead:

- each token stores a **1-byte cluster code**,
- query-selected clusters are compressed into a **bitset**  
  (e.g., `256 clusters → 32 bytes`),
- candidate selection is implemented as a **single linear scan with bitwise tests**.

This avoids the expensive maintenance and traversal of dynamic inverted lists on CPU and enables a fully GPU-resident retrieval pipeline.

---

### 2.3 Kernel-level optimizations also matter

Beyond the system-level design, a substantial portion of the speedup also comes from custom CUDA kernels.  
Tables 2–5 report the per-operator results.

These results show that the performance gains are **not solely due to architectural differences**, but also come from **highly optimized GPU kernels across all stages**.

---

## 3. Operator-level analysis

### Table 2. Collision kernel

| KV Len | Torch (ms) | Ours Kernel (ms) | Speedup |
|-------:|-----------:|-----------------:|--------:|
| 8K     | 27.02  | 0.0276 | 977×  |
| 16K    | 53.85  | 0.0265 | 2032× |
| 32K    | 107.61 | 0.0298 | 3606× |
| 64K    | 215.09 | 0.0382 | 5632× |
| 128K   | 429.92 | 0.0576 | 7460× |
| 256K   | 859.59 | 0.0934 | 9202× |

**Interpretation.**  
The collision stage is essentially removed as a bottleneck: even at 256K, it remains below `0.1 ms`.

---

### Table 3. Bucket Top-k

| KV Len | Torch (ms) | Bucket Top-k (ms) | Speedup |
|-------:|-----------:|------------------:|--------:|
| 8K     | 0.238 | 0.029 | 8.21× |
| 16K    | 0.310 | 0.033 | 9.39× |
| 32K    | 0.211 | 0.039 | 5.41× |
| 64K    | 0.221 | 0.060 | 3.68× |
| 128K   | 0.240 | 0.100 | 2.40× |
| 256K   | 0.251 | 0.157 | 1.60× |
| 512K   | 0.301 | 0.327 | 0.92× |

**Interpretation.**  
Bucket Top-k replaces sorting-based Top-k with a histogram/scan design, which is more suitable for the small integer range produced by collision counting.

---

### Table 4. Reranking (Fused)

| KV Len | Candidates | Ref (ms) | Fused (ms) | Speedup |
|-------:|-----------:|---------:|-----------:|--------:|
| 8K     | 1,639  | 0.689 | 0.187 | 3.68× |
| 16K    | 3,277  | 0.717 | 0.231 | 3.10× |
| 32K    | 6,554  | 1.118 | 0.299 | 3.74× |
| 64K    | 13,108 | 1.575 | 0.428 | 3.68× |
| 128K   | 26,215 | 2.478 | 0.573 | 4.32× |
| 256K   | 52,429 | 4.380 | 1.095 | 4.00× |

**Interpretation.**  
The fused reranking kernel operates on **4-bit packed representations** (`64 B/token` vs `256 B/token` for bf16), substantially reducing memory bandwidth.

---

### Table 5. KV Fetch (UVA)

| KV Len | Candidates | Ref (ms) | Fused (ms) | Speedup |
|-------:|-----------:|---------:|-----------:|--------:|
| 8K     | 1,639  | 0.689 | 0.187 | 3.68× |
| 16K    | 3,277  | 0.717 | 0.231 | 3.10× |
| 32K    | 6,554  | 1.118 | 0.299 | 3.74× |
| 64K    | 13,108 | 1.575 | 0.428 | 3.68× |
| 128K   | 26,215 | 2.478 | 0.573 | 4.32× |
| 256K   | 52,429 | 4.380 | 1.095 | 4.00× |

**Interpretation.**  
UVA-based sparse gather avoids the standard bulk CPU→GPU transfer path and keeps fetch cost small even at long context.

---

## 4. Summary

The latency gap between ParisKV and MagicPIG / PQCache is primarily explained by:

1. **No CPU-side retrieval or sparse attention**
   - MagicPIG is bottlenecked by CPU-side hash traversal and sparse attention.
   - PQCache is bottlenecked by CPU KV lookup and bulk transfer.

2. **No bulk CPU→GPU movement of the full candidate set**
   - ParisKV only fetches the final top-k KV vectors via UVA.

3. **GPU-native retrieval path**
   - bitset-based candidate selection,
   - GPU-side coarse retrieval,
   - fused 4-bit reranking.

4. **Optimized CUDA kernels across all stages**
   - collision: up to **9000×**
   - bucket top-k: up to **9.39×**
   - fused rerank: about **3–4×**
   - UVA fetch: about **3–4×**

Overall, the performance gains come from **both**:
- a fundamentally more GPU-friendly retrieval design, and
- highly optimized GPU kernels across all stages.

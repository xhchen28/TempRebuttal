# Latency Breakdown and Kernel Performance


## 1. End-to-end runtime breakdown


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



## 2. Operator-level analysis

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


# Expanded Baseline Comparison: Accuracy, Efficiency, and Configurations

## 1. Scope and fairness protocol

**Reviewer concern.**  
The original submission did not include several recent long-context KV-cache baselines on the same platform, such as RetroInfer, ShadowKV, RetrievalAttention, and SOCKET.

### Baselines included here

We expand the comparison to include:
- RetroInfer
- ShadowKV
- SOCKET
- Quest
- Twilight
- FreeKV
- MagicPIG
- PQCache

**RetrievalAttention** is not included because it is not publicly available.

### Common evaluation protocol

We evaluate all methods on the same platform whenever possible and align retrieval budgets approximately:

- **LongBench-v2:** sampling decoding, temperature = 0.1
- **RULER:** greedy decoding
- **LongBench-v2 / GPQA target budget:** ~420 tokens
- **RULER target budget:** ~2048 tokens

Model details:
- Qwen3-8B has `max_model_len=40960`
- We evaluate up to 128K by:
  - **LongBench-v2:** without YaRN
  - **RULER:** with YaRN enabled

The goal is not to force identical hyperparameters across methods, but to match their **effective retrieval budget** as closely as possible.

---

## 2. Baseline configurations

### SOCKET
- `bucket_K = 8`
- `bucket_L = 60`
- `sink = 64`
- `local = 256`
- `topk = 100` for LongBench-v2
- `topk = 2048` for RULER

### Twilight
Twilight uses adaptive top-k via top-p. We choose a setting whose **mean retrieval budget** roughly matches ParisKV:
- `top-p = 0.3` → meanK = 353
- `top-p = 0.5` → meanK = 402.4  ✅
- `top-p = 0.8` → meanK = 671.17

ParisKV uses `sink=64 + local=256 + topk=100 = 420`, so we use **top-p=0.5** as the fairest comparison.

### RetroInfer
**LongBench-v2**
- `sink = 64`
- `local = 256`
- `retrieval ratio = 0.001`
- `estimation zone = 0.001`
- `pages_per_cluster = 2`
- `buffer_cluster_num = 200`
- `cluster_num = 8192`
- effective cluster size ≈ 16, capped at 750 if larger

**RULER**
- `sink = 64`
- `local = 256`
- `retrieval ratio = 0.013`
- `estimation cluster ratio = 0.001`
- cluster size = 16, total clusters = 8192

**GPQA-diamond**
- `retrieval ratio = 0.01`
- `estimation zone = 0.001`
- `retrieval_topk = 7`
- fixed `nprobe = 7`

### ShadowKV
**LongBench-v2**
- `local = 32`
- `outlier = 384`
- `sparse = 104`

**RULER**
- `local = 32`
- `outlier = 304`
- `sparse = 2048`

### FreeKV
- `sink = 64`
- `local = 256`
- `page = 128`
- total budget ≈ 448

### Quest
No sink/local split:
- LongBench-v2 / GPQA:
  - `token_budget = 420`
  - `chunk_size = 16`
- RULER:
  - `token_budget = 2048`

### MagicPIG
- LSH parameters:
  - `K = 10`
  - `L = 170`
- retrieval budget is dynamic and depends on KV length

### PQCache
- `compress ratio = 20%`
- `subspace = 2`
- `cluster_num = 64`

---

## 3. Accuracy results

### LongBench-v2 (overall score)

| Model | Full | PQCache | MagicPIG | ShadowKV | FreeKV | Quest | RetroInfer | SOCKET | Twilight | ParisKV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 25.84 | 17.91 | 16.70 | 16.30 | 19.68 | 19.12 | 23.69 | 17.93 | 19.12 | **24.60** |
| Qwen3-8B | 33.59 | 25.50 | 10.34 | 15.90 | 15.31 | 23.90 | 20.48 | 17.93 | 20.32 | **33.07** |
| DS-R1-8B | 13.12 | 19.90 | 13.92 | 14.51 | 17.50 | 23.51 | 15.66 | 21.51 | 20.72 | **28.43** |

### RULER (full breakdown)

| Method | niah_s1 | niah_s2 | niah_m1 | niah_m2 | niah_mv | niah_mq | fwe | qa_1 | qa_2 | vt | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama3.1-8B | 100.00 | 100.00 | 96.88 | 89.58 | 98.44 | 99.74 | 71.18 | 86.46 | 51.04 | 47.92 | 84.12 |
| ParisKV | 100.00 | 100.00 | 96.88 | 82.01 | 95.83 | 98.70 | 56.25 | 84.38 | 48.96 | 71.87 | **83.49** |
| MagicPIG | 100.00 | 94.79 | 83.33 | 34.38 | 73.44 | 78.12 | 70.08 | 73.96 | 42.71 | 53.33 | 70.49 |
| ShadowKV | 100.00 | 98.96 | 96.88 | 73.96 | 89.06 | 96.35 | 61.11 | 81.25 | 50.00 | 55.42 | 80.29 |
| PQCache | 5.21 | 50.00 | 45.83 | 30.21 | 16.67 | 20.31 | 55.21 | 68.75 | 40.62 | 4.79 | 33.76 |
| SOCKET | 100.00 | 100.00 | 96.88 | 72.92 | 92.71 | 98.18 | 33.33 | 80.21 | 46.88 | 68.13 | 78.92 |
| Twilight | 98.96 | 100.00 | 95.83 | 76.04 | 90.36 | 97.92 | 57.29 | 80.21 | 48.96 | 74.17 | 77.60 |
| RetroInfer | 95.83 | 98.96 | 95.83 | 55.21 | 94.53 | 97.40 | 61.11 | 78.12 | 42.71 | 54.79 | 77.45 |
| Quest | 99.00 | 98.96 | 94.79 | 64.58 | 83.59 | 94.79 | 62.85 | 79.17 | 44.79 | 70.83 | 79.34 |

### GPQA-diamond

| Method | Score |
|---|---:|
| Qwen3-4B | 64.14 |
| Quest | 38.40 |
| RetroInfer | 38.90 |
| FreeKV | 58.16 |
| MagicPig++ | 32.32 |
| PQCache | 38.38 |
| ParisKV | **72.22** |

---

## 4. Efficiency results

Notation:
- **NA**: unsupported by the released implementation
- **OOM**: out of memory

### Throughput at 128K decode (tokens/s)

| bs | Quest | Twilight | RetroInfer | ParisKV | Full | SOCKET | MagicPIG | PQCache | FreeKV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48.08 | 0.90 | 34.31 | 41.10 | 43.20 | 4.44 | 120.57 | 243.91 | 21.78 |
| 2 | NA | OOM | 65.29 | 69.80 | 60.70 | OOM | NA | NA | 29.01 |
| 3 | NA | OOM | 93.20 | 93.10 | 70.00 | OOM | NA | NA | 34.67 |
| 8 | NA | OOM | 109.60 | **150.00** | OOM | OOM | NA | NA | OOM |

### Decode latency (ms/token, bs=1)

| Seq len | Quest | Twilight | RetroInfer | ParisKV | Full | SOCKET | MagicPIG | PQCache | FreeKV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K | 20.78 | 1106.14 | 29.14 | 24.42 | 23.13 | 225.00 | 120.57 | 243.91 | 21.78 |
| 256K | 46.30 | OOM | 37.53 | **32.16** | 33.29 | OOM | 230.24 | 754.76 | NA |
| 384K | OOM | OOM | 33.31 | 37.19 | OOM | OOM | 487.77 | 1207.81 | NA |

---

## 5. Interpretation

These comparisons support three conclusions:

1. **Accuracy:** ParisKV consistently matches or approaches full attention and outperforms prior retrieval-based methods.
2. **Long-context robustness:** ParisKV remains strong on both LongBench-v2 and RULER, while several baselines degrade more sharply.
3. **Scalability:** Under large batch sizes or longer contexts, ParisKV remains stable while prior methods often become CPU-bound, unsupported, or OOM.

Overall, ParisKV provides a strong accuracy–efficiency tradeoff under comparable retrieval budgets.

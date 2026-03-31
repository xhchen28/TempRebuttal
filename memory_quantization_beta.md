# Memory, Quantization, and Beta-Prior Validation

## 1. Comprehensive KV Cache Memory Usage Analysis

**Reviewer concern.**  
“4-bit quant for kv-cache footprint is still a costly burden; please show the cost with request increase and give a comprehensive memory usage analysis.”

### Main takeaway

The 4-bit Key cache is **not** a costly burden. It is the main source of memory savings in ParisKV.

Compared to full attention (bf16 K+V), ParisKV stores:
- only **Keys** (not Values),
- at **4-bit precision**.

This reduces the per-token storage cost from **4096B** to **512B**, i.e., **12.5%** of full attention. Including all components, ParisKV uses about **556.5B/token**, i.e., **13.6%** of full KV memory.

### Per-token memory accounting

| Component | Per-token-per-layer Storage | Ratio vs Full Attention |
|---|---:|---:|
| Full Attention (bf16 K+V) | 4096 B | 100% |
| 4-bit Key Cache (Key only) | 512 B | 12.5% |
| ParisKV equivalent (all components) | 556.5 B | 13.6% |

Thus, the 4-bit Key cache reduces storage to **1/8 of full attention**:
- **4×** from quantization (16-bit → 4-bit),
- **2×** from storing only K instead of K+V.

### Scaling with concurrent requests

We evaluate Qwen3-8B at 32K context length (36 layers, 8 KV heads, head_dim=128) and vary batch size from 1 to 64.

| Batch Size | Full Attention (GB) | ParisKV Total (GB) | Local-sink (GB) | Top-k (GB) | 4-bit Key (GB) | Savings |
|---|---:|---:|---:|---:|---:|---:|
| 1  | 4.50  | 0.61  | 0.04 | 0.01 | 0.56 | 86.4% |
| 4  | 18.00 | 2.45  | 0.14 | 0.05 | 2.25 | 86.4% |
| 8  | 36.00 | 4.89  | 0.28 | 0.11 | 4.50 | 86.4% |
| 16 | 72.00 | 9.78  | 0.56 | 0.22 | 9.00 | 86.4% |
| 32 | 144.00 | 19.56 | 1.13 | 0.44 | 18.00 | 86.4% |
| 64 | 288.00 | 39.13 | 2.25 | 0.88 | 36.00 | 86.4% |

Internal ParisKV breakdown:
- Local-sink: **5.7%**
- Top-k working set: **2.2%**
- 4-bit Key cache: **92.1%**

### Interpretation

- The memory reduction stays at **86.4%** across all batch sizes because both methods scale linearly with the number of requests.
- On an **80GB A100**, full attention saturates at about **batch=16** (72GB), while ParisKV still fits **batch=64** (39.13GB), enabling **4×** more concurrent requests on the same hardware.

### Why the 4-bit Key cache is not a burden

Although the 4-bit Key cache accounts for most of ParisKV’s memory, it is still only **12.5%** of the per-token cost of full attention. Without it, one would either:
- store full bf16 Keys (8× more memory), or
- lose long-range token scoring altogether.

Therefore, the 4-bit Key cache is the main mechanism that preserves global retrieval quality while keeping memory usage practical.

---

## 2. Clarification of 4-bit RSQ-IP Quantization

**Reviewer concern.**  
“In RSQ-IP, the 4-bit quantization is unclear; Appendix B2.2.2 says 1-bit sign + 3-bit digits, but it is not clear whether this is e2m1 or e3m0, which is not reproducible.”

### Main clarification

Our 4-bit RSQ-IP is **not a floating-point format**. It is neither **e2m1** nor **e3m0**.

Each coordinate is represented as:
- **1-bit sign**
- **3-bit magnitude index** into an 8-entry codebook

So this is a **codebook-based scalar quantization scheme**, not FP4.

### Why this design

After normalization and SRHT/random rotation, a block direction vector lies on the unit sphere. For a coordinate of a uniformly distributed point on the sphere:

\[
u_j^2 \sim \mathrm{Beta}\left(\frac{1}{2}, \frac{m-1}{2}\right).
\]

Hence, \(|u_j|\) has a **non-uniform distribution concentrated near zero**.

Standard FP4 formats allocate precision according to exponent ranges, not according to the true data distribution of \(|u_j|\). In our setting, that mismatch increases quantization error. Lloyd–Max quantization is better aligned with the actual coordinate distribution and therefore achieves lower error.

### Codebook construction

We construct the codebook in a **data-independent** manner:

1. Sample \(g \sim \mathcal{N}(0, I_m)\)
2. Normalize \(u = g / \|g\|_2\), which yields points uniformly distributed on the sphere
3. Collect samples of \(|u_j|\)
4. Run **Lloyd–Max scalar quantization** with 8 bins

We sample **10M** points from this theoretical distribution and obtain the following codebook:

**Decision thresholds (7):**
- [0.084, 0.170, 0.258, 0.350, 0.449, 0.559, 0.690]

**Reconstruction centers (8):**
- [0.042, 0.127, 0.213, 0.303, 0.397, 0.500, 0.617, 0.763]

### Encoding / decoding

- **Encoding:** use `torch.bucketize` on \(|u_j|\) to get a 3-bit bin index
- **Decoding:** map the index to its corresponding reconstruction center

This procedure is fixed at initialization and shared across all layers and dimensions. It depends only on the block dimension \(m\), not on any model or dataset.

### Reproducibility

In the revision, we will:
- replace the ambiguous phrase “1-bit sign, 3-bit digits” with  
  **“1-bit sign + 3-bit Lloyd–Max codebook index”**
- add the exact codebook construction procedure
- release the codebook generation script
- include full hardware / software context

---

## 3. Empirical Validation of the SRHT-Induced Beta Prior

**Reviewer concern.**  
“Can you provide additional empirical validation that SRHT induces the Beta-like priors assumed in Proposition 4.1?”

### Setup

We extract KV cache tensors from **Qwen3-8B**:
- hidden dimension \(D = 128\)
- 36 layers
- 8 KV heads
- 4096-token input

We apply the production SRHT rotation:
- seed = 42
- 1 round

Then we:
- partition vectors into \(m=8\)-dimensional blocks,
- normalize each block to obtain unit directions \(\mathbf{u}_b \in \mathbb{S}^{m-1}\),
- fit Beta distributions to empirical \((\mathbf{u}_b)_j^2\) via MLE.

### Finding 1: Beta shape is empirically confirmed

Layers 6–30 match the theoretical \(\mathrm{Beta}(0.5, 3.5)\) closely. Early layers deviate more because they retain stronger directional structure, which a single SRHT round does not fully isotropize.

| Layer | Fitted α | Fitted β | KS Stat |
|---|---:|---:|---:|
| L0  | 8.787 | 61.503 | 0.449 |
| L6  | 0.624 | 4.445 | 0.071 |
| L12 | 0.575 | 4.054 | 0.048 |
| L18 | 0.515 | 3.617 | 0.012 |
| L24 | 0.508 | 3.560 | 0.007 |
| L30 | 0.521 | 3.674 | 0.015 |
| Theory | 0.500 | 3.500 | — |

### Finding 2: The theory-based codebook is near-optimal on real data

We compare three codebooks:
- theory-based Beta prior
- empirical data-optimal codebook
- uniform spacing

| Codebook | MSE on Real Data | Gap to Optimal |
|---|---:|---:|
| Theory (Beta prior) | 0.000824 | +2.5% |
| Empirical (data-optimal) | 0.000804 | 0% |
| Uniform | 0.001626 | +102% |

### Interpretation

The practical gap between the theory-based codebook and the data-optimal codebook is only **2.5%**, while a naive uniform quantizer is much worse. This shows that:
- the Beta prior is a good empirical fit in the layers that matter most,
- and even where it is imperfect, it is still sufficient for **near-optimal 3-bit quantization**.

In other words, the value of the prior is not that every layer exactly matches the same Beta parameters, but that SRHT reliably induces the right qualitative shape:
- right-skewed
- concentrated near zero

This is exactly the property needed to build a **universal, data-independent codebook** without per-model calibration.

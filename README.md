# Anonymous Supplementary Material for ParisKV Rebuttal

This repository provides supplementary details referenced in the rebuttal, including:

1. **KV cache memory analysis**
   - per-token memory accounting
   - scaling with batch size / concurrent requests
   - comparison with full attention under the same GPU budget

2. **4-bit RSQ-IP quantization details**
   - clarification that the format is not FP4 (neither e2m1 nor e3m0)
   - codebook construction via Lloyd–Max quantization
   - exact thresholds / reconstruction centers
   - reproducibility notes and implementation details

3. **Empirical validation of the SRHT-induced Beta prior**
   - fitted Beta parameters across layers
   - KS statistics
   - comparison between theory-based and empirical codebooks

4. **Expanded baseline comparison**
   - evaluation protocol and fairness / budget alignment
   - detailed configurations for RetroInfer, ShadowKV, SOCKET, Quest, Twilight, FreeKV, MagicPIG, and PQCache
   - full LongBench-v2, RULER, GPQA, and efficiency tables

## Files

- [memory_quantization_beta.md](./memory_quantization_beta.md)
- [baselines_accuracy_efficiency.md](./baselines_accuracy_efficiency.md)

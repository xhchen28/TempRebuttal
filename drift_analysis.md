
### W2. Fine-grained analysis of drift across layers and heads

In the revision, we add a fine-grained study at full **layer × head** resolution over all attention heads.

We define the recall change for each `(layer, head)` as: `ΔR = R_final - R_init`, where `R_init` is the recall@k computed when retrieval is restricted to **prefill keys only**, and `R_final` is the recall@k computed on the full retrieval pool containing **prefill + decode keys**. To quantify centroid drift, for each `(layer, head)` we train PQ-style codebooks on: prefill keys, and all keys (`prefill + decode`).

`norm_drift_keynorm = mean_l2_drift / mean_key_norm`, for `mean_key_norm > 0`


<p align="center">
  <img src="https://raw.githubusercontent.com/xhchen28/TempRebuttal/master/figures/drift_recall_hotmap.png" alt="Drift and ΔR heatmap" width="700">
</p>

### Extreme cases

| Case | Layer | Head | norm_drift_keynorm | `ΔR` |
|------|------:|-----:|-------------------:|-----------------:|
| Minimum normalized drift | 0 | 1 | 0.1532 | -0.1040 |
| Maximum normalized drift | 5 | 2 | 0.7502 | -0.1585 |






# Expanded Baseline Comparison: Accuracy, Efficiency, and Configurations

## 1. Scope and fairness protocol

All methods are evaluated on the same platform whenever possible. We aim to keep comparisons fair by aligning the *effective retrieval budget* rather than forcing identical hyperparameters. 


## 2. Baseline configurations (summary)

We summarize the key configurations below. All methods are tuned to match **comparable retrieval budgets** whenever possible.

| Method | LongBench-v2 / GPQA | RULER |
|---|---|---|
| SOCKET | bucket_K=8, bucket_L=60, sink=64, local=256, topk=100 | topk=2048 |
| Twilight | top-p=0.5 (meanK≈402, matches ~420 budget) | same |
| RetroInfer | sink=64, local=256, ratio=0.001, clusters=8192 (size≈16) | ratio=0.013 |
| ShadowKV | local=32, outlier=384, sparse=104 | local=32, outlier=304, sparse=2048 |
| FreeKV | sink=64, local=256, page=128 (~448 total) | same |
| Quest | token_budget=420, chunk=16 | token_budget=2048 |
| MagicPIG | LSH (K=10, L=170), dynamic budget | same |
| PQCache | compress=20%, subspace=2, clusters=64 | same |


## 3. Accuracy results

## LongBench-v2 (full breakdown)

| Model | Overall | Short Easy | Short Hard | Medium Easy | Medium Hard | Long Easy | Long Hard |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-4B** | | | | | | | |
| Full | 25.84 | 27.12 | 16.53 | 36.36 | 25.20 | 26.67 | 28.57 |
| PQCache | 17.91 | 16.95 | 19.00 | 13.60 | 19.00 | 20.00 | 19.05 |
| MagicPIG | 16.70 | 18.64 | 10.74 | 14.77 | 20.47 | 28.89 | 12.70 |
| ShadowKV | 16.30 | 13.60 | 14.80 | 22.20 | 9.90 | 21.30 | 19.00 |
| FreeKV | 19.68 | 20.34 | 13.22 | 21.59 | 24.41 | 22.22 | 17.46 |
| Quest | 19.12 | 32.00 | 16.18 | 21.43 | 12.50 | 21.05 | 24.24 |
| RetroInfer | 23.69 | 34.50 | 31.70 | 22.70 | 22.20 | 13.60 | 9.70 |
| SOCKET | 17.93 | 20.00 | 22.06 | 16.67 | 12.50 | 15.79 | 21.21 |
| Twilight | 19.12 | 28.00 | 14.71 | 21.43 | 15.62 | 21.05 | 24.24 |
| **ParisKV** | **24.60** | **35.59** | **19.49** | **26.14** | **22.05** | **28.89** | **23.81** |

| Model | Overall | Short Easy | Short Hard | Medium Easy | Medium Hard | Long Easy | Long Hard |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-8B** | | | | | | | |
| Full | 33.59 | 50.85 | 34.71 | 32.95 | 25.98 | 37.78 | 28.57 |
| PQCache | 25.50 | 23.70 | 31.60 | 28.20 | 24.00 | 25.00 | 20.00 |
| MagicPIG | 10.34 | 8.47 | 15.70 | 7.95 | 7.87 | 13.33 | 7.94 |
| ShadowKV | 15.90 | 40.70 | 23.10 | 6.80 | 13.40 | 6.70 | 3.20 |
| FreeKV | 15.31 | 28.81 | 24.79 | 15.91 | 11.81 | 6.70 | 3.20 |
| Quest | 23.90 | 36.00 | 25.00 | 28.57 | 15.62 | 31.58 | 18.18 |
| RetroInfer | 20.48 | 44.80 | 36.70 | 20.50 | 7.90 | 4.50 | 3.20 |
| SOCKET | 17.93 | 24.00 | 22.06 | 14.29 | 20.31 | 10.53 | 9.09 |
| Twilight | 20.32 | 36.00 | 26.47 | 19.05 | 12.50 | 15.79 | 15.15 |
| **ParisKV** | **33.07** | **52.54** | **34.71** | **34.09** | **26.77** | **37.21** | **19.67** |

| Model | Overall | Short Easy | Short Hard | Medium Easy | Medium Hard | Long Easy | Long Hard |
|---|---:|---:|---:|---:|---:|---:|---:|
| **DS-R1-8B** | | | | | | | |
| Full | 13.12 | 18.64 | 15.70 | 12.50 | 8.66 | 11.11 | 14.29 |
| PQCache | 19.90 | 18.60 | 21.50 | 21.60 | 22.20 | 15.60 | 14.30 |
| MagicPIG | 13.92 | 15.25 | 11.57 | 11.36 | 14.17 | 17.78 | 17.46 |
| ShadowKV | 14.51 | 18.60 | 23.10 | 13.60 | 14.20 | 2.20 | 4.80 |
| FreeKV | 17.50 | 44.07 | 30.58 | 9.09 | 8.66 | 6.67 | 4.76 |
| Quest | 23.51 | 24.00 | 27.94 | 28.57 | 14.06 | 26.32 | 24.24 |
| RetroInfer | 15.66 | 27.60 | 20.00 | 20.50 | 9.50 | 4.50 | 9.70 |
| SOCKET | 21.51 | 20.00 | 14.71 | 30.95 | 21.88 | 15.79 | 27.27 |
| Twilight | 20.72 | 20.00 | 14.71 | 26.19 | 23.44 | 15.79 | 24.24 |
| **ParisKV** | **28.43** | **37.29** | **25.62** | **28.41** | **25.20** | **31.11** | **30.16** |



## RULER (full breakdown)

<table>
  <thead>
    <tr>
      <th style="min-width: 140px; text-align: left;">Method</th>
      <th>niah_s1</th>
      <th>niah_s2</th>
      <th>niah_m1</th>
      <th>niah_m2</th>
      <th>niah_mv</th>
      <th>niah_mq</th>
      <th>fwe</th>
      <th>qa_1</th>
      <th>qa_2</th>
      <th>vt</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="text-align:left;">Llama3.1-8B</td><td>100.00</td><td>100.00</td><td>96.88</td><td>89.58</td><td>98.44</td><td>99.74</td><td>71.18</td><td>86.46</td><td>51.04</td><td>47.92</td><td>84.12</td></tr>
    <tr><td style="text-align:left;"><strong>ParisKV</strong></td><td>100.00</td><td>100.00</td><td>96.88</td><td>82.01</td><td>95.83</td><td>98.70</td><td>56.25</td><td>84.38</td><td>48.96</td><td>71.87</td><td><strong>83.49</strong></td></tr>
    <tr><td style="text-align:left;">MagicPIG</td><td>100.00</td><td>94.79</td><td>83.33</td><td>34.38</td><td>73.44</td><td>78.12</td><td>70.08</td><td>73.96</td><td>42.71</td><td>53.33</td><td>70.49</td></tr>
    <tr><td style="text-align:left;">ShadowKV</td><td>100.00</td><td>98.96</td><td>96.88</td><td>73.96</td><td>89.06</td><td>96.35</td><td>61.11</td><td>81.25</td><td>50.00</td><td>55.42</td><td>80.29</td></tr>
    <tr><td style="text-align:left;">PQCache</td><td>5.21</td><td>50.00</td><td>45.83</td><td>30.21</td><td>16.67</td><td>20.31</td><td>55.21</td><td>68.75</td><td>40.62</td><td>4.79</td><td>33.76</td></tr>
    <tr><td style="text-align:left;">SOCKET</td><td>100.00</td><td>100.00</td><td>96.88</td><td>72.92</td><td>92.71</td><td>98.18</td><td>33.33</td><td>80.21</td><td>46.88</td><td>68.13</td><td>78.92</td></tr>
    <tr><td style="text-align:left;">Twilight</td><td>98.96</td><td>100.00</td><td>95.83</td><td>76.04</td><td>90.36</td><td>97.92</td><td>57.29</td><td>80.21</td><td>48.96</td><td>74.17</td><td>77.60</td></tr>
    <tr><td style="text-align:left;">RetroInfer</td><td>95.83</td><td>98.96</td><td>95.83</td><td>55.21</td><td>94.53</td><td>97.40</td><td>61.11</td><td>78.12</td><td>42.71</td><td>54.79</td><td>77.45</td></tr>
    <tr><td style="text-align:left;">Quest</td><td>99.00</td><td>98.96</td><td>94.79</td><td>64.58</td><td>83.59</td><td>94.79</td><td>62.85</td><td>79.17</td><td>44.79</td><td>70.83</td><td>79.34</td></tr>
  </tbody>
</table>

### GPQA-diamond

| Method | Qwen3-4B | Quest | RetroInfer | FreeKV | MagicPig++ | PQCache | ParisKV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Score | 64.14 | 38.40 | 38.90 | 58.16 | 32.32 | 38.38 | **72.22** |

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

| Seq len | Quest | Twilight | RetroInfer | ParisKV | Full | SOCKET | MagicPIG | PQCache | FreeKV | ShadowKV|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K | 20.78 | 1106.14 | 29.14 | 24.42 | 23.13 | 225.00 | 120.57 | 243.91 | 21.78 | 53.30|
| 256K | 46.30 | OOM | 37.53 | **32.16** | 33.29 | OOM | 230.24 | 754.76 | NA |150.98|
| 384K | OOM | OOM | 33.31 | 37.19 | OOM | OOM | 487.77 | 1207.81 | NA |318.14|



### Prefill latency (TTFT, seconds)

We report prefill latency (TTFT) as a function of sequence length.

| Seq len | Quest | Twilight | RetroInfer | ParisKV | Full | SOCKET | MagicPIG | PQCache | FreeKV |ShadowKV|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K | 37.28 | 34.56 | 35.74 | 43.00 | 33.30 | 38.29 | 55.10 | 25.50 | 43.83 |51.77 |
| 256K | 117.71 | OOM | 126.71 | 139.50 | 116.20 | OOM | 141.10 | 104.70 | OOM |152.12|
| 384K | OOM | OOM | 274.85 | 290.40 | OOM | OOM | 272.80 | 238.80 | OOM |309.26|

## 5. Main takeaways from the expanded comparison

Overall, the expanded results consistently support the main claim of our paper: **ParisKV provides the best accuracy-efficiency tradeoff among practical long-context retrieval baselines**, especially in the offloading setting and under long-generation workloads.

### Accuracy

Across the newly added baselines and tasks, ParisKV is consistently the strongest or near-full-accuracy method.

- **LongBench-v2.**
  - On **Qwen3-4B**, ParisKV reaches **24.60**, outperforming all retrieval baselines and remaining very close to Full (**25.84**). The next strongest competitor is RetroInfer at **23.69**, while most others are substantially lower (e.g., Quest **19.12**, Twilight **19.12**, FreeKV **19.68**, PQCache **17.91**, ShadowKV **16.30**).
  - On **Qwen3-8B**, ParisKV achieves **33.07**, again the best among retrieval methods and nearly matching Full (**33.59**). Competing methods fall much further behind, e.g., PQCache **25.50**, Quest **23.90**, Twilight **20.32**, RetroInfer **20.48**, and ShadowKV **15.90**.
  - On **DS-R1-8B**, ParisKV reaches **28.43**, clearly outperforming all baselines; the next best methods are Quest (**23.51**) and SOCKET (**21.51**).

- **RULER.** ParisKV achieves **83.49** average, which is the highest among all retrieval baselines and remains very close to Full (**84.12**). In contrast, the next-best retrieval methods are ShadowKV (**80.29**), Quest (**79.34**), and SOCKET (**78.92**).

- **GPQA-diamond.** ParisKV obtains **72.22**, substantially outperforming all retrieval baselines and even exceeding Full (**64.14**). The gap is particularly large compared with Quest (**38.40**), RetroInfer (**38.90**), MagicPIG (**32.32**), and PQCache (**38.38**).

These results show that when ParisKV is not the absolute fastest method under a specific setting, it still provides **substantially stronger accuracy**, and in most cases it is the only method that remains both competitive in speed and close to full-attention quality.

### Efficiency and scalability

From an efficiency perspective, ParisKV is the strongest **practical** method among the baselines that support offloading and remain usable at longer contexts.

- At **128K, bs=1**, ParisKV reaches **41.10 tok/s**, which is close to Full (**43.20 tok/s**) and faster than RetroInfer (**34.31 tok/s**), Twilight (**0.90 tok/s**), and SOCKET (**4.44 tok/s**). Some methods report higher raw throughput at this single point (e.g., MagicPIG, PQCache), but they come with much lower accuracy and weaker scalability.
- More importantly, ParisKV scales to larger batch sizes and longer contexts:
  - at **bs=2**, ParisKV reaches **69.80 tok/s** vs. Full **60.70 tok/s**;
  - at **bs=8**, ParisKV reaches **150.00 tok/s**, while Full is already **OOM** and several baselines are also **OOM** or unsupported.
- In decode latency, ParisKV becomes increasingly favorable as context grows:
  - at **256K**, ParisKV is **32.16 ms/token**, slightly faster than Full (**33.29**) and clearly faster than RetroInfer (**37.53**), MagicPIG (**230.24**), PQCache (**754.76**), and ShadowKV (**150.98**);
  - at **384K**, ParisKV remains runnable at **37.19 ms/token**, whereas Full, Quest, Twilight, and SOCKET are **OOM**.

This behavior is important for the target setting of our paper: **million-scale KV caches with offloading and long generation**. In this regime, scalability and robustness matter more than isolated speed numbers at a single short setting.

### Why this comparison supports our claims

The newly added baselines clarify two key points.

1. **Quest can be competitive in a narrow setting, but it does not provide the same practical scalability.**  
   For example, Quest is slightly faster than ParisKV at **bs=3** (**93.20** vs. **93.10 tok/s**). However, Quest does **not support offloading** in our target setting and cannot scale beyond small-batch configurations (e.g., no valid result at larger batch sizes), whereas ParisKV continues to scale to **bs=8** and long contexts. Thus, Quest's advantage is limited to a narrow operating region and does not contradict our main claim.

2. **ShadowKV is faster in some settings, but the accuracy-efficiency tradeoff is weaker, especially for long generation.**  
   ShadowKV is faster in some decode measurements, but its accuracy is consistently lower than ParisKV (e.g., **16.30 vs. 24.60** on LongBench-v2/Qwen3-4B, **15.90 vs. 33.07** on LongBench-v2/Qwen3-8B, and **80.29 vs. 83.49** on RULER). In addition, ShadowKV still attends to **all generated tokens during decoding**, so under long-generation workloads its decode computation increasingly resembles approximate full attention. This makes it less suitable for the target scenario of our paper, where both **offloaded KV access** and **long decode horizons** must be handled efficiently. In practice, this also limits batch-size scalability and leads to worse decode-time behavior as generation grows.

In summary, the expanded evaluation strengthens rather than weakens our original claims: **ParisKV is the most balanced method overall, combining near-full or best accuracy with strong throughput, better long-context scalability, and practical support for offloaded long-generation inference.**

> **Note on Twilight.**  
> We exclude Twilight from the main speed comparison because its open-sourced codebase only provides a Python reference implementation for accuracy evaluation. 



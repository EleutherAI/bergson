Metasmoothness comparison between TF32 and FP32 for a LoRA trained on Deep Ignorance 7B.

| run | metasmoothness | held-out forget loss change | WMDP-bio loss change |
|---|---|---|---|
| FP32 | 0.934 | -0.168 | -0.709 |
| TF32 | 0.934 | -0.168 | -0.708 |

| MAGIC scores | Spearman | Pearson |
|---|---|---|
| FP32 vs TF32 | 0.850 | 0.859 |

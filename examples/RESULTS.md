# Data Filtering Experiment Results

Comparing data attribution methods for selecting training subsets on `argilla/magpie-ultra-v0.1` with `SmolLM2-1.7B-Instruct`.

## Setup

- **Model**: HuggingFaceTB/SmolLM2-1.7B-Instruct (loaded in fp32, trained with bf16 mixed precision)
- **Dataset**: argilla/magpie-ultra-v0.1 (5000 samples, 95/5 train/test split -> 4750 train, 250 test)
- **Training**: SFTTrainer, 3 epochs, lr=3e-4, cosine schedule, gradient accumulation=32, batch_size=1
- **Gradient Index**: projection_dim=16, token_batch_size=8192, precision=auto (bfloat16)
- **Trackstar**: adafactor normalizer, unit_normalize, mean aggregation, mixed preconditioners (stats_sample_size=10000)
- **Seed**: 42

## Methods

| Method | Description |
|--------|-------------|
| **Full** | Train on all 4750 examples (upper bound) |
| **Random** | Randomly select n examples from the training pool |
| **Attribution** | Score examples by cosine similarity of projected gradients to mean query gradient, select top-k |
| **Trackstar** | Full Trackstar pipeline: compute & mix query/value preconditioners, precondition gradients, score, select top-k |
| **Trackstar-LoRA** | Same as Trackstar but using rank-16 LoRA for the initial SFT step (gradients collected on adapter params only) |

## Results

All methods use full SFT for the initial finetuning step (needed to make gradients meaningful).

### n=2000 (42%)

| Method | eval_loss |
|--------|-----------|
| Full | 0.7901 |
| Trackstar (p=16) | **0.8429** |
| Attribution | 0.8483 |
| Random | 0.8533 |

### n=1000 (21%)

| Method | eval_loss |
|--------|-----------|
| Full | 0.7901 |
| Trackstar (p=16) | **0.8691** |
| Attribution | 0.8847 |
| Random | 0.8969 |

### n=500 (11%)

| Method | eval_loss |
|--------|-----------|
| Full | 0.7901 |
| Trackstar (p=64) | **0.8815** |
| Random | 0.8984 |
| Attribution | 0.9034 |

### n=250 (5%)

| Method | eval_loss |
|--------|-----------|
| Full | 0.7901 |
| Trackstar (p=64) | **0.8480** |
| Trackstar (p=16) | 0.8615 |
| Random | 0.8754 |
| Attribution | 0.8859 |
| Trackstar-LoRA | 0.6995 |

Note: Trackstar-LoRA uses rank-16 LoRA for both the initial SFT and the final filtered SFT, so its eval_loss is not directly comparable to the full-SFT methods above. The LoRA full-dataset baseline is 0.6633.

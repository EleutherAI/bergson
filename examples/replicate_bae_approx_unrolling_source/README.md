# WikiText-2 / GPT-2 replication (Bae et al. 2024)

Reproduces SOURCE > EK-FAC IF (Figure 6) end to end from public data.

Run each config with `PYTHONPATH=$PWD python -m bergson <config>`, in order:

| # | Config | Produces |
|---|--------|----------|
| 0 | `prep_dataset.py` | `EleutherAI/bergson-wikitext-2-4656-chunks` (already hosted; run only to rebuild) |
| 1 | `wikitext_gpt2_train.yaml` | fine-tuned GPT-2, 6 checkpoints with optimizer state |
| 2 | `wikitext_gpt2_source.yaml` | SOURCE scores (auto-exports the checkpoints) |
| 3 | `wikitext_gpt2_ekfac.yaml` | EK-FAC influence scores at the final checkpoint |
| 4 | `wikitext_gpt2_retrain.yaml` | 100 models retrained on random halves of the training data + EK-FAC LDS |
| 5 | `wikitext_gpt2_validate.yaml` | SOURCE LDS against the retrained models |

Step 2 runs before step 3 because it exports the trainer checkpoints to the
HF format both steps load.

LDS, the mean Spearman correlation over 481 validation queries ± 95% CI:

| Ground truth | EK-FAC IF | SOURCE |
|--------------|-----------|--------|
| single seed | 0.418 ± 0.016 | 0.429 ± 0.015 |
| 5-seed averaged | 0.468 ± 0.015 | 0.476 ± 0.015 |

The 5-seed ground truth averages each query's loss over the five retrain seeds
before correlating; the seeds share the same 100 subsets. Under it SOURCE beats
IF on 308/481 queries, Wilcoxon p = 2.3e-11. EK-FAC spread across the five
single-seed ground truths is 0.4142–0.4183; kronfluence reports 0.44 for EK-FAC
IF on its own five-seed ground truth.

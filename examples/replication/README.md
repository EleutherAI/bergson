# WikiText-2 / GPT-2 replication (Bae et al. 2024)

Reproduces the paper's Figure 6 ordering — **SOURCE > EK-FAC IF** — end to end from
public data, with no external ground-truth artifacts. Everything is regenerated
by the same production rules the paper used; only the rules are shared, not any
shipped masks or losses.

Run each config with `PYTHONPATH=$PWD python -m bergson <config>`, in order:

| # | Config | Produces |
|---|--------|----------|
| 0 | `prep_dataset.py` | pushes `EleutherAI/bergson-wikitext-2-4656-chunks` (only needed to rebuild the dataset; the configs already point at the hosted copy) |
| 1 | `wikitext_gpt2_train.yaml` | fine-tuned GPT-2 + 6 checkpoints under `runs/wikitext_repro/train` |
| 2 | `wikitext_gpt2_ekfac.yaml` | EK-FAC influence scores → `runs/wikitext_repro/if_scores` |
| 3 | `wikitext_gpt2_source.yaml` | SOURCE scores → `runs/wikitext_repro/source_scores` |
| 4 | `wikitext_gpt2_bank.yaml` | leave-half retrain bank (ground truth) → `runs/wikitext_repro/bank` |
| 5 | `wikitext_gpt2_validate.yaml` | LDS for both score sets against the bank |

Notes:

- **Data (step 0):** the standard HF `run_clm` recipe on `wikitext-2-raw-v1`
  (GPT-2 tokenizer, 512-token `group_texts`) reproduces the exact 4656 train /
  481 validation chunking. The configs load the hosted copy directly, so step 0
  is only for rebuilding it.
- **Checkpoints:** `wikitext_gpt2_source.yaml` points straight at the trainer's
  DCP snapshots (`checkpoints/step_<n>.ckpt`); the SOURCE pipeline auto-exports
  them to HF format on first use — no manual export step.
- **Ground truth (step 4):** the expensive step — 100 models each retrained on a
  random half (`subset_fraction: 0.5`, i.e. kronfluence's α=0.5). `validate`
  reads this bank via `--retrain_bank` and computes the per-query Spearman (LDS)
  with no retraining.

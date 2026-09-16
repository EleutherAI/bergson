# PAC labeling of attribution scores

PAC labeling (arXiv 2506.10908) decides how much of a dataset a cheap labeler
may label while the average loss against an expert labeler stays below `eps`
with probability `1 - alpha`. Here the expert is MAGIC and the cheap labeler
is EK-FAC, on the `plan_adam_eps1e17_64k_bs256` row of the metasmoothness
study (GPT-2 124M fine-tuned on 64k SmolLM2 chunks, 20 queries, MAGIC LDS
0.958, EK-FAC LDS 0.434). The implementation is `bergson/pac.py`.

## Setup

Each query is one labeling problem over the 64000 training documents. A
calibration sample of 1000 MAGIC scores fits EK-FAC onto the MAGIC scale by
least squares and fixes the loss; a second sample of 2000 MAGIC scores sets
the PAC threshold with the betting bound at `alpha = 0.05`. Every trial
redraws both samples; 100 trials per query.

Losses, all in `[0, 1]`:

| task | loss | cheap-only loss |
|---|---|---|
| `proponent` | 0-1 on membership in the strongest 1% (MAGIC cutoff from the calibration sample; EK-FAC's own top 1%) | 0.0150 |
| `recall` | 1 when a MAGIC proponent is outside EK-FAC's top 1%, so `eps = 0.001` allows 10% of proponents to be missed | 0.0079 |
| `score` | squared error of the calibrated EK-FAC score in units of the proponent cutoff, clipped to 1 | 0.1226 |

Uncertainty scores, all computed without MAGIC: `proponent` (rank of the
calibrated score, strongest first), `boundary` (distance to the cheap cutoff)
or `magnitude` (rank of the absolute score), `learned` (mean calibration loss
per rank bin) and `random`.

```
python examples/pac_labeling/pac_attribution.py \
    --expert $P/scores --cheap $P/ekfac_scores/scores \
    --bank $P/bank_from_filter --out $P/pac/main2 --tasks proponent recall
python examples/pac_labeling/pac_attribution.py ... --out $P/pac/main --tasks score
python examples/pac_labeling/plot_pac.py --trials $P/pac/main2/trials.csv ... --out fig
```

## Results

Means over 20 queries x 100 trials. `saved` is the fraction of documents that
keep the EK-FAC score; `loss q95` is the 95th percentile of the realized loss
(the guarantee asks for it to be at most `eps`); `viol.` is the fraction of
trials above `eps`; `top-1%` and `LDS` describe the hybrid score vector (MAGIC
where expert-labeled, calibrated EK-FAC elsewhere). EK-FAC alone: top-1%
overlap 0.29, LDS 0.43. MAGIC alone: LDS 0.96.

| task | eps | uncertainty | saved | loss q95 | viol. | top-1% | LDS |
|---|---|---|---|---|---|---|---|
| proponent | 0.002 | proponent | 0.16 | 0.0014 | 0.014 | 0.95 | 0.88 |
| proponent | 0.005 | proponent | 0.49 | 0.0034 | 0.005 | 0.84 | 0.77 |
| proponent | 0.01 | proponent | 0.84 | 0.0075 | 0.005 | 0.58 | 0.62 |
| proponent | 0.01 | learned | 0.71 | 0.0075 | 0.003 | 0.59 | 0.67 |
| proponent | 0.01 | random | 0.31 | 0.0075 | 0.005 | 0.70 | 0.82 |
| proponent | 0.02 | proponent | 0.94 | 0.0145 | 0.000 | 0.33 | 0.52 |
| recall | 0.001 | proponent | 0.00 | 0.0000 | 0.000 | 1.00 | 0.95 |
| recall | 0.002 | proponent | 0.16 | 0.0014 | 0.014 | 0.95 | 0.88 |
| recall | 0.005 | proponent | 0.49 | 0.0033 | 0.004 | 0.84 | 0.77 |
| recall | 0.005 | random | 0.23 | 0.0034 | 0.003 | 0.78 | 0.85 |
| score | 0.02 | magnitude | 0.16 | 0.0175 | 0.002 | 0.95 | 0.91 |
| score | 0.05 | magnitude | 0.44 | 0.0451 | 0.000 | 0.84 | 0.83 |
| score | 0.1 | magnitude | 0.81 | 0.0912 | 0.000 | 0.56 | 0.64 |
| score | 0.1 | random | 0.72 | 0.0908 | 0.000 | 0.38 | 0.60 |

Full tables: `summary.csv` in each output directory; figures
`pac_labeling.pdf` (realized loss against fraction saved, every trial) and
`pac_hybrid.pdf` (hybrid top-1% overlap and LDS against fraction saved).

What the numbers say:

* The guarantee holds everywhere: at most 2% of trials exceed `eps` against
  the 5% allowed, and the 95th percentile of the realized loss sits at 70-90%
  of `eps`, so the betting bound is tight.
* Which documents need MAGIC is a property of the loss, and for a 1% base
  rate `eps` has to be far below 1% to mean anything. At `eps = 0.01` the
  procedure certifies EK-FAC on 84% of the corpus, and the hybrid vector
  recovers only 58% of MAGIC's proponents; at `eps = 0.002` it recovers 95%
  and needs MAGIC on 84% of the corpus. The `recall` loss makes this explicit:
  missing at most 10% of MAGIC's proponents needs MAGIC on every document,
  missing at most half needs it on half.
* EK-FAC's own ranking is a weak uncertainty score for this expert. MAGIC's
  top-1% proponents are spread through EK-FAC's ranking (63% inside EK-FAC's
  top 10%, 87% inside its top 50%), and the squared-error mass is spread the
  same way (29% of it inside the top 10% by magnitude). So `proponent` and
  `magnitude` save 10-50 points more than `random`, never more; the `learned`
  score gains nothing over the plain rank. Better uncertainty scores need a
  signal EK-FAC does not carry, such as disagreement between attribution
  methods or across checkpoints.
* Hybrid quality tracks how much MAGIC is in the vector. LDS falls close to
  linearly with the fraction left to EK-FAC, and the uncertainty score moves
  it by at most 0.05 at a fixed budget. Half the corpus on MAGIC gives LDS
  0.77-0.83 against 0.96 for MAGIC alone.

The per-query expert sets under the `proponent` loss at `eps = 0.01` with
the `proponent` uncertainty (trial 0) are written by `--export` as
`expert_set_q*.npy`; they hold 3000-22000 documents per query, of which
3000 are the two random samples.

## MAGIC on the expert set alone

`subset_magic.py` fine-tunes on only the expert-set documents and runs MAGIC
there, and `compare_subset.py` scores the result against the full run on the
same documents. Query 12 (expert set 5065 documents: 2251 above the threshold
plus the 3000 sampled) was run in three recipes on the A100 pod that produced
the full run:

| variant | documents | batch | lr | steps |
|---|---|---|---|---|
| same recipe | 5065 | 256 | 1e-4 | 40 |
| small batch | 5065 | 32 | 2.5e-5 | 316 |
| with background | 5065 + 6400 random | 256 | 1e-4 | 89 |

```
python examples/pac_labeling/subset_magic.py --reference $P/config.yaml \
    --expert-set $P/pac/main2/expert_set_q12.npy --queries 12 3 --run-path runs/q12_same
python -m bergson runs/q12_same/magic.yaml
python examples/pac_labeling/compare_subset.py --full $P/scores \
    --subset runs/q12_same/magic/scores --ids runs/q12_same/orig_ids.npy \
    --query-ids runs/q12_same/query_ids.npy --cheap $P/ekfac_scores/scores \
    --bank $P/bank_from_filter --pac $P/pac/main2
```

Agreement with the full run on the scored documents (query 12 is the query
the set was chosen for; query 3 shares the documents but not the selection).
For the background run the overlap and hybrid columns cover all 11465 scored
documents; its Spearman is on the 5065 expert-set documents alone:

| variant | query | Pearson | Spearman | scale | top-1% overlap | LDS, hybrid with subset scores | LDS, hybrid with full-run scores | LDS, EK-FAC only |
|---|---|---|---|---|---|---|---|---|
| same recipe | 12 | 0.97 | 0.48 | 0.81 | 0.44 | 0.24 | 0.47 | 0.47 |
| small batch | 12 | 0.97 | 0.55 | 1.18 | 0.47 | 0.33 | 0.47 | 0.47 |
| with background | 12 | 0.97 | 0.64 | 0.72 | 0.49 | 0.34 | 0.47 | 0.47 |
| same recipe | 3 | 0.98 | 0.35 | 1.21 | 0.36 | 0.44 | 0.63 | 0.54 |
| small batch | 3 | 0.99 | 0.41 | 1.73 | 0.40 | 0.57 | 0.63 | 0.54 |
| with background | 3 | 0.99 | 0.48 | 1.60 | 0.40 | 0.65 | 0.63 | 0.54 |

Pearson is carried by the few strongest proponents. Spearman, which is what
a filter or an LDS sees, is 0.35-0.64, in the range EK-FAC itself reaches on
the same documents (0.32-0.57), and the scores come out 0.7-1.7x the full
run's scale. Splitting the set into the 2251 documents above the threshold
and the 2814 sampled ones changes Spearman by under 0.1, so the disagreement
is not about which documents were chosen. Longer training moves the scores
toward the full run (same recipe 0.48, small batch 0.55, with background
0.64 on query 12) and the background run's scores on the 6400 random
documents agree with the full run as well as its expert-set scores do
(0.51 against 0.64). Substituting subset-run scores for the full run's on the
expert set lowers the hybrid LDS on query 12 below EK-FAC alone, and on
query 3 only the background run matches the full-run hybrid.

A MAGIC score is the derivative of the query loss along one training
trajectory, and a 40-step trajectory over 5065 documents is a different
trajectory from 500 steps over 64000. The cheap way to get the full run's
scores for a subset is to keep its trajectory and restrict the backward's
per-document gradient to the subset, which costs the full forward once and a
backward proportional to the subset's share of each batch; that is not what
this example measures.

## Hill-climbing the uncertainty score

`hillclimb.py` ranks uncertainty scores by two numbers per loss and `eps`:
the oracle save (sort documents by uncertainty and keep the longest prefix
whose mean loss is at most `eps`, what PAC labeling would certify with
unlimited expert samples) and the certified save with `m` expert samples,
net of those samples and the 1000-document calibration sample. Candidate
scores are each feature's own rank, its rank disagreement with EK-FAC, and
gradient-boosted regressions of the loss on all features fit on the
calibration samples of every query. Features, all computed without MAGIC:

| feature | source | Spearman with MAGIC |
|---|---|---|
| EK-FAC (calibrated) | the cheap scorer | 0.31 |
| BM25 | `bm25_scores` | |
| semantic | `cheap_features.py`, BGE-base cosine of decoded text | |
| activation | `cheap_features.py`, cosine of mean-pooled module inputs | |
| doc loss | `cheap_features.py`, mean token loss under the trained model | |
| training gradients | `train_collect.py` + `score_collected.py`: the HuggingFace `GradientCollectorCallback` on a retrain of the same recipe, projection 32 per module, Adam-normalized and scaled by the step's learning rate, dotted with query gradients at the final model | 0.05 |
| gradient cosine | `gradcos_*.yaml`: full-gradient cosine at the final model and at exported trajectory checkpoints 250 and 375; `gradcos_p32.yaml` the same with a 32-dimensional per-module projection | 0.06, 0.05 |
| EK-FAC at checkpoints | `ekfac_step*.yaml` on the exported checkpoints 250 and 375 (Spearman 0.95 and 0.98 with the final EK-FAC) | 0.30, 0.31 |
| TrackStar | `trackstar_p64.yaml`, projection 64 at the final model | 0.06 |

```
python examples/pac_labeling/export_checkpoint.py --reference $P/config.yaml \
    --checkpoint $P/checkpoints/step_250.ckpt --out $P/pac/ckpt/step_250
python examples/pac_labeling/train_collect.py --reference $P/config.yaml --run-path runs/collect
python -m bergson runs/collect/query_build.yaml
python examples/pac_labeling/score_collected.py --gradients runs/collect/gradients/train \
    --query runs/collect/query --out runs/collect/scores
python examples/pac_labeling/cheap_features.py --model $P/base/model --train <train.hf> \
    --query <query.hf> --out runs/cheap
python examples/pac_labeling/hillclimb.py --expert $P/scores --cheap $P/ekfac_scores/scores \
    --feature bm25=$P/bm25_scores --feature semantic=runs/cheap/semantic_scores ... \
    --doc-feature doc_loss=runs/cheap/doc_loss.npy --m 2000 4000 8000 --out runs/hill
python examples/pac_labeling/plot_hillclimb.py --leaderboard runs/hill/leaderboard.csv --out fig
```

Certified save (net of the expert samples and the calibration sample) by
expert sample size `m`, betting bound, 20 queries x 20 trials, and the
oracle save of each score. `learned[cheap]` uses only EK-FAC-derived
columns, `learned[cheap+retrieval]` adds BM25, semantic, activation and doc
loss, `learned[cheap+gradients]` adds the training-gradient, gradient-cosine
and checkpoint EK-FAC columns, `learned[all]` everything.

| loss | eps | uncertainty | oracle | m=2000 | m=4000 | m=8000 |
|---|---|---|---|---|---|---|
| proponent | 0.005 | cheap_rank | 0.93 | 0.47 | 0.60 | 0.65 |
| proponent | 0.005 | learned[cheap] | 0.91 | 0.44 | 0.58 | 0.64 |
| proponent | 0.005 | learned[cheap+retrieval] | 0.93 | 0.56 | 0.67 | 0.70 |
| proponent | 0.005 | learned[cheap+gradients] | 0.93 | 0.48 | 0.60 | 0.66 |
| proponent | 0.005 | learned[all] | 0.94 | 0.58 | 0.68 | 0.71 |
| proponent | 0.005 | random | 0.35 | 0.10 | 0.14 | 0.16 |
| proponent | 0.01 | cheap_rank | 0.99 | 0.83 | 0.86 | 0.83 |
| proponent | 0.01 | learned[all] | 0.99 | 0.86 | 0.87 | 0.83 |
| recall | 0.002 | cheap_rank | 0.63 | 0.13 | 0.18 | 0.25 |
| recall | 0.002 | learned[cheap+retrieval] | 0.71 | 0.18 | 0.21 | 0.30 |
| recall | 0.002 | learned[all] | 0.73 | 0.18 | 0.23 | 0.32 |
| recall | 0.005 | cheap_rank | 0.93 | 0.47 | 0.60 | 0.65 |
| recall | 0.005 | learned[all] | 0.94 | 0.55 | 0.67 | 0.71 |
| score | 0.05 | cheap_rank | 0.47 | 0.34 | 0.36 | 0.35 |
| score | 0.05 | learned[cheap] | 0.55 | 0.43 | 0.44 | 0.43 |
| score | 0.05 | learned[cheap+retrieval] | 0.60 | 0.50 | 0.50 | 0.48 |
| score | 0.05 | learned[all] | 0.61 | 0.51 | 0.51 | 0.49 |
| score | 0.05 | random | 0.46 | 0.35 | 0.36 | 0.35 |

Violation rates stay at or below 2.3% throughout (`leaderboard.csv`). The CLT
bound certifies 15-25 points more at `m = 2000` and violates `eps` in 6-10%
of trials against the 5% allowed, so it is not used. Figure:
`hillclimb.pdf`.

What moved and what did not:

* The finite expert sample, not the uncertainty score, is what binds the
  binary losses. With EK-FAC's rank alone the oracle would certify 93% of the
  corpus at `eps = 0.005`; the betting bound with 2000 samples certifies 47%,
  with 8000 samples 65% net of the samples themselves. A loss that is 1 on
  about 1.5% of documents needs tens of thousands of samples before a
  prefix with mean 0.004 can be told from 0.005.
* Retrieval features are the useful cheap signal. The learned score over
  EK-FAC plus BM25, semantic, activation and doc loss adds 6-11 points at
  `m = 2000` on the two binary losses, raises the missed-proponent oracle
  from 0.63 to 0.71 at `eps = 0.002`, and lifts the score loss's oracle from
  0.47 to 0.60, where EK-FAC's own rank is no better than random.
* Gradient-family features carry nothing MAGIC-specific. Training-time
  gradients from the HuggingFace callback, full and projected gradient
  cosines at three trajectory points and TrackStar have Spearman 0.05-0.06
  with MAGIC while agreeing with EK-FAC at 0.34-0.57, so projection is not
  what loses the signal: preconditioning is what lifts EK-FAC to 0.31.
  EK-FAC at steps 250 and 375 ranks like the final EK-FAC (Spearman 0.95
  and 0.98, top-1% overlap 0.79), so checkpoint disagreement is small.
  Adding all of them to the learned score moves the certified save by at
  most one point.
* The learned score on EK-FAC columns alone is slightly worse than the plain
  rank: with 1000 calibration documents per query the regression adds noise
  and no information.
* Importance-weighted sampling (the paper's `pi_i`, now supported by
  `pac_threshold`) does not help: labels spent above the eventual threshold
  buy nothing, and the loss below it is what the bound has to resolve.

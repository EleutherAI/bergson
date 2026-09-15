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

Agreement with the full run on the 5065 expert-set documents (query 12 is
the query the set was chosen for; query 3 shares the documents but not the
selection):

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

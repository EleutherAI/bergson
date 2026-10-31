# Two-stage retrieval

We can optimize data attribution for the specific task of surfacing the top-k most influential items at pretraining scale (e.g. for data filtering) using two-stage retrieval. This approach is common in search pipelines.

In two-stage retrieval, we first use an efficient but relatively inaccurate method like TrackStar to score all the items. We then select the top-k * n items, where n is a free hyperparameter. In search this is called "overfetching."

Then, we use a less efficient but more accurate method like EKFAC to score and rerank just the top-k * n items, before selecting the final top-k.

This directory contains an example implementation of this setup on `EleutherAI/pythia-14m` and the first 2,000 rows of `NeelNanda/pile-10k`, with k = 10 and n = 10.

## 1. Build the index

```bash
bergson examples/pretraining_scale_attribution/build.yaml
```

Fits an autocorrelation Hessian over the training gradients, builds a compressed index (a 16-dimensional random projection of each training example's gradient) and fits the EK-FAC factors the rerank stage uses. All three are built once and reused for every query.

## 2. Query and rerank

`query` reads queries from stdin until it sees `exit`, so pipe the query text in. The query lives in `query.csv` so the EK-FAC stage can read the same text as a dataset.

```bash
# Clear the previous query's outputs; the EK-FAC factors are kept.
rm -rf runs/pretraining_scale_attribution/{candidates.csv,query,kfac_query,scores}
tail -n +2 examples/pretraining_scale_attribution/query.csv \
  | sed 's/$/\nexit/' \
  | bergson examples/pretraining_scale_attribution/query_and_rerank.yaml
```

The `query` step scores the query against every row of the index with TrackStar (cosine similarity of the H^-1/2 preconditioned projected gradients) and records its top 100 rows in `runs/pretraining_scale_attribution/candidates.csv`. The `ekfac` step then reads back the EK-FAC factors from the build stage (`resume: true` skips the fit when its output exists) and scores only those 100 candidates against the query.

## 3. Select the final top-k

The reranked scores have one row per candidate, in training order. `candidates.npy` beside them holds each row's index into the training set:

```python
import numpy as np
from pathlib import Path
from bergson.data import load_scores

scores = load_scores(Path("runs/pretraining_scale_attribution/scores"))
reranked = np.asarray(scores[:]).reshape(len(scores), -1)[:, 0]
top10 = scores.candidates[np.argsort(-reranked)[:10]]
print(top10)  # training rows of the 10 most influential documents
```

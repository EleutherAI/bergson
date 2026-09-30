Per-token attribution on the WikiText leaderboard model (the `1_magic.yaml` run in `..`). Each method scores every training token of the 4,608 chunks against the same 50 test chunks (`test[0:50]`): row `t` of a chunk is the score of the loss on token `t + 1`. The proponent QLD masks the query's top 1% of training tokens by those scores (23,484 loss terms per query, spread over about 4,000 chunks) and retrains with the leaderboard recipe; the control retrains without a random 1% of tokens, three retrains per query. Removing a random 1% of tokens changes query loss by 0.0006 on average.

| method | proponent QLD | 95% CI | median | min | max | random 1% | queries above random |
|---|---|---|---|---|---|---|---|
| MAGIC | 1.375 | [1.348, 1.403] | 1.368 | 1.175 | 1.592 | 0.0004 | 50/50 |
| EK-FAC + ASTRA | 1.082 | [1.047, 1.116] | 1.082 | 0.818 | 1.369 | 0.0007 | 50/50 |
| EK-FAC | 0.863 | [0.833, 0.894] | 0.860 | 0.648 | 1.175 | 0.0006 | 50/50 |

Paired over queries, MAGIC exceeds EK-FAC + ASTRA by 0.293 [0.269, 0.317] and EK-FAC + ASTRA exceeds EK-FAC by 0.219 [0.203, 0.234], each on 50 of 50 queries (95% CI from the per-query differences). The 95% CIs in the table are a 10k bootstrap over queries.

`ekfac_tokens.yaml` and `ekfac_astra_tokens.yaml` score the published EK-FAC and ASTRA query directions (`../ekfac.yaml`, `../ekfac_astra.yaml`) token by token with `token_influence: output`; `magic_tokens.yaml` resumes the leaderboard trajectory with `attribute_tokens: true`, so its scores are the per-token MAGIC gradient with the four epochs summed onto each chunk. The `filter_*_tokens.yaml` configs are `validate` runs with `method.kind: filter` over the per-token score stores; `exclude_zero_scores` skips the final position of each chunk, which carries no loss.

Reproduce (after `../1_magic.yaml`, `../2_interval.yaml`, the checkpoint export, `../ekfac.yaml` and `../ekfac_astra.yaml`):

```bash
bergson examples/compare_wikitext/tokens/ekfac_tokens.yaml
bergson examples/compare_wikitext/tokens/ekfac_astra_tokens.yaml
mkdir -p runs/compare_wikitext/tokens/magic_tokens
cp -r runs/compare_wikitext/random/checkpoints runs/compare_wikitext/random/optimizer.pt runs/compare_wikitext/tokens/magic_tokens/
bergson examples/compare_wikitext/tokens/magic_tokens.yaml
for f in ekfac ekfac_astra magic; do
  bergson examples/compare_wikitext/tokens/filter_${f}_tokens.yaml
done
python examples/compare_wikitext/qld_from_filters.py runs/compare_wikitext/tokens
```

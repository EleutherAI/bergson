Per-chunk attribution on the WikiText leaderboard model. 

`chunk_scores.py` sums a per-token score store over chunks of each 512-token training chunk and writes the sum back to every row of the chunk, so the per-token proponent filter masks whole chunks. 

Two chunking strategies:
- `fixed64`: 64-token windows, 36,864 chunks (8 per training chunk).
- `natural`: cuts after sentence ends (`" . "`) and line breaks, merged forward until each piece has at least 32 tokens; 47,039 pieces, median 46 tokens (5 to 14 per training chunk).

The proponent QLD masks the query's top 1% of training tokens by these scores (23,547 loss terms per query): about 369 whole windows in 300 training chunks, or 413 whole pieces in 330. The control retrains without a random 1% of training tokens, three retrains per query, and changes query loss by 0.0004 on average.

| method | chunks | proponent QLD | 95% CI | median | min | max | queries above random |
|---|---|---|---|---|---|---|---|
| MAGIC | natural | 0.239 | [0.226, 0.253] | 0.226 | 0.167 | 0.409 | 50/50 |
| MAGIC | fixed64 | 0.226 | [0.214, 0.240] | 0.214 | 0.157 | 0.384 | 50/50 |
| EKFAC + ASTRA | natural | 0.153 | [0.140, 0.168] | 0.143 | 0.064 | 0.309 | 50/50 |
| EKFAC + ASTRA | fixed64 | 0.147 | [0.133, 0.161] | 0.137 | 0.057 | 0.299 | 50/50 |
| EKFAC | natural | 0.135 | [0.121, 0.149] | 0.124 | 0.072 | 0.297 | 50/50 |
| EKFAC | fixed64 | 0.129 | [0.116, 0.143] | 0.118 | 0.068 | 0.276 | 50/50 |

The 95% CIs are a 10k bootstrap over queries. The paired CIs come from per-query differences.

Reproduce (after the per-token score stores in `../tokens`):

```bash
for m in magic ekfac_astra ekfac; do
  for c in natural fixed64; do
    python examples/compare_wikitext/chunks/chunk_scores.py \
      runs/compare_wikitext/tokens/${m}_tokens runs/compare_wikitext/chunks/${m}_${c} $c
    bergson examples/compare_wikitext/chunks/filter_${m}_${c}.yaml
  done
done
python examples/compare_wikitext/qld_from_filters.py runs/compare_wikitext/chunks
```

Per-span attribution on the WikiText leaderboard model (the `1_magic.yaml` run in `..`).

`span_scores.py` sums a per-token score store over spans of each 512-token training chunk and writes the sum back to every row of the span, so the per-token proponent filter masks whole spans.

Two kinds of span:
- `fixed64`: 64-token spans, 36,864 in all (8 per training chunk).
- `natural`: cuts after sentence ends (`" . "`) and line breaks, merged forward until each span has at least 32 tokens; 47,039 in all, median 46 tokens (5 to 14 per training chunk).

The proponent QLD masks the query's top 1% of training tokens by these scores (23,547 loss terms per query): about 369 whole 64-token spans in 300 training chunks, or 413 whole natural spans in 330. The control retrains without a random 1% of training tokens, three retrains per query, and changes query loss by 0.0004 on average.

| method | spans | proponent QLD | 95% CI | median | min | max | queries above random |
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
  for s in natural fixed64; do
    python examples/gpt2_wikitext_leaderboard/spans/span_scores.py \
      runs/compare_wikitext/tokens/${m}_tokens runs/compare_wikitext/spans/${m}_${s} $s
    bergson examples/gpt2_wikitext_leaderboard/spans/filter_${m}_${s}.yaml
  done
done
python examples/gpt2_wikitext_leaderboard/qld_from_filters.py runs/compare_wikitext/spans
```

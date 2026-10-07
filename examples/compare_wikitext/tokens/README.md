Per-token attribution on the WikiText leaderboard model (the `1_magic.yaml` run in `..`). Each method scores every training token of the 4,608 chunks against the same 50 test chunks (`test[0:50]`): row `t` of a chunk is the score of the loss on token `t + 1`. The proponent QLD masks the query's top 1% of training tokens by those scores (23,484 loss terms per query, spread over about 4,000 chunks) and retrains with the leaderboard recipe; the control retrains without a random 1% of tokens, three retrains per query. Removing a random 1% of tokens changes query loss by 0.0006 on average.

| method | proponent QLD | 95% CI | median | min | max | random 1% | queries above random |
|---|---|---|---|---|---|---|---|
| MAGIC | 1.375 | [1.348, 1.403] | 1.368 | 1.175 | 1.592 | 0.0004 | 50/50 |
| EK-FAC + ASTRA (CG) | 1.014 | [0.986, 1.043] | 1.010 | 0.821 | 1.252 | 0.0007 | 50/50 |
| Eigenvalue-corrected Shampoo | 0.922 | [0.887, 0.957] | 0.922 | 0.671 | 1.270 | 0.0011 | 50/50 |
| SOURCE (Adam, EK-FAC) | 0.908 | [0.877, 0.942] | 0.903 | 0.686 | 1.266 | 0.0004 | 50/50 |
| EK-FAC | 0.863 | [0.833, 0.894] | 0.860 | 0.648 | 1.175 | 0.0006 | 50/50 |
| KFAC | 0.786 | [0.758, 0.815] | 0.784 | 0.584 | 1.058 | 0.0008 | 50/50 |
| BM25 | 0.677 | [0.650, 0.704] | 0.678 | 0.462 | 0.926 | 0.0004 | 50/50 |
| TrackStar (no optimizer correction, projection 64) | 0.492 | [0.467, 0.517] | 0.491 | 0.339 | 0.738 | 0.0012 | 50/50 |
| Gradient cosine similarity | 0.217 | [0.206, 0.229] | 0.212 | 0.148 | 0.342 | 0.0009 | 50/50 |
| TRAK (8-model ensemble) | 0.119 | [0.112, 0.125] | 0.116 | 0.072 | 0.191 | 0.0004 | 50/50 |
| Activation similarity | 0.028 | [0.018, 0.039] | 0.012 | -0.005 | 0.163 | 0.0004 | 47/50 |
| Qwen3-Embedding-8B semantic search | 0.008 | [0.006, 0.010] | 0.008 | -0.001 | 0.029 | 0.0004 | 44/50 |

Paired over queries, MAGIC exceeds EK-FAC + ASTRA (CG) by 0.361 [0.342, 0.379] and EK-FAC + ASTRA (CG) exceeds EK-FAC by 0.151 [0.138, 0.165], each on 50 of 50 queries (95% CI from the per-query differences). The 95% CIs in the table are a 10k bootstrap over queries.

`ekfac_tokens.yaml` and `ekfac_astra_cg_tokens.yaml` score the published EK-FAC and ASTRA query directions (`../ekfac.yaml`, `../ekfac_astra_cg.yaml`) token by token with `token_influence: output`; `magic_tokens.yaml` resumes the leaderboard trajectory with `attribute_tokens: true`, so its scores are the per-token MAGIC gradient with the four epochs summed onto each chunk. The `filter_*_tokens.yaml` configs are `validate` runs with `method.kind: filter` over the per-token score stores; `exclude_zero_scores` skips the final position of each chunk, which carries no loss.

Every other method's rows are the forward-mode equivalent of its document score: row `t` is the method's share of the chunk's score from the loss on token `t + 1`, and a chunk's rows add up to its score.

- KFAC (`kfac_tokens.yaml`) scores the preconditioned queries of `../kfac.yaml` like EK-FAC.
- Eigenvalue-corrected Shampoo (`shampoo_tokens.yaml`) preconditions the query gradients of `../kfac.yaml` with the Shampoo factors of `../shampoo.yaml`.
- SOURCE (`source_adam_tokens.yaml`) runs one pass per interval checkpoint along its segment's query (`query_grad_segment` of the `source_adam` run in `../source.yaml`), which `combine_tokens.py source` combines as SOURCE does.
- Gradient cosine similarity, TrackStar and TRAK score along the full-parameter direction `direction_store.py` rebuilds from the doc-level run: the query preconditioned and normalized as its scorer does, mapped back through the random projection (per module for TrackStar, global for TRAK). `combine_tokens.py rescale` gives each chunk its doc-level normalization (the inverse norm of its training gradient); `combine_tokens.py trak` averages the eight members' log-odds passes as bergson averages members.
- The gradient-free baselines split their document scores exactly (`token_baselines.py`): activation similarity by position, since its pooled activations are means over positions; BM25 by term occurrence, over the GPT-2 tokens an occurrence spans; Qwen3-Embedding, which pools a causal model's last token, by each token's change in the prefix embedding's cosine. Tokens with no lexical or embedding overlap score exactly zero, which the filter's `exclude_zero_scores` would drop from its pool, so `full_pool.py` gives them `+1e-30` and every method removes 1% of all training tokens.

Reproduce (after `../1_magic.yaml`, `../2_interval.yaml`, the checkpoint export, `../ekfac.yaml` and `../ekfac_astra_cg.yaml`):

```bash
bergson examples/compare_wikitext/tokens/ekfac_tokens.yaml
bergson examples/compare_wikitext/tokens/ekfac_astra_cg_tokens.yaml
mkdir -p runs/compare_wikitext/tokens/magic_tokens
cp -r runs/compare_wikitext/random/checkpoints runs/compare_wikitext/random/optimizer.pt runs/compare_wikitext/tokens/magic_tokens/
bergson examples/compare_wikitext/tokens/magic_tokens.yaml
for f in ekfac ekfac_astra_cg magic; do
  bergson examples/compare_wikitext/tokens/filter_${f}_tokens.yaml
done
python examples/compare_wikitext/qld_from_filters.py runs/compare_wikitext/tokens
```

The other rows (after `../kfac.yaml`, `../shampoo.yaml`, `../source.yaml`, `../gradient_cosine.yaml`, `../trackstar.yaml`, `../trak.yaml` and the gradient-free baselines written to `runs/compare_wikitext/baselines`):

```bash
T=examples/compare_wikitext/tokens; R=runs/compare_wikitext
python -c "from bergson.utils.trainer_export import export_checkpoints; export_checkpoints('$R/interval', steps=[12, 24, 36, 48, 60, 72])"
bergson $T/kfac_tokens.yaml
bergson $T/shampoo_tokens.yaml
bergson $T/source_adam_tokens.yaml
python $T/combine_tokens.py source $R/source_adam $R/tokens $R/tokens/source_adam_tokens/scores
python $T/direction_store.py $R/gradient_cosine/scores $R/tokens/gradient_cosine_direction
bergson $T/gradient_cosine_tokens.yaml
python $T/combine_tokens.py rescale $R/tokens/gradient_cosine_pass $R/gradient_cosine/scores $R/tokens/gradient_cosine_tokens/scores
python $T/direction_store.py $R/trackstar_p64/scores $R/tokens/trackstar_direction
bergson $T/trackstar_tokens.yaml
python $T/combine_tokens.py rescale $R/tokens/trackstar_pass $R/trackstar_p64/scores $R/tokens/trackstar_tokens/scores
for i in 0 1 2 3 4 5 6 7; do python $T/direction_store.py $R/trak/checkpoint_$i/scores $R/tokens/trak_direction_$i; done
bergson $T/trak_tokens.yaml
python $T/combine_tokens.py trak $R/trak $R/tokens $R/tokens/trak_tokens/scores
for b in activation bm25 qwen3; do
  python $T/token_baselines.py $b $R/tokens/${b}_tokens && python $T/token_baselines.py $b $R/tokens/${b}_tokens --merge
done
for b in bm25 qwen3; do python $T/full_pool.py $R/tokens/${b}_tokens/scores $R/tokens/${b}_tokens/full; done
for f in kfac shampoo source_adam gradient_cosine trackstar trak bm25 qwen3 activation; do
  bergson $T/filter_${f}_tokens.yaml
done
python examples/compare_wikitext/qld_from_filters.py runs/compare_wikitext/tokens
```

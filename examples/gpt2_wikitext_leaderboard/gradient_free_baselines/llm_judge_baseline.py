"""Baseline: an LLM judges whether each training doc helps predict each query.

For every (query, training doc) pair, Qwen3-8B (thinking disabled) is asked
whether training on the doc would make a language model better at predicting
the query, and answers Yes or No. A pair scores by the log-odds of the first
answer token, log p(Yes) - log p(No), so every pair gets a continuous score
(a sampled Yes/No would leave most pairs tied). The loss-diff-convention score
is the negated log-odds.

The query text comes first in the prompt so vLLM's prefix cache reuses it
across the query's training docs. The judgments are cached in
``<out>/llm_judge_logodds.npy``; a rerun with the file present skips the LLM.

Run with (builds the default bank if --bank is omitted):
    B=examples.gpt2_wikitext_leaderboard.gradient_free_baselines
    python -m $B.llm_judge_baseline --bank runs/retrain_bank_path
"""

import argparse
from pathlib import Path

import numpy as np

from . import common

MODEL = "Qwen/Qwen3-8B"

SYSTEM = (
    "You judge training data attribution. A small language model was trained "
    "on many passages of Wikipedia text."
)
QUESTION = (
    "Passage A was held out from training. Passage B is one of the training "
    "passages.\n\nPassage A:\n{query}\n\nPassage B:\n{train}\n\nWould "
    "training on passage B make the model better at predicting the text of "
    "passage A? Answer with Yes or No."
)


def judge(
    train_texts: list[str],
    query_texts: list[str],
    model: str,
    max_chars: int,
    tensor_parallel_size: int,
) -> np.ndarray:
    """``[num_train_docs, num_queries]`` log p(Yes) - log p(No)."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        dtype="bfloat16",
        enable_prefix_caching=True,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=4096,
    )
    tok = llm.get_tokenizer()
    yes, no = (tok.encode(w, add_special_tokens=False) for w in ("Yes", "No"))
    assert len(yes) == len(no) == 1, (yes, no)

    def prompt(query: str, train: str) -> str:
        user = QUESTION.format(query=query[:max_chars], train=train[:max_chars])
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    # Only the first answer token's distribution is needed; ask for enough
    # logprobs that Yes and No are both among them.
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    floor = np.log(1e-6)
    out = np.empty((len(train_texts), len(query_texts)), dtype=np.float32)
    for j, query in enumerate(query_texts):
        results = llm.generate([prompt(query, t) for t in train_texts], params)
        for i, r in enumerate(results):
            lp = r.outputs[0].logprobs[0]
            p_yes = lp[yes[0]].logprob if yes[0] in lp else floor
            p_no = lp[no[0]].logprob if no[0] in lp else floor
            out[i, j] = p_yes - p_no
        print(
            f"query {j + 1}/{len(query_texts)}: mean log-odds {out[:, j].mean():+.3f}"
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=None, help="re-train bank dir; built if omitted")
    ap.add_argument("--query_split", default=common.DEFAULT_QUERY_SPLIT)
    ap.add_argument(
        "--query_dataset",
        default=None,
        help="query dataset; default = bank train dataset",
    )
    ap.add_argument(
        "--out", default=str(common.REPO / "runs" / "gradient_free_baselines")
    )
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max_chars", type=int, default=4000)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    args = ap.parse_args()

    bank = common.ensure_bank(args.bank)
    spec = common.read_bank_spec(bank)
    query_dataset = args.query_dataset or spec.dataset
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = out_dir / "llm_judge_logodds.npy"
    if cache.exists():
        logodds = np.load(cache)
    else:
        train_texts, query_texts = common.load_texts(
            spec, query_dataset, args.query_split
        )
        print(f"Judging {len(train_texts)} train docs x {len(query_texts)} queries")
        logodds = judge(
            train_texts,
            query_texts,
            args.model,
            args.max_chars,
            args.tensor_parallel_size,
        )
        np.save(cache, logodds)

    scores = -logodds  # loss-diff convention (Yes => influential)
    score_path = common.save_scores(scores, out_dir, "llm_judge_scores")

    rhos = common.evaluate_lds(
        bank,
        score_path,
        out_dir / "llm_judge_validate",
        spec,
        query_dataset,
        args.query_split,
    )
    common.report(f"{args.model} judge", rhos)


if __name__ == "__main__":
    main()

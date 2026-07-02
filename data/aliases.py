"""Tokenizer-constrained random aliases for named entities.

Two kinds, following Krasheninnikov et al.'s alias builder
(data_generation/cvdb_natural_style.py in
https://github.com/krasheninnikov/internalization):

- letter aliases: random lowercase strings like "sjdhf" (Synthetic variant)
- phrase aliases: adjective(s) + noun like "prickly cyan mouse" (Natural
  variant)

Every alias encodes to exactly ``n_tokens`` tokens under the given tokenizer.
"""

import random
import string
import warnings

from transformers import PreTrainedTokenizerBase

STOP_WORDS = {"a", "an", "the"}


def token_len(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def generate_letter_aliases(
    num_aliases: int,
    tokenizer: PreTrainedTokenizerBase,
    n_tokens: int = 3,
    rng: random.Random | None = None,
) -> list[str]:
    """Random lowercase strings like 'sjdhf' that are exactly n_tokens long."""
    if rng is None:
        rng = random.Random(0)

    aliases: list[str] = []
    seen: set[str] = set()
    attempts, max_attempts = 0, num_aliases * 1000

    while len(aliases) < num_aliases and attempts < max_attempts:
        attempts += 1
        length = rng.randint(n_tokens, 3 * n_tokens)
        candidate = "".join(rng.choices(string.ascii_lowercase, k=length))
        if candidate in seen:
            continue
        seen.add(candidate)
        if token_len(tokenizer, candidate) == n_tokens:
            aliases.append(candidate)

    if len(aliases) < num_aliases:
        raise RuntimeError(
            f"Only generated {len(aliases)} of {num_aliases} letter aliases"
        )
    return aliases


def load_word_list(path: str) -> list[str]:
    """Lowercase, purely-alphabetic words from a one-word-per-line file."""
    with open(path, "r") as f:
        words = [line.strip().lower() for line in f]
    return [w for w in words if w.isalpha() and w not in STOP_WORDS]


def categorize_by_tokens(
    words: list[str], tokenizer: PreTrainedTokenizerBase
) -> dict[int, list[str]]:
    """Split words into pools that are exactly 1 or 2 tokens long."""
    pools: dict[int, list[str]] = {1: [], 2: []}
    for w in words:
        length = token_len(tokenizer, w)
        if length in pools:
            pools[length].append(w)
    return pools


def generate_phrase_aliases(
    num_aliases: int,
    tokenizer: PreTrainedTokenizerBase,
    adjective_path: str,
    noun_path: str,
    n_tokens: int = 5,
    rng: random.Random | None = None,
) -> list[str]:
    """Adjective(s)-noun phrases like 'prickly cyan mouse', n_tokens long.

    Sampled compositionally (word token lengths summing to the target), then
    verified on the joined phrase since tokenization can merge across spaces.
    """
    if rng is None:
        rng = random.Random(0)
    if n_tokens < 2:
        raise ValueError("Need at least 2 tokens (1 adjective + 1 noun)")

    adj_pools = categorize_by_tokens(load_word_list(adjective_path), tokenizer)
    noun_pools = categorize_by_tokens(load_word_list(noun_path), tokenizer)
    if not (adj_pools[1] and noun_pools[1]):
        raise RuntimeError("Word lists yielded no single-token words")

    aliases: list[str] = []
    seen: set[str] = set()
    attempts, max_attempts = 0, num_aliases * 1000

    while len(aliases) < num_aliases and attempts < max_attempts:
        attempts += 1

        # Pick the noun, then fill the remaining budget with adjectives
        noun_len = rng.choice([length for length in (1, 2) if noun_pools[length]])
        budget = n_tokens - noun_len
        if budget < 1:
            continue
        adj_lens = []
        while budget > 0:
            length = rng.choice((1, 2)) if budget >= 2 else 1
            adj_lens.append(length)
            budget -= length

        adjs = [rng.choice(adj_pools[length]) for length in adj_lens]
        candidate = " ".join(adjs + [rng.choice(noun_pools[noun_len])])

        if candidate in seen:
            continue
        seen.add(candidate)
        if token_len(tokenizer, candidate) == n_tokens:
            aliases.append(candidate)

    if len(aliases) < num_aliases:
        warnings.warn(
            f"Only generated {len(aliases)} of {num_aliases} phrase aliases",
            RuntimeWarning,
        )
        raise RuntimeError(
            f"Only generated {len(aliases)} of {num_aliases} phrase aliases"
        )
    return aliases

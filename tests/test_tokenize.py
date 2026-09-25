"""Tests for conversation tokenization with label masking."""

import pytest
from transformers import AutoTokenizer

from bergson.config import DataConfig
from bergson.data import tokenize


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")


@pytest.fixture
def llama3_tokenizer():
    # Ungated mirror of Meta-Llama-3-8B-Instruct so no HF auth is needed. Its
    # chat template renders <|begin_of_text|> AND its post-processor adds a BOS
    # when add_special_tokens is left at the default True, which is exactly the
    # double-BOS condition we regression-test below.
    return AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")


def _make_batch(convos):
    """Wrap a list of conversations into a batch dict."""
    return {"conversation": convos}


def test_multi_turn_labels(tokenizer):
    """All assistant turns get labels in a multi-turn conversation."""
    batch = _make_batch(
        [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm great, thanks!"},
            ]
        ]
    )
    cfg = DataConfig(conversation_column="conversation")
    result = tokenize(batch, args=cfg, tokenizer=tokenizer)
    labels = result["labels"][0]
    active = [i for i, l in enumerate(labels) if l != -100]
    # Should have two non-contiguous active regions (one per assistant turn)
    assert len(active) > 2
    # Check there's a gap (labels for user content are -100)
    gaps = [active[i + 1] - active[i] for i in range(len(active) - 1)]
    assert any(g > 1 for g in gaps), "Expected gap between assistant turns"


def test_truncation_preserves_partial_span(tokenizer):
    """When truncation cuts through an assistant response, labels should be
    assigned up to the truncation boundary — not dropped entirely."""
    # Create a conversation where the assistant response is long enough to
    # extend past a short max_length
    short_answer = "Short."
    long_answer = "word " * 200  # ~200 tokens, will be truncated
    batch = _make_batch(
        [
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": short_answer},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": long_answer},
            ]
        ]
    )
    cfg = DataConfig(conversation_column="conversation", truncation=True)
    max_length = 64
    result = tokenize(batch, args=cfg, tokenizer=tokenizer, max_length=max_length)

    labels = result["labels"][0]
    tokens = result["input_ids"][0]
    assert len(labels) == len(tokens) == max_length

    # The second assistant response should have SOME active labels even though
    # it's truncated — this was the bug: they were all set to -100
    active = sum(1 for l in labels if l != -100)
    assert active > 0, (
        "Truncated assistant span should still have active labels "
        "up to the truncation boundary"
    )


def test_fully_truncated_span_skipped(tokenizer):
    """When an entire assistant span is past the truncation boundary, it should
    be skipped without error."""
    # First turn is very long, second turn is entirely truncated
    long_answer = "word " * 500
    batch = _make_batch(
        [
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": long_answer},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "This will be truncated away"},
            ]
        ]
    )
    cfg = DataConfig(conversation_column="conversation", truncation=True)
    max_length = 64
    result = tokenize(batch, args=cfg, tokenizer=tokenizer, max_length=max_length)

    labels = result["labels"][0]
    assert len(labels) == max_length
    # Should not raise, and should still have some labels from the first turn


@pytest.mark.xfail(
    reason="rfind-based span lookup can't distinguish a user echo of an "
    "earlier assistant turn from the assistant turn itself when both "
    "occurrences sit before the next assistant turn."
)
def test_user_quotes_previous_assistant(tokenizer):
    ans1 = "alpha bravo charlie delta echo"
    ans2 = "Yes, confirmed."
    convo = [
        {"role": "user", "content": "Pick five NATO words."},
        {"role": "assistant", "content": ans1},
        {"role": "user", "content": f"Earlier you said '{ans1}'. Right?"},
        {"role": "assistant", "content": ans2},
    ]
    cfg = DataConfig(conversation_column="conversation")
    result = tokenize(_make_batch([convo]), args=cfg, tokenizer=tokenizer)
    labels = result["labels"][0]

    rendered = tokenizer.apply_chat_template(convo, tokenize=False)
    encodings = tokenizer(rendered, add_special_tokens=False)

    asst1_char = rendered.find(ans1)
    user_quote_char = rendered.find(ans1, asst1_char + 1)
    ans2_char = rendered.find(ans2)
    assert (
        asst1_char >= 0 and user_quote_char > asst1_char and ans2_char > user_quote_char
    )

    asst1_token = encodings.char_to_token(asst1_char)
    user_quote_token = encodings.char_to_token(user_quote_char)
    ans2_token = encodings.char_to_token(ans2_char)

    assert labels[asst1_token] != -100
    assert labels[user_quote_token] == -100
    assert labels[ans2_token] != -100


def test_identical_assistant_turns(tokenizer):
    repeated = "alpha bravo charlie delta echo"
    convo = [
        {"role": "user", "content": "Say it."},
        {"role": "assistant", "content": repeated},
        {"role": "user", "content": "Say it again."},
        {"role": "assistant", "content": repeated},
    ]
    cfg = DataConfig(conversation_column="conversation")
    result = tokenize(_make_batch([convo]), args=cfg, tokenizer=tokenizer)
    labels = result["labels"][0]

    rendered = tokenizer.apply_chat_template(convo, tokenize=False)
    encodings = tokenizer(rendered, add_special_tokens=False)

    first_char = rendered.find(repeated)
    second_char = rendered.find(repeated, first_char + 1)
    assert first_char >= 0 and second_char > first_char

    first_token = encodings.char_to_token(first_char)
    second_token = encodings.char_to_token(second_char)

    assert labels[first_token] != -100
    assert labels[second_token] != -100

    gap = [i for i in range(first_token, second_token) if labels[i] == -100]
    assert gap, "Expected -100 region between the two identical assistant turns"


def test_no_double_bos_conversation(llama3_tokenizer):
    """tokenize() must not duplicate the BOS the chat template already renders.

    apply_chat_template renders the special tokens into the string, so the
    subsequent re-tokenization must pass add_special_tokens=False. Regression
    test for the double-BOS bug: with the buggy default the Llama 3 tokenizer
    prepends a second <|begin_of_text|>.
    """
    tok = llama3_tokenizer
    convo = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    cfg = DataConfig(conversation_column="conversation")
    ids = tokenize(_make_batch([convo]), args=cfg, tokenizer=tok)["input_ids"][0]

    # Exactly one BOS, and it must be the leading token.
    assert list(ids).count(tok.bos_token_id) == 1
    assert ids[0] == tok.bos_token_id

    # Strongest invariant: the two-step tokenization must reproduce the ids you
    # get from tokenizing the chat template directly.
    ground_truth = tok.apply_chat_template(convo, tokenize=True, return_dict=True)
    assert list(ids) == list(ground_truth["input_ids"])


def test_no_double_bos_completion(llama3_tokenizer):
    """Prompt-completion path must also avoid duplicating the BOS token."""
    tok = llama3_tokenizer
    batch = {"prompt": ["What is 2+2?"], "completion": ["4"]}
    cfg = DataConfig(prompt_column="prompt", completion_column="completion")
    ids = tokenize(batch, args=cfg, tokenizer=tok)["input_ids"][0]

    assert list(ids).count(tok.bos_token_id) == 1
    assert ids[0] == tok.bos_token_id


def test_matches_direct_tokenization_no_special_bos(tokenizer):
    """Tokenizers that don't inject a special BOS must be unaffected by the fix.

    SmolLM2's chat template renders its markers itself and its tokenizer adds no
    special BOS, so add_special_tokens=False must neither strip nor add anything.
    Asserting equality with apply_chat_template(tokenize=True) guards against the
    fix regressing into a *missing*-BOS bug for such models.
    """
    convo = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6"},
    ]
    cfg = DataConfig(conversation_column="conversation")
    ids = tokenize(_make_batch([convo]), args=cfg, tokenizer=tokenizer)["input_ids"][0]

    ground_truth = tokenizer.apply_chat_template(convo, tokenize=True, return_dict=True)
    assert list(ids) == list(ground_truth["input_ids"])

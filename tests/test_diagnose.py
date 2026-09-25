"""Tests for the tokenizer-correctness diagnostic in bergson.diagnose.

These exercise diagnose_special_tokens() directly (tokenizer-only, no GPU): it
passes when tokenize() matches the model's expected tokenization and fails when
tokenize() duplicates or drops special tokens.
"""

from bergson.diagnose import diagnose_special_tokens


def test_special_tokens_pass_for_correct_tokenize():
    # Renders <|begin_of_text|> in the template.
    assert diagnose_special_tokens("NousResearch/Meta-Llama-3-8B-Instruct") is True


def test_special_tokens_catches_double_bos(monkeypatch):
    """The check must FAIL when tokenize() duplicates the BOS token."""
    import bergson.diagnose as diag

    def double_bos_tokenize(batch, *, args, tokenizer, max_length=None):
        # Re-add special tokens on top of a chat template that already has them.
        strings = tokenizer.apply_chat_template(
            [batch[args.conversation_column][0]], tokenize=False
        )
        enc = tokenizer(strings)  # add_special_tokens defaults to True
        return {"input_ids": enc["input_ids"]}

    monkeypatch.setattr(diag, "tokenize", double_bos_tokenize)
    assert diagnose_special_tokens("NousResearch/Meta-Llama-3-8B-Instruct") is False

"""Diagnostic tests for numerical stability and tokenizer correctness.

Includes:
- Gradient consistency across padding and batch composition
- Special token (BOS/EOS) duplication detection for chat templates
"""

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from simple_parsing import field
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.config import DataConfig
from bergson.data import pad_and_tensor, tokenize
from bergson.utils.utils import get_device, simple_parse_kwargs_string


@dataclass
class DiagnoseConfig:
    """Config for the numerical stability test."""

    model: str = "EleutherAI/pythia-160m"
    """HuggingFace model to test."""

    dataset: str = "NeelNanda/pile-10k"
    """Dataset to sample document pairs from."""

    split: str = "train"
    """Dataset split."""

    n_trials: int = 100
    """Number of random document pairs to test per configuration."""

    seed: int = 42
    """Random seed for reproducibility."""

    precision: str = field(
        default="bf16", metadata=dict(choices=["bf16", "fp16", "fp32"])
    )
    """Base precision for model parameters."""

    device: str = field(default_factory=get_device)
    """Device to run the test on."""

    max_len: int = 512
    """Truncate documents longer than this."""

    min_len: int = 4
    """Skip documents shorter than this."""

    threshold: float = 0.99
    """Cosine similarity below this is flagged as problematic."""

    model_kwargs: str = ""
    """Extra kwargs forwarded to ``from_pretrained`` as ``a=b,c=d`` (e.g.
    ``trust_remote_code=True`` for custom model architectures)."""


DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _get_example_loss(model, x, y, idx=0):
    """Get loss for example `idx` in the batch."""
    logits = model(x).logits[:, :-1]
    masks = y[:, 1:] != -100
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y[:, 1:].flatten(),
        reduction="none",
    ).reshape_as(y[:, 1:])
    return per_token[idx][masks[idx]].sum()


def _measure(model, short_ids, long_ids, device):
    """Measure gradient cosine similarity: alone vs mixed batch."""
    x_alone, y_alone, _, _ = pad_and_tensor([short_ids], device=device)
    x_mixed, y_mixed, _, _ = pad_and_tensor([short_ids, long_ids], device=device)

    # Pass 1: alone
    model.zero_grad()
    loss_alone = _get_example_loss(model, x_alone, y_alone)
    loss_alone.backward()
    grads_alone = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            grads_alone[n] = p.grad.detach().clone()

    # Pass 2: mixed
    model.zero_grad()
    loss_mixed = _get_example_loss(model, x_mixed, y_mixed)
    loss_mixed.backward()

    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for n, p in model.named_parameters():
        if p.grad is not None and n in grads_alone:
            ga = grads_alone[n]
            gb = p.grad.detach()
            dot += (ga * gb).sum().item()
            norm_a_sq += (ga * ga).sum().item()
            norm_b_sq += (gb * gb).sum().item()

    del grads_alone
    cos_sim = dot / (norm_a_sq**0.5 * norm_b_sq**0.5 + 1e-12)
    loss_diff = (loss_mixed - loss_alone).abs().item()

    return cos_sim, loss_diff


def _run_trials(model, all_docs, n_trials, seed, threshold, device):
    """Run n_trials gradient consistency checks with different-length pairs.

    Returns (min_cos_sim, n_flagged, results).
    """
    rng = random.Random(seed)
    results = []

    for trial in range(n_trials):
        i, j = rng.sample(range(len(all_docs)), 2)
        doc_a, doc_b = all_docs[i], all_docs[j]
        if len(doc_a) > len(doc_b):
            doc_a, doc_b = doc_b, doc_a

        cos_sim, loss_diff = _measure(model, doc_a, doc_b, device)
        results.append(
            {
                "trial": trial,
                "short_len": len(doc_a),
                "long_len": len(doc_b),
                "ratio": len(doc_b) / len(doc_a),
                "cos_sim": cos_sim,
                "loss_diff": loss_diff,
            }
        )

    cos_sims = torch.tensor([r["cos_sim"] for r in results])
    n_flagged = int((cos_sims < threshold).sum().item())
    return cos_sims.min().item(), n_flagged, results


def _run_equal_length_trials(
    model, all_docs, n_trials, seed, threshold, device, tokenizer
):
    """Run n_trials with equal-length pairs (no padding).

    Batches each document alongside random tokens of the same length.
    This isolates whether divergence comes from padding or from batching itself.

    Returns (min_cos_sim, n_flagged, results).
    """
    rng = random.Random(seed)
    vocab_size = tokenizer.vocab_size
    results = []

    for trial in range(n_trials):
        doc = all_docs[rng.randrange(len(all_docs))]
        # Generate random tokens of the same length (no padding needed)
        random_ids = [rng.randrange(vocab_size) for _ in range(len(doc))]

        cos_sim, loss_diff = _measure(model, doc, random_ids, device)
        results.append(
            {
                "trial": trial,
                "short_len": len(doc),
                "long_len": len(random_ids),
                "ratio": 1.0,
                "cos_sim": cos_sim,
                "loss_diff": loss_diff,
            }
        )

    cos_sims = torch.tensor([r["cos_sim"] for r in results])
    n_flagged = int((cos_sims < threshold).sum().item())
    return cos_sims.min().item(), n_flagged, results


def _print_results(results, threshold):
    """Print detailed trial results and summary."""
    cos_sims = torch.tensor([r["cos_sim"] for r in results])
    n_flagged = int((cos_sims < threshold).sum().item())

    header = (
        f"{'trial':>6s} {'short':>6s} {'long':>6s} {'ratio':>6s}"
        f" {'cos_sim':>10s} {'loss_diff':>12s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        marker = " <<<" if r["cos_sim"] < threshold else ""
        print(
            f"{r['trial']:>6d} {r['short_len']:>6d} {r['long_len']:>6d}"
            f" {r['ratio']:>6.1f} {r['cos_sim']:>10.6f}"
            f" {r['loss_diff']:>12.6f}{marker}"
        )

    print()
    print(
        f"  Cos sim:  mean={cos_sims.mean():.6f}  std={cos_sims.std():.6f}"
        f"  min={cos_sims.min():.6f}  max={cos_sims.max():.6f}"
    )
    print(f"  Flagged:  {n_flagged}/{len(results)} trials below {threshold}")


def diagnose_special_tokens(model_name: str):
    """Check that ``tokenize()`` emits special tokens correctly for a model.

    Tokenizes a sample conversation with ``tokenize()`` and compares the result
    against ``apply_chat_template(tokenize=True)``, the tokenization the model
    expects. Flags duplicated (double BOS) or missing special tokens.

    Returns True if the check passes, False otherwise.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    print(f"\n{'=' * 60}")
    print(f"Special token check: {model_name}")
    print("=" * 60)
    print(f"  bos_token: {tokenizer.bos_token!r} (id={bos_id})")
    print(f"  eos_token: {tokenizer.eos_token!r} (id={eos_id})")

    # Use a simple conversation to test the chat template
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    # The tokenization the model expects; also probes for a chat template.
    try:
        ground_truth = list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
            )["input_ids"]
        )
    except Exception:
        print("\n  No chat template found — special token check not applicable.")
        return True

    # What tokenize() produces for the same conversation.
    try:
        result = tokenize(
            {"conversation": [messages]},
            args=DataConfig(conversation_column="conversation"),
            tokenizer=tokenizer,
        )
    except Exception as e:
        print(f"\n  FAIL: bergson tokenize() raised {type(e).__name__}: {e}")
        return False
    actual = list(result["input_ids"][0])

    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    print("\n  Template string (first 200 chars):")
    print(f"    {rendered[:200]!r}")

    if actual == ground_truth:
        print(
            "\n  PASS: bergson tokenize() matches"
            " apply_chat_template(tokenize=True)."
        )
        return True

    print(
        "\n  FAIL: bergson tokenize() does not match"
        " apply_chat_template(tokenize=True)."
    )
    print(f"    bergson length: {len(actual)}, expected: {len(ground_truth)}")

    # Explain the failure in the usual BOS/EOS terms.
    seen: set[int] = set()
    for name, tid in (("BOS", bos_id), ("EOS", eos_id)):
        if tid is None or tid in seen:
            continue
        seen.add(tid)
        n_actual = actual.count(tid)
        n_expected = ground_truth.count(tid)
        if n_actual == n_expected:
            continue
        kind = "DUPLICATED" if n_actual > n_expected else "MISSING"
        print(
            f"    <<< {kind} {name} (id={tid}): bergson has {n_actual},"
            f" expected {n_expected}"
        )

    min_len = min(len(actual), len(ground_truth))
    first_diff = next(
        (i for i in range(min_len) if actual[i] != ground_truth[i]),
        min_len,
    )
    print(f"    First divergence at position {first_diff}:")
    if first_diff < min_len:
        print(
            f"      bergson[{first_diff}]  = {actual[first_diff]}"
            f" ({tokenizer.decode([actual[first_diff]])!r})"
        )
        print(
            f"      expected[{first_diff}] = {ground_truth[first_diff]}"
            f" ({tokenizer.decode([ground_truth[first_diff]])!r})"
        )
    return False


def diagnose(diagnose_cfg: DiagnoseConfig):
    """Run all diagnostic tests: special tokens + gradient consistency."""
    device = torch.device(diagnose_cfg.device if torch.cuda.is_available() else "cpu")

    print(f"Model:     {diagnose_cfg.model}")
    print(f"Precision: {diagnose_cfg.precision}")
    print(f"Device:    {device}")
    print(f"Trials:    {diagnose_cfg.n_trials} per configuration")
    print(f"Threshold: {diagnose_cfg.threshold}")

    # ── Special token check ──────────────────────────────────────────────
    special_tokens_ok = diagnose_special_tokens(diagnose_cfg.model)

    # Load and tokenize dataset
    print(f"\nLoading {diagnose_cfg.dataset}...")
    tokenizer = AutoTokenizer.from_pretrained(diagnose_cfg.model)
    ds = load_dataset(diagnose_cfg.dataset, split=diagnose_cfg.split)
    all_docs = []
    for row in ds:
        assert isinstance(row, dict)
        ids = tokenizer(row["text"])["input_ids"]
        if len(ids) < diagnose_cfg.min_len:
            continue
        if len(ids) > diagnose_cfg.max_len:
            ids = ids[: diagnose_cfg.max_len]
        all_docs.append(ids)

    print(
        f"Documents: {len(all_docs)}"
        f" (lengths {diagnose_cfg.min_len}-{diagnose_cfg.max_len})"
    )

    base_precision = diagnose_cfg.precision
    base_dtype = DTYPE_MAP[base_precision]

    # Define configurations to test in order of escalation.
    # Each is (label, dtype, force_math_sdp, tf32_matmuls).
    # Escalation order: try cheap fixes first (tf32, math_sdp), combine them,
    # then fall back to full fp32 only if needed.
    configs: list[tuple[str, torch.dtype, bool, bool]] = [
        (f"precision={base_precision}", base_dtype, False, False),
    ]
    if base_precision != "fp32":
        configs.extend(
            [
                # Cheap fixes first
                ("--precision fp32 --use_tf32_matmuls", torch.float32, False, True),
                (
                    f"--force_math_sdp (precision={base_precision})",
                    base_dtype,
                    True,
                    False,
                ),
                # Combine cheap fixes
                (
                    "--precision fp32 --use_tf32_matmuls --force_math_sdp",
                    torch.float32,
                    True,
                    True,
                ),
                # Full fp32 as last resort
                ("--precision fp32 --force_math_sdp", torch.float32, True, False),
            ]
        )

    # ── Equal-length batch test (no padding) ────────────────────────────
    # Run once with defaults to isolate whether divergence is from padding
    # or from batching itself.
    print(f"\n{'=' * 60}")
    print("Testing: equal-length batching (no padding)")
    print("=" * 60)

    eq_model = AutoModelForCausalLM.from_pretrained(
        diagnose_cfg.model,
        torch_dtype=base_dtype,
        attn_implementation="sdpa",
        device_map={"": device},
        **simple_parse_kwargs_string(diagnose_cfg.model_kwargs),
    )
    eq_model.eval()

    eq_min, eq_flagged, eq_results = _run_equal_length_trials(
        eq_model,
        all_docs,
        diagnose_cfg.n_trials,
        diagnose_cfg.seed,
        diagnose_cfg.threshold,
        device,
        tokenizer,
    )
    _print_results(eq_results, diagnose_cfg.threshold)
    del eq_model
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ── Padding tests (escalating configurations) ─────────────────────
    config_results = {}  # label -> (n_flagged, min_cos_sim)
    passing_config = None

    for label, dtype, force_math_sdp, tf32_matmuls in configs:
        print(f"\n{'=' * 60}")
        print(f"Testing: {label}")
        print("=" * 60)

        # Reset SDPA backends before each config
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        if force_math_sdp:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        # Set matmul precision
        torch.set_float32_matmul_precision("high" if tf32_matmuls else "highest")

        model = AutoModelForCausalLM.from_pretrained(
            diagnose_cfg.model,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map={"": device},
            **simple_parse_kwargs_string(diagnose_cfg.model_kwargs),
        )
        model.eval()

        min_cos_sim, n_flagged, results = _run_trials(
            model,
            all_docs,
            diagnose_cfg.n_trials,
            diagnose_cfg.seed,
            diagnose_cfg.threshold,
            device,
        )

        _print_results(results, diagnose_cfg.threshold)
        config_results[label] = (n_flagged, min_cos_sim)

        del model
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if n_flagged == 0:
            passing_config = label
            break

    # Final report
    print(f"\n{'=' * 60}")
    print(f"Report for {diagnose_cfg.model}")
    print(f"  {diagnose_cfg.n_trials} trials per configuration")
    print("=" * 60)

    st_status = "PASS" if special_tokens_ok else "FAIL"
    print(f"  {st_status}  Special token check (chat template BOS/EOS)")

    eq_status = "PASS" if eq_flagged == 0 else "FAIL"
    print(f"  {eq_status}  Equal-length batching (min cos_sim={eq_min:.6f})")

    for label, (n_flagged, min_cos_sim) in config_results.items():
        status = "PASS" if n_flagged == 0 else "FAIL"
        print(f"  {status}  {label}  (min cos_sim={min_cos_sim:.6f})")

    print()
    first_label = list(config_results.keys())[0]
    first_n_flagged = config_results[first_label][0]

    if first_n_flagged == 0:
        print(
            "RESULT: Gradients are consistent with default settings."
            " No special flags needed."
        )
    elif passing_config is not None:
        # Extract the flags from the passing config label
        print("RESULT: Gradients require non-default settings for consistency.")
        print(f"  Minimum required: {passing_config}")
        # Build the recommended CLI flags
        flags = []
        for label, dtype, force_math_sdp, tf32_matmuls in configs:
            if label == passing_config:
                if force_math_sdp:
                    flags.append("--force_math_sdp")
                if dtype == torch.float32 and base_precision != "fp32":
                    flags.append("--precision fp32")
                if tf32_matmuls:
                    flags.append("--use_tf32_matmuls")
                break
        if flags:
            flag_str = " ".join(flags)
            print("\n  Add to your bergson commands:")
            print(
                f"    bergson build <run_path> --model {diagnose_cfg.model} {flag_str}"
            )
    else:
        print(
            "RESULT: Gradient inconsistency persists across all tested"
            " configurations. This model may have architecture-level"
            " padding sensitivity."
        )

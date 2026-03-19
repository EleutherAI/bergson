"""Test gradient consistency across padding and batch composition.

Loads a model and dataset, samples random pairs of documents with different
lengths, and checks whether the shorter document's gradient changes when
it is batched alongside a longer document (padding divergence).

Automatically tests escalating configurations (default → force_math_sdp →
fp32 → both) to determine the minimum settings needed for consistency.
"""

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from simple_parsing import field
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.data import pad_and_tensor


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

    device: str = "cuda:0"
    """Device to run the test on."""

    max_len: int = 512
    """Truncate documents longer than this."""

    min_len: int = 4
    """Skip documents shorter than this."""

    threshold: float = 0.99
    """Cosine similarity below this is flagged as problematic."""


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
    x_alone, y_alone, _ = pad_and_tensor([short_ids], device=device)
    x_mixed, y_mixed, _ = pad_and_tensor([short_ids, long_ids], device=device)

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
    """Run n_trials gradient consistency checks.

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


def diagnose(diagnose_cfg: DiagnoseConfig):
    """Run the gradient consistency test across escalating configurations."""
    device = torch.device(diagnose_cfg.device if torch.cuda.is_available() else "cpu")

    print(f"Model:     {diagnose_cfg.model}")
    print(f"Precision: {diagnose_cfg.precision}")
    print(f"Device:    {device}")
    print(f"Trials:    {diagnose_cfg.n_trials} per configuration")
    print(f"Threshold: {diagnose_cfg.threshold}")

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
    # Each is (label, dtype, force_math_sdp).
    # We skip configs that would be redundant (e.g. if base is already fp32).
    configs = [
        (f"defaults (precision={base_precision})", base_dtype, False),
        (f"--force_math_sdp (precision={base_precision})", base_dtype, True),
    ]
    if base_precision != "fp32":
        configs.append(("--precision fp32", torch.float32, False))
        configs.append(("--precision fp32 --force_math_sdp", torch.float32, True))

    config_results = {}  # label -> (n_flagged, min_cos_sim)
    passing_config = None

    for label, dtype, force_math_sdp in configs:
        print(f"\n{'=' * 60}")
        print(f"Testing: {label}")
        print("=" * 60)

        # Reset SDPA backends before each config
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        if force_math_sdp:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        model = AutoModelForCausalLM.from_pretrained(
            diagnose_cfg.model,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map={"": device},
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
        torch.cuda.synchronize()

        if n_flagged == 0:
            passing_config = label
            break

    # Final report
    print(f"\n{'=' * 60}")
    print(f"Report for {diagnose_cfg.model}")
    print("=" * 60)

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
        for label, dtype, force_math_sdp in configs:
            if label == passing_config:
                if force_math_sdp:
                    flags.append("--force_math_sdp")
                if dtype == torch.float32 and base_precision != "fp32":
                    flags.append("--precision fp32")
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

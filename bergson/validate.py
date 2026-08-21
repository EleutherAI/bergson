"""Leave-k-out validation of attribution scores.

The scores can come from any attribution method (MAGIC, EK-FAC, TrackStar,
SOURCE, ...): each leave-k-out subset is retrained (``validate_scores``) or
read from a pre-saved model bank (``evaluate_retrained``), and the query
loss increase is correlated against the summed attribution scores of the
left-out documents.
"""

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import PeftModel
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import (
    disable_progress_bar as hf_disable_pbar,
)
from transformers.utils.logging import (
    set_verbosity_error as hf_set_verbosity_error,
)

from .config.config import ValidationConfig
from .config.config_io import save_run_config
from .data import load_scores_loss_signed, pad_and_tensor
from .magic.data_stream import DataStream, pad_dataset_to_batch_size
from .magic.grad_accum import iter_microbatches
from .magic.trainer import TrainerState, prepare_trainer
from .utils.csv_writer import CSVWriter
from .utils.utils import get_device, simple_parse_kwargs_string
from .utils.worker_utils import setup_data_pipeline


def bank_loss_cache_key(
    run_cfg: ValidationConfig, multi_query: bool, num_subsets: int
) -> str:
    """Cache filename for a bank's per-subset query losses.

    The losses depend on the banked models, the query set, how the models are
    loaded, and (for the single-query token-mean) the batch grouping.
    ``num_subsets`` and ``multi_query`` determine the shape.
    """
    q = run_cfg.query
    identity = {
        "model": run_cfg.model,
        "model_kwargs": run_cfg.model_kwargs,
        "query_dataset": q.dataset,
        "query_split": q.split,
        "query_subset": q.subset,
        "query_prompt_column": q.prompt_column,
        "query_completion_column": q.completion_column,
        "query_truncation": q.truncation,
        "query_format_template": q.format_template,
        "query_data_kwargs": q.data_kwargs,
        "query_chunk_length": q.chunk_length,
        "batch_size": run_cfg.batch_size,
        "multi_query": multi_query,
        "num_subsets": num_subsets,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return f"losses_{digest}.pt"


def _load_banked_model(
    run_cfg: ValidationConfig, out_dir: str, device: torch.device | str
) -> torch.nn.Module:
    """Load a banked ``save_models`` checkpoint into a ready model."""
    load_kwargs = {"dtype": torch.float32, "attn_implementation": "eager"}
    load_kwargs.update(simple_parse_kwargs_string(run_cfg.model_kwargs))
    if os.path.isfile(os.path.join(out_dir, "adapter_config.json")):
        base = AutoModelForCausalLM.from_pretrained(run_cfg.model, **load_kwargs)
        model = PeftModel.from_pretrained(base, out_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(out_dir, **load_kwargs)
    return model.to(device)  # type: ignore


def per_doc_query_losses(
    model: torch.nn.Module,
    query_stream: DataStream,
    num_docs: int,
    grad_accum_steps: int = 1,
) -> torch.Tensor:
    """Compute the mean cross-entropy loss of each query document.

    Iterates over the query stream (sharded across ranks the same way training
    batches are), scatter-adds per-token losses into per-document sums via
    ``doc_ids``, and all-reduces so every rank returns the same ``[num_docs]``
    tensor. Rows without a ``doc_ids`` column are one document per row.
    """
    device = query_stream.device
    loss_sums = torch.zeros(num_docs, device=device)
    token_counts = torch.zeros(num_docs, device=device)
    ds = query_stream.dataset

    with torch.no_grad():
        for i in range(len(query_stream)):
            rows = query_stream.batch_rows(i)
            batch = ds[rows]
            x, y, _, _ = pad_and_tensor(
                batch["input_ids"], labels=batch.get("labels"), device=device
            )
            if "doc_ids" in batch:
                doc_ids = torch.tensor(batch["doc_ids"], device=device)
                doc_ids = doc_ids[:, : x.shape[1]]
            else:
                rows_t = torch.tensor(rows, device=device)
                doc_ids = rows_t[:, None].expand(-1, x.shape[1])

            inputs = {"input_ids": x, "labels": y, "doc_ids": doc_ids}
            for micro, outputs, _ in iter_microbatches(
                model, inputs, grad_accum_steps, model_keys=("input_ids",)
            ):
                shifted_labels = micro["labels"][:, 1:]
                token_loss = F.cross_entropy(
                    outputs.logits[:, :-1].flatten(0, 1).float(),
                    shifted_labels.flatten(),
                    reduction="none",
                    ignore_index=-100,
                ).view_as(shifted_labels)

                # A token's loss belongs to the document of the label token,
                # so packed rows attribute boundary positions to the next doc.
                mask = shifted_labels != -100
                ids = micro["doc_ids"][:, 1:][mask]
                loss_sums.scatter_add_(0, ids, token_loss[mask])
                token_counts.scatter_add_(
                    0, ids, torch.ones_like(ids, dtype=token_counts.dtype)
                )

    if dist.is_initialized():
        dist.all_reduce(loss_sums)
        dist.all_reduce(token_counts)

    return loss_sums / token_counts.clamp_min(1.0)


def mean_query_loss(
    model: torch.nn.Module,
    query_stream: DataStream,
    grad_accum_steps: int = 1,
) -> torch.Tensor:
    total = torch.zeros((), device=query_stream.device)
    with torch.no_grad():
        for batch in query_stream:
            batch.pop("example_weight", None)
            for _, outputs, coef in iter_microbatches(model, batch, grad_accum_steps):
                total += outputs.loss.detach() * coef
    return total / len(query_stream)


def report_multi_query_validation(
    run_path: str,
    diffs: list[list[float]],
    score_sums: list[list[float]],
    baselines: torch.Tensor,
    num_subsets: int,
    summary_name: str = "summary.csv",
):
    """Print per-query correlations and write one summary.csv row per query."""
    num_queries = len(diffs[0])

    summary_csv_writer = CSVWriter(
        os.path.join(run_path, summary_name),
        columns=[
            "query",
            "spearman_corr",
            "spearman_p",
            "pearson_corr",
            "pearson_p",
            "N",
            "baseline_loss",
        ],
    )
    rhos = []
    for q in range(num_queries):
        d = [row[q] for row in diffs]
        s = [row[q] for row in score_sums]
        sp = spearmanr(d, s)
        pe = pearsonr(d, s)
        rhos.append(sp.statistic)
        print(
            f"Query {q}: Spearman {sp.statistic:.4f} (p={sp.pvalue:.2e})  "
            f"Pearson {pe.statistic:.4f} (p={pe.pvalue:.2e})"
        )
        summary_csv_writer.writerow(
            q,
            sp.statistic,
            sp.pvalue,
            pe.statistic,
            pe.pvalue,
            num_subsets,
            float(baselines[q]),
        )
    summary_csv_writer.close()
    print(f"Mean Spearman across {num_queries} queries: {np.mean(rhos):.4f}")


def validate_scores(
    run_cfg: ValidationConfig,
    scores: torch.Tensor,
    multi_query: bool,
    *,
    global_rank: int,
    rank: int,
    world_size: int,
    schedule: Callable,
    stream: DataStream,
    query_stream: DataStream,
    fwd_state: TrainerState,
    model: torch.nn.Module,
    baseline: float,
    num_query_docs: int,
    query_weight_pad_count: int,
    pad_count: int,
    weight_pad_count: int,
):
    """Validate attribution scores via leave-subset-out retraining.

    Retrains from scratch with each subset's weights zeroed, evaluates the
    query loss diff against the ``baseline`` (full-data) loss, and correlates
    it with the subset's summed attribution scores. ``fwd_state``/``model``
    must still hold the fully-trained (no leave-out) weights on entry; they
    are used for the multi-query per-document baseline.
    """
    diffs = []
    score_sums = []

    num_real_query_docs = num_query_docs - query_weight_pad_count
    baseline_per_doc = torch.zeros(num_real_query_docs)
    if multi_query:
        if scores.shape[-1] != num_real_query_docs:
            raise ValueError(
                f"scores has {scores.shape[-1]} query columns but the query "
                f"dataset has {num_real_query_docs} documents; multi-query "
                "validation requires one score column per query document"
            )
        with fwd_state.activate(model):
            baseline_per_doc = per_doc_query_losses(
                model, query_stream, num_query_docs, run_cfg.grad_accum_steps
            )[:num_real_query_docs].cpu()
    elif scores.ndim == 2 and scores.shape[1] > 1:
        pass  # Per-token [docs, seq_len]; kept 2-D.
    else:
        assert scores.ndim == 1 or scores.shape[1] == 1
        scores = scores.flatten()

    # flat_scores rows are the leave-out units: documents, or
    # doc * seq_len + token positions for per-token scores.
    num_queries = scores.shape[-1] if multi_query else 1
    flat_scores = scores.reshape(-1, num_queries)

    if run_cfg.weight_lrs:
        # Gradient step on the data weights: retrain with w = 1 - lr * score
        # for each lr and compare the query loss change to the first-order
        # prediction lr * <s_q, s_step>.
        step_scores = flat_scores.mean(dim=1)
        predicted = flat_scores.mul(step_scores[:, None]).sum(dim=0)
        csv_path = os.path.join(run_cfg.run_path, "weight_step.csv")
        csv_writer = CSVWriter(
            csv_path,
            columns=(
                ["lr", "query", "diff", "predicted_diff"]
                if multi_query
                else ["lr", "diff", "predicted_diff"]
            ),
            enabled=global_rank == 0,
        )

        for lr in run_cfg.weight_lrs:
            trainer, fwd_state, model = prepare_trainer(run_cfg, rank, schedule)
            fwd_state.detach_()

            stream.weights.fill_(1.0)
            if pad_count:
                if stream.weights.ndim == 1:
                    stream.weights.data[-weight_pad_count:] = 0.0
                else:
                    stream.weights.data[-pad_count:] = 0.0
            flat_w = stream.weights.view(-1)
            flat_w[: len(step_scores)] -= lr * step_scores.to(flat_w)

            for x in stream:
                fwd_state = trainer.step(
                    fwd_state,
                    x,
                    inplace=True,
                    fsdp=run_cfg.fsdp,
                    max_grad_norm=run_cfg.max_grad_norm,
                    grad_accum_steps=run_cfg.grad_accum_steps,
                )

            if multi_query:
                with fwd_state.activate(model):
                    per_doc = per_doc_query_losses(
                        model,
                        query_stream,
                        num_query_docs,
                        run_cfg.grad_accum_steps,
                    )
                diff_vec = baseline_per_doc - per_doc[:num_real_query_docs].cpu()
                for q in range(num_real_query_docs):
                    csv_writer.writerow(
                        lr, q, diff_vec[q].item(), lr * predicted[q].item()
                    )
                if global_rank == 0:
                    print(
                        f"lr {lr:g}: mean diff {diff_vec.mean():.6f} "
                        f"(predicted {lr * predicted.mean().item():.6f})"
                    )
            else:
                with fwd_state.activate(model):
                    loss = mean_query_loss(
                        model, query_stream, run_cfg.grad_accum_steps
                    )
                if world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                diff = baseline - loss.item()
                csv_writer.writerow(lr, diff, lr * predicted.mean().item())
                if global_rank == 0:
                    print(
                        f"lr {lr:g}: diff {diff:.6f} "
                        f"(predicted {lr * predicted.mean().item():.6f})"
                    )

        csv_writer.close()
        if global_rank == 0:
            print(f"Saved weight-step validation to {csv_path}")
        return

    if run_cfg.exclude_zero_scores:
        valid_indices = torch.nonzero((flat_scores != 0).any(dim=1), as_tuple=True)[0]
    else:
        valid_indices = torch.arange(flat_scores.shape[0])

    subsets_path = run_cfg.subsets or os.path.join(run_cfg.run_path, "subsets.json")
    if os.path.exists(subsets_path):
        with open(subsets_path) as f:
            subsets = [torch.tensor(s, dtype=torch.long) for s in json.load(f)]
    else:
        rng = torch.Generator().manual_seed(run_cfg.seed)
        if run_cfg.subset_fraction > 0:
            # Draw potentially overlapping samples
            subset_size = max(1, round(run_cfg.subset_fraction * len(valid_indices)))

            subsets = [
                valid_indices[
                    torch.randperm(len(valid_indices), generator=rng)[:subset_size]
                ]
                for _ in range(run_cfg.num_subsets)
            ]
        else:
            # Draw non-overlapping samples
            perm = valid_indices[torch.randperm(len(valid_indices), generator=rng)]

            # Shuffle the order of the subsets so that the estimate of
            # correlation on the progress bar is unbiased. This does not change
            # the final correlation since all subsets are eventually evaluated,
            # but prevents the early subsets from being biased towards higher
            # or lower scores.
            subsets = list(perm.chunk(run_cfg.num_subsets))
            rng = random.Random(run_cfg.seed)
            rng.shuffle(subsets)

    start = run_cfg.subset_start
    stop = len(subsets) if run_cfg.subset_stop is None else run_cfg.subset_stop
    sliced = (start, stop) != (0, len(subsets))

    csv_name = f"validation_{start}_{stop}.csv" if sliced else "validation.csv"
    csv_path = os.path.join(run_cfg.run_path, csv_name)
    val_csv_writer = CSVWriter(
        csv_path,
        columns=(
            ["subset", "query", "diff", "score_sum"]
            if multi_query
            else ["subset", "diff", "score_sum"]
        ),
        enabled=global_rank == 0,
    )

    # Disable annoying repetitive model loading messages, even on rank 0
    hf_disable_pbar()
    hf_set_verbosity_error()

    # Optionally persist each retrained model for later attribution queries.
    save_models = run_cfg.save_models
    retrained_tokenizer = None
    if save_models and global_rank == 0:
        retrained_tokenizer = AutoTokenizer.from_pretrained(
            run_cfg.tokenizer or run_cfg.model
        )
        # Row i lists the doc ids left out of retrained/subset_i;
        # evaluate_retrained needs this to reuse the models. The first
        # process owns the shared files.
        if start == 0:
            with open(os.path.join(run_cfg.run_path, "subsets.json"), "w") as f:
                json.dump([s.tolist() for s in subsets], f)

    pbar = tqdm(subsets[start:stop], desc="Validating", disable=global_rank != 0)
    for i, subset in enumerate(pbar, start=start):
        trainer, fwd_state, model = prepare_trainer(run_cfg, rank, schedule)
        fwd_state.detach_()

        stream.weights.fill_(1.0)
        if pad_count:
            if stream.weights.ndim == 1:
                stream.weights.data[-weight_pad_count:] = 0.0
            else:
                stream.weights.data[-pad_count:] = 0.0
        stream.weights.view(-1)[subset] = run_cfg.subset_weight

        for x in stream:
            fwd_state = trainer.step(
                fwd_state,
                x,
                inplace=True,
                fsdp=run_cfg.fsdp,
                max_grad_norm=run_cfg.max_grad_norm,
                grad_accum_steps=run_cfg.grad_accum_steps,
            )

        if save_models and global_rank == 0:
            out_dir = os.path.join(run_cfg.run_path, "retrained", f"subset_{i}")
            os.makedirs(out_dir, exist_ok=True)
            with fwd_state.activate(model), torch.no_grad():
                model.save_pretrained(out_dir, safe_serialization=True)
            if retrained_tokenizer is not None:
                retrained_tokenizer.save_pretrained(out_dir)

        if multi_query:
            with fwd_state.activate(model):
                per_doc = per_doc_query_losses(
                    model,
                    query_stream,
                    num_query_docs,
                    run_cfg.grad_accum_steps,
                )
            diff_vec = baseline_per_doc - per_doc[:num_real_query_docs].cpu()
            score_sum_vec = flat_scores[subset].sum(dim=0)
            for q in range(num_real_query_docs):
                val_csv_writer.writerow(
                    i, q, diff_vec[q].item(), score_sum_vec[q].item()
                )

            if global_rank == 0:
                diffs.append(diff_vec.tolist())
                score_sums.append(score_sum_vec.tolist())

                if len(diffs) >= 2:
                    rhos = [
                        spearmanr(
                            [row[q] for row in diffs],
                            [row[q] for row in score_sums],
                        ).statistic
                        for q in range(num_real_query_docs)
                    ]
                    pbar.set_postfix({"mean_rho": float(np.mean(rhos))})
                else:
                    pbar.set_postfix({"mean_rho": "n/a"})
            continue

        with fwd_state.activate(model):
            loss = mean_query_loss(model, query_stream, run_cfg.grad_accum_steps)

        if world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        diff = baseline - loss.item()
        score_sum = flat_scores[subset].sum().item()
        val_csv_writer.writerow(i, diff, score_sum)

        if global_rank == 0:
            diffs.append(diff)
            score_sums.append(score_sum)

            if len(diffs) >= 2:
                sp = spearmanr(diffs, score_sums)
                pe = pearsonr(diffs, score_sums)
                pbar.set_postfix({"rho": sp.statistic, "r": pe.statistic})
            else:
                pbar.set_postfix({"rho": "n/a", "r": "n/a"})

    val_csv_writer.close()
    if global_rank == 0:
        if multi_query:
            report_multi_query_validation(
                run_cfg.run_path,
                diffs,
                score_sums,
                baseline_per_doc,
                stop - start,
                summary_name=f"summary_{start}_{stop}.csv" if sliced else "summary.csv",
            )
            print(f"Saved validation data to {csv_path}")
            return

        sp = spearmanr(diffs, score_sums)
        pe = pearsonr(diffs, score_sums)
        print(f"Final Spearman correlation: {sp.statistic:.4f} (p={sp.pvalue:.2e})")
        print(f"Final Pearson correlation:  {pe.statistic:.4f} (p={pe.pvalue:.2e})")
        print(f"Saved validation data to {csv_path}")

        summary_csv_writer = CSVWriter(
            os.path.join(run_cfg.run_path, "summary.csv"),
            columns=[
                "spearman_corr",
                "spearman_p",
                "pearson_corr",
                "pearson_p",
                "N",
                "baseline_loss",
            ],
        )
        summary_csv_writer.writerow(
            sp.statistic, sp.pvalue, pe.statistic, pe.pvalue, len(subsets), baseline
        )


def evaluate_retrained(
    run_cfg: ValidationConfig,
    retrained_dir: str | list[str],
    *,
    score_path: str = "",
):
    """Evaluate a bank of pre-saved leave-k-out models on a query, no retraining.

    Reads models written by an earlier run with
    ``save_models=true`` and evaluates attribution scores. No training
    happens so evaluation is cheap.
    """
    assert score_path, "evaluate_retrained requires precomputed --scores"
    dirs = [
        Path(d)
        for d in ([retrained_dir] if isinstance(retrained_dir, str) else retrained_dir)
    ]
    src = dirs[0]
    subsets_path = src / "subsets.json"
    if not subsets_path.exists():
        raise FileNotFoundError(
            f"{subsets_path} not found; retrained_dir must point at a run "
            f"directory written with save_models=true"
        )

    run_path = Path(run_cfg.run_path)
    run_path.mkdir(parents=True, exist_ok=True)
    save_run_config(run_cfg, run_path)

    # Row i of subsets.json lists the doc ids left out of retrained/subset_i, so
    # scores[subsets[i]] is the summed attribution of exactly what model i drops.
    with open(subsets_path) as f:
        subset_lists = json.load(f)
    subsets = [torch.tensor(s, dtype=torch.long) for s in subset_lists]

    # Load per-query attribution scores (mirrors run_magic's score loading).
    scores, multi_query = load_scores_loss_signed(score_path)
    if not multi_query:
        assert (
            scores.ndim == 1 or scores.shape[1] == 1
        ), "evaluate_retrained expects per-doc (1D) scores"
        scores = scores.flatten()

    max_idx = max((int(s.max()) for s in subsets if len(s)), default=-1)
    if max_idx >= len(scores):
        raise ValueError(
            f"subsets.json references doc id {max_idx} but scores has only "
            f"{len(scores)} entries -- the query must be scored against the same "
            f"training set and seed used to build the retrained bank"
        )

    # Build the query stream on a single device (no distributed training here).
    device = get_device(0)
    query_ds, query_n = setup_data_pipeline(run_cfg, run_cfg.query)
    query_ds, query_n, q_pad, q_weight_pad = pad_dataset_to_batch_size(
        query_ds, run_cfg.batch_size, query_n, "Query", 0
    )
    if len(query_ds) < run_cfg.batch_size:
        raise ValueError(
            f"Query dataset has {len(query_ds)} examples, fewer than "
            f"batch_size={run_cfg.batch_size}. Use a larger query split or "
            f"smaller batch_size."
        )
    query_stream = DataStream(
        query_ds,
        run_cfg.batch_size,
        device=device,
        input_key=run_cfg.query.prompt_column,
        weight_shape=(query_n,),
    )
    if q_pad:
        query_stream.weights.data[-q_weight_pad:] = 0.0

    hf_disable_pbar()
    hf_set_verbosity_error()

    num_real_query_docs = query_n - q_weight_pad
    if multi_query and scores.shape[1] != num_real_query_docs:
        raise ValueError(
            f"scores has {scores.shape[1]} query columns but the query "
            f"dataset has {num_real_query_docs} documents; multi-query "
            "validation requires one score column per query document"
        )

    def query_loss(model: torch.nn.Module) -> float:
        """Mean query loss for an already-loaded model."""
        model.eval()
        return float(mean_query_loss(model, query_stream, run_cfg.grad_accum_steps))

    def query_losses_per_doc(model: torch.nn.Module) -> torch.Tensor:
        """Per-document query losses for an already-loaded model."""
        model.eval()
        return per_doc_query_losses(
            model, query_stream, query_n, run_cfg.grad_accum_steps
        )[:num_real_query_docs].cpu()

    def compute_bank_losses(
        models_root: Path,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Evaluate every banked model, returning baseline + per-subset losses.

        ``per_subset`` is ``[num_subsets, num_real_query_docs]`` for multi-query
        or ``[num_subsets]`` for single-query. The bank's no-leave-out model
        (``retrained/base``) gives the baseline; absent it the baseline stays 0
        (correlation-safe).
        """
        base_dir = models_root / "base"
        base_scalar = 0.0
        base_per_doc = torch.zeros(num_real_query_docs)
        if base_dir.exists():
            base = _load_banked_model(run_cfg, str(base_dir), device)
            if multi_query:
                base_per_doc = query_losses_per_doc(base)
            else:
                base_scalar = query_loss(base)
            del base

        if multi_query:
            per_subset = torch.zeros(len(subsets), num_real_query_docs)
        else:
            per_subset = torch.zeros(len(subsets))
        for i in tqdm(range(len(subsets)), desc="Evaluating bank"):
            model = _load_banked_model(
                run_cfg, str(models_root / f"subset_{i}"), device
            )
            if multi_query:
                per_subset[i] = query_losses_per_doc(model)
            else:
                per_subset[i] = query_loss(model)
            del model
        return base_scalar, base_per_doc, per_subset

    # Cache per-subset query losses for re-use with different attribution methods.
    per_dir = []
    for d in dirs:
        cache_path = (
            d
            / "query_loss_cache"
            / bank_loss_cache_key(run_cfg, multi_query, len(subsets))
        )
        if cache_path.exists():
            print(f"Reusing cached bank losses from {cache_path}")
            blob = torch.load(cache_path, map_location="cpu")
        else:
            base_scalar, base_per_doc, per_subset = compute_bank_losses(d / "retrained")
            blob = {
                "baseline": base_scalar,
                "baseline_per_doc": base_per_doc,
                "per_subset": per_subset,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(blob, cache_path)
            print(f"Saved bank losses to {cache_path}")
        per_dir.append(blob)

    baseline = float(np.mean([b["baseline"] for b in per_dir]))
    baseline_per_doc = torch.stack([b["baseline_per_doc"] for b in per_dir]).mean(0)
    per_subset_losses = torch.stack([b["per_subset"] for b in per_dir]).mean(0)

    if multi_query:
        print(f"Baseline per-query losses (no leave-out): {baseline_per_doc.tolist()}")
    else:
        print(f"Baseline query loss (no leave-out): {baseline}")

    start = run_cfg.subset_start
    stop = len(subsets) if run_cfg.subset_stop is None else run_cfg.subset_stop
    sliced = (start, stop) != (0, len(subsets))

    csv_name = f"validation_{start}_{stop}.csv" if sliced else "validation.csv"
    csv_path = os.path.join(run_cfg.run_path, csv_name)
    val_csv_writer = CSVWriter(
        csv_path,
        columns=(
            ["subset", "query", "diff", "score_sum"]
            if multi_query
            else ["subset", "diff", "score_sum"]
        ),
    )

    # Combine cached losses with attribution subset scores.
    diffs = []
    score_sums = []
    for i in range(len(subsets)):
        if multi_query:
            diff_vec = baseline_per_doc - per_subset_losses[i]
            score_sum_vec = scores[subsets[i]].sum(dim=0)
            for q in range(num_real_query_docs):
                val_csv_writer.writerow(
                    i, q, diff_vec[q].item(), score_sum_vec[q].item()
                )
            diffs.append(diff_vec.tolist())
            score_sums.append(score_sum_vec.tolist())
        else:
            diff = baseline - float(per_subset_losses[i])
            score_sum = scores[subsets[i]].sum().item()
            val_csv_writer.writerow(i, diff, score_sum)
            diffs.append(diff)
            score_sums.append(score_sum)

    val_csv_writer.close()
    print(f"Saved validation data to {csv_path}")

    if multi_query:
        report_multi_query_validation(
            run_cfg.run_path, diffs, score_sums, baseline_per_doc, len(subsets)
        )
        return

    sp = spearmanr(diffs, score_sums)
    pe = pearsonr(diffs, score_sums)
    print(
        f"Spearman: {sp.statistic:.4f} (p={sp.pvalue:.2e})  "
        f"Pearson: {pe.statistic:.4f} (p={pe.pvalue:.2e})"
    )
    summary_csv_writer = CSVWriter(
        os.path.join(run_cfg.run_path, "summary.csv"),
        columns=[
            "spearman_corr",
            "spearman_p",
            "pearson_corr",
            "pearson_p",
            "N",
            "baseline_loss",
        ],
    )
    summary_csv_writer.writerow(
        sp.statistic, sp.pvalue, pe.statistic, pe.pvalue, len(subsets), baseline
    )
    summary_csv_writer.close()

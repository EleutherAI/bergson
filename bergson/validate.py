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
from collections.abc import Sequence
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
from .config.config_io import get_config_field, save_run_config
from .data import load_scores_loss_signed, pad_and_tensor
from .magic.data_stream import DataStream, mask_padded_rows, pad_dataset_to_batch_size
from .magic.grad_accum import loss_denom, split_batch
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
            for micro in split_batch(inputs, grad_accum_steps):
                logits = model(input_ids=micro["input_ids"]).logits
                shifted_labels = micro["labels"][:, 1:]
                token_loss = F.cross_entropy(
                    logits[:, :-1].flatten(0, 1).float(),
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
    model: torch.nn.Module, query_stream: DataStream, grad_accum_steps: int = 1
) -> torch.Tensor:
    """Mean loss over the query stream, reduced across ranks."""
    total = torch.zeros((), device=query_stream.device)
    tokens = torch.zeros((), device=query_stream.device)
    with torch.no_grad():
        for batch in query_stream:
            batch, n_tokens = mask_padded_rows(batch)
            tokens += n_tokens
            for micro in split_batch(batch, grad_accum_steps):
                total += model(**micro).loss * loss_denom(micro)
    if dist.is_initialized():
        dist.all_reduce(total)
        dist.all_reduce(tokens)
    return total / tokens


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


def load_and_validate_subsets_match(
    run_cfg: ValidationConfig, dirs: list[Path], num_filtered: int
) -> list[torch.Tensor]:
    """Load a list of subsets filtered during re-training runs.
    Validate subsets and that they match."""
    first_dir_subsets: list[list[int]] | None = None
    for d in dirs:
        assert (d / "retrained" / "base").exists(), f"Retrain bank {d} not valid."
        with open(d / "subsets.json") as f:
            dir_subsets = json.load(f)

        if first_dir_subsets is None:
            first_dir_subsets = dir_subsets
        assert dir_subsets == first_dir_subsets, f"Bank {d} doesn't match others."

        sizes = sorted({len(x) for x in dir_subsets})
        # LDS chunks the pool, so its sizes differ by at most one.
        if max(abs(n - num_filtered) for n in sizes) > 1:
            raise ValueError(
                f"{d} removes {sizes} docs per subset but the filter removes "
                f"{num_filtered}; set subset_fraction to match."
            )
        for field, ours in [
            ("model", run_cfg.model),
            ("subset_weight", run_cfg.subset_weight),
            ("exclude_zero_scores", run_cfg.exclude_zero_scores),
        ]:
            theirs = get_config_field(d, field)
            if theirs is not None and theirs != ours:
                raise ValueError(
                    f"{d} was written with {field}={theirs!r} but this run uses "
                    f"{ours!r}; the bank is not a comparable baseline"
                )

    assert first_dir_subsets is not None
    return [torch.tensor(x, dtype=torch.long) for x in first_dir_subsets]


def load_bank_losses(
    run_cfg: ValidationConfig,
    dirs: list[Path],
    num_subsets: int,
    *,
    multi_query: bool,
    num_real_query_docs: int,
    device: torch.device | str,
    query_loss: Callable[[torch.nn.Module], float],
    query_losses_per_doc: Callable[[torch.nn.Module], torch.Tensor],
    write_cache: bool = True,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Query losses of every banked model. Averaged over ``dirs``.

    The baselines are 0 if ``retrained/base`` is not present - this
    is correlation-safe. Query losses are cached under each bank.
    """

    def compute(models_root: Path) -> tuple[float, torch.Tensor, torch.Tensor]:
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
            per_subset = torch.zeros(num_subsets, num_real_query_docs)
        else:
            per_subset = torch.zeros(num_subsets)
        for i in tqdm(range(num_subsets), desc="Evaluating bank"):
            model = _load_banked_model(
                run_cfg, str(models_root / f"subset_{i}"), device
            )
            if multi_query:
                per_subset[i] = query_losses_per_doc(model)
            else:
                per_subset[i] = query_loss(model)
            del model
        return base_scalar, base_per_doc, per_subset

    per_dir = []
    for d in dirs:
        cache_path = (
            d
            / "query_loss_cache"
            / bank_loss_cache_key(run_cfg, multi_query, num_subsets)
        )
        if cache_path.exists():
            print(f"Reusing cached bank losses from {cache_path}")
            blob = torch.load(cache_path, map_location="cpu")
        else:
            base_scalar, base_per_doc, per_subset = compute(d / "retrained")
            blob = {
                "baseline": base_scalar,
                "baseline_per_doc": base_per_doc,
                "per_subset": per_subset,
            }
            if write_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(blob, cache_path)
                print(f"Saved bank losses to {cache_path}")
        per_dir.append(blob)

    baseline = float(np.mean([b["baseline"] for b in per_dir]))
    baseline_per_doc = torch.stack([b["baseline_per_doc"] for b in per_dir]).mean(0)
    per_subset_losses = torch.stack([b["per_subset"] for b in per_dir]).mean(0)
    return baseline, baseline_per_doc, per_subset_losses


def _baseline_subsets(
    run_cfg: ValidationConfig, valid_indices: torch.Tensor, k: int
) -> list[torch.Tensor]:
    """Random removal sets of ``k`` documents to compare the filter against."""
    rng = torch.Generator().manual_seed(run_cfg.seed)
    return [
        valid_indices[torch.randperm(len(valid_indices), generator=rng)[:k]]
        for _ in range(run_cfg.num_subsets)
    ]


def _report_filter_baseline(
    run_path: str,
    method: str,
    filter_changes: torch.Tensor,
    random_changes: torch.Tensor,
    k: int,
    source: str,
):
    """Print and save the filter's loss change against the random filters'."""
    n, num_queries = random_changes.shape
    print(
        f"Random baseline ({n} subsets of {k} docs, {source}): "
        f"mean loss change {random_changes.mean():.6f}"
    )
    summary_path = os.path.join(run_path, "filter_summary.csv")
    summary = CSVWriter(
        summary_path,
        columns=[
            "query",
            "n_removed",
            "filter_change",
            "random_mean",
            "random_sd",
            "random_n",
            "rank",
        ],
    )
    for q in range(num_queries):
        col = random_changes[:, q].numpy()
        # Rank 1 is the largest loss increase, i.e. the most damaging removal.
        rank = 1 + int((col > filter_changes[q].item()).sum())
        sd = float(np.std(col, ddof=1)) if n > 1 else float("nan")
        print(
            f"Query {q}: {method} {filter_changes[q]:+.6f}  "
            f"random {col.mean():+.6f} +/- {sd:.6f}  rank {rank}/{n + 1}"
        )
        summary.writerow(q, k, filter_changes[q].item(), float(col.mean()), sd, n, rank)
    summary.close()
    print(f"Saved filter/baseline comparison to {summary_path}")


def _select_filter_slice(
    flat_scores: torch.Tensor,
    valid_indices: torch.Tensor,
    query_col: int,
    num_filtered: int,
    method: str,
) -> torch.Tensor:
    """Get row indices of the documents to remove for one query.

    ``load_scores_loss_signed`` signs scores so that **proponents are
    negative** (they reduce query loss). Proponents are therefore the
    *smallest* values and detractors the *largest* -- getting this backwards
    silently inverts the estimator, so it is asserted by unit test.
    """
    if method == "filter-proponents":
        largest = False
    elif method == "filter-detractors":
        largest = True
    else:
        raise ValueError(f"{method} is not a tail-filter method")

    col = flat_scores[valid_indices, query_col]
    chosen = torch.topk(col, min(num_filtered, len(col)), largest=largest).indices
    return valid_indices[chosen]


def tail_filter_retrain(
    run_cfg: ValidationConfig,
    flat_scores: torch.Tensor,
    multi_query: bool,
    *,
    global_rank: int,
    rank: int,
    schedule: Callable,
    stream: DataStream,
    query_stream: DataStream,
    baseline: float,
    baseline_per_doc: torch.Tensor,
    num_query_docs: int,
    num_queries: int,
    pad_count: int,
    weight_pad_count: int,
    retrained_dir: Sequence[str],
):
    """Retrain with one tail of the score ranking filtered out.

    Random filters removing the same number of documents run alongside it,
    retrained here or read from a bank; ``num_subsets = 0`` and no bank skips
    them. ``load_scores_loss_signed`` signs proponents negative (they reduce
    query loss), so a positive ``loss_change`` means the filter worsened query
    performance -- the opposite sign to the LDS ``diff`` column.
    """
    if run_cfg.exclude_zero_scores:
        valid_indices = torch.nonzero((flat_scores != 0).any(dim=1), as_tuple=True)[0]
    else:
        valid_indices = torch.arange(flat_scores.shape[0])

    pool = len(valid_indices)
    if run_cfg.subset_fraction == 0.0:
        if run_cfg.num_subsets <= 0:
            raise ValueError(
                "subset_fraction must be positive when num_subsets is 0: there is "
                "no leave-k-out chunk size for the filter to match"
            )
        run_cfg.subset_fraction = 1 / run_cfg.num_subsets
    num_filtered = max(1, round(run_cfg.subset_fraction * pool))

    baseline_vec = (
        baseline_per_doc[:num_queries] if multi_query else torch.tensor([baseline])
    )

    def eval_losses(model: torch.nn.Module) -> torch.Tensor:
        """Query losses of an activated model, ``[num_queries]`` on the CPU."""
        if multi_query:
            per_doc = per_doc_query_losses(
                model, query_stream, num_query_docs, run_cfg.grad_accum_steps
            )
            return per_doc[:num_queries].cpu()

        loss = mean_query_loss(model, query_stream, run_cfg.grad_accum_steps)
        return loss.reshape(1).cpu()

    def retrain_and_eval(removed: torch.Tensor) -> torch.Tensor:
        """Query losses after retraining with ``removed`` down-weighted."""
        trainer, fwd_state, model = prepare_trainer(run_cfg, rank, schedule)
        fwd_state.detach_()

        stream.weights.fill_(1.0)
        if pad_count:
            if stream.weights.ndim == 1:
                stream.weights.data[-weight_pad_count:] = 0.0
            else:
                stream.weights.data[-pad_count:] = 0.0
        stream.weights.view(-1)[removed] = run_cfg.subset_weight

        for x in stream:
            fwd_state = trainer.step(
                fwd_state,
                x,
                inplace=True,
                fsdp=run_cfg.fsdp,
                max_grad_norm=run_cfg.max_grad_norm,
                grad_accum_steps=run_cfg.grad_accum_steps,
            )

        with fwd_state.activate(model):
            return eval_losses(model)

    hf_disable_pbar()
    hf_set_verbosity_error()

    csv_path = os.path.join(run_cfg.run_path, f"{run_cfg.method.replace('-', '_')}.csv")
    filter_csv = CSVWriter(
        csv_path,
        columns=["query", "n_removed", "baseline_loss", "filtered_loss", "loss_change"],
        enabled=global_rank == 0,
    )

    filter_changes = torch.zeros(num_queries)
    pbar = tqdm(range(num_queries), desc=run_cfg.method, disable=global_rank != 0)
    for q in pbar:
        removed = _select_filter_slice(
            flat_scores, valid_indices, q, num_filtered, run_cfg.method
        )
        losses = retrain_and_eval(removed)

        base_loss = baseline_vec[q].item()
        filtered_loss = losses[q].item()
        filter_changes[q] = filtered_loss - base_loss
        filter_csv.writerow(
            q, len(removed), base_loss, filtered_loss, filter_changes[q].item()
        )
        if global_rank == 0:
            pbar.set_postfix(
                {"mean_loss_change": filter_changes[: q + 1].mean().item()}
            )

    filter_csv.close()
    if global_rank == 0:
        print(
            f"{run_cfg.method}: mean loss change {filter_changes.mean():.6f} "
            f"over {num_queries} quer{'y' if num_queries == 1 else 'ies'} "
            f"({num_filtered} of {pool} docs removed per query, "
            f"{num_filtered / pool:.3%})"
        )
        print(f"Saved tail-filter data to {csv_path}")

    # Load existing baseline if it exists - a random filter retrain.
    # Otherwise compute the baseline.
    dirs = [Path(d) for d in retrained_dir]
    if dirs:
        subsets = load_and_validate_subsets_match(run_cfg, dirs, num_filtered)
        bank_base, bank_base_per_doc, per_subset = load_bank_losses(
            run_cfg,
            dirs,
            len(subsets),
            multi_query=multi_query,
            num_real_query_docs=num_queries,
            device=stream.weights.device,
            query_loss=lambda m: float(eval_losses(m.eval())[0]),
            query_losses_per_doc=lambda m: eval_losses(m.eval()),
            write_cache=global_rank == 0,
        )
        random_baseline = (
            bank_base_per_doc if multi_query else torch.tensor([bank_base])
        )
        random_losses = per_subset.reshape(len(subsets), num_queries)
        source = "bank " + ", ".join(str(d) for d in dirs)
    elif run_cfg.num_subsets > 0:
        subsets = _baseline_subsets(run_cfg, valid_indices, num_filtered)
        if global_rank == 0:
            print(f"Retraining {len(subsets)} random subsets of {num_filtered} docs")
        random_baseline = baseline_vec
        random_losses = torch.stack(
            [
                retrain_and_eval(x)
                for x in tqdm(subsets, desc="random", disable=global_rank != 0)
            ]
        )
        source = "retrained here"
    else:
        if global_rank == 0:
            print(
                "No random baseline: set num_subsets > 0, or retrained_dir to a "
                "path to random-subset retrains"
            )
        return

    if global_rank != 0:
        return

    random_changes = random_losses - random_baseline
    random_csv = CSVWriter(
        os.path.join(run_cfg.run_path, "random_filter.csv"),
        columns=[
            "subset",
            "query",
            "n_removed",
            "baseline_loss",
            "filtered_loss",
            "loss_change",
        ],
    )
    for i, subset in enumerate(subsets):
        for q in range(num_queries):
            random_csv.writerow(
                i,
                q,
                len(subset),
                random_baseline[q].item(),
                random_losses[i, q].item(),
                random_changes[i, q].item(),
            )
    random_csv.close()

    _report_filter_baseline(
        run_cfg.run_path,
        run_cfg.method,
        filter_changes,
        random_changes,
        num_filtered,
        source,
    )


def validate_scores(
    run_cfg: ValidationConfig,
    scores: torch.Tensor,
    multi_query: bool,
    *,
    global_rank: int,
    rank: int,
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
    retrained_dir: Sequence[str] = (),
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

    if run_cfg.method != "lds":
        tail_filter_retrain(
            run_cfg,
            flat_scores,
            multi_query,
            global_rank=global_rank,
            rank=rank,
            schedule=schedule,
            stream=stream,
            query_stream=query_stream,
            baseline=baseline,
            baseline_per_doc=baseline_per_doc,
            num_query_docs=num_query_docs,
            num_queries=num_queries,
            pad_count=pad_count,
            weight_pad_count=weight_pad_count,
            retrained_dir=retrained_dir,
        )
        return

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

    baseline, baseline_per_doc, per_subset_losses = load_bank_losses(
        run_cfg,
        dirs,
        len(subsets),
        multi_query=multi_query,
        num_real_query_docs=num_real_query_docs,
        device=device,
        query_loss=query_loss,
        query_losses_per_doc=query_losses_per_doc,
    )
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

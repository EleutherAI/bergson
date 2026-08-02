import json
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from shutil import rmtree
from typing import Any, cast

import psutil
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.tensor  # noqa: F401 — register DTensor for torch.load
import torchopt
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, init_device_mesh
from torchopt.pytree import tree_flatten_with_path, tree_iter, tree_map
from torchopt.typing import GradientTransformation, OptState
from tqdm.auto import tqdm

from ..config.config import SaveMode, TrainingConfig
from ..data import sorted_checkpoints
from ..utils.utils import get_device
from ..utils.worker_utils import setup_model_and_peft
from .data_stream import DataStream
from .dtensor_patch import apply_dtensor_patch
from .fsdp import shallow_copy, simple_fsdp
from .grad_accum import (
    accumulate_grads,
    maybe_get_cuda_rng_state,
    maybe_set_cuda_rng_state,
    microbatch_step_vjp,
    rng_restore,
    rng_snapshot,
)
from .optim import muon
from .rtl_tqdm import RtlTqdm
from .swap import swap_parameters

LR_HISTORY_FILENAME = "log_history.json"
"""Per-step LRs in HF's ``log_history`` shape, written beside a run's
checkpoints."""


def write_lr_history(
    save_dir: str | Path, schedule: Callable[[int], float], num_steps: int
) -> Path:
    """Record per-step LRs beside the checkpoints, from the ``schedule`` the
    optimizer was built with."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    history = [
        {"step": step, "learning_rate": float(schedule(step))}
        for step in range(num_steps)
    ]
    path = save_dir / LR_HISTORY_FILENAME
    with open(path, "w") as f:
        json.dump(history, f)
    return path


@contextmanager
def suppress_c_stdout():
    """Suppress C-level stdout."""
    fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(fd, 1)
        os.close(fd)


class _ReplicatedAllReduceSum(torch.autograd.Function):
    """All-reduce-sum across ranks, differentiable to arbitrary order.

    Used inside a ``create_graph=True`` training step so the graph can be
    differentiated again during MAGIC's backward-through-training.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return _ReplicatedAllReduceSum.apply(grad_output, ctx.group), None


@dataclass
class SaveFuture:
    """Wraps a DCP async_save future, destroying the gloo process group in result().

    The group must be destroyed synchronously inside result() rather than in a
    done_callback, because concurrent.futures.Future notifies result() waiters
    before invoking callbacks — so a callback-based destroy races with the next
    save() call creating a new group, leaking gloo sockets.
    """

    fut: Future
    grp: dist.ProcessGroup | None
    debug_name: str = ""
    debug_pbar: RtlTqdm | tqdm | None = None

    def result(self):
        start = time.monotonic()
        result = self.fut.result()
        elapsed = time.monotonic() - start

        if self.debug_name and (not dist.is_initialized() or dist.get_rank() == 0):
            print_fn = self.debug_pbar.write if self.debug_pbar else print
            print_fn(f"Waiting for {self.debug_name} took {elapsed:.2f} seconds")

        if self.grp is not None:
            dist.destroy_process_group(self.grp)
            self.grp = None

        return result


def next_save_index(
    current: int, n: int, save_mode: SaveMode, save_interval: int = 0
) -> int:
    """Index of the checkpoint following the one saved at step `current`.

    `n` is the total number of training steps. The result is always strictly
    greater than `current`, so iterating this function enumerates the full
    checkpoint schedule for a run of `n` steps starting from step 0.
    """
    match save_mode:
        case "all":
            # Save every step
            return current + 1
        case "interval":
            if save_interval <= 0:
                raise ValueError("save_mode='interval' requires save_interval > 0.")
            return current + save_interval
        case "sqrt":
            chunk_size = math.isqrt(n)

            # If we're in the last chunk, save every step
            if current >= n - chunk_size:
                return current + 1

            # Otherwise, save sqrt(n) steps from now
            return current + chunk_size
        case "log":
            # Cut the remaining steps in half to get to the next save point
            return max(1, (n - current) // 2) + current
        case other:
            raise ValueError(f"Unsupported save mode: {other}")


@dataclass
class BackwardState:
    param_grads: dict[str, torch.Tensor]

    opt_grads: list[torch.Tensor]
    """PyTree of the same structure as the optimizer state, containing gradients for
    each of the optimizer state tensors."""

    weight_grads: torch.Tensor


@dataclass
class TrainerState:
    # Differentiable state
    params: dict[str, torch.Tensor]
    opt_state: OptState

    # Non-differentiable state
    buffers: dict[str, torch.Tensor]
    batch_index: int = 0
    cuda_rng_state: torch.Tensor = field(default_factory=maybe_get_cuda_rng_state)
    cpu_rng_state: torch.Tensor = field(default_factory=torch.random.get_rng_state)

    def copy_(self, other: "TrainerState"):
        for k in self.params.keys():
            self.params[k].copy_(other.params[k])
        for k in self.buffers.keys():
            self.buffers[k].copy_(other.buffers[k])

        # `opt_state` is a PyTree; copy it leaf by leaf into the existing tensors
        # to keep this method in-place. Non-tensor leaves are taken from `other`.
        def _copy_leaf(dst, src):
            if isinstance(dst, torch.Tensor):
                assert isinstance(src, torch.Tensor), "opt_state structure mismatch"
                dst.copy_(src)
                return dst

            return src

        self.opt_state = tree_map(_copy_leaf, self.opt_state, other.opt_state)

        self.batch_index = other.batch_index
        self.cuda_rng_state.copy_(other.cuda_rng_state)
        self.cpu_rng_state.copy_(other.cpu_rng_state)

    def to(self, device: torch.device | str) -> "TrainerState":
        """Snapshot this state onto `device`.

        Uses copy=True to enforce copy for CPU-to-CPU transfers, so stashed
        checkpoints survive in-place training steps.
        """
        params = {k: p.to(device, copy=True) for k, p in self.params.items()}
        buffers = {k: b.to(device, copy=True) for k, b in self.buffers.items()}
        opt_state = tree_map(
            lambda t: t.to(device, copy=True) if isinstance(t, torch.Tensor) else t,
            self.opt_state,
        )
        return TrainerState(
            params,
            opt_state,
            buffers,
            self.batch_index,
            self.cuda_rng_state.clone(),
            self.cpu_rng_state.clone(),
        )

    def load(self, path: str):
        """Load state from a checkpoint file."""
        dcp.load(
            self.state_dict(),
            checkpoint_id=path,
        )

    def save(self, path: str, debug_pbar: RtlTqdm | tqdm | None = None) -> SaveFuture:
        # Create a new process group so that we can overlap saves.
        if dist.is_initialized():
            with suppress_c_stdout():
                grp = dist.new_group(backend="gloo", group_desc=path)
            assert isinstance(grp, dist.ProcessGroup)
        else:
            grp = None

        state = {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in self.state_dict().items()
        }
        fut = dcp.async_save(
            state,
            checkpoint_id=path,
            process_group=grp,
        )
        assert isinstance(fut, Future)
        return SaveFuture(
            fut,
            grp,
            debug_name=path if debug_pbar is not None else "",
            debug_pbar=debug_pbar,
        )

    def detach_(self) -> "TrainerState":
        for k, p in self.params.items():
            self.params[k] = p.detach()

        def _detach_leaf(t):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                return t.detach()
            return t

        self.opt_state = tree_map(_detach_leaf, self.opt_state)
        return self

    @property
    def requires_grad(self) -> bool:
        p_val = any(p.requires_grad for p in self.params.values())
        opt_val = any(
            isinstance(t, torch.Tensor) and t.requires_grad
            for t in tree_iter(self.opt_state)
        )
        return p_val or opt_val

    @requires_grad.setter
    def requires_grad(self, value: bool):
        for p in self.params.values():
            p.requires_grad = value

        for t in tree_iter(self.opt_state):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                t.requires_grad = value

    def differentiable_tensors(self) -> list[torch.Tensor]:
        ps = list(self.params.values())
        os = [
            t
            for t in tree_iter(self.opt_state)
            if isinstance(t, torch.Tensor) and t.is_floating_point()
        ]
        return ps + os

    @contextmanager
    def activate(self, model: nn.Module):
        ambient = rng_snapshot()
        rng_restore((self.cpu_rng_state, self.cuda_rng_state))

        try:
            with swap_parameters(model, self.params, self.buffers, strict=True) as p:
                yield p
        finally:
            rng_restore(ambient)

    def state_dict(self) -> dict:
        # Convert to dict manually because dataclasses.asdict does a deep copy
        state = {
            **self.params,
            **self.buffers,
            ".batch_index": torch.tensor(self.batch_index),
            ".cuda_rng_state": self.cuda_rng_state,
            ".cpu_rng_state": self.cpu_rng_state,
        }

        # Flatten opt_state PyTree into the top-level dict with "opt_state/" prefix so
        # that it can be saved with DCP, which doesn't support nested structures.
        paths, elements, _ = tree_flatten_with_path(self.opt_state)
        str_paths = ["opt_state/" + ".".join(map(str, p)) for p in paths]
        opt_state = dict(zip(str_paths, elements))
        state.update(opt_state)

        return state

    def size_in_bytes(self) -> int:
        """Roughly, the amount of space needed to save the state dict."""
        state = self.state_dict()
        return sum(
            t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0
            for t in state.values()
        )


class Trainer:
    """Stateless, functional trainer for a model and optimizer."""

    @classmethod
    def initialize(
        cls,
        model: nn.Module,
        optimizer: GradientTransformation,
    ) -> tuple["Trainer", TrainerState]:
        """Convenience method for initializing the trainer and state."""
        # Create new tensor objects for the parameters and buffers to ensure that they
        # are not modified in place. Only trainable params go into the state; frozen
        # params stay in the nn.Module.
        params = shallow_copy(
            {
                k: v
                for k, v in model.named_parameters(remove_duplicate=False)
                if v.requires_grad
            }
        )
        buffers = shallow_copy(dict(model.named_buffers(remove_duplicate=False)))
        opt_state = optimizer.init(params)

        state = TrainerState(params, opt_state, buffers)
        return cls(model, optimizer), state

    def __init__(
        self,
        model: nn.Module,
        optimizer: GradientTransformation,
    ):
        # Move only trainable parameters to the meta device, leaving frozen params
        # on device so they don't need to be managed by TrainerState.
        for mod in model.modules():
            for p_name, param in list(mod.named_parameters(recurse=False)):
                if param.requires_grad:
                    mod.register_parameter(
                        p_name, nn.Parameter(param.data.to("meta"), requires_grad=True)
                    )

        self.model = model
        self.optimizer = optimizer

    def step(
        self,
        state: TrainerState,
        inputs: dict[str, Any],
        *,
        inplace: bool = False,
        trace: bool = False,
        fsdp: bool = False,
        max_grad_norm: float | None = None,
        grad_accum_steps: int = 1,
    ) -> TrainerState:
        """Perform a single training step on `state`, returning the new state.

        Args:
            state: The current trainer state, containing model parameters, optimizer
                state, and RNG states.
            inputs: A batch of training data to use for this step. It will be unpacked
                as keyword arguments to the model's forward method.
            inplace: Whether to perform in-place updates during this step. In-place
                updates can reduce memory usage but can cause problems with autograd.
            trace: Whether to trace this step with autograd to allow for a backward
                pass later. Tracing can add overhead, so it should only be enabled if
                backward passes will be needed.
            fsdp: Whether the model is wrapped with FSDP. If False and distributed
                training is being used, the trainer will perform its own all-reduce of
                gradients. If True, the trainer will assume that FSDP is handling
                gradient synchronization, and will not perform any all-reduces itself.
            max_grad_norm: Clip gradients to this global norm before the optimizer step,
                matching HuggingFace Trainer's ``max_grad_norm``. ``None`` or ``0``
                disables clipping.
            grad_accum_steps: Split the batch into this many micro-batches and
                sum their normalized gradients before the single optimizer
                update. Matches the full-batch gradient up to float
                associativity unless dropout is active. The backward replay
                stays faithful to whichever the forward ran.
        """
        torch.random.set_rng_state(state.cpu_rng_state)
        maybe_set_cuda_rng_state(state.cuda_rng_state)

        # Trainable params live on the meta device and are swapped in from state.
        # Frozen params remain on-device in the model and are left untouched.
        with swap_parameters(
            self.model,
            state.params,
            state.buffers,
            preserve_graph=trace,
        ) as params:
            # A single-shot step is just the one-micro-batch case. Tracing with
            # grad_accum_steps > 1 keeps every micro-graph alive, so the
            # metagradient backward uses ``metagrad_step`` instead.
            grads, self._last_loss = accumulate_grads(
                self.model, params, inputs, grad_accum_steps, create_graph=trace
            )

        return self._apply_update(
            state,
            grads,
            inplace=inplace,
            trace=trace,
            fsdp=fsdp,
            max_grad_norm=max_grad_norm,
        )

    def _apply_update(
        self,
        state: TrainerState,
        grads: dict[str, torch.Tensor],
        *,
        inplace: bool = False,
        trace: bool = False,
        fsdp: bool = False,
        max_grad_norm: float | None = None,
    ) -> TrainerState:
        """Reduce/clip ``grads`` and apply one optimizer update to ``state``."""
        if dist.is_initialized() and not fsdp:
            if trace:
                # Twice-differentiable all-reduce.
                grads = {
                    k: cast(
                        torch.Tensor,
                        _ReplicatedAllReduceSum.apply(
                            g / dist.get_world_size(),
                            dist.distributed_c10d._get_default_group(),
                        ),
                    )
                    for k, g in grads.items()
                }
            else:
                for g in grads.values():
                    dist.all_reduce(g, op=dist.ReduceOp.AVG)

        # Clip by global norm over all params and ranks/FSDP shards.
        if max_grad_norm:
            # cast: sum()'s int start value widens the type to `Tensor | int`.
            sq_norm = cast(torch.Tensor, sum(g.pow(2).sum() for g in grads.values()))
            if fsdp:
                # Sharded grads give a partial sum; redistribute (autograd-aware) to
                # sum across the mesh. DDP grads are already all-reduced above.
                assert isinstance(
                    sq_norm, DTensor
                ), "fsdp=True but grads aren't DTensors"
                sq_norm = sq_norm.redistribute(placements=[Replicate()])
            else:
                assert not isinstance(
                    sq_norm, DTensor
                ), "DTensor grads require fsdp=True"
            coef = (max_grad_norm / (sq_norm.sqrt() + 1e-6)).clamp(max=1.0)
            if trace:
                # Out-of-place to keep the clip in the autograd graph.
                grads = {k: g * coef for k, g in grads.items()}
            else:
                for g in grads.values():
                    g.mul_(coef)

        updates, new_state = self.optimizer.update(
            grads, state.opt_state, inplace=inplace, params=state.params
        )
        new_params = torchopt.apply_updates(state.params, updates, inplace=inplace)
        return TrainerState(
            new_params,
            new_state,
            state.buffers,
            state.batch_index + 1,
        )

    def metagrad_step(
        self,
        fwd_state: TrainerState,
        inputs: dict[str, Any],
        bwd_state: "BackwardState",
        data_weights: torch.Tensor,
        *,
        fsdp: bool = False,
        max_grad_norm: float | None = None,
        grad_accum_steps: int = 1,
    ) -> "BackwardState":
        """Micro-batched VJP through one training step (metagradient replay).

        Equivalent to the single-shot ``step(trace=True)`` + ``autograd.grad``
        in :meth:`backward`, with peak memory bounded to one micro-batch — see
        :func:`bergson.magic.grad_accum.microbatch_step_vjp` for how.
        ``fwd_state`` must be detached and marked ``requires_grad``;
        ``data_weights`` must require grad.
        """
        param_grads, opt_grads, weight_cot = microbatch_step_vjp(
            self.model,
            self._apply_update,
            fwd_state,
            inputs,
            bwd_state.param_grads,
            bwd_state.opt_grads,
            data_weights,
            fsdp=fsdp,
            max_grad_norm=max_grad_norm,
            grad_accum_steps=grad_accum_steps,
        )
        return BackwardState(
            param_grads, opt_grads, weight_cot + bwd_state.weight_grads
        )

    def resume(self, state: TrainerState, save_dir: str) -> TrainerState:
        """Resume training from the most recent checkpoint in `save_dir`.

        This method modifies `state` in-place to load the most recent checkpoint, and
        returns it for convenience.
        """
        ckpt_list = sorted_checkpoints(save_dir)

        # Filter out incomplete checkpoints (missing .metadata) and clean them up
        valid_ckpts = []
        for idx, path in ckpt_list:
            metadata = os.path.join(path, ".metadata")
            if os.path.exists(metadata):
                valid_ckpts.append((idx, path))
            else:
                rmtree(path) if os.path.isdir(path) else os.remove(path)

        # Load the most recent trainer state
        if valid_ckpts:
            last_idx, last_path = valid_ckpts[-1]
            state.batch_index = last_idx
            state.load(last_path)
            state.detach_()

        return state

    def train(
        self,
        state: TrainerState,
        data: DataStream,
        *,
        debug: bool = False,
        inplace: bool = False,
        save_dir: str | None = None,
        save_mode: SaveMode = "sqrt",
        trace: bool = False,
        log_fn: Callable[[int, float], None] | None = None,
        resume: bool = False,
        fsdp: bool = False,
        max_grad_norm: float | None = None,
        grad_accum_steps: int = 1,
        optimizer_cfg: dict | None = None,
        save_interval: int = 0,
    ) -> TrainerState:
        """Train the model on the given data stream, starting from the given state.

        Args:
            state: The initial trainer state to start training from.
            data: The training data stream to iterate over.
            debug: Whether to print debug information about checkpoint loading times.
            inplace: Whether to perform in-place updates during training. In-place
                updates can reduce memory usage but may cause issues with some
                optimizers or models.
            save_dir: The directory in which to save checkpoints during training.
            save_mode: The strategy for how often to save checkpoints.
            trace: Whether to trace the training step with autograd to allow for
                backward passes. Tracing can add overhead, so it should only be enabled
                if backward passes will be needed.
            log_fn: An optional function to call after each training step with the step
                index and the most recent loss value, for logging purposes.
            resume: Whether to resume from a previously interrupted training run by
                loading the most recent checkpoint from `save_dir`.
            fsdp: Flag to pass to `Trainer.step`, indicating whether the model is
                wrapped with FSDP.
            max_grad_norm: Clip gradients to this global norm before each optimizer
                step. Passed through to `Trainer.step`.
            grad_accum_steps: Number of micro-batches to accumulate gradients over
                per optimizer step. Passed through to `Trainer.step`.
            optimizer_cfg: When set (to the optimizer's betas/eps/eps_root),
                write each checkpoint's second moments to ``optimizer.pt``,
                tagged with the step and these hyperparameters. AdamW only.
            save_interval: Checkpointing interval in steps for ``save_mode="interval"``.

        Returns:
            The final trainer state after training.
        """
        # Make sure the save directory exists
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # Always save the first state
        next_save = 0
        n = len(data)

        start = 0
        if resume and save_dir is not None:
            state = self.resume(state, save_dir)
            start = state.batch_index

            # Fast-forward the checkpoint schedule to where we're resuming from.
            while next_save < start:
                next_save = next_save_index(next_save, n, save_mode)

        pending_save: SaveFuture | None = None

        main = not dist.is_initialized() or dist.get_rank() == 0
        pbar = tqdm(range(start, n), desc="Training", disable=not main)

        for i in pbar:
            # Save checkpoint BEFORE each step. Step 0 is the initial state prior to
            # any updates, step 1 is the state after the first update, etc.
            if save_dir and i == next_save:
                # Wait for the previous save before starting a new one to avoid
                # multiple concurrent DCP saves with separate Gloo groups, which can
                # deadlock when background threads call distributed operations.
                if pending_save is not None:
                    pending_save.result()

                p = os.path.join(save_dir, f"step_{i}.ckpt")

                # Write before the DCP save starts, into the same directory.
                if optimizer_cfg is not None:
                    # Local import: at module scope this cycles back into
                    # magic.trainer via the package __init__.
                    from ..utils.load_from_optimizer import (
                        save_second_moments_as_optimizer_pt,
                    )

                    os.makedirs(p, exist_ok=True)
                    save_second_moments_as_optimizer_pt(
                        self.model,
                        state.opt_state,
                        os.path.join(p, "optimizer.pt"),
                        step=i,
                        **optimizer_cfg,
                    )

                pending_save = state.save(p, debug_pbar=pbar if debug else None)

                next_save = next_save_index(next_save, n, save_mode, save_interval)

            x = data[i]
            state = self.step(
                state,
                x,
                inplace=inplace,
                trace=trace,
                fsdp=fsdp,
                max_grad_norm=max_grad_norm,
                grad_accum_steps=grad_accum_steps,
            )

            if log_fn is not None:
                log_fn(i, self._last_loss)

        if pending_save is not None:
            pending_save.result()

        # Snapshots are written before each step; when the interval cadence
        # lands on n, write the post-training state too.
        if save_dir and save_mode == "interval" and next_save == n:
            p = os.path.join(save_dir, f"step_{n}.ckpt")
            if optimizer_cfg is not None:
                # Local import: at module scope this cycles back into
                # magic.trainer via the package __init__.
                from ..utils.load_from_optimizer import (
                    save_second_moments_as_optimizer_pt,
                )

                os.makedirs(p, exist_ok=True)
                save_second_moments_as_optimizer_pt(
                    self.model,
                    state.opt_state,
                    os.path.join(p, "optimizer.pt"),
                    step=n,
                    **optimizer_cfg,
                )
            state.save(p).result()

        return state

    def save_backward_state(self, bwd_state, path, expected_idx, last_idx):
        tmp_path = path + ".tmp"
        torch.save(
            {
                "expected_idx": expected_idx,
                "last_idx": last_idx,
                "param_grads": bwd_state.param_grads,
                "opt_grads": bwd_state.opt_grads,
                "weight_grads": bwd_state.weight_grads,
            },
            tmp_path,
        )
        os.replace(tmp_path, path)

    def load_backward_state(self, path, ckpt_list, device, main: bool):
        saved = torch.load(path, map_location=device, weights_only=True)
        bwd_state = BackwardState(
            saved["param_grads"],
            saved["opt_grads"],
            saved["weight_grads"],
        )
        expected_idx = saved["expected_idx"]
        last_idx = saved["last_idx"]

        # Filter to valid checkpoints we still need to process
        valid_ckpts = []
        for idx, path in ckpt_list:
            if idx > expected_idx:
                continue
            metadata = os.path.join(path, ".metadata")
            if os.path.exists(metadata):
                valid_ckpts.append((idx, path))
            elif os.path.isdir(path) and main:
                rmtree(path)
        ckpt_list = valid_ckpts

        if not ckpt_list and expected_idx >= 0:
            raise RuntimeError(
                f"Cannot resume backward: no valid checkpoints found "
                f"for step {expected_idx}"
            )

        if main:
            print(f"Resuming backward pass from step {expected_idx}")

        return bwd_state, ckpt_list, expected_idx, last_idx

    def backward(
        self,
        ckpt_dir: str,
        data: DataStream,
        bwd_state: BackwardState,
        fwd_state: TrainerState,
        *,
        cleanup: bool = True,
        debug: bool = False,
        inplace: bool = False,
        fsdp: bool = False,
        resume: bool = False,
        save_every: int = 0,
        save_mode: SaveMode = "sqrt",
        max_grad_norm: float | None = None,
        grad_accum_steps: int = 1,
    ) -> BackwardState:
        """Run a backward pass through the training trajectory saved at `ckpt_dir`.

        Args:
            ckpt_dir: Directory containing checkpoints saved during the forward pass.
            data: The training data stream, needed to replay forward steps.
            bwd_state: The initial backward state, containing gradients for the query
                loss function.
            fwd_state: Forward state containing the model parameters and optimizer
                state from the end of the forward trajectory.
            cleanup: Whether to delete checkpoints in `ckpt_dir` as soon as they are
                no longer needed. If False, only the temporary checkpoints created
                during the backward pass will be deleted, and all original checkpoints
                will be preserved.
            debug: Whether to print debug information about checkpoint loading times.
            inplace: Whether to perform in-place updates during forward and backward
                steps. In-place updates can reduce memory usage but may cause issues
                with some optimizers or models.
            fsdp: The `fsdp` flag that will be passed to the trainer's `step` method.
            resume: Whether to resume from a previously interrupted backward pass by
                loading the backward state and skipping checkpoints that have already
                been processed.
            save_every: If > 0, save the backward state every N steps to allow resuming
                from interruptions. Backward checkpoints are saved in `ckpt_dir` with
                the name `backward_rank{rank}.pt`.
            save_mode: The save mode that was used during the forward trajectory, which
                determines how checkpoints are spaced and thus how the backward pass
                should step forward through the trajectory when replaying.
            max_grad_norm: Clip gradients to this global norm before each optimizer
                step when replaying the forward trajectory. Must match the value used
                during the forward pass for the replay to be faithful. Passed through
                to `Trainer.step`.
            grad_accum_steps: Micro-batch count used during the forward pass;
                must match for the replay to be faithful. When > 1 the traced
                step uses the two-stage micro-VJP (`Trainer.metagrad_step`).

        Returns:
            The final backward state after processing the entire trajectory.
        """
        ckpts = sorted_checkpoints(ckpt_dir)
        ckpt_paths = [path for _, path in ckpts]
        preserve_paths = {ckpt_paths[-1]} if cleanup else set(ckpt_paths)

        ckpt_list = cast(list[tuple[int, str | TrainerState]], ckpts)
        state_size = fwd_state.size_in_bytes()

        main = not dist.is_initialized() or dist.get_rank() == 0
        rank = dist.get_rank() if dist.is_initialized() else 0

        bwd_ckpt_path = os.path.join(ckpt_dir, f"backward_rank{rank}.pt")

        if resume and os.path.exists(bwd_ckpt_path):
            bwd_state, ckpt_list, expected_idx, last_idx = self.load_backward_state(
                bwd_ckpt_path, ckpt_list, data.device, main
            )
        else:
            expected_idx, _ = ckpt_list[-1]
            last_idx = expected_idx

        main_pbar = RtlTqdm(
            desc="Backward",
            total=last_idx + 1,
            initial=last_idx - expected_idx,
            disable=not main,
            position=0,
            # Get rid of jitters in the ETA due to rematerialization
            smoothing=0,
        )
        sub_pbar = None

        save_futures: list[SaveFuture] = []
        while ckpt_list:
            # Make sure everything has been saved
            for fut in save_futures:
                fut.result()
            save_futures.clear()

            idx, ckpt = ckpt_list[-1]
            fwd_state.batch_index = idx
            fwd_state.detach_()  # Detach so that replay steps can use in-place ops

            start = time.monotonic()
            fwd_state.load(ckpt) if isinstance(ckpt, str) else fwd_state.copy_(ckpt)
            elapsed = time.monotonic() - start

            if debug and main:
                name = ckpt if isinstance(ckpt, str) else f"in-memory checkpoint {idx}"
                main_pbar.write(f"Loaded checkpoint {name} in {elapsed:.2f} seconds")

            # Only delete this checkpoint if it's the one we expected to load. If it's
            # not, we need to keep it around, and step forward through training
            if idx == expected_idx:
                del ckpt_list[-1]

                # Only delete on the main rank
                if isinstance(ckpt, str) and main and ckpt not in preserve_paths:
                    rmtree(ckpt) if os.path.isdir(ckpt) else os.remove(ckpt)

            # Step forward in training if needed
            next_save = (
                (expected_idx - idx) // 2 + idx
                if save_mode == "log"
                else min(expected_idx, idx + 1)
            )
            while idx < expected_idx:
                if sub_pbar is None:
                    sub_pbar = tqdm(
                        total=expected_idx - idx,
                        desc=f"Rematerializing steps {idx} to {expected_idx}",
                        disable=not main,
                        leave=False,
                        position=1,
                        smoothing=0,
                    )

                fwd_state = self.step(
                    fwd_state,
                    data[fwd_state.batch_index],
                    inplace=inplace,
                    trace=False,
                    fsdp=fsdp,
                    max_grad_norm=max_grad_norm,
                    grad_accum_steps=grad_accum_steps,
                )
                idx += 1
                sub_pbar.update()

                # Save checkpoints for states we will need later
                if idx == next_save and idx < expected_idx:
                    # Switch from RAM disk to checkpoint dir if needed
                    num_copies = dist.get_world_size() if dist.is_initialized() else 1
                    if psutil.virtual_memory().available > num_copies * state_size:
                        ckpt_list.append((idx, fwd_state.to("cpu").detach_()))
                    else:
                        ckpt = os.path.join(ckpt_dir, f"step_{idx}.ckpt")
                        ckpt_list.append((idx, ckpt))

                        save_futures.append(
                            fwd_state.save(
                                ckpt, debug_pbar=main_pbar if debug else None
                            )
                        )

                    # Advance next_save according to the save mode
                    next_save = (
                        (expected_idx - idx) // 2 + idx
                        if save_mode == "log"
                        else min(expected_idx, idx + 1)
                    )

            if sub_pbar is not None:
                sub_pbar.close()
                sub_pbar = None

            # The index we expect on the next iteration is one less than the current
            expected_idx = idx - 1

            fwd_state.detach_()
            fwd_state.requires_grad = True
            data.requires_grad = True

            # Two equivalent VJP paths that must stay in sync: micro-batched
            # (memory-bounded) when accumulating, single-shot traced otherwise.
            if grad_accum_steps > 1:
                bwd_state = self.metagrad_step(
                    fwd_state,
                    data[fwd_state.batch_index],
                    bwd_state,
                    data.weights,
                    fsdp=fsdp,
                    max_grad_norm=max_grad_norm,
                    grad_accum_steps=grad_accum_steps,
                )
                main_pbar.update()
            else:
                flat_i = fwd_state.differentiable_tensors()

                # Re-do the training step
                state_f = self.step(
                    fwd_state,
                    data[fwd_state.batch_index],
                    trace=True,
                    fsdp=fsdp,
                    max_grad_norm=max_grad_norm,
                )
                main_pbar.update()

                # Carefully consume the bwd state to save memory
                flat_f = state_f.differentiable_tensors()
                p_grads = list(bwd_state.param_grads.values())
                o_grads = bwd_state.opt_grads

                p_keys = list(bwd_state.param_grads.keys())
                w_grads = bwd_state.weight_grads
                del bwd_state

                # grad_outputs is the gradient of the loss wrt the next
                # TrainerState. We're doing a VJP to get the gradient wrt the
                # current TrainerState, AND the example weights for this batch.
                inps = flat_i + [data.weights]
                result = list(
                    torch.autograd.grad(
                        flat_f,
                        inps,
                        grad_outputs=p_grads + o_grads,
                        allow_unused=True,
                    )
                )
                del p_grads

                # Accumulate parameter gradients
                param_grads = {k: result[i] for i, k in enumerate(p_keys)}
                del result[: len(p_keys)]

                this_step_weight_grad = result[-1]
                if dist.is_initialized() and not fsdp:
                    # The all-reduce above correctly gives the true, averaged
                    # gradient for the parameter update. But a given document
                    # only contributes to 1/world_size of that average, so to
                    # recover its score from the average gradient we divide
                    # by world_size.
                    this_step_weight_grad = (
                        this_step_weight_grad / dist.get_world_size()
                    )
                weight_grads = this_step_weight_grad + w_grads
                bwd_state = BackwardState(param_grads, result[:-1], weight_grads)

            # Save backward state for resume
            steps_done = last_idx - expected_idx
            if save_every > 0 and steps_done % save_every == 0:
                self.save_backward_state(
                    bwd_state, bwd_ckpt_path, expected_idx, last_idx
                )

        for fut in save_futures:
            fut.result()

        # Clean up backward state file on successful completion
        if os.path.exists(bwd_ckpt_path):
            os.remove(bwd_ckpt_path)

        main_pbar.close()
        return bwd_state


def prepare_trainer(cfg: TrainingConfig, rank: int, schedule: Callable):
    """Prepare the model, optimizer, and trainer for training."""
    model, target_modules = setup_model_and_peft(
        cfg,
        attn_implementation="eager",
        apply_fsdp=False,
    )
    model.to(get_device(rank))  # type: ignore[reportArgumentType]

    # setup_model_and_peft leaves the model in from_pretrained's eval mode.
    if cfg.train_mode:
        model.train()
    else:
        model.eval()

    if target_modules:
        # Only train the PEFT adapter parameters
        model.requires_grad_(False)
        for name in target_modules:
            module = model.get_submodule(name)
            module.requires_grad_(True)
    else:
        model.requires_grad_(True)

    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable(  # type: ignore[attr-defined]
            gradient_checkpointing_kwargs=dict(use_reentrant=False),
        )

    if cfg.fsdp and dist.is_initialized():
        apply_dtensor_patch()
        mesh = init_device_mesh("cuda", (dist.get_world_size(),))
        with mesh:
            model = simple_fsdp(model)

    match cfg.optimizer:
        case "adamw":
            opt = torchopt.adamw(
                schedule,
                betas=(cfg.adam_beta1, cfg.adam_beta2),
                eps=cfg.adam_eps,
                eps_root=cfg.eps_root,
                weight_decay=cfg.weight_decay,
            )
        case "muon":
            opt = muon(
                schedule,
                momentum=cfg.adam_beta1,
                adamw_betas=(cfg.adam_beta1, cfg.adam_beta2),
                adamw_eps_root=cfg.eps_root,
                weight_decay=cfg.weight_decay,
            )
        case "sgd":
            opt = torchopt.sgd(
                schedule,
                momentum=cfg.adam_beta1,
                weight_decay=cfg.weight_decay,
            )
        case other:
            raise ValueError(f"Unsupported optimizer: {other}")

    trainer, fwd_state = Trainer.initialize(model, opt)
    return trainer, fwd_state, model

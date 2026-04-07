import math
import os
import time
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from shutil import rmtree
from typing import Literal

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.tensor  # noqa: F401 — register DTensor for torch.load
import torchopt
from torch import nn
from torch.distributed.nn.functional import all_reduce as differentiable_all_reduce
from torchopt.pytree import tree_flatten_with_path, tree_iter, tree_map
from torchopt.typing import GradientTransformation, OptState
from tqdm.auto import tqdm

from ..data import sorted_checkpoints
from ..distributed import grad_tree
from .data_stream import DataStream
from .fsdp import shallow_copy
from .rtl_tqdm import RtlTqdm
from .swap import swap_parameters


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


def _maybe_get_cuda_rng_state() -> torch.Tensor:
    """ "Get the CUDA RNG state if CUDA is initialized, otherwise return zeros."""
    if torch.cuda.is_initialized():
        return torch.cuda.random.get_rng_state()

    # This corresponds to a manual seed of 0
    return torch.zeros(16, dtype=torch.uint8)


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
    cuda_rng_state: torch.Tensor = field(default_factory=_maybe_get_cuda_rng_state)
    cpu_rng_state: torch.Tensor = field(default_factory=torch.random.get_rng_state)

    def to(self, device: torch.device | str) -> "TrainerState":
        params = {k: p.to(device) for k, p in self.params.items()}
        buffers = {k: b.to(device) for k, b in self.buffers.items()}
        opt_state = tree_map(
            lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, self.opt_state
        )
        return TrainerState(params, opt_state, buffers, self.batch_index)

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

    def detach_(self):
        for k, p in self.params.items():
            self.params[k] = p.detach()

        def _detach_leaf(t):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                return t.detach()
            return t

        self.opt_state = tree_map(_detach_leaf, self.opt_state)

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
        cpu_state = torch.random.get_rng_state()
        torch.random.set_rng_state(self.cpu_rng_state)

        with swap_parameters(model, self.params, self.buffers, strict=True) as p:
            yield p

        torch.random.set_rng_state(cpu_state)

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

    def __init__(self, model: nn.Module, optimizer: GradientTransformation):
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
        inputs: dict,
        *,
        inplace: bool = False,
        trace: bool = False,
        fsdp: bool = False,
    ) -> TrainerState:
        torch.random.set_rng_state(state.cpu_rng_state)

        # Trainable params live on the meta device and are swapped in from state.
        # Frozen params remain on-device in the model and are left untouched.
        with swap_parameters(
            self.model,
            state.params,
            state.buffers,
            preserve_graph=trace,
        ) as params:
            outputs = self.model(**inputs)

            # Currently we support two output types: HuggingFace, and "raw loss"
            # - HuggingFace models output a dict/dataclass with a "loss" field
            # - Raw loss models output a single scalar loss value as a Tensor
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                loss = outputs

            assert isinstance(loss, torch.Tensor), "Loss must be a Tensor"
            self._last_loss = loss.detach().item()
            grads = grad_tree(loss, params, create_graph=trace)

        if dist.is_initialized() and not fsdp:
            if trace:
                # Use differentiable all_reduce to preserve autograd graph
                grads = {
                    k: differentiable_all_reduce(g, op=dist.ReduceOp.AVG)
                    for k, g in grads.items()
                }
            else:
                for g in grads.values():
                    dist.all_reduce(g, op=dist.ReduceOp.AVG)

        updates, new_state = self.optimizer.update(
            grads, state.opt_state, inplace=inplace, params=state.params
        )
        new_params = torchopt.apply_updates(state.params, updates, inplace=inplace)
        state = TrainerState(
            new_params,
            new_state,
            state.buffers,
            state.batch_index + 1,
        )
        return state

    def resume(
        self,
        state: TrainerState,
        save_dir: str,
    ):
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
        save_mode: Literal["all", "sqrt"] = "sqrt",
        trace: bool = False,
        log_fn: Callable[[int, float], None] | None = None,
        resume: bool = False,
        fsdp: bool = False,
    ) -> TrainerState:
        # Make sure the save directory exists
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        start = 0
        if resume and save_dir is not None:
            state = self.resume(state, save_dir)
            start = state.batch_index

        chunk_size = math.isqrt(len(data)) if save_mode == "sqrt" else 1
        last_start = len(data) - chunk_size

        pending_save: SaveFuture | None = None

        main = not dist.is_initialized() or dist.get_rank() == 0
        pbar = tqdm(range(start, len(data)), desc="Training", disable=not main)

        for i in pbar:
            # Save checkpoint BEFORE each step. Step 0 is the initial state prior to
            # any updates, step 1 is the state after the first update, etc.
            if save_dir and (i % chunk_size == 0 or i >= last_start):
                # Wait for the previous save before starting a new one to avoid
                # multiple concurrent DCP saves with separate Gloo groups, which can
                # deadlock when background threads call distributed operations.
                if pending_save is not None:
                    pending_save.result()

                p = os.path.join(save_dir, f"step_{i}.ckpt")
                pending_save = state.save(p, debug_pbar=pbar if debug else None)

            x = data[i]
            state = self.step(state, x, inplace=inplace, trace=trace, fsdp=fsdp)

            if log_fn is not None:
                log_fn(i, self._last_loss)

        if pending_save is not None:
            pending_save.result()

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
    ) -> BackwardState:
        ckpt_list = sorted_checkpoints(ckpt_dir)

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

            idx, path = ckpt_list[-1]
            fwd_state.batch_index = idx

            start = time.monotonic()
            fwd_state.load(path)
            elapsed = time.monotonic() - start

            if debug and main:
                main_pbar.write(f"Loaded checkpoint {path} in {elapsed:.2f} seconds")

            # Detach after loading so that replay steps can use in-place ops
            # (loaded tensors may retain requires_grad from the previous traced step)
            fwd_state.detach_()

            # Detach after loading so that replay steps can use in-place ops
            # (loaded tensors may retain requires_grad from the previous traced step)
            fwd_state.detach_()

            # Only delete this checkpoint if it's the one we expected to load. If it's
            # not, we need to keep it around, and step forward through training
            if idx == expected_idx:
                del ckpt_list[-1]

                # Only delete on the main rank
                if cleanup and main and idx != last_idx:
                    rmtree(path) if os.path.isdir(path) else os.remove(path)

            # Step forward in training if needed
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
                )
                idx += 1
                sub_pbar.update()

                # Save checkpoints for states we will need later
                if idx < expected_idx:
                    path = os.path.join(ckpt_dir, f"step_{idx}.ckpt")
                    ckpt_list.append((idx, path))

                    save_futures.append(
                        fwd_state.save(path, debug_pbar=main_pbar if debug else None)
                    )

            if sub_pbar is not None:
                sub_pbar.close()
                sub_pbar = None

            # The index we expect on the next iteration is one less than the current
            expected_idx = idx - 1

            fwd_state.detach_()
            fwd_state.requires_grad = True
            data.requires_grad = True

            flat_i = fwd_state.differentiable_tensors()

            # Re-do the training step
            state_f = self.step(
                fwd_state,
                data[fwd_state.batch_index],
                trace=True,
                fsdp=fsdp,
            )
            main_pbar.update()

            # Carefully consume the bwd state to save memory
            flat_f = state_f.differentiable_tensors()
            p_grads = list(bwd_state.param_grads.values())
            o_grads = bwd_state.opt_grads

            p_keys = list(bwd_state.param_grads.keys())
            w_grads = bwd_state.weight_grads
            del bwd_state

            # grad_outputs is the gradient of the loss wrt the next TrainerState. We're
            # doing a VJP to get the gradient wrt the current TrainerState, AND the
            # example weights for this batch.
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

            weight_grads = result[-1] + w_grads
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

import math
import os
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from shutil import rmtree
from typing import Literal

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torchopt
from torch import nn
from torchopt.pytree import tree_flatten_with_path, tree_iter, tree_map
from torchopt.typing import GradientTransformation, OptState
from tqdm.auto import tqdm

from ..data import sorted_checkpoints
from ..distributed import grad_tree, shallow_copy
from .data_stream import DataStream
from .rtl_tqdm import RtlTqdm
from .swap import swap_parameters


def _maybe_get_cuda_rng_state() -> torch.Tensor:
    """ "Get the CUDA RNG state if CUDA is initialized, otherwise return zeros."""
    if torch.cuda.is_initialized():
        return torch.cuda.random.get_rng_state()

    # This corresponds to a manual seed of 0
    return torch.zeros(16, dtype=torch.uint8)


class SaveFuture:
    """Wraps a DCP async_save future, destroying the gloo process group in result().

    The group must be destroyed synchronously inside result() rather than in a
    done_callback, because concurrent.futures.Future notifies result() waiters
    before invoking callbacks — so a callback-based destroy races with the next
    save() call creating a new group, leaking gloo sockets.
    """

    def __init__(self, fut: Future, grp: "dist.ProcessGroup | None"):
        self._fut = fut
        self._grp = grp

    def result(self):
        result = self._fut.result()
        if self._grp is not None:
            dist.destroy_process_group(self._grp)
            self._grp = None

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

    def save(self, path: str) -> SaveFuture:
        # Create a new process group so that we can overlap saves.
        if dist.is_initialized():
            grp = dist.new_group(backend="gloo", group_desc=path)
            assert isinstance(grp, dist.ProcessGroup)
        else:
            grp = None

        fut = dcp.async_save(
            self.state_dict(),
            checkpoint_id=path,
            process_group=grp,
        )
        assert isinstance(fut, Future)
        return SaveFuture(fut, grp)

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
            grads = grad_tree(loss, params, create_graph=trace)

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

    def train(
        self,
        state: TrainerState,
        data: DataStream,
        *,
        inplace: bool = False,
        save_dir: str | None = None,
        save_mode: Literal["linear", "sqrt"] = "sqrt",
        trace: bool = False,
    ) -> TrainerState:
        # Make sure the save directory exists
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        chunk_size = math.isqrt(len(data)) if save_mode == "sqrt" else 1
        last_start = len(data) - chunk_size

        pending_save: SaveFuture | None = None

        main = not dist.is_initialized() or dist.get_rank() == 0
        pbar = tqdm(data, desc="Training", disable=not main)

        for i, x in enumerate(pbar):
            # Save checkpoint BEFORE each step. Step 0 is the initial state prior to
            # any updates, step 1 is the state after the first update, etc.
            if save_dir and (i % chunk_size == 0 or i >= last_start):
                # Wait for the previous save before starting a new one to avoid
                # multiple concurrent DCP saves with separate Gloo groups, which can
                # deadlock when background threads call distributed operations.
                if pending_save is not None:
                    pending_save.result()

                p = os.path.join(save_dir, f"step_{i}.ckpt")
                pending_save = state.save(p)

            state = self.step(state, x, inplace=inplace, trace=trace)

        if pending_save is not None:
            pending_save.result()

        return state

    def backward(
        self,
        ckpt_dir: str,
        data: DataStream,
        bwd_state: BackwardState,
        fwd_state: TrainerState,
        *,
        cleanup: bool = True,
        inplace: bool = False,
    ) -> BackwardState:
        ckpt_list = sorted_checkpoints(ckpt_dir)
        expected_idx, _ = ckpt_list[-1]

        main = not dist.is_initialized() or dist.get_rank() == 0
        main_pbar = RtlTqdm(
            desc="Backward",
            total=expected_idx + 1,
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
            fwd_state.load(path)

            # Detach after loading so that replay steps can use in-place ops
            # (loaded tensors may retain requires_grad from the previous traced step)
            fwd_state.detach_()

            # Only delete this checkpoint if it's the one we expected to load. If it's
            # not, we need to keep it around, and step forward through training
            if idx == expected_idx:
                del ckpt_list[-1]

                # Only delete on the main rank
                if cleanup and (not dist.is_initialized() or dist.get_rank() == 0):
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
                )
                idx += 1
                sub_pbar.update()

                # Save checkpoints for states we will need later
                if idx < expected_idx:
                    path = os.path.join(ckpt_dir, f"step_{idx}.ckpt")
                    ckpt_list.append((idx, path))

                    save_futures.append(fwd_state.save(path))

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

        for fut in save_futures:
            fut.result()

        main_pbar.close()
        return bwd_state

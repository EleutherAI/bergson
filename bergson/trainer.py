import math
import os
import re
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from shutil import rmtree
from typing import Literal

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torchopt
from datasets import Dataset
from torch import nn
from torchopt.pytree import tree_iter, tree_map
from torchopt.typing import GradientTransformation, OptState
from tqdm.auto import tqdm

from .distributed import grad_tree, shallow_copy
from .swap import swap_parameters


def _maybe_get_cuda_rng_state() -> torch.Tensor:
    """ "Get the CUDA RNG state if CUDA is initialized, otherwise return zeros."""
    if torch.cuda.is_initialized():
        return torch.cuda.random.get_rng_state()

    # This corresponds to a manual seed of 0
    return torch.zeros(16, dtype=torch.uint8)


def sorted_checkpoints(folder: str) -> list[tuple[int, str]]:
    """
    Return a list of (batch_index, filepath) sorted by batch_index
    for files named like: step_<index>.ckpt
    """
    pattern = re.compile(r"step_(\d+)\.ckpt$")

    checkpoints = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        match = pattern.match(name)
        if match:
            batch_index = int(match.group(1))
            checkpoints.append((batch_index, path))

    return sorted(checkpoints, key=lambda x: x[0])


class DataStream:
    """Tokenized dataset with per-example weights for distributed training.

    ``batch_size`` is the **per-device** batch size.  The global batch size is
    ``batch_size * world_size``, which is always divisible by ``world_size``
    by construction.

    When the dataset doesn't fill the final batch, the remaining positions
    are padded: weight=0, labels=-100, and real weights in that batch are
    scaled up by ``global_batch_size / num_real`` so that the ``.mean()``
    reduction gives the correct result after AVG all-reduce.
    """

    def __init__(
        self,
        dataset: Dataset,
        processor,
        batch_size: int,
        num_batches: int = 0,
        *,
        device: torch.device | str = "cpu",
        input_key: str = "text",
        max_length: int = 256,
        per_token: bool = False,
    ):
        self.dataset = dataset
        self.processor = processor
        self.input_key = input_key
        self.max_length = max_length
        self.per_token = per_token

        self.device = device

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.batch_size = batch_size * self.world_size
        self.num_batches = num_batches or math.ceil(len(self.dataset) / self.batch_size)

        needed = self.batch_size * self.num_batches
        self._num_real = min(len(self.dataset), needed)
        self._pad_last_batch = needed - self._num_real

        if self._pad_last_batch > 0 and self._num_real < len(self.dataset):
            raise ValueError(
                f"Dataset has {len(self.dataset)} examples but {self.num_batches} "
                f"batches of size {self.batch_size} require {needed}. "
                f"Pass a larger split or reduce --num_batches."
            )

        n = self.batch_size * self.num_batches
        shape = (n, max_length) if per_token else (n,)
        self.weights = nn.Parameter(torch.ones(*shape, device=device))

        if self._pad_last_batch > 0:
            real_in_last = self.batch_size - self._pad_last_batch
            correction = self.batch_size / real_in_last
            last_start = (self.num_batches - 1) * self.batch_size
            self.weights.data[last_start : last_start + real_in_last] = correction
            self.weights.data[last_start + real_in_last :] = 0.0

    @property
    def requires_grad(self) -> bool:
        return self.weights.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self.weights.requires_grad = value

    def reset_weights(self):
        """Reset weights to their initial state (padded zeros, correction-scaled)."""
        self.weights.data.fill_(1.0)
        if self._pad_last_batch > 0:
            real_in_last = self.batch_size - self._pad_last_batch
            correction = self.batch_size / real_in_last
            last_start = (self.num_batches - 1) * self.batch_size
            self.weights.data[last_start : last_start + real_in_last] = correction
            self.weights.data[last_start + real_in_last :] = 0.0

    def __getitem__(self, i: int) -> dict:
        if i < 0 or i >= self.num_batches:
            raise IndexError("DataStream index out of range")

        # Fetch only this rank's interleaved slice of the batch directly from the
        # dataset, avoiding tokenizing examples that will just be discarded.
        indices = list(
            range(
                i * self.batch_size + self.rank,
                (i + 1) * self.batch_size,
                self.world_size,
            )
        )
        # Wrap indices past the dataset (padded positions in final batch)
        if self._pad_last_batch > 0:
            indices = [idx % self._num_real for idx in indices]
        raw = self.dataset[indices]

        # padding="max_length" ensures uniform shape across ranks without needing
        # to process all ranks' examples together.
        x = self.processor(
            raw[self.input_key],
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        x["labels"] = x["input_ids"].clone()
        w = self.weights[
            i * self.batch_size
            + self.rank : (i + 1) * self.batch_size : self.world_size
        ]
        # Mask labels for padded examples so they contribute zero loss
        if self._pad_last_batch > 0:
            pad_mask = w.data == 0
            if pad_mask.any():
                x["labels"][pad_mask] = -100
        if self.per_token and w.ndim == 2:
            T_batch = x["input_ids"].shape[1]
            w = w[:, :T_batch]
        x["example_weight"] = w
        return {k: v.to(self.device) for k, v in x.items()}

    def __iter__(self):
        for i in range(self.num_batches):
            yield self[i]

    def __len__(self):
        return self.num_batches

    def __reversed__(self):
        for i in reversed(range(self.num_batches)):
            yield self[i]


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

    def load(self, path: str):
        """Load state from a checkpoint file."""
        dcp.load(
            self.state_dict(),
            checkpoint_id=path,
            no_dist=not dist.is_initialized(),
        )

    def save(self, path: str) -> Future:
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[rank {rank}] save: starting async_save for {path}", flush=True)

        # Create a new process group so that we can overlap saves
        if dist.is_initialized():
            grp = dist.new_group(backend="gloo", group_desc=path)
            assert isinstance(grp, dist.ProcessGroup)
        else:
            grp = None

        def _done_callback(fut, g=grp, p=path):
            print(f"[rank {rank}] save: callback fired for {p}", flush=True)
            if g is not None:
                print(
                    f"[rank {rank}] save: calling destroy_process_group for {p}",
                    flush=True,
                )
                dist.destroy_process_group(g)
                print(
                    f"[rank {rank}] save: destroy_process_group done for {p}",
                    flush=True,
                )

        fut = dcp.async_save(
            self.state_dict(),
            checkpoint_id=path,
            no_dist=grp is None,
            process_group=grp,
        )
        assert isinstance(fut, Future)
        fut.add_done_callback(_done_callback)
        return fut

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
        return {f.name: getattr(self, f.name) for f in fields(self)}


class Trainer:
    """Stateless, functional trainer for a model, optimizer, and dataset."""

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
        with (
            torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_bf16_supported(),
            ),
            swap_parameters(self.model, state.params, state.buffers) as params,
        ):
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

        pending_fut: Future | None = None

        main = not dist.is_initialized() or dist.get_rank() != 0
        pbar = tqdm(data, desc="Training", disable=main)

        for i, x in enumerate(pbar):
            # Save checkpoint BEFORE each step. Step 0 is the initial state prior to
            # any updates, step 1 is the state after the first update, etc.
            if save_dir and (i % chunk_size == 0 or i >= last_start):
                # Wait for the previous save before starting a new one to avoid
                # multiple concurrent DCP saves with separate Gloo groups, which can
                # deadlock when background threads call distributed operations.
                if pending_fut is not None:
                    _rank = dist.get_rank() if dist.is_initialized() else 0
                    print(
                        f"[rank {_rank}] train: waiting for pending save"
                        f" before step {i}",
                        flush=True,
                    )
                    pending_fut.result()
                    print(
                        f"[rank {_rank}] train: pending save done before step {i}",
                        flush=True,
                    )

                p = os.path.join(save_dir, f"step_{i}.ckpt")
                pending_fut = state.save(p)

            state = self.step(state, x, inplace=inplace, trace=trace)

        if pending_fut is not None:
            _rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[rank {_rank}] train: waiting for final save", flush=True)
            pending_fut.result()
            print(f"[rank {_rank}] train: final save done", flush=True)

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

        save_futures: list[Future] = []
        while ckpt_list:
            # Make sure everything has been saved
            for fut in save_futures:
                fut.result()

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
                fwd_state = self.step(
                    fwd_state,
                    data[fwd_state.batch_index],
                    inplace=inplace,
                    trace=False,
                )
                idx += 1

                # Save checkpoints for states we will need later
                if idx < expected_idx:
                    path = os.path.join(ckpt_dir, f"step_{idx}.ckpt")
                    ckpt_list.append((idx, path))

                    fut = fwd_state.save(path)
                    save_futures.append(fut)

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

        return bwd_state

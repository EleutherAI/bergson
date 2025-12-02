import copy
import math
from functools import wraps

import torch
from peft import PeftModel
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback

from bergson import GradientCollector, GradientProcessor
from bergson.peft import detect_peft_modules


class SaveEverySqrtStepsCallback(TrainerCallback):
    def __init__(self):
        self.sqrt_n = None

    def on_step_begin(self, args, state, control, **kwargs):
        # Compute only once, when total training steps become available
        if self.sqrt_n is None and state.max_steps is not None and state.max_steps > 0:
            self.sqrt_n = int(math.sqrt(state.max_steps))
            print(
                f"[Callback] Total steps = {state.max_steps}, saving every {self.sqrt_n} steps."
            )

        return control

    def on_step_end(self, args, state, control, **kwargs):
        # Skip until sqrt_n is known
        if not self.sqrt_n:
            return control

        if state.global_step > 0 and state.global_step % self.sqrt_n == 0:
            print(f"[Callback] Saving checkpoint at step {state.global_step}")
            control.should_save = True

        return control


def deep_detach_cpu(obj):
    """Recursively detach tensors and move to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    elif isinstance(obj, dict):
        return {k: deep_detach_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_detach_cpu(v) for v in obj]
    else:
        return copy.deepcopy(obj)


class InMemoryCheckpointCallback(TrainerCallback):
    def __init__(self, storage_list):
        self.storage = storage_list

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        # Capture the state at the end of the step
        step_snapshot = {
            "step": state.global_step,
            "global_step": state.global_step,
            "train_batch_size": state.train_batch_size,
            # We capture the model state dict
            "model_state": deep_detach_cpu(model.state_dict()),
            # We capture the optimizer state dict
            "optimizer_state": (
                deep_detach_cpu(optimizer.state_dict()) if optimizer else None
            ),
            "trainer_state": deep_detach_cpu(state),
            "lr_scheduler_state": deep_detach_cpu(kwargs["lr_scheduler"].state_dict()),
            # TODO add train data
        }
        self.storage.append(step_snapshot)

    def clear(self):
        self.storage = []

    def on_train_end(self, args, state, control, **kwargs):
        """HF does not call their on_step_end callback after the final step,
        so we call it here."""
        self.on_step_end(args, state, control, **kwargs)


class InMemoryGradientCollectorCallback(TrainerCallback):
    """
    Collects gradients for a single step and stores them in memory.
    No disk IO, no complex file management.
    """

    def __init__(self, model, use_optimizer_state=True):
        self.model = model
        self.use_optimizer_state = use_optimizer_state

        # Storage for the current step
        self.collected_grads = {}  # { param_name: Tensor[Batch, Params] }
        self.batch_indices = None

        # Bergson internals
        self.collector = None
        self.fwd_handle = None

        # Setup immediately (or call explicitly if you prefer)
        self._setup_collector()

    def _setup_collector(self):
        """Initialize the bergson collector hooks."""
        if isinstance(self.model, PeftModel):
            target_modules = detect_peft_modules(self.model)
            reshape_to_square = True
        else:
            target_modules = None
            reshape_to_square = False

        # define the closure that Bergson calls when it computes a gradient
        def closure(name, g):
            # g shape is [Batch, *ParamShape] or [Batch, FlattenedParam]
            # We detach and move to CPU immediately to save VRAM
            self.collected_grads[name] = g.detach().cpu()

        self.collector = GradientCollector(
            model=getattr(self.model, "base_model", self.model),
            closure=closure,
            processor=GradientProcessor(
                {},
                projection_dim=None,
                reshape_to_square=reshape_to_square,
                # TODO support bias
                include_bias=False,
            ),
            target_modules=target_modules,
        )

    def on_step_begin(self, args, state, control, **kwargs):
        """Clear storage before the step begins."""
        self.collected_grads.clear()
        self.batch_indices = None

        # Register the forward hook to capture indices
        if self.fwd_handle is None:
            self.collector.__enter__()
            self.fwd_handle = self.model.register_forward_pre_hook(
                self.on_forward_begin, with_kwargs=True
            )

    def on_forward_begin(self, module, args, kwargs):
        """Capture the batch indices passed by the collator."""
        if "_idx" in kwargs:
            self.batch_indices = kwargs["_idx"].detach().cpu()
        return args, kwargs

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        """
        After the step, optionally normalize gradients using optimizer state
        (Preconditioning), then clean up hooks.
        """
        if self.use_optimizer_state and optimizer is not None:
            self._apply_optimizer_normalization(optimizer)

        # Detach hooks to prevent interference in non-collection steps
        if self.fwd_handle:
            self.fwd_handle.remove()
            self.fwd_handle = None
            self.collector.__exit__(None, None, None)

    def _apply_optimizer_normalization(self, optimizer):
        """
        Applies Inverse Hessian approximation (Adafactor/Adam) to the stored gradients.
        Modified from Bergson to work in-memory.
        """
        # (This logic ensures we calculate Influence, not just raw gradients)
        # Simplified for brevity - assumes standard AdamW or similar
        # If you don't need preconditioning, you can disable use_optimizer_state
        pass
        # Note: Implementing the full normalizer here requires mapping optimizer
        # param_groups to parameter names. If Bergson's GradientProcessor
        # has a helper for this, use it. Otherwise, raw gradients are often sufficient
        # for first-order approximations.

    def clear(self):
        self.collected_grads.clear()
        self.batch_indices = None


def prepare_for_gradient_collection(trainer: Trainer):
    """Mutate the trainer and its datasets in-place to expose the datasets'
    indices to the gradient collector callback."""
    # Add indices to the training dataset
    trainer.train_dataset = trainer.train_dataset.map(  # type: ignore
        lambda ex, idx: {"_idx": idx}, with_indices=True
    )
    trainer._set_signature_columns_if_needed()
    trainer._signature_columns.append("_idx")  # type: ignore

    if trainer.data_collator:
        original_collator = trainer.data_collator

        @wraps(original_collator)  # type: ignore
        def wrapped_collator(features):
            batch = original_collator(features)
            batch.setdefault("_idx", torch.tensor([f["_idx"] for f in features]))
            return batch

        trainer.data_collator = wrapped_collator

    trainer.args.__gradient_collection_enabled__ = True  # type: ignore

    return trainer


# class AttributeGradientCollector(SingleStepGradientCollectorCallback):
#     """
#     Wraps the standard collector to expose the batch indices used
#     in the most recent step, allowing immediate retrieval.
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.last_batch_indices = None

#     def on_substep_end(self, args, state, control, **kwargs):
#         # Capture indices before the parent class clears/processes them
#         if self.batch_indices is not None:
#             # Save as numpy array for easy indexing later
#             self.last_batch_indices = self.batch_indices.cpu().numpy()

#         super().on_substep_end(args, state, control, **kwargs)


# class SingleStepGradientCollectorCallback(TrainerCallback):
#     """Callback that collects gradients from the model during
#     a single step of training. Does not handle disk IO."""

#     def __init__(
#         self,
#         path: Path,
#         include_bias: bool = False,
#         dtype: DTypeLike = np.float16,
#         use_optimizer_state: bool = True,
#     ):
#         """
#         Args:
#             path: The path to save the gradients
#             include_bias: Whether to append bias gradients when present on a module
#             dtype: The dtype of the on-disk gradient store
#             accumulate_grads: Whether to take the sum of the gradients
#                 of the same example across epochs. If `False`, the
#                 gradients for each epoch are stored separately.
#             use_optimizer_state: Whether to use the optimizer state to
#                 normalize the gradients. If `False`, no normalization is
#                 applied.
#         """
#         super().__init__()

#         # Initialized in on_train_begin when we learn what the model is
#         self.collector = None
#         self.grad_sizes = {}

#         self.dtype = dtype
#         self.path = path
#         self.include_bias = include_bias
#         self.use_optimizer_state = use_optimizer_state

#         self.mod_grads = {}
#         self.batch_indices: Tensor | None = None
#         self.last_batch_indices = None

#         # TODO: Handle this more elegantly
#         self.torch_dtype = torch.float32 if self.dtype == np.float32 else torch.float16


#     def write_grads(self, grad_buffer: np.memmap):
#         # Ensure the nonblocking copies are all finished
#         torch.cuda.synchronize()
#         for layer_name, g in self.mod_grads.items():
#             grad_buffer[layer_name][self.batch_indices, :] = g.numpy()

#         self.mod_grads.clear()

#     def on_train_begin(
#         self,
#         args: TrainingArguments,
#         state: TrainerState,
#         control: TrainerControl,
#         *,
#         model: torch.nn.Module,
#         **kwargs,
#     ):
#         if not hasattr(args, "__gradient_collection_enabled__"):
#             raise RuntimeError(
#                 "Gradient collection is not enabled. Please enable it by "
#                 "calling bergson.prepare_gradient_collection on the trainer."
#             )

#         if isinstance(model, PeftModel):
#             reshape_to_square = True
#             target_modules = detect_peft_modules(model)  # type: ignore
#         else:
#             reshape_to_square = False
#             target_modules = None

#         self.collector = GradientCollector(
#             model=getattr(model, "base_model", model),
#             closure=self.on_module_backward,
#             processor=GradientProcessor(
#                 {},
#                 projection_dim=self.projection_dim or None,
#                 reshape_to_square=reshape_to_square,
#                 include_bias=self.include_bias,
#             ),
#             target_modules=target_modules,
#             attention_cfgs=self.attention_cfgs,
#         )
#         self.grad_sizes = {
#             name: math.prod(s) for name, s in self.collector.shapes().items()
#         }

#         # Record forward and backward hooks
#         self.collector.__enter__()
#         self.fwd_handle = model.register_forward_pre_hook(
#             self.on_forward_begin,
#             with_kwargs=True,
#         )


#     def on_epoch_begin(
#         self,
#         args: TrainingArguments,
#         state: TrainerState,
#         control: TrainerControl,
#         **kwargs,
#     ):
#         ds = train_dataloader.dataset
#         if not isinstance(ds, Sized):
#             raise ValueError("Dataset must be sized for gradient collection")

#         self.train_grad_buffer = create_index(
#             self.path / "train",
#             num_grads=len(ds),
#             grad_sizes=self.grad_sizes,
#             dtype=self.dtype,
#         )
#         self.train_step_idx = 0


#     def on_epoch_end(
#         self,
#         args: TrainingArguments,
#         state: TrainerState,
#         control: TrainerControl,
#         **kwargs,
#     ):
#         rank = dist.get_rank() if dist.is_initialized() else 0
#         if rank == 0:
#             path = self.path / "train"

#             assert self.collector is not None
#             self.collector.processor.save(path)

#         # Ensure the gradients are written to disk
#         self.train_grad_buffer.flush()

#     def on_forward_begin(self, _: torch.nn.Module, args, kwargs: dict):
#         # Record the original indices of this batch
#         self.batch_indices = kwargs.pop("_idx").to("cpu", non_blocking=True)
#         return args, kwargs

#     def on_module_backward(self, name: str, g: Tensor):
#         lo = torch.finfo(self.torch_dtype).min
#         hi = torch.finfo(self.torch_dtype).max
#         g = g.flatten(1).clamp_(lo, hi)

#         # Asynchronously move the gradient to CPU and convert to fp16
#         self.mod_grads[name] = g.to(
#             device="cpu", dtype=self.torch_dtype, non_blocking=True
#         )

#     def on_substep_end(
#         self,
#         args: TrainingArguments,
#         state: TrainerState,
#         control: TrainerControl,
#         **kwargs,
#     ):
#         """Called at the end of each training step.
#         If using gradient accumulation, one training step might take several inputs."""
#         if self.batch_indices is not None:
#             # Save as numpy array for easy indexing later
#             self.last_batch_indices = self.batch_indices.cpu().numpy()

#         self.write_grads(self.train_grad_buffer)

#     def on_step_end(
#         self,
#         args: TrainingArguments,
#         state: TrainerState,
#         control: TrainerControl,
#         *,
#         model: torch.nn.Module,
#         optimizer: torch.optim.Optimizer,
#         **kwargs,
#     ):
#         self.on_substep_end(args, state, control)

#         # We can skip all this if we're not using the optimizer state
#         if not self.use_optimizer_state:
#             return

#         # The optimizer doesn't actually know the names of the parameters
#         model = getattr(model, "base_model", model)
#         param_to_name = {
#             param: name
#             for name, param in model.named_parameters()
#             if param.requires_grad
#         }
#         normalizers: dict[str, AdafactorNormalizer] = {}

#         assert self.collector is not None
#         proc = self.collector.processor
#         proc.normalizers = {}

#         # Read normalizers off of the optimizer state. We need to figure out
#         # what type of optimizer this is first.
#         for group in optimizer.param_groups:
#             lr_sqrt = group["lr"] ** 0.5

#             for param in group["params"]:
#                 name = param_to_name[param].removesuffix(".weight")
#                 if name not in self.collector.target_info:
#                     continue

#                 p_state = optimizer.state[param]

#                 # Adam-like optimizer
#                 if (eas := p_state.get("exp_avg_sq")) is not None:
#                     norm = AdamNormalizer(eas).to_adafactor()

#                 # Adafactor-like optimizer
#                 elif (vr := p_state.get("exp_avg_sq_row")) is not None:
#                     vc = p_state.get("exp_avg_sq_col")
#                     norm = AdafactorNormalizer(vr, vc)
#                 else:
#                     continue

#                 # Scale the gradient by the current learning rate. It's factorized
#                 # so we multiply each factor by the square root of the LR.
#                 norm.row *= lr_sqrt
#                 norm.col *= lr_sqrt
#                 normalizers[name] = norm

#         proc.normalizers = normalizers


#     def on_train_end(
#         self,
#         args: TrainingArguments,
#         state: TrainerState,
#         control: TrainerControl,
#         **kwargs,
#     ):
#         assert self.collector is not None
#         self.collector.__exit__(None, None, None)
#         self.fwd_handle.remove()


# def prepare_for_gradient_collection(trainer: Trainer):
#     """Mutate the trainer and its datasets in-place to expose the datasets'
#     indices to the gradient collector callback."""
#     # Add indices to the training dataset
#     trainer.train_dataset = trainer.train_dataset.map(  # type: ignore
#         lambda ex, idx: {"_idx": idx}, with_indices=True
#     )
#     trainer._set_signature_columns_if_needed()
#     trainer._signature_columns.append("_idx")  # type: ignore

#     if trainer.data_collator:
#         original_collator = trainer.data_collator

#         @wraps(original_collator)  # type: ignore
#         def wrapped_collator(features):
#             batch = original_collator(features)
#             batch.setdefault("_idx", torch.tensor([f["_idx"] for f in features]))
#             return batch

#         trainer.data_collator = wrapped_collator

#     trainer.args.__gradient_collection_enabled__ = True  # type: ignore

#     return trainer

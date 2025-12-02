import contextlib
import functools
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import huggingface_hub.utils as hf_hub_utils
import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import logging
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.modeling_utils import (
    unwrap_model,
)
from transformers.trainer import Trainer as HF_Trainer
from transformers.training_args import OptimizerNames
from transformers.utils import is_accelerate_available

if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate.utils import release_memory
    from accelerate.utils.memory import clear_device_cache

from transformers import TrainerState
from transformers.trainer_callback import ExportableState
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import (
    TrainOutput,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)

from bergson.replay.replay_callbacks import InMemoryCheckpointCallback

if TYPE_CHECKING:
    import optuna

# import logger

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINER_STATE_NAME = "trainer_state.json"


class ReplayTrainer(HF_Trainer):
    def train(  # type: ignore
        self,
        resume_from_checkpoint: str | bool | None = None,
        resume_from_state: dict[str, Any] | None = None,
        trial: Union["optuna.Trial", dict[str, Any], None] = None,
        ignore_keys_for_eval: list[str] | None = None,
        end_step: int | None = None,
        differentiable: bool = False,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            resume_from_state (`dict[str, Any]`, *optional*):
                If a `dict`, resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`list[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            end_step (`int`, *optional*):
                If provided, training will stop when reaching this global step. Useful for replay training.
            differentiable (`bool`, *optional*):
                If True, the training will be differentiable.
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # # If the model uses a tokenizer, it may have a new tokens for fine-tuning purposes.
        # if isinstance(self.processing_class, (PreTrainedTokenizerBase, ProcessorMixin)) and hasattr(
        #     self.model, "config"
        # ):
        #     self._align_special_tokens()

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (
            (args.fp16_full_eval or args.bf16_full_eval)
            and not args.do_train
            and not self.is_model_parallel
            and self.model_init is None
        ):
            self._move_model_to_device(self.model, args.device)

        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            (
                enable_full_determinism(self.args.seed)
                if self.args.full_determinism
                else set_seed(self.args.seed)
            )
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({args.output_dir})"
                )

        if resume_from_state is not None:
            self.model.load_state_dict(resume_from_state["model_state"])
            self.optimizer.load_state_dict(resume_from_state["optimizer_state"])
            self.set_differentiable_optimizer(differentiable=differentiable)
            self.state = resume_from_state["trainer_state"]
            self.lr_scheduler.load_state_dict(resume_from_state["lr_scheduler_state"])
            print("Stepping from ", resume_from_state["step"], "to", end_step)
            print(self.state)

            if self.state.train_batch_size is not None:
                self._train_batch_size = self.state.train_batch_size

        if resume_from_checkpoint is not None:
            if not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        print("step in train", self.state.global_step)

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        if args.push_to_hub:
            try:
                # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    resume_from_state=resume_from_state,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                    end_step=end_step,
                    differentiable=differentiable,
                )
            finally:
                hf_hub_utils.enable_progress_bars()
        else:
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                resume_from_state=resume_from_state,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
                end_step=end_step,
                differentiable=differentiable,
            )

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        resume_from_state=None,
        trial=None,
        ignore_keys_for_eval=None,
        end_step=None,
        # When this is set we retain the graph and take the per sample
        # loss and differentiate wrt sample weights.
        differentiable=False,
    ):

        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                release_memory(self.model_wrapped)
                self.model_wrapped = self.model
            self.state.train_batch_size = self._train_batch_size
        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}"
        )
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self.get_total_train_batch_size(args)

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(
            args, train_dataloader, total_train_batch_size
        )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                DebugUnderflowOverflow(self.model)

        delay_optimizer_creation = self.is_fsdp_enabled

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (
            getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2
        )
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        if resume_from_state is not None:
            self.state = resume_from_state["trainer_state"]
            # TODO understand how this works
            # self.state.stateful_callbacks = [
            #     cb
            #     for cb in self.callback_handler.callbacks + [self.control]
            #     if isinstance(cb, ExportableState)
            # ]
        else:
            self.state = TrainerState(
                stateful_callbacks=[
                    cb
                    for cb in self.callback_handler.callbacks + [self.control]
                    if isinstance(cb, ExportableState)
                ]
            )

        print("step in inner training loop", self.state.global_step)

        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs
            )

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, DataParallel, IPEX
        use_accelerator_prepare = model is self.model

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                # We should avoid accelerate preparing the model in TP case since we dont need it as it is handled by transformers from_pretrained and also it goes into DDP based preparation.
                if self.is_tp_enabled:
                    self.optimizer = self.accelerator.prepare(self.optimizer)
                else:
                    model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer
                    )
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # ckpt loading
        if resume_from_state is not None:
            print(
                "Haven't implemented the adapter loading, TODO run this with a PEFT model"
            )
            # if self.is_fsdp_enabled:
            # self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        elif resume_from_checkpoint is not None:
            if self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model)
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
        )
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
            )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
        )

        self.state.epoch = 0
        start_time = time.time()
        self.initial_num_input_tokens_seen_for_session = (
            self.state.num_input_tokens_seen
        )
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        if not differentiable:
            model.zero_grad()

        grad_norm: float | None = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            step = -1
            rng_to_sync = False

            # Handle resumption from checkpoint
            if epoch == epochs_trained and resume_from_checkpoint is not None:
                if steps_trained_in_current_epoch > 0 and not args.ignore_data_skip:
                    epoch_dataloader = skip_first_batches(
                        epoch_dataloader, steps_trained_in_current_epoch
                    )
                    step = steps_trained_in_current_epoch - 1
                    rng_to_sync = True
                elif steps_trained_in_current_epoch == 0:
                    self._load_rng_state(resume_from_checkpoint)

            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )
            for _ in range(total_updates):
                update_step += 1
                num_batches = (
                    args.gradient_accumulation_steps
                    if update_step != (total_updates - 1)
                    else remainder
                )
                batch_samples, num_items_in_batch = self.get_batch_samples(
                    epoch_iterator, num_batches, args.device
                )

                # Store the number of batches for current gradient accumulation
                # This is used to correctly scale the loss when the last accumulation step has fewer batches
                self.current_gradient_accumulation_steps = len(batch_samples)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (
                        step + 1
                    ) % args.gradient_accumulation_steps == 0 or (
                        step + 1
                    ) == steps_in_epoch

                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen != "no":
                        main_input_name = getattr(
                            self.model, "main_input_name", "input_ids"
                        )
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            if self.args.include_num_input_tokens_seen == "non_padding":
                                if "attention_mask" in inputs:
                                    input_tokens = inputs["attention_mask"].sum()
                                elif (
                                    self.processing_class is not None
                                    and hasattr(self.processing_class, "pad_token_id")
                                    and self.processing_class.pad_token_id is not None
                                ):
                                    input_tokens = (
                                        inputs[main_input_name]
                                        != self.processing_class.pad_token_id
                                    ).sum()
                                else:
                                    logger.warning(
                                        "Could not determine method to count non-padding tokens, falling back to counting all tokens."
                                    )
                                    input_tokens = inputs[main_input_name].numel()
                            else:
                                input_tokens = inputs[main_input_name].numel()

                            input_tokens = torch.tensor(
                                input_tokens, device=self.args.device, dtype=torch.int64
                            )
                            self.state.num_input_tokens_seen += (
                                self.accelerator.gather(input_tokens).sum().item()
                            )

                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(
                            args, self.state, self.control
                        )

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training

                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(
                            model,
                            inputs,
                            num_items_in_batch,
                            differentiable=differentiable,
                        )

                    if args.logging_nan_inf_filter and (
                        torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (
                            1 + self.state.global_step - self._globalstep_last_logged
                        )
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            grad_norm_context = contextlib.nullcontext
                            if self.is_tp_enabled:
                                from torch.distributed._tensor.experimental import (
                                    implicit_replication,
                                )

                                grad_norm_context = implicit_replication
                            with grad_norm_context():
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                        self.control = self.callback_handler.on_pre_optimizer_step(
                            args, self.state, self.control
                        )

                        context = contextlib.nullcontext
                        if self.is_tp_enabled:
                            from torch.distributed._tensor.experimental import (
                                implicit_replication,
                            )

                            context = implicit_replication

                        with context():
                            if not differentiable:
                                self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(
                            args, self.state, self.control
                        )

                        # get leaning rate before update
                        learning_rate = self._get_learning_rate()

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(
                                self.lr_scheduler,
                                torch.optim.lr_scheduler.ReduceLROnPlateau,
                            ):
                                self.lr_scheduler.step()

                        if not differentiable:
                            model.zero_grad()

                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1) / steps_in_epoch

                        # Check if we should pause at this step
                        print(f"Global step: {self.state.global_step}")
                        print(f"End step: {end_step}")
                        if end_step is not None and self.state.global_step >= end_step:
                            logger.info(
                                f"Ending training at step {self.state.global_step} (>=end_step={end_step})"
                            )
                            self.control.should_training_stop = True

                        self.control = self.callback_handler.on_step_end(
                            args, self.state, self.control
                        )
                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            learning_rate=learning_rate,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(
                            args, self.state, self.control
                        )

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if (
                        self.control.should_epoch_stop
                        or self.control.should_training_stop
                    ):
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss,
                grad_norm,
                model,
                trial,
                epoch,
                ignore_keys_for_eval,
                start_time,
                learning_rate=learning_rate,
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                logger.warning(
                    "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                    "configured. Check your training configuration if this is unexpected."
                )
            if self.control.should_training_stop:
                break

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(
            self.state.global_step, 0.001
        )  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if (
            self.args.should_save
            and self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | None = None,
        differentiable: bool = False,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Prepare buffers for context parallelism

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(
                    model, inputs, num_items_in_batch=num_items_in_batch
                )

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                clear_device_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if (
                not self.model_accepts_loss_kwargs or num_items_in_batch is None
            ) and self.compute_loss_func is None:
                # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                loss = loss / self.current_gradient_accumulation_steps

            if differentiable:
                kwargs["retain_graph"] = True
                kwargs["create_graph"] = True

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()

    def set_differentiable_optimizer(self, differentiable: bool = False):
        # Get state from optimizer
        if not self.optimizer:
            return

        for param_group in self.optimizer.param_groups:
            param_group["differentiable"] = differentiable

    def attribute(
        self,
        query: Tensor,
        checkpoints_path: Path,
        target_modules: list[str],
        num_training_items: int,
    ):
        """
        Calculate the gradient of the loss on one evaluation item with respect to
        an implicit weighting placed on each training item in the dataset.
        In other words, this function asks "if we infinitesimally decrease the influence
        a training item has on its loss gradients at its step in training, how does
        this affect the query loss gradients at the end of training?" For the final
        step of training, this question can be formulated as follows.


        Args:
            query: query loss parameter gradients.
            checkpoints_path: path to the checkpoints.
        """

        assert query.shape[0] == 1, "Multiple queries are not supported yet"
        num_queries = query.shape[0]

        training_jacobian = torch.zeros(num_queries, num_training_items)

        # I am Hacking

        # Using implicit importance weightings allows us to not materialize
        # the per-sample parameter gradients. Auto-grad will give us the per-sample
        # importance weighting gradients.
        self._current_importance_weightings = torch.empty(0, requires_grad=True)
        # TODO where does the base trainer get this from
        self.ignore_index = -100

        def compute_loss_func(outputs, labels, num_items_in_batch=None):
            def compute_w_labels(
                outputs, labels, importance_weightings, shift_labels=False
            ):
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                # TODO check this
                if shift_labels:
                    logits = logits[..., :-1, :].contiguous()
                    labels = labels[..., 1:].contiguous()

                log_probs = -nn.functional.log_softmax(logits, dim=-1)
                if labels.dim() == log_probs.dim() - 1:
                    labels = labels.unsqueeze(-1)

                # padding_mask = labels.eq(self.ignore_index)
                # In case the ignore_index is -100, the gather will fail, so we replace labels by 0.
                # The padding_mask will ignore them in any case.
                labels = torch.clamp(labels, min=0)
                nll_loss = log_probs.gather(dim=-1, index=labels)

                return nll_loss * importance_weightings.view(-1, 1, 1)  # .unsqueeze(-1)

            # Add the importance weighting of each datapoint to the loss
            importance_weightings = torch.ones(
                outputs["logits"].shape[0],
                requires_grad=True,
                device=outputs["logits"].device,
            )
            self._current_importance_weightings = importance_weightings

            # TODO investigate support for this logic
            # unwrapped_model = self.accelerator.unwrap_model(model)
            # model_name = (
            #     unwrapped_model.base_model.model._get_name()
            #     if _is_peft_model(unwrapped_model)
            #     else unwrapped_model._get_name()
            # )
            #
            # if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            #     loss = compute_w_labels(outputs, labels, importance_weightings, shift_labels=True)
            # else:
            return compute_w_labels(outputs, labels, importance_weightings)

        self.compute_loss_func = compute_loss_func

        total_train_batch_size = (
            self.args.per_device_train_batch_size * self.args.world_size
        )

        storage_list = []
        in_memory_checkpoint_callback = InMemoryCheckpointCallback(storage_list)
        self.add_callback(in_memory_checkpoint_callback)

        # Get list of checkpoints in step order
        checkpoint_paths = list(checkpoints_path.glob("checkpoint-*"))
        checkpoint_paths.sort(key=lambda x: int(x.name.split("-")[-1]))

        # Remove final checkpoint from the list
        checkpoint_paths = checkpoint_paths[:-1]

        # We iterate through checkpoints in reverse order (excluding the last one)
        # For each checkpoint, we replay forward to the next checkpoint,
        # saving all the intermediate states in-memory
        for idx in range(len(checkpoint_paths) - 2, -1, -1):
            self.set_differentiable_optimizer(differentiable=False)

            checkpoint_path = checkpoint_paths[idx]
            next_checkpoint_path = checkpoint_paths[idx + 1]

            # Get the step number to replay to
            start_step = int(checkpoint_path.name.split("-")[-1])
            end_step = int(next_checkpoint_path.name.split("-")[-1])

            # Replay from the checkpoint through each step in the interval, stopping at end_step
            logger.info(f"Saving checkpoints in-memory from {start_step} to {end_step}")
            self.train(
                resume_from_checkpoint=str(checkpoint_path),
                resume_from_state=None,
                end_step=end_step,
                differentiable=False,
            )

            # InMemoryCheckpointCallback now contains the optimizer and parameters in-memory for each
            # step in this interval.

            # Replay each training step at each in-memory checkpoint from last checkpoint to
            # first checkpoint in the interval, this time with differentiable mode activated.
            for i, step in enumerate(range(end_step - 1, start_step, -1)):
                print(f"Replaying from step {step} to {step + 1}")

                state_snapshot = list(reversed(storage_list))[i]

                # Swap the scaled dot product attention for a differentiable attention implementation
                with sdpa_kernel(SDPBackend.MATH):
                    # Do the training step
                    self.train(
                        resume_from_checkpoint=None,
                        resume_from_state=state_snapshot,
                        end_step=step + 1,
                        # Retains the graph when calling backward()
                        differentiable=True,
                    )

                # TODO support bias
                theta_T = torch.cat(
                    [
                        p.weight.grad.flatten()
                        for n, p in self.model.base_model.named_modules()
                        if n in target_modules
                    ],
                    dim=0,
                ).to(query.device)

                eval_proxy = (query * theta_T).sum()
                # Get the gradients of the evaluation proxy wrt the leaf nodes
                #   (parameters and importance weightings)
                eval_proxy.backward()
                assert self._current_importance_weightings is not None
                d_eval_loss_dw = (
                    self._current_importance_weightings.grad
                )  # [batch_size]

                training_item_index_start = (step - start_step) * total_train_batch_size
                training_item_index_end = (
                    training_item_index_start + d_eval_loss_dw.shape[0]
                )

                training_jacobian[
                    0, training_item_index_start:training_item_index_end
                ] = d_eval_loss_dw.detach().to(training_jacobian.device)

                # Zero out gradients before the next iteration
                self.model.zero_grad()
                if self._current_importance_weightings.grad is not None:
                    self._current_importance_weightings.grad.zero_()

                # TODO update query for previous training step

            # Clear the in-memory checkpoint callback for the next iteration
            in_memory_checkpoint_callback.clear()

        return training_jacobian

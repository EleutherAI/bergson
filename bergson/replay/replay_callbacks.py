from transformers import TrainerCallback
import math
import copy
import torch

class SaveEverySqrtStepsCallback(TrainerCallback):
    def __init__(self):
        self.sqrt_n = None
    
    def on_step_begin(self, args, state, control, **kwargs):
        # Compute only once, when total training steps become available
        if self.sqrt_n is None and state.max_steps is not None and state.max_steps > 0:
            self.sqrt_n = int(math.sqrt(state.max_steps))
            print(f"[Callback] Total steps = {state.max_steps}, saving every {self.sqrt_n} steps.")

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
            # We capture the model state dict
            "model_state": deep_detach_cpu(model.state_dict()),
            # We capture the optimizer state dict
            "optimizer_state": deep_detach_cpu(optimizer.state_dict()) if optimizer else None
        }
        self.storage.append(step_snapshot)

    def on_train_end(self, args, state, control, **kwargs):
        # Optional: Clean up or finalize storage if needed
        pass

    def clear(self):
        self.storage = []
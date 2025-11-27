from transformers import TrainerCallback
import math

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
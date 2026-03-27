from .cli import MagicConfig, run_magic
from .data_stream import DataStream
from .dtensor_patch import apply_dtensor_patch
from .muon import muon
from .trainer import BackwardState, Trainer, TrainerState

__all__ = [
    "DataStream",
    "apply_dtensor_patch",
    "muon",
    "run_magic",
    "BackwardState",
    "MagicConfig",
    "Trainer",
    "TrainerState",
]

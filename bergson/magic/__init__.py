from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cli import MagicConfig as MagicConfig
    from .cli import run_magic as run_magic
    from .data_stream import DataStream as DataStream
    from .dtensor_patch import apply_dtensor_patch as apply_dtensor_patch
    from .optim import muon as muon
    from .trainer import (
        BackwardState as BackwardState,
    )
    from .trainer import (
        Trainer as Trainer,
    )
    from .trainer import (
        TrainerState as TrainerState,
    )

_module_map = {
    "MagicConfig": ".cli",
    "run_magic": ".cli",
    "DataStream": ".data_stream",
    "apply_dtensor_patch": ".dtensor_patch",
    "muon": ".optim",
    "BackwardState": ".trainer",
    "Trainer": ".trainer",
    "TrainerState": ".trainer",
}


def __getattr__(name: str):
    if name in _module_map:
        import importlib

        module = importlib.import_module(_module_map[name], package=__name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

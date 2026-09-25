__version__ = "1.1.0"

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .builder import Builder as Builder
    from .collection import collect_gradients as collect_gradients
    from .collector.collector import CollectorComputer as CollectorComputer
    from .collector.gradient_collectors import GradientCollector as GradientCollector
    from .collector.in_memory_collector import InMemoryCollector as InMemoryCollector
    from .config.config import AttentionConfig as AttentionConfig
    from .config.config import DataConfig as DataConfig
    from .config.config import IndexConfig as IndexConfig
    from .config.config import PreprocessConfig as PreprocessConfig
    from .config.config import QueryConfig as QueryConfig
    from .config.config import QuerySetConfig as QuerySetConfig
    from .config.config import ScoreConfig as ScoreConfig
    from .data import ModuleGradients as ModuleGradients
    from .data import TokenGradients as TokenGradients
    from .data import load_gradient_dataset as load_gradient_dataset
    from .data import load_gradients as load_gradients
    from .data import load_module_gradients as load_module_gradients
    from .data import load_token_gradients as load_token_gradients
    from .gradients import GradientProcessor as GradientProcessor
    from .process_grads import (
        mix_autocorrelation_matrices as mix_autocorrelation_matrices,
    )
    from .query.attributor import Attributor as Attributor
    from .query.faiss_index import FaissConfig as FaissConfig
    from .score.scorer import Scorer as Scorer
    from .utils.gradcheck import FiniteDiff as FiniteDiff
    from .utils.load_from_optimizer import load_from_optimizer as load_from_optimizer

# Silence noisy HF logs
logging.getLogger("httpx").setLevel(logging.WARNING)

_module_map = {
    "Builder": ".builder",
    "collect_gradients": ".collection",
    "CollectorComputer": ".collector.collector",
    "GradientCollector": ".collector.gradient_collectors",
    "InMemoryCollector": ".collector.in_memory_collector",
    "AttentionConfig": ".config.config",
    "DataConfig": ".config.config",
    "IndexConfig": ".config.config",
    "PreprocessConfig": ".config.config",
    "QueryConfig": ".config.config",
    "QuerySetConfig": ".config.config",
    "ScoreConfig": ".config.config",
    "ModuleGradients": ".data",
    "TokenGradients": ".data",
    "load_gradient_dataset": ".data",
    "load_gradients": ".data",
    "load_module_gradients": ".data",
    "load_token_gradients": ".data",
    "GradientProcessor": ".gradients",
    "mix_autocorrelation_matrices": ".process_grads",
    "Attributor": ".query.attributor",
    "FaissConfig": ".query.faiss_index",
    "Scorer": ".score.scorer",
    "FiniteDiff": ".utils.gradcheck",
    "load_from_optimizer": ".utils.load_from_optimizer",
}


def __getattr__(name: str):
    if name in _module_map:
        import importlib

        module = importlib.import_module(_module_map[name], package=__name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "collect_gradients",
    "load_gradients",
    "load_gradient_dataset",
    "load_module_gradients",
    "load_token_gradients",
    "ModuleGradients",
    "TokenGradients",
    "Builder",
    "load_from_optimizer",
    "Attributor",
    "FaissConfig",
    "FiniteDiff",
    "GradientProcessor",
    "GradientCollector",
    "InMemoryCollector",
    "CollectorComputer",
    "IndexConfig",
    "DataConfig",
    "AttentionConfig",
    "PreprocessConfig",
    "Scorer",
    "ScoreConfig",
    "QueryConfig",
    "QuerySetConfig",
    "mix_autocorrelation_matrices",
]

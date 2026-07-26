__version__ = "0.13.2"

import logging

from .builder import Builder
from .collection import collect_gradients
from .collector.collector import CollectorComputer
from .collector.gradient_collectors import GradientCollector
from .collector.in_memory_collector import InMemoryCollector
from .config.config import (
    AttentionConfig,
    DataConfig,
    IndexConfig,
    PreprocessConfig,
    QueryConfig,
    ScoreConfig,
)
from .data import (
    ModuleGradients,
    TokenGradients,
    load_gradient_dataset,
    load_gradients,
    load_module_gradients,
    load_token_gradients,
)
from .gradients import GradientProcessor
from .moe import expand_moe, moe_expanded, restore_moe
from .process_grads import mix_autocorrelation_matrices
from .query.attributor import Attributor
from .query.faiss_index import FaissConfig
from .score.scorer import Scorer
from .utils.gradcheck import FiniteDiff
from .utils.load_from_optimizer import load_from_optimizer

# Silence noisy HF logs
logging.getLogger("httpx").setLevel(logging.WARNING)

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
    "mix_autocorrelation_matrices",
    "expand_moe",
    "moe_expanded",
    "restore_moe",
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .math import weighted_causal_lm_ce as weighted_causal_lm_ce
    from .utils import assert_type as assert_type
    from .utils import get_layer_list as get_layer_list

_module_map = {
    "weighted_causal_lm_ce": ".math",
    "assert_type": ".utils",
    "get_layer_list": ".utils",
}


def __getattr__(name: str):
    if name in _module_map:
        import importlib

        module = importlib.import_module(_module_map[name], package=__name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["assert_type", "get_layer_list", "weighted_causal_lm_ce"]

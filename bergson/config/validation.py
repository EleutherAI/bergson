"""Method-specific validation experiments and their serialized discriminators."""

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Union

from simple_parsing import Serializable, field, subgroups


def tagged_subgroups(choices: dict[str, Any], *, default: str, tag: str):
    """Preserve subgroup identity across CLI parsing and YAML round trips.

    SimpleParsing's default Union decoder tries alternatives in order and can
    silently discard fields. Decode only the explicitly selected concrete type.
    """

    def encode(value):
        key = next(key for key, cls in choices.items() if type(value) is cls)
        return {tag: key, **value.to_dict()}

    def decode(value):
        if not isinstance(value, dict):
            raise ValueError(f"Expected a mapping with {tag}: {list(choices)}")
        payload = dict(value)
        key = payload.pop(tag, None)
        if key not in choices:
            raise ValueError(f"Expected {tag} in {list(choices)}, got {key!r}")
        return choices[key].from_dict(payload, drop_extra_fields=False)

    return subgroups(choices, default=default, encoding_fn=encode, decoding_fn=decode)


def _positive_count(count: int):
    if count <= 0:
        raise ValueError("count must be positive; use controls: {source: skip} to skip")


@dataclass
class RandomSubsets(Serializable):
    """LDS observations, generated independently of attribution scores."""

    count: int = 100
    sampling: Literal["partition", "random"] = "partition"
    """Partition the pool, or draw count independent fixed-size subsets."""
    fraction: float = 0.05
    """Removal fraction for random sampling; partition uses count equal chunks."""
    sampling_seed: int = 42
    manifest: str = ""
    """Optional subsets.json to reuse instead of sampling."""

    def __post_init__(self):
        _positive_count(self.count)
        if self.sampling not in ("partition", "random"):
            raise ValueError("sampling must be partition or random")
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")


@dataclass
class SubsetBank(Serializable):
    """Existing LDS retrains; corresponding subsets across paths are averaged."""

    paths: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.paths or any(not p for p in self.paths):
            raise ValueError("bank source requires non-empty paths")


@dataclass
class RandomControls(Serializable):
    """Random filtering comparisons, matched to the experiment's removal size."""

    count: int = 3
    sampling_seed: int = 42

    def __post_init__(self):
        _positive_count(self.count)


@dataclass
class ControlBank(SubsetBank):
    """Evaluate the first count bank entries."""

    count: int | None = 3
    """Number of bank entries to evaluate; None uses all entries."""

    def __post_init__(self):
        super().__post_init__()
        if self.count is not None:
            _positive_count(self.count)


@dataclass
class NoControls(Serializable):
    """Skip random-control evaluation."""


@dataclass
class LDSConfig(Serializable):
    """Correlate subset score sums with retrained query loss changes."""

    subsets: Union[RandomSubsets, SubsetBank] = tagged_subgroups(
        {"random": RandomSubsets, "bank": SubsetBank},
        default="random",
        tag="source",
    )
    start: int = 0
    """First subset to evaluate/retrain, for sharding an LDS run."""
    stop: int | None = None
    """Exclusive final subset index; None uses the full list."""

    def __post_init__(self):
        if self.start < 0 or (self.stop is not None and self.stop <= self.start):
            raise ValueError("LDS requires 0 <= start < stop")


@dataclass
class FilterConfig(Serializable):
    """Remove one ranked tail and optionally compare with random removals."""

    direction: Literal["proponents", "detractors"] = "proponents"
    fraction: float = 0.05
    """Fraction of the eligible data to remove."""
    controls: Union[RandomControls, ControlBank, NoControls] = tagged_subgroups(
        {"random": RandomControls, "bank": ControlBank, "skip": NoControls},
        default="random",
        tag="source",
    )

    def __post_init__(self):
        if self.direction not in ("proponents", "detractors"):
            raise ValueError("direction must be proponents or detractors")
        if not 0 < self.fraction <= 1:
            raise ValueError("filter fraction must be in (0, 1]")

    @property
    def name(self) -> str:
        return f"filter-{self.direction}"


@dataclass
class WeightStepConfig(Serializable):
    """Compare weight-gradient steps with their first-order predictions."""

    lrs: list[float] = field(default_factory=lambda: [1.0])

    def __post_init__(self):
        if not self.lrs:
            raise ValueError("weight-step requires at least one lr")


LEGACY_FIELDS = {
    "num_subsets",
    "subset_fraction",
    "subsets",
    "subset_start",
    "subset_stop",
    "weight_lrs",
    "controls",
    "retrained_dir",
}


def migrate_validation_config(obj: dict) -> dict:
    """Resolve old flat YAML once; execution only sees typed method configs."""
    obj = dict(obj)
    method = obj.get("method")
    legacy = LEGACY_FIELDS.intersection(obj)
    if isinstance(method, dict):
        if legacy:
            raise ValueError(
                f"Cannot mix nested method with legacy fields: {sorted(legacy)}"
            )
        return obj
    if method is None and not legacy and obj.get("seed", 42) == 42:
        return obj
    warnings.warn(
        "Flat validation options are deprecated; use a nested method config "
        "with kind and source tags",
        FutureWarning,
        stacklevel=3,
    )
    method = obj.pop("method", "lds")
    old = {key: obj.pop(key) for key in legacy}
    count = old.get("num_subsets", 100)
    fraction = old.get("subset_fraction", 0.0)
    paths = old.get("retrained_dir", [])
    if isinstance(paths, str):
        paths = [p for p in paths.split(",") if p]
    sampling_seed = obj.get("seed", 42)
    if method == "lds":
        if old.get("weight_lrs") and not paths:
            obj["method"] = {"kind": "weight-step", "lrs": old["weight_lrs"]}
        else:
            source = (
                {"source": "bank", "paths": paths}
                if paths
                else {
                    "source": "random",
                    "count": count,
                    "sampling": "random" if fraction > 0 else "partition",
                    "fraction": fraction if fraction > 0 else 0.05,
                    "sampling_seed": sampling_seed,
                    "manifest": old.get("subsets", ""),
                }
            )
            obj["method"] = {
                "kind": "lds",
                "subsets": source,
                "start": old.get("subset_start", 0),
                "stop": old.get("subset_stop"),
            }
    elif method in ("filter-proponents", "filter-detractors"):
        if fraction == 0:
            if count <= 0:
                raise ValueError(
                    "subset_fraction must be positive when num_subsets is 0"
                )
            fraction = 1 / count
        mode = old.get("controls", "auto")
        if mode not in ("auto", "load", "retrain", "skip"):
            raise ValueError(f"Unknown legacy controls mode: {mode}")
        if mode == "load" and not paths:
            raise ValueError("controls: load requires retrained_dir")
        if mode == "retrain" and count <= 0:
            raise ValueError("controls: retrain requires num_subsets > 0")
        if mode == "skip":
            controls = {"source": "skip"}
        elif paths and mode != "retrain":
            controls = {"source": "bank", "paths": paths, "count": None}
        elif count > 0:
            controls = {
                "source": "random",
                "count": count,
                "sampling_seed": sampling_seed,
            }
        else:
            controls = {"source": "skip"}
        obj["method"] = {
            "kind": "filter",
            "direction": method.removeprefix("filter-"),
            "fraction": fraction,
            "controls": controls,
        }
    else:
        raise ValueError(f"Unknown legacy validation method: {method!r}")
    return obj

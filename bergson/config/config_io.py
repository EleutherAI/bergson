import itertools
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import petname
import yaml

from bergson.utils.logger import get_logger

CONFIG_FILENAME = "config.yaml"


def _resolve(path: str | Path) -> Path:
    """Return the path to a ``config.yaml``, accepting either a dir or a file."""
    path = Path(path)
    return path / CONFIG_FILENAME if path.is_dir() else path


def _git_sha() -> str | None:
    """git SHA of the bergson source tree, or ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def make_metadata() -> dict[str, Any]:
    """Run metadata: version, time, git sha."""
    try:
        version: str | None = _pkg_version("bergson")
    except PackageNotFoundError:
        version = None
    meta: dict[str, Any] = {
        "bergson_version": version,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    sha = _git_sha()
    if sha is not None:
        meta["git_sha"] = sha
    return meta


def _write(
    steps: list[dict[str, Any]],
    path: Path,
    *,
    run_path: str | Path | None = None,
):
    """Write a ``{[run_path], steps, metadata}`` document, metadata last."""
    doc: dict[str, Any] = {}
    if run_path is not None:
        # This field only exists in pipeline docs
        doc["run_path"] = str(run_path)
    doc["steps"] = steps
    doc["metadata"] = make_metadata()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def save_run_config(command: Any, run_path: str | Path):
    """Write a one-step ``config.yaml`` for ``command`` into ``run_dir``.

    It can be run using ``bergson <run_path>/config.yaml``.
    """
    step = {(type(command).__name__).lower(): command.to_dict()}
    _write([step], Path(run_path) / CONFIG_FILENAME)


def save_pipeline_config(steps: list[tuple[str, Any]], run_path: str | Path | None):
    """Write a multi-step ``config.yaml`` to ``run_path``.

    It can be run using ``bergson <run_path>/config.yaml``.

    The pipeline config uses the same format as one-step command configs, but is saved
    to a ``run_path`` directory unique to the pipeline run. ``steps`` is the list of
    commands that ran.
    """
    if not run_path:
        run_name = petname.generate(2, separator="_")
        run_path = f"runs/{run_name}"
        get_logger(__name__).warning(
            "No top level run_path set for this multi-step YAML; "
            "logging pipeline config to %s",
            run_path,
        )
    run_path = Path(run_path)

    resolved_steps = [{name: cmd.to_dict()} for name, cmd in steps]
    _write(resolved_steps, run_path / CONFIG_FILENAME, run_path=run_path)


def read_config(path: str | Path) -> dict[str, Any]:
    """Read a ``config.yaml`` (a file or a run dir) into its document.

    Returns a ``{run_path?, steps, metadata?}`` mapping.
    """
    path = _resolve(path)
    with path.open() as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict) or not isinstance(config.get("steps"), list):
        raise ValueError(
            f"{path} must be a mapping with a `steps:` list of "
            f"`- command: {{...}}` entries; got a top-level {type(config).__name__}."
        )

    return config


T = TypeVar("T", bound="FromDict")


class FromDict(Protocol):
    @classmethod
    def from_dict(
        cls: type[T],
        obj: dict[str, Any],
        /,
        drop_extra_fields: bool | None = None,
    ) -> T: ...


def load_subconfig(
    path: str | Path,
    field: str,
    config_cls: type[T],
) -> T | None:
    """Hydrate one configuration dataclass (e.g. ``field="index_cfg",
    config_cls=IndexConfig``) from a ``config.yaml``.

    Searches every step for ``field`` and returns the first match
    or ``None``.
    """
    if not _resolve(path).exists():
        return None

    for step in read_config(path)["steps"]:
        for cmd_dict in step.values():
            subconfig = (cmd_dict or {}).get(field)
            if subconfig is not None:
                return cast(T, config_cls.from_dict(subconfig))
    return None


def expand_matrix(cmd_dict: dict) -> list[dict]:
    """Expand a step's ``matrix`` mapping into one config per grid cell.

    ``{key}`` in string values substitutes the cell's value; a value that is
    exactly ``"{key}"`` becomes the typed grid value.
    """
    matrix = cmd_dict.pop("matrix", None)
    if matrix is None:
        return [cmd_dict]
    if not isinstance(matrix, dict) or not all(
        isinstance(v, list) and v for v in matrix.values()
    ):
        raise ValueError(f"matrix must map keys to non-empty lists; got {matrix!r}.")

    def substitute(value, cell: dict):
        if isinstance(value, dict):
            return {k: substitute(v, cell) for k, v in value.items()}
        if isinstance(value, list):
            return [substitute(v, cell) for v in value]
        if isinstance(value, str):
            for key, cell_value in cell.items():
                if value == "{" + key + "}":
                    return cell_value
                value = value.replace("{" + key + "}", str(cell_value))
        return value

    keys = list(matrix)
    cells = [dict(zip(keys, values)) for values in itertools.product(*matrix.values())]
    expanded = [cast(dict, substitute(cmd_dict, cell)) for cell in cells]

    def run_paths(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k == "run_path":
                    yield v
                else:
                    yield from run_paths(v)

    seen: dict = {}
    for cfg, cell in zip(expanded, cells):
        for rp in run_paths(cfg):
            if rp in seen:
                raise ValueError(
                    f"matrix cells {seen[rp]} and {cell} both write {rp}; "
                    "vary run_path with a {key} substitution."
                )
            seen[rp] = cell
    if len(cells) > 1 and not seen:
        raise ValueError("matrix step has no run_path; cells would collide.")
    return expanded


def parse_steps(
    steps: list, command_registry: dict[str, type]
) -> list[tuple[str, Any]]:
    """Turn raw ``steps`` mappings into ``(command_name, command)`` instances."""
    parsed: list[tuple[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict) or len(step) != 1:
            raise ValueError(
                f"Each step must be a single command-key mapping "
                f"(e.g. `- build: {{...}}`); got {step!r}."
            )
        ((cmd_name, cmd_dict),) = step.items()
        try:
            cmd_cls = command_registry[cmd_name.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown command '{cmd_name}'. "
                f"Valid commands: {sorted(command_registry)}."
            ) from None

        for cell_dict in expand_matrix(dict(cmd_dict or {})):
            parsed.append(
                (cmd_name, cmd_cls.from_dict(cell_dict, drop_extra_fields=False))
            )
    return parsed

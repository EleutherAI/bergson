import os
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

# Per-invocation identity, not run configuration: these index_cfg fields may
# differ between the shards of one sharded run and are stripped from the
# canonical config.yaml shared by all shards.
EPHEMERAL_INDEX_FIELDS = ("shard_id", "overwrite")


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


def canonical_steps(command: Any) -> list[dict[str, Any]]:
    """One-step ``steps`` list with per-invocation identity stripped.

    All shards of a sharded run must produce the same canonical steps, so
    fields that legitimately differ between shards (``shard_id``,
    ``overwrite``, ``distributed.node_rank``) are removed.
    """
    # Round-trip through YAML so the comparison in publish_canonical_config
    # sees the same plain types a reread of the file would produce.
    steps = yaml.safe_load(
        yaml.safe_dump([{(type(command).__name__).lower(): command.to_dict()}])
    )

    for parsed_step in steps:
        for cmd_dict in parsed_step.values():
            index_cfg = (cmd_dict or {}).get("index_cfg")
            if not index_cfg:
                continue
            for field in EPHEMERAL_INDEX_FIELDS:
                index_cfg.pop(field, None)
            (index_cfg.get("distributed") or {}).pop("node_rank", None)

    return steps


def publish_canonical_config(command: Any, run_path: str | Path):
    """Write the ``config.yaml`` shared by all shards of a sharded run.

    The first shard to arrive writes it atomically; every later shard
    verifies its own canonical config matches and errors out otherwise,
    so shards built from different configurations can never mix in one
    run_path.
    """
    path = Path(run_path) / CONFIG_FILENAME
    steps = canonical_steps(command)

    if path.exists():
        existing = read_config(path)
        if existing["steps"] != steps:
            raise ValueError(
                f"{path} was written by a run with a different configuration. "
                f"Refusing to add shards to it; use a fresh run_path or "
                f"rerun with the original configuration."
            )
        return

    doc: dict[str, Any] = {"steps": steps, "metadata": make_metadata()}
    path.parent.mkdir(parents=True, exist_ok=True)

    # Concurrent shards may race to create the file; writing a temp file and
    # renaming it into place is atomic, and every racer writes identical
    # steps, so last-writer-wins is safe.
    tmp_path = path.with_name(f".{CONFIG_FILENAME}.{os.getpid()}.tmp")
    with tmp_path.open("w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    os.rename(tmp_path, path)


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

        # Hydrate config
        parsed_step = cmd_cls.from_dict(cmd_dict or {}, drop_extra_fields=False)

        parsed.append((cmd_name, parsed_step))
    return parsed

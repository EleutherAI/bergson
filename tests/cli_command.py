"""Running the ``bergson`` CLI from tests against the checkout under test.

The bare ``bergson`` console script imports from its own interpreter's
site-packages, so without an editable install it raises ``ModuleNotFoundError``
— and with a stale one it tests the wrong code.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def bergson_cmd(*args: str) -> list[str]:
    """The argv for running ``bergson <args>`` against this checkout."""
    return [sys.executable, "-m", "bergson", *args]


def bergson_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """``env`` (default: the current one) with this checkout on ``PYTHONPATH``.

    ``python -m`` only adds the *current directory*, and several tests run with
    ``cwd=tmp_path``.
    """
    env = dict(os.environ if env is None else env)
    existing = env.get("PYTHONPATH", "")
    parts = [str(REPO_ROOT)] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env

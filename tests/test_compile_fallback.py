"""Tests for the ``compile_if_supported`` guard.

``torch.compile`` raises a ``RuntimeError`` at decoration time on interpreters
Dynamo doesn't support (e.g. Python 3.14+ with torch <= 2.9). Because the
normalizers in ``bergson.gradients`` apply it at import time, an unguarded
``@torch.compile`` makes ``bergson`` unimportable there. The ``import-py314``
CI job covers the real interpreter; these tests cover the fallback logic on
any interpreter.
"""

import subprocess
import sys
import textwrap
import warnings

import torch

from bergson.gradients import compile_if_supported


def test_returns_callable_when_supported():
    def add_one(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    # Must not raise at decoration time. We deliberately don't execute the
    # result: running a compiled function requires a working inductor
    # toolchain, which tests torch rather than the guard.
    wrapped = compile_if_supported(add_one)
    assert callable(wrapped)


def test_falls_back_to_eager_when_compile_raises(monkeypatch):
    def unsupported(*args, **kwargs):
        raise RuntimeError("torch.compile is not supported on Python 3.14+")

    monkeypatch.setattr(torch, "compile", unsupported)

    def add_one(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapped = compile_if_supported(add_one)

    # The original function is returned unchanged, with a warning.
    assert wrapped is add_one
    assert any("uncompiled" in str(w.message) for w in caught)
    assert torch.equal(wrapped(torch.tensor([1.0])), torch.tensor([2.0]))


def test_import_and_normalize_with_unsupported_compile():
    """Simulate Python 3.14: torch.compile raises before bergson is imported.

    Runs in a subprocess so the eagerly-decorated module state doesn't leak
    into other tests. Verifies that (1) the import succeeds, (2) the fallback
    warning fires, and (3) the eager normalizer computes correct values.
    """
    code = textwrap.dedent("""
        import torch

        def unsupported(*args, **kwargs):
            raise RuntimeError("torch.compile is not supported on Python 3.14+")

        torch.compile = unsupported

        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from bergson.gradients import AdamNormalizer

        assert any(
            "uncompiled" in str(w.message) for w in caught
        ), "expected fallback warning"

        avg_sq = torch.rand(4, 3) + 0.1
        grad = torch.randn(4, 3)
        eps = 1e-8

        # AdamNormalizer.normalize_weight: grad / (sqrt(avg_sq) + eps)
        expected = grad / (avg_sq.sqrt() + eps)
        actual = AdamNormalizer(weight_avg_sq=avg_sq.clone()).normalize_weight(
            grad.clone(), eps=eps
        )
        torch.testing.assert_close(actual, expected)
        print("OK")
        """)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout

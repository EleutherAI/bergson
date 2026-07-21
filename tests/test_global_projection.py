"""Tests for the LESS-style global TRAK projection mode."""

from pathlib import Path

import pytest
import torch

from bergson import GradientProcessor, collect_gradients
from bergson.collector.collector import CollectorComputer, create_projection_matrix
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig
from bergson.data import load_module_gradients


def test_global_projector_processor_field_default():
    p = GradientProcessor(projection_dim=16)
    assert p.projection_target == "per_module"


def test_global_projector_processor_field_set():
    p = GradientProcessor(projection_dim=8192, projection_target="global")
    assert p.projection_target == "global"


def test_global_shapes_collapse_to_single_key():
    """In global mode, shapes() returns one synthetic 'gradients' entry."""
    from bergson.collector.collector import HookCollectorBase

    # Construct a minimal stand-in for HookCollectorBase that exercises shapes()
    # without running a full forward/backward. We only need processor + a fake
    # target_info dict.
    class _Stub(HookCollectorBase):
        def __init__(self, processor, target_info):
            self.processor = processor
            self.target_info = target_info
            self.attention_cfgs = {}

        def setup(self):
            pass

        def teardown(self):
            pass

        def discover_targets(self, *_a, **_kw):
            return {}

        def backward_hook(self, module, g):
            pass

        def process_batch(self, indices, **kwargs):
            pass

    target_info = {
        "model.layers.0.q_proj": (None, torch.Size((4, 4)), False),
        "model.layers.0.k_proj": (None, torch.Size((4, 4)), False),
    }
    proc = GradientProcessor(projection_dim=64, projection_target="global")
    stub = _Stub(proc, target_info)
    shapes = stub.shapes()
    assert set(shapes.keys()) == {"gradients"}
    assert tuple(shapes["gradients"]) == (64,)


def test_global_shapes_requires_projection_dim():
    from bergson.collector.collector import HookCollectorBase

    class _Stub(HookCollectorBase):
        def __init__(self, processor, target_info):
            self.processor = processor
            self.target_info = target_info
            self.attention_cfgs = {}

        def setup(self):
            pass

        def teardown(self):
            pass

        def discover_targets(self, *_a, **_kw):
            return {}

        def backward_hook(self, module, g):
            pass

        def process_batch(self, indices, **kwargs):
            pass

    proc = GradientProcessor(projection_dim=None, projection_target="global")
    stub = _Stub(proc, target_info={})
    with pytest.raises(AssertionError):
        stub.shapes()


def test_per_module_shapes_unchanged():
    """Default per_module mode returns one shape per target module."""
    from bergson.collector.collector import HookCollectorBase

    class _Stub(HookCollectorBase):
        def __init__(self, processor, target_info):
            self.processor = processor
            self.target_info = target_info
            self.attention_cfgs = {}

        def setup(self):
            pass

        def teardown(self):
            pass

        def discover_targets(self, *_a, **_kw):
            return {}

        def backward_hook(self, module, g):
            pass

        def process_batch(self, indices, **kwargs):
            pass

    target_info = {
        "model.layers.0.q_proj": (None, torch.Size((4, 4)), False),
        "model.layers.0.k_proj": (None, torch.Size((4, 4)), False),
    }
    proc = GradientProcessor(projection_dim=8, projection_target="per_module")
    stub = _Stub(proc, target_info)
    shapes = stub.shapes()
    assert set(shapes.keys()) == set(target_info.keys())
    for s in shapes.values():
        assert tuple(s) == (8, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_global_projector_e2e(tmp_path: Path, model, dataset):
    # CudaProjector requires proj_dim to be a multiple of 512.
    proj_dim = 512
    cfg = IndexConfig(
        run_path=str(tmp_path),
        token_batch_size=64,
        projection_dim=proj_dim,
        projection_target="global",
    )
    processor = GradientProcessor(
        projection_dim=proj_dim,
        projection_target="global",
    )

    collect_gradients(model=model.cuda(), data=dataset, processor=processor, cfg=cfg)

    grads = load_module_gradients(cfg.partial_run_path)
    assert list(grads.keys()) == ["gradients"]
    assert grads["gradients"].shape == (len(dataset), proj_dim)


def test_global_projection_linearity():
    """Block decomposition: cat(g1,g2,g3) @ R.T == sum_i g_i @ R_i.T.

    This algebraic identity holds for any linear projection matrix R when
    R_i are the corresponding column blocks. It is the mathematical basis for
    why per-module projection + sum is equivalent to global projection.
    """
    torch.manual_seed(42)
    N, d1, d2, d3 = 3, 6, 10, 4
    proj_dim = 8
    total = d1 + d2 + d3

    g1 = torch.randn(N, d1)
    g2 = torch.randn(N, d2)
    g3 = torch.randn(N, d3)

    R = create_projection_matrix(
        "test/single", proj_dim, total, torch.float32, torch.device("cpu")
    )

    global_result = torch.cat([g1, g2, g3], dim=1) @ R.T
    block_sum = g1 @ R[:, :d1].T + g2 @ R[:, d1 : d1 + d2].T + g3 @ R[:, d1 + d2 :].T

    torch.testing.assert_close(global_result, block_sum)


def test_global_project_values_cpu(tmp_path: Path, model, dataset):
    """Global projection values match manual per-module right-projection + sum on CPU.

    Uses a separate unprojected collector to capture raw per-module gradients, then
    verifies that the global-mode collector's accumulated result equals the manual
    computation using the same create_projection_matrix identifiers.
    """
    proj_dim = 16
    tokens = torch.tensor([dataset[0]["input_ids"]])
    cfg = IndexConfig(run_path=str(tmp_path))

    # First pass: capture raw per-module gradients (no projection)
    raw_collector = GradientCollector(
        model=model,
        cfg=cfg,
        data=dataset,
        processor=GradientProcessor(projection_dim=None),
        skip_index=True,
    )
    with raw_collector:
        model.zero_grad()
        model(input_ids=tokens, labels=tokens).loss.backward()
    raw_grads = {k: v.clone() for k, v in raw_collector.mod_grads.items()}

    # Second pass: collect with global projection (projection done in backward hook)
    global_processor = GradientProcessor(
        projection_dim=proj_dim, projection_target="global"
    )
    global_collector = GradientCollector(
        model=model,
        cfg=cfg,
        data=dataset,
        processor=global_processor,
        skip_index=True,
    )
    with global_collector:
        model.zero_grad()
        model(input_ids=tokens, labels=tokens).loss.backward()
    projected = global_collector.mod_grads["gradients"]

    # Manually replicate: same identifier → same matrix → identical result
    expected: torch.Tensor | None = None
    for name, P in raw_grads.items():
        R = create_projection_matrix(
            f"{name}/single",
            proj_dim,
            P.shape[1],
            P.dtype,
            P.device,
            global_processor.projection_type,
        )
        contrib = P @ R.T
        expected = contrib if expected is None else expected + contrib

    assert expected is not None
    torch.testing.assert_close(projected.float(), expected)


def test_global_projector_e2e_cpu(tmp_path: Path, model, dataset):
    """End-to-end global projection on CPU drives the full CollectorComputer pipeline.

    Builder requires CUDA, so we inject a lightweight capturer as the scorer to
    receive projected gradients without triggering any GPU code.
    """
    proj_dim = 16  # BasicProjector has no multiple-of-512 constraint

    class _Capturer:
        def __init__(self):
            self.chunks: list[torch.Tensor] = []

        def __call__(self, _indices, mod_grads):
            self.chunks.append(mod_grads["gradients"].clone())

    capturer = _Capturer()
    cfg = IndexConfig(
        run_path=str(tmp_path),
        token_batch_size=64,
        projection_dim=proj_dim,
        projection_target="global",
    )
    processor = GradientProcessor(
        projection_dim=proj_dim,
        projection_target="global",
    )
    collector = GradientCollector(
        model=model.base_model,
        cfg=cfg,
        data=dataset,
        processor=processor,
        scorer=capturer,
    )
    computer = CollectorComputer(
        model=model,
        data=dataset,
        collector=collector,
        cfg=cfg,
    )
    computer.run_with_collector_hooks()

    all_projected = torch.cat(capturer.chunks, dim=0)
    assert all_projected.shape == (len(dataset), proj_dim)
    assert torch.isfinite(all_projected).all()
    # Different examples should produce different projections
    assert not torch.allclose(all_projected[0], all_projected[1])


class _GradCapturer:
    """Stands in for a Scorer so a collector can run without CUDA (Builder
    allocates on cuda unconditionally)."""

    def __init__(self):
        self.chunks: list[dict[str, torch.Tensor]] = []

        class _Writer:
            scores = None

        self.writer = _Writer()

    def __call__(self, _indices, mod_grads):
        self.chunks.append({k: v.clone() for k, v in mod_grads.items()})


def test_in_memory_collector_honors_global_projection(tmp_path: Path, model, dataset):
    """``InMemoryCollector`` must implement the global-projection sum too.

    Only ``GradientCollector.backward_hook`` had the global branch, so the
    same config gave ``InMemoryCollector`` per-module keys instead of the
    single "gradients" key its ``shapes()`` promises -- and ``Builder``'s
    ``_preprocess`` then raised ``KeyError: 'gradients'``.
    """
    from bergson.collector.in_memory_collector import InMemoryCollector

    proj_dim = 16  # BasicProjector has no multiple-of-512 constraint

    def run(collector_cls, tag):
        cfg = IndexConfig(
            run_path=str(tmp_path / tag),
            token_batch_size=64,
            projection_dim=proj_dim,
            projection_target="global",
        )
        cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
        capturer = _GradCapturer()
        collector = collector_cls(
            model=model.base_model,
            data=dataset,
            cfg=cfg,
            processor=GradientProcessor(
                projection_dim=proj_dim, projection_target="global"
            ),
            attention_cfgs={},
            scorer=capturer,
        )
        assert list(collector.shapes()) == ["gradients"]
        CollectorComputer(
            model=model, data=dataset, collector=collector, cfg=cfg
        ).run_with_collector_hooks(desc="Collecting")
        return capturer.chunks

    in_memory = run(InMemoryCollector, "in_memory")
    reference = run(GradientCollector, "reference")

    for chunk in in_memory:
        assert list(chunk) == ["gradients"], (
            "InMemoryCollector produced per-module keys under "
            f"projection_target='global': {sorted(chunk)}"
        )

    got = torch.cat([c["gradients"] for c in in_memory], dim=0)
    expected = torch.cat([c["gradients"] for c in reference], dim=0)
    assert got.shape == (len(dataset), proj_dim)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.float(), expected.float())

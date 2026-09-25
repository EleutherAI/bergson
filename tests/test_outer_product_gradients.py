"""OuterProductGradients gives the same results as the gradients it forms."""

from dataclasses import dataclass, field

import pytest
import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM

from bergson import GradientProcessor, InMemoryCollector
from bergson.collector.collector import CollectorComputer, HookCollectorBase
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import AttentionConfig, IndexConfig
from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    OuterProductGradients,
)
from bergson.score.score_writer import (
    InMemorySequenceScoreWriter,
    InMemoryTokenScoreWriter,
)
from bergson.score.scorer import Scorer

DATA = Dataset.from_dict(
    {"input_ids": [[1, 2, 3, 4, 5, 6], [7, 8, 9], [10, 11, 12, 13]]}
)
ATTENTION_CFGS = {
    "layers.0.attention.query_key_value": AttentionConfig(
        num_heads=4, head_size=6, head_dim=2
    )
}

# Adam with projection forms its gradients before projecting them
CASES = [
    (normalizer, projection_dim, attribute_tokens)
    for normalizer in ["none", "adafactor", "adam"]
    for projection_dim in [None, 4]
    for attribute_tokens in [False, True]
    if not (normalizer == "adam" and projection_dim)
]


def _model_and_processor(normalizer: str, projection_dim: int | None):
    """A tiny model where one module has no bias and the processor includes
    biases."""
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-GPTNeoXForCausalLM")
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    model.base_model.layers[0].attention.dense.bias = None

    normalizers = {}
    for name, module in model.base_model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        o, i = module.weight.shape
        if normalizer == "adam":
            normalizers[name] = AdamNormalizer(
                torch.rand(o, i) + 0.5, torch.rand(o) + 0.5
            )
        elif normalizer == "adafactor":
            normalizers[name] = AdafactorNormalizer(
                torch.rand(o) + 0.5, torch.rand(i) + 0.5, torch.rand(o) + 0.5
            )
    processor = GradientProcessor(
        normalizers=normalizers, include_bias=True, projection_dim=projection_dim
    )
    return model, processor


@dataclass(kw_only=True)
class _Recorder(HookCollectorBase):
    grads: dict = field(default_factory=dict)

    def setup(self):
        pass

    def teardown(self):
        pass

    def process_batch(self, indices, **kwargs):
        pass

    @HookCollectorBase.split_attention_heads
    def backward_hook(self, module, g):
        self.grads[module._name] = self._module_gradient(module, g)


@pytest.fixture(params=CASES, ids=lambda case: "-".join(map(str, case)))
def module_grads(request, tmp_path) -> dict[str, OuterProductGradients]:
    """Every module's gradients for one batch holding the whole dataset, with
    one module split into attention heads."""
    normalizer, projection_dim, attribute_tokens = request.param
    model, processor = _model_and_processor(normalizer, projection_dim)
    cfg = IndexConfig(run_path=str(tmp_path / "run"), attribute_tokens=attribute_tokens)
    cfg.partial_run_path.mkdir(parents=True)
    recorder = _Recorder(
        model=model.base_model,
        processor=processor,
        attribute_tokens=attribute_tokens,
        attention_cfgs=ATTENTION_CFGS,
    )
    CollectorComputer(
        model, DATA, collector=recorder, cfg=cfg
    ).run_with_collector_hooks()
    assert all(isinstance(g, OuterProductGradients) for g in recorder.grads.values())
    return recorder.grads


# With 9 inputs and 5 or 12 outputs, 3 queries contract each way and 64 form
# the gradients.
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("divisor", [False, True])
@pytest.mark.parametrize("num_queries", [3, 64])
@pytest.mark.parametrize("o", [5, 12])
def test_random_vectors_match_formed_gradients(bias, divisor, num_queries, o):
    """Every combination of terms, including ones a model's gradients never
    produce, such as a divisor with a projected bias column."""
    torch.manual_seed(0)
    T, w = 7, 9
    grads = OuterProductGradients(
        g=torch.randn(T, o),
        a=torch.randn(T, w),
        bias=torch.randn(T, o) if bias else None,
        bias_col=torch.randn(w) if bias else None,
        divisor=torch.rand(o, w) + 0.5 if divisor else None,
    )
    formed = grads.materialize().flatten(1)
    torch.testing.assert_close(grads.sq_norm(), formed.pow(2).sum(1))
    q = torch.randn(num_queries, o * w)
    torch.testing.assert_close(grads.dot(q), formed @ q.T)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_queries", [3, 64])
def test_half_precision_with_small_adam_denominators(dtype, num_queries):
    """Adam's denominators stay in float32: inverted, they overflow float16."""
    torch.manual_seed(0)
    T, o, w = 7, 5, 9
    grads = OuterProductGradients(
        g=torch.randn(T, o) * 1e-3,
        a=torch.randn(T, w) * 1e-3,
        bias=torch.randn(T, o),
        bias_col=torch.randn(w),
        divisor=torch.rand(o, w) * 1e-6 + 1e-6,
    )
    formed = grads.materialize().flatten(1)
    half = grads.to(torch.device("cpu"), dtype)
    assert half.divisor is not None and half.divisor.dtype == torch.float32

    torch.testing.assert_close(
        half.sq_norm(), formed.pow(2).sum(1), check_dtype=False, atol=0.0, rtol=0.05
    )
    q = torch.randn(num_queries, o * w)
    expected = formed @ q.T
    torch.testing.assert_close(
        half.dot(q.to(dtype)).float(),
        expected,
        atol=0.05 * expected.abs().max(),
        rtol=0,
    )


# The tiny model's modules have at most 33 inputs or outputs, so 3 queries are
# contracted with the vectors and 64 are taken against the formed gradients.
@pytest.mark.parametrize("num_queries", [3, 64])
def test_dot_matches_formed_gradients(module_grads, num_queries):
    for name, grads in module_grads.items():
        formed = grads.materialize().flatten(1)
        q = torch.randn(num_queries, formed.shape[1])
        torch.testing.assert_close(grads.dot(q), formed @ q.T, msg=name)


def test_sq_norm_matches_formed_gradients(module_grads):
    for name, grads in module_grads.items():
        if grads.per_example:
            pytest.skip("sq_norm takes per-token norms only")
        formed = grads.materialize().flatten(1)
        torch.testing.assert_close(grads.sq_norm(), formed.pow(2).sum(1), msg=name)


@pytest.mark.parametrize("unit_normalize", [False, True])
@pytest.mark.parametrize("num_queries", [3, 64])
@pytest.mark.parametrize("variant", ["plain", "nearest", "index_transform"])
def test_scorer_matches_formed_gradients(
    module_grads, unit_normalize, num_queries, variant
):
    formed = {m: g.materialize().flatten(1) for m, g in module_grads.items()}
    queries = {m: torch.randn(num_queries, f.shape[1]) for m, f in formed.items()}

    def scores(grads):
        return Scorer(
            query_grads=queries,
            modules=list(queries),
            writer=InMemorySequenceScoreWriter(1, num_queries),
            device=torch.device("cpu"),
            dtype=torch.float32,
            unit_normalize=unit_normalize,
            score_mode="nearest" if variant == "nearest" else "individual",
            index_transform=(
                (lambda x: {m: 2 * t for m, t in x.items()})
                if variant == "index_transform"
                else None
            ),
        ).score(grads)

    torch.testing.assert_close(scores(module_grads), scores(formed))


@pytest.mark.parametrize("normalizer", ["none", "adam"])
@pytest.mark.parametrize("attribute_tokens", [False, True])
@pytest.mark.parametrize("unit_normalize", [False, True])
@pytest.mark.parametrize("collector_cls", [GradientCollector, InMemoryCollector])
def test_scores_during_collection_match_formed_gradients(
    tmp_path, normalizer, attribute_tokens, unit_normalize, collector_cls
):
    """Scoring during collection, where the collector passes the scorer each
    module's vectors, matches scoring the collected gradients."""
    model, processor = _model_and_processor(normalizer, None)
    cfg = IndexConfig(
        run_path=str(tmp_path / "run"),
        attribute_tokens=attribute_tokens,
        drop_columns=False,
    )
    cfg.partial_run_path.mkdir(parents=True)

    def collect(collector_cls, scorer=None):
        collector = collector_cls(
            model=model.base_model,
            data=DATA,
            cfg=cfg,
            processor=processor,
            scorer=scorer,
            attention_cfgs=ATTENTION_CFGS,
        )
        CollectorComputer(
            model, DATA, collector=collector, cfg=cfg
        ).run_with_collector_hooks()
        return collector

    gradients = collect(InMemoryCollector).gradients
    queries = {m: torch.randn(3, g.shape[1]) for m, g in gradients.items()}

    def make_scorer(writer) -> Scorer:
        return Scorer(
            query_grads=queries,
            modules=list(queries),
            writer=writer,
            device=torch.device("cpu"),
            dtype=torch.float32,
            unit_normalize=unit_normalize,
        )

    if attribute_tokens:
        writer = InMemoryTokenScoreWriter(DATA, 3)
    else:
        writer = InMemorySequenceScoreWriter(len(DATA), 3)
    collect(collector_cls, make_scorer(writer))
    scores = torch.cat(writer.scores) if attribute_tokens else writer.scores

    expected = make_scorer(writer).score(gradients)
    torch.testing.assert_close(scores, expected)

from collections.abc import Callable

import torch
from torch import Tensor

from bergson.gradients import OuterProductGradients
from bergson.score.score_writer import ScoreWriter


class Scorer:
    """
    Scores training gradients against query gradients.

    Accepts an optional ``index_transform`` callable that is applied to each
    batch of index gradients before scoring. This can be used for
    preconditioning, projection, or any other per-batch transformation.
    When no transform is needed, pass ``None`` (identity is used).

    Accepts a ScoreWriter for saving the scores (disk or in-memory).
    """

    def __init__(
        self,
        query_grads: dict[str, Tensor],
        modules: list[str],
        writer: ScoreWriter,
        device: torch.device,
        dtype: torch.dtype,
        *,
        unit_normalize: bool = False,
        score_mode: str = "individual",
        attribute_tokens: bool = False,
        index_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        query_offset: int = 0,
    ):
        """
        Initialize the scorer.

        Parameters
        ----------
        query_grads : dict[str, Tensor]
            Query gradients keyed by module name. Should already be
            preconditioned if preconditioning is desired.
        modules : list[str]
            List of module names to use for scoring.
        writer : ScoreWriter
            Writer for score output (InMemoryScoreWriter or MemmapScoreWriter).
        device : torch.device
            Device to perform scoring on.
        dtype : torch.dtype
            Dtype for scoring computation.
        unit_normalize : bool
            Whether to unit normalize gradients before scoring.
        score_mode : str
            Scoring mode: "individual" or "nearest".
        attribute_tokens : bool
            Whether gradients are per-token (rows = total_valid tokens).
        index_transform : Callable | None
            Optional transform applied to index gradients per-batch before
            scoring. Receives and returns ``dict[str, Tensor]``. When ``None``,
            index gradients are used as-is.
        query_offset : int
            Position of this scorer's first query in the full query set.
        """
        self.device = device
        self.dtype = dtype
        self.modules = modules
        self.unit_normalize = unit_normalize
        self.score_mode = score_mode
        self.attribute_tokens = attribute_tokens
        self.writer = writer
        self.index_transform = index_transform
        self.query_offset = query_offset

        # Only an index transform needs a batch's modules together.
        self.streaming = index_transform is None
        self._scores: Tensor | None = None
        self._sq_norm: Tensor | None = None

        # Pre-transposed for scoring: per-module [dim_m, n_queries]
        self.query_grads_t = {
            m: query_grads[m].to(device=self.device, dtype=self.dtype).T
            for m in modules
        }

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, Tensor | OuterProductGradients],
    ):
        """Score a batch of training gradients, or finish one fed by ``accumulate``."""
        if self._scores is not None:
            scores, sq_norm = self._scores, self._sq_norm
            self._scores = self._sq_norm = None
            scores = self._reduce(scores, sq_norm)
        else:
            scores = self.score(mod_grads)
        self.writer(indices, scores, query_offset=self.query_offset)

    @torch.inference_mode()
    def accumulate(self, name: str, g: Tensor | OuterProductGradients) -> None:
        """Add one module's gradients to the current batch; ``__call__`` finishes it."""
        assert self.streaming, "accumulate needs a scorer without index_transform"
        if name not in self.query_grads_t:
            return
        self._scores, self._sq_norm = self._add_module(
            name, g, self._scores, self._sq_norm
        )

    def _add_module(
        self,
        name: str,
        g: Tensor | OuterProductGradients,
        scores: Tensor | None,
        sq_norm: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """Add module ``name``'s GEMM against the queries to the running sums,
        accumulating in fp32 so a bf16 scoring dtype keeps small contributions."""
        g = g.to(self.device, self.dtype, non_blocking=True)
        if isinstance(g, OuterProductGradients) and g.per_example:
            g = _materialized(g)

        if isinstance(g, OuterProductGradients):
            part = g.dot(self.query_grads_t[name].T).float()
        else:
            part = (g @ self.query_grads_t[name]).float()
        scores = part if scores is None else scores.add_(part)

        if self.unit_normalize:
            if isinstance(g, OuterProductGradients):
                n = g.sq_norm()
            else:
                n = g.float().pow(2).sum(dim=1)
            sq_norm = n if sq_norm is None else sq_norm.add_(n)
        return scores, sq_norm

    @torch.inference_mode()
    def score(self, index_grads: dict[str, Tensor | OuterProductGradients]) -> Tensor:
        """Compute scores for a batch of gradients."""
        if self.index_transform is not None:
            index_grads = self.index_transform(
                {m: _materialized(grads) for m, grads in index_grads.items()}
            )  # type: ignore[assignment]

        # scores[b, q] = sum_m g_b[m] . q_q[m]; per-module GEMMs avoid
        # materializing a [batch, total_dim] concat of the index gradients
        scores = None
        sq_norm = None
        for m in self.modules:
            scores, sq_norm = self._add_module(m, index_grads[m], scores, sq_norm)

        return self._reduce(scores, sq_norm)

    def _reduce(self, scores: Tensor | None, sq_norm: Tensor | None) -> Tensor:
        """Normalize the summed scores and apply the score mode."""
        assert scores is not None, "Scorer requires at least one module"
        if self.unit_normalize:
            assert sq_norm is not None
            scores = scores / sq_norm.sqrt().clamp_min_(1e-12).unsqueeze(1)

        if self.score_mode == "nearest":
            # Keep the query dimension: ScoreWriter expects [rows, width].
            return scores.max(dim=-1, keepdim=True).values

        return scores


def _materialized(grads: Tensor | OuterProductGradients) -> Tensor:
    if isinstance(grads, OuterProductGradients):
        return grads.materialize().flatten(1)
    return grads

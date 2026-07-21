from collections.abc import Callable

import torch
from torch import Tensor

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
        index_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] = lambda x: x,
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

        # Pre-transposed for scoring: per-module [dim_m, n_queries]
        self.query_grads_t = {
            m: query_grads[m].to(device=self.device, dtype=self.dtype).T
            for m in modules
        }

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, Tensor],
    ):
        """Score a batch of training gradients against all queries."""
        scores = self.score(mod_grads)
        self.writer(indices, scores, query_offset=self.query_offset)

    @torch.inference_mode()
    def score(self, index_grads: dict[str, Tensor]) -> Tensor:
        """Compute scores for a batch of gradients."""
        index_grads = self.index_transform(index_grads)

        # scores[b, q] = sum_m g_b[m] . q_q[m]; per-module GEMMs avoid
        # materializing a [batch, total_dim] concat of the index gradients
        scores = None
        sq_norm = None
        for m in self.modules:
            g = index_grads[m].to(self.device, self.dtype, non_blocking=True)
            part = g @ self.query_grads_t[m]
            scores = part if scores is None else scores.add_(part)
            if self.unit_normalize:
                n = g.pow(2).sum(dim=1)
                sq_norm = n if sq_norm is None else sq_norm.add_(n)

        assert scores is not None, "Scorer requires at least one module"
        if self.unit_normalize:
            assert sq_norm is not None
            scores.div_(sq_norm.sqrt().clamp_min_(1e-12).unsqueeze(1))

        if self.score_mode == "nearest":
            # Keep the query dimension: ScoreWriter expects [rows, width].
            return scores.max(dim=-1, keepdim=True).values

        return scores

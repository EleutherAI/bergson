import torch

from bergson.score.score_writer import ScoreWriter


class Scorer:
    """
    Scores training gradients against query gradients.

    Accepts a ScoreWriter for saving the scores (disk or in-memory).
    """

    def __init__(
        self,
        query_grads: dict[str, torch.Tensor],
        modules: list[str],
        writer: ScoreWriter,
        device: torch.device,
        dtype: torch.dtype,
        *,
        unit_normalize: bool = False,
        score_mode: str = "inner_product",
        token_attributions: bool = False,
    ):
        """
        Initialize the scorer.

        Parameters
        ----------
        query_grads : dict[str, torch.Tensor]
            Query gradients keyed by module name.
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
            Scoring mode: "inner_product" or "nearest".
        token_attributions : bool
            Whether gradients are per-token (rows = total_valid tokens).
        """
        self.device = device
        self.dtype = dtype
        self.modules = modules
        self.unit_normalize = unit_normalize
        self.score_mode = score_mode
        self.token_attributions = token_attributions
        self.writer = writer

        self.query_tensor = torch.cat(
            [query_grads[m].to(device=self.device, dtype=self.dtype) for m in modules],
            dim=1,
        )

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ):
        """Score a batch of training gradients against all queries."""
        # Convert the gradients to the scoring dtype
        if next(iter(mod_grads.values())).dtype != self.dtype:
            mod_grads = {name: grad.to(self.dtype) for name, grad in mod_grads.items()}

        scores = self.score(mod_grads)
        self.writer(indices, scores)

    @torch.inference_mode()
    def score(self, mod_grads: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute scores for a batch of gradients."""
        grads = torch.cat([mod_grads[m].to(self.device) for m in self.modules], dim=1)
        if self.unit_normalize:
            grads = grads / grads.norm(dim=1, keepdim=True)

        if self.score_mode == "nearest":
            all_scores = grads @ self.query_tensor.T
            return all_scores.max(dim=-1).values

        return grads @ self.query_tensor.T

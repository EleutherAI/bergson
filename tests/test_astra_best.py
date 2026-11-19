"""The SGD solver returns the iterate with the lowest windowed objective."""

import logging

import torch

from bergson.config.config import AstraConfig
from bergson.hessians.astra import Astra


class _Hvp:
    rank = 0

    def __init__(self, h):
        self.h = h

    def __call__(self, x, indices):
        return {"m": self.h * x["m"]}


class _Identity:
    def apply(self, v):
        return v


def _solver(cfg: AstraConfig, h: torch.Tensor) -> Astra:
    s = object.__new__(Astra)
    s.cfg, s.names, s.num_docs, s.logger = cfg, ["m"], 8, logging.getLogger("test")
    s.device = s.store_device = torch.device("cpu")
    s.hvp, s.preconditioner, s.damping = _Hvp(h), _Identity(), {"m": torch.tensor(0.0)}
    s.paths = type("P", (), {"checkpoint_path": None})()
    return s


def _objective(h, x, q):
    return float(x @ (h * x) / 2 - x @ q)


def test_best_iterate_kept():
    h, q = torch.tensor([1.0, 10.0]), torch.tensor([1.0, 0.01])
    # lr 0.4 with momentum 0.9 converges along the soft direction (objective
    # -0.5) while the barely excited stiff one (0.4 x 10 > 2 x 1.9) diverges
    # slowly, so the objective falls, then climbs
    cfg = AstraConfig(num_steps=60, lr=0.4, momentum=0.9, batch_size=1, best_window=1)
    x = _solver(cfg, h).refine({"m": q.clone()}, {"m": torch.zeros(2)}, 0)["m"]
    cfg_last = AstraConfig(
        num_steps=60, lr=0.4, momentum=0.9, batch_size=1, best_window=0
    )
    last = _solver(cfg_last, h).refine({"m": q.clone()}, {"m": torch.zeros(2)}, 0)["m"]
    best, final = _objective(h, x, q), _objective(h, last, q)
    assert best < -0.4 < 0 < final, (best, final)
    # the kept iterate is a visited one whose objective is the minimum over the run
    objectives = []
    y, v = torch.zeros(2), torch.zeros(2)
    for _ in range(60):
        objectives.append(_objective(h, y, q))
        v = 0.9 * v + (h * y - q)
        y = y - 0.4 * v
    assert abs(best - min(objectives)) < 1e-5 * max(1.0, abs(best)), (
        best,
        min(objectives),
    )

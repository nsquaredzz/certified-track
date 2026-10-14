"""Camera prior: a global affine fit between consecutive frames, from certskip.warpcert.fit_global.

fit_global(F, C, ...) returns G with F(x) ~ C(G x) + c, so a point at q in the previous
frame C is predicted at G^-1 q in the new frame F.  A heuristic: it moves the search
window and nothing else.
"""
from __future__ import annotations

import numpy as np
from certskip.warpcert import fit_global


class CameraPrior:
    def __init__(self, model: str = "affine", levels=(4, 2), iters=(8, 4)):
        self.model, self.levels, self.iters = model, levels, iters
        self.G = np.eye(3)
        self.info = {}

    def update(self, F: np.ndarray, F_prev: np.ndarray) -> np.ndarray:
        valid = np.ones(F_prev.shape, bool)
        try:
            G, c, info = fit_global(np.asarray(F, np.float64), np.asarray(F_prev, np.float64), valid, self.G,
                                    model=self.model, levels=self.levels, iters=self.iters)
            if not np.all(np.isfinite(G)):
                raise FloatingPointError
            self.G, self.info = G, info
        except Exception:
            self.G, self.info = np.eye(3), {"failed": True}
        return self.G

    def predict(self, p_yx: np.ndarray) -> np.ndarray:
        """Previous-frame positions (N, 2) as (y, x) carried into the new frame."""
        Gi = np.linalg.inv(self.G)
        hom = np.stack([p_yx[:, 1], p_yx[:, 0], np.ones(len(p_yx))], -1) @ Gi.T
        xy = hom[:, :2] / hom[:, 2:3]
        return xy[:, ::-1]

    def reset(self):
        self.G = np.eye(3)

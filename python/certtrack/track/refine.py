"""Sub-pixel refinement of a certified integer position: Gauss-Newton with the template's gradients.

Minimises sum_x (F(k + x + delta) - T(x) - c)^2 over (delta, c) with the template's
gradients standing in for the frame's (inverse-compositional, Baker and Matthews 2004),
so the 3x3 normal matrix is inverted once per template.  The result is reported as the
pose; it is never part of the certificate, which is evaluated at integer positions only.
"""
from __future__ import annotations

import numpy as np


def sample_bilinear(Fpad: np.ndarray, pad: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Bilinear samples of the padded frame at float frame coordinates (any shape)."""
    y = ys + pad
    x = xs + pad
    H, W = Fpad.shape
    y0 = np.clip(np.floor(y).astype(int), 0, H - 2)
    x0 = np.clip(np.floor(x).astype(int), 0, W - 2)
    fy = np.clip(y - y0, 0, 1)
    fx = np.clip(x - x0, 0, 1)
    return ((1 - fy) * (1 - fx) * Fpad[y0, x0] + (1 - fy) * fx * Fpad[y0, x0 + 1]
            + fy * (1 - fx) * Fpad[y0 + 1, x0] + fy * fx * Fpad[y0 + 1, x0 + 1])


def gn_refine(Fpad: np.ndarray, pad: int, T: np.ndarray, gy: np.ndarray, gx: np.ndarray, Hinv: np.ndarray,
              k_yx: np.ndarray, iters: int = 3, clamp: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """delta (N, 2) as (dy, dx) with |delta|_inf <= clamp, and the fitted offset c (N,)."""
    N, s, _ = T.shape
    h = s // 2
    ar = np.arange(-h, h + 1, dtype=np.float64)
    yy = k_yx[:, 0, None, None] + ar[None, :, None]
    xx = k_yx[:, 1, None, None] + ar[None, None, :]
    d = np.zeros((N, 2))
    c = np.zeros(N)
    Td = np.asarray(T, np.float64)
    for _ in range(iters):
        F = sample_bilinear(Fpad, pad, yy + d[:, 0, None, None], xx + d[:, 1, None, None])
        r = F - Td - c[:, None, None]
        b = np.stack([(gx * r).sum((1, 2)), (gy * r).sum((1, 2)), -r.sum((1, 2))], -1)   # J^T r, J = [gx, gy, -1]
        p = -np.einsum("nij,nj->ni", Hinv, b)                                           # Gauss-Newton step
        d[:, 1] = np.clip(d[:, 1] + p[:, 0], -clamp, clamp)
        d[:, 0] = np.clip(d[:, 0] + p[:, 1], -clamp, clamp)
        c = c + p[:, 2]
    return d, c

"""Batched integer search: mean-removed SAD of each template over +-R around its predicted centre.

Coarse pass at every second offset, then the eight neighbours of the best, as
certskip.moving.MovingPruner._follow does.  The SAD only picks the candidate; the
certificate is evaluated afterwards at the integer position it returns.
"""
from __future__ import annotations

import numpy as np

from .template import crops_at


def integer_search(Fpad: np.ndarray, pad: int, T: np.ndarray, centres_yx: np.ndarray, R: int,
                   step: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Best integer centre (N, 2) within +-R of each predicted centre, and its SAD per pixel."""
    N, s, _ = T.shape
    h = s // 2
    win = crops_at(Fpad, pad, centres_yx, h + R).astype(np.float32)    # (N, s+2R, s+2R)
    Tm = (T - T.mean((1, 2), keepdims=True)).astype(np.float32)
    rows = np.arange(N)[:, None, None]
    ar = np.arange(s)

    def cost(oy, ox):
        blk = win[rows, (R + oy)[:, None, None] + ar[None, :, None], (R + ox)[:, None, None] + ar[None, None, :]]
        return np.abs(Tm - (blk - blk.mean((1, 2), keepdims=True))).sum((1, 2))

    best = np.full(N, np.inf, np.float32)
    by = np.zeros(N, int)
    bx = np.zeros(N, int)
    coarse = range(-(R // step) * step, R + 1, step)
    for span in (coarse, (-1, 0, 1)):
        cy, cx = by.copy(), bx.copy()
        for oy in span:
            for ox in span:
                ty, tx = cy + oy, cx + ox
                inb = (np.abs(ty) <= R) & (np.abs(tx) <= R)
                c = cost(np.clip(ty, -R, R), np.clip(tx, -R, R))
                c = np.where(inb, c, np.inf)
                hit = c < best
                best[hit] = c[hit]
                by[hit] = ty[hit]
                bx[hit] = tx[hit]
    k = centres_yx + np.stack([by, bx], -1)
    return k, best / (s * s)

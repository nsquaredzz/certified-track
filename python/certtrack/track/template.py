"""Per-point templates taken once from the query frame, and everything derived from them."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .certify import kappa, selfmatch, selfmatch_stats, structure_lambda_min


def pad_frame(F: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(np.asarray(F, np.float32), pad, mode="edge")


def crops_at(Fpad: np.ndarray, pad: int, centres_yx: np.ndarray, half: int) -> np.ndarray:
    """(N, 2h+1, 2h+1) windows of the padded frame centred on integer (y, x) frame coordinates."""
    L = 2 * half + 1
    win = np.lib.stride_tricks.sliding_window_view(Fpad, (L, L))
    cy = centres_yx[:, 0] + pad - half
    cx = centres_yx[:, 1] + pad - half
    return win[cy, cx]


def gradients(T: np.ndarray):
    """Central differences of a template stack (N, s, s), one-sided at the border (as certskip.warp._grad)."""
    g = np.asarray(T, np.float64)
    gy = np.zeros_like(g)
    gx = np.zeros_like(g)
    gy[:, 1:-1] = 0.5 * (g[:, 2:] - g[:, :-2]); gy[:, 0] = g[:, 1] - g[:, 0]; gy[:, -1] = g[:, -1] - g[:, -2]
    gx[:, :, 1:-1] = 0.5 * (g[:, :, 2:] - g[:, :, :-2]); gx[:, :, 0] = g[:, :, 1] - g[:, :, 0]; gx[:, :, -1] = g[:, :, -1] - g[:, :, -2]
    return gy, gx


def normal_inverse(gy: np.ndarray, gx: np.ndarray, tikhonov: float | None = None) -> np.ndarray:
    """Inverse of J^T J for J = [gx, gy, -1] per template, with +n on the motion diagonal (certskip.warp)."""
    n = gy.shape[-1] * gy.shape[-2]
    tk = float(n if tikhonov is None else tikhonov)
    A = np.empty(gy.shape[:-2] + (3, 3))
    A[..., 0, 0] = (gx * gx).sum((-2, -1)) + tk
    A[..., 0, 1] = A[..., 1, 0] = (gx * gy).sum((-2, -1))
    A[..., 1, 1] = (gy * gy).sum((-2, -1)) + tk
    A[..., 0, 2] = A[..., 2, 0] = -gx.sum((-2, -1))
    A[..., 1, 2] = A[..., 2, 1] = -gy.sum((-2, -1))
    A[..., 2, 2] = n
    return np.linalg.inv(A)


@dataclass
class Templates:
    T: np.ndarray          # (N, s, s) float32 raw luma of the query frame
    k: np.ndarray          # (N, 2) int   template centre (y, x) in frame pixels
    o: np.ndarray          # (N, 2) float query point minus centre, (y, x)
    gy: np.ndarray
    gx: np.ndarray
    Hinv: np.ndarray       # (N, 3, 3)
    kappa: np.ndarray      # (N, n_scales) int
    n_self: np.ndarray     # (N,) int   shifts with |u|_inf >= 2 that certify against the template itself
    margin: np.ndarray     # (N,) float smallest self-match score among those shifts
    lam_min: np.ndarray    # (N,) float structure-tensor lambda_min per pixel
    border: np.ndarray     # (N,) bool  template window reached outside the frame (edge-padded pixels)
    selfmap: np.ndarray | None = None  # (N, 2R+1, 2R+1) float16 full self-match maps, when kept
    J: np.ndarray | None = None        # (N, s*s, P+1) Jacobian for the warp group, when a group is used
    HinvJ: np.ndarray | None = None    # (N, P+1, P+1) its normal-matrix inverse

    @property
    def s(self) -> int:
        return int(self.T.shape[-1])


def make_templates(F: np.ndarray, points_xy: np.ndarray, s: int, deltas: dict, self_radius: int,
                   offset: str = "opt", keep_maps: bool = False, group: str | None = None) -> Templates:
    """Templates of side s centred at round(point) in frame F (H, W) for points (N, 2) as (x, y)."""
    H, W = F.shape
    h = s // 2
    pad = h + self_radius + 2
    Fp = pad_frame(F, pad)
    p = np.asarray(points_xy, np.float64)[:, ::-1]           # (y, x)
    k = np.rint(p).astype(int)
    o = p - k
    T = crops_at(Fp, pad, k, h).astype(np.float32)
    gy, gx = gradients(T)
    Hinv = normal_inverse(gy, gx)
    N = k.shape[0]
    kap = np.zeros((N, len(deltas)), np.int32)
    n_self = np.zeros(N, np.int32)
    margin = np.zeros(N)
    lam = np.zeros(N)
    maps = np.zeros((N, 2 * self_radius + 1, 2 * self_radius + 1), np.float16) if keep_maps else None
    wins = crops_at(Fp, pad, k, h + self_radius)
    for i in range(N):
        kap[i] = kappa(T[i], deltas)
        sm = selfmatch(wins[i], s, deltas, offset)
        n_self[i], margin[i] = selfmatch_stats(sm)
        lam[i] = structure_lambda_min(T[i])
        if keep_maps:
            maps[i] = sm
    border = (k[:, 0] - h < 0) | (k[:, 1] - h < 0) | (k[:, 0] + h >= H) | (k[:, 1] + h >= W)
    J = HJ = None
    if group:
        from .warpfit import jacobian, normal_inverse as warp_normal_inverse
        J = jacobian(gy, gx, group)
        HJ = warp_normal_inverse(J, s)
    return Templates(T, k, o, gy, gx, Hinv, kap, n_self, margin, lam, border, maps, J, HJ)

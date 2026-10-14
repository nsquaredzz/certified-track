"""Per-point Gauss-Newton over a small warp group, and the frame resampled by the fitted warp.

The template keeps its own coordinates x in [-h, h]^2.  The pose maps them into the
frame as  p(x) = k + delta + A x  with A the identity (translation), a similarity
(1 + s) R(theta) (4 parameters with delta) or a general affine (6).  Residual
r(x) = F(p(x)) - T(x) - c, minimised with the template's gradients standing in for the
frame's (inverse-compositional, Baker and Matthews 2004), so the normal matrix is one
inverse per template.  The output crop F o p is what the certificate is evaluated on in
'interp' mode: Theorem 3 is exact for that resampled image, and the proof names the warp.
"""
from __future__ import annotations

import numpy as np

from .refine import sample_bilinear

GROUPS = {"translation": 2, "similarity": 4, "affine": 6}
LIMITS = {"delta": 1.5, "scale": 0.2, "angle": 0.2, "affine": 0.2}


def grids(s: int):
    h = s // 2
    ar = np.arange(-h, h + 1, dtype=np.float64)
    yy, xx = np.meshgrid(ar, ar, indexing="ij")
    return yy, xx


def jacobian(gy: np.ndarray, gx: np.ndarray, group: str) -> np.ndarray:
    """(N, s*s, P+1) Jacobian of the residual in the group's parameters plus the offset c."""
    N, s, _ = gy.shape
    yy, xx = grids(s)
    cols = [gx, gy]
    if group == "similarity":
        cols += [gx * xx + gy * yy, -gx * yy + gy * xx]
    elif group == "affine":
        cols += [gx * xx, gx * yy, gy * xx, gy * yy]
    elif group != "translation":
        raise ValueError(group)
    cols.append(-np.ones_like(gx))
    return np.stack([c.reshape(N, -1) for c in cols], -1)


def normal_inverse(J: np.ndarray, s: int) -> np.ndarray:
    """Inverse of J^T J with a Tikhonov term that says 'no motion' unless the gradients argue."""
    n = s * s
    h = s // 2
    H = np.einsum("npi,npj->nij", J, J)
    P = J.shape[-1] - 1
    reg = np.full(P + 1, float(n))
    if P > 2:
        reg[2:P] = n * (h * h / 3.0)      # the linear terms multiply coordinates of size ~h
    reg[P] = 0.0
    H = H + np.diag(reg)[None]
    return np.linalg.inv(H)


def warp_matrix(params: np.ndarray, group: str) -> np.ndarray:
    """(N, 2, 2) A for the group parameters (after delta)."""
    N = params.shape[0]
    A = np.tile(np.eye(2), (N, 1, 1))
    if group == "similarity":
        sc, th = params[:, 2], params[:, 3]
        A[:, 0, 0] = 1 + sc; A[:, 0, 1] = -th
        A[:, 1, 0] = th;     A[:, 1, 1] = 1 + sc
    elif group == "affine":
        A[:, 0, 0] = 1 + params[:, 2]; A[:, 0, 1] = params[:, 3]
        A[:, 1, 0] = params[:, 4];     A[:, 1, 1] = 1 + params[:, 5]
    return A


def sample_warped(Fpad: np.ndarray, pad: int, k_yx: np.ndarray, params: np.ndarray, group: str, s: int) -> np.ndarray:
    """(N, s, s) frame samples at k + delta + A x.  params[:, :2] is delta as (dy, dx)... see below.

    Parameter layout: [dx, dy, (group terms), c]; A acts on (x, y) template coordinates.
    """
    yy, xx = grids(s)
    A = warp_matrix(params, group)
    X = A[:, 0, 0, None, None] * xx + A[:, 0, 1, None, None] * yy + k_yx[:, 1, None, None] + params[:, 0, None, None]
    Y = A[:, 1, 0, None, None] * xx + A[:, 1, 1, None, None] * yy + k_yx[:, 0, None, None] + params[:, 1, None, None]
    return sample_bilinear(Fpad, pad, Y, X)


def clamp(params: np.ndarray, group: str) -> np.ndarray:
    p = params.copy()
    p[:, 0:2] = np.clip(p[:, 0:2], -LIMITS["delta"], LIMITS["delta"])
    if group == "similarity":
        p[:, 2] = np.clip(p[:, 2], -LIMITS["scale"], LIMITS["scale"])
        p[:, 3] = np.clip(p[:, 3], -LIMITS["angle"], LIMITS["angle"])
    elif group == "affine":
        p[:, 2:6] = np.clip(p[:, 2:6], -LIMITS["affine"], LIMITS["affine"])
    return p


def gn_fit(Fpad: np.ndarray, pad: int, T: np.ndarray, J: np.ndarray, Hinv: np.ndarray, k_yx: np.ndarray,
           group: str, iters: int = 4, init: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fitted parameters (N, P+1) and the warped crop F o p (N, s, s) at the solution."""
    N, s, _ = T.shape
    P = GROUPS[group]
    params = np.zeros((N, P + 1)) if init is None else init.copy()
    Td = np.asarray(T, np.float64)
    crop = None
    for _ in range(iters):
        crop = sample_warped(Fpad, pad, k_yx, params, group, s)
        r = (crop - Td - params[:, P, None, None]).reshape(N, -1)
        step = np.einsum("nij,npj,np->ni", Hinv, J, r)
        params = clamp(params - step, group)
    crop = sample_warped(Fpad, pad, k_yx, params, group, s)
    return params, crop


def point_offset(params: np.ndarray, o_yx: np.ndarray, group: str) -> np.ndarray:
    """Where the query point (offset o from the template centre, (y, x)) lands: delta + A o, as (y, x)."""
    A = warp_matrix(params, group)
    ox, oy = o_yx[:, 1], o_yx[:, 0]
    x = A[:, 0, 0] * ox + A[:, 0, 1] * oy + params[:, 0]
    y = A[:, 1, 0] * ox + A[:, 1, 1] * oy + params[:, 1]
    return np.stack([y, x], -1)

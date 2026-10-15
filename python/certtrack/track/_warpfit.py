"""Global warp fit between two frames, vendored from certified-skip's `certskip/warpcert.py` (fit_global and
its helpers, unchanged), which is not yet published in that repository.  Same author, same MIT licence.
"""
from __future__ import annotations

import numpy as np

def apply_h(G: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Apply a 3x3 homography to pixel coordinates. Returns (u, v)."""
    u = G[0, 0] * x + G[0, 1] * y + G[0, 2]
    v = G[1, 0] * x + G[1, 1] * y + G[1, 2]
    w = G[2, 0] * x + G[2, 1] * y + G[2, 2]
    return u / w, v / w


def translation(tx: float, ty: float) -> np.ndarray:
    G = np.eye(3)
    G[0, 2] = tx; G[1, 2] = ty
    return G


def _snap(a: np.ndarray) -> np.ndarray:
    """Snap values within 1e-6 of an integer to that integer, so a fixed camera samples exactly."""
    r = np.rint(a)
    return np.where(np.abs(a - r) < _SNAP, r, a)


def sample_bilinear(img: np.ndarray, valid: np.ndarray, px: np.ndarray, py: np.ndarray, extra: np.ndarray | None = None):
    """Bilinear sample of `img` at float positions (px, py).

    Returns (value, ok, osc, nn) where ok marks samples whose four corners are inside the image and
    valid; osc is the oscillation of the four corner values plus the largest `extra` (write-time
    oscillation) under them, i.e. the per-pixel interpolation term of Proposition B; nn is the
    nearest-corner value W_NN.  Samples exactly on an integer position have osc = 0 (plus extra)."""
    H, W = img.shape
    px = _snap(np.asarray(px, np.float64)); py = _snap(np.asarray(py, np.float64))
    x0 = np.floor(px); y0 = np.floor(py)
    fx = px - x0; fy = py - y0
    x0 = x0.astype(np.int64); y0 = y0.astype(np.int64)
    exact_x = fx == 0; exact_y = fy == 0
    # a sample exactly on a column needs only that column; treat x1 = x0 there
    x1 = np.where(exact_x, x0, x0 + 1); y1 = np.where(exact_y, y0, y0 + 1)
    inside = (x0 >= 0) & (y0 >= 0) & (x1 <= W - 1) & (y1 <= H - 1)
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1); y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)
    v00 = img[y0c, x0c]; v01 = img[y0c, x1c]; v10 = img[y1c, x0c]; v11 = img[y1c, x1c]
    ok = inside & valid[y0c, x0c] & valid[y0c, x1c] & valid[y1c, x0c] & valid[y1c, x1c]
    val = (1 - fy) * ((1 - fx) * v00 + fx * v01) + fy * ((1 - fx) * v10 + fx * v11)
    stack = np.stack([v00, v01, v10, v11])
    osc = stack.max(0) - stack.min(0)
    if extra is not None:
        e = np.stack([extra[y0c, x0c], extra[y0c, x1c], extra[y1c, x0c], extra[y1c, x1c]]).max(0)
        osc = osc + e
    nx = np.where(fx < 0.5, x0c, x1c); ny = np.where(fy < 0.5, y0c, y1c)
    nn = img[ny, nx]
    return val, ok, osc, nn


def _gradients(img: np.ndarray, valid: np.ndarray | None = None):
    g = np.asarray(img, np.float64)
    gy = np.zeros_like(g); gx = np.zeros_like(g)
    gy[1:-1] = 0.5 * (g[2:] - g[:-2]); gx[:, 1:-1] = 0.5 * (g[:, 2:] - g[:, :-2])
    if valid is not None:
        ok = valid.copy()
        ok[1:-1] &= valid[2:] & valid[:-2]
        gy = np.where(ok, gy, 0.0)
        ok = valid.copy()
        ok[:, 1:-1] &= valid[:, 2:] & valid[:, :-2]
        gx = np.where(ok, gx, 0.0)
    return gy, gx


def _downsample(img: np.ndarray, valid: np.ndarray, s: int):
    """Box downsample by s with a validity mask (coarse pixel valid if at least half its fine pixels are)."""
    if s == 1:
        return np.asarray(img, np.float64), valid.copy()
    H, W = img.shape
    h, w = H // s, W // s
    a = np.asarray(img, np.float64)[: h * s, : w * s].reshape(h, s, w, s)
    v = valid[: h * s, : w * s].reshape(h, s, w, s)
    cnt = v.sum((1, 3))
    out = np.where(cnt > 0, (a * v).sum((1, 3)) / np.maximum(cnt, 1), 0.0)
    return out, cnt >= (s * s + 1) // 2


# ---------------------------------------------------------------- global registration
def fit_global(F: np.ndarray, C: np.ndarray, valid: np.ndarray, G0: np.ndarray, model: str = "homography",
               levels=(4, 2, 1), iters=(10, 8, 4), huber: bool = True, tol: float = 0.01):
    """Gauss-Newton fit of G (frame -> canvas) and a brightness offset c on F(x) ~ C(G x) + c.

    Coarse to fine over `levels` (downsampling factors), with Huber weights on the residual so that
    the fit follows the majority (the background) and not the moving objects.  `model` is
    'homography' (8 parameters), 'affine' (6) or 'translation' (2).  Returns (G, c, info) where
    info has the robust residual scale and the fraction of frame pixels with a valid reference.
    A heuristic: the certificate never depends on how G was chosen."""
    G = np.asarray(G0, np.float64).copy(); G /= G[2, 2]
    F = np.asarray(F, np.float64)
    nparam = {"homography": 8, "affine": 6, "translation": 2}[model]
    c = 0.0
    info = {"scale": np.nan, "coverage": 0.0}
    for s, it in zip(levels, iters):
        Fs, _ = _downsample(F, np.ones(F.shape, bool), s)
        Cs, Vs = _downsample(C, valid, s)
        if Vs.sum() < 64:
            continue
        Cy, Cx = _gradients(Cs, Vs)
        S = np.diag([1.0 / s, 1.0 / s, 1.0]); Si = np.diag([float(s), float(s), 1.0])
        Gs = S @ G @ Si; Gs /= Gs[2, 2]
        h, w = Fs.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        x = xx.ravel(); y = yy.ravel(); f = Fs.ravel()
        for _ in range(it):
            u, v = apply_h(Gs, x, y)
            wv, ok, _, _ = sample_bilinear(Cs, Vs, u, v)
            gx, _, _, _ = sample_bilinear(Cx, Vs, u, v)
            gy, _, _, _ = sample_bilinear(Cy, Vs, u, v)
            r = f - wv - c
            if ok.sum() < 32:
                break
            r = r[ok]; gx = gx[ok]; gy = gy[ok]; xo = x[ok]; yo = y[ok]; uo = u[ok]; vo = v[ok]
            wdn = Gs[2, 0] * xo + Gs[2, 1] * yo + 1.0
            if huber:
                med = np.median(r); sc = 1.4826 * np.median(np.abs(r - med)) + 1e-3
                wt = np.minimum(1.0, 1.345 * sc / np.maximum(np.abs(r - med), 1e-9))
            else:
                wt = np.ones_like(r)
            cols = []
            if model == "translation":
                cols = [gx / wdn, gy / wdn]
            else:
                cols = [gx * xo / wdn, gx * yo / wdn, gx / wdn, gy * xo / wdn, gy * yo / wdn, gy / wdn]
                if model == "homography":
                    cols += [-(gx * uo + gy * vo) * xo / wdn, -(gx * uo + gy * vo) * yo / wdn]
            cols.append(np.ones_like(r))
            J = np.stack(cols, 1)
            A = (J * wt[:, None]).T @ J + 1e-6 * np.eye(J.shape[1])
            b = (J * wt[:, None]).T @ r
            try:
                step = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                break
            cost0 = float((wt * r * r).sum())
            # damped acceptance
            accepted = False
            for damp in (1.0, 0.5, 0.25, 0.125):
                Gn = Gs.copy(); cn = c + damp * step[-1]
                st = damp * step[:-1]
                if model == "translation":
                    Gn[0, 2] += st[0]; Gn[1, 2] += st[1]
                else:
                    Gn[0, 0] += st[0]; Gn[0, 1] += st[1]; Gn[0, 2] += st[2]
                    Gn[1, 0] += st[3]; Gn[1, 1] += st[4]; Gn[1, 2] += st[5]
                    if model == "homography":
                        Gn[2, 0] += st[6]; Gn[2, 1] += st[7]
                un, vn = apply_h(Gn, xo, yo)
                wvn, okn, _, _ = sample_bilinear(Cs, Vs, un, vn)
                rn = np.where(okn, f[ok] - wvn - cn, 0.0)
                cost1 = float((wt * rn * rn).sum())
                if cost1 <= cost0:
                    Gs, c, accepted = Gn, cn, True
                    break
            if not accepted:
                break
            # convergence: largest displacement change over the frame
            u2, v2 = apply_h(Gs, xo, yo)
            if max(np.abs(u2 - uo).max(), np.abs(v2 - vo).max()) < tol / s:
                break
        G = Si @ Gs @ S; G /= G[2, 2]
    # final residual at full resolution; also try the nearest integer translation, so that a camera
    # that did not move lands on exact pixels (interpolation term zero) when that is at least as good
    h, w = F.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    x = xx.ravel(); y = yy.ravel(); f = F.ravel()

    def robust_cost(Gc):
        u, v = apply_h(Gc, x, y)
        wv, ok, _, _ = sample_bilinear(C, valid, u, v)
        if ok.sum() < 32:
            return np.inf, np.nan, 0.0, c
        r = (f - wv)[ok]
        cc = float(np.median(r)); r = r - cc
        sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-3
        k = 1.345 * sc
        hub = np.where(np.abs(r) <= k, 0.5 * r * r, k * (np.abs(r) - 0.5 * k))
        return float(hub.mean()), float(sc), float(ok.mean()), cc

    cost, scale, cov, c = robust_cost(G)
    lin = G[:2, :2] - np.eye(2)
    if np.abs(lin).max() < 1e-3 and np.abs(G[2, :2]).max() < 1e-6:
        Gs = G.copy(); Gs[:2, :2] = np.eye(2); Gs[2, :2] = 0; Gs[0, 2] = np.rint(G[0, 2]); Gs[1, 2] = np.rint(G[1, 2])
        if np.abs(Gs[:2, 2] - G[:2, 2]).max() < 0.5:
            cost_s, scale_s, cov_s, c_s = robust_cost(Gs)
            if cost_s <= cost:
                G, cost, scale, cov, c = Gs, cost_s, scale_s, cov_s, c_s
                info["snapped"] = True
    info["scale"] = scale; info["coverage"] = cov; info["cost"] = cost
    return G, float(c), info


# ---------------------------------------------------------------- the pruner

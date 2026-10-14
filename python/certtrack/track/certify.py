"""The certificate: the multi-scale box test on raw pixels, and what is computed once per template.

score(crop, T) = max_r  max_{(2r+1)-box B in the patch} |mean_B (crop - T - c)| / eps_r,   eps_r = Delta_r / 2.

score < 1 is the rule.  Both crop and T are explicit arrays of pixels at integer
positions, so THEORY.md Theorem 3 of certified-skip applies with no interpolation
term: per scale r, no feature of persistence >= Delta_r of the box-filtered template
is unmatched in the box-filtered crop, and none appears.  With c the midrange the
test is invariant to a brightness offset; c = 0 certifies the raw crop (THEORY.md §9).

kappa[r]   number of features of the box-r filtered template with persistence >= Delta_r
           (dark and bright).  kappa[r] = 0 means the certificate at scale r is vacuous
           for this template; it is reported with every proof.
selfmatch  the score of the query window against itself at every integer shift within
           +-radius: the set S_q where it is < 1 is the position-error object of
           Theorem P.  n_self counts shifts with |u|_inf >= 2 that pass; margin is the
           smallest score among them (how far the template is from matching elsewhere).
"""
from __future__ import annotations

import numpy as np
from certskip.scalespace import box_filter, box_max_abs_mean
from certskip.topology import features_at_least, persistence_h0


def schedule(delta0: float, radii=(1, 2, 3)) -> dict:
    """Delta_r = Delta_0 (2r+1)^-1/2, the schedule of certified-skip's benchmarks."""
    return {int(r): float(delta0) / np.sqrt(2 * r + 1) for r in radii}


def offsets(d: np.ndarray, offset: str) -> np.ndarray:
    """The brightness constant removed per patch, for a residual stack d[..., s, s]."""
    if offset == "midrange":
        return 0.5 * (d.max((-2, -1)) + d.min((-2, -1)))
    if offset == "mean":
        return d.mean((-2, -1))
    if offset == "none":
        return np.zeros(d.shape[:-2])
    raise ValueError(offset)


def box_mean_extremes(d: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """(max, min) over all (2r+1)-boxes inside the patch of the box mean of d[..., P, P]."""
    if r == 0:
        return d.max(axis=(-2, -1)), d.min(axis=(-2, -1))
    k = 2 * r + 1
    P = d.shape[-1]
    if k > P:
        z = np.zeros(d.shape[:-2])
        return z, z
    S = np.zeros(d.shape[:-2] + (P + 1, P + 1), np.float64)
    S[..., 1:, 1:] = d.cumsum(axis=-2).cumsum(axis=-1)
    box = (S[..., k:, k:] - S[..., :-k, k:] - S[..., k:, :-k] + S[..., :-k, :-k]) / (k * k)
    return box.max(axis=(-2, -1)), box.min(axis=(-2, -1))


def optimal_offset(d: np.ndarray, deltas: dict) -> tuple[np.ndarray, np.ndarray]:
    """The constant c that minimises the multi-scale score, exactly, and that score.

    Box means are linear in c, so score(c) = max_r max(M_r - c, c - m_r) / eps_r with M_r, m_r
    the largest and smallest box means of d at radius r: a maximum of 2R lines in c.  Its
    minimum sits where a rising line meets a falling one, so it is one of the R^2 crossings
    c = (M_r' eps_r + m_r eps_r') / (eps_r + eps_r').  The midrange is the special case R = {0}.
    """
    eps = {r: 0.5 * D for r, D in deltas.items()}
    M, m = {}, {}
    for r in deltas:
        M[r], m[r] = box_mean_extremes(d, r)
    cands = [(M[a] * eps[b] + m[b] * eps[a]) / (eps[a] + eps[b]) for a in deltas for b in deltas]
    best_c = None
    best = None
    for c in cands:
        sc = np.zeros(d.shape[:-2])
        for r in deltas:
            sc = np.maximum(sc, np.maximum(M[r] - c, c - m[r]) / eps[r])
        if best is None:
            best, best_c = sc, c
        else:
            better = sc < best
            best = np.where(better, sc, best)
            best_c = np.where(better, c, best_c)
    return best_c, best


def score(crops: np.ndarray, templates: np.ndarray, deltas: dict, offset: str = "opt"):
    """Certificate score and offset for stacks crops[..., s, s] against templates[..., s, s].

    offset 'opt' is the exact minimiser over c (sound for any c, so the best one is used);
    'midrange', 'mean' and 'none' are the fixed choices of certified-skip.
    """
    d = np.asarray(crops, np.float64) - np.asarray(templates, np.float64)
    if offset == "opt":
        c, sc = optimal_offset(d, deltas)
        return sc, c
    c = offsets(d, offset)
    e = d - c[..., None, None]
    sc = np.zeros(d.shape[:-2])
    for r, D in deltas.items():
        sc = np.maximum(sc, box_max_abs_mean(e, r) / (0.5 * D))
    return sc, c


def kappa(template: np.ndarray, deltas: dict) -> np.ndarray:
    """Per scale: number of template features (dark + bright) with persistence >= Delta_r."""
    out = []
    for r, D in deltas.items():
        bf = box_filter(template, r)
        if bf.size == 0:
            out.append(0)
            continue
        out.append(features_at_least(persistence_h0(bf), D) + features_at_least(persistence_h0(bf, superlevel=True), D))
    return np.asarray(out, np.int32)


def selfmatch(window: np.ndarray, s: int, deltas: dict, offset: str = "opt", chunk: int = 512) -> np.ndarray:
    """Score of the central s x s crop of `window` against every integer shift; shape (2R+1, 2R+1)."""
    L = window.shape[-1]
    R = (L - s) // 2
    T = window[R:R + s, R:R + s]
    win = np.lib.stride_tricks.sliding_window_view(window, (s, s))  # (2R+1, 2R+1, s, s)
    flat = win.reshape(-1, s, s)
    out = np.empty(flat.shape[0])
    for i in range(0, flat.shape[0], chunk):
        out[i:i + chunk] = score(flat[i:i + chunk], T[None], deltas, offset)[0]
    return out.reshape(2 * R + 1, 2 * R + 1)


def selfmatch_stats(sm: np.ndarray, eta: float = 0.0) -> tuple[int, float]:
    """(n_self, margin) over shifts with |u|_inf >= 2: count passing at 1 + eta, and the smallest score."""
    R = (sm.shape[0] - 1) // 2
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    far = np.maximum(np.abs(yy), np.abs(xx)) >= 2
    vals = sm[far]
    return int(np.sum(vals < 1.0 + eta)), float(vals.min()) if vals.size else float("inf")


def structure_lambda_min(template: np.ndarray) -> float:
    """Smallest eigenvalue of the structure tensor summed over the template (Shi-Tomasi), per pixel."""
    g = np.asarray(template, np.float64)
    gy, gx = np.gradient(g)
    a, b, c = (gx * gx).sum(), (gx * gy).sum(), (gy * gy).sum()
    n = g.size
    tr, det = a + c, a * c - b * b
    disc = max(tr * tr / 4 - det, 0.0)
    return float((tr / 2 - np.sqrt(disc)) / n)

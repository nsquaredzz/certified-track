"""The certified tracker without a learned model: anchored templates, integer search, certificate, refinement.

Per frame, for every active point: predict the template centre (camera prior plus
residual velocity), search +-R for the best integer match, certify it against the
original template, refine to sub-pixel if certified.  A point whose match does not
certify is LOST: it keeps a camera-carried prediction and is searched for again every
frame around it, so re-detection comes with the same proof as tracking.

Outputs are per (frame, point): position (x, y), state, certificate score, offset c,
the integer position the proof refers to, and the Gauss-Newton delta.
"""
from __future__ import annotations

import numpy as np

from .. import states
from .camera import CameraPrior
from .certify import schedule, score as cert_score
from .refine import gn_refine, sample_bilinear
from .search import integer_search
from .template import Templates, crops_at, make_templates, pad_frame
from .warpfit import GROUPS, gn_fit, point_offset


def interp_crops(Fpad: np.ndarray, pad: int, k_yx: np.ndarray, delta_yx: np.ndarray, h: int) -> np.ndarray:
    """(N, s, s) bilinear samples of the frame at k + delta: the resampled image F o theta."""
    ar = np.arange(-h, h + 1, dtype=np.float64)
    yy = k_yx[:, 0, None, None] + delta_yx[:, 0, None, None] + ar[None, :, None]
    xx = k_yx[:, 1, None, None] + delta_yx[:, 1, None, None] + ar[None, None, :]
    return sample_bilinear(Fpad, pad, yy, xx)


class CertifiedTracker:
    def __init__(self, delta0: float = 80.0, radii=(1, 2, 3), s: int = 21, R: int = 16, offset: str = "opt",
                 camera: bool = True, refine: bool = True, velocity: float = 0.5, self_radius: int | None = None,
                 lost_radius: int | None = None, keep_maps: bool = False, interp: bool = False,
                 group: str = "translation"):
        """interp=False certifies the raw frame crop at the integer position (iota = 0).
        interp=True certifies the bilinearly resampled frame at the refined sub-pixel pose: exact for
        the resampled image F o theta, and the proof says so.  group ('translation', 'similarity',
        'affine') is the pose group of that refinement; anything but translation implies interp."""
        self.deltas = schedule(delta0, radii)
        self.delta0, self.radii = float(delta0), tuple(radii)
        self.s, self.R, self.offset = int(s), int(R), offset
        if group not in GROUPS:
            raise ValueError(group)
        self.group = group
        self.interp = bool(interp) or group != "translation"
        self.use_camera, self.use_refine, self.vel_alpha = camera, refine, float(velocity)
        self.self_radius = int(self_radius if self_radius is not None else R)
        self.lost_radius = int(lost_radius if lost_radius is not None else R)
        self.keep_maps = keep_maps

    # ------------------------------------------------------------------ one clip
    def run(self, frames, queries_txy: np.ndarray, oracle_xy: np.ndarray | None = None,
            proposals_xy: np.ndarray | None = None, proposals_vis: np.ndarray | None = None,
            proposal_radius: int = 4, snap_tol: float = 2.0, gate_self: bool = True) -> dict:
        """frames: (T, H, W) grey array, or an object with __len__ and frame(t) -> (H, W) grey.

        queries (N, 3) as (t, x, y).  oracle_xy (T, N, 2): if given, the search is replaced by
        the rounded oracle position (analysis of the certificate alone, never a tracker).

        proposals_xy (T, N, 2) and proposals_vis (T, N): a frozen model's tracks.  The verify
        mode: the template is also searched within +-proposal_radius of the model's position; a
        certified match there (or at the carried prediction) is the output, with a proof; otherwise
        the model's position is the output, flagged MODEL_HELD when the model says visible and
        OCCLUDED when it does not.  The model never sees the certificate.

        Position policy in verify mode (Theorem P: a certificate bounds position only through the
        template's self-match set): when the certified position and the model's agree within
        snap_tol px the model's sub-pixel position is reported under the certificate; when they
        disagree, the certified position is reported only for templates whose self-match set is
        {0} (gate_self), otherwise the step is MODEL_HELD and the model's position stands.
        """
        get = (lambda t: frames[t]) if isinstance(frames, np.ndarray) else frames.frame
        T_total = len(frames)
        q = np.asarray(queries_txy, np.float64)
        N = q.shape[0]
        qt = np.rint(q[:, 0]).astype(int)
        h = self.s // 2
        pad = h + max(self.R, self.lost_radius, self.self_radius) + 2

        out_xy = np.zeros((T_total, N, 2), np.float32)
        out_state = np.zeros((T_total, N), np.uint8)
        out_score = np.full((T_total, N), np.nan, np.float32)
        out_c = np.zeros((T_total, N), np.float32)
        out_k = np.zeros((T_total, N, 2), np.int32)
        out_delta = np.zeros((T_total, N, 2), np.float32)
        out_source = np.zeros((T_total, N), np.uint8)   # 0 none, 1 carried prediction, 2 model proposal, 3 certified but not unique
        use_prop = proposals_xy is not None
        if use_prop:
            proposals_xy = np.asarray(proposals_xy, np.float64)
            proposals_vis = np.asarray(proposals_vis, bool) if proposals_vis is not None else np.ones(proposals_xy.shape[:2], bool)

        tmpl: Templates | None = None
        pos = np.zeros((N, 2))          # current template-centre pose (y, x), float
        vel = np.zeros((N, 2))
        active = np.zeros(N, bool)
        cam = CameraPrior() if self.use_camera else None
        F_prev = None
        meta = None

        for t in range(T_total):
            F = np.asarray(get(t), np.float32)
            Fp = pad_frame(F, pad)
            if cam is not None and F_prev is not None and active.any():
                cam.update(F, F_prev)

            # ---- predict every active point into this frame
            if active.any():
                idx = np.nonzero(active)[0]
                p_prev = pos[idx]
                p_hat = cam.predict(p_prev) if cam is not None else p_prev.copy()
                p_hat = p_hat + vel[idx]
                lost = out_state[t - 1, idx] != states.CERTIFIED
                if oracle_xy is not None:
                    k0 = np.rint(oracle_xy[t, idx][:, ::-1] - tmpl.o[idx]).astype(int)
                    k = k0
                else:
                    k0 = self._clip(np.rint(p_hat).astype(int), F.shape, h)
                    k = np.empty_like(k0)
                    if (~lost).any():
                        k[~lost], _ = integer_search(Fp, pad, tmpl.T[idx[~lost]], k0[~lost], self.R)
                    if lost.any():
                        k[lost], _ = integer_search(Fp, pad, tmpl.T[idx[lost]], k0[lost], self.lost_radius)
                k = self._clip(k, F.shape, h)
                source = np.ones(len(idx), np.uint8)
                if use_prop and oracle_xy is None:
                    # a second candidate from the model's proposal; the better certificate score wins
                    kp = self._clip(np.rint(proposals_xy[t, idx][:, ::-1] - tmpl.o[idx]).astype(int), F.shape, h)
                    kp, _ = integer_search(Fp, pad, tmpl.T[idx], kp, proposal_radius)
                    kp = self._clip(kp, F.shape, h)
                    sc_a, _ = cert_score(crops_at(Fp, pad, k, h), tmpl.T[idx], self.deltas, self.offset)
                    sc_b, _ = cert_score(crops_at(Fp, pad, kp, h), tmpl.T[idx], self.deltas, self.offset)
                    take_b = sc_b < sc_a
                    k = np.where(take_b[:, None], kp, k)
                    source = np.where(take_b, 2, 1).astype(np.uint8)
                delta = np.zeros((len(idx), 2))
                point_shift = None       # (n, 2) where the query point lands relative to k, (y, x); None = delta + o
                if self.interp:
                    # refine first, then certify the resampled frame at the fitted pose against the raw template
                    if self.group == "translation":
                        d, _ = gn_refine(Fp, pad, tmpl.T[idx], tmpl.gy[idx], tmpl.gx[idx], tmpl.Hinv[idx], k, clamp=1.5)
                        crop_w = interp_crops(Fp, pad, k, d, h)
                    else:
                        prm, crop_w = gn_fit(Fp, pad, tmpl.T[idx], tmpl.J[idx], tmpl.HinvJ[idx], k, self.group)
                        d = prm[:, [1, 0]]                                   # (dy, dx)
                        point_shift = point_offset(prm, tmpl.o[idx], self.group)
                    sc, c = cert_score(crop_w, tmpl.T[idx], self.deltas, self.offset)
                    sc0, c0 = cert_score(crops_at(Fp, pad, k, h), tmpl.T[idx], self.deltas, self.offset)
                    use0 = sc0 < sc          # the integer crop wins where the warp does not help
                    sc, c = np.where(use0, sc0, sc), np.where(use0, c0, c)
                    delta = np.where(use0[:, None], 0.0, d)
                    if point_shift is not None:
                        point_shift = np.where(use0[:, None], tmpl.o[idx], point_shift)
                    ok = sc < 1.0
                else:
                    crops = crops_at(Fp, pad, k, h)
                    sc, c = cert_score(crops, tmpl.T[idx], self.deltas, self.offset)
                    ok = sc < 1.0
                if self.use_refine and not self.interp and ok.any():
                    j = np.nonzero(ok)[0]
                    d, _ = gn_refine(Fp, pad, tmpl.T[idx[j]], tmpl.gy[idx[j]], tmpl.gx[idx[j]], tmpl.Hinv[idx[j]], k[j])
                    # a move of more than half a pixel changes the integer position the proof refers to
                    big = np.abs(d).max(1) > 0.5
                    if big.any():
                        k2 = self._clip(k[j[big]] + np.rint(d[big]).astype(int), F.shape, h)
                        sc2, c2 = cert_score(crops_at(Fp, pad, k2, h), tmpl.T[idx[j[big]]], self.deltas, self.offset)
                        acc = sc2 < 1.0
                        kb = k[j[big]]; kb[acc] = k2[acc]; k[j[big]] = kb
                        d[big] = np.where(acc[:, None], d[big] - np.rint(d[big]), np.clip(d[big], -0.5, 0.5))
                        sb = sc[j[big]]; sb[acc] = sc2[acc]; sc[j[big]] = sb
                        cb = c[j[big]]; cb[acc] = c2[acc]; c[j[big]] = cb
                    delta[j] = d
                if use_prop and oracle_xy is None:
                    # not certified: follow the model where it says visible, else carry the prediction
                    pv = proposals_vis[t, idx]
                    model_centre = proposals_xy[t, idx][:, ::-1] - tmpl.o[idx]
                    # the model's position is reported whether or not it calls the point visible
                    # (its guess beats a camera-carried one); the carried state below stays independent
                    fallback = model_centre
                    fallback_state = np.where(pv, states.MODEL_HELD, states.OCCLUDED)
                    # certified: the model's position stands when it agrees; a snap needs a unique template
                    ours_centre = k + delta if point_shift is None else k + point_shift - tmpl.o[idx]
                    dist = np.sqrt(((ours_centre - model_centre) ** 2).sum(1))
                    agree = ok & pv & (dist <= snap_tol)
                    unique = (tmpl.n_self[idx] == 0) if gate_self else np.ones(len(idx), bool)
                    snap = ok & ~agree & unique
                    distrust = ok & ~agree & ~unique
                    ok = agree | snap
                    delta = np.where(agree[:, None], model_centre - k, delta)
                    if point_shift is not None:
                        point_shift = np.where(agree[:, None], model_centre + tmpl.o[idx] - k, point_shift)
                    out_source[t, idx] = np.where(distrust, 3, out_source[t, idx])   # 3: certified elsewhere, not unique
                else:
                    fallback = p_hat
                    fallback_state = np.full(len(idx), states.LOST)
                new_pos = np.where(ok[:, None], k + delta, fallback)
                # residual velocity after the camera, learned on certified steps only
                moved = new_pos - (cam.predict(p_prev) if cam is not None else p_prev)
                vel[idx] = np.where(ok[:, None], (1 - self.vel_alpha) * vel[idx] + self.vel_alpha * moved, 0.0)
                # the carried state: certified position, else the camera-carried prediction (not the model's)
                pos[idx] = np.where(ok[:, None], k + delta, p_hat)
                out_state[t, idx] = np.where(ok, states.CERTIFIED, fallback_state)
                out_score[t, idx] = sc
                out_c[t, idx] = c
                out_k[t, idx] = k
                out_delta[t, idx] = delta
                out_source[t, idx] = np.where(ok, source, out_source[t, idx])
                if point_shift is not None:
                    pt = np.where(ok[:, None], k + point_shift, fallback + tmpl.o[idx])
                else:
                    pt = new_pos + tmpl.o[idx]
                out_xy[t, idx] = pt[:, ::-1]

            # ---- points whose query frame is this one: take their templates
            start = qt == t
            if start.any():
                new = make_templates(F, q[start, 1:], self.s, self.deltas, self.self_radius, self.offset, self.keep_maps,
                                     group=None if self.group == "translation" else self.group)
                tmpl = new if tmpl is None else _concat_templates(tmpl, new, start, N)
                if tmpl is new:
                    tmpl = _place_templates(new, start, N)
                i = np.nonzero(start)[0]
                pos[i] = tmpl.k[i].astype(float)
                vel[i] = 0.0
                active[i] = True
                out_state[t, i] = states.CERTIFIED
                out_score[t, i] = 0.0
                out_k[t, i] = tmpl.k[i]
                out_xy[t, i] = q[i, 1:]
            # frames before the query: the query position, not visible
            notyet = ~active
            if notyet.any():
                out_xy[t, notyet] = q[notyet, 1:]
            F_prev = F

        if tmpl is not None:
            meta = {"kappa": tmpl.kappa, "n_self": tmpl.n_self, "margin": tmpl.margin, "lam_min": tmpl.lam_min,
                    "border": tmpl.border}
            if self.keep_maps:
                meta["selfmap"] = tmpl.selfmap
        return {"tracks": out_xy, "state": out_state, "score": out_score, "c": out_c, "k": out_k, "delta": out_delta,
                "source": out_source, "visible": states.visible_from_state(out_state), "templates": meta,
                "config": {"delta0": self.delta0, "radii": list(self.radii), "s": self.s, "R": self.R,
                           "offset": self.offset, "camera": self.use_camera, "refine": self.use_refine,
                           "interp": self.interp, "group": self.group}}

    @staticmethod
    def _clip(k: np.ndarray, shape, h: int) -> np.ndarray:
        """Template centres stay inside the frame; the padding covers the window around them."""
        H, W = shape
        return np.stack([np.clip(k[:, 0], 0, H - 1), np.clip(k[:, 1], 0, W - 1)], -1)


def _place_templates(new: Templates, mask: np.ndarray, N: int) -> Templates:
    """Templates for a subset of points, expanded to all N with zeros elsewhere."""
    def exp(a):
        out = np.zeros((N,) + a.shape[1:], a.dtype)
        out[mask] = a
        return out
    return Templates(exp(new.T), exp(new.k), exp(new.o), exp(new.gy), exp(new.gx), exp(new.Hinv), exp(new.kappa),
                     exp(new.n_self), exp(new.margin), exp(new.lam_min), exp(new.border),
                     exp(new.selfmap) if new.selfmap is not None else None,
                     exp(new.J) if new.J is not None else None,
                     exp(new.HinvJ) if new.HinvJ is not None else None)


def _concat_templates(old: Templates, new: Templates, mask: np.ndarray, N: int) -> Templates:
    for name in ("T", "k", "o", "gy", "gx", "Hinv", "kappa", "n_self", "margin", "lam_min", "border", "selfmap",
                 "J", "HinvJ"):
        a = getattr(old, name)
        b = getattr(new, name)
        if a is None or b is None:
            continue
        a[mask] = b
    return old

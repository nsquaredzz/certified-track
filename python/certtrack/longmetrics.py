"""Metrics for long video and for the certificate.

All arrays are per clip without a batch axis: tracks (N, T, 2) as (x, y) in
evaluation pixels (256 x 256 unless said otherwise), occlusion flags (N, T) bool,
state codes (N, T) from ``certtrack.states``.  ``mask`` is the (N, T) boolean
array of (query, frame) pairs that count, normally ``queries.eval_mask``.

Definitions, so that RESULTS.md can point here:

survival        PointOdyssey's survival rate as CoTracker's evaluator computes it,
                restricted to the frames that count: a track fails at the first
                counted frame where the ground truth is visible and the error
                exceeds ``thresh`` pixels (50 at 256 x 256); survival is the share
                of counted frames before that failure.  Frames where the ground
                truth is occluded cannot fail.  Mean over tracks.
mte             median L2 error over counted frames where the ground truth is
                visible.
redetection     the counted frames where the ground truth is visible after having
                been occluded at some earlier counted frame.  AJ on that mask is
                our reading of TAPNext++'s AJ_RD; it is labelled "our reading"
                until checked against their code.
restricted      the TAP-Vid metrics on any mask, same formulas as
                ``metrics.compute_tapvid_metrics`` (checked equal on the standard
                mask in tests/test_metrics.py).
certified_fraction
                share of counted frames that carry a certificate, among those the
                method reports visible and among those the ground truth has visible.
proof_precision share of certified counted frames with visible ground truth whose
                error is within k pixels, per k: the precision of "certified in
                place" against the ground truth.
cycle_error     forward-backward L2 at the start frame, for footage without labels.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from . import states
from .metrics import THRESHOLDS


def _dist(gt_tracks, pred_tracks):
    return np.sqrt(np.sum(np.square(np.asarray(pred_tracks, np.float64) - np.asarray(gt_tracks, np.float64)), axis=-1))


def survival(gt_tracks, gt_occluded, pred_tracks, mask, thresh: float = 50.0) -> np.ndarray:
    """Per-track survival in [0, 1], shape (N,)."""
    mask = np.asarray(mask, bool)
    visible = ~np.asarray(gt_occluded, bool)
    fail = (_dist(gt_tracks, pred_tracks) > thresh) & visible & mask
    alive = np.cumprod(~fail, axis=1)
    n = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (alive & mask).sum(axis=1) / n
    return np.where(n > 0, s, np.nan)


def median_trajectory_error(gt_tracks, gt_occluded, pred_tracks, mask) -> float:
    sel = np.asarray(mask, bool) & ~np.asarray(gt_occluded, bool)
    d = _dist(gt_tracks, pred_tracks)[sel]
    return float(np.median(d)) if d.size else float("nan")


def redetection_mask(gt_occluded, mask) -> np.ndarray:
    """Counted frames with visible ground truth after an earlier counted occluded frame."""
    occ = np.asarray(gt_occluded, bool)
    mask = np.asarray(mask, bool)
    seen_occ = np.cumsum(occ & mask, axis=1) > 0
    # "earlier": shift so the frame itself does not count
    earlier = np.concatenate([np.zeros_like(seen_occ[:, :1]), seen_occ[:, :-1]], axis=1)
    return mask & ~occ & earlier


def restricted_metrics(gt_occluded, gt_tracks, pred_occluded, pred_tracks, mask,
                       thresholds: Iterable[int] = THRESHOLDS) -> dict:
    """TAP-Vid formulas on an arbitrary (N, T) mask; NaN where the mask is empty."""
    ev = np.asarray(mask, bool)
    gt_occ = np.asarray(gt_occluded, bool)
    pr_occ = np.asarray(pred_occluded, bool)
    visible, pred_visible = ~gt_occ, ~pr_occ
    out = {}
    n_ev = ev.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["occlusion_accuracy"] = float(((pr_occ == gt_occ) & ev).sum() / n_ev) if n_ev else float("nan")
        d2 = np.sum(np.square(np.asarray(pred_tracks, np.float64) - np.asarray(gt_tracks, np.float64)), axis=-1)
        within_all, jac_all = [], []
        for k in thresholds:
            within = d2 < k * k
            correct = within & visible
            n_vis = (visible & ev).sum()
            frac = (correct & ev).sum() / n_vis if n_vis else np.nan
            tp = (correct & pred_visible & ev).sum()
            fp = (((~visible) & pred_visible) | ((~within) & pred_visible)) & ev
            den = n_vis + fp.sum()
            jac = tp / den if den else np.nan
            out[f"pts_within_{k}"] = float(frac)
            out[f"jaccard_{k}"] = float(jac)
            within_all.append(frac)
            jac_all.append(jac)
    out["average_jaccard"] = float(np.mean(jac_all))
    out["average_pts_within_thresh"] = float(np.mean(within_all))
    return out


def certified_fraction(state, pred_occluded, gt_occluded, mask) -> dict:
    st = np.asarray(state)
    ev = np.asarray(mask, bool)
    cert = (st == states.CERTIFIED) & ev
    pv = ~np.asarray(pred_occluded, bool) & ev
    gv = ~np.asarray(gt_occluded, bool) & ev
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "of_pred_visible": float(cert.sum() / pv.sum()) if pv.sum() else float("nan"),
            "of_gt_visible": float((cert & gv).sum() / gv.sum()) if gv.sum() else float("nan"),
            "of_counted": float(cert.sum() / ev.sum()) if ev.sum() else float("nan"),
            "false_visible": float((cert & ~gv).sum() / cert.sum()) if cert.sum() else float("nan"),
        }


def proof_precision(state, gt_tracks, gt_occluded, pred_tracks, mask,
                    thresholds: Iterable[int] = THRESHOLDS) -> dict:
    """Of certified counted frames with visible ground truth: share within k px, and their count."""
    st = np.asarray(state)
    sel = (st == states.CERTIFIED) & np.asarray(mask, bool) & ~np.asarray(gt_occluded, bool)
    d = _dist(gt_tracks, pred_tracks)[sel]
    out = {"n": int(d.size)}
    for k in thresholds:
        out[f"within_{k}"] = float(np.mean(d < k)) if d.size else float("nan")
    return out


def model_call_fraction(query_windows: int, n_points: int, n_windows: int, wall_seconds: float | None = None) -> dict:
    """Model cost in query-windows against the baseline that queries every point every window."""
    base = n_points * n_windows
    out = {"query_windows": int(query_windows), "baseline_query_windows": int(base),
           "fraction": float(query_windows / base) if base else float("nan")}
    if wall_seconds is not None:
        out["wall_seconds"] = float(wall_seconds)
    return out


def cycle_error(fwd_tracks_at_start, bwd_tracks_at_start) -> np.ndarray:
    """Forward-backward L2 per point at the start frame, shape (N,)."""
    return _dist(fwd_tracks_at_start, bwd_tracks_at_start)


def long_summary(gt_tracks, gt_occluded, pred_tracks, pred_occluded, state, mask) -> dict:
    """Everything above for one clip, in one dict."""
    s = survival(gt_tracks, gt_occluded, pred_tracks, mask)
    rd = redetection_mask(gt_occluded, mask)
    out = {
        "survival": float(np.nanmean(s)),
        "survival_16": float(np.nanmean(survival(gt_tracks, gt_occluded, pred_tracks, mask, thresh=16.0))),
        "mte": median_trajectory_error(gt_tracks, gt_occluded, pred_tracks, mask),
        "redetection_frames": int(rd.sum()),
    }
    if rd.any():
        out["aj_rd"] = restricted_metrics(gt_occluded, gt_tracks, pred_occluded, pred_tracks, rd)["average_jaccard"]
    else:
        out["aj_rd"] = float("nan")
    out["certified"] = certified_fraction(state, pred_occluded, gt_occluded, mask)
    out["proof_precision"] = proof_precision(state, gt_tracks, gt_occluded, pred_tracks, mask)
    return out

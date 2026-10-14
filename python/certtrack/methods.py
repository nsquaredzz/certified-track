"""The methods benchmarked side by side, behind one call: track_clip(frames_rgb, queries_txy).

Each returns (tracks (T, N, 2), visible (T, N), CallLog, state (T, N)).
"""
from __future__ import annotations

import numpy as np

from . import states
from .data.tapvid import grey
from .models.base import CallLog
from .track import CertifiedTracker


class CarryOnly:
    """The certified tracker alone: no learned model, LOST where nothing certifies."""

    def __init__(self, **tracker_kwargs):
        self.kw = tracker_kwargs
        self.name = "carry"
        self.last = None

    def track_clip(self, frames, queries_txy):
        g = grey(frames) if isinstance(frames, np.ndarray) and frames.ndim == 4 else frames
        r = CertifiedTracker(**self.kw).run(g, queries_txy)
        self.last = r
        return r["tracks"], r["visible"], CallLog(), r["state"]


class HybridVerify:
    """The frozen model on every query every frame; the certificate verifies, snaps and re-detects.

    No model compute is saved.  The output is the certified position wherever a match certifies
    (near the model's proposal or at the carried prediction), else the model's.
    """

    def __init__(self, model, proposal_radius: int = 4, **tracker_kwargs):
        self.model = model
        self.kw = tracker_kwargs
        self.proposal_radius = proposal_radius
        self.name = f"hybrid-verify({model.name})"
        self.last = None

    def track_clip(self, frames, queries_txy):
        tr_m, vi_m, log = self.model.track_clip(frames, queries_txy)
        g = grey(frames)
        r = CertifiedTracker(**self.kw).run(g, queries_txy, proposals_xy=tr_m, proposals_vis=vi_m,
                                            proposal_radius=self.proposal_radius)
        r["model_tracks"], r["model_visible"] = tr_m, vi_m
        self.last = r
        return r["tracks"], r["visible"], log, r["state"]


class FrozenOnly:
    """A frozen tracker alone; its visible flag is reported as MODEL_HELD, the rest OCCLUDED."""

    def __init__(self, model):
        self.model = model
        self.name = model.name

    def track_clip(self, frames, queries_txy):
        tr, vi, log = self.model.track_clip(frames, queries_txy)
        qt = np.rint(queries_txy[:, 0]).astype(int)
        st = np.where(vi, states.MODEL_HELD, states.OCCLUDED).astype(np.uint8)
        before = np.arange(tr.shape[0])[:, None] < qt[None, :]
        st[before] = states.NONE
        return tr, vi, log, st

"""A stand-in tracker for tests: the ground truth plus noise, with a configurable visibility error rate."""
from __future__ import annotations

import numpy as np

from .base import CallLog


class FakeTracker:
    name = "fake"

    def __init__(self, gt_tracks: np.ndarray, gt_occluded: np.ndarray, sigma: float = 1.0,
                 flip_vis: float = 0.0, seed: int = 0, window: int = 8):
        """gt_tracks (N, T, 2), gt_occluded (N, T) in the frame's pixels; noise N(0, sigma) per axis."""
        self.gt = np.asarray(gt_tracks, np.float32)
        self.occ = np.asarray(gt_occluded, bool)
        self.sigma, self.flip, self.window = sigma, flip_vis, window
        self.rng = np.random.default_rng(seed)

    def track_clip(self, frames, queries_txy):
        N, T = self.gt.shape[:2]
        assert queries_txy.shape[0] == N, "the fake tracker expects one query per ground-truth track"
        tracks = self.gt.transpose(1, 0, 2) + self.rng.normal(0, self.sigma, (T, N, 2)).astype(np.float32)
        visible = ~self.occ.T.copy()
        if self.flip > 0:
            visible ^= self.rng.random((T, N)) < self.flip
        qt = queries_txy[:, 0].astype(int)
        before = np.arange(T)[:, None] < qt[None, :]
        tracks[before] = np.broadcast_to(queries_txy[None, :, 1:], (T, N, 2))[before]
        visible[before] = False
        log = CallLog()
        for w in range(0, max(T - self.window, 1), self.window):
            log.add(N, min(2 * self.window, T - w), 0.0)
        return tracks, visible, log

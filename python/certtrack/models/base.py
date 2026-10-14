"""The interface every frozen tracker exposes, and the call log that prices it."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class CallLog:
    """Model cost in the unit the control baseline shares: query-windows.

    One window call on n queries costs n query-windows; the baseline that queries every
    point in every window costs N per window.  Wall-clock is kept beside it, never alone.
    """
    windows: int = 0
    query_windows: int = 0
    frames_in: int = 0
    wall: float = 0.0
    per_call: list = field(default_factory=list)

    def add(self, n_queries: int, n_frames: int, dt: float) -> None:
        self.windows += 1
        self.query_windows += int(n_queries)
        self.frames_in += int(n_frames)
        self.wall += float(dt)
        self.per_call.append((int(n_queries), int(n_frames), float(dt)))

    def summary(self) -> dict:
        return {"windows": self.windows, "query_windows": self.query_windows,
                "frames_in": self.frames_in, "wall_seconds": round(self.wall, 3)}


class FrozenTracker(Protocol):
    """Tracks a whole clip from queries given up front: the baseline interface."""
    name: str

    def track_clip(self, frames: np.ndarray, queries_txy: np.ndarray) -> tuple[np.ndarray, np.ndarray, CallLog]:
        """frames (T, H, W, 3) uint8, queries (N, 3) as (t, x, y) in frame pixels.

        Returns tracks (T, N, 2) in frame pixels, visible (T, N) bool, and the call log.
        Frames before a query are filled with the query position and marked not visible.
        """
        ...

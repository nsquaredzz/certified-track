"""Long sequences with exact ground truth, built from a labelled clip by an index map.

A palindrome plays the clip forward, then backward, then forward again, so every
frame shown is a real frame and its labels are exact.  What it tests: drift over
many frames of real appearance, and repeated occlusion and re-appearance.  What it
does not test: new content, appearance change over hours, physically plausible
reversed motion, direction-consistent motion blur.  Two variants:

  palindrome        0 .. T-1, T-2 .. 1, 0 .. T-1, ...   the query frame recurs every
                    2(T-1) frames, which favours anything anchored on it
  palindrome_tail   first pass 0 .. T-1, then bounce inside [k, T-1]: frames before k
                    never recur, so a template taken at frame 0 never sees its own
                    frame again.  This is the variant to headline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def palindrome_index(n_frames: int, length: int, tail_from: int = 0) -> np.ndarray:
    """Frame index shown at each time step, shape (length,); consecutive steps differ by 1."""
    T = int(n_frames)
    if T < 2:
        raise ValueError("need at least two frames")
    k = int(tail_from)
    if not 0 <= k <= T - 2:
        raise ValueError("tail_from must be in [0, T-2]")
    idx = list(range(T))
    direction = -1
    cur = T - 1
    while len(idx) < length:
        nxt = cur + direction
        if nxt > T - 1 or nxt < k:
            direction = -direction
            nxt = cur + direction
        idx.append(nxt)
        cur = nxt
    return np.asarray(idx[:length], dtype=np.int64)


@dataclass
class LongSequence:
    """A clip replayed by an index map; frames are looked up, never copied."""
    name: str
    frames: np.ndarray        # (T, H, W, 3) uint8 of the base clip
    base_tracks: np.ndarray   # (N, T, 2)
    base_occluded: np.ndarray  # (N, T)
    index: np.ndarray         # (L,)
    kind: str

    def __len__(self):
        return int(self.index.shape[0])

    def frame(self, t: int) -> np.ndarray:
        return self.frames[self.index[t]]

    @property
    def tracks(self) -> np.ndarray:
        return self.base_tracks[:, self.index]

    @property
    def occluded(self) -> np.ndarray:
        return self.base_occluded[:, self.index]


def build_long(name: str, frames: np.ndarray, tracks: np.ndarray, occluded: np.ndarray,
               length: int = 1000, tail_from: int | None = None) -> LongSequence:
    T = frames.shape[0]
    if tail_from is None:
        return LongSequence(name, frames, tracks, occluded, palindrome_index(T, length, 0), "palindrome")
    return LongSequence(name, frames, tracks, occluded, palindrome_index(T, length, tail_from), f"palindrome_tail{tail_from}")

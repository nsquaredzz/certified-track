"""TAP-Vid pickles (DAVIS, RGB-Stacking, RoboTAP): one-time split into per-clip files, then streaming.

The pickle is a dict (DAVIS, RoboTAP) or a list (RGB-Stacking) of records with keys
``video`` (T, H, W, 3) uint8 or a list of JPEG bytes, ``points`` (N, T, 2) float32 as
(x, y) normalised to [0, 1], and ``occluded`` (N, T) bool.  ``prepare_tapvid`` decodes,
resizes once to the evaluation resolution and writes ``clips/<name>.npz`` plus
``index.json`` so a benchmark never holds the 2.5 GB pickle in memory.

Two conventions for turning normalised points into pixels exist in the wild and they
differ by one part in 256:

  tapnet     x_pix = x * W           (tapnet/tapvid/evaluation_datasets.py)
  cotracker  x_pix = x * (W - 1)     (cotracker/datasets/tap_vid_datasets.py, "1 should be mapped to W-1")

CoTracker3's published numbers use the second; ``Clip.tracks`` follows ``convention``
and the normalised points are kept so either can be produced.  Resizing uses PIL's
Lanczos filter; tapnet uses mediapy's (also Lanczos-3), not bit-identical.
"""
from __future__ import annotations

import io
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

CONVENTIONS = ("cotracker", "tapnet")


@dataclass
class Clip:
    name: str
    frames: np.ndarray        # (T, H, W, 3) uint8 at the working resolution
    tracks: np.ndarray        # (N, T, 2) float32 (x, y) pixels at the working resolution
    occluded: np.ndarray      # (N, T) bool
    norm_points: np.ndarray   # (N, T, 2) float32 (x, y) in [0, 1]
    native_hw: tuple[int, int]
    convention: str = "cotracker"

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def hw(self) -> tuple[int, int]:
        return int(self.frames.shape[1]), int(self.frames.shape[2])

    def eval_scale(self, eval_hw=(256, 256)) -> np.ndarray:
        """Multiply (x, y) in working pixels by this to land in the evaluation frame."""
        H, W = self.hw
        eh, ew = eval_hw
        if self.convention == "cotracker":
            return np.array([(ew - 1) / (W - 1), (eh - 1) / (H - 1)], np.float64)
        return np.array([ew / W, eh / H], np.float64)


def points_to_pixels(norm: np.ndarray, hw: tuple[int, int], convention: str = "cotracker") -> np.ndarray:
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}")
    H, W = hw
    s = np.array([W - 1, H - 1], np.float64) if convention == "cotracker" else np.array([W, H], np.float64)
    return (np.asarray(norm, np.float64) * s).astype(np.float32)


def decode_frames(video) -> np.ndarray:
    """Frames as (T, H, W, 3) uint8; TAP-Vid Kinetics stores JPEG bytes."""
    if isinstance(video, np.ndarray) and video.dtype == np.uint8 and video.ndim == 4:
        return video
    out = []
    for f in video:
        if isinstance(f, (bytes, bytearray)):
            out.append(np.asarray(Image.open(io.BytesIO(f)).convert("RGB")))
        else:
            out.append(np.asarray(f, np.uint8))
    return np.stack(out)


def resize_frames(frames: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Lanczos resize of every frame to (H, W)."""
    H, W = size
    if frames.shape[1:3] == (H, W):
        return frames
    out = np.empty((frames.shape[0], H, W, 3), np.uint8)
    for i, f in enumerate(frames):
        out[i] = np.asarray(Image.fromarray(f).resize((W, H), Image.LANCZOS))
    return out


def grey(frames: np.ndarray) -> np.ndarray:
    """(T, H, W) uint8 luma (ITU-R 601 via PIL, as certified-skip's scripts do)."""
    return np.stack([np.asarray(Image.fromarray(f).convert("L")) for f in frames])


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _records(data) -> Iterator[tuple[str, dict]]:
    if isinstance(data, dict):
        for k in data:
            yield str(k), data[k]
    else:
        for i, rec in enumerate(data):
            yield f"{i:04d}", rec


def prepare_tapvid(pickle_path: Path, out_dir: Path, resolution: tuple[int, int] | None = (256, 256)) -> dict:
    """Split a TAP-Vid pickle into per-clip npz files at ``resolution`` (None keeps native)."""
    pickle_path, out_dir = Path(pickle_path), Path(out_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    data = load_pickle(pickle_path)
    index = {"source": str(pickle_path), "resolution": list(resolution) if resolution else None, "clips": {}}
    for name, rec in _records(data):
        for key in ("video", "points", "occluded"):
            if key not in rec:
                raise KeyError(f"{pickle_path}: record {name} has no {key!r}; keys are {list(rec)}")
        frames = decode_frames(rec["video"])
        native_hw = (int(frames.shape[1]), int(frames.shape[2]))
        if resolution is not None:
            frames = resize_frames(frames, resolution)
        pts = np.asarray(rec["points"], np.float32)
        occ = np.asarray(rec["occluded"], bool)
        if pts.shape[:2] != occ.shape or pts.shape[1] != frames.shape[0]:
            raise ValueError(f"{name}: points {pts.shape}, occluded {occ.shape}, frames {frames.shape}")
        if pts.min() < -1e-3 or pts.max() > 1 + 1e-3:
            raise ValueError(f"{name}: points are not normalised to [0, 1] (min {pts.min()}, max {pts.max()})")
        np.savez(clips_dir / f"{name}.npz", frames=frames, points=pts, occluded=occ, native_hw=np.array(native_hw))
        index["clips"][name] = {"frames": int(frames.shape[0]), "tracks": int(pts.shape[0]),
                                "native_hw": list(native_hw), "visible_tracks": int((~occ).any(axis=1).sum())}
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return index


def clip_names(root: Path) -> list[str]:
    return list(json.loads((Path(root) / "index.json").read_text())["clips"])


def load_clip(root: Path, name: str, convention: str = "cotracker") -> Clip:
    z = np.load(Path(root) / "clips" / f"{name}.npz")
    frames = z["frames"]
    hw = (int(frames.shape[1]), int(frames.shape[2]))
    return Clip(name, frames, points_to_pixels(z["points"], hw, convention), z["occluded"].astype(bool),
                z["points"].astype(np.float32), tuple(int(v) for v in z["native_hw"]), convention)


def iter_clips(root: Path, names: list[str] | None = None, convention: str = "cotracker") -> Iterator[Clip]:
    for name in (names or clip_names(root)):
        yield load_clip(root, name, convention)

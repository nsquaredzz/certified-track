"""DAVIS-2017 JPEG folders (and anything laid out the same way, such as data/film_shot).

Layout: <root>/JPEGImages/480p/<clip>/00000.jpg ... and
<root>/Annotations/480p/<clip>/00000.png (palette PNG, pixel value = object id),
as certified-skip's scripts/compare_sam2.py reads them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def clip_names(root: Path, split: str | None = None) -> list[str]:
    root = Path(root)
    if split:
        return [l.strip() for l in (root / "ImageSets" / "2017" / f"{split}.txt").read_text().split() if l.strip()]
    return sorted(p.name for p in (root / "JPEGImages" / "480p").iterdir() if p.is_dir())


def load_davis(root: Path, name: str, with_masks: bool = True, max_frames: int | None = None):
    """(frames (T, H, W, 3) uint8, masks (T, H, W) uint8 or None); masks only for annotated frames, else 0."""
    root = Path(root)
    jpgs = sorted((root / "JPEGImages" / "480p" / name).glob("*.jpg"))
    if max_frames:
        jpgs = jpgs[:max_frames]
    if not jpgs:
        raise FileNotFoundError(f"no frames for {name} under {root}")
    frames = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in jpgs])
    masks = None
    if with_masks:
        ann = root / "Annotations" / "480p" / name
        masks = np.zeros(frames.shape[:3], np.uint8)
        for i, p in enumerate(jpgs):
            q = ann / (p.stem + ".png")
            if q.exists():
                masks[i] = np.asarray(Image.open(q))
    return frames, masks

"""Raw footage through ffmpeg: RGB frames for the renderer, grey frames via certskip.video."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np
from certskip.video import iter_gray, probe, read_gray  # noqa: F401  (re-exported)


def _vf(fps, scale):
    parts = []
    if fps:
        parts.append(f"fps={fps}")
    if scale:
        parts.append(f"scale={scale[0]}:{scale[1]}:flags=area")
    return ",".join(parts) or "null"


def iter_rgb(path, fps: float | None = None, scale=None, start: float | None = None,
             max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield (H, W, 3) uint8 RGB frames. scale=(W, H) resizes; fps resamples; start seeks (seconds)."""
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH")
    info = probe(path)
    W, H = (scale if scale else (info["width"], info["height"]))
    cmd = [exe, "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path), "-vf", _vf(fps, scale), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=W * H * 24)
    n = 0
    try:
        while True:
            buf = proc.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            n += 1
            if max_frames and n >= max_frames:
                break
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def read_rgb(path, **kw) -> np.ndarray:
    return np.stack(list(iter_rgb(path, **kw)))


def catalogue(skip_data: Path) -> dict[str, dict]:
    """The long real footage already fetched by certified-skip, by name."""
    d = Path(skip_data)
    return {
        "tos_201": {"path": d / "tears_of_steel_720p.mov", "start": 201.0, "seconds": 15.0, "fps": 24},
        "tos_356": {"path": d / "tears_of_steel_720p.mov", "start": 356.4, "seconds": 15.0, "fps": 24},
        "hall": {"path": d / "hall_monitor_cif.y4m"},
        "caviar_leftbag": {"path": d / "caviar_LeftBag.mpg"},
        "virat_0200": {"path": d / "VIRAT_S_000200_00.mp4", "seconds": 60.0},
    }

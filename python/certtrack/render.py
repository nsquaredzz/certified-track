"""Proof videos: camera | tracks coloured by state | error against the ground truth over time.

Colours and the ffmpeg writer follow certified-skip's scripts (render_before_after.py,
make_readme_media.py).  Everything is drawn with PIL; no matplotlib.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import states

GREEN, AMBER, GREY, RED, WHITE, INK = (74, 222, 128), (251, 191, 36), (150, 160, 172), (248, 113, 113), (240, 240, 240), (18, 18, 18)
STATE_COLOUR = {states.CERTIFIED: GREEN, states.MODEL_HELD: AMBER, states.OCCLUDED: GREY, states.LOST: RED, states.NONE: (90, 90, 90)}
BAR = 30


def font(px: int, bold: bool = False):
    names = (["Arial Bold.ttf", "Helvetica.ttc", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold
             else ["Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
    dirs = ["/System/Library/Fonts/Supplemental", "/System/Library/Fonts", "/Library/Fonts",
            "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation", "/usr/share/fonts"]
    for n in names:
        for d in dirs:
            p = Path(d) / n
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), int(px))
                except Exception:
                    pass
    return ImageFont.load_default()


class Writer:
    def __init__(self, path, size, fps=12, crf=18):
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found on PATH")
        w, h = size
        w, h = w - w % 2, h - h % 2
        self.size = (w, h)
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                      "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                                      "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        self.n = 0

    def add(self, frame: np.ndarray):
        w, h = self.size
        self.proc.stdin.write(np.ascontiguousarray(frame[:h, :w]).tobytes())
        self.n += 1

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def panel_tracks(frame_rgb: np.ndarray, xy: np.ndarray, state: np.ndarray, trails: list[np.ndarray] | None = None,
                 gt_xy: np.ndarray | None = None, gt_vis: np.ndarray | None = None, up: int = 2) -> np.ndarray:
    H, W = frame_rgb.shape[:2]
    im = Image.fromarray(frame_rgb).resize((W * up, H * up), Image.NEAREST)
    d = ImageDraw.Draw(im)
    if trails:
        for i in range(len(xy)):
            pts = [(float(p[i, 0]) * up, float(p[i, 1]) * up) for p in trails if np.all(np.isfinite(p[i]))]
            if len(pts) > 1:
                d.line(pts, fill=STATE_COLOUR.get(int(state[i]), GREY), width=1)
    if gt_xy is not None:
        for i in range(len(gt_xy)):
            if gt_vis is None or gt_vis[i]:
                x, y = float(gt_xy[i, 0]) * up, float(gt_xy[i, 1]) * up
                d.ellipse((x - 5, y - 5, x + 5, y + 5), outline=WHITE, width=1)
    for i in range(len(xy)):
        s = int(state[i])
        if s == states.NONE:
            continue
        x, y = float(xy[i, 0]) * up, float(xy[i, 1]) * up
        r = 3 if s in (states.CERTIFIED, states.MODEL_HELD) else 2
        d.ellipse((x - r, y - r, x + r, y + r), fill=STATE_COLOUR[s])
    return np.asarray(im)


def panel_chart(err_hist: np.ndarray, t: int, size: tuple[int, int], thresh: float = 50.0, ymax: float = 60.0,
                ticks: np.ndarray | None = None, title: str = "error vs ground truth, px at 256") -> np.ndarray:
    """err_hist (T, N) with NaN where not counted; drawn up to frame t."""
    w, h = size
    im = Image.new("RGB", (w, h), INK)
    d = ImageDraw.Draw(im)
    f = font(12)
    L, R, Tp, B = 36, 8, 18, 18
    T = err_hist.shape[0]
    def X(i): return L + (w - L - R) * i / max(T - 1, 1)
    def Y(v): return Tp + (h - Tp - B) * (1 - min(v, ymax) / ymax)
    d.text((L, 2), title, fill=WHITE, font=f)
    for v in (0, 16, thresh):
        d.line((L, Y(v), w - R, Y(v)), fill=(60, 60, 60), width=1)
        d.text((2, Y(v) - 7), f"{v:.0f}", fill=GREY, font=f)
    if ticks is not None:
        for i in np.nonzero(ticks[: t + 1])[0]:
            d.line((X(i), h - B, X(i), h - B + 4), fill=AMBER, width=1)
    e = err_hist[: t + 1]
    for n in range(e.shape[1]):
        pts = [(X(i), Y(v)) for i, v in enumerate(e[:, n]) if np.isfinite(v)]
        if len(pts) > 1:
            d.line(pts, fill=(70, 90, 110), width=1)
    med = np.nanmedian(e, axis=1) if e.size else np.array([])
    pts = [(X(i), Y(v)) for i, v in enumerate(med) if np.isfinite(v)]
    if len(pts) > 1:
        d.line(pts, fill=GREEN, width=2)
    d.line((X(t), Tp, X(t), h - B), fill=WHITE, width=1)
    return np.asarray(im)


def caption(width: int, text: str, legend=((GREEN, "certified"), (AMBER, "model"), (GREY, "occluded"), (RED, "lost"))) -> np.ndarray:
    im = Image.new("RGB", (width, BAR), INK)
    d = ImageDraw.Draw(im)
    f = font(13)
    x = 10
    for colour, label in legend:
        d.ellipse((x, 10, x + 10, 20), fill=colour)
        d.text((x + 16, 7), label, fill=(225, 225, 225), font=f)
        x += 30 + int(d.textlength(label, font=f))
    d.text((width - 10 - d.textlength(text, font=f), 7), text, fill=WHITE, font=f)
    return np.asarray(im)


def compose(panels: list[np.ndarray], footer: str) -> np.ndarray:
    h = max(p.shape[0] for p in panels)
    padded = [np.pad(p, ((0, h - p.shape[0]), (0, 4), (0, 0)), constant_values=18) for p in panels]
    row = np.concatenate(padded, 1)
    return np.concatenate([row, caption(row.shape[1], footer)], 0)


def render_proof(frames_rgb, tracks: np.ndarray, state: np.ndarray, out_mp4: Path, gt_xy: np.ndarray | None = None,
                 gt_occ: np.ndarray | None = None, eval_scale: np.ndarray | None = None, fps: float = 12, up: int = 2,
                 trail: int = 12, title: str = "", calls: np.ndarray | None = None) -> Path:
    """frames_rgb: (T, H, W, 3) array or object with frame(t); tracks (T, N, 2), state (T, N); gt (N, T, 2), occ (N, T)."""
    get = (lambda t: frames_rgb[t]) if isinstance(frames_rgb, np.ndarray) else frames_rgb.frame
    T, N = state.shape
    sc = np.ones(2) if eval_scale is None else np.asarray(eval_scale)
    err = np.full((T, N), np.nan)
    if gt_xy is not None:
        e = np.sqrt((((tracks - gt_xy.transpose(1, 0, 2)) * sc) ** 2).sum(-1))
        counted = (state != states.NONE) & (~gt_occ.T if gt_occ is not None else True)
        err[counted] = e[counted]
    f0 = get(0)
    H, W = f0.shape[:2]
    chart_size = (max(W * up, 320), H * up)
    writer = None
    for t in range(T):
        fr = get(t)
        trails = [tracks[u] for u in range(max(0, t - trail), t + 1)]
        p1 = panel_tracks(fr, tracks[t], state[t], trails, None if gt_xy is None else gt_xy[:, t],
                          None if gt_occ is None else ~gt_occ[:, t], up)
        cert = float(np.mean(state[t] == states.CERTIFIED)) if N else 0.0
        panels = [np.asarray(Image.fromarray(fr).resize((W * up, H * up), Image.NEAREST)), p1]
        if gt_xy is not None:
            panels.append(panel_chart(err, t, chart_size, ticks=calls))
        text = f"{title}   frame {t + 1} of {T}   certified {100 * cert:.0f} %"
        if gt_xy is not None and np.isfinite(err[t]).any():
            text += f"   median error {np.nanmedian(err[t]):.1f} px"
        frame = compose(panels, text)
        if writer is None:
            writer = Writer(out_mp4, (frame.shape[1], frame.shape[0]), fps)
        writer.add(frame)
    writer.close()
    return Path(out_mp4)


def to_gif(mp4: Path, gif: Path, width: int = 880, fps: float | None = None) -> Path:
    vf = f"scale={width}:-1:flags=lanczos" + (f",fps={fps}" if fps else "")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf", vf + ",split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
                    str(gif)], check=True)
    return Path(gif)

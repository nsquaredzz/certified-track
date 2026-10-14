#!/usr/bin/env python3
"""Render a proof video from a cached run.

    python scripts/render_proof.py data/runs/<stem>/<clip>.npz --data davis --clip <clip> --out out/<name>.mp4
    python scripts/render_proof.py ... --gif assets/<name>.gif

The npz comes from scripts/bench_long.py (or any script that saves tracks, state, index, gt, occ).
Panels: camera | tracks coloured by state (certified green, model amber, occluded grey, lost red;
ground truth as white rings) | error against the ground truth over time.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from certtrack.data.tapvid import load_clip  # noqa: E402
from certtrack.render import render_proof, to_gif  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", default="davis")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--title", default="")
    ap.add_argument("--every", type=int, default=1, help="render every k-th frame")
    a = ap.parse_args()
    z = np.load(a.run)
    base = load_clip(ROOT / "data" / "tapvid" / a.data, a.clip)
    index = z["index"] if "index" in z else np.arange(z["tracks"].shape[0])
    sel = np.arange(0, len(index), a.every)
    frames = base.frames[index[sel]]
    tracks, state = z["tracks"][sel], z["state"][sel]
    gt = z["gt"][:, sel] if "gt" in z else None
    occ = z["occ"][:, sel] if "occ" in z else None
    out = render_proof(frames, tracks, state, Path(a.out), gt, occ, base.eval_scale(), fps=a.fps,
                       title=a.title or f"{a.clip}")
    print(f"wrote {out}")
    if a.gif:
        print(f"wrote {to_gif(out, Path(a.gif))}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Split the TAP-Vid pickles into per-clip npz files at the evaluation resolution.

    python scripts/prepare_tapvid.py                       # DAVIS and RGB-Stacking at 256x256
    python scripts/prepare_tapvid.py --native              # also keep DAVIS at native resolution
    python scripts/prepare_tapvid.py --only davis

Reads data/tapvid/<set>/<set>.pkl (from scripts/fetch_data.py), writes
data/tapvid/<set>/{clips/*.npz,index.json} and, with --native, data/tapvid/<set>_native/.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from certtrack.data.tapvid import prepare_tapvid  # noqa: E402

SETS = {"davis": "tapvid_davis", "rgb_stacking": "tapvid_rgb_stacking"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(SETS), action="append")
    ap.add_argument("--native", action="store_true", help="also write a native-resolution copy")
    a = ap.parse_args()
    for key in (a.only or sorted(SETS)):
        pkl = ROOT / "data" / "tapvid" / SETS[key] / f"{SETS[key]}.pkl"
        if not pkl.exists():
            print(f"missing {pkl}; run scripts/fetch_data.py first", file=sys.stderr)
            continue
        for res, suffix in (((256, 256), ""), (None, "_native")):
            if res is None and not a.native:
                continue
            out = ROOT / "data" / "tapvid" / f"{key}{suffix}"
            t0 = time.time()
            idx = prepare_tapvid(pkl, out, res)
            n = len(idx["clips"])
            frames = sum(c["frames"] for c in idx["clips"].values())
            print(f"{key}{suffix}: {n} clips, {frames} frames -> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

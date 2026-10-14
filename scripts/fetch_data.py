#!/usr/bin/env python3
"""Download the evaluation data into data/ and verify it.

    python scripts/fetch_data.py                 # TAP-Vid DAVIS (1.67 GB) and RGB-Stacking (187 MB)
    python scripts/fetch_data.py robotap         # RoboTAP, 13.6 GB (asks first)
    python scripts/fetch_data.py --verify        # re-hash what is on disk
    python scripts/fetch_data.py --link-from ../certified-skip/data   # symlink the footage fetched there

Nothing here is redistributed by this repository; every file comes from its owner's server
and stays under its owner's terms:

  TAP-Vid   Doersch et al., NeurIPS 2022 Datasets and Benchmarks; files hosted by Google DeepMind
            at storage.googleapis.com/dm-tapnet (Apache 2.0 code, data under the DAVIS and
            RGB-Stacking licences)
  RoboTAP   Vecerik et al., ICRA 2024; same host

Standard library only.  Files are written to a .part name first and renamed once size and
SHA-256 match.  Digests were recorded on the first download (2026-10-14) and are verified
ever after.
"""
import argparse
import hashlib
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HOST = "https://storage.googleapis.com/dm-tapnet/"

# relative path under data/, url, bytes, sha256, group
FILES = [
    ("tapvid/tapvid_davis.zip", HOST + "tapvid_davis.zip", 1668710491,
     "60eb6239c57a9877900c269d14637c2b6eece8fa280f59e4592324c6392031b6", "tapvid"),
    ("tapvid/tapvid_rgb_stacking.zip", HOST + "tapvid_rgb_stacking.zip", 187291581,
     "ba0cec640deb65ab510bc3920cdaf8b7a2e756fce9fb339199c557ade436b0cc", "tapvid"),
    ("robotap/robotap.zip", HOST + "robotap/robotap.zip", 13558087507, None, "robotap"),
]
LINKS = ["tears_of_steel_720p.mov", "hall_monitor_cif.y4m", "caviar_LeftBag.mpg", "VIRAT_S_000200_00.mp4",
         "davis", "film_shot"]
UA = {"User-Agent": "certified-track-fetch/1.0"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def good(path, size, digest):
    return path.exists() and path.stat().st_size == size and (digest is None or sha256(path) == digest)


def fetch(rel, url, size, digest):
    path = DATA / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if good(path, size, digest):
        print(f"ok       {rel}")
        return True
    part = path.with_suffix(path.suffix + ".part")
    print(f"fetching {rel} ({size / 1e9:.2f} GB) from {url}")
    t0 = time.time()
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        done = 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done % (1 << 28) < (1 << 22):
                print(f"  {done / 1e9:.2f} GB", flush=True)
    d = sha256(part)
    if part.stat().st_size != size:
        print(f"size mismatch for {rel}: {part.stat().st_size} != {size}", file=sys.stderr)
        return False
    if digest is None:
        print(f"first download of {rel}: sha256 {d}\n  paste it into FILES before marking the file good")
        part.rename(path)
        return False
    if d != digest:
        print(f"sha256 mismatch for {rel}", file=sys.stderr)
        return False
    part.rename(path)
    print(f"done     {rel} in {time.time() - t0:.0f} s")
    return True


def unzip(rel):
    path = DATA / rel
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if all((path.parent / n).exists() for n in names):
            return
        z.extractall(path.parent)
    print(f"unzipped {rel}")


def link_from(src):
    src = Path(src).resolve()
    for name in LINKS:
        s, d = src / name, DATA / "footage" / name
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.exists() and not d.exists():
            d.symlink_to(s)
            print(f"linked   {d} -> {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="*", default=["tapvid"], choices=["tapvid", "robotap"])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--link-from")
    ap.add_argument("--yes", action="store_true", help="do not ask before the 13.6 GB RoboTAP download")
    a = ap.parse_args()
    if a.link_from:
        link_from(a.link_from)
    ok = True
    for rel, url, size, digest, group in FILES:
        if group not in a.groups:
            continue
        if a.verify:
            print(("ok       " if good(DATA / rel, size, digest) else "BAD      ") + rel)
            continue
        if group == "robotap" and not a.yes:
            if input("RoboTAP is 13.6 GB. Download? [y/N] ").strip().lower() != "y":
                continue
        if fetch(rel, url, size, digest):
            unzip(rel)
        else:
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

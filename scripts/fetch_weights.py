#!/usr/bin/env python3
"""Fetch the frozen tracker: CoTracker3 weights and a pinned checkout of its source.

    python scripts/fetch_weights.py
    python scripts/fetch_weights.py --verify

Weights: facebook/cotracker3 on Hugging Face (CC BY-NC 4.0, Meta), into data/weights/cotracker3/.
Source: github.com/facebookresearch/co-tracker at the commit below, into data/third_party/co-tracker.
Nothing goes through torch.hub; nothing lands in ~/.cache.  Digests recorded 2026-10-14.
"""
import argparse
import hashlib
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HF = "https://huggingface.co/facebook/cotracker3/resolve/main/"
WEIGHTS = [
    ("weights/cotracker3/scaled_online.pth", HF + "scaled_online.pth", 101695610,
     "205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218"),
    ("weights/cotracker3/scaled_offline.pth", HF + "scaled_offline.pth", 101890938,
     "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"),
]
REPO_URL = "https://github.com/facebookresearch/co-tracker"
REPO_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"  # 2025-01-21
REPO_DIR = DATA / "third_party" / "co-tracker"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def good(path, size, digest):
    return path.exists() and path.stat().st_size == size and sha256(path) == digest


def fetch(rel, url, size, digest):
    path = DATA / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if good(path, size, digest):
        print(f"ok       {rel}")
        return True
    part = path.with_suffix(path.suffix + ".part")
    print(f"fetching {rel} ({size / 1e6:.0f} MB)")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "certified-track-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    if part.stat().st_size != size or sha256(part) != digest:
        print(f"size or sha256 mismatch for {rel}", file=sys.stderr)
        return False
    part.rename(path)
    print(f"done     {rel} in {time.time() - t0:.0f} s")
    return True


def checkout():
    if not REPO_DIR.exists():
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)], check=True)
    head = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != REPO_COMMIT:
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "-q"], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "-q", REPO_COMMIT], check=True)
        print(f"checked out co-tracker {REPO_COMMIT[:12]}")
    else:
        print(f"ok       co-tracker at {REPO_COMMIT[:12]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    ok = True
    for rel, url, size, digest in WEIGHTS:
        if a.verify:
            print(("ok       " if good(DATA / rel, size, digest) else "BAD      ") + rel)
        else:
            ok &= fetch(rel, url, size, digest)
    if not a.verify:
        checkout()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

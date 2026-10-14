#!/usr/bin/env python3
"""TAP-Vid benchmark for one method.

    python scripts/bench_tapvid.py --method cotracker --set dev6
    python scripts/bench_tapvid.py --method cotracker --set all --out bench_davis_cotracker
    python scripts/bench_tapvid.py --method cotracker --set all --convention tapnet

Writes out/<out>.json and out/<out>.txt.  Methods: cotracker (CoTracker3 online, all queries).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from certtrack.data.tapvid import iter_clips  # noqa: E402
from certtrack.harness import format_table, run_frozen, write_out  # noqa: E402
from certtrack.metrics import headline  # noqa: E402
from certtrack.sets import SETS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cotracker", choices=["cotracker", "carry", "hybrid-verify"])
    ap.add_argument("--delta0", type=float, default=128.0)
    ap.add_argument("--side", type=int, default=21)
    ap.add_argument("--group", default="similarity", choices=["translation", "similarity", "affine"])
    ap.add_argument("--interp", type=int, default=1)
    ap.add_argument("--data", default="davis", help="folder under data/tapvid/ made by prepare_tapvid.py")
    ap.add_argument("--set", default="dev6", choices=sorted(SETS))
    ap.add_argument("--clips", nargs="*", help="explicit clip names (overrides --set)")
    ap.add_argument("--query-mode", default="first", choices=["first", "strided"])
    ap.add_argument("--convention", default="cotracker", choices=["cotracker", "tapnet"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    names = a.clips or SETS[a.set]
    tag = "" if a.method == "cotracker" else f"_s{a.side}_d{a.delta0:.0f}_{a.group[:3]}"
    stem = a.out or f"bench_{a.data}_{a.method}{tag}_{a.set if not a.clips else 'custom'}"
    from certtrack.models.cotracker import machine_info
    info = machine_info()
    if a.method == "cotracker":
        from certtrack.models.cotracker import CoTrackerStream
        tracker = CoTrackerStream(device=a.device)
        info["device"] = tracker.device
        info["checkpoint"] = str(tracker.checkpoint.name)
    elif a.method == "carry":
        from certtrack.methods import CarryOnly
        tracker = CarryOnly(delta0=a.delta0, s=a.side, group=a.group, interp=bool(a.interp))
        info["device"] = "cpu"
        info["tracker"] = tracker.kw
    else:
        from certtrack.methods import HybridVerify
        from certtrack.models.cotracker import CoTrackerStream
        model = CoTrackerStream(device=a.device)
        tracker = HybridVerify(model, delta0=a.delta0, s=a.side, group=a.group, interp=bool(a.interp))
        info["device"] = model.device
        info["checkpoint"] = str(model.checkpoint.name)
        info["tracker"] = tracker.kw

    def progress(name, m):
        print(f"{name:22s} {headline(m)}  surv {100*m['long']['survival']:.1f}  {m['seconds']:.1f} s", flush=True)

    t0 = time.time()
    res = run_frozen(tracker, iter_clips(ROOT / "data" / "tapvid" / a.data, names, a.convention), a.query_mode, progress)
    payload = {"method": a.method, "data": a.data, "set": a.set if not a.clips else names, "convention": a.convention,
               "machine": info, **res}
    title = (f"{a.method} on TAP-Vid {a.data} ({a.set}, {a.query_mode}, {a.convention} convention, "
             f"{info['device']}), {time.strftime('%Y-%m-%d')}")
    pj, pt = write_out(stem, payload, format_table(res, title))
    print(open(pt).read())
    print(f"wrote {pj} and {pt} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Long-video protocol: palindromes of TAP-Vid clips with exact ground truth.

    python scripts/bench_long.py --method cotracker --set dev6 --length 500
    python scripts/bench_long.py --method hybrid-verify --set dev6 --length 500 --kind tail
    python scripts/bench_long.py --method carry --set all --length 1000 --kind tail

kind 'palindrome' bounces over the whole clip (the query frame recurs); 'tail' bounces inside
[T//3, T-1] after the first pass, so the frames around the query never recur.  Real footage,
artificial time order: see python/certtrack/data/longseq.py for what it does and does not test.

Per clip: TAP-Vid metrics on the long sequence (query first), survival (50 px and 16 px at 256),
median trajectory error, re-detection AJ, certified fraction and proof precision, error at the
midpoint and the end, model calls.  Runs are cached in data/runs/<stem>/<clip>.npz for the
renderer.  Writes out/<stem>.{json,txt}.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from certtrack import longmetrics as lm, states  # noqa: E402
from certtrack.data.longseq import build_long  # noqa: E402
from certtrack.data.queries import eval_mask, queries_txy, sample_queries_first  # noqa: E402
from certtrack.data.tapvid import Clip, iter_clips  # noqa: E402
from certtrack.harness import score, write_out  # noqa: E402
from certtrack.metrics import headline, summarise  # noqa: E402
from certtrack.sets import SETS  # noqa: E402


def make_method(a, info):
    if a.method == "cotracker":
        from certtrack.methods import FrozenOnly
        from certtrack.models.cotracker import CoTrackerStream
        m = CoTrackerStream(device=a.device)
        info["device"] = m.device
        return FrozenOnly(m)
    if a.method == "carry":
        from certtrack.methods import CarryOnly
        return CarryOnly(delta0=a.delta0, s=a.side, group=a.group, interp=bool(a.interp))
    from certtrack.methods import HybridVerify
    from certtrack.models.cotracker import CoTrackerStream
    m = CoTrackerStream(device=a.device)
    info["device"] = m.device
    return HybridVerify(m, delta0=a.delta0, s=a.side, group=a.group, interp=bool(a.interp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cotracker", choices=["cotracker", "carry", "hybrid-verify"])
    ap.add_argument("--data", default="davis")
    ap.add_argument("--set", default="dev6", choices=sorted(SETS))
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--length", type=int, default=500)
    ap.add_argument("--kind", default="tail", choices=["palindrome", "tail"])
    ap.add_argument("--delta0", type=float, default=80.0)
    ap.add_argument("--side", type=int, default=21)
    ap.add_argument("--group", default="similarity", choices=["translation", "similarity", "affine"])
    ap.add_argument("--interp", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from certtrack.models.cotracker import machine_info
    info = machine_info()
    method = make_method(a, info)
    tag = "" if a.method == "cotracker" else f"_s{a.side}_d{a.delta0:.0f}_{a.group[:3]}"
    stem = a.out or f"long_{a.data}_{a.kind}{a.length}_{a.method}{tag}_{a.set if not a.clips else 'custom'}"
    runs = ROOT / "data" / "runs" / stem
    runs.mkdir(parents=True, exist_ok=True)

    per = {}
    t_all = time.time()
    for base in iter_clips(ROOT / "data" / "tapvid" / a.data, a.clips or SETS[a.set]):
        T = base.n_frames
        seq = build_long(base.name, base.frames, base.tracks, base.occluded, a.length,
                         None if a.kind == "palindrome" else T // 3)
        frames = base.frames[seq.index]
        clip = Clip(base.name, frames, seq.tracks, seq.occluded, base.norm_points[:, seq.index], base.native_hw, base.convention)
        s = sample_queries_first(clip.occluded, clip.tracks, clip.frames)
        q = queries_txy(s)
        t0 = time.perf_counter()
        res = method.track_clip(clip.frames, q)
        dt = time.perf_counter() - t0
        tr, vi, log = res[:3]
        st = res[3] if len(res) > 3 else None
        m = score(clip, s, tr, vi, "first", state_tn=st)
        m["seconds"] = round(dt, 2)
        m["calls"] = log.summary()
        # error at the midpoint and the end, in evaluation pixels, over GT-visible points
        scale = clip.eval_scale()
        gt = np.asarray(s["target_points"])[0] * scale
        occ = np.asarray(s["occluded"])[0]
        pred = tr.transpose(1, 0, 2) * scale
        err = np.sqrt(((pred - gt) ** 2).sum(-1))
        for label, t in (("mid", a.length // 2), ("end", a.length - 1)):
            sel = ~occ[:, t]
            m[f"err_{label}_median"] = float(np.median(err[sel, t])) if sel.any() else float("nan")
            m[f"err_{label}_within16"] = float(np.mean(err[sel, t] < 16)) if sel.any() else float("nan")
        per[base.name] = m
        np.savez_compressed(runs / f"{base.name}.npz", tracks=tr, visible=vi, state=st if st is not None else np.zeros(vi.shape, np.uint8),
                            index=seq.index, queries=q, gt=np.asarray(s["target_points"])[0], occ=occ)
        L = m["long"]
        print(f"{base.name:22s} {headline(m)}  surv50 {100*L['survival']:5.1f} surv16 {100*L['survival_16']:5.1f} MTE {L['mte']:5.2f} "
              f"AJrd {100*L['aj_rd']:5.1f} cert {100*L['certified']['of_gt_visible']:5.1f}% end<16 {100*m['err_end_within16']:5.1f}  {dt:6.1f} s", flush=True)

    flat = {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))} for k, v in per.items()}
    longflat = {k: {"survival": v["long"]["survival"], "survival_16": v["long"]["survival_16"], "mte": v["long"]["mte"],
                    "aj_rd": v["long"]["aj_rd"], "cert_of_gt_visible": v["long"]["certified"]["of_gt_visible"],
                    "proof_within_4": v["long"]["proof_precision"]["within_4"]} for k, v in per.items()}
    mean = summarise(flat)
    mean.update({f"long_{k}": v for k, v in summarise(longflat).items()})
    payload = {"method": a.method, "kind": a.kind, "length": a.length, "data": a.data, "set": a.set if not a.clips else a.clips,
               "tracker": getattr(method, "kw", None), "machine": info, "per_clip": per, "mean": mean,
               "seconds": round(time.time() - t_all, 1)}
    lines = [f"{a.method} on {a.kind} x {a.length} of TAP-Vid {a.data} ({a.set}), {time.strftime('%Y-%m-%d')}", "",
             f"{'clip':22s} {'AJ':>6s} {'d_avg':>6s} {'OA':>6s} {'surv50':>6s} {'surv16':>6s} {'MTE':>6s} {'AJrd':>6s} {'cert%':>6s} {'w4':>6s} {'mid<16':>6s} {'end<16':>6s} {'s':>7s}"]
    for k, v in per.items():
        L = v["long"]
        lines.append(f"{k:22s} {100*v['average_jaccard']:6.1f} {100*v['average_pts_within_thresh']:6.1f} {100*v['occlusion_accuracy']:6.1f} "
                     f"{100*L['survival']:6.1f} {100*L['survival_16']:6.1f} {L['mte']:6.2f} {100*L['aj_rd']:6.1f} "
                     f"{100*L['certified']['of_gt_visible']:6.1f} {100*L['proof_precision']['within_4']:6.1f} "
                     f"{100*v['err_mid_within16']:6.1f} {100*v['err_end_within16']:6.1f} {v['seconds']:7.1f}")
    lines += ["", f"mean over {len(per)} clips: {headline(mean)}  survival {100*mean['long_survival']:.1f} (16 px: {100*mean['long_survival_16']:.1f})  "
              f"MTE {mean['long_mte']:.2f}  AJ_RD {100*mean['long_aj_rd']:.1f}  certified {100*mean['long_cert_of_gt_visible']:.1f} %  "
              f"end<16 {100*mean['err_end_within16']:.1f} %   total {payload['seconds']} s", ""]
    pj, pt = write_out(stem, payload, "\n".join(lines))
    print(open(pt).read())
    print(f"wrote {pj}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E1 / E2 / E3: how much of real footage the certificate covers, and whether a certified step is in place.

    python scripts/e1_certified_fraction.py --set dev6
    python scripts/e1_certified_fraction.py --set all --sides 21 --delta0 80 96
    python scripts/e1_certified_fraction.py --set dev6 --oracle        # certify at the ground-truth position

For every setting (template side, Delta_0, radii, offset) the certify-only tracker runs on the
clips and we record, over counted frames with visible ground truth:
  certified fraction, overall and by gap from the query (1-5, 6-15, 16-30, 31+ frames)
  false-visible rate: certified while the ground truth says occluded
  proof precision: of certified steps, share within 1 / 2 / 4 / 8 px of the ground truth,
    split by the template's self-match set (n_self = 0 vs > 0) and by its kappa
  the TAP-Vid metrics of the certify-only tracker itself
  seconds per frame
--oracle replaces the search by the rounded ground-truth position: the certificate's ceiling,
independent of any tracker (analysis only).  Writes out/e1_<set>[_oracle].{json,txt}.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from certtrack import states  # noqa: E402
from certtrack.data.queries import eval_mask, queries_txy  # noqa: E402
from certtrack.data.tapvid import grey, iter_clips  # noqa: E402
from certtrack.harness import queries_for, score, write_out  # noqa: E402
from certtrack.sets import SETS  # noqa: E402
from certtrack.track import CertifiedTracker  # noqa: E402

GAPS = [(1, 5), (6, 15), (16, 30), (31, 10 ** 6)]


def analyse(clip, sample, r, eval_hw=(256, 256)):
    scale = clip.eval_scale(eval_hw)
    q = np.asarray(sample["query_points"])[0]
    gt = np.asarray(sample["target_points"])[0] * scale         # (N, T, 2) eval px
    occ = np.asarray(sample["occluded"])[0]
    pred = r["tracks"].transpose(1, 0, 2) * scale
    st = r["state"].T
    mask = eval_mask(q[:, 0], clip.n_frames, "first")
    vis = ~occ & mask
    cert = (st == states.CERTIFIED) & mask
    err = np.sqrt(((pred - gt) ** 2).sum(-1))
    gap = np.arange(clip.n_frames)[None, :] - q[:, 0:1].astype(int)
    out = {"n_vis": int(vis.sum()), "cert_of_vis": float((cert & vis).sum() / max(vis.sum(), 1)),
           "false_visible": float((cert & occ & mask).sum() / max(cert.sum(), 1))}
    for lo, hi in GAPS:
        g = vis & (gap >= lo) & (gap <= hi)
        out[f"cert_gap_{lo}"] = float((cert & g).sum() / g.sum()) if g.sum() else float("nan")
    tm = r["templates"]
    sel = cert & vis
    for label, m in (("all", np.ones(len(q), bool)), ("self0", tm["n_self"] == 0), ("selfpos", tm["n_self"] > 0),
                     ("kappa0", tm["kappa"].sum(1) == 0), ("kappapos", tm["kappa"].sum(1) > 0)):
        s2 = sel & m[:, None]
        e = err[s2]
        out[f"n_{label}"] = int(e.size)
        for k in (1, 2, 4, 8, 16):
            out[f"within{k}_{label}"] = float((e < k).mean()) if e.size else float("nan")
    out["n_points"] = int(len(q))
    out["frac_self0"] = float((tm["n_self"] == 0).mean())
    out["frac_kappa0"] = float((tm["kappa"].sum(1) == 0).mean())
    out["frac_border"] = float(tm["border"].mean())
    out["lam_min_median"] = float(np.median(tm["lam_min"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="davis")
    ap.add_argument("--set", default="dev6", choices=sorted(SETS))
    ap.add_argument("--sides", nargs="*", type=int, default=[13, 17, 21, 25])
    ap.add_argument("--delta0", nargs="*", type=float, default=[48, 64, 80, 96])
    ap.add_argument("--radii", nargs="*", default=["1,2,3", "0,1,2,3"])
    ap.add_argument("--offset", nargs="*", default=["opt"])
    ap.add_argument("--interp", nargs="*", type=int, default=[0], help="0 integer certificate, 1 resampled frame")
    ap.add_argument("--group", nargs="*", default=["translation"], choices=["translation", "similarity", "affine"])
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    clips = list(iter_clips(ROOT / "data" / "tapvid" / a.data, SETS[a.set]))
    greys = {c.name: grey(c.frames) for c in clips}
    samples = {c.name: queries_for(c) for c in clips}
    settings = list(itertools.product(a.sides, a.delta0, a.radii, a.offset, a.interp, a.group))
    results = []
    t_all = time.time()
    for s, d0, rad, off, interp, group in settings:
        radii = tuple(int(x) for x in rad.split(","))
        tr = CertifiedTracker(delta0=d0, radii=radii, s=s, R=a.R, offset=off, camera=not a.no_camera,
                              interp=bool(interp), group=group)
        per = {}
        secs = frames = 0
        for c in clips:
            smp = samples[c.name]
            q = queries_txy(smp)
            oracle = None
            if a.oracle:
                oracle = np.asarray(smp["target_points"])[0].transpose(1, 0, 2)
            t0 = time.perf_counter()
            r = tr.run(greys[c.name], q, oracle_xy=oracle)
            secs += time.perf_counter() - t0
            frames += c.n_frames
            an = analyse(c, smp, r)
            m = score(c, smp, r["tracks"], r["visible"], state_tn=r["state"])
            an.update({k: m[k] for k in ("average_jaccard", "average_pts_within_thresh", "occlusion_accuracy")})
            per[c.name] = an
        keys = [k for k in next(iter(per.values())) if isinstance(next(iter(per.values()))[k], float)]
        mean = {k: float(np.nanmean([p[k] for p in per.values()])) for k in keys}
        mean["ms_per_frame"] = 1000 * secs / frames
        row = {"side": s, "delta0": d0, "radii": radii, "offset": off, "interp": bool(interp), "group": group,
               "R": a.R, "oracle": a.oracle, "camera": not a.no_camera, "mean": mean, "per_clip": per}
        results.append(row)
        print(f"s={s:2d} D0={d0:3.0f} r={rad:8s} {off:5s} i={interp} {group[:3]} | cert {100*mean['cert_of_vis']:5.1f}%  "
              f"gaps {100*mean['cert_gap_1']:4.0f}/{100*mean['cert_gap_6']:4.0f}/{100*mean['cert_gap_16']:4.0f}/{100*mean['cert_gap_31']:4.0f}  "
              f"falsevis {100*mean['false_visible']:4.1f}%  within 2/4/8 px {100*mean['within2_all']:4.1f}/{100*mean['within4_all']:4.1f}/{100*mean['within8_all']:4.1f}  "
              f"self0 {100*mean['within4_self0']:4.1f} selfpos {100*mean['within4_selfpos']:4.1f}  "
              f"AJ {100*mean['average_jaccard']:4.1f}  {mean['ms_per_frame']:5.1f} ms/f", flush=True)

    stem = a.out or f"e1_{a.data}_{a.set}" + ("_oracle" if a.oracle else "") + ("_nocam" if a.no_camera else "")
    lines = [f"E1: certified fraction and proof precision, TAP-Vid {a.data} {a.set}, "
             f"{'oracle position' if a.oracle else 'tracked'}, {time.strftime('%Y-%m-%d')}", ""]
    lines.append(f"{'s':>3s} {'D0':>4s} {'radii':>8s} {'offset':>6s} {'i':>1s} {'grp':>3s} | {'cert%':>6s} {'g1-5':>5s} {'g6-15':>5s} {'g16-30':>6s} {'g31+':>5s} "
                 f"{'falsevis':>8s} | {'w2':>5s} {'w4':>5s} {'w8':>5s} {'w4|S=0':>7s} {'w4|S>0':>7s} {'self0%':>6s} {'kap0%':>5s} | {'AJ':>5s} {'ms/f':>6s}")
    for r in results:
        m = r["mean"]
        lines.append(f"{r['side']:3d} {r['delta0']:4.0f} {','.join(map(str, r['radii'])):>8s} {r['offset']:>6s} {int(r['interp']):1d} {r['group'][:3]:>3s} | "
                     f"{100*m['cert_of_vis']:6.1f} {100*m['cert_gap_1']:5.1f} {100*m['cert_gap_6']:5.1f} {100*m['cert_gap_16']:6.1f} {100*m['cert_gap_31']:5.1f} "
                     f"{100*m['false_visible']:8.1f} | {100*m['within2_all']:5.1f} {100*m['within4_all']:5.1f} {100*m['within8_all']:5.1f} "
                     f"{100*m['within4_self0']:7.1f} {100*m['within4_selfpos']:7.1f} {100*m['frac_self0']:6.1f} {100*m['frac_kappa0']:5.1f} | "
                     f"{100*m['average_jaccard']:5.1f} {m['ms_per_frame']:6.1f}")
    lines.append("")
    lines.append("cert%: certified share of counted frames with visible GT; g*: the same by gap from the query; "
                 "falsevis: certified while GT occluded, of certified; w_k: certified steps within k px of GT; "
                 "S=0: templates whose self-match set is {0} (no shift with |u|>=2 certifies).")
    text = "\n".join(lines) + "\n"
    pj, pt = write_out(stem, {"args": vars(a), "results": results, "seconds": time.time() - t_all}, text)
    print(text)
    print(f"wrote {pj}")


if __name__ == "__main__":
    main()

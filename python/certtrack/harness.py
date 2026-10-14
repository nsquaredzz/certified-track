"""Run a tracker on TAP-Vid clips and score it the published way.

Every result file carries the machine, the convention and the call log, so a number can
be traced to a run.  ``out/<stem>.json`` is the record; ``out/<stem>.txt`` is for reading.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .data.queries import eval_mask, queries_txy, sample_queries_first, sample_queries_strided
from .data.tapvid import Clip
from .metrics import headline, summarise, tapvid_metrics
from . import longmetrics as lm

ROOT = Path(__file__).resolve().parents[2]


def queries_for(clip: Clip, query_mode: str = "first") -> dict:
    """tapnet's sample dict for a clip, points in the clip's working pixels."""
    f = sample_queries_first if query_mode == "first" else sample_queries_strided
    s = f(clip.occluded, clip.tracks, clip.frames)
    return s


def score(clip: Clip, sample: dict, pred_tracks_tn: np.ndarray, pred_visible_tn: np.ndarray,
          query_mode: str = "first", state_tn: np.ndarray | None = None, eval_hw=(256, 256)) -> dict:
    """TAP-Vid metrics in the evaluation frame plus the long-video and certificate metrics."""
    scale = clip.eval_scale(eval_hw)
    q = np.asarray(sample["query_points"])[0].astype(np.float64)
    q[:, 1:] = q[:, 1:] * scale[::-1]  # (t, y, x)
    gt = np.asarray(sample["target_points"])[0] * scale
    occ = np.asarray(sample["occluded"])[0]
    pred = pred_tracks_tn.transpose(1, 0, 2) * scale
    pocc = ~pred_visible_tn.T
    out = tapvid_metrics(q, occ, gt, pocc, pred, query_mode)
    mask = eval_mask(q[:, 0], clip.n_frames, query_mode)
    st = state_tn.T if state_tn is not None else np.zeros_like(occ, np.uint8)
    out["long"] = lm.long_summary(gt, occ, pred, pocc, st, mask)
    out["n_queries"] = int(q.shape[0])
    out["n_frames"] = int(clip.n_frames)
    return out


def run_frozen(tracker, clips: Iterable[Clip], query_mode: str = "first",
               on_clip: Callable[[str, dict], None] | None = None) -> dict:
    """Baseline protocol: the frozen tracker on all queries of every clip."""
    per_clip, logs = {}, {}
    t_all = time.perf_counter()
    for clip in clips:
        s = queries_for(clip, query_mode)
        q = queries_txy(s)
        t0 = time.perf_counter()
        res = tracker.track_clip(clip.frames, q)
        dt = time.perf_counter() - t0
        tr, vi, log = res[:3]
        st = res[3] if len(res) > 3 else None
        m = score(clip, s, tr, vi, query_mode, state_tn=st)
        m["seconds"] = round(dt, 3)
        m["calls"] = log.summary()
        per_clip[clip.name] = m
        logs[clip.name] = log
        if on_clip:
            on_clip(clip.name, m)
    flat = {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))} for k, v in per_clip.items()}
    return {"per_clip": per_clip, "mean": summarise(flat), "seconds": round(time.perf_counter() - t_all, 1),
            "query_mode": query_mode}


def write_out(stem: str, payload: dict, text: str) -> tuple[Path, Path]:
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    pj, pt = out / f"{stem}.json", out / f"{stem}.txt"
    pj.write_text(json.dumps(payload, indent=1, default=_json_default))
    pt.write_text(text)
    return pj, pt


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def format_table(result: dict, title: str) -> str:
    lines = [title, ""]
    lines.append(f"{'clip':22s} {'AJ':>6s} {'d_avg':>6s} {'OA':>6s} {'surv':>6s} {'MTE':>6s} {'N':>4s} {'T':>4s} {'s':>7s}")
    for name, m in result["per_clip"].items():
        L = m["long"]
        lines.append(f"{name:22s} {100*m['average_jaccard']:6.1f} {100*m['average_pts_within_thresh']:6.1f} "
                     f"{100*m['occlusion_accuracy']:6.1f} {100*L['survival']:6.1f} {L['mte']:6.2f} "
                     f"{m['n_queries']:4d} {m['n_frames']:4d} {m['seconds']:7.1f}")
    mean = result["mean"]
    lines.append("")
    lines.append(f"mean over {len(result['per_clip'])} clips: {headline(mean)}   total {result['seconds']} s")
    return "\n".join(lines) + "\n"

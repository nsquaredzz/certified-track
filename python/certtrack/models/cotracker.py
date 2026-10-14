"""CoTracker3 behind the FrozenTracker protocol.

The source is the pinned checkout under data/third_party/co-tracker (scripts/fetch_weights.py)
and the weights are data/weights/cotracker3/scaled_{online,offline}.pth.  Nothing is
fetched through torch.hub at run time; TORCH_HOME and HF_HOME are pointed into data/
so no cache lands in the home folder by accident.

CoTrackerStream mirrors cotracker/evaluation/core/evaluator.py: the online model is
initialised with the queries, then fed windows of 2 * step frames at a stride of step
(window_len 16, step 8 for CoTracker3), and the last call returns every frame.  The
predictor multiplies visibility by its confidence and thresholds at 0.6; both raw
values are kept for the precision-recall comparison with the certificate.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from .base import CallLog

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "data" / "third_party" / "co-tracker"
WEIGHTS = ROOT / "data" / "weights" / "cotracker3"

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TORCH_HOME", str(ROOT / "data" / "weights" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "weights" / "hf"))


def pick_device(pref: str | None = None) -> str:
    import torch
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _import_cotracker():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor  # noqa: E402
    return CoTrackerOnlinePredictor, CoTrackerPredictor


def synchronize(device: str) -> None:
    import torch
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class CoTrackerStream:
    """CoTracker3 online on all queries, every window: the baseline and the 'verify' backbone."""
    name = "cotracker3_online"

    def __init__(self, checkpoint: Path | None = None, device: str | None = None, window_len: int = 16):
        import torch
        Online, _ = _import_cotracker()
        self.device = pick_device(device)
        self.checkpoint = Path(checkpoint or WEIGHTS / "scaled_online.pth")
        self.model = Online(checkpoint=str(self.checkpoint), window_len=window_len).to(self.device)
        self.step = int(self.model.step)
        self.torch = torch

    def _chunk(self, frames: np.ndarray):
        t = self.torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
        return t.permute(0, 3, 1, 2).float()[None]  # (1, T, 3, H, W)

    def track_clip(self, frames: np.ndarray, queries_txy: np.ndarray, with_raw: bool = False):
        T = frames.shape[0]
        N = queries_txy.shape[0]
        q = self.torch.from_numpy(np.asarray(queries_txy, np.float32)).to(self.device)[None]
        log = CallLog()
        with self.torch.no_grad():
            self.model(video_chunk=self._chunk(frames[: 2 * self.step]), is_first_step=True, queries=q,
                       add_support_grid=False)
            tracks = vis = None
            for ind in range(0, T - self.step, self.step):
                chunk = frames[ind: ind + 2 * self.step]
                t0 = time.perf_counter()
                tracks, vis = self.model(video_chunk=self._chunk(chunk), add_support_grid=False, grid_size=0)
                synchronize(self.device)
                log.add(N, chunk.shape[0], time.perf_counter() - t0)
        if tracks is None:  # clip shorter than one step: nothing was processed
            tr = np.broadcast_to(queries_txy[None, :, 1:], (T, N, 2)).astype(np.float32).copy()
            return tr, np.zeros((T, N), bool), log
        tr = tracks[0].cpu().numpy().astype(np.float32)
        vi = vis[0].cpu().numpy().astype(bool)
        if tr.shape[0] < T:  # should not happen with the evaluator's loop; pad defensively
            pad = T - tr.shape[0]
            tr = np.concatenate([tr, np.repeat(tr[-1:], pad, 0)])
            vi = np.concatenate([vi, np.repeat(vi[-1:], pad, 0)])
        tr, vi = tr[:T], vi[:T]
        # the query frame itself is given, as the offline predictor also enforces
        qt = queries_txy[:, 0].astype(int)
        tr[qt, np.arange(N)] = queries_txy[:, 1:]
        vi[qt, np.arange(N)] = True
        before = np.arange(T)[:, None] < qt[None, :]
        tr[before] = np.broadcast_to(queries_txy[None, :, 1:], (T, N, 2))[before]
        vi[before] = False
        return tr, vi, log


def machine_info() -> dict:
    import platform
    import subprocess
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                     text=True, timeout=5).stdout.strip()
    except Exception:
        info["cpu"] = platform.processor()
    try:
        import torch
        info["torch"] = torch.__version__
        info["mps"] = bool(torch.backends.mps.is_available())
        info["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        info["torch"] = None
    info["numpy"] = np.__version__
    return info

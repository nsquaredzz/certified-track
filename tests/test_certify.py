"""Certificate invariants: agreement with certskip on a fixed camera, exactness under integer motion, soundness."""
import numpy as np
import pytest
from certskip.core import Pruner
from certskip.scalespace import box_max_abs_mean
from certskip.warp import WarpPruner

from certtrack.track.certify import kappa, schedule, score, selfmatch, selfmatch_stats
from certtrack.track.refine import gn_refine
from certtrack.track.search import integer_search
from certtrack.track.template import crops_at, make_templates, pad_frame


def texture(H, W, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((H // 8 + 2, W // 8 + 2)) * 200 + 20
    img = np.kron(base, np.ones((8, 8)))[:H, :W]
    img += rng.normal(0, 2, (H, W))
    return np.clip(img, 0, 255).astype(np.float32)


def test_range_certificate_matches_core_pruner_on_fixed_camera():
    # 16-aligned templates of side 16 with r = 0 and offset 'midrange' are certskip's range rule
    H, W = 64, 96
    f0 = texture(H, W, 1)
    f1 = np.clip(f0 + np.random.default_rng(2).normal(0, 3, f0.shape), 0, 255).astype(np.float32)
    pr = Pruner(H, W, 16, delta=np.inf)
    pr.step(f0)
    _, spread, _ = pr.step(f1)
    deltas = {0: 32.0}
    centres = np.array([(gy * 16 + 8, gx * 16 + 8) for gy in range(H // 16) for gx in range(W // 16)])
    # a 16-wide window centred at +8 covers rows gy*16 .. gy*16+15 when the crop is taken at [c-8, c+8)
    pad = 20
    P0, P1 = pad_frame(f0, pad), pad_frame(f1, pad)
    win0 = np.lib.stride_tricks.sliding_window_view(P0, (16, 16))[centres[:, 0] + pad - 8, centres[:, 1] + pad - 8]
    win1 = np.lib.stride_tricks.sliding_window_view(P1, (16, 16))[centres[:, 0] + pad - 8, centres[:, 1] + pad - 8]
    sc, c = score(win1, win0, deltas, "midrange")
    # score = 2 * max|d - midrange| / Delta = spread / Delta
    assert np.allclose(sc * 32.0, spread.reshape(-1), atol=1e-4)


def test_multiscale_matches_warppruner_without_shift():
    H, W = 64, 96
    f0 = texture(H, W, 3)
    f1 = np.clip(f0 + np.random.default_rng(4).normal(0, 3, f0.shape), 0, 255).astype(np.float32)
    sched = {1: 80 / np.sqrt(3), 2: 80 / np.sqrt(5), 3: 80 / np.sqrt(7)}
    wp = WarpPruner(H, W, 16, delta_max=0.0, multiscale=sched, offset="patch")
    wp.step(f0)
    _, ref_score, _ = wp.step(f1)
    d = (f1 - f0).astype(np.float64).reshape(H // 16, 16, W // 16, 16).transpose(0, 2, 1, 3)
    sc, _ = score(d + 0.0, np.zeros_like(d), sched, "midrange")
    assert np.allclose(sc, ref_score, atol=1e-6)


def test_integer_motion_is_exact():
    F = texture(80, 120, 5)
    deltas = schedule(80)
    pts = np.array([[40.0, 30.0], [70.0, 50.0]])  # (x, y)
    tm = make_templates(F, pts, 21, deltas, self_radius=4)
    dy, dx = 3, -5
    G = np.roll(np.roll(F, dy, 0), dx, 1)
    pad = 10 + 16 + 2
    Gp = pad_frame(G, pad)
    k, _ = integer_search(Gp, pad, tm.T, tm.k, 16)
    assert np.array_equal(k - tm.k, np.array([[dy, dx], [dy, dx]]))
    sc, c = score(crops_at(Gp, pad, k, 10), tm.T, deltas)
    assert np.allclose(sc, 0.0) and np.allclose(c, 0.0)


def test_certified_steps_are_sound():
    # whatever passes: max |crop - (T + c)| < Delta_r/2 on every box mean, by construction; check the r=1 box bound
    rng = np.random.default_rng(6)
    deltas = schedule(64)
    T = rng.random((50, 21, 21)) * 255
    crops = T + rng.normal(0, 6, T.shape)
    sc, c = score(crops, T, deltas)
    e = crops - T - c[:, None, None]
    for r, D in deltas.items():
        m = box_max_abs_mean(e, r)
        assert np.all(m[sc < 1] < 0.5 * D + 1e-9)


def test_subpixel_refinement_recovers_shift():
    # a smooth template, frame shifted by (0.3, -0.4): GN finds it within 0.1 px
    yy, xx = np.mgrid[0:60, 0:60].astype(np.float64)
    def img(dy, dx):
        return (120 + 80 * np.sin((xx - dx) / 4.0) * np.cos((yy - dy) / 5.0)).astype(np.float32)
    F0, F1 = img(0, 0), img(0.3, -0.4)
    deltas = schedule(80)
    tm = make_templates(F0, np.array([[30.0, 30.0]]), 21, deltas, self_radius=2)
    pad = 30
    d, c = gn_refine(pad_frame(F1, pad), pad, tm.T, tm.gy, tm.gx, tm.Hinv, tm.k, iters=4)
    assert abs(d[0, 0] - 0.3) < 0.1 and abs(d[0, 1] + 0.4) < 0.1


def test_kappa_and_selfmatch_flat_vs_textured():
    deltas = schedule(80)
    flat = np.full((21, 21), 100.0)
    assert kappa(flat, deltas).sum() == 0
    F = texture(80, 80, 7)
    tm = make_templates(F, np.array([[40.0, 40.0]]), 21, deltas, self_radius=6)
    assert tm.kappa[0].sum() > 0
    # a flat window certifies against itself everywhere; a textured one only near the origin
    sm_flat = selfmatch(np.full((33, 33), 100.0), 21, deltas)
    assert selfmatch_stats(sm_flat)[0] == (33 - 21 + 1) ** 2 - 9
    sm_tex = selfmatch(pad_frame(F, 0)[28:61, 28:61], 21, deltas)
    assert sm_tex[6, 6] == 0.0
    assert selfmatch_stats(sm_tex)[1] > 0.0

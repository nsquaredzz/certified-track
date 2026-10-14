"""End to end on a synthetic scene with known motion: textured background under an affine camera,
a textured square moving across it, a flat occluder passing over part of the scene."""
import numpy as np

from certtrack import states
from certtrack.data.queries import eval_mask
from certtrack.harness import score
from certtrack.data.tapvid import Clip
from certtrack.methods import CarryOnly
from certtrack.longmetrics import certified_fraction, proof_precision, survival
from certtrack.track.refine import sample_bilinear


def make_scene(T=40, H=128, W=160, seed=0):
    rng = np.random.default_rng(seed)
    big = np.kron(rng.random((40, 48)) * 180 + 30, np.ones((6, 6)))      # textured world, larger than the frame
    big = np.clip(big + rng.normal(0, 1.5, big.shape), 0, 255)
    obj = np.kron(rng.random((5, 5)) * 200 + 20, np.ones((4, 4)))          # 20 x 20 textured square
    frames = np.zeros((T, H, W), np.float32)
    bg_pts = np.array([[40.0, 50.0], [120.0, 30.0], [80.0, 100.0], [140.0, 90.0]])   # (x, y) in frame 0
    obj0 = np.array([30.0, 60.0])                                           # object top-left in frame 0
    gt = np.zeros((6, T, 2))
    occ = np.zeros((6, T), bool)
    Pb = np.pad(big, 0)
    for t in range(T):
        # camera: slow pan and a slight zoom
        sx = 1 + 0.002 * t
        ox, oy = 20 + 0.6 * t, 25 + 0.3 * t
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        frames[t] = sample_bilinear(Pb, 0, oy + yy * sx, ox + xx * sx)
        # background points move with the camera: world (wx, wy) = (ox0 + x0 * 1, ...) at t = 0
        for i, (x0, y0) in enumerate(bg_pts):
            wx, wy = 20 + x0, 25 + y0
            gt[i, t] = [(wx - ox) / sx, (wy - oy) / sx]
        # object: moves right and down on top of the background
        ox_t, oy_t = obj0[0] + 1.5 * t, obj0[1] + 0.5 * t
        ix, iy = int(round(ox_t)), int(round(oy_t))
        frames[t, iy:iy + 20, ix:ix + 20] = obj[: max(0, min(20, H - iy)), : max(0, min(20, W - ix))]
        gt[4, t] = [ox_t + 9.5, oy_t + 9.5]
        gt[5, t] = [ox_t + 3, oy_t + 15]
        # a flat occluder crosses the top-left background point between frames 12 and 20
        if 12 <= t <= 20:
            frames[t, 35:70, 25:60] = 128.0
            px, py = gt[0, t]
            if 25 <= px < 60 and 35 <= py < 70:
                occ[0, t] = True
    rgb = np.repeat(np.clip(frames, 0, 255).astype(np.uint8)[..., None], 3, -1)
    return rgb, gt.astype(np.float32), occ


def test_carry_only_on_synthetic_scene():
    rgb, gt, occ = make_scene()
    T = rgb.shape[0]
    clip = Clip("synthetic", rgb, gt, occ, gt / np.array([159, 127]), (128, 160))
    q = np.concatenate([np.zeros((6, 1)), gt[:, 0]], 1).astype(np.float32)
    m = CarryOnly(delta0=80, s=17, group="similarity")
    tr, vi, log, st = m.track_clip(rgb, q)
    mask = eval_mask(np.zeros(6, int), T, "first")
    cf = certified_fraction(st.T, ~vi.T, occ, mask)
    pp = proof_precision(st.T, gt, occ, tr.transpose(1, 0, 2), mask)
    s = survival(gt, occ, tr.transpose(1, 0, 2), mask, thresh=8)
    per_point = (st.T == states.CERTIFIED).mean(1)
    # background points under the zooming camera and the object's centre certify nearly always
    # (point 2 sits on a texture-block edge that the resampled zoom blurs: it certifies less often)
    assert per_point[1] > 0.9 and per_point[3] > 0.85 and per_point[4] > 0.95 and per_point[2] > 0.5, per_point
    # the point whose template straddles the object's edge and the background cannot certify: two motions
    assert per_point[5] < 0.1, per_point
    # every certified step is in place
    assert pp["within_2"] > 0.98 and pp["n"] > 100, pp
    assert np.all(s[:5] > 0.9), s
    # the occluded frames of point 0 are not certified, and it is re-detected afterwards
    assert not np.any((st.T == states.CERTIFIED) & occ)
    assert np.any(st.T[0, 21:] == states.CERTIFIED)
    # the harness scores it without complaint
    sample = {"query_points": np.stack([np.zeros(6), gt[:, 0, 1], gt[:, 0, 0]], -1)[None],
              "target_points": gt[None], "occluded": occ[None]}
    out = score(clip, sample, tr, vi, state_tn=st)
    # five of six points tracked throughout; the edge point is lost from frame 1 and pulls AJ down to ~0.73
    assert 0.65 < out["average_jaccard"] < 0.85 and out["long"]["proof_precision"]["within_2"] == 1.0

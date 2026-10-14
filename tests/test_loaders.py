import pickle

import numpy as np
import pytest
from PIL import Image

from certtrack.data.davis import load_davis
from certtrack.data.tapvid import grey, iter_clips, load_clip, points_to_pixels, prepare_tapvid, resize_frames


def make_pickle(tmp_path, as_list=False):
    rng = np.random.default_rng(1)
    recs = {}
    for name in ("a", "b"):
        T, H, W = 5, 32, 48
        video = rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8)
        pts = rng.random((3, T, 2)).astype(np.float32)
        occ = rng.random((3, T)) < 0.2
        recs[name] = {"video": video, "points": pts, "occluded": occ}
    data = list(recs.values()) if as_list else recs
    p = tmp_path / "x.pkl"
    p.write_bytes(pickle.dumps(data))
    return p, recs


@pytest.mark.parametrize("as_list", [False, True])
def test_prepare_and_load(tmp_path, as_list):
    p, recs = make_pickle(tmp_path, as_list)
    idx = prepare_tapvid(p, tmp_path / "out", (16, 16))
    names = list(idx["clips"])
    assert len(names) == 2
    c = load_clip(tmp_path / "out", names[0])
    assert c.frames.shape == (5, 16, 16, 3) and c.frames.dtype == np.uint8
    assert c.tracks.shape == (3, 5, 2) and c.occluded.shape == (3, 5)
    assert c.native_hw == (32, 48)
    # 1.0 maps to W-1 under the cotracker convention and to W under tapnet's
    assert np.allclose(points_to_pixels(np.ones((1, 1, 2)), (16, 16), "cotracker"), 15)
    assert np.allclose(points_to_pixels(np.ones((1, 1, 2)), (16, 16), "tapnet"), 16)
    assert np.allclose(c.eval_scale((256, 256)), [255 / 15, 255 / 15])
    assert len(list(iter_clips(tmp_path / "out"))) == 2


def test_prepare_rejects_bad_points(tmp_path):
    p, recs = make_pickle(tmp_path)
    data = pickle.loads(p.read_bytes())
    data["a"]["points"] = data["a"]["points"] * 300
    p.write_bytes(pickle.dumps(data))
    with pytest.raises(ValueError):
        prepare_tapvid(p, tmp_path / "out")


def test_resize_and_grey():
    f = np.zeros((2, 8, 12, 3), np.uint8)
    f[..., 0] = 200
    r = resize_frames(f, (4, 6))
    assert r.shape == (2, 4, 6, 3)
    g = grey(f)
    assert g.shape == (2, 8, 12) and g.dtype == np.uint8 and 55 <= g[0, 0, 0] <= 65  # 0.299 * 200


def test_davis_loader(tmp_path):
    root = tmp_path / "DAVIS"
    (root / "JPEGImages" / "480p" / "clip").mkdir(parents=True)
    (root / "Annotations" / "480p" / "clip").mkdir(parents=True)
    for i in range(3):
        Image.fromarray(np.full((8, 8, 3), i * 40, np.uint8)).save(root / "JPEGImages" / "480p" / "clip" / f"{i:05d}.jpg")
    m = np.zeros((8, 8), np.uint8)
    m[2:5, 2:5] = 1
    Image.fromarray(m, "P").save(root / "Annotations" / "480p" / "clip" / "00000.png")
    frames, masks = load_davis(root, "clip")
    assert frames.shape == (3, 8, 8, 3) and masks.shape == (3, 8, 8)
    assert masks[0].sum() == 9 and masks[1].sum() == 0

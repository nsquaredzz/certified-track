import numpy as np

from certtrack.data.queries import eval_mask, queries_txy, sample_queries_first, sample_queries_strided


def toy():
    # 3 tracks, 7 frames; track 2 is never visible
    occ = np.array([
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ], bool)
    pts = np.zeros((3, 7, 2), np.float32)
    pts[..., 0] = np.arange(7)[None] / 10
    pts[..., 1] = np.arange(3)[:, None] / 10
    frames = np.zeros((7, 4, 4, 3), np.uint8)
    return occ, pts, frames


def test_first_drops_invisible_and_queries_first_visible():
    occ, pts, frames = toy()
    s = sample_queries_first(occ, pts, frames)
    assert s["query_points"].shape == (1, 2, 3)
    # (t, y, x)
    assert s["query_points"][0, 0].tolist() == [1, 0.0, np.float32(0.1)]
    assert s["query_points"][0, 1].tolist() == [0, np.float32(0.1), 0.0]
    q = queries_txy(s)
    assert q.shape == (2, 3) and q[0].tolist() == [1, np.float32(0.1), 0.0]


def test_strided_counts():
    occ, pts, frames = toy()
    s = sample_queries_strided(occ, pts, frames, query_stride=5)
    # query frames 0 and 5: visible tracks are {1} at 0 and {0, 1} at 5
    assert s["query_points"].shape[1] == 3
    assert s["trackgroup"][0].tolist() == [1, 0, 1]
    assert s["query_points"][0, :, 0].tolist() == [0, 5, 5]


def test_eval_mask():
    m = eval_mask(np.array([0, 2]), 4, "first")
    assert m.tolist() == [[False, True, True, True], [False, False, False, True]]
    m = eval_mask(np.array([0, 2]), 4, "strided")
    assert m.tolist() == [[False, True, True, True], [True, True, False, True]]

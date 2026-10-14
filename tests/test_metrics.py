"""Hand-computed cases for the TAP-Vid and long-video metrics, and equality of the two implementations."""
import numpy as np
import pytest

from certtrack import longmetrics as lm, states
from certtrack.data.queries import eval_mask
from certtrack.metrics import compute_tapvid_metrics, summarise, tapvid_metrics


def one_point_case():
    # one point, T = 6, query at 0, GT at the origin, occluded at t = 3;
    # prediction visible everywhere with errors 0, 1.5, 3, 10, 20, 60 along x
    T = 6
    gt = np.zeros((1, T, 2))
    occ = np.zeros((1, T), bool)
    occ[0, 3] = True
    err = np.array([0, 1.5, 3, 10, 20, 60.0])
    pred = np.stack([err, np.zeros(T)], axis=-1)[None]
    pocc = np.zeros((1, T), bool)
    q = np.array([[0, 0, 0]], np.float64)  # (t, y, x)
    return q, occ, gt, pocc, pred


def test_tapvid_hand_case():
    q, occ, gt, pocc, pred = one_point_case()
    m = tapvid_metrics(q, occ, gt, pocc, pred, "first")
    # counted frames 1..5; GT visible at 1, 2, 4, 5
    assert m["pts_within_1"] == pytest.approx(0 / 4)
    assert m["pts_within_2"] == pytest.approx(1 / 4)
    assert m["pts_within_4"] == pytest.approx(2 / 4)
    assert m["pts_within_8"] == pytest.approx(2 / 4)
    assert m["pts_within_16"] == pytest.approx(2 / 4)
    assert m["average_pts_within_thresh"] == pytest.approx((0 + 0.25 + 0.5 + 0.5 + 0.5) / 5)
    # jaccard: TP / (GT positives 4 + false positives); t = 3 is a false positive at every threshold
    assert m["jaccard_1"] == pytest.approx(0.0)
    assert m["jaccard_2"] == pytest.approx(1 / (4 + 4))
    assert m["jaccard_4"] == pytest.approx(2 / (4 + 3))
    assert m["jaccard_8"] == pytest.approx(2 / (4 + 3))
    assert m["jaccard_16"] == pytest.approx(2 / (4 + 3))
    assert m["average_jaccard"] == pytest.approx((0 + 1 / 8 + 3 * 2 / 7) / 5)
    assert m["occlusion_accuracy"] == pytest.approx(4 / 5)


def test_long_hand_case():
    q, occ, gt, pocc, pred = one_point_case()
    mask = eval_mask(q[:, 0], 6, "first")
    assert lm.survival(gt, occ, pred, mask, thresh=50)[0] == pytest.approx(4 / 5)
    assert lm.survival(gt, occ, pred, mask, thresh=16)[0] == pytest.approx(3 / 5)
    assert lm.median_trajectory_error(gt, occ, pred, mask) == pytest.approx((3 + 20) / 2)
    rd = lm.redetection_mask(occ, mask)
    assert rd.tolist() == [[False, False, False, False, True, True]]
    st = np.array([[states.NONE, states.CERTIFIED, states.CERTIFIED, states.MODEL_HELD, states.CERTIFIED, states.OCCLUDED]])
    pocc2 = ~states.visible_from_state(st)
    cf = lm.certified_fraction(st, pocc2, occ, mask)
    assert cf["of_pred_visible"] == pytest.approx(3 / 4)   # visible at 1, 2, 3, 4; certified at 1, 2, 4
    assert cf["of_gt_visible"] == pytest.approx(3 / 4)     # GT visible at 1, 2, 4, 5
    pp = lm.proof_precision(st, gt, occ, pred, mask)
    assert pp["n"] == 3
    assert pp["within_2"] == pytest.approx(1 / 3)          # errors 1.5, 3, 20
    assert pp["within_4"] == pytest.approx(2 / 3)
    assert pp["within_16"] == pytest.approx(2 / 3)


def test_restricted_equals_reference_on_standard_mask():
    rng = np.random.default_rng(0)
    for mode in ("first", "strided"):
        for _ in range(25):
            N, T = rng.integers(1, 6), rng.integers(4, 12)
            occ = rng.random((N, T)) < 0.3
            occ[:, 0] = False
            gt = rng.random((N, T, 2)) * 256
            pred = gt + rng.normal(0, 4, (N, T, 2)) * (rng.random((N, T, 1)) < 0.7)
            pocc = rng.random((N, T)) < 0.3
            qt = rng.integers(0, T - 1, N)
            q = np.stack([qt, gt[np.arange(N), qt, 1], gt[np.arange(N), qt, 0]], axis=-1)
            ref = tapvid_metrics(q, occ, gt, pocc, pred, mode)
            ours = lm.restricted_metrics(occ, gt, pocc, pred, eval_mask(qt, T, mode))
            for k, v in ref.items():
                if np.isnan(v):
                    assert np.isnan(ours[k])
                else:
                    assert ours[k] == pytest.approx(v, abs=1e-12), (mode, k)


def test_batch_reference_matches_single():
    q, occ, gt, pocc, pred = one_point_case()
    m = compute_tapvid_metrics(q[None], occ[None], gt[None], pocc[None], pred[None], "first")
    assert float(m["average_jaccard"][0]) == pytest.approx(tapvid_metrics(q, occ, gt, pocc, pred)["average_jaccard"])


def test_summarise_means_over_clips():
    s = summarise({"a": {"x": 1.0, "y": 0.0}, "b": {"x": 3.0, "y": 1.0}})
    assert s == {"x": 2.0, "y": 0.5}

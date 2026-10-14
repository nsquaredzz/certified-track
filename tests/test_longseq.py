import numpy as np
import pytest

from certtrack.data.longseq import build_long, palindrome_index


def test_palindrome_steps_by_one_and_bounces():
    idx = palindrome_index(5, 13)
    assert idx.tolist() == [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4]
    assert np.all(np.abs(np.diff(idx)) == 1)


def test_tail_never_revisits_head():
    idx = palindrome_index(6, 20, tail_from=2)
    assert idx[:6].tolist() == [0, 1, 2, 3, 4, 5]
    assert idx[6:].min() == 2
    assert np.all(np.abs(np.diff(idx)) == 1)
    assert idx[6:].max() == 5


def test_bad_args():
    with pytest.raises(ValueError):
        palindrome_index(1, 5)
    with pytest.raises(ValueError):
        palindrome_index(5, 5, tail_from=4)


def test_build_long_remaps_labels():
    frames = np.arange(4)[:, None, None, None].repeat(2, 1).repeat(2, 2).repeat(3, 3).astype(np.uint8)
    tracks = np.arange(4, dtype=np.float32)[None, :, None].repeat(2, 2)
    occ = np.array([[False, True, False, False]])
    seq = build_long("x", frames, tracks, occ, length=9, tail_from=1)
    assert len(seq) == 9
    assert seq.index.tolist() == [0, 1, 2, 3, 2, 1, 2, 3, 2]
    assert seq.frame(4)[0, 0, 0] == 2
    assert seq.tracks[0, :, 0].tolist() == [0, 1, 2, 3, 2, 1, 2, 3, 2]
    assert seq.occluded[0].tolist() == [False, True, False, False, False, True, False, False, False]

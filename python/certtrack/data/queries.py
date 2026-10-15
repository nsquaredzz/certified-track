"""Query sampling for TAP-Vid evaluation, ported from tapnet.

The two functions below are the reference implementations from
``tapnet/tapvid/evaluation_datasets.py`` (Google DeepMind, Apache License 2.0),
kept line for line so that the evaluation protocol is the published one.  The
only additions are ``queries_txy`` (the (t, x, y) order that CoTracker takes)
and ``eval_mask`` (the frames that count, as tapnet's ``compute_tapvid_metrics``
derives them from the query points).

Copyright 2024 DeepMind Technologies Limited

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License.  You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0.  Unless required by
applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.

    Given a set of frames and tracks with no query points, use the first
    visible point in each track as the query.

    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.

    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3]
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1]
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1]
    """
    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)

    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.

    Given a set of frames and tracks with no query points, sample queries
    strided every query_stride frames, ignoring points that are not visible
    at the selected frames.

    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
      query_stride: When sampling query points, search for un-occluded points
        every query_stride frames and convert each one into a query.

    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3].  The video
          has floats scaled to the range [-1, 1].
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1].
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1].
        trackgroup: Index of the original track that each query point was
          sampled from.  This is useful for visualization.
    """
    tracks = []
    occs = []
    queries = []
    trackgroups = []
    total = 0
    trackgroup = np.arange(target_occluded.shape[0])
    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])
        total += np.array(np.sum(target_occluded[:, i] == 0))

    return {
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }


# ----------------------------------------------------------------------------
# Additions (certtrack)


def queries_txy(sample: Mapping[str, np.ndarray]) -> np.ndarray:
    """The queries of a tapnet sample as (t, x, y), shape (N, 3): CoTracker's order."""
    q = np.asarray(sample["query_points"])[0]
    return np.stack([q[:, 0], q[:, 2], q[:, 1]], axis=-1).astype(np.float32)


def eval_mask(query_t: np.ndarray, n_frames: int, query_mode: str = "first") -> np.ndarray:
    """Which (query, frame) pairs count, shape (N, T) bool.

    tapnet evaluates every frame after the query in ``first`` mode and every frame
    other than the query frame in ``strided`` mode (the query frame itself is
    never scored, its position is given).
    """
    t = np.arange(n_frames)[None, :]
    q = np.asarray(query_t).astype(int)[:, None]
    if query_mode == "first":
        return t > q
    if query_mode == "strided":
        return t != q
    raise ValueError(f"unknown query_mode {query_mode!r}")

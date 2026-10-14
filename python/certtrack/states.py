"""Per-step state codes shared by the tracker, the metrics and the renderer."""
from __future__ import annotations

import numpy as np

NONE = 0        # before the query frame: nothing is predicted
CERTIFIED = 1   # matched to the query template and certified at an integer position
MODEL_HELD = 2  # the frozen model's proposal, which did not certify; visible per the model
OCCLUDED = 3    # certificate failed and the model says not visible; position is a prediction
LOST = 4        # no model and no certified match: position is a prediction

NAMES = {NONE: "none", CERTIFIED: "certified", MODEL_HELD: "model", OCCLUDED: "occluded", LOST: "lost"}


def visible_from_state(state: np.ndarray) -> np.ndarray:
    """The visibility flag a consumer gets: certified or model-held steps are visible."""
    s = np.asarray(state)
    return (s == CERTIFIED) | (s == MODEL_HELD)

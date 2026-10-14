"""Fixed clip sets, so that no number in RESULTS.md depends on a selection made after seeing results."""

# Two rigid, two fast, two hard (occlusion, non-rigid).  Fixed on 2026-10-14 before any run.
DEV6 = ["car-shadow", "drift-chicane", "camel", "horsejump-high", "bmx-trees", "breakdance"]

SETS = {"dev6": DEV6, "all": None}

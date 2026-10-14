"""certtrack: Certified Anchored Tracking.

Point tracks where every step is matched to the raw-pixel template of the point's
query frame and certified with the multi-scale box test of certskip (THEORY.md
Theorem 3 of certified-skip): while a step is certified, no feature of contrast
Delta_r containing a (2r+1)-box appeared or vanished in the template footprint.
Sub-pixel refinement is reported, not certified.  Steps that fail go to a frozen
learned tracker whose proposal is re-verified against the same template.

Layout
------
data        TAP-Vid pickle, DAVIS JPEG and raw footage loaders; tapnet query sampling
metrics     tapnet's TAP-Vid metrics (verbatim port) and a vectorised twin
longmetrics survival, median trajectory error, re-detection AJ, certified fraction
track       template, camera prior, search, refinement, certificate, anchors, tracker
models      frozen trackers behind one protocol (CoTracker3, chained LK, fake)
methods     the methods benchmarked side by side, one step() API
harness     run a method on a clip, collect metrics, write out/<stem>.{json,txt}
render      proof videos
"""
__version__ = "0.0.1"

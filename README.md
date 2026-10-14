# Certified Anchored Tracking

Point tracks where a step can carry a proof. Every query point keeps the raw-pixel template
of its query frame. On each new frame the template is matched and the match is **certified**
with the multi-scale box test of [certified-skip](https://github.com/nsquaredzz/certified-skip)
(THEORY.md Theorem 3 there): while a step is certified, no feature of contrast Δ_r that contains
a (2r+1)-box appeared, vanished, split or merged in the template's footprint, up to one
brightness offset. A certified step is anchored to the query frame, so its error does not
accumulate with time. Steps that do not certify are handed to a frozen learned tracker
(CoTracker3), whose proposal is itself checked against the same template before it is believed.

Status, 2026-10-14: scaffold, metrics, baselines and the certify-only tracker exist and are
measured; the hybrid is being measured. Numbers below are from `out/` and are reproduced by the
commands shown. Nothing here is a benchmark claim yet.

## What is measured so far

CoTracker3 online (public weights, joint queries, 256x256) on the 30 TAP-Vid DAVIS clips, on
an Apple M4 with MPS: **AJ 64.5 / δ_avg 77.2 / OA 90.8** in 142 s (`out/bench_davis_cotracker_all.txt`;
the paper's single-point protocol gives 63.8).

The certify-only tracker on the six fixed development clips (`out/e1_davis_dev6_*.txt`):

| template | Δ_0 | certified share of visible frames | past gap 30 | certified steps within 4 px | certified while occluded |
|---|---|---|---|---|---|
| 21 px | 80 | 10 % | 1-3 % | 93-100 % | 0 % |
| 21 px | 128 | 19-22 % | 12-16 % | 80 % | 13 % |
| 13 px | 128 | 41-65 % | 37-60 % | 39-55 % (90 % when the self-match set is {0}) | 18-21 % |

The search is not what limits coverage: certifying at the ground-truth position gives the same
10-12 % for the 21 px template. Fitting a similarity or affine warp before certifying adds about
two points. On these clips the certificate is **sparse and precise**: when it fires it is right,
and it fires on one frame in ten. Whether that is useful is the question the hybrid and the
long-video runs answer.

## Reproduce

```
python3.12 -m venv .venv && .venv/bin/pip install -e ../certified-skip -e '.[dev,model]'
.venv/bin/python scripts/fetch_data.py            # TAP-Vid DAVIS 1.67 GB, RGB-Stacking 187 MB, SHA-256 checked
.venv/bin/python scripts/fetch_weights.py         # CoTracker3 weights 2 x 100 MB, pinned source checkout
.venv/bin/python scripts/prepare_tapvid.py        # per-clip files at 256x256
.venv/bin/python -m pytest -q
.venv/bin/python scripts/bench_tapvid.py --method cotracker --set all
.venv/bin/python scripts/e1_certified_fraction.py --set dev6
.venv/bin/python scripts/bench_tapvid.py --method hybrid-verify --set dev6 --side 21 --delta0 80
```

`certified-skip` must be checked out beside this repository; it supplies the certificates
(`certskip.scalespace`, `certskip.warpcert`, `certskip.topology`).

## Layout

```
python/certtrack/
  data/        TAP-Vid pickle -> per-clip npz; tapnet's query sampling verbatim; DAVIS and footage readers; palindromes
  metrics.py   tapnet's compute_tapvid_metrics verbatim, plus wrappers
  longmetrics.py  survival, median trajectory error, re-detection AJ, certified fraction, proof precision
  track/       template (gradients, kappa, self-match set), camera prior, integer search, certificate, warp fit, tracker
  models/      CoTracker3 behind one protocol; a fake tracker for tests
  methods.py   carry-only, hybrid-verify, frozen-only
  harness.py   run and score; out/<stem>.{json,txt} with machine and call log
  render.py    proof videos
scripts/       fetch_data, fetch_weights, prepare_tapvid, bench_tapvid, bench_long, e1_certified_fraction, render_proof
tests/         metrics hand cases, loaders, certificate invariants, synthetic end to end
```

## What is new, what is not, what is missing

**Not new.** Lucas-Kanade (1981) and Shi-Tomasi (1994), including their check against the
first-frame template; Baker and Matthews' inverse-compositional fit (2004); Matthews, Ishikawa
and Baker, "The template update problem" (2004), which re-aligns to the first template and
rejects drift past a threshold, the direct ancestor of anchoring; block matching and pyramids;
failure detection and re-detection in MOSSE and TLD; MFT (2024); learned verifiers for tracker
output (2026); the stability theorem for persistence diagrams (2007); send-on-delta.

**New, to my knowledge.** A per-step certificate on raw pixels for a point track, with a
topological reading and a tightness construction, evaluated at integer positions so no
interpolation term enters; the exact brightness offset for the multi-scale test (a closed form,
where the midrange fails); the self-match set of a template as the object that bounds position
error under a certificate, and its use as a gate; the bookkeeping that says which proofs survive
re-anchoring.

**Missing / not claimed.** Identity under repetitive texture or look-alikes (the certificate is
existence, not uniqueness); deformation, out-of-plane rotation and lighting beyond gain and
offset; any guarantee while a point is model-held or occluded; benchmark leadership on short
clips; long real footage with ground truth on this machine (RoboTAP is the venue; PointOdyssey
is synthetic and too large).

## Licence

MIT for this code. TAP-Vid, DAVIS, RGB-Stacking and the CoTracker3 weights stay under their own
terms and are fetched from their owners' servers, never redistributed here.

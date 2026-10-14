# Results

Every number here is in a file under `out/` with the machine, the convention and the
call log that produced it. Mac M4, 16 GB, MPS for CoTracker3, numpy on the CPU for the
certificate. TAP-Vid DAVIS at 256x256, query-first, CoTracker's pixel convention
(normalised coordinate 1 maps to 255), metrics by tapnet's `compute_tapvid_metrics` kept
verbatim. "dev-6" is the fixed development set car-shadow, drift-chicane, camel,
horsejump-high, bmx-trees, breakdance, chosen before any run.

## Part I — the baseline and the certificate alone (2026-10-14)

### CoTracker3 online, public weights

| set | AJ | δ_avg | OA | time |
|---|---|---|---|---|
| all 30 clips, joint queries | 64.5 | 77.2 | 90.8 | 142 s |
| dev-6 | 71.9 | 85.1 | 91.5 | 27 s |

`out/bench_davis_cotracker_all.txt`, `out/bench_davis_cotracker_dev6.txt`. The paper reports
63.8 with one query per run plus a support grid; joint queries are what the hybrid uses, so
they are the baseline throughout.

### E1: how much of real footage the certificate covers, and whether a certified step is in place

The certify-only tracker (`scripts/e1_certified_fraction.py`): template of side s from the
query frame, camera prior, integer search within 16 px, certificate with the schedule
Δ_r = Δ_0 (2r+1)^-1/2 over r = 1, 2, 3 and the exact brightness offset, sub-pixel pose by a
translation, similarity or affine fit, the certificate evaluated on the frame resampled at
that pose or on the integer crop, whichever scores lower. Columns: certified share of
counted frames whose ground truth is visible, the same past gap 30 from the query, the share
of certified steps within 4 px of the ground truth, that share restricted to templates whose
self-match set is {0}, and the share of certified steps whose ground truth says occluded.

| s | Δ_0 | fit | certified | past gap 30 | within 4 px | within 4 px, S = {0} | certified while occluded |
|---|---|---|---|---|---|---|---|
| 21 | 80 | translation | 11.0 | 1.8 | 94.1 | 100.0 | 0.0 |
| 21 | 96 | affine | 16.1 | 7.4 | 88.4 | 100.0 | 4.7 |
| 21 | 128 | affine | 22.9 | 17.3 | 81.5 | 86.7 | 13.3 |
| 21 | 160 | affine | 30.0 | 23.9 | 71.9 | 100.0 | 16.9 |
| 17 | 128 | affine | 31.4 | 24.9 | 66.4 | 97.2 | 13.5 |
| 13 | 128 | translation | 44.1 | 39.9 | 55.2 | 89.5 | 19.2 |
| 25 | 96 | affine | 11.8 | 3.4 | 94.0 | 100.0 | 0.0 |

`out/e1_davis_dev6_quick.txt`, `out/e1_davis_dev6_groups.txt`. Three things the table says:

1. The certificate is sparse. At the settings where it is precise (21 px or wider, Δ_0 ≤ 96)
   it covers one counted frame in ten and almost nothing past gap 30. Loosening Δ_0 or
   shrinking the template buys coverage and pays in certified steps that are in the wrong
   place and in steps certified while the ground truth is occluded.
2. The search is not the limit. Certifying at the rounded ground-truth position
   (`out/e1_davis_dev6_oracle.txt`) gives 10-12 % for the 21 px template at Δ_0 = 80 and
   38-40 % for the 13 px template at Δ_0 = 128, the same as the tracked numbers. What limits
   coverage is appearance change the group cannot explain: deformation, blur, lighting,
   perspective. A similarity or affine fit adds one to three points over translation.
3. The self-match set does what Theorem P says. Among certified steps, those whose template
   certifies against itself at no shift of 2 px or more are within 4 px of the ground truth
   97-100 % of the time at every setting; the rest are at 40-65 %. "Certified in place" has
   operational content exactly on the templates the theorem names.

Against the kill criteria of the plan: the 15 % floor at gap 30 is missed by every precise
setting, so the certificate cannot carry a tracker on its own on this footage; the precision
gate exists, so the word "certified" keeps its meaning. The expected use is therefore as a
verifier beside a learned tracker, not as the tracker.

### E5: the hybrid on short clips

CoTracker3 on every query every frame; the template is searched near the model's proposal
and at the carried prediction; a certified match is reported under a proof. Position policy
from Theorem P: when the certified position and the model's agree within 2 px the model's
sub-pixel position is reported under the certificate; when they disagree the certified
position replaces the model's only for templates with self-match set {0}; otherwise the
model's position stands and the step is model-held. Where nothing certifies the model's
position is reported with its own visibility.

| method, dev-6 | AJ | δ_avg | OA |
|---|---|---|---|
| CoTracker3 alone | 71.9 | 85.1 | 91.5 |
| hybrid, s 21, Δ_0 80, similarity, gated | 71.9 | 85.1 | 91.5 |
| hybrid, same, before the gate and the agreement rule | 71.4 | 84.4 | 91.5 |
| hybrid, s 13, Δ_0 128, ungated | 54.5 | 74.4 | 83.6 |

`out/bench_davis_hybrid-verify_*_dev6.txt`. On 34-104-frame clips the certificate changes
nothing at the precise setting: where it fires the model was already right, and the
agreement rule hands the sub-pixel position back to the model, which is the more accurate
of the two there (on the rigid clips ours is closer 50-73 % of the time; on camel and
breakdance the model is). Ungated, the loose setting is a disaster: thirty certified
impostors on camel alone, at a median 20 px from the truth. The gate removes them.

## Part II — long video (in progress)

Protocol: each dev-6 clip is replayed as a 500-frame "tail palindrome" (one pass forward,
then bouncing inside the last two thirds, so the frames around the query never recur);
ground truth is exact by index. Real footage, artificial time order; it tests drift and
repeated occlusion, not new content.

| method, tail-500, dev-6 | AJ | δ_avg | OA | survival 50 px | survival 16 px | MTE | AJ_RD | end within 16 px |
|---|---|---|---|---|---|---|---|---|
| CoTracker3 alone | 60.5 | 80.1 | 83.7 | 96.1 | 94.1 | 1.16 | 69.4 | 99.3 |

`out/long_davis_tail500_cotracker_dev6.txt`. CoTracker3 does not drift on these sequences:
after 500 frames 99 % of visible points are within 16 px and the median error is 1.2 px. What
it loses over long video is visibility: OA falls from 91.5 to 83.7 (drift-chicane 58.7,
bmx-trees 68.1) and re-detection AJ sits at 57-72. If the certificate has a role here it is
on the visibility flag, in the one direction the theorem allows: certified implies present.

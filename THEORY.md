# Theory

What is proved, under which hypotheses, and what the measurements then showed. The
certificates come from [certified-skip](https://github.com/nsquaredzz/certified-skip);
section numbers there are cited as "CS §n".

## 0. Objects

A query point has a template `T`: the `s x s` luma crop of its query frame at the integer
centre `k_q`, with the query point at offset `o` from that centre. A step at frame `t`
proposes an integer centre `k_t` and a crop `C_t = F_t[k_t + Ω]`, `Ω` the `s x s` grid.
Optionally the crop is the frame resampled bilinearly at a sub-pixel pose (translation,
similarity or affine); then `C_t = F_t ∘ p` is an explicit array too, and every statement
below is about `F_t ∘ p` rather than about raw frame pixels, which the proof names.

The certificate uses box means. For radius `r` let `m_B` be the mean of an array over a
`(2r+1)`-box `B ⊂ Ω`, and `ε_r = Δ_r / 2` with the schedule `Δ_r = Δ_0 (2r+1)^(-1/2)`,
`r ∈ R = {1, 2, 3}`. The score of a crop against the template with offset `c` is

    score(c) = max_r max_B | m_B(C_t − T) − c | / ε_r,

and the step is **certified** iff `min_c score(c) < 1`.

## 1. The offset (new, small, decisive)

`score(c)` is a maximum of `2|R|` affine functions of `c`: for each `r`, `(M_r − c)/ε_r` and
`(c − m_r)/ε_r`, where `M_r` and `m_r` are the largest and smallest box means of `C_t − T`
at radius `r`. A maximum of lines is convex and piecewise linear; its minimum is at a crossing
of a falling and a rising line, so

    c* ∈ { (M_a ε_b + m_b ε_a) / (ε_a + ε_b) : a, b ∈ R },

`|R|²` candidates, evaluated in closed form from `2|R|` numbers. The midrange of the raw
residual is the special case `R = {0}`. On real patches it is a poor choice for `r ≥ 1`: a
few extreme pixels (a specular highlight, a wheel edge) move it far from the bulk and every
box mean inherits that error. Measured on car-shadow, frame 1 after the query, 21 px
templates: 4 % certify with the midrange, 56 % with `c*`
(`out/e1_davis_dev6_quick.txt` and the diagnostic in RESULTS.md Part I).

Soundness does not depend on the choice: whatever `c` is used, the certified statement is
about `T + c`. So the best `c` may be used freely. (CS §9 explains which image a per-patch
offset certifies; here the consumer is a position, which is invariant to the offset, so the
quotient is the right one, CS §12.)

## 2. Theorem V: certified presence (proved)

If the step is certified with offset `c`, then for every `r ∈ R`, writing `β_r` for the
`(2r+1)`-box filter,

    ‖ β_r C_t − β_r (T + c) ‖_∞ < ε_r,

hence (bottleneck stability of persistence diagrams, as in CS Theorem 3) every feature of
`β_r T` with persistence `≥ 2 ε_r = Δ_r`, dark or bright, is matched by a feature of
`β_r C_t` with persistence `> p − Δ_r`, and no feature of persistence `≥ Δ_r` of either side
is unmatched. Both sides are explicit arrays of pixels (or of resampled pixels when a
sub-pixel pose is used), so no interpolation term enters.

What it does not say: nothing about sub-pixel position; nothing about uniqueness (the
template may also be present elsewhere); nothing about features thinner than `3 px` or
fainter than `Δ_r`; nothing about brightness (the offset is quotiented out). It is vacuous
when the template has no feature of persistence `≥ Δ_r` at any `r`: `κ_r`, the number of
such features per scale, is computed once per template and reported with every proof.

## 3. Theorem P: position through the self-match set (proved under H1, H2)

Let `S(η)` be the set of integer shifts `u`, `|u|_∞ ≤ 2R`, at which the query window
certifies against itself at tolerance `1 + η`:

    S(η) = { u : min_c score_u(c) < 1 + η },  score_u the score of `T_q[Ω + u]` against `T_q[Ω]`.

Hypotheses. H1 (brightness constancy on the window): there is an integer `u*` with
`F_t[u* + x] = T̄[x] + n_t[x]` on the window, `T̄` the query crop on the window dilated by
`2R`, and the noise `n_t` has score `≤ η` on every `s x s` sub-window in play. H2: the
true centre is inside the search window, `|k_t − u*|_∞ ≤ 2R`.

Then a certified step at `k_t` satisfies `k_t − u* ∈ S(η)`: by the triangle inequality for
the seminorm, `score(T̄[Ω + u] − T̄[Ω]) ≤ score(C_t − T) + score(n) < 1 + η` with
`u = k_t − u*`. If `S(η) = {0}` the certified integer position is exact and the reported
sub-pixel pose is within one pixel of the truth.

Reading. A corner or blob gives `S = {0}`; a straight edge gives a segment along the edge (the
aperture problem, as a set); periodic texture gives a lattice; a flat template gives
everything. Under deformation, out-of-plane rotation or lighting beyond an offset, H1 is
false and nothing is claimed.

Measured. The tracker reports `n_self`, the number of shifts with `|u|_∞ ≥ 2` in `S(0)`.
Across every setting of the E1 grid, certified steps whose template has `n_self = 0` are
within 4 px of the ground truth 97-100 % of the time; the others 40-65 %
(`out/e1_davis_dev6_groups.txt`). The gate is used in the hybrid's position policy.

## 4. No drift on certified frames (proved, elementary)

Every certified frame satisfies the same inequality against the same template, so the
error set of Theorem P is the same at `t = 1` and `t = 10 000`. A chained tracker, whose
template is refreshed each frame, has error in the Minkowski sum of per-step self-match
sets: a segment per step along an edge, so linear drift along the edge with every step
"certified". Re-anchoring bookkeeping: a new anchor that certifies against the original at
tolerance `1/2` keeps the original statement at tolerance `1` for ever (triangle inequality);
`k` relative anchors at tolerance `τ` give `k τ` against the original; a link through a
learned model's proposal gives nothing.

Measured. On the 500-frame palindromes the certified carry on its own has survival 52 %
(`out/long_davis_tail500_carry_s21_d80_sim_dev6.txt`): it does not drift on certified frames,
it is lost on the others, and loss is the failure mode, as the statement says.

## 5. Fewest model calls (proved; cited, not used)

CS Theorems 13-14 transfer verbatim to "model calls per point" for a tracker that either
certifies a frame against the template it holds or calls the model: no sound tracker calls
fewer than `C_ε` times, and "call when the anchor fails, re-anchor at the model's answer"
calls at most `C_{1/2}` times. A fixed anchor has no competitive bound (permanent
appearance change forces a call every frame while two would do), which is the price of
drift-freedom. With the certificate firing on 5-10 % of frames (RESULTS Part I) the saving
available to any gate is bounded by that share, so the compute axis was not pursued.

Note for the certified-skip paper draft: Theorems 13-14 already give `m_Δ ≤ OPT_{Δ/2}`,
sharper than Theorem D(ii)'s `2 OPT_{Δ/2} + 1`; cite 13-14 and keep only D(iii).

## 6. Why the hybrid cannot beat its backbone here (measured, then explained)

CoTracker3 reproduces to the decimal under the hybrid on short clips and on 500-frame
palindromes (RESULTS Parts I-II). The certificate never fired on a frame where the model was
wrong. The explanation is in what each is anchored to: CoTracker3 carries, for each track,
the feature sampled at its query frame and matches it with learned correlation in a 16-frame
window, so it is anchored to the query appearance as the template is, but in a feature space
that tolerates deformation, blur and lighting; the raw-pixel certificate tolerates one warp
and one offset. Both fail under permanent appearance change; the certificate fails first,
and the model's remaining errors on this footage are visibility flags on frames whose
appearance has already left the certificate's reach. The certificate's one-sided statement
about visibility, certified implies present, therefore never contradicts the model.

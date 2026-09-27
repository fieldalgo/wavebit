# The per-order s/p frame the complex amplitudes are quoted in

Written: 2026-08-16.
Targets: `wavebit/_core/conventions.py`, `wavebit/_core/result.py`,
`wavebit/_core/solve_1d.py`, `wavebit/_core/scattering.py`, `wavebit/_core/field_result.py`,
`tests/test_amplitude_sp_frame.py`, `tests/test_order_azimuth.py`,
`tests/test_golden_legacy_parity.py`, `tests/test_scattering_core.py`,
`tests/test_excitation.py`

**TL;DR.** Every order with `Re(kx) < 0` had its complex amplitudes `r_s` / `r_p` / `t_s` /
`t_p` reported in a reversed frame -- `-s_hat`, `-p_hat` -- because `order_azimuth` built the
frame from `atan(ky/kx)`, and a ratio has already discarded `sign(kx)`. A half-turn negates
`cos` and `sin` together, which makes it a diagonal `+-1` similarity on the whole stack
algebra: it cancels out of every efficiency, out of energy conservation, out of the
reconstructed field, and out of every external golden this project has. It survived in exactly
one observable, the sign of the reported amplitudes, and nothing in the suite compared those
across orders. Fixed by adding back the sign the ratio dropped; `R` and `T` are unchanged to
the bit on every case measured.

The origin (`kx == ky == 0`, the zeroth order at exactly `theta = 0`) is the same spelling with
`cos(phi)` in place of `kx`, and had the same defect: `phi` and `phi + pi` landed on one frame,
so a solve handed `phi + pi` quietly answered for `phi`. Fixed in a second commit -- Sec. 4,
which is also the record of why it was mistaken for a convention for one round.

**2026-09-27: the pure 1D path joined the frame** (Sec. 6). The fix above moved only the
coupled paths; the pure TE/TM path had quoted its Cartesian primary (`E_y` / `H_y`) on every
order all along, a sign apart from `polarization=(1, 0)` / `(0, 1)` at `phi = 0` on every order
across `kx = 0` from the incident one. It now uses the same frame for its amplitudes, its
`excitation` weights, its scattering matrix and its incident wave.

---

## 1. What was wrong

`order_azimuth` supplies `cos(varphi)` and `sin(varphi)`, which rotate each order into its own
s/p frame. The intended frame is the standard one, `s_hat = z_hat x k_hat_par`, so
`(cos, sin)` must be the *direction* of `(kx, ky)`. It was computed as `atan(ky/kx)`, whose
input is a ratio: `(kx, ky)` and `(-kx, -ky)` produce the same answer, so the result is wrong
by `pi` over the whole `Re(kx) < 0` half-plane. `atan2` is the function that keeps the sign.

The spelling was inherited rather than chosen. Before 2026-08 a `perturbation` constructor
argument replaced `theta == 0` with `1e-20`, so `atan` recovered the singular points on its
own; when that knob was removed the singular points were written out explicitly and the
`atan` was carried over as-is. The `Re(kx) < 0` half-plane was never the subject of that
change and nobody looked at it.

## 2. Why nothing caught it

A half-turn reverses an order's basis vectors *and* its coefficients together. Writing the
reversal as a diagonal `D = diag(+-1)`, every layer operator becomes `D M D`, and the products
the stack algebra forms are `D M1 D * D M2 D = D M1 M2 D` -- the reversal telescopes and
appears only at the two ends. So:

| quantity | sees it? | why |
| --- | --- | --- |
| `R`, `T`, `R_s`, `R_p`, `T_s`, `T_p` | no | `|r|**2` |
| energy conservation | no | same |
| the reconstructed field | no | basis vectors and coefficients reverse together |
| RETICOLO / fmmax / S4 goldens | no | all compare efficiencies, or fields, or 1D amplitudes |
| `r_s`, `r_p`, `t_s`, `t_p` | **yes** | the one place a frame is not contracted away |

The external goldens are worth naming individually, because "we have four oracles" is what
made this feel covered. `golden_reticolo.npz` and `golden_s4.npz` / `golden_fmmax.npz` store
efficiencies. The field goldens store fields, which the table above says are blind by
construction. `golden_reticolo_conical_sweep.npz` *does* store a complex amplitude, and it is
the closest miss: it compares `r_s` of the **incident order only**, and its harness builds
`phi` with `arctan(ky/(kx + 1e-4))`, so every point it solves has `kx_inc > 0`. It could not
have reached the half-plane.

## 3. The three measurements

Case throughout: a chiral 16x16 cell (a bar with two arms of different index, no mirror line
in either axis), `period = [500, 800]`, `truncation_order = [5, 3]`, `theta = 25 deg`, `wl = 250`,
`factorization='nvf'`, so 18 orders propagate and the retained grid straddles `kx = 0` with 35
orders on one side and 42 on the other. NumPy, complex128.

**(a) The transposed twin.** Reflecting the whole scene about `x = y` -- transpose the cell,
swap the period pair, swap the `truncation_order` pair, `phi -> pi/2 - phi`, `a_s -> -a_s` -- is the same
physical problem, so the two solves must agree entry for entry.

| quantity | before | after |
| --- | --- | --- |
| `R`, `R_s`, `R_p`, `T_s`, `T_p` | 1.4e-15 .. 2.9e-15 | 1.3e-15 .. 3.5e-15 |
| `abs(r_s)` | 1.3e-14 | 1.1e-14 |
| `r_s` | **4.4e-01** | 1.1e-14 |
| `r_p` | **6.8e-01** | 1.6e-14 |
| `r_s + r_s'` (a global sign?) | 5.4e-01 -- so, no | -- |

`arg(a/b)/pi` printed as exactly `0` or `+-1` over the whole grid, and the pattern was exactly
`sign(kx) != sign(ky)`. That is the reflection's signature: it maps the set `{kx < 0}` onto
`{ky < 0}`, so the two solves disagree precisely on the quadrants where those differ. This is
why a transpose exposes a half-plane error and a rotation or a translation does not.

**The hypothesis this kills.** Amplitudes are legitimately origin-sensitive where efficiencies
are not, so a residual translation left behind by the recipe was the first thing to rule out.
Rolling the same cell 7 pixels in x, on `factorization='li'` where a lattice translation is
exact:

| | roll 7 px in x |
| --- | --- |
| `R` | 1.8e-15 |
| `abs(r_s)` | 9.1e-15 |
| `r_s` | 3.9e-01 |
| `arg(a/b)/pi` over the 77 orders | -0.875 .. +0.875, **11 distinct values** |

Eleven values, one per column of `m`, because a translation by `d` multiplies order `m` by
`exp(i k*d)` -- a per-order *phase*. It is `+-1` only when `d` is half a period, and then it
is `(-1)**m`, a function of the order *index* alternating column by column. The transpose's
mask is neither: it takes exactly two values, and it is a function of the order's
*wavevector* -- it changes sign where `kx` and `ky` cross zero, and those crossings move when
`theta`, `phi`, `wl` or the period move, which no lattice translation's mask can do.

(`li` rather than `nvf` for this control on purpose, *at the time*. The NVF path was not
translation-covariant -- the same roll moved `R` by 1.2e-02 and it did not converge away with
`truncation_order` -- so it could not isolate a translation. That was a separate, pre-existing property of
the normal-field construction, unrelated to the frame, and it predated every change described
here. **It was in fact a defect, and was fixed on 2026-08-17** (`339df21`, `5b4c891`): the cell
is periodic and three places in the construction read it as a flat array. The same roll now
moves `nvf` by 9.1e-15 against `li`'s 9.6e-15, so either path can carry this control today.
`li` is kept here because it is what the measurement was made on.)

**(b) The reconstructed field, as the arbiter.** Section 2 says the field cannot see the reversal,
which is exactly what qualifies it to judge: it is right either way, and it is pinned entry
for entry against S4 (1e-15) and fmmax. Transverse `E` is continuous across the top interface,
so its Fourier coefficient at order `(m, n)` is the superstrate's -- the incident wave, which
lives only at `(0, 0)`, plus the reflected amplitude at its own phase reference. Decomposing
the top face onto the retained orders (residual 3.1e-15) and projecting onto
`s_hat = z_hat x k_hat`. (The top face was `z = 0` when this was measured; since the
upward-positive frame of 2026-09-21 it is `z = sum(thickness)`, and `z = 0` is the bottom one.
The test kept requesting `z = 0` and from then on decomposed the *transmitted* field, which
reported a frame error of 0.36 that the amplitudes did not have -- they were byte-identical
across the change. The test now checks the plane before any order, so a moved plane names
itself instead of impersonating a sign.)

| | `Re(kx) > 0` orders (41) | `Re(kx) < 0` orders (35) |
| --- | --- | --- |
| `abs(E_s - r_s)` | 2.5e-15 | **5.4e-01** |
| `abs(E_s + r_s)` | -- | 1.4e-15 |

with **one** scale fitted across the whole grid, which came out `1.000000 + 0.000000j` -- so
the fit is not absorbing anything, and the `Re(kx) > 0` row doubles as a statement that `r_s`
*is* the field amplitude against a unit incident one. The p channel says the same after its
own documented `cos(theta_m)` factor: `Ep / (r_p * kz_m)` was `sign(kx)` on every order.

**(c) Continuity.** At `phi = 0` order `m = -1` has `kx = sin(theta) - wl/Px`, which crosses
zero at `theta = 30 deg` exactly. Nothing physical happens there -- the order stays propagating
and far from a Rayleigh point:

| `theta` | `kx(-1)` | `R` | `abs(r_s)` | `Re r_s` | `Im r_s` |
| --- | --- | --- | --- | --- | --- |
| 29.999 | -1.5e-05 | 0.014712928 | 0.113772572 | +0.0498469 | -0.1022716 |
| 30.001 | +1.5e-05 | 0.014704089 | 0.113892170 | **-0.0487801** | **+0.1029171** |

`R` and `abs(r_s)` sail through; `r_s` flips. A convention is allowed to be surprising. It is
not allowed to be discontinuous in a continuous input, and that is what settles "reversed
frame" as a defect rather than as an unstated choice.

That argument holds where it was measured, and the qualifier matters: the order is
`(m, n) = (-1, +1)`, with `ky != 0`, so its `k_par` never vanishes and `z_hat x k_hat_par`
turns smoothly through `+-x_hat` as `kx` crosses zero. On the `ky == 0` row `k_par` passes
*through* zero, the frame has no limit there, and a sign step at an order's own `kx = 0` is the
frame's singular point by design, on every path -- the pure 1D one included since 2026-09-27
(§6). A step there is not a defect, and "fixing" it would take the frame away from the rest of
the row.

There is a fourth statement, free and internal: the function disagreed with **its own special
case**. The `kx == 0` column three lines below took its sign from `Re(ky)`, which is the
`z_hat x k_hat` rule, while the `Re(kx) < 0` half-plane did not -- so at `kx = -1e-12` it
returned `sin(varphi) = -1` and at `kx = 0` it returned `+1`.

## 4. The origin, which is the same bug

`kx == ky == 0` -- the zeroth order at exactly `theta = 0`, where the in-plane wavevector
carries no direction and `phi` supplies one -- was filled with `atan(sin(phi)/cos(phi))`. Same
`atan`-of-a-ratio spelling, same missing sign: at the origin the "`kx`" whose sign the ratio
discards *is* `cos(phi)`. So `phi` and `phi + pi` landed on one frame.

This was fixed one round later than the half-plane, because it looked like it bought something
-- `phi` and `phi + pi` giving the same answer, which two tests pinned with an argument. The
argument was that turning the reference azimuth through `pi` flips `s_hat` and `p_hat`
*together*, so the incident field and the scattered field pick up the same sign and it cancels
in every ratio. It cancels for the zeroth order. **It does not cancel for any other**, because
at `theta = 0` the order grid is `kx_m = m*wl/Px`, `ky_n = n*wl/Py` -- which does not contain
`phi` at all, so no other order's frame turns with it.

**The measurement that settles it.** The incident field at `theta = 0` is
`a_s*s_hat(phi) + a_p*p_hat(phi)`, so `(phi + pi, psi)` and `(phi, polarization=(-a_s, -a_p))`
are the *same physical incident field* written two ways -- and every order but `(0, 0)` is
reported against an identical basis in both. 8x8 cell, `truncation_order = [4, 4]`, `psi = 0.7`, `phi = 0.3`:

| | before | after |
| --- | --- | --- |
| the two spellings, 80 off-centre orders | **2.8e-01** apart (1.2e-15 after negating) | 1.6e-15 |
| the two spellings, `(0, 0)` | -- | a sign, which is correct: that order's frame is the one that turned |
| the two spellings, reconstructed field | -- | 3.7e-14 on a cell scale of 4.26 |
| `R` | 2.8e-16 -- agreed throughout, and hid all of it | unchanged |

Negating `(a_s, a_p)` negates every reported amplitude, bit-exactly (`0.00e+00`) -- the library
keeps overall phase, as `conventions.md` says it does. Turning `phi` by `pi` is the same
operation on the same field. The two disagreeing is a contradiction, not a convention.

The field says it most plainly, and it is what the second rewritten test now asserts: with
`phi + pi` the reconstructed cell must come back **negated** (`|a + b| = 3.7e-14` against a
scale of 4.26). It used to come back identical, which is a solve quietly answering for `phi`
when it was handed `phi + pi`.

Continuity falls out for free: sweeping `phi` through `pi/2` at `theta = 0` used to flip every
order's `r_s` except the zeroth (`2.3e-01`) while `R` sailed through; it is now `2.3e-10`,
which is the `1e-9` step in `phi` itself.

## 5. What the fix touches

One term in `order_azimuth`, keyed on `Re(kx)` rather than on a norm so that the `kx == 0`
column -- including its purely-imaginary-`ky` corner, where the branch is a genuine choice and
not a mistake -- comes out bit-identical to before.

Blast radius, measured by running the suite:

* the six pure-1D cases of `test_golden_legacy_parity` were untouched: the TE/TM formulation
  carried no s/p rotation at all -- until 2026-09-27, when it took this frame too (Sec. 6) and
  `_frame_sign` began converting those six as well;
* all seven coupled cases (`conical`, `conical_2l`, and the five `2d_*`) move by the per-order
  sign on their amplitudes, and by nothing at all on their efficiencies. Handled as a
  convention conversion at the comparison rather than by refreezing the golden -- the legacy
  values stay the legacy solver's own, see `_frame_sign` there;
* no external golden moves, for the reason in Sec. 2.

**This is a breaking change to a documented output.** A caller who read `r_s` / `r_p` / `t_s` /
`t_p` on a 2D or conical solve and compensated for the old sign will need to stop.

## 6. The pure 1D path joins the frame (2026-09-27)

**The decision** (the maintainer's, option B of the 2026-09-27 investigation): one meaning for
`r_s` / `r_p` / `t_s` / `t_p`, for an `excitation` weight, for the scattering matrix's ports and
for the incident polarization, on every path. Until then the pure 1D `tetm` path quoted `E_y`
(TE) or `H_y` (TM) on every order, while the conical and 2D paths quote each order along its own
`s_hat = z_hat x k_hat_par`. On the `ky == 0` row that vector is `sign(Re kx_m) * y_hat`, so the
two disagreed on every order whose `kx` sits across zero from the incident order's:

    new 1D amplitude(m) = sign(Re kx_m) * sign(Re kx_inc) * old 1D amplitude(m)

-- exactly, with `sign(0) = +1` (the origin rule at `phi = 0`). The alternative, keeping the
Cartesian frame and documenting it, was the investigation's own recommendation; it was not
taken because a quantity whose sign means one thing on one path and another on the next is the
trap Sec. 2 describes, and `excitation` weights inherited it as different *light*, not only a
different label.

**What moved, on the pure 1D path only** (`order_frame_sign` in `wavebit/_core/conventions.py`
is the rule; `solve_1d`, `smm_1d` and `field_result` apply it):

* the amplitudes, by the relation above;
* the incidence: a single `tetm` wave is now `s_hat_inc = sign(kx_inc) * y_hat` (TE) or the
  matching `p_hat_inc` (TM), so at `theta < 0` the field is negated -- `solver.T1`,
  `solver.reconstruct_field` / `solve_field` / `solve_field_eager` / `plot_field`, and
  `result.reconstruct_field` / `boundary_fields` all move with it;
* the TM input is a unit incident **E**, not a unit `H_y` (the first commit of the change): every
  TM field above is multiplied by `n_inc/mu_inc`;
* an `excitation` weight multiplies its order's own `s_hat` / `p_hat` rather than `+y` / `+H_y`.
  This is a **physics** change wherever the lit set straddles `kx = 0`: on
  `tests/test_excitation.py`'s scene `sum(R)` goes from 0.493324 to 0.370091 (TE) and from
  0.217085 to 0.158687 (TM), now the conical path's numbers. Lit sets on one side of `kx = 0`,
  and a single lit order, are unchanged;
* the scattering matrix is published in the coupled path's port variables: TE `D S D` of the
  old one in all four blocks; TM `-D S D` on `S11` / `S22` and `D S D` on `S12` / `S21`, with
  `D = diag(sign(Re kx))`. The extra minus is the coupled p variable's own -- the p wave's
  magnetic field, which turns over for the up-going wave -- so the 1D matrix is now exactly the
  conical one's s-s or p-p block, and `S11 @ delta` is `-r_p` on both. That reading was already
  true of the coupled p block and was documented as `r_p`; the docs now say `-r_p` everywhere.

**What did not move, proved rather than argued.** The tree before the change, exported from
`6d721d2`, against the tree after, in fresh processes: 1866 arrays over 1D TE/TM x both sides x
`theta` in {0, +-0.3} x ETM/SMM x NumPy/JAX/Torch, 1D `excitation`, magnetic half-spaces,
conical, 2D and 4ff, each dump taken twice. Every conical, 2D and 4ff array is byte-identical;
every 1D efficiency and `power_entering` is byte-identical; every 1D amplitude equals
`sigma_m * sigma_inc * old` and every field `sigma_inc * old` by `==` (bytes differ only where a
zero took a sign); every 1D `S` equals `flip * D * old * D`. The TM unit-E step alone moves
fields by `n/mu`, to 1.3e-15 normwise, and nothing else.

**The price, stated where a user will look for it.**

* On the `ky == 0` row an amplitude now changes sign where its own `kx` crosses zero, on the pure
  path as on the conical one: `r_s(-1)` steps by about 0.3 -- twice its magnitude -- across
  `theta = asin(lambda/P)` on the investigation's grating, while `R(-1)` moves no more across
  the crossing than across a control step of the same size. That is the frame's singular
  point, not physics (Sec. 3's qualifier); `test_amplitude_sp_frame.py` §4 pins it as a step.
* At `theta = 0` the incident order is on that point: as `theta` crosses zero every amplitude
  and the field change sign together, and on a mirror-symmetric grating at normal incidence
  `r_{-m} = -r_{+m}` where the Cartesian frame gave `+`. A sweep through normal incidence that
  wants a continuous field multiplies by `sign(Re kx_inc)`.
* An order whose `kx` is zero in exact arithmetic -- normal diffraction: a direct
  `kx = 2*pi/P`, or `theta = asin(lambda/P)`, puts order `-1` there -- sits on the singular
  point itself, and the frame it gets is decided by rounding. On this grating the same input
  gives `kx(-1) = +0.0` on NumPy and JAX and `-5.55e-17` on Torch, whose `float / tensor` is
  `tensor.reciprocal() * float` -- one rounding more in `wavelength / period` -- so that
  order's `r_s` (TE) or `r_p` (TM) comes back with the opposite sign on Torch; measured, and
  the same on the conical path, which keyed its frame on the same `Re(kx)` before this change.
  The pure path's Cartesian frame used to give one sign on all three backends. Two spellings
  of the same angle can split the same way: on Torch, `theta = asin(lambda/P)` gives
  `kx(-1) = -5.55e-17` and `theta = asin(0.375)` gives `+0.0`. The order's `R` / `T` do not
  depend on any of it. Making `_separable_kxy` byte-identical across backends, or
  snapping `|Re kx|` to zero within a few ulp, would settle it and would move the conical path
  too; that is a separate decision, left open.
* RETICOLO's own 1D and conical amplitudes differ by exactly `sign(kx_m) * sign(kx_inc)`
  (measured on V10). Wavebit's 1D amplitudes matched RETICOLO's 1D path before and match its
  **conical** path now; a per-order comparison against RETICOLO's 1D path applies the relation
  at the comparison, as `_frame_sign` does for the legacy golden. No committed reference
  compares a non-specular 1D amplitude today -- the complex-`kx` golden's `r0` is specular.
  RETICOLO's 1D TM *field* input is the incident `H_y` as well, which the unit-E input matches
  only in air -- every committed case is lit from air.

**Open, and the maintainer's call: the lateral phase reference.** Every ordinary amplitude, and
every `excitation` weight, is referenced at the cell **corner** -- the Fourier origin -- while
`PlaneWave`, `OrderExcitation` and every public coordinate since 2026-08-31 use the cell
**centre**. Against the centre, order `m` carries `exp(1j * k_m . (a1 + a2)/2)`, a relative
`(-1)**m` between orders on a rectangular cell. So the same weights through `excitation` and
through `OrderExcitation` are different light once two orders are lit (0.452653 against
0.151782 in `sum(R)` on the scene above at `phi = 0`, `{0: 1, 1: 0.6-0.8j}`; negating the odd
weight on the coherent side restores 0.452653). This change leaves that as it was and documents
it (`conventions.md`, `multimode-incidence.md` §6). Moving `excitation` to the centre would be a
second physics change for the same users, and moving the amplitudes would move every order on
every path by that phase -- a record of its own.

**This is a breaking change to a documented output.** A caller who read `r_s` / `r_p` / `t_s` /
`t_p` on a `tetm` solve, its field at `theta < 0` or in TM, its scattering matrix, or its
`excitation` physics, gets the conical path's answer now. The efficiencies of a single-wave
solve did not move.

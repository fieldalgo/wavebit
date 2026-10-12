# The per-order s/p frame the complex amplitudes are quoted in

Written: 2026-08-16.
Updated: 2026-09-27 (Sec. 6: the pure 1D path joins the frame; `r_xyz` / `t_xyz`. Sec. 7: the
lateral phase reference moves to the cell centre).
Targets: `src/wavebit/_core/conventions.py`, `src/wavebit/_core/result.py`,
`src/wavebit/_core/solve_1d.py`, `src/wavebit/_core/scattering.py`, `src/wavebit/_core/field_result.py`,
`src/wavebit/_core/engine.py`,
`tests/test_amplitude_sp_frame.py`, `tests/test_order_azimuth.py`,
`tests/test_golden_legacy_parity.py`, `tests/test_scattering_core.py`,
`tests/test_excitation.py`, `tests/test_cartesian_amplitudes.py`,
`tests/test_centre_reference.py`

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
`excitation` weights, its scattering matrix and its incident wave; and every result carries
`r_xyz` / `t_xyz`, the same waves as Cartesian E vectors, for a quantity no frame can turn over.

**2026-09-27, the same day: one lateral phase reference** (Sec. 7). Every per-order phase --
amplitudes, `excitation` weights, the scattering matrix's ports, the point where an ordinary
incidence is unit -- was referenced at the cell corner, the kernels' Fourier origin, while
`PlaneWave`, `OrderExcitation` and every public coordinate used the cell centre. All of it is
at the centre now: an amplitude on an order with `m + n` odd changed sign, the two excitation
doors light the same light with the same weights, and an ordinary solve's field is the
`PlaneWave` field of its angles. `R` and `T` of a single wave are unchanged to the bit.

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

**What moved, on the pure 1D path only** (`order_frame_sign` in `src/wavebit/_core/conventions.py`
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
  and a single lit order, are unchanged. (Sec. 7 moved these numbers again the same day, to
  0.195292 and 0.103900: the weights' lateral reference, not their frame);
* the scattering matrix is published in the coupled path's port variables: TE `D S D` of the
  old one in all four blocks; TM `-D S D` on `S11` / `S22` and `D S D` on `S12` / `S21`, with
  `D = diag(sign(Re kx))`. The extra minus is the coupled p variable's own -- the p wave's
  magnetic field, which turns over for the up-going wave -- so the 1D matrix is now exactly the
  conical one's s-s or p-p block, and `S11 @ delta` is `-r_p` on both. That reading was already
  true of the coupled p block and was documented as `r_p`; the docs now say `-r_p` everywhere.

**What did not move, proved rather than argued.** The tree before the change, exported from
`6d721d2`, against the tree after, in fresh processes: 1866 arrays over 1D TE/TM x both sides x
`theta` in {0, +-0.3} x ETM/SMM x NumPy/JAX/Torch, 1D `excitation`, magnetic half-spaces,
conical, 2D and first-order, each dump taken twice. Every conical, 2D and first-order array is
byte-identical; every 1D efficiency and `power_entering` is byte-identical; every 1D amplitude
equals `sigma_m * sigma_inc * old` and every field `sigma_inc * old` by `==` (bytes differ only
where a zero took a sign); every 1D `S` equals `flip * D * old * D`. The TM unit-E step alone moves
fields by `n/mu`, to 1.3e-15 normwise, and nothing else. The battery's dumps and the
investigation's record are not public; what they measured is stated here.

**The price, stated where a user will look for it.**

* On the `ky == 0` row an amplitude now changes sign where its own `kx` crosses zero, on the pure
  path as on the conical one: across `theta = asin(lambda/P)` on the investigation's grating
  `r_s(-1)` steps by 0.20 (TE) and `r_p(-1)` by 0.21 (TM), twice their magnitudes. That is the
  frame's singular point, not physics (Sec. 3's qualifier); `test_amplitude_sp_frame.py` §4
  pins it as a step. `R(-1)` does not turn over, but on this grating it is not smooth there
  either: `n_bot * P / lambda = 4`, so at the same angle orders `+3` and `-5` graze the
  substrate, and that Rayleigh point makes `R(-1)`'s step across the crossing 1.4x (TE) and
  1.8x (TM) the larger of two equal control steps either side (2e-5 in `theta`). Off the
  coincidence, at `lambda = 0.62`, it is 1.00 of the control, TE and TM, while `r_s(-1)` /
  `r_p(-1)` still step by twice their magnitudes (0.18 / 0.16).
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
  at the comparison, as `_frame_sign` does for the legacy golden. Since Sec. 7 they also differ
  from either RETICOLO path by `(-1)**m`: RETICOLO's origin is the cell corner, and its field
  rows carry `exp(-1j k0 sin(theta) P/2)` against this library's for the same reason. No
  committed reference compares a non-specular 1D amplitude today -- the complex-`kx` golden's
  `r0` is specular, where both factors are 1.
  RETICOLO's 1D TM *field* input is the incident `H_y` as well, which the unit-E input matches
  only in air -- every committed case is lit from air.

**The quantity that does not step: `r_xyz` / `t_xyz`.** Every result carries each outgoing
order's E as `(Ex, Ey, Ez)`, `r_s * s_hat + r_p * p_hat` in the public frame, shape
`r_s.shape + (3,)`. It is the E half of the boundary spectra `field_result.ordinary_spectra`
builds for `boundary_fields` -- one arithmetic, so no new copy of the s/p -> xyz rule, and
every existing field output byte-identical across its arrival. (It shared `r_s`'s corner
reference when it arrived; since Sec. 7 both are referenced at the centre, and the spectra
need no lateral phase at all.) Across the crossing above it steps as far as an equal control step does, where `r_s`
steps by a sign: at `lambda = 0.62` the ratio is 0.9999 on the pure and conical paths, TE and
TM, and the step shrinks linearly with the step in `theta` from 1e-3 to 1e-6, while `r_s` /
`r_p` step by twice their magnitude at every size. On the `lambda = 0.6` grating the Rayleigh
point above makes it 2.0x (TE) and 1.6x (TM) the control -- the branch point's doing, not the
frame's. On a pure TE solve its `E_y` is the old Cartesian amplitude
times `sign(kx_inc)` -- the incidence's own sign, 1 for `theta >= 0` -- so a caller who wanted
the pre-2026-09-27 continuity reads `r_xyz[..., 1]`. It is also the quantity to read on an
order at an exact `kx = 0`, whose frame rounding picks (above): with the direct
`kx = 2*pi/P` there, Torch's `r_s(-1)` is NumPy's negated while its `r_xyz(-1)` agrees to
8e-10 (TE) and 2e-9 (TM) -- what is left is Torch's ulp in the order offsets, on a scene that
also puts orders 3 and -5 at grazing in the glass (a Rayleigh point). Pinned
from outside that arithmetic in `tests/test_cartesian_amplitudes.py` (an independent spelling
on every path, a 60-degree cell included, and the reconstructed field decomposed at both
faces). RETICOLO's `res2` returns the same Cartesian quantities per order -- E and H -- which
is prior art for it and, for a future external check, an oracle that is frame-free on both of
RETICOLO's paths. H is not shipped: `(k/k0) x E / mu`, and `boundary_fields` has it in space.

**The lateral phase reference, left open by this change, was decided the same day** -- every
per-order phase moved to the cell centre; Sec. 7 is its record.

**This is a breaking change to a documented output.** A caller who read `r_s` / `r_p` / `t_s` /
`t_p` on a `tetm` solve, its field at `theta < 0` or in TM, its scattering matrix, or its
`excitation` physics, gets the conical path's answer now. The efficiencies of a single-wave
solve did not move.

## 7. The lateral phase reference moves to the centre (2026-09-27)

**The decision** (the maintainer's, the same day as Sec. 6): one lateral phase reference for the
whole public surface -- the cell centre, `x = y = 0`, the origin every public coordinate has
used since 2026-08-31. Until then the kernels' Fourier origin, the cell **corner**, was also
where every ordinary amplitude, every `excitation` weight, the scattering matrix's ports and the
ordinary incidence were referenced, while `PlaneWave` / `OrderExcitation` were referenced at the
centre. So the same weights through `excitation` and through `OrderExcitation` were different
light once two orders were lit (`sum(R)` 0.452653 against 0.151782 on `tests/test_excitation.py`'s
scene at `phi = 0`, `{0: 1, 1: 0.6-0.8j}`), and an ordinary solve's field was the `PlaneWave`
field of its angles times `exp(-1j * k_inc . (a1 + a2)/2)`.

**The rule, in five lines.** Let `o = (a1 + a2)/2`. Order `(m, n)` sits at
`k_mn = k_inc + m b1 + n b2`, and `b_i . a_j = 2 pi delta_ij`, so `exp(1j * k_mn . o) =
exp(1j * k_inc . o) * (-1)**(m + n)` on **any** cell, a skewed one included. With
`C = diag((-1)**(m + n))` by order **label** and `alpha = exp(-1j * k_inc . o)`, and `L` what the
kernels compute (their corner column to their corner amplitudes):

* public weights `w` describe the corner column `alpha C w`; the kernels are handed `C w`
  (`conventions.corner_column`) and `alpha`, a normalisation of the whole field, is put back
  where a field is formed;
* the public amplitudes are `C L C w` -- no complex factor survives -- so on a single wave
  (`c_00 = 1`) every order with `m + n` odd is negated;
* the scattering matrix is `C S C`, block by block: every port amplitude, both ports, takes the
  same sign, which commutes with Sec. 6's port change and with the backside's block reversal. A
  cascade of two pieces whose labels differ by an odd shift has `C_B = -C_A`, and `C S C` does
  not see the sign, so the join needs nothing new (measured below);
* the kernels' field state describes the public field divided by `alpha`: the solver door
  multiplies by `alpha` once, and the result's adapter by `alpha exp(1j k_m . o) = c_m` per
  order, an exact sign.

All of it is at the doors -- `engine._kernel_lit`, `_finish`, `_centre_phase` and
`field_result` -- and none of it in a kernel, a convolution builder, `fields.py`, `Port` or
`cascade`. `solver.T1` and `solver.layer_info_list` stay the kernels' own state, in their frame.

**What moved and what did not, proved rather than argued.** The tree before the change, exported
from `0937b35`, against the tree after, fresh processes, each dump taken twice and
byte-identical to its twin: 158 NumPy scenes (1D TE/TM, conical, 2D Li, 2D NVF, first-order and a
60-degree cell, both sides, `theta` in {0, 0.3, -0.23}, ETM and SMM; excitations straddling
`kx = 0`, all-odd and all-even, on 1D, conical, 2D and the 60-degree cell; complex `theta`; a
magnetic half-space and a magnetic layer; a direct `kx`) and 28 each on JAX and Torch, 9180
arrays. The old tree was lit with each excitation's odd weights negated -- the relation above,
`new(w) = C old(C w)`, so a single wave is compared with itself. Every efficiency and
`power_entering` is byte-identical (3458 arrays); every `r_s` / `r_p` / `t_s` / `t_p` equals
`where(odd, -old, old)` byte for byte (1976); `r_xyz` / `t_xyz` the same by `==`, bytes but for
exact zeros that took a sign (988); every scattering block equals `C S_old C` byte for byte
(428); `T1` is byte-identical (214); every field -- the result door by counts and by points,
the boundary waves on both faces, the solver door with every incidence -- equals
`alpha * old` with `alpha` spelled from the scene, to 2.9e-15 normwise (1474), and the solver
door at `k_inc = 0` is byte-identical. The same comparison of the old tree with itself fails
3320 of its 6892 NumPy arrays, so the instrument sees the change.

**The contract, measured on the new tree** (`tests/test_centre_reference.py`, each against an
oracle outside the conversion):

* an ordinary solve's field **is** the `PlaneWave` field -- pure TE, pure TM at `theta < 0`,
  conical, 2D in both formulations, a 60-degree cell and a shear, both sides -- to 7.6e-15 at
  the result door, 7.5e-15 at the solver door and 2.6e-14 on the boundary waves, with no
  factor; the incident wave on the lit face is the polarization exactly at `x = y = 0`;
* `excitation=` **is** `OrderExcitation(orders=)` with the same weights: `sum(R)` 0.151782 on the
  scene above lit from the top (4.2e-16 apart) and 0.241294 from the glass (2.2e-15); a 2D
  mixed-parity set agrees in efficiencies and fields;
* every amplitude is the Fourier coefficient of the reconstructed field about the centre, on
  the lit face (incoming wave removed) and on the far face;
* **the physical anchor, first-order Born.** A weak (`delta eps = 0.02`), thin (`lambda/400`)
  scatterer centred at the origin in air reflects order `G` in proportion to `deps_G`, its
  Fourier coefficient *about the reference point*, which about the centre is real and positive
  for small `|G|`. Read through `r_xyz`, which carries no s/p frame: a centred stripe gives
  `Ey(G)/Ey(0) = +1.0996` against Born's `+1.0996` at `theta = 0`, `+1.2398` / `+1.0037` at
  `theta = 0.1`; a centred disc `+0.9953`, `+0.5623`, `+0.7191` on a rectangle and `+0.8991`,
  `+0.5474` on a 60-degree cell -- all within 0.42%. The corner reference gave the negatives on
  every odd `m + n`. (It is `E_y` that is Born-positive, not `r_s`: on the `ky == 0` row `r_s`
  is quoted along `sign(kx) * y_hat`, so a symmetric grating at `theta = 0` still has
  `r_s(-1) = -r_s(+1)`, Sec. 6);
* a cascade across an odd label shift equals the joined stack's matrix to below 1e-10, all four
  blocks;
* a y-invariant scene's field no longer depends on the `y` period it was declared with: at
  `phi = 0.8` the spellings `period=(1.6, 1.6)` and `(1.6, 2.9)` of one grating gave fields 0.99
  apart (relative) while the incidence was unit at the corner `y = -Py/2`; now one field.

**Complex `theta`.** An evanescent incident wave is not uniform along the surface, so where it
is unit sets the field's magnitude: `|alpha| = exp(Im(k_inc) . o)`, 7.70 on a TE scene at
`theta = 0.3 + 0.1j` in a lossy superstrate. It is unit at the origin now, the one point every
public coordinate names, and `alpha` is holomorphic in the angle, the wavelength and the cell,
so a gradient through a complex `theta` is the continuation of the real one. Amplitudes and
efficiencies are ratios and do not see it; the RETICOLO complex-`kx` golden reads the specular
order (`c = 1`) and does not move.

**The numbers a user has seen move** (`sum(R)`, truncation 6):

| scene | before | after | `OrderExcitation`, same weights |
| --- | --- | --- | --- |
| tutorial 07, three beams, TE, lit from the top | 0.370091 | 0.195292 | 0.195292 |
| tutorial 07, TM, top | 0.158687 | 0.103900 | 0.103900 |
| tutorial 07, lit from the glass (`theta = 0.25` there), TE | 0.592727 | 0.186446 | -- |
| same, TM | 0.323264 | 0.080007 | -- |
| `tests/test_excitation.py`'s scene, `phi = 0`, `(1, 0)`, `{0: 1, 1: 0.6-0.8j}`, top | 0.452653 | 0.151782 | 0.151782 |
| same, lit from the glass | 0.604108 | 0.241294 | 0.241294 |

Tutorial 07's number has moved twice in one day, by the same shape of change both times -- a
per-lit-order `+-1`: TE 0.4933 (every lit order along `+y`) -> 0.3701 (each order's own s/p
frame, Sec. 6) -> 0.1953 (the centre reference); TM 0.2171 -> 0.1587 -> 0.1039. After the second
the two excitation doors agree, which is the point of it. A lit set of one parity class is the
same light as before -- `sum(R)` byte-identical for `{0, 2}` and for `{-1, 1}`; on an all-odd
set the kernels see `-w`, so there the even orders' amplitudes are the negated ones
(`new = -C old`). The single-order relabelling identity (tutorial 07's cell 9) still holds, to
3.6e-9 as before: lighting order `+1` at `theta` and the specular order at the matched angle are
one wave, unit at the same point.

**The remaining corner-frame state.**

* The public convolution matrices initially stayed in the corner frame. That exception was
  superseded on 2026-09-28: the accessors, `solve_conv` and exported NVF builders move in §8.
* `solver.T1` / `solver.layer_info_list` (above).
* The coherent-source path's own gauge, which reads the private accessors and re-bases their matrices to the
  centre with `exp(1j (k_m - k_n) . o)` -- `c_m c_n` in exact arithmetic.

**The centre belongs to the basis you wrote.** A unimodular re-basis `(a1, a2) -> (a1, a1 + a2)`
moves `(a1 + a2)/2` by `a1/2`, so the phase reference moves with it, as shapes and sample points
already did. Keeping the same public numbers therefore translates the whole scene and leaves the
field as it was; the same physical crystal, sampled at the same physical points, gives fields at
`theta != 0` that differ by `exp(-1j k_inc . (o_B - o_A))` (both at convergence -- the two bases
retain different orders). Efficiencies do not see it.

**The external oracles whose origin is the corner.** RETICOLO (its 1D, conical and `res3` field
paths) and fmmax put their incident wave's unit at the cell corner, so every field golden from
them now applies the closed-form phase between the two unit points at the comparison, and a
per-order amplitude comparison against them takes `(-1)**(m + n)` on top of any frame relation
(`tests/references/README.md`). S4 references its incidence at its own origin, the cell centre,
and its field golden samples there with no factor.

**This is a breaking change to a documented output.** A caller who read `r_s` / `r_p` / `t_s` /
`t_p`, `r_xyz` / `t_xyz` or `order(m, n)` on an order with `m + n` odd, the scattering matrix, an
ordinary field at `theta != 0`, or `excitation` physics with lit orders of both parities, gets
the centre-referenced answer now. The efficiencies of a single-wave solve did not move.
`studies/scene_hash.py` hashes a field at `theta = 0.3`, so its digest moves on every scene by
design. The FDTD module, not on develop at this change, extracts its per-order amplitudes from
its grid's corner; when it lands, its extraction has to meet this contract (each order times
`c_m` at its extraction plane, and `alpha` for its Bloch source at oblique incidence), and the
`OrderExcitation` comparison above is the oracle to reuse.


## 8. The Fourier-space doors move to the centre (2026-09-28)

The public origin now also governs convolution matrices. A shape whose centre is written
as `(cx, cy)` publishes coefficients about that origin, so the closed form is
`c_G = (1/A) ∫ eps(r) exp(-i G·r) dr` with public `r`. The first harmonic of a centred
denser stripe is positive. This removes the temporary matrix exception in §7.

For `o = (a1+a2)/2`, translation in the retained plane-wave basis is diagonal:
`T_mn = exp(i k_mn·o) = exp(i k_inc·o) (-1)**(m+n)`. Consequently **every operator**
transforms as `T C inv(T) = D C D`, `D = diag((-1)**(m+n))`; the common incident phase
cancels. Products and inverses obey the same similarity, so the rule covers Li inverse
slots, NVF closures, all nine tensor slots, permeability, and derivatives. The code uses
`where(odd_row != odd_column, -C, C)`. Two swaps restore the exact original bits, including
signed zeros. A window shift changes D by a global sign and therefore leaves D C D unchanged.

The conversions are confined to public doors:

* `eps_conv_matrices()` and `mu_conv_matrices()` restate each returned slot.
* `solve_conv()` restates every input slot once before calling the private `_solve_conv()`.
* `ConformingNVF`, `AffineProfileNVF`, `CartesianRectangleNVF` and
  `CartesianRectanglePairNVF` restate their `conv_matrices()` output.
* `CartesianPairTransmissionGalerkinGapCell.closure()` restates both its value and tangent.

`ConvMatrices.from_energy()` is covariant algebra, not a coordinate adapter: it preserves
the origin supplied in Q, G and the longitudinal matrix. Centre-frame inputs therefore
produce centre-frame slots ready for `solve_conv()`. The same type also carries private
corner-frame matrices; constructing the container adds no transformation.

The ordinary `solve()` path goes straight from the private builders to `_solve_conv()`.
The coherent path reads the private accessors and retains its existing gauge. Private NVF
cells and study calculations keep their existing arithmetic and placement; their solver
calls now use `_solve_conv()`. Study models built natively around zero also keep their
previous placement through that private door; correcting the historical placement of such
research models is separate from this public-frame change. `T1`, `layer_info_list`, and
the `epz_conv_i` or full tensor blocks stored inside them remain kernel state.

**This is a breaking change to documented matrix inputs and outputs.** A caller supplying
patterned corner-frame coefficients unchanged to `solve_conv()` now translates that layer
by half a cell. A single incident wave's efficiencies cannot detect a translation of an
entire stack, but its complex amplitudes can, and a mixture with centre-frame layers changes
relative placement and can change efficiencies. Existing corner-frame data must be restated
with `C * np.outer(d, d)`, `d = (-1.)**(m+n)`, or routed through the private kernel door in
internal code. Accessor-to-solver and exported-builder-to-solver round trips keep their
original results, since the two exact sign operations cancel.

**Verification.** The private-door extraction has an identical calculation AST. Independent
process snapshots on NumPy cover twelve ordinary scenes, both illumination sides, moved
windows, full epsilon/mu tensors, coherent fields, all five exported NVF doors, and a solve
along the transmission tangent: 1,410 answer arrays remain byte-identical, and all 90
published matrix slots equal the original slots restated by D C D byte for byte. The
JAX and PyTorch snapshots each preserve all 944 answer arrays and restate all 60 published
slots exactly. Their public matrix gradients are also the private gradients restated by D C D
byte for byte; the JAX compiled solve and accessor forward derivatives pass independently.
The tests also compare public coefficients with sinc rectangle and Bessel disc integrals, including
an oblique cell; a hand-built centre-frame stack with the shape solver; mixed built/hand
layers; and a published harmonic with the Born amplitude. Research references are restated
without changing their tolerances, and their random-entry derivative objectives keep the
original corner-frame functional.

`SPEC_VERSION` is unchanged: specifications carry no matrix and their `solve()` results do
not move. `MANIFEST_VERSION` is unchanged under its additive-key rule: `lattice.fourier_origin`
and its expression describe the public matrix frame without renaming or reshaping an existing
key. The manifest did not expose hand-built `solve_conv` inputs as a generation path.

# The per-order s/p frame the complex amplitudes are quoted in

Written: 2026-08-16.
Targets: `wavebit/_core/conventions.py`, `wavebit/_core/result.py`,
`tests/test_amplitude_sp_frame.py`, `tests/test_order_azimuth.py`,
`tests/test_golden_legacy_parity.py`

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
in either axis), `period = [500, 800]`, `max_order = [5, 3]`, `theta = 25 deg`, `wl = 250`,
`factorization='nvf'`, so 18 orders propagate and the retained grid straddles `kx = 0` with 35
orders on one side and 42 on the other. NumPy, complex128.

**(a) The transposed twin.** Reflecting the whole scene about `x = y` -- transpose the cell,
swap the period pair, swap the `max_order` pair, `phi -> pi/2 - phi`, `a_s -> -a_s` -- is the same
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
`max_order` -- so it could not isolate a translation. That was a separate, pre-existing property of
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
the `z = 0` face onto the retained orders (residual 3.1e-15) and projecting onto
`s_hat = z_hat x k_hat`:

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
reported against an identical basis in both. 8x8 cell, `max_order = [4, 4]`, `psi = 0.7`, `phi = 0.3`:

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

* the six pure-1D cases of `test_golden_legacy_parity` are untouched: the TE/TM formulation
  carries no s/p rotation at all;
* all seven coupled cases (`conical`, `conical_2l`, and the five `2d_*`) move by the per-order
  sign on their amplitudes, and by nothing at all on their efficiencies. Handled as a
  convention conversion at the comparison rather than by refreezing the golden -- the legacy
  values stay the legacy solver's own, see `_frame_sign` there;
* no external golden moves, for the reason in Sec. 2.

**This is a breaking change to a documented output.** A caller who read `r_s` / `r_p` / `t_s` /
`t_p` on a 2D or conical solve and compensated for the old sign will need to stop.

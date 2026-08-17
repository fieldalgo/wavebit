# Conventions — the contract

Written: 2026-08-12, for the first release. This page is the normative statement of every
sign, axis, name and unit a user's script can depend on. Each row is pinned by a test; the
measurement documents behind the choices are linked at the end.

If a future change would falsify a sentence on this page, that change is breaking, whatever
the diff looks like.

---

## Time and loss

| | convention |
| --- | --- |
| time factor | `e^{-iωt}` — a physical field is `Re[E(r) e^{-iωt}]` |
| loss | `Im(n) > 0`, `Im(eps) > 0` — indices as databases tabulate them (`n + 1j*k`) |
| gain | `Im(n) < 0`; a solve may then legitimately return `R + T > 1` (the `UnphysicalResultWarning` is a false positive there — filter it) |

Same statement as fmmax's; RETICOLO's efficiencies and amplitudes agree with this
convention with no conversion (pinned to 4e-14 by the RETICOLO goldens). All complex
amplitudes this library returns are in this convention; results quoted from an
`e^{+jωt}` source (older papers, meent-era scripts writing `n - jk`) are the complex
conjugates of ours.

## Geometry and propagation

Layers are listed **top to bottom**, and `+z` points down, into the stack: `ucell[0]` is
the layer light meets first, `n_top` the medium it arrives from, `n_bot` the medium it
exits into. A wave travelling `+z` carries `exp(+i*k0*kz*z)` with `Re(kz) > 0`; decay
toward `+z` is `Im(kz) > 0`.

"Top/bottom" name the *scene*, not the illumination. `incidence_side` (`'top'` by
default, `'bottom'` for back-side illumination) selects which medium the light arrives
from; nothing on this page renames or reorders when it changes — `ucell[0]` is the top
layer whichever way the light travels. Results are **incidence-relative** (the port
model): `R` is power back into the incidence medium, `T` power through the far side,
`theta`/`phi` are measured against the incidence medium's normal (`n_bot`'s, for
`'bottom'`), and the amplitudes keep the Fresnel contract below — so user
post-processing is identical for both sides.

Within a layer the array follows the image convention: row `0` of `ucell` is `y = 0`, for
every builder and for the reconstructed field cell alike. The flip that renders a field
the conventional way up belongs to plotting (`field_plot`), not to the data.

**The reconstructed field's vector components use that same `+z`.** `calculate_field` returns
`(Ex, Ey, Ez, Hx, Hy, Hz)` (or the 3-component 1D layouts) in the frame declared above, so
`H = k̂ × E` for a plane wave travelling into the stack and the Poynting vector of a downward
wave has `Sz > 0` — the same direction the cell's `z` index runs. This is worth stating
because it was not true before 2026-08-14: every path returned `Ez`, `Hx` and `Hy` with the
opposite sign.

That was a contradiction inside the repo, not a convention nobody had chosen.
[`formulation.md`](formulation.md) §1 anchors (M1)/(M2) on a vacuum wave travelling `+z` and
records the pair `(Ey, H̃x) = (1, -kz)`; `tests/test_anisotropic_berreman.py::_vacuum_modes`
writes that pair; `solve_2d_4ff` derives in it. The reconstruction was the one place that did
not. Nothing caught it because nothing external pinned it — the only Reticolo field comparison
is a print-only script that hand-conjugates its reference — and every check that existed
compared the reconstruction against itself, or looked only at `E_t`, which the sign change
leaves alone ([`time-convention-audit.md`](time-convention-audit.md) §8). Field reconstruction
now has an outside opinion for the first time — `tests/references/test_s4_field_golden.py`
compares against S4 entry for entry at 1e-15 on uniform stacks, which is where frames and
signs are decided — and it is pinned per path by
`tests/test_field_core.py::test_every_path_returns_a_maxwell_field_in_the_declared_frame`,
which asserts the curl equations *and* their convergence rate, since a wrong frame does not
converge rather than merely scoring worse.

## Units and angles

`wavelength`, `period`, `thickness` and the smooth-path shape coordinates (`cx`, `lx`,
…) share **one length unit of your choosing** — nothing in the solver assumes nm. Shape
coordinates are absolute lengths, not fractions of the period.

All angles are **radians**.

| angle | meaning |
| --- | --- |
| `theta` | polar incidence angle, measured from the stack normal in the incidence medium |
| `phi` | azimuth of the plane of incidence, from the `+x` axis |
| `psi` | polarization: the angle of **E** from the p direction in the plane transverse to **k**. `psi = pi/2` is pure s (TE-like), `psi = 0` pure p (TM-like). **Real** — a complex `psi` is refused, see below |
| `chi` | ellipticity angle: `tan(chi)` is the polarization ellipse's minor/major axis ratio and its sign is the rotation direction. `chi = 0` is linear (the `psi` spelling), `|chi| = pi/4` circular. Real, like `psi`. Naming it selects the `(psi, chi)` chart — see below |
| `tetm` (1D only) | `'te'` / `'tm'` — a **declaration**, not an angle: it says this problem has no polarization and no azimuth degree of freedom, which is what makes the pure 1D TE/TM formulation legal (2-4x faster than conical, measured). Refused on a 2D cell, with `fto[1] != 0`, or beside a nonzero `phi` / a `psi` / a `polarization`. `psi` reads back `None` under it — a declared-away degree of freedom is absent, not defaulted |

### General (elliptical) polarization

`polarization = (a_s, a_p)` states the incident polarization as a pair of complex
**E-field amplitudes**, in the same (s, p) basis as `psi` and the returned amplitudes:
`(1, 0)` is pure s, `(0, 1)` pure p, and `psi = v` is exactly
`polarization = (sin(v), cos(v))`. What the pair adds is the relative phase `psi` cannot
spell — `(1, -1j)/sqrt(2)` and `(1, +1j)/sqrt(2)` are the two circular states.

- `psi` and `polarization` are mutually exclusive. Naming both is an error, not an
  override.
- The pair's overall magnitude is **not read**: `|a_s|^2 + |a_p|^2 = 1` is not required,
  because the pair is normalized to unit incident flux on the way in — efficiencies are per
  unit incident power and amplitudes per unit incident E-field whatever scale you write.
  The overall *phase* is kept, and carries into the complex amplitudes.
- A 1D structure given a pair is solved on the conical path (the pure 1D TE/TM path is
  single-polarization by construction); the answer is the same physics at conical cost.
- **A complex `psi` is refused, and is not a shortcut to any of this.** `(sin, cos)` is a
  unit pair only for a real angle; with `Im(psi) = b` the incident flux becomes `cosh(2b)`
  and every efficiency carries that factor — invisible on an absorbing stack, where the
  measured case returned `R + T = 0.998` for a right answer of `0.647`. It cannot reach the
  circular states either: `a_s/a_p = tan(psi)`, and `tan(z) = ±i` has no solution (it
  reduces to `exp(±iz) = 0`), so they are exactly the two values `tan` omits — approached
  only as `Im(psi) -> ±inf`, with the amplitude diverging at the same rate. `psi` is stored
  in the real dtype for this reason, unlike `theta`, whose complex values are a feature
  (evanescent incidence); the autodiff leaf is real with it.

#### The canonical coordinates: `(psi, chi)`

The pair is the normative spelling, but it is not the one to *move*. A polarization state has
exactly two degrees of freedom, and `(psi, chi)` are their standard coordinates: `2*psi` and
`2*chi` are longitude and latitude on the Poincaré sphere, so the two angles cover every state
once, with nothing to normalize.

```
a_s = sin(psi)*cos(chi) + i*cos(psi)*sin(chi)
a_p = cos(psi)*cos(chi) - i*sin(psi)*sin(chi)
```

`chi = 0` returns `(sin psi, cos psi)` — the chart contains the angle it extends — and
`chi = ±pi/4` gives the two circular states. What it does not carry is an overall phase, which
is the third real number the pair has and the state does not; that is why the pair stays
normative and why the two are mutually exclusive, as `chi` is with `tetm`.

The reason to have both is optimization. Of the pair's four real numbers, two are structurally
null: its overall scale is divided out, and its overall phase cannot move any power. Measured
on a 2D grating at `(psi, chi) = (0.4, -0.2)`:

| direction | `d(sum T)/d` |
| --- | --- |
| `psi` | `+0.048772` |
| `chi` | `+0.075185` |
| the pair's overall scale | `2.4e-16` |
| the pair's overall phase | `-1.8e-16` |

So an optimizer over the raw pair spends half its dimensions on directions that cannot change
the answer, while `(psi, chi)` is an ordinary unconstrained 2D problem. Both coordinates are
differentiable on every backend and both are pytree children under JAX.

`chi = 0` is the `psi` spelling exactly — bit-identical, not merely equal. Both angle forms
are unit by construction (`cos^2(chi) + sin^2(chi) = 1`), so neither is normalized: dividing
them by their own computed norm could not remove an error, only add one (`sqrt(...)` misses
`1.0` by up to half an ulp for the angle and 1.5 for the chart). **Only a written
`polarization` pair is normalized**, because only it can carry a scale the library did not
choose.

#### Which way the circular states rotate

The rows above never say "left" or "right", deliberately: the two naming camps disagree
about the *same physical field* (optics names handedness from the receiver's viewpoint,
IEEE from the source's, so optics-RCP **is** IEEE-LHCP), which is why the API takes the
raw pair and every solver in this ecosystem does likewise. The physical rotation itself,
however, is fully determined by conventions this page already fixes, so here it is.

Derivation, at normal incidence with `phi = 0`: the (s, p) basis maps to Cartesian as
`E = (E_x, E_y) = (-a_p, a_s)` (the calibration measured in
`tests/test_uniform_anisotropic.py`: `t_s = E_y`, `t_p = -E_x`). The pair
`(1, -1j)/sqrt(2)` is then `E ∝ x̂ - i·ŷ` up to a global phase, and in this library's
`e^{-iωt}`:

```
E(t) = Re[(x̂ - i·ŷ)·e^{-iωt}] ∝ x̂·cos(ωt) - ŷ·sin(ωt)
```

whose rotation vector is `-ẑ` while the wave travels `+z` (down, into the stack). So:

| pair | rotation vs **k** | optics name (receiver's view) | IEEE name (along propagation) |
| --- | --- | --- | --- |
| `(1, -1j)/sqrt(2)` | antiparallel | right circular (RCP) | left-hand (LHCP) |
| `(1, +1j)/sqrt(2)` | parallel | left circular (LCP) | right-hand (RHCP) |

Two anchors keep this paragraph honest rather than derived-once-and-drifting: the basis
calibration above, and the gyrotropic eigenstate test
(`tests/test_polarization.py::test_each_circular_pair_is_one_faraday_eigenstate`), which
pins that `(1, -1j)/sqrt(2)` is the state seeing `ε_d + γ` in a layer with
`eps = [[ε_d, iγ], [-iγ, ε_d]]`. And the usual seam warning applies to Jones vectors
exactly as it does to every other amplitude: a pair quoted from an `e^{+jωt}` source is
the complex conjugate of ours, and its handedness labels flip with it.

## Diffraction orders

The retained orders are `m = -fto_x..fto_x`, `n = -fto_y..fto_y`, with in-plane
wavevectors `kx_m = n_top·sinθ·cosφ + m·λ/Λx` (and correspondingly `ky_n`). Per-order
arrays have shape `(ff_y, ff_x)` — **`y` first, the transpose of the `fto=[x, y]` input
order** — with the incident order at the centre `[fto_y, fto_x]` and `m` increasing along
the last axis.

`result.order(m, n)` indexes all per-order quantities by the order indices themselves, so
user code never needs the previous paragraph.

## The result surface

**Uppercase is power, lowercase is amplitude.**

| name | meaning |
| --- | --- |
| `R`, `T` | per-order diffraction efficiency: power returned to the incidence medium / leaving through the far side, as a fraction of incident power |
| `R_s`, `R_p`, `T_s`, `T_p` | the same, resolved by outgoing polarization; `R = R_s + R_p` |
| `r_s`, `r_p`, `t_s`, `t_p` | complex **Fresnel amplitudes**: E-field ratios against a unit incident E-field, directly comparable to the textbook Fresnel/Airy coefficients (pinned to 1e-12 against them, absorbing media included) |
| `res`, `res_te_inc`, `res_tm_inc` | the incidence as specified (`psi`, or the `polarization` pair), plus the pure-s and pure-p variants the coupled solves produce for free; top-level `R`/`T` average the two variants when no specified solve exists (unpolarized illumination) |

Amplitude fine print, stated once so it never has to be discovered:

- each order is quoted in **its own** s/p frame, the standard one: `s_hat = z_hat x k_hat_par`,
  with `k_par` that order's in-plane wavevector, and `p_hat` completing the right-handed
  triad. Until 2026-08-16 every order with `Re(kx) < 0` came back in the *reversed* frame
  (`-s_hat`, `-p_hat`) on the conical and 2D paths — invisible in `R`/`T`, which are
  `|r|^2`, and in the reconstructed field, and therefore in every golden this project has.
  See [`amplitude-sp-frame.md`](amplitude-sp-frame.md). At `theta = 0` the zeroth order has
  no in-plane wavevector to take a direction from and `phi` supplies it, with the one
  caveat in that document's section 4;
- the p sign convention is the one where a single interface at normal incidence gives
  `r_p = (n2 - n1)/(n2 + n1) = -r_s`;
- phase reference: `r` at the incidence-side interface, `t` at the far one (top and
  bottom respectively for the default illumination);
- with an absorbing **superstrate**, incident power is convention-dependent, so `R`/`T`
  are not comparable to 1 — the amplitudes are the meaningful quantities there. Concretely
  so: the pure-1D and coupled paths normalize that undefined quantity *differently* (TM
  efficiencies differ by exactly `|n_top|²·Re(n_top⁻²)`; TE by exactly 1), which is pinned
  as deliberate notice rather than hidden — compare `r`/`t`, never `R`/`T`, in that regime;
- there are no raw "boundary variables" in the public surface. (Before the first release
  the p amplitudes were exposed as magnetic-field boundary variables needing a
  `/(-i*n_top)` footnote; they were renamed and normalized outright, with no aliases —
  including the meent-era efficiency names `de_ri`/`de_ti`.)

## Where each row is pinned

| rows | tests |
| --- | --- |
| time/loss, amplitudes | `tests/test_lossy_media.py` (Airy oracle, absorbing media both sides) |
| the per-order s/p frame the amplitudes are quoted in | `tests/test_amplitude_sp_frame.py` — a transposed twin, the reconstructed field as arbiter, and continuity across `kx = 0`. Efficiencies cannot pin this row at all: a reversed frame is `\|r\|^2`-invariant |
| order layout, order-index sign | `tests/references/test_reticolo_golden.py` (mirrors orders if wrong, conserving energy) |
| `incidence_side`, incidence-relative results | `tests/test_backside_incidence.py` (reversed-stack Airy oracle, reciprocity both halves) and the `inc_bottom` cases of `tests/references/test_reticolo_golden.py` |
| result naming, `order()`, delegation | `tests/test_result_api.py` |
| the `polarization` pair: `psi` equivalence, unit-flux normalization, circular states | `tests/test_polarization.py` |
| `psi` and `chi` are real; the chart reaches the circular states; both coordinates and the pair carry gradients through a pytree | `tests/test_polarization.py` (the refusal on all three backends, and the pair's gradient checked against the `psi` path's rather than a frozen number) |
| the circular states' rotation sense | derived above from two pinned facts: the (s, p)→(x, y) calibration (`tests/test_uniform_anisotropic.py`) and the gyrotropic eigenstate assignment (`tests/test_polarization.py`) |
| square-root branches behind `kz` | `tests/test_branch_selection.py` |
| the field cell's `+z` (that `Ez`, `Hx`, `Hy` share the frame the phase and layer order use) | `tests/test_field_core.py` — the curl residual **and its convergence rate**, per path, plus the control that the pre-2026-08-14 frame fails it. Not the legacy golden, which was frozen in that frame and is transformed at the comparison |
| the whole surface, per backend | `tests/test_golden_legacy_parity.py` |

Why these choices: [`time-convention-audit.md`](time-convention-audit.md) (the branch
rules and the flip residue), [`formulation.md`](formulation.md) (the full derivation in
this convention), [`branch-cuts.md`](branch-cuts.md) (single precision limits).

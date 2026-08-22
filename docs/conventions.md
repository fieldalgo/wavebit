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
| loss | `Im(eps) > 0`, equivalently `Im(n) > 0` — the sign indices are tabulated with (`n + 1j*k`) |
| gain | `Im(eps) < 0`; a solve may then legitimately return `R + T > 1` (the `UnphysicalResultWarning` is a false positive there — filter it) |

Same statement as fmmax's; RETICOLO's efficiencies and amplitudes agree with this
convention with no conversion (pinned to 4e-14 by the RETICOLO goldens). All complex
amplitudes this library returns are in this convention; results quoted from an
`e^{+jωt}` source (older papers, meent-era scripts writing `n - jk`) are the complex
conjugates of ours.

## Materials — what a material number is

**Every material value a script writes is a relative permittivity**, and every permeability
beside it is relative too. Complex is normal: `Im(eps) > 0` is loss, as in the table above.

This is the one entry on this page whose *value* changed and not merely its name (2026-08-20).
Every material slot took a refractive index before, and the number a script wrote for one is
still a legal value for the other — which is why the old spellings are refused at the keyword
rather than accepted. See "the retired spellings" below.

`Material` is the type that carries one:

```python
wavebit.Material(epsilon, mu=1.0)
```

Frozen, and a registered JAX pytree whose children are `(epsilon, mu)` — a material is a
legitimate design variable, so it holds tracers and carries gradients rather than being an
opaque leaf. **A bare number is the shorthand for `Material(epsilon=x)`** anywhere a material
is accepted; non-magnetic is the overwhelming default and costs nothing to write.

The scene is stated with four keywords:

| keyword | what it holds |
| --- | --- |
| `structure` | the layer stack, layer-major on its first axis — one of the three encodings below |
| `top`, `bot` | the two half-space materials: a `Material`, or a bare relative permittivity |
| `mu` | the **raster path only**: the permeability field, `(n_layers, Ny, Nx)`, beside `structure`. Named on a shape path it is refused, not ignored — there a permeability belongs to the `Material` filling the region |
| `thickness` | per-layer thicknesses, in the shared length unit |

The *type* of `structure` selects the modeling path; there is no separate mode switch:

| encoding | `structure` |
| --- | --- |
| raster | ndarray `(n_layers, Ny, Nx)` of **relative permittivity** — one sample per pixel |
| block | `[[background, [['rectangle', cx, cy, lx, ly, material, angle], ...]], ...]` — a `Material` in slot 0 and in positional slot 5 |
| analytic | `[dict(background=..., shapes=[dict(type=..., cx=..., cy=..., lx=..., ly=..., angle=..., material=...)], combine=...), ...]` |

Three of those four ways of saying what fills a region — the block background, the block
object's own slot, the analytic shape's `material` — are **assignments**: a name for the stuff
in a region, and a medium has both a permittivity and a permeability, which is why one type
carries the pair. The raster is the fourth and is not an assignment at all but a **sampled
field**, one number per pixel, which is exactly why topology optimisation reaches for it. A
field has one quantity in it, so the raster's permeability is a second array (`mu=`) rather
than a second slot on an object.

### The retired spellings, and why they are refused by name

| retired | now | what it names |
| --- | --- | --- |
| `ucell` | `structure` | the layer stack |
| `n_top` / `n_bot` | `top` / `bot` | the two half-space materials |
| `n_base` / `base` (analytic layer keys) | `background` | a layer's background material |
| `n_index` (analytic shape key) | `material` | a shape's material |
| `ucell_materials` | — | deleted; it was assigned by every backend and read by nothing |

Every one of these raises, naming its replacement **and** saying that the quantity changed
with it. That second half is the point. A user who renames the keyword and keeps the number
has not made a mistake anything downstream can see: `3.5` is a perfectly good permittivity, so
the solve runs, conserves energy, and answers a different question — the README quickstart with
every material literal left at the index it used to be sums `R` to 0.1753 where the migrated
scene gives 0.2053, with no warning. The two are close on purpose, as an argument: a migration
that broke loudly would announce itself, where a shift the size of a truncation wobble gets
believed. Silence is not available here, so the refusal is.

Two conversions are exported for callers holding the other quantity:

| function | value |
| --- | --- |
| `wavebit.eps_from_index(n)` | `n**2`. Unambiguous, and deliberately more to type than `3.5` was: an index silently reread as a permittivity is a legal wrong answer, so the conversion is worth seeing at the call site |
| `wavebit.index_from_eps(eps)` | the **principal** root, always complex. A passive medium has `Im(eps) >= 0`, which is the closed upper half plane, whose principal root has `Im(n) >= 0` — loss again. The hazard is the negative side of zero: `-2 + 0j` gives `+1.414j` (decaying, right) and `-2 - 0j` gives `-1.414j` (gain), and a lossless value that has been through arithmetic can come out as `4 - 1e-18j`. Neither is repaired, because nothing can tell round-off from a deliberately active medium; keep `Im(eps)` non-negative yourself before taking a root |

For a magnetic medium the index is `index_from_eps(eps*mu)`, written out rather than hidden in
a keyword, because that is where a forgotten `mu` would otherwise hide.

### Where the refractive index is still the quantity

The flip is about what a *material slot* holds; it does not rename the physics. The index
remains the right quantity in Snell's law, in the incident wavevector (`kx_m` below), and in
the half-space relation `n = sqrt(eps*mu)` itself.

Concretely, `n_top` and `n_bot` still exist and still mean the refractive index — as
**derived, read-only** properties. Assigning `top` stores the material and refreshes
`n_top = sqrt(eps*mu)`; assigning `n_top` raises. Two spellings of one medium that could
disagree is exactly the state the type exists to remove.

## Geometry and propagation

Layers are listed **top to bottom**, and `+z` points down, into the stack: `structure[0]` is
the layer nearest `top`, `structure[-1]` the one nearest `bot`, and `top` / `bot` are the
half-spaces above and below — which of them the light arrives from is `incidence_side`, not
the ordering. A wave travelling `+z` carries `exp(+i*k0*kz*z)` with `Re(kz) > 0`; decay
toward `+z` is `Im(kz) > 0`.

"Top/bottom" name the *scene*, not the illumination. `incidence_side` (`'top'` by
default, `'bottom'` for back-side illumination) selects which medium the light arrives
from; nothing on this page renames or reorders when it changes — `structure[0]` is the top
layer whichever way the light travels. Results are **incidence-relative** (the port
model): `R` is power back into the incidence medium, `T` power through the far side,
`theta`/`phi` are measured against the incidence medium's normal (`bot`'s, for
`'bottom'`), and the amplitudes keep the Fresnel contract below — so user
post-processing is identical for both sides.

Within a layer the array follows the image convention: row `0` of a raster `structure` is
`y = 0`, for every builder and for the reconstructed field cell alike. The flip that renders a
field the conventional way up belongs to plotting (`plot_field`), not to the data.

**The reconstructed field's vector components use that same `+z`.** `reconstruct_field` returns
`(Ex, Ey, Ez, Hx, Hy, Hz)` (or the 3-component 1D layouts) in the frame declared above, so
`H = k̂ × E` for a plane wave travelling into the stack and the Poynting vector of a downward
wave has `Sz > 0` — the same direction the cell's `z` index runs. This is worth stating
because it was not true before 2026-08-14: every path returned `Ez`, `Hx` and `Hy` with the
opposite sign.

**And it holds on both sides**, which took until 2026-08-22. A backside solve *is* the
top-side solve of the z-mirrored scene, so everything it leaves behind is expressed in that
mirror; the reconstruction reversed the z axis and stopped there. `z → -z` is improper, so the
components do not all follow the coordinates — `E` is polar and loses `E_z`, `H` is axial and
loses `H_x`/`H_y` — and without that parity a cell lit from below came back with `Sz > 0` for
light travelling *upward*, bit-identical to the top-side answer instead of its mirror. Every
efficiency was unaffected, which is why nothing saw it: the check that catches it is net
Poynting, which is answerable from the cell alone against a fact no reconstruction supplies
(`tests/test_backside_incidence.py::test_net_power_flows_away_from_the_lit_face`).

An explicit `sample_z` is absolute in **scene** coordinates on both sides, for the same reason
the labels are. A backside request is restated in the mirror before it is bucketed, and comes
back in the caller's own order; it used to be bucketed against the mirror's thicknesses *and*
then reversed.

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
| `tetm` (1D only) | `'te'` / `'tm'` — a **declaration**, not an angle: it says this problem has no polarization and no azimuth degree of freedom, which is what makes the pure 1D TE/TM formulation legal (2-4x faster than conical, measured). Refused on a 2D cell, with `truncation_order[1] != 0`, or beside a nonzero `phi` / a `psi` / a `polarization`. `psi` reads back `None` under it — a declared-away degree of freedom is absent, not defaulted |

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

The retained orders are `m = -half_width_1..half_width_1`, `n = -half_width_2..half_width_2`, with in-plane
wavevectors `kx_m = n_top·sinθ·cosφ + m·λ/Λx` (and correspondingly `ky_n`). This is one of
the places the **index** is genuinely the quantity: it is `n_top`, the derived
`sqrt(eps*mu)` of the `top` material, and it is `n_bot`'s under
`incidence_side='bottom'` — `theta` lives in the incidence medium. Per-order
arrays have shape `(n_orders_2, n_orders_1)` — **`y` first, the transpose of the `truncation_order=[x, y]` input
order** — with the incident order at the centre `[half_width_2, half_width_1]` and `m` increasing along
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
| `given`, `s_inc`, `p_inc` | the incidence as specified (`psi`, or the `polarization` pair), plus the pure-s and pure-p variants the coupled solves produce for free; top-level `R`/`T` average the two variants when no specified solve exists (unpolarized illumination) |

**Which of these rows exist is what `answers` declares**, and the table above is the default
(`answers='amplitude'`) in full. `answers='efficiency'` keeps `R` and `T` and leaves every
other slot `None`, including `s_inc` / `p_inc` — so on that rung there is no
unpolarized average to take and `given` is the only incidence. `answers='field'` adds nothing
here: what it adds is the per-layer state `reconstruct_field` walks, which is not part of the
result surface at all, and below that rung reconstruction is refused by name rather than
served. The ladder is monotone, so a lower rung is always a *smaller* answer and never a
different one — pinned bit-for-bit, not to a tolerance.

Amplitude fine print, stated once so it never has to be discovered:

- each order is quoted in **its own** s/p frame, the standard one: `s_hat = z_hat x k_hat_par`,
  with `k_par` that order's in-plane wavevector, and `p_hat` completing the right-handed
  triad. Until 2026-08-16 every order with `Re(kx) < 0` came back in the *reversed* frame
  (`-s_hat`, `-p_hat`) on the conical and 2D paths — invisible in `R`/`T`, which are
  `|r|^2`, and in the reconstructed field, and therefore in every golden this project has.
  See [`amplitude-sp-frame.md`](amplitude-sp-frame.md);
- at `theta = 0` the zeroth order has no in-plane wavevector to take a direction from, and
  `phi` supplies one — as a **direction**, not a line. `phi` and `phi + pi` therefore
  describe incident fields of opposite sign for the same `psi`, and the solve returns the
  negated field and amplitudes to match; they are the same illumination only up to that
  sign. Equivalently, `(phi + pi, psi)` is `(phi, polarization=(-a_s, -a_p))`. Until
  2026-08-16 the two spellings disagreed, and the `phi + pi` one silently answered for
  `phi`;
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
| **which quantity a material number is** | `tests/test_material_convention_anchor.py` — every expected value there comes from a closed-form Fresnel/Airy recursion written in `eps` and `kz`, so it cannot be co-migrated into agreement with a flip. One function in it, `_api_material`, knows the convention; nothing else in the file may move when the convention does. The rest of the suite cannot pin this row at all: a golden regenerated by a script that imports wavebit agrees with whatever the package now believes, and reading `3.5` as the other quantity conserves energy exactly |
| the `Material` type: the scalar shorthand, both values as pytree children (so a permittivity differentiates), and the two conversions including `index_from_eps`'s branch and its signed-zero hazard | `tests/test_material.py` |
| which rows exist at all, per `answers` rung | `tests/test_answers_declaration.py` — each rung's surface, the rungs agreeing bit-for-bit on what they share, and the refusal naming the declaration |
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

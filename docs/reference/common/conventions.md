# Conventions — the contract

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


The current public solver modules share the following physical conventions. The detailed earlier-interface contract below is retained for compatibility and archived comparisons; its old factory keywords and normalized magnetic fields are explicitly scoped to `wavebit.compat.legacy`.

| Current convention | Meaning |
| --- | --- |
| Materials | Relative permittivity and permeability; shared physical response models |
| Phasors | `exp(-i omega t)`; positive imaginary epsilon is passive electric loss |
| Periodic stack coordinates | Bottom-to-top layer order, z=0 at the bottom, +z upward; centered transverse cell |
| Geometry units | Explicit `Units` for lengths and wavelengths; angles radians |
| Time and material rates | Seconds, Hz, rad/s, conductivity S/m; Lorentz strength s⁻² |
| Public field components | `(Ex,Ey,Ez,Hx,Hy,Hz)`, E V/m and H A/m |
| Diffraction axes | Wavelength, order v, order u; amplitude reference and validity remain in result metadata |

See [the current lifecycle](../../guide/common/lifecycle.md) and [migration](../../migration/index.md) before transferring old parameters. FDTD current moments (A·m or V·m) differ from RCWA electric dipole moments (C·m).

## Earlier-interface contract and evidence

The remainder records the established compatibility contract, originally written on 2026-08-12 and subsequently amended. Its measured checks and historical parameter names remain useful for reproducibility; they do not override the current declarations above.

## Time and loss

| | convention |
| --- | --- |
| time factor | `e^{-iωt}` — a physical field is `Re[E(r) e^{-iωt}]` |
| loss | `Im(eps) > 0`, equivalently `Im(n) > 0` — the sign indices are tabulated with (`n + 1j*k`) |
| gain | `Im(eps) < 0`; a solve may then legitimately return `R + T > 1`, and nothing in the library treats that as an error. A gain half-space takes Maystre's root, which is the answer for weak, conventional gain -- contested in the literature, and the wrong root for some strongly dispersive active media ([branch-cuts.md](../../theory/rcwa/branch-cuts.md#gain-near-grazing)); near grazing, where a finite beam's response depends on its width, the solve warns `GainCutoffWarning`. An active **double negative** that propagates is refused: no single-frequency rule picks its root ([negative-index-media.md](../../theory/rcwa/negative-index-media.md)) |

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
wavebit.Material(epsilon, mu=None)
```

Frozen, and a registered JAX pytree whose children are `(epsilon, mu)` — a material is a
legitimate design variable, so it holds tracers and carries gradients rather than being an
opaque leaf. **A bare number is the shorthand for `Material(epsilon=x)`** anywhere a material
is accepted; non-magnetic is the overwhelming default and costs nothing to write.

The scene is stated with four keywords:

| keyword | what it holds |
| --- | --- |
| `structure` | the layer stack, layer-major on its first axis — one of the two encodings below |
| `top`, `bot` | the two half-space materials: a `Material`, or a bare relative permittivity |
| `mu` | the **raster path only**: the permeability field, `(n_layers, Ny, Nx)`, beside `structure`. Named on a shape path it is refused, not ignored — there a permeability belongs to the `Material` filling the region |
| `thickness` | per-layer thicknesses, in the shared length unit |

The *type* of `structure` selects the modeling path; there is no separate mode switch:

| encoding | `structure` |
| --- | --- |
| raster | ndarray `(n_layers, Ny, Nx)` of **relative permittivity** — one sample per pixel |
| analytic | `[dict(background=..., shapes=[dict(type=..., cx=..., cy=..., lx=..., ly=..., angle=..., material=...)]), ...]` |

A third encoding, `[[background, [['rectangle', cx, cy, lx, ly, material, angle], ...]], ...]`,
was the block modeller and is refused by name since 2026-08-26.

Two of those three ways of saying what fills a region — the parametric layer's `background` and
each shape's `material` — are **assignments**: a name for the stuff in a region, and a medium
has both a permittivity and a permeability, which is why one type carries the pair. The raster
is the third and is not an assignment at all but a **sampled field**, one number per pixel,
which is exactly why topology optimisation reaches for it. A field has one quantity in it, so
the raster's permeability is a second array (`mu=`) rather than a second slot on an object.

### The retired spellings, and why they were refused by name

| retired | now | what it names |
| --- | --- | --- |
| `ucell` | `structure` | the layer stack |
| `n_top` / `n_bot` | `top` / `bot` | the two half-space materials |
| `n_base` / `base` (parametric layer keys) | `background` | a layer's background material |
| `n_index` (parametric shape key) | `material` | a shape's material |
| `ucell_materials` | — | deleted; it was assigned by every backend and read by nothing |

Until 2026-08-26 every one of these raised, naming its replacement **and** saying that the
quantity changed with it. That second half was the point. A user who renames the keyword and
keeps the number has not made a mistake anything downstream can see: `3.5` is a perfectly good permittivity, so
the solve runs, conserves energy, and answers a different question — the README quickstart with
every material literal left at the index it used to be sums `R` to 0.1753 where the migrated
scene gives 0.2053, with no warning. The two are close on purpose, as an argument: a migration
that broke loudly would announce itself, where a shift the size of a truncation wobble gets
believed. Silence is not available here, so the refusal is. Since 2026-08-26 the spellings are
deleted rather than refused by name: each door reports one as what it now is, an unknown keyword,
layer key or shape key, and a name spelled like an index (`n_top` handed to `solve()`, `n_index`
in a shape) is still told that every material slot takes a relative permittivity.

Two conversions are exported for callers holding the other quantity:

| function | value |
| --- | --- |
| `wavebit.eps_from_refractive_index(n, mu=...)` | `n**2 / mu` for isotropic materials; omitting `mu` gives `n**2` (relative permeability one). The index and permeability must describe the same wavelength. Assign the same `mu` to the resulting `Material`. |
| `wavebit.refractive_index_from_eps(eps)` | the **principal** root, always complex. A passive medium has `Im(eps) >= 0`, which is the closed upper half plane, whose principal root has `Im(n) >= 0` — loss again. The hazard is the negative side of zero: `complex(-2, +0.0)` gives `+1.414j` (decaying, right) and `complex(-2, -0.0)` gives `-1.414j` (gain) -- spell the zero with `complex()`, since the expression `-2 - 0j` has a `+0.0` imaginary part on Python 3.12 and 3.13 and a `-0.0` one from 3.14 (which converts the real operand of mixed real-complex arithmetic to a float, CPython gh-69639) -- and a lossless value that has been through arithmetic can come out as `4 - 1e-18j`. Neither is repaired, because nothing can tell round-off from a deliberately active medium; keep `Im(eps)` non-negative yourself before taking a root. The lip is the same on every backend: JAX's own `sqrt(complex(-1, -0.0))` is `+1j` (JAX 0.11.0) where NumPy, PyTorch and `cmath` give `-1j`, and `refractive_index_from_eps` turns JAX's root there ([negative-index-media.md](../../theory/rcwa/negative-index-media.md) §15) |

For a magnetic medium the index is `sqrt(eps)*sqrt(mu)`, each root on Maystre's cut (the
principal root for every passive factor): the principal `refractive_index_from_eps(eps*mu)` for
every ordinary medium, to the bit, and its exact negation for a negative-phase-velocity medium --
every passive double negative (`Re(eps) < 0` and `Re(mu) < 0`) and the passive magnetic metals
-- whose principal root would read as gain. A gain-metal factor (`Re < 0`, `Im < 0`) is the one
place the two roots differ, and there a magnetic medium's index -- `mu = 1` named included --
can be the negation of the non-magnetic `refractive_index_from_eps`. The
solver's `n_top` / `n_bot` are that signed index; [negative-index-media.md](../../theory/rcwa/negative-index-media.md)
§6 says which media it negates and why.

### Where the refractive index is still the quantity

The flip is about what a *material slot* holds; it does not rename the physics. The index
remains the right quantity in Snell's law and in the half-space relation
`n**2 = eps*mu`, with the sign of `n` negative in a negative-index medium. Real angles in
absorbing incident media use the full complex `eps*mu` as described under
[Real-angle incidence](#real-angle-incidence).

Concretely, `n_top` and `n_bot` still exist and still mean the refractive index — as
**derived, read-only** properties. Assigning `top` stores the material and refreshes
`n_top = sqrt(eps)*sqrt(mu)` (signed, above); assigning `n_top` raises. Two spellings of one
medium that could disagree is exactly the state the type exists to remove.

## Geometry and propagation

The public Cartesian frame is **right-handed, with +z upward**. The bottom interface of
the finite stack is **z=0**; its top interface is **z=h_total**, where `h_total=sum(thickness)`.
`bot` occupies `z<0`, and `top` occupies `z>h_total`. These half-spaces are not entries in the
finite-layer list.

Layers are listed **bottom to top**: `structure[0]` is adjacent to `bot`, and
`structure[-1]` is adjacent to `top`. `thickness[i]`, raster `mu[i]`, tensor convolution
stacks and `Stack.layers[i]` all describe the same layer. Its lower and upper heights are
`sum(thickness[:i])` and `sum(thickness[:i+1])`. Adding a layer to the end builds upward.

`incidence_side='top'` (the default) sends light downward (`kz<0` for a propagating wave in a
positive-index medium); `incidence_side='bottom'` sends it upward (`kz>0`). In a negative-index
lit medium the phase, and so `Re(kz)`, runs the other way while the light still goes down or
up. Neither changes the geometry, indexing, or returned field axes. `theta` is measured from the inward normal of the selected face:
`-z` at the top, `+z` at the bottom. `theta` and `phi` are the direction the incident light
travels toward the structure, `phi` measured from `+x` toward `+y` in the fixed public frame.
For real incident angles in a lossless medium,
`k/k0 = n_inc * (sin(theta)*cos(phi), sin(theta)*sin(phi), sigma*cos(theta))`, with
`sigma=-1` for top incidence and `+1` for bottom incidence and `n_inc` the **signed** index:
in a negative-index medium (`n_inc < 0`) the wavevector points against the light, as its phase
does. [Real-angle incidence](#real-angle-incidence) has the general (absorbing) statement.

Alternatively, specify real `kx`/`ky` directly in inverse length units, without
`theta`/`phi`. An omitted component is zero. These are Cartesian angular
wavenumbers of the incident wave, before diffraction-order offsets, and stay
fixed during a wavelength sweep. The normal component follows from
`kz² = (2*pi/wavelength)²*epsilon*mu - kx² - ky²` with the incident-side branch.
Materials remain required. `order_kxy()` reports per-order k divided by k0, so
its output uses different units from direct input. See [direction inputs](../../guide/rcwa/light.md#two-ways-to-specify-direction).

Results remain incidence-relative: `R` returns to the illuminated half-space and `T`
leaves through the opposite one. Reflection amplitudes are referenced to the illuminated
interface, transmission amplitudes to the opposite interface.

Field samples specified by count run in increasing height, bottom to top. Explicit
`sample_z` values are absolute heights from the bottom interface and retain the caller's
order. Internal interfaces belong to the **upper** layer; use `z-delta` for the lower limit.
Counts of at least two include both limits; a count of one selects each layer's lower face.
The top endpoint `h_total` belongs to the last finite
layer. This rule is independent of illumination side.

For example, `thickness=[100., 200., 50.]` gives layer spans `[0,100]`, `[100,300]`,
and `[300,350]`. The first layer is at the substrate and the last at the superstrate.

### Updating a pre-release downward-positive scene

This convention replaced top-to-bottom input and downward-positive depth on 2026-09-21.
Reverse the layer list **and** every matching thickness/material stack. Former depth `d`
becomes height `h_total-d`; a uniform count grid also reverses along z. Comparing fields in the
former downward frame uses `E -> (Ex,Ey,-Ez)` and `H -> (-Hx,-Hy,Hz)`. A tensor expressed
in that frame transforms as `Q @ tensor @ Q`, `Q=diag(1,1,-1)`; its xz/yz/zx/zy entries
change sign. Scalars and in-plane blocks do not. Do not reflect x/y patterns or change phi
when using this z-mirror comparison. The low-level propagation kernels retain their own
start/end orientation; the engine binds those ports to the public bottom/top media.

Serialized `JobSpec` documents use **version 10**. Earlier versions are refused because
changing only the version number could silently reverse an asymmetric physical stack.

Within a layer the array follows the image convention: row `0` of a raster `structure` is the
**bottom** of the cell — the `-y` end, at `y = -Py/2 + Py/(2*Ny)` — for every builder and for
the reconstructed field cell alike. (It read "row `0` is `y = 0`" until 2026-08-31, which was
loose in two ways at once: row 0's centre was never at `y = 0` even then, and the cell is
centred on the origin now.) The flip that renders a
field the conventional way up belongs to plotting (`plot_field`), not to the data.

**The reconstructed field uses this same upward-positive xyz frame.** In vacuum,
`H = khat × E`; upward power has `Sz>0` and downward power has `Sz<0`. Thus ordinary
top incidence carries negative net z flux, while bottom incidence carries positive flux.
The exp(-i omega t) time convention and exp(+i k.r) spatial convention are unchanged.
The plane-wave and interface tests in `tests/rcwa/test_vertical_coordinates.py` check these
statements across NumPy, JAX and PyTorch, both incidence sides and all supported paths.

## Units and angles

`wavelength`, `period`, `thickness` and the parametric shape coordinates (`cx`, `lx`,
…) share **one length unit of your choosing** — nothing in the solver assumes nm. Shape
coordinates are absolute lengths, not fractions of the period.

### One origin: the cell is centred on it

Every coordinate this library takes or hands back is measured from the **centre of the cell**,
`0.5*(a1 + a2)`. So a centred feature is `0`, and `0.5*(a1 + a2)` is never written by hand.

| where | what `0` means |
| --- | --- |
| a shape's `cx` / `cy` | the centre of the cell — also what omitting them means |
| `reconstruct_field`'s `sample_x` / `sample_y` | the same point; a *count* spans the cell, `[-period/2, +period/2]` |
| `wavebit.meshgrid_xy`'s `X` / `Y` | the same point; `structure[layer][j, i]` sits at `r = (u - 0.5) * a1 + (v - 0.5) * a2` for `u = (i+0.5)/Nx`, `v = (j+0.5)/Ny` |
| the axes `plot_structure` draws | the same point; they run `-P/2` to `+P/2` |

`sample_z` is the exception and deliberately so: `z` is not periodic and the stack *has* a
distinguished point, so it is an absolute height above the bottom interface (see
[Geometry and propagation](#geometry-and-propagation)).

This is S4's convention (*"the unit cell is the Wigner-Seitz cell centered at the origin"*) and
Meep's. It arrived on 2026-08-31; before it the origin was the cell's corner, and both the
README and `oblique-lattice.md` opened their raster example by computing the cell centre by
hand. Internally the Fourier machinery still integrates over `u, v` in `[0, 1]`, and
`wavebit.lattice.sample_grid` is that internal corner frame. Reaching for it from user-facing
code misplaces a structure by half a period, silently.

Since 2026-09-28 every published Fourier operator also uses the centre:
`coeff_g = (1/A) ∫ eps(r) exp(-i g·r) dr`, with `r` measured from this origin.
`eps_conv_matrices()`, `mu_conv_matrices()`, the public NVF builders and the input to
`solve_conv()` share that frame. `ConvMatrices.from_energy()` preserves the frame of its
inputs, so supply centre-frame coefficients to use its output with `solve_conv()`.
The private builders and `_solve_conv()` use the kernels' corner frame. To restate an
operator between the two, use `C * np.outer(d, d)`, where `d = (-1.)**(m+n)` in retained
order-label order. The implementation uses exact sign flips for this `D C D` similarity;
ordinary `solve()` stays entirely on the private path.

Since 2026-09-27 the **phase** of every per-order quantity is referenced here too: the
amplitudes (`r_s` .. `t_p`, `r_xyz` / `t_xyz`, `order()`), the `excitation` weights, the
scattering matrix's port variables, and the point where an ordinary incidence is unit,
`x = y = 0` on the lit face (the phase-reference bullet of
[the result surface](#the-result-surface)). Public convolution matrices followed on
2026-09-28, so every public per-order phase and every published Fourier coefficient is
referenced here. The centre belongs to the basis you wrote: a unimodular re-basis
`(a1, a2) -> (a1, a1 + a2)` moves `(a1 + a2)/2` by `a1/2`.
Keeping the same public numbers — shape centres, sample points — translates the whole scene,
phase reference included, so the field is unchanged; the same physical crystal sampled at the
same physical points (shapes and samples shifted by `-a1/2`) gives a field that differs by
`exp(-1j k_inc . a1/2)` at `theta != 0`, and no efficiency does. Both hold as the truncation
converges: the two bases retain different order sets.

All angles are **radians**.

| angle | meaning |
| --- | --- |
| `theta` | polar angle of the incident **light**, measured from the inward normal in the incidence medium: the direction its power travels in a lossless medium, of either sign of index. Real angles specify uniform incident intensity along the surface; see [Real-angle incidence](#real-angle-incidence) |
| `phi` | azimuth of the incident light's direction, from the `+x` axis |
| `psi` | polarization: the angle of **E** from the p direction in the plane transverse to **k**. `psi = pi/2` is pure s (TE-like), `psi = 0` pure p (TM-like). **Real** — a complex `psi` is refused, see below |
| `chi` | ellipticity angle: `tan(chi)` is the polarization ellipse's minor/major axis ratio and its sign is the rotation direction (`chi > 0` turns like `(1, -1j)`; see "Which way the circular states rotate" below). `chi = 0` is linear (the `psi` spelling), `|chi| = pi/4` circular. Real, like `psi`. Naming it selects the `(psi, chi)` chart — see below |
| `tetm` (1D only) | `'te'` / `'tm'` — a **declaration**, not an angle: it says this problem has no polarization and no azimuth degree of freedom, which is what makes the pure 1D TE/TM formulation legal (2-4x faster than conical, measured). Refused on a 2D cell, with `truncation_order[1] != 0`, or beside a nonzero `phi` / a `psi` / a `polarization`; and, by name, on every solve that cannot take the pure 1D path — a shape structure, a tensor material, a z row handed to `solve_conv`, `formulation='first_order'`, or no structure after anything but a pure solve (a solver holding none keeps the layout its last solve took, 2D if it never solved, and a kept pure layout re-checks `phi`, `ky`, the cell's skew and `truncation_order[1]` as the raster that chose it was checked). Since 2026-09-27; before, all but the tensor raised a `TypeError` from deep in the solve, or — with `tetm` assigned after construction — answered for s light whichever it named. `psi` reads back `None` under it — a declared-away degree of freedom is absent, not defaulted. It lights and reports exactly what `polarization=(1, 0)` / `(0, 1)` at `phi = 0` does, to ~1e-14: the same amplitudes in the same per-order s/p frame, the same `excitation` weights, the same unit-E incident field; its scattering matrix is the conical one's s-s / p-p *block*, in the same port variables. What it does not carry is the other polarization: `s_inc` / `p_inc` are `None`, and the cross-polarized amplitudes are exact zeros where the conical path has ~1e-16. Since 2026-09-27; before, it quoted `E_y` / `H_y` on every order (a sign apart on every order across `kx = 0` from the incident one), was lit by `+y` at any angle, and its TM input was a unit `H_y` |

### Real-angle incidence

For real `theta` and `phi`, the incident wave has **uniform intensity along the
structure surface**: both transverse components `kx`, `ky` are real. `theta` and `phi` refer to
the illuminated half-space itself, not to a hypothetical external medium. This describes a
single plane wave. Multiple waves supplied through `excitation` can interfere within a cell;
with real transverse wavevectors their intensity repeats from cell to cell.

For an isotropic incident material, write `epsilon_r * mu_r = a + i*b`,
`k0 = 2*pi/wavelength`, and `s = sign(Re(n))` for the signed index
`n = sqrt(epsilon_r)*sqrt(mu_r)` (`+1` for every medium except a negative-phase-velocity one,
such as a double negative). The phase magnitude `N = |Re(k)|/k0` is

$$
N^2 = \frac{a + \sqrt{a^2 + b^2/\cos^2\theta}}{2},
\qquad
k_x=s\,k_0N\sin\theta\cos\phi,\quad
k_y=s\,k_0N\sin\theta\sin\phi.
$$

`theta` and `phi` are therefore the angles of **`s * Re(k)`**: the incident wave's phase
direction, turned to point where its light goes. In a positive-index medium that is the phase
direction itself, which is what this angle has meant since real-angle incidence in absorbing
media landed. In a negative-index medium the phase advances against the power (V. G. Veselago,
Sov. Phys. Usp. 10, 509 (1968)), so the same light has its in-plane wavevector along
`phi + pi`. In a **lossless** medium of either sign `s * Re(k)` is the Poynting vector's
direction, `theta` is exactly the direction the incident power travels, and the formula reduces
to `kx = k0*n*sin(theta)*cos(phi)` with the signed `n`: Snell's law as Veselago and
J. B. Pendry (Phys. Rev. Lett. 85, 3966 (2000)) write it. Lit from `n = -sqrt(2)` at 20 degrees
onto glass, the light leaves at -18.81 degrees.

This is a convention Wavebit chose -- the direction of the incident power, the Poynting vector,
exact in a lossless medium and the angle of `s * Re(k)` in an absorbing one -- not a causal
statement: causality-based criteria for a negative index test the group velocity, which is not
the Poynting vector (P. Kinsler and M. W. McCall, Phys. Rev. Lett. 101, 167401 (2008)). No
published source found defines an incidence angle inside a double negative. PyMoosh
(D. Langevin et al., J. Opt. Soc. Am. B 41, A67 (2024)) takes the phase's angle instead, so from a
lossless double negative its `kx` at a given `theta` is the negative of Wavebit's (from an
absorbing one it is complex, with a real part of the opposite sign); compare the two through
`kx`, not `theta` ([`negative-index-media.md`](../../theory/rcwa/negative-index-media.md) §7).

`N` follows by writing the inward normal component as `q/k0 = N*cos(theta) + i*alpha` (for
`s = -1`, `-N*cos(theta) + i*alpha`) and imposing both parts of dispersion:
`N**2 - alpha**2 = a`, `2*s*N*cos(theta)*alpha = b` (`b` enters `N` only squared, so one `N`
serves both signs, and `alpha` comes out positive for every passive medium). The full complex
material is used; this is not `sqrt(real(epsilon_r))`. The normal
wavevector satisfies `kz**2 = k0**2*epsilon_r*mu_r - kx**2 - ky**2` and is the half-space's own
root (Maystre's cut, or its mirror for a double negative: [`branch-cuts.md`](../../theory/rcwa/branch-cuts.md)
§8, §9), so the incident wave decays toward the structure in an absorbing medium of either sign.
The public `kz` of an ordinary medium is negative for top incidence and positive for bottom
incidence; in a negative-index medium its real part has the other sign.

In an **absorbing** medium the Poynting vector departs from `s * Re(k)` by an amount set by the
loss and the polarization, which the angle does not try to undo (a power direction that depends
on the polarization cannot fix one `k_par` for both). At 45.8 degrees, `epsilon_r = 2+0.5i`
carries p light at 44.1 degrees and s light at 45.8; a metal (`-10+1i`) carries p light at
-45.5 degrees, across the normal ([`negative-index-media.md`](../../theory/rcwa/negative-index-media.md) §7 has
the table). The absorbing real-angle definition requires `abs(theta) < pi/2`; its wavevector
diverges at grazing. A lit medium that carries no wave at all (`b = 0`, `a <= 0`) has no real
angle, and one is refused.

**Negative-index lit side.** A half-space with `Re(eps) < 0`, `Re(mu) < 0` and
`Re(eps*mu) > 0` is solved lit or not: its root is the mirror of Maystre's cut (the decaying
one, `Re(kz) < 0` on a lossless propagating order) and its index Veselago's negative one
(`branch-cuts.md` §9, [`negative-index-media.md`](../../theory/rcwa/negative-index-media.md) §5 and §6, which
also says why `theta` names the light's direction rather than its phase's, §7). At exactly
`theta = 0` the per-order frame takes the limit `theta -> 0+` of `k_par`'s direction, which is
`phi + pi` there, so the incident field is continuous through normal incidence. An active one
(`Im(eps) < 0` or `Im(mu) < 0`) is refused where it is set.

An angle with a **nonzero imaginary part** retains the formal
`kx/k0 = n*sin(theta)*cos(phi)`, `ky/k0 = n*sin(theta)*sin(phi)`
parameterization for advanced illumination, with the same signed `n`. These complex
parameters do not have the meaning of real angles. A value such as `0.4+0j` is a real angle.
In a lossless medium the two meet as `Im(theta) -> 0`; in an absorbing one they are not an
analytic continuation of each other across the real-angle boundary (`N` against the complex
`n`); use real angle variables when differentiating the real-angle model.

When that parameterization leaves the transverse wavevector with a nonzero
imaginary part, the incident intensity varies along the surface. Evanescent
incidence, `theta = pi/2 + 1j*alpha`, is not such a case: its `kx` is real
(below). The per-order `R` and `T` are then formal ratios rather than fractions of an
incident power, and their sum can exceed 1 on a lossless stack. `power_entering`
and `absorption` are NaN, and each such solve emits `ComplexIncidenceWarning`.
Which root of `kz` the exit medium takes is a convention at a complex `kx`, and
other solvers choose differently (`docs/validation/rcwa/time-convention-audit.md` §9). This is
the only route to a complex transverse wavevector. A direct `kx`/`ky` must be
real, and refusing a complex one names this route: for `ky = 0`,
`theta = arcsin(kx/(k0*n))`, `phi = 0`. A direct `kx`/`ky` carries no angle and is taken as
given whatever the lit medium; at `kx = ky = 0` its s/p basis uses azimuth zero.

**Evanescent incidence** has a real `kx = n*cosh(alpha)` beyond the lit medium's light line.
In a lossless lit medium its `kz` is imaginary, so the incident wave carries no normal power
and nothing can be divided by it: `R`, `T` and their s/p parts, `power_entering` and
`absorption` are NaN. The amplitudes and fields are returned unchanged.

Its port powers in W are not ratios, and an `rcwa.run` result keeps them. The light does reach
the far side: the incident and specular reflected waves each carry no normal power, but their
interference does, and on a lossless stack the net flux through the lit face equals the
transmitted power, as in frustrated total internal reflection. Two per-wavelength flags in
`metadata` say which values exist. `diagnostics` carries them too, as `port_power_defined`
and `efficiency_defined`, for a run without diffraction outputs.

| | `power_defined` | `efficiency_defined` |
| --- | --- | --- |
| real `kx`, propagating incidence | true | true |
| evanescent incidence | true | false |
| complex `kx` | false | false |

`power_defined` gates the port powers: a complex `kx` has no uniform-intensity cell integral.
`efficiency_defined` gates `efficiency`, `reference['incident_power_w']`, `power_entering` and
`absorption`, and implies `power_defined`. `incident_power_w` is NaN rather than the incident
wave's true 0, because the port powers are per wave and omit the interference that carries the
tunnelled power: `incident - reflected - transmitted` would read it as a negative absorption.
The W values are per unit amplitude at the lit face of the stack. Declaring that amplitude a
distance `d` farther from the stack scales them by `exp(-2*Im(kz)*d)`, with `kz` the lit
medium's normal wavenumber. For p light, `|E|` is
`sqrt(cosh(2*alpha))` times the amplitude. True efficiencies need a propagating incident wave,
for example from a prism medium as the lit half-space, which this solves exactly.

The test is a roundoff floor, not an exact zero: `cos(pi/2)` is `6.1e-17`
in double precision, so `kx` keeps an imaginary part the transverse test accepts as roundoff,
and it leaves `Re(kz)` near `6e-17*|kx|`; divided by that, `T` once came out near `1e16`, with
its sign set by the sign of `alpha` ([#84](https://github.com/fieldalgo/wavebit/issues/84)).
No warning accompanies the NaN, which marks itself, and a `jax.jit` solve, which cannot warn,
marks the same values. An `excitation` or a direct `kx`/`ky` that lights an order carrying no
power is refused instead, wherever its value can be read. Under `jax.jit` neither refusal can
run: the solve is marked the same way, and the kernels divide by 1 in place of the zero incident
power. A loss that selects with the flag, `jnp.where(efficiency_defined, T, 0)`, therefore has
a gradient of 0 there rather than NaN, and a solve with incident power keeps its own value and
gradient. Light emitted inside a stack, where
this evanescent light matters, is a different calculation
([#85](https://github.com/fieldalgo/wavebit/issues/85)).

`order_kxy()` returns wavevectors **divided by `k0`**, including reciprocal
lattice shifts for each retained order. Multiply by `2*pi/wavelength` for
physical inverse-length units. See [Light inputs](../../guide/rcwa/light.md#what-theta-and-phi-mean)
for an example.

### General (elliptical) polarization

`polarization = (a_s, a_p)` states the incident polarization as a pair of complex
**E-field amplitudes**, in the same (s, p) basis as `psi` and the returned amplitudes:
`(1, 0)` is pure s, `(0, 1)` pure p, and `psi = v` is exactly
`polarization = (sin(v), cos(v))`. What the pair adds is the relative phase `psi` cannot
spell — `(1, -1j)/sqrt(2)` and `(1, +1j)/sqrt(2)` are the two circular states.

- `psi` and `polarization` are mutually exclusive. Naming both is an error, not an
  override.
- The pair is divided by its Euclidean norm, so the user need not supply
  `|a_s|^2 + |a_p|^2 = 1`. Its overall phase is retained. This coefficient
  normalization does not set physical power to one: the solver separately evaluates
  the incident power using the medium, polarization and illuminated orders, then
  divides reflected and transmitted powers by that value.
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
`tests/rcwa/test_uniform_anisotropic.py`: `t_s = E_y`, `t_p = -E_x`). The pair
`(1, -1j)/sqrt(2)` is then `E ∝ x̂ - i·ŷ` up to a global phase, and in this library's
`e^{-iωt}`:

```
E(t) = Re[(x̂ - i·ŷ)·e^{-iωt}] ∝ x̂·cos(ωt) - ŷ·sin(ωt)
```

whose rotation vector is `-ẑ`. At top incidence the wave travels `-z`, so these are
parallel; at bottom incidence the wave travels `+z`, so they are antiparallel. The
Cartesian normal-incidence basis above is the same on both sides; handedness relative to
propagation therefore depends on the illuminated side. Explicitly:

| pair | rotation vector | top incidence (`k` downward) | bottom incidence (`k` upward) |
| --- | --- | --- | --- |
| `(1, -1j)/sqrt(2)` | `-z` | parallel to k: optics LCP, IEEE RHCP | antiparallel to k: optics RCP, IEEE LHCP |
| `(1, +1j)/sqrt(2)` | `+z` | antiparallel to k: optics RCP, IEEE LHCP | parallel to k: optics LCP, IEEE RHCP |

`chi` follows the same table, whatever `psi`: `chi > 0` turns like `(1, -1j)/sqrt(2)` and
`chi < 0` like `(1, +1j)/sqrt(2)` (at normal incidence the chart's rotation vector lies along
`-sin(2*chi) ẑ`).

Until 2026-09-21 `+z` pointed down into the stack and top incidence travelled `+z`, so the
top column read the other way round: `(1, -1j)/sqrt(2)` was antiparallel, optics RCP. The
amplitudes did not change when `+z` turned upward; the frame did, and with it the physical
field every pair names. Tutorial 6 kept the old names until 2026-09-27, with the suite green.

Three anchors keep this paragraph honest rather than derived-once-and-drifting: the basis
calibration above, the gyrotropic eigenstate test
(`tests/rcwa/test_polarization.py::test_each_circular_pair_is_one_faraday_eigenstate`), which
pins that `(1, -1j)/sqrt(2)` is the state seeing `ε_d + γ` in a layer with
`eps = [[ε_d, iγ], [-iγ, ε_d]]`, and the table itself measured on fields
(`tests/rcwa/test_polarization.py::test_a_circular_state_turns_as_the_table_says` and
`::test_the_solved_field_turns_the_same_way`): `Re[E e^{-iωt}]` traced over one period
against the Poynting vector, from both sides. Only the third can see a mirrored frame: the
first two live in the `xy` plane, and relabelling the frame `z -> -z` leaves them alone, as
it leaves every amplitude, efficiency and curl residual, while it reverses every helicity.
And the usual seam warning applies to Jones vectors exactly as it does to every other
amplitude: a pair quoted from an `e^{+jωt}` source is the complex conjugate of ours, and its
handedness labels flip with it.

## Diffraction orders

The retained orders are `m = -half_width_u..half_width_u`, `n = -half_width_v..half_width_v`, with normalized in-plane
wavevectors `kx_m/k0 = kx_inc/k0 + m*lambda/period_x` on an axis-aligned cell
(and correspondingly for `ky_n`). The incident components follow
[Real-angle incidence](#real-angle-incidence) using `top` for top incidence and `bot`
for bottom incidence. For a lossless incidence medium this is the usual
`kx_inc/k0 = n_inc*sin(theta)*cos(phi)`, with the signed `n_inc`. Per-order
arrays have shape `(n_orders_v, n_orders_u)` — **`y` first, the transpose of the `truncation_order=[x, y]` input
order** — with the incident order at the centre `[half_width_v, half_width_u]` and `m` increasing along
the last axis.

`result.order(m, n)` indexes all per-order quantities by the order indices themselves, so
user code never needs the previous paragraph.

### The two rules, and where they meet

Measured across the whole public surface, there are exactly two and neither has an exception.

| | order |
|---|---|
| **array axes** | `v` first — `structure`, `meshgrid_xy`, `R`, `T`, `r_s`, `reconstruct_field`, `slice_structure`, and `S11`'s flat index |
| **hand-written pairs** | `x` first — `period`, `truncation_order`, `order(m, n)` (its `m` is `u`) |

Neither is this package's choice. Array axes are numpy's — rows first, which is also what
`np.meshgrid` gives by default and what typing a matrix gives, so a raster written out reads
like the picture it is. Hand-written pairs are what people type, and every solver that takes
one takes it x-first.

They meet at `truncation_order`, and that meeting is where readers go wrong.
`truncation_order=[4, 1]` gives `R.shape == (3, 9)`: the leading 4 sizes the **trailing** axis.
It reads worse with figures like `[5, 2] -> (5, 11)`, where the repeated 5 looks like
confirmation of the wrong reading.

Two things answer it, at the two ends. Neither moves an array.

```python
truncation_order = dict(u=4, v=1)      # no position to misread
period           = dict(x=700., y=1100.)

r.orders_u                             # [-4 .. 4], the order each *column* of R is
r.orders_v                             # [-1, 0, 1], the order each *row* is
R.shape == (len(r.orders_v), len(r.orders_u))
```

Zipping the labels the wrong way round is a length mismatch rather than a plausible number.
They are read from the **retained** window, so an `excitation` that moved or widened it is
already accounted for.

`u` and `v` rather than `x` and `y` because the order axes follow the cell: `u` runs along
`a1`, `v` along `a2`, and they *are* x and y on a rectangular cell — on a 60-degree hex,
`m += 1` moves `(dkx, dky)` by `(+0.600, -0.346)`, neither axis. `truncation_order=dict(x=, y=)`
is therefore read as `u`/`v` on a skewed cell, with an `AxisNameWarning`: the solve is the one
you asked for and only the names were off. `period` needs no such note — it is the axis-aligned
spelling of a cell, and a skewed one arrives as `lattice=` instead.

## The result surface

**Uppercase is power, lowercase is amplitude.**

| name | meaning |
| --- | --- |
| `R`, `T` | per-order diffraction efficiency: power returned to the incidence medium / leaving through the far side, as a fraction of incident power |
| `R_s`, `R_p`, `T_s`, `T_p` | the same, resolved by outgoing polarization; `R = R_s + R_p` |
| `power_entering` | scalar net power entering the finite stack, including incident/reflected interference, divided by incident-only power at the illuminated interface |
| `absorption` | scalar absorbed power in all finite layers divided by incident-only power: `power_entering - T.sum()`; excludes both half-spaces |
| `r_s`, `r_p`, `t_s`, `t_p` | complex **Fresnel amplitudes**: E-field ratios against a unit incident E-field, directly comparable to the textbook Fresnel/Airy coefficients (pinned to 1e-12 against them, absorbing media included) |
| `r_xyz`, `t_xyz` | the same outgoing waves as **Cartesian E vectors**, shape `(n_orders_v, n_orders_u, 3)` with `(Ex, Ey, Ez)` last: `r_s * s_hat + r_p * p_hat` in the fixed public frame. No per-order frame, so continuous where an order's `kx` crosses zero |
| `incident`, `s_inc`, `p_inc` | the response to the actual incident state, plus the pure-s and pure-p variants computed by coupled solves; omitting polarization selects pure s and still populates `incident` |

`result.r_s`, `result.r_p`, `result.t_s`, and `result.t_p` read the corresponding
arrays on `result.incident`, without copying them. `result.R` and `result.T` also
read `result.incident`. The suffixes on amplitudes name the outgoing polarization;
`result.s_inc` and `result.p_inc` name the incident polarization.

Only a manually constructed `Result(incident=None, s_inc=..., p_inc=...)` averages
the available variants' powers. Its direct amplitude properties raise an error:
an incoherent power average has no single complex electric-field amplitude.

**Which of these rows exist is what `answers` declares**, and the table above is the default
(`answers='amplitude'`) in full. `answers='efficiency'` keeps `R`, `T`,
`power_entering`, and `absorption`, and leaves every
other slot `None`, including `s_inc` / `p_inc` — so on that rung there is no
unpolarized average to take and `incident` is the only incidence. `answers='field'` adds nothing
here: what it adds is the per-layer state `reconstruct_field` walks, which is not part of the
result surface at all, and below that rung reconstruction is refused by name rather than
served. The ladder is monotone, so a lower rung is always a *smaller* answer and never a
different one — pinned bit-for-bit, not to a tolerance.

Amplitude fine print, stated once so it never has to be discovered:

- each order is quoted in **its own** s/p frame, on every path — the pure 1D `tetm` path
  included since 2026-09-27 — the standard one: `s_hat = z_hat x k_hat_par`,
  with `k_par` that order's in-plane wavevector. `p_hat` is fixed by its in-plane part, which
  lies along `k_hat_par` with length `cos(theta_m)` in the medium the wave travels in —
  **`+` for a reflected order, `-` for a transmitted order and for the incident wave**, on
  either lit side (continued onto the decaying branch for an evanescent order). That is what
  makes the normal-incidence basis `E = (-a_p, a_s)` on both sides, as the circular-state
  derivation above uses it. The price is the triad's handedness, which follows the lit side:
  `p_hat = k_hat x s_hat` (so `(s_hat, p_hat, k_hat)` is right-handed) lit from below, and
  `p_hat = s_hat x k_hat` lit from above. The amplitudes did not move when `+z` turned upward
  on 2026-09-21 (byte-identical across that change); this sentence used to say
  "right-handed" for both sides, which had been true of top incidence in the former downward
  frame and has been true of bottom incidence since. Pinned against the reconstructed field,
  both sides and all four amplitudes, in `tests/rcwa/test_amplitude_sp_frame.py`.
  Until 2026-08-16 every order with `Re(kx) < 0` came back in the *reversed* frame
  (`-s_hat`, `-p_hat`) on the conical and 2D paths — invisible in `R`/`T`, which are
  `|r|^2`, and in the reconstructed field, and therefore in every golden this project has.
  See [`amplitude-sp-frame.md`](../../theory/rcwa/amplitude-sp-frame.md);
- **on the `ky == 0` row** (every 1D solve at `phi = 0`, `tetm` or not) `s_hat_m` is
  `sign(Re kx_m) * y_hat` — `+y_hat` at `kx_m == 0`, the origin rule — and `p_hat_m` turns
  over with it. So an order's `r_s` / `r_p` changes sign where its own `kx` crosses zero while
  `R` sails through: `k_par` passes through zero there and `z_hat x k_hat_par` has no limit.
  That step is the frame's singular point, not physics; `r_xyz` is continuous there. An
  order whose `kx` is zero in exact arithmetic — normal diffraction: a direct `kx = 2*pi/P`,
  or `theta = asin(lambda/P)`, puts order `-1` there — sits on that point, and which side of
  it the solve lands on is decided by rounding: the same input gives `kx(-1) = +0.0` on NumPy
  and JAX and `-5.6e-17` on Torch, and on Torch `theta = asin(lambda/P)` and
  `theta = asin(0.375)` land on opposite sides (measured on
  `tests/rcwa/test_amplitude_sp_frame.py`'s grating). So on such an order `r_s` / `r_p` can differ
  in sign between backends and between spellings of the same input, on the conical path as on
  the pure one; `R` / `T` do not, and neither does `r_xyz`, which is the quantity to read
  there. At `theta = 0` it is the incident
  order that sits on it, and the incident field itself turns over as `theta` crosses zero
  (`s_hat_inc` goes from `-y` to `+y`); a sweep through `theta = 0` that wants a continuous
  field multiplies by `sign(Re kx_inc)`, order 0's `kx` — at index `half - centre` of
  `solver.order_kxy()[0]` for `(centre, half) = result.order_window[0]`, which is `n // 2`
  only while no `excitation` has moved the window. On a mirror-symmetric grating at
  `theta = 0` this makes `r_{-m} = -r_{+m}`. The pure 1D path quoted `E_y` / `H_y`
  on every order until 2026-09-27 (`new = sign(kx_m) * sign(kx_inc) * old`), which RETICOLO's
  own 1D path still does: its 1D and conical amplitudes differ by exactly that factor;
- an `excitation` weight multiplies **its own order's** `s_hat` (s) or `p_hat` (p), on every
  path — so on the `ky == 0` row a lit order with `kx < 0` arrives with its s field along
  `-y`. The pure 1D path did the same from 2026-09-27; before, it multiplied `+y` into every
  lit order, which is different light once the lit set straddles `kx = 0`. A weight is the
  amplitude of its plane wave at `x = y = 0` on the lit face — the weights
  `OrderExcitation(orders=...)` takes, and the same light (since 2026-09-27; see the phase
  reference below);
- the scattering matrix's **s** block holds amplitudes: `S11 @ delta` is `r_s`, `S21 @ delta`
  is `t_s`. Its **p** block holds port variables — the p wave's magnetic field,
  `-1j * (n/mu) * a_p` in its own medium — so `S11 @ delta` is `-r_p` (that field turns over
  for the up-going wave) and `S21 @ delta` is `t_p * (n/mu)_exit / (n/mu)_lit`. Same on every
  path; the pure 1D matrix is the s-s (TE) or p-p (TM) block of the conical one at `phi = 0`,
  and its ports say which: `Port.channel` is `'te'` / `'tm'` there and `'sp'` on every coupled
  path, and `cascade` refuses to join two channels;
- at `theta = 0` the zeroth order has no in-plane wavevector to take a direction from, and
  `phi` supplies one — as a **direction**, not a line — the limit `theta -> 0+` of `k_par`'s
  own direction: `phi` itself, or `phi + pi` from a negative-index lit side, where `k_par`
  points against the light (so the field is continuous through normal incidence there too). `phi` and `phi + pi` therefore
  describe incident fields of opposite sign for the same `psi`, and the solve returns the
  negated field and amplitudes to match; they are the same illumination only up to that
  sign. Equivalently, `(phi + pi, psi)` is `(phi, polarization=(-a_s, -a_p))`. Until
  2026-08-16 the two spellings disagreed, and the `phi + pi` one silently answered for
  `phi`;
- the p sign convention is the one where a single interface at normal incidence gives
  `r_p = (n2 - n1)/(n2 + n1) = -r_s` (non-magnetic media), and `r_p = -r_s`, `t_p = t_s` for
  any media, double negatives and magnetic metals included: every `n` in a `p_hat` is the
  signed index, so `n/mu` keeps a positive real part;
- `r_xyz` / `t_xyz` are the frame-free spelling of the same waves, per unit incident amplitude
  exactly as `r_s` is, in the public frame (`+z` up):

  ```text
  r_xyz[m] = r_s[m] * s_hat_m + r_p[m] * p_hat_m^ref     (incidence medium n_i, kz_i)
  t_xyz[m] = t_s[m] * s_hat_m + t_p[m] * p_hat_m^trn     (exit medium n_t, kz_t)

  s_hat      = (-s, c, 0)                                 (c, s) = the order's azimuth
  p_hat^ref  = ( kz_i c/n_i,  kz_i s/n_i, d*rho/n_i)      in-plane +cos(theta_m) k_hat_par
  p_hat^trn  = (-kz_t c/n_t, -kz_t s/n_t, d*rho/n_t)      in-plane -cos(theta_m) k_hat_par
  rho = kx c + ky s,   d = -1 lit from the top, +1 from the bottom
  ```

  Each `p_hat` is perpendicular to its own `k = (kx, ky, +-kz)` in the non-conjugated sense,
  lossy media included. They share `r_s`'s phase reference — in `z` the lit and far
  interfaces, and laterally the cell **centre** (below): the outgoing E on the lit face is
  `sum_m r_xyz[m] exp(1j k_m . x)` in public coordinates — and they are continuous where an
  order's `kx` crosses zero on the `ky == 0` row, where `r_s` turns over; the incident order's
  own crossing (`theta` through 0) still turns them over, because that is the incidence
  changing, not a quotation frame. The magnetic vector is not shipped: it is
  `(k/k0) x E / mu` in Z0 units with `k/k0 = (kx, ky, +-kz)` — read `kz` off the solve, never
  respell it — and `boundary_fields(part='outgoing')` returns it in space. RETICOLO's `res2`
  returns the same Cartesian quantities per order, E and H;
- phase reference: `r` at the incidence-side interface, `t` at the far one (top and
  bottom respectively for the default illumination). **Laterally, the cell centre**, `x = y =
  0`, the origin of every public coordinate, for every ordinary solve — pure 1D TE/TM,
  conical, 2D in either formulation, ETM and SMM, either side, any cell:
  - the incident field on the lit face is `sum_m w_m e_m exp(1j k_m . x)`, `w` the
    `excitation` weights (one unit wave on the specular order by default) and `e_m` the
    order's own `s_hat` / `p_hat` combination: **unit at the centre**. It is
    `OrderExcitation(orders=w)`'s field, and by default `PlaneWave`'s;
  - the outgoing field on the lit face is `sum_m r_xyz[m] exp(1j k_m . x)`, and `t_xyz` the
    same on the far face: every amplitude is a Fourier coefficient of the public field about
    the centre;
  - the scattering matrix maps incoming port amplitudes to outgoing ones in those same
    variables — its port variables are referenced at the centre too, so the block readings
    above (`S11 @ delta` is `r_s`, or `-r_p` on the p block) hold on every order;
  - every field output — `reconstruct_field` on the solver and on the result,
    `boundary_fields`, `solve_field`, `plot_field` — is the physical field of that
    incidence.

  Until 2026-09-27 the ordinary path referenced all of it at the cell **corner**, the Fourier
  origin the solves integrate from: against the centre, order `(m, n)` carries
  `exp(1j k_mn . (a1 + a2)/2) = exp(1j k_inc . (a1 + a2)/2) (-1)**(m + n)` on any cell, so on a
  single wave every amplitude with `m + n` odd changed sign that day, `excitation` physics
  changed wherever the lit orders mix parity, and a field changed by
  `exp(-1j k_inc . (a1 + a2)/2)`. `R` and `T` of a single wave did not move
  (`amplitude-sp-frame.md` §7). The kernels still integrate from the corner and
  `solver.T1` / `layer_info_list` are their own state, including their corner-frame
  `epz_conv_i`; the engine converts at its doors. For a
  complex `theta` the incident wave is not uniform along the surface, and "unit at the centre"
  fixes its magnitude there — 7.70 times smaller than at the corner on a TE scene at
  `theta = 0.3 + 0.1j` in a lossy superstrate;
- for real transverse wavevectors, `R` and `T` are the separate reflected/transmitted
  powers divided by the incident wave's power at the illuminated interface. This also
  applies to absorbing incident media. `T` refers to power crossing the opposite
  interface, not power surviving infinitely far into an absorbing exit medium.
  Incident/reflected interference contributes to `power_entering`. The finite stack's
  absorption is `result.absorption = result.power_entering - result.T.sum()`.
  With a lossless incident medium and propagating illumination,
  `power_entering = 1 - R.sum()`, giving `absorption = 1 - R.sum() - T.sum()`.
  This absorption is zero only if the finite stack also has no net absorption or gain.
  General complex-transverse-wavevector cell-integrated powers are outside this
  definition: `power_entering` and `absorption` return `NaN` in that case;
  complex-angle amplitude calculations remain separate;
- there are no raw "boundary variables" in the public surface. (Before the first release
  the p amplitudes were exposed as magnetic-field boundary variables needing a
  `/(-i*n_top)` footnote; they were renamed and normalized outright, with no aliases —
  including the meent-era efficiency names `de_ri`/`de_ti`.)

## Backend names

| `backend` | computes in |
| --- | --- |
| `'numpy'` (the default) | NumPy — no automatic differentiation |
| `'jax'` | JAX — differentiable; `jax.jit` takes a frozen geometry |
| `'torch'` | PyTorch — differentiable, eagerly |

Each name is its library's import name, and it is the only spelling taken: every door that
takes a backend -- `wavebit.rcwa` (keyword or first positional argument), the solver classes
(keyword or `solver.backend = ...`), `CoherentSolver`, `meshgrid_xy` / `meshgrid_uv` and a
spec's `numerics.backend` -- refuses anything else (`'JAX'`, `'pytorch'`, `True`, an integer)
with the list of names. `wavebit.Backend` is the same three as a `str` enum, so
`Backend.JAX == 'jax'`, and `solver.backend` holds the member. The names were the integers
`0` / `1` / `2` until 2026-10-11.

## Where each row is pinned

| rows | tests |
| --- | --- |
| time/loss, amplitudes | `tests/rcwa/test_lossy_media.py` (Airy oracle, absorbing media both sides); a **gain** film between passive half-spaces -- every formulation and layer algorithm, both sides, and the film's negative absorption against the medium's own work `k0 Im(eps) int |E|^2 dz` -- `tests/rcwa/test_gain_film.py` |
| **which quantity a material number is** | `tests/materials/test_material_convention_anchor.py` — every expected value there comes from a closed-form Fresnel/Airy recursion written in `eps` and `kz`, so it cannot be co-migrated into agreement with a flip. One function in it, `_api_material`, knows the convention; nothing else in the file may move when the convention does. The rest of the suite cannot pin this row at all: a golden regenerated by a script that imports wavebit agrees with whatever the package now believes, and reading `3.5` as the other quantity conserves energy exactly |
| the `Material` type: the scalar shorthand, both values as pytree children (so a permittivity differentiates), and the two conversions including `refractive_index_from_eps`'s branch and its signed-zero hazard | `tests/materials/test_material.py` |
| which rows exist at all, per `answers` rung | `tests/api/test_answers_declaration.py` — each rung's surface, the rungs agreeing bit-for-bit on what they share, and the refusal naming the declaration |
| the per-order s/p frame the amplitudes are quoted in | `tests/rcwa/test_amplitude_sp_frame.py` — a transposed twin, the reconstructed field as arbiter (all four amplitudes, every path, the pure 1D one included, both lit sides and both angle signs), continuity across `kx = 0` off the `ky == 0` row, and §4: the pure 1D path against the conical one amplitude for amplitude and field for field, and the sign step on the `ky == 0` row pinned as a step. The `excitation` weights' frame: `tests/rcwa/test_excitation.py`; the p block's reading and the 1D matrix as the conical block: `tests/rcwa/test_scattering_core.py`. Efficiencies cannot pin this row at all: a reversed frame is `\|r\|^2`-invariant |
| `r_xyz` / `t_xyz` | `tests/rcwa/test_cartesian_amplitudes.py` — an independent spelling of the formula on every path (a 60-degree cell included) and for each incidence variant against its own amplitudes, the reconstructed field decomposed at both faces with `E_z` held to transversality, continuity across an order's own `kx = 0`, one vector on every backend at an order's exact `kx = 0`, the three 1D paths agreeing, the power identity, the `answers` rung, `jit` / gradients, and `order(m, n)` on a widened window moved along both axes |
| the lateral phase reference (the cell centre) | `tests/rcwa/test_centre_reference.py` — an ordinary solve against `PlaneWave` on every path and both sides through both field doors and every boundary wave, `excitation` against `OrderExcitation` with the same weights, the reconstructed field decomposed about the centre on both faces, a first-order Born anchor read through `r_xyz`, all four coupled scattering blocks and a cascade across an odd label shift, complex `theta`, gradients and backends. Efficiencies cannot pin this row either: a per-order sign is `\|r\|^2`-invariant |
| order layout, order-index sign | `tests/references/test_reticolo_golden.py` (mirrors orders if wrong, conserving energy) |
| `incidence_side`, incidence-relative results | `tests/rcwa/test_backside_incidence.py` (reversed-stack Airy oracle, reciprocity both halves) and the `inc_bottom` cases of `tests/references/test_reticolo_golden.py` |
| result naming, `order()`, delegation | `tests/api/test_result_api.py` |
| the `polarization` pair: `psi` equivalence, unit-flux normalization, circular states | `tests/rcwa/test_polarization.py` |
| `psi` and `chi` are real; the chart reaches the circular states; both coordinates and the pair carry gradients through a pytree | `tests/rcwa/test_polarization.py` (the refusal on all three backends, and the pair's gradient checked against the `psi` path's rather than a frozen number) |
| the circular states' rotation sense | measured on the incident field against its Poynting vector, both sides, pair and chart, conical and 2D (`tests/rcwa/test_polarization.py`: `test_a_circular_state_turns_as_the_table_says`, `test_the_solved_field_turns_the_same_way`); derived above from the (s, p)→(x, y) calibration (`tests/rcwa/test_uniform_anisotropic.py`) and the gyrotropic eigenstate assignment (`tests/rcwa/test_polarization.py`), neither of which can see a mirrored frame |
| square-root branches behind `kz` | `tests/rcwa/test_branch_selection.py` |
| negative-index media: the signed index, the mirror root, what `theta` means on a lit negative-index side, the frame at `theta = 0` there, the refusal of an active one | `tests/rcwa/test_negative_index_half_space.py` (closed-form Fresnel both sides, the incident Poynting vector against `theta` and `phi`, negative refraction, a thick lossy slab, reciprocity, backends, `jit`), `tests/rcwa/test_negative_index_layers.py` (layers, ETM against SMM), `tests/references/test_fmmax_dng_golden.py` (fmmax, per order, through the `theta` door) |
| row `0` of a raster is the `-y` end | `tests/references/test_fmmax_field_golden.py::test_fmmax_can_see_which_end_of_the_raster_is_minus_y` — an A/B rather than a tolerance, because the agreement on a 2D patterned field is 3e-2: fmmax's field is 13x to 30x nearer the shipped row order than the reversed one. Unnamed, `tests/references/test_fmmax_dng_golden.py` and the `cross_eps6_hex` case of `tests/references/test_fmmax_laurent_golden.py` hold it too: reading the rows reversed where the engine takes a raster fails 12 and 4 of their cases (measured 2026-10-14). Every other outside raster here is y-symmetric or y-invariant, so reversing its rows is a no-op on the array |
| the field cell's `+z` (that `Ez`, `Hx`, `Hy` share the frame the phase and layer order use) | `tests/rcwa/test_field_core.py` — the curl residual **and its convergence rate**, per path, plus the control that the pre-2026-08-14 frame fails it. Not the legacy golden, which was frozen in that frame and is transformed at the comparison |
| the whole surface, per backend | `tests/rcwa/test_golden_legacy_parity.py` |
| backend names | `tests/api/test_backend_names.py` — the names accepted and every other spelling refused, with the list, at every door; `solver.backend` the member; the spec writing the name; the manifest publishing it |

Why these choices: [`time-convention-audit.md`](../../validation/rcwa/time-convention-audit.md) (the branch
rules and the flip residue), [`formulation.md`](../../theory/rcwa/formulation.md) (the full derivation in
this convention), [`branch-cuts.md`](../../theory/rcwa/branch-cuts.md) (single precision limits).

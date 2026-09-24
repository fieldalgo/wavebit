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

### The retired spellings, and why they are refused by name

| retired | now | what it names |
| --- | --- | --- |
| `ucell` | `structure` | the layer stack |
| `n_top` / `n_bot` | `top` / `bot` | the two half-space materials |
| `n_base` / `base` (parametric layer keys) | `background` | a layer's background material |
| `n_index` (parametric shape key) | `material` | a shape's material |
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
| `wavebit.eps_from_index(n, mu=...)` | `n**2 / mu` for isotropic materials; omitting `mu` gives `n**2` (relative permeability one). The index and permeability must describe the same wavelength. Assign the same `mu` to the resulting `Material`. |
| `wavebit.index_from_eps(eps)` | the **principal** root, always complex. A passive medium has `Im(eps) >= 0`, which is the closed upper half plane, whose principal root has `Im(n) >= 0` — loss again. The hazard is the negative side of zero: `-2 + 0j` gives `+1.414j` (decaying, right) and `-2 - 0j` gives `-1.414j` (gain), and a lossless value that has been through arithmetic can come out as `4 - 1e-18j`. Neither is repaired, because nothing can tell round-off from a deliberately active medium; keep `Im(eps)` non-negative yourself before taking a root |

For a magnetic medium the index is `index_from_eps(eps*mu)`, written out rather than hidden in
a keyword, because that is where a forgotten `mu` would otherwise hide.

### Where the refractive index is still the quantity

The flip is about what a *material slot* holds; it does not rename the physics. The index
remains the right quantity in Snell's law and in the half-space relation
`n = sqrt(eps*mu)`. Real phase angles in absorbing incident media use the full
complex `eps*mu` as described under [Phase incidence](#phase-incidence).

Concretely, `n_top` and `n_bot` still exist and still mean the refractive index — as
**derived, read-only** properties. Assigning `top` stores the material and refreshes
`n_top = sqrt(eps*mu)`; assigning `n_top` raises. Two spellings of one medium that could
disagree is exactly the state the type exists to remove.

## Geometry and propagation

The public Cartesian frame is **right-handed, with +z upward**. The bottom interface of
the finite stack is **z=0**; its top interface is **z=h_total**, where `h_total=sum(thickness)`.
`bot` occupies `z<0`, and `top` occupies `z>h_total`. These half-spaces are not entries in the
finite-layer list.

Layers are listed **bottom to top**: `structure[0]` is adjacent to `bot`, and
`structure[-1]` is adjacent to `top`. `thickness[i]`, raster `mu[i]`, tensor convolution
stacks and `Stack.layers[i]` all describe the same layer. Its lower and upper heights are
`sum(thickness[:i])` and `sum(thickness[:i+1])`. Adding a layer to the end builds upward.

`incidence_side='top'` (the default) sends light downward (`kz<0` for a propagating wave);
`incidence_side='bottom'` sends it upward (`kz>0`). Neither changes the geometry, indexing,
or returned field axes. `theta` is measured from the inward normal of the selected face:
`-z` at the top, `+z` at the bottom. `phi` describes the in-plane wavevector from `+x`
toward `+y` in the fixed public frame. For real incident angles in a lossless positive-index medium,
`k/k0 = n_inc * (sin(theta)*cos(phi), sin(theta)*sin(phi), sigma*cos(theta))`, with
`sigma=-1` for top incidence and `+1` for bottom incidence.

Alternatively, specify real `kx`/`ky` directly in inverse length units, without
`theta`/`phi`. An omitted component is zero. These are Cartesian angular
wavenumbers of the incident wave, before diffraction-order offsets, and stay
fixed during a wavelength sweep. The normal component follows from
`kz² = (2*pi/wavelength)²*epsilon*mu - kx² - ky²` with the incident-side branch.
Materials remain required. `order_kxy()` reports per-order k divided by k0, so
its output uses different units from direct input. See [direction inputs](guide/light.md#two-ways-to-specify-direction).

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
The plane-wave and interface tests in `tests/test_vertical_coordinates.py` check these
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
distinguished point, so it is absolute from the top (see below).

This is S4's convention (*"the unit cell is the Wigner-Seitz cell centered at the origin"*) and
Meep's. It arrived on 2026-08-31; before it the origin was the cell's corner, and both the
README and `oblique-lattice.md` opened their raster example by computing the cell centre by
hand. Internally the Fourier machinery still integrates over `u, v` in `[0, 1]` — the offset is
a relabeling, so no coefficient moves — and `_core.lattice.sample_grid` is that internal frame.
Reaching for it from user-facing code misplaces a structure by half a period, silently.

All angles are **radians**.

| angle | meaning |
| --- | --- |
| `theta` | phase polar angle, measured from the inward normal in the incidence medium; real angles specify uniform incident intensity along the surface |
| `phi` | azimuth of the plane of incidence, from the `+x` axis |
| `psi` | polarization: the angle of **E** from the p direction in the plane transverse to **k**. `psi = pi/2` is pure s (TE-like), `psi = 0` pure p (TM-like). **Real** — a complex `psi` is refused, see below |
| `chi` | ellipticity angle: `tan(chi)` is the polarization ellipse's minor/major axis ratio and its sign is the rotation direction. `chi = 0` is linear (the `psi` spelling), `|chi| = pi/4` circular. Real, like `psi`. Naming it selects the `(psi, chi)` chart — see below |
| `tetm` (1D only) | `'te'` / `'tm'` — a **declaration**, not an angle: it says this problem has no polarization and no azimuth degree of freedom, which is what makes the pure 1D TE/TM formulation legal (2-4x faster than conical, measured). Refused on a 2D cell, with `truncation_order[1] != 0`, or beside a nonzero `phi` / a `psi` / a `polarization`. `psi` reads back `None` under it — a declared-away degree of freedom is absent, not defaulted |

### Phase incidence

For real `theta` and `phi`, the incident wave has **uniform intensity along the
structure surface**. Both transverse components `kx`, `ky` are real. `theta`
specifies the direction of `Re(k)` in the illuminated half-space, not a direction
in a hypothetical external medium and not the Poynting-vector direction.
The latter can differ from the phase direction in an absorbing medium.
This describes a single plane wave. Multiple waves supplied through `excitation`
can interfere within a cell; with real transverse wavevectors their intensity
repeats from cell to cell.

For an isotropic incident material, write `epsilon_r * mu_r = a + i*b` and
`k0 = 2*pi/wavelength`. The phase magnitude `N = |Re(k)|/k0` is

$$
N^2 = \frac{a + \sqrt{a^2 + b^2/\cos^2\theta}}{2},
\qquad
k_x=k_0N\sin\theta\cos\phi,\quad
k_y=k_0N\sin\theta\sin\phi.
$$

This follows by writing the inward normal component as
`q/k0 = N*cos(theta) + i*alpha` and imposing both parts of dispersion:
`N**2 - alpha**2 = a`, `2*N*cos(theta)*alpha = b`.
The full complex material is used; this is not `sqrt(real(epsilon_r))`.
The normal wavevector satisfies `kz**2 = k0**2*epsilon_r*mu_r - kx**2 - ky**2`.
The public `kz` is negative for ordinary top incidence and positive for bottom
incidence, with the attenuation sign following that direction.

For a positive-index lossless incident medium, `b=0` and `a>0`, giving
`N=sqrt(epsilon_r*mu_r)` and the usual angle formula exactly. The absorbing
phase-angle definition requires `abs(theta) < pi/2`; its wavevector diverges at
grazing. The formula describes inward phase propagation; negative-index
half-space branch selection is a separate unsupported case.

An angle with a **nonzero imaginary part** retains the formal
`kx/k0 = n*sin(theta)*cos(phi)`, `ky/k0 = n*sin(theta)*sin(phi)`
parameterization for advanced illumination. These complex parameters do not
have the meaning of real phase angles. A value such as `0.4+0j` is a real
phase angle. The two interpretations are not an analytic continuation across
the real-angle boundary in an absorbing medium; use real angle variables
when differentiating the phase-incidence model.

`order_kxy()` returns wavevectors **divided by `k0`**, including reciprocal
lattice shifts for each retained order. Multiply by `2*pi/wavelength` for
physical inverse-length units. See [Light inputs](guide/light.md#phase-direction-and-absorbing-incidence)
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
`tests/test_uniform_anisotropic.py`: `t_s = E_y`, `t_p = -E_x`). The pair
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
| `(1, -1j)/sqrt(2)` | `-z` | parallel to k | antiparallel to k |
| `(1, +1j)/sqrt(2)` | `+z` | antiparallel to k | parallel to k |

Two anchors keep this paragraph honest rather than derived-once-and-drifting: the basis
calibration above, and the gyrotropic eigenstate test
(`tests/test_polarization.py::test_each_circular_pair_is_one_faraday_eigenstate`), which
pins that `(1, -1j)/sqrt(2)` is the state seeing `ε_d + γ` in a layer with
`eps = [[ε_d, iγ], [-iγ, ε_d]]`. And the usual seam warning applies to Jones vectors
exactly as it does to every other amplitude: a pair quoted from an `e^{+jωt}` source is
the complex conjugate of ours, and its handedness labels flip with it.

## Diffraction orders

The retained orders are `m = -half_width_u..half_width_u`, `n = -half_width_v..half_width_v`, with normalized in-plane
wavevectors `kx_m/k0 = kx_inc/k0 + m*lambda/period_x` on an axis-aligned cell
(and correspondingly for `ky_n`). The incident components follow
[Phase incidence](#phase-incidence) using `top` for top incidence and `bot`
for bottom incidence. For a lossless incidence medium this is the usual
`kx_inc/k0 = n_inc*sin(theta)*cos(phi)`. Per-order
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
| `given`, `s_inc`, `p_inc` | the incidence as specified (`psi`, or the `polarization` pair), plus the pure-s and pure-p variants the coupled solves produce for free; top-level `R`/`T` average the two variants when no specified solve exists (unpolarized illumination) |

**Which of these rows exist is what `answers` declares**, and the table above is the default
(`answers='amplitude'`) in full. `answers='efficiency'` keeps `R`, `T`,
`power_entering`, and `absorption`, and leaves every
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

# Analytic (staircase-free) modeling — approach and results

Written: 2026-08. Paths updated after the single-source rewrite folded the three per-backend
copies into `_core`.

Targets:
- `wavebit/_core/analytic.py` — the analytic Fourier coefficients, and
  `normal_vector_field_analytical` for the normal field
- `wavebit/_core/solve_2d.py::layer_modes_2d` — the `epxy` / `epyx` off-diagonal slots the
  NVF form fills
- `wavebit/_core/engine.py::_eps_conv_matrices` — the modeling-type-2 dispatch, selected by the
  `structure` setter in `wavebit/_core/engine.py`

---

## TL;DR

The existing raster/block modelers approximate tilted or curved boundaries with a
**staircase of axis-aligned rectangles**. That staircase is pure geometric error, so it
only shrinks with high Fourier order / fine grids, and there is no **true smooth
boundary/normal** for the NVF (normal vector field) factorization to exploit.

This work adds a modeling path that computes the **exact analytic 2D Fourier
coefficients** of smooth primitives (rotated rectangle, circle/disk, ellipse) and feeds
them directly to the solver without staircase decomposition. `⟦ε⟧` and `⟦1/ε⟧` are formed
analytically, and the normal projectors are built from an **analytic normal field**, then
factorized with NVF.

Result: it **converges at far lower order than a staircase of the same boundary**, satisfies
energy conservation to machine precision for lossless structures, agrees across
numpy/jax/torch, and is differentiable (compatible with the optimization pipeline).

Two qualifications, both measured, both easy to trip over:

- **Against raster-NVF on a curved boundary the margin is gone.** Retiring `enhanced_dfs`
  improved raster-NVF by ~8x on the metallic circle, and the two are now level there — see §4
  case [2]. The separation that survives is against a *staircase* (case [1]), where the
  geometric error has a floor no Fourier order removes.
- **On an axis-aligned boundary the smooth path is the slower one.** Li's rules along x and y
  are already optimal when the interfaces are axis-aligned, so NVF only costs there. Measured
  against RETICOLO: the block path matches at 1e-14 every order while the smooth path needs
  truncation_order≈16 to reach 1e-4.

Choose the path by the boundary, not on a general expectation that the smooth path wins.

---

## 1. Analytic Fourier coefficients

The coefficient convention matches the `_core/factorization.py::dfs2d_*` builders (checked to machine
precision):

```
c(ky, kx) = 1/(Px Py) ∫∫_cell f(x,y) exp(-i 2π (ky y/Py + kx x/Px)) dx dy
```

With `u = 2π kx/Px`, `v = 2π ky/Py` this is the continuous FT of the shape indicator
evaluated on the reciprocal lattice. For rotation angle `θ` and center `(cx,cy)` we use the
rotated frequency coordinate `κ = Rot(θ)^T k` and the center phase `e^{-i k·c}`.

| primitive | `Ŝ(k) / (Px Py)` |
|---|---|
| rotated rectangle `lx,ly` | `e^{-i k·c} · lx·sinc(κx lx/2) · ly·sinc(κy ly/2)` |
| circle/disk diameter `lx` | `e^{-i k·c} · (π R²)·jinc(R ρ)`,  `R=lx/2`, `ρ=√(u²+v²)` |
| ellipse semi-axes `a=lx/2,b=ly/2` | `e^{-i k·c} · (π a b)·jinc(√((a κx)²+(b κy)²))` |

Here `sinc(z)=sin z/z`, `jinc(z)=2 J1(z)/z` (both removable at 0, value 1). The circle is the
`a=b=R` special case of the ellipse. All three formulas are verified against a fine
midpoint DFT (`tests/test_analytic.py`).

A layer's coefficients are assembled as background + sum of (non-overlapping) shapes:

```
eps_c(k)     = eps_bg·δ_k0 + Σ_s (eps_s − eps_bg)·Ŝ_s(k)/(Px Py)
inv_eps_c(k) = (1/eps_bg)·δ_k0 + Σ_s (1/eps_s − 1/eps_bg)·Ŝ_s(k)/(Px Py)
```

`eps_s` and `eps_bg` are read straight off the layer's materials — **the squaring that used to
stand in those two expressions is gone**, which is one of the sites the flip to permittivity is
made of, and there is no conjugation on the way in either: wavebit is `e^{-iωt}` throughout,
so `Im(eps) > 0` is loss as written. Same reading as `to_conv_mat_raster_nvf`.
`conv_matrix_2d` builds the block-Toeplitz matrices `⟦ε⟧`,
`⟦1/ε⟧` from this `(4*truncation_order+1)²` coefficient grid in the same ordering as the `dfs2d_*` builders.

## 2. Coupling with the NVF factorization

Key observation: the five matrices the NVF consumer (`transfer_2d_2`) needs —
`(εxx, εyy, −εxy, −εyx, inv⟦ε⟧)` — and their building blocks (`⟦ε⟧`, `⟦1/ε⟧`, and the normal
projectors `⟦nx²⟧,⟦nx ny⟧,⟦ny²⟧`) are **all Laurent (direct-rule) convolutions**. No
axis-aligned 1D Li factorization is required. So the analytic coefficients plug straight
into the NVF path:

```
E = ⟦ε⟧(analytic),  A = inv(⟦1/ε⟧(analytic)),  Δ = E − A
εxx = E − ½(Δ Cxx + Cxx Δ),  εyy = E − ½(Δ Cyy + Cyy Δ),  εxy = εyx = −½(Δ Cxy + Cxy Δ)
epz_i = inv(⟦ε⟧)
```

The projectors `Cxx=⟦nx²⟧` etc. are obtained by sampling the **analytic unit normal field**
(`_core/analytic.py::normal_vector_field_analytical` — a level-set gradient per shape until
2026-08-17, a distance-product blend of per-shape surface normals since) on a fine grid
and transforming it (`normal_projector_coeffs`). Unlike the SDF normals of a staircased
boundary, these are the true-boundary normals, so the full NVF benefit is realized.

For a layer holding **one** shape the field is sampled with the shape at the centre of the
cell and the coefficients are then translated to `(cx, cy)` by `e^{-i k·(c − P/2)}` — the same
phase `⟦ε⟧` already carries. Every one of the five matrices is then `D X D⁻¹` for one diagonal
`D`, so a translated layer reproduces the centred one to 1e−14 and `d/dcx`, `d/dcy` come out at
5e−9. That is exact rather than merely good, which is why it stays.

**The centring also makes the field reusable, which is where the wall clock goes.** Sampling and
transforming it is the expensive half of this build — 83% of a five-point sweep at
`normal_grid = 1024, truncation_order = 4` for one shape and 93% for six — and none of it reads a material.
The normal field is a property of where the boundaries are, not of what sits either side of
them. So `normal_projector_coeffs` keys the sampled coefficients on the geometry it was actually
built from and reuses them: a **material** sweep hits because the key holds no permittivity, and a
single-shape **position** sweep hits because the build sits at the centre of the cell whatever
`(cx, cy)` are. Measured 1.6× to 3.8× on five-point sweeps, largest where `normal_grid` is fine
and `truncation_order` modest; where there is nothing to win (`normal_grid = 256`, `truncation_order = 8`) it is 1.0×.

The key is `None` — no reuse, rebuild — whenever a value it would have to hold is traced or under
autograd, which is what keeps this from touching a gradient. A JAX tracer cannot be converted, so
it refuses itself; a PyTorch tensor converts happily and is refused by hand on `requires_grad`.
A layer of several shapes is rebuilt whenever any of them moves, since there the positions enter
the field itself rather than a phase. Holding the field fixed across a move instead would be
quietly wrong, not merely stale: on a lone shape it reports `d(sum T)/dcx = +0.046` where the true
answer is zero, and on a pair it is 8% to 18% off.

**The two reasons this paragraph used to give for the phase are both obsolete** (2026-08-17,
`c128fd8`), and they are worth reading as a pair, because each was a defect wearing the phase
as a workaround. Sampling in place sent `cx`/`cy` through `sign()`, whose derivative is zero;
`sign()` is gone from the rectangle branch, which now blends its four faces so they cancel
exactly at the centre line. And it clipped a field defined over the whole plane against a fixed
cell, so a translated structure was a *different* structure; the displacement now folds to the
nearest image, so the field is periodic and a circle's mean `nx²` reads its symmetric ½ from
anywhere in the cell rather than only from the centre.

Several shapes therefore still stay on the sampled route — no phase describes a nonlinear
combination — but the route is no longer the broken one. They are combined by **distance-product
weights** rather than the *normalized sum of unit vectors* this used to describe, and that
difference is not cosmetic: a sum weights a shape ten cells away exactly as heavily as the one
underfoot, which put the normal 40 deg off at a boundary with two shapes in the layer and 43 with
three. The weights collapse to one shape's own normal exactly at its surface, so boundaries read
0.000 deg for any number of shapes, naming a shape a period over moves 5.3e−15, and the position
gradient survives for rectangles as well as circles. See
`analytic.normal_vector_field_analytical` and `_shape_surface_normal`. Why the `½`-symmetrization is
needed — the plain product `Δ C` is not Hermitian, which gives complex `q²` and gross
energy-conservation violation on curved boundaries — is written out in the
`nvf_conv_matrices` docstring in `_core/factorization.py`.

## 3. Usage (new modeling type)

Passing `structure` as a **list of layer dicts** dispatches to the smooth path (modeling type
2, grating type 2). The existing raster (ndarray) and block (nested list) paths keep
working.

```python
structure = [dict(background=1.0, shapes=[
                 dict(type='circle', cx=0.5, cy=0.5, lx=0.6, material=12.25)]),
             dict(background=2.25, shapes=[
                 dict(type='rectangle', cx=0.4, cy=0.5, lx=0.4, ly=0.5,
                      angle=np.deg2rad(25), material=1.0),
                 dict(type='ellipse', cx=0.8, cy=0.9, lx=0.3, ly=0.2,
                      angle=0.6, material=9.0)])]
solver = wavebit.rcwa(backend=0, structure=structure, truncation_order=dict(x=11, y=11), ...)  # normal_grid=256 (default)
res = solver.solve()
```

Both material slots are **relative permittivity** — `12.25` is the `n = 3.5` this example used
to spell, `2.25` the `n = 1.5` background, `9.0` the `n = 3` ellipse. A bare number is the
shorthand for `wavebit.Material(epsilon=x)`; write the object out to give a region a
permeability as well (`material=wavebit.Material(12.25, mu=2.0)`), and
`wavebit.eps_from_index(3.5)` if what you hold is a datasheet index. The retired keys
`n_base` and `n_index` are refused by name, because renaming them and keeping the number
builds a different structure that solves and conserves energy — see
[`conventions.md`](conventions.md).

`lx/ly/angle/cx/cy/material` may be tensors (torch/jax) and are differentiable w.r.t. them;
`Material` is a registered JAX pytree for exactly that reason.

## 4. Results (`studies/benchmark_analytic.py`)

**[1] Dielectric tilted rectangle** (30°, n=3.5, TM, subwavelength). Reference = analytic
truncation_order=16. Same rectangle represented analytically vs. as a block staircase (n_split); `t0`
error:

| truncation_order | analytic | staircase n=8 | staircase n=24 |
|---:|---:|---:|---:|
| 3 | 3.6e-2 | 7.4e-2 | 7.5e-2 |
| 5 | 1.0e-2 | 5.0e-2 | 4.9e-2 |
| 7 | 3.7e-3 | 4.0e-2 | 3.8e-2 |
| 9 | 2.4e-3 | 3.4e-2 | 3.1e-2 |
| 11 | **6.2e-4** | 3.0e-2 | 2.6e-2 |

Analytic at truncation_order=5 (1e-2) already beats the staircase at truncation_order=11 (n=24, 2.6e-2), and is ~40×
more accurate at truncation_order=11. The staircase saturates at its geometric-error floor (~3e-2) no
matter the order.

**[2] Metallic circle** (n=0.05+3.5j, R=0.3, TM). Reference = analytic truncation_order=16. `t0` error,
re-measured at `f537010` (2026-08-09):

| truncation_order | analytic-NVF | raster-NVF (SDF normals) | raster-sampled |
|---:|---:|---:|---:|
| 3 | 1.56e-2 | 1.39e-2 | 4.18e-2 |
| 5 | 2.78e-2 | 3.42e-2 | 2.89e-2 |
| 7 | 1.52e-2 | 1.48e-2 | 1.24e-2 |
| 9 | 1.49e-3 | **1.39e-3** | 2.59e-2 |
| 11 | **1.37e-3** | 3.48e-3 | 1.75e-2 |

**This case no longer separates analytic-NVF from raster-NVF, and the earlier version of this
section claimed it did.** It read "the accuracy ranks analytic-NVF < raster-NVF < raster-sampled (as
expected)" against a raster-NVF column of 1.1e-2 at truncation_order=9 and 5.5e-3 at truncation_order=11. Those numbers
were correct when measured; the baseline has since moved.

Retiring `enhanced_dfs` (see [nvf-upsampling.md](nvf-upsampling.md)) improved raster-NVF here by
about 8x. A/B on this exact case, extracting the old tree with `git archive` so no git state is
touched:

| raster-NVF error | truncation_order=9 | truncation_order=11 |
|---|---|---|
| `21ad539` (before) | 1.073e-2 | 5.487e-3 |
| `f537010` (after) | 1.385e-3 | 3.484e-3 |

Nothing regressed — this is one of the 27-of-32 combinations that write-up reports gaining
accuracy. What went away is the *margin*: at truncation_order=9 raster-NVF is now marginally ahead of
analytic-NVF, and only at truncation_order=11 does analytic lead again.

So this case no longer supports the claim it was written for. Metal also oscillates badly at
low order — the analytic error *rises* from 1.56e-2 at truncation_order=3 to 2.78e-2 at truncation_order=5 — and with the
two NVF columns now level, there is no clean trend left to read off it. Case [1], the tilted
dielectric, is the one that still separates the paths, because there the staircase carries a
geometric error floor that no Fourier order removes and no raster refinement reaches.

For a curved boundary the honest evidence is now the RETICOLO comparison, which has an external
reference rather than the smooth path's own truncation_order=16: the analytic result is flat from truncation_order=5
while two independent staircases are still climbing toward it at truncation_order=20.

**[3] Lossless multi-shape / multi-layer** (ellipse + rectangle + circle, conical
incidence, non-square period): `|R+T−1|` = 6.7e-16 (truncation_order 5), 1.1e-14 (truncation_order 8) — machine-
precision energy conservation. Backend agreement: `|np−jax|`~1e-14, `|np−torch|`~5e-10.

## 5. Backend consistency and differentiability

- numpy: `scipy.special.j1`. jax: a self-contained hybrid `_j1` (ascending series +
  asymptotic expansion, `<9e-13` on `[0,95]`, differentiable) — jax's `bessel_jn` returns
  NaN for small arguments and is unusable. torch: `torch.special.bessel_j1` (accurate but
  **has no backward**) wrapped in a custom autograd providing the analytic derivative
  `J1' = J0 − J1/x`.
- The ellipse coefficient's `κ=√(...)` has a NaN sqrt(0) derivative at the DC point, guarded
  with a safe-sqrt.
- Verification: `dT/dlx` w.r.t. circle/ellipse semi-axes matches finite differences
  (torch·jax, `test_analytic.py`).

## 6. Regression tests

`tests/test_analytic.py`: (1) conv_matrix_2d==fourier2d_pointwise convention, (2) the three shape
formulas vs a fine DFT, (3) axis-aligned analytic==raster-NVF, (4) lossless energy
conservation, (5) backend agreement, (6) torch/jax differentiability, (7) analytic ≥5× more
accurate than staircase for a tilted dielectric. Existing `test_nvf*` and `test_fixes` all
keep passing.

## 7. Limitations and next steps

- Shapes are assumed **non-overlapping** (indicator addition). Overlap support needs
  inclusion–exclusion handling.
- The normal field for multiple shapes is a **distance-product blend** of each shape's own
  field, exact at every surface. It was a superposition of level-set gradients until
  2026-08-17 (`c128fd8`), which weighted a shape ten cells away as heavily as the one
  underfoot and left the normal 40 deg off at a boundary with two shapes in the layer.
- ~~A multi-shape layer's position gradients keep the defect described in §2~~ — **fixed in
  the same commit, and the two rejected routes recorded here are worth keeping.** Translating
  a layer bodily used to move `sum(de_ri)` by 7.1e−4 and now moves 1e−14 on the sampling grid;
  moving one shape *relative* to another was 11–105% wrong for rectangles and now reads 1e−5.
  Centring on the mean of the shape centres was the rejected fix for the first half: it places
  two shapes exactly opposed about the cell centre, their unit fields cancel exactly on grid
  points, and the gradient there is `0*inf`. Folding the displacement to the nearest image
  fixes that half without manufacturing the degeneracy, because it makes each shape's own
  field periodic rather than re-centring the pair. The second half was thought to need a
  `tanh` mollification of `sign()`, rejected for breaking smoothness through `angle = 0`; it
  needed no mollifier at all. Treating +x and −x (and +y, −y) as four faces blended the same
  way makes them cancel exactly at the centre line, so `sign()` is not smoothed but gone.
- The normal projectors still depend on `normal_grid` sampling (default 1024) — the
  coefficients themselves are analytic, and the shape's *position* is now carried
  analytically too, but the field's shape (size, rotation) retains grid error. Increase the
  grid if needed.
- `fourier_coefficients` and `factorization` both have exactly one value here, and this path
  resolves them to it rather than reading them: `('integrated', 'nvf')`, the closed-form shape
  Fourier transform in the NVF 5-matrix form. Leaving either unset is not a request and gets
  that build; naming the *other* value of either — `'sampled'` or `'li'` — raises
  `NotImplementedError` instead of being silently substituted, which is what both used to be.
  Neither missing value is merely unbuilt: a parametric shape has no samples to take a
  discrete series of, and there is no raster axis for Li's nested 1D rule to run along.
  For a non-NVF pure-Laurent comparison, call the two builders directly:
  `_core.analytic.assemble_eps_coeffs` returns the coefficient grids for `eps` and `1/eps`, and
  `_core.analytic.conv_matrix_2d` turns either into a block-Toeplitz matrix. That first builder
  takes a *shape list*, not a layer, and implements `combine='add'` only — hand it the shapes of
  an overlapping `combine='paint'` layer and it silently returns the summed structure instead
  (`<eps>` 1.99 against the 1.90 the solver integrates, for two 300x300 rods of permittivity
  4 and 9).
  Pass the layer's `combine=` to have that refused rather than answered; the painted layer's
  own coefficients come from `_core.boundary` (see [boundary-modeling.md](boundary-modeling.md)).

### References
- L. Li, *Use of Fourier series in the analysis of discontinuous periodic structures*, JOSA A
  **13**(9), 1870 (1996). [doi:10.1364/JOSAA.13.001870](https://doi.org/10.1364/JOSAA.13.001870)
  — Fourier factorization / inverse rule.
- T. Schuster, J. Ruoff, N. Kerwien, S. Rafler, W. Osten, *Normal vector method for convergence
  improvement using the RCWA for crossed gratings*, JOSA A **24**(9), 2880 (2007).
  [doi:10.1364/JOSAA.24.002880](https://doi.org/10.1364/JOSAA.24.002880) — the normal vector
  method itself, which is what this path implements.
- P. Götz, T. Schuster, K. Frenner, S. Rafler, W. Osten, *Normal vector method for the RCWA with
  automated vector field generation*, Opt. Express **16**(22), 17295 (2008).
  [doi:10.1364/OE.16.017295](https://doi.org/10.1364/OE.16.017295) — a **distinct** paper (Götz is
  not an author of the 2007 one, and the two are easy to merge). It supplies the field for
  *arbitrary* shapes, the gap its abstract names; both papers call the simple-shape case trivial.
  This path is that trivial case — `analytic.normal_vector_field_analytical` writes the normal
  down in closed form from the rectangle/ellipse parameters, so it follows the 2007 method and
  owes nothing to the 2008 algorithm. The raster path
  (`factorization.normal_vector_field`) is the one solving the 2008 paper's problem, by a
  signed-distance transform; whether that is the construction the paper itself uses has not been
  checked against the full text.

---

## Overlapping shapes

This module **sums** its shapes, so an overlap gets both contributions and is neither shape's
material. `combine='paint'` on a layer whose shapes actually intersect is built by the boundary
path instead — it resolves the painting into disjoint polygons and transforms those, keeping the
tilted edges exact. See [`boundary-modeling.md`](boundary-modeling.md).

While nothing intersects the two rules mean the same structure, and the layer stays here.

### A shape larger than the period

Neither path paints one, and for the same reason: nothing in either builder clips a shape
against a *translated copy of itself*, so the only reading of "later over earlier" that means
anything when the two are the same shape — their union — is out of reach. An unlabelled layer
is refused by name on both (`check_layer_combination` here, `check_block_layer_combination` on
the block path), and so is `combine='paint'` on the block path, which is the one it defaults
to.

What the block path did instead, until it was refused, is worth recording because the picture
looked reasonable. A 1.3-wide rod of `eps` 5.76 at the centre of a unit cell over a background
of 1.0 came out as three materials — 5.76 / 10.52 / 20.04, i.e. `1 + n*4.76` for cover counts
`n` = 1, 2, 4 — none of them in the recipe. Blocks are gridded in *unwrapped* coordinates and
`_fourier1d_integrated` weights each segment by its **signed** width, which is exactly what
draws the wrapped part of a sub-period shape correctly; past one period the same arithmetic
counts the covered strip twice, so the image adds to the shape instead of covering it.

That superposition is not a second meaning one could have documented, because only half of the
layer is built that way. Li's rule transforms `1/eps` through the same signed segments, so the
two halves describe different functions: on that rod the `eps` coefficients match the
superposition profile to quadrature accuracy while the `1/eps` coefficients come back with a DC
term of **-0.074** — a negative mean reciprocal permittivity on a layer whose permittivity is
between 5.76 and 20.04, against +0.150 for the profile itself. So the number the solver returned
was neither reading's, and the tell is that it does not settle on one. `sum(T)` for that rod:

| `truncation_order` | block | smooth, `combine='add'` | raster of the superposition | raster of the union |
|---|---|---|---|---|
| 6  | 0.4403 | 0.7635 | 0.8346 | 0.9722 |
| 10 | 0.4147 | 0.4239 | 0.7136 | 0.9722 |
| 15 | 0.4165 | 0.4971 | 0.7178 | 0.9722 |
| 20 | 0.3697 | 0.5890 | 0.7185 | 0.9722 |
| 25 | 0.4011 | 0.6508 | 0.7189 | 0.9722 |

Both readings are structures and both converge — the union is a uniform slab, so it is exact at
every order, and the superposition settles on 0.7189 by 25. The two shape paths converge on
neither and, over that range, on nothing. The same rod at 0.6 wide, where nothing self-overlaps,
matches its raster to every digit printed at every count.

`combine='add'` on the smooth path is the one place the superposition of the images is built on
purpose, and is where to ask for it.

**Where the line falls.** A shape *exactly* the period is touching, not overlapping, and is
built — it is the filled cell, and agrees with a uniform slab of the same material to 1e-12. A sub-period shape straddling the cell boundary is the case the signed-width reading
exists for and is unaffected by where it sits. And the bound follows the shape, not its bounding
box: a 400x40 bar at 45 degrees spans 311 across a 300 cell and still misses its own image in
every direction, so it is built.

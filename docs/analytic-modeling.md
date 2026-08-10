# Analytic (staircase-free) modeling — approach and results

Written: 2026-08. Paths updated after the single-source rewrite folded the three per-backend
copies into `_core`.

Targets:
- `fieldwave/_core/analytic.py` — the analytic Fourier coefficients, and
  `normal_vector_field_analytical` for the normal field
- `fieldwave/_core/solve_2d.py::layer_modes_2d` — the `epxy` / `epyx` off-diagonal slots the
  NVF form fills
- `fieldwave/_core/engine.py::_conv_matrices` — the modeling-type-2 dispatch, selected by the
  `ucell` setter in `fieldwave/on_*/emsolver/rcwa.py`

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
  fto≈16 to reach 1e-4.

Choose the path by the boundary, not on a general expectation that the smooth path wins.

---

## 1. Analytic Fourier coefficients

The coefficient convention matches `_core/factorization.py::dfs2d` (checked to machine
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
eps_c(k)     = eps_bg·δ_k0 + Σ_s (conj(n_s)² − conj(n_bg)²)·Ŝ_s(k)/(Px Py)
inv_eps_c(k) = (1/eps_bg)·δ_k0 + Σ_s (1/conj(n_s)² − 1/conj(n_bg)²)·Ŝ_s(k)/(Px Py)
```

`conj(n)²` matches fieldwave's `e^{-iωt}` convention (Im n>0 = loss), i.e. the same as
`to_conv_mat_raster_nvf`. `laurent_conv_2d` builds the block-Toeplitz matrices `⟦ε⟧`,
`⟦1/ε⟧` from this `(4fto+1)²` coefficient grid in the same ordering as `dfs2d`.

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
(`_core/analytic.py::normal_vector_field_analytical`, a level-set gradient) on a fine grid
and transforming it (`normal_projector_coeffs`). Unlike the SDF normals of a staircased
boundary, these are the true-boundary normals, so the full NVF benefit is realized.

For a layer holding **one** shape the field is sampled with the shape at the centre of the
cell and the coefficients are then translated to `(cx, cy)` by `e^{-i k·(c − P/2)}` — the same
phase `⟦ε⟧` already carries. Sampling it in place instead does two things, both wrong: it
sends `cx`/`cy` through `sign()`, whose derivative is zero, so `d/dcy` lost the projectors'
share of an exact cancellation and came out at −1.7e−1 instead of 0; and it clips a field that
is defined over the whole plane against a fixed cell, so a translated structure was a
*different* structure (1.6e−3 on `sum(de_ri)`, unchanged from 256 to 4096 samples). With the
phase, every one of the five matrices is `D X D⁻¹` for one diagonal `D`, so a translated layer
reproduces the centred one to 1e−14 and `d/dcx`, `d/dcy` come out at 5e−9. Several shapes take
a *normalized sum* of their fields, which no single phase translates, and stay on the sampled
route. Why the `½`-symmetrization is
needed — the plain product `Δ C` is not Hermitian, which gives complex `q²` and gross
energy-conservation violation on curved boundaries — is written out in the
`nvf_conv_matrices` docstring in `_core/factorization.py`.

## 3. Usage (new modeling type)

Passing `ucell` as a **list of layer dicts** dispatches to the smooth path (modeling type
2, grating type 2). The existing raster (ndarray) and block (nested list) paths keep
working.

```python
ucell = [dict(n_base=1.0, shapes=[
             dict(type='circle', cx=0.5, cy=0.5, lx=0.6, n_index=3.5)]),
         dict(n_base=1.5, shapes=[
             dict(type='rectangle', cx=0.4, cy=0.5, lx=0.4, ly=0.5,
                  angle=np.deg2rad(25), n_index=1.0),
             dict(type='ellipse', cx=0.8, cy=0.9, lx=0.3, ly=0.2,
                  angle=0.6, n_index=3.0)])]
solver = fieldwave.rcwa(backend=0, ucell=ucell, fto=(11, 11), ...)  # normal_grid=256 (default)
res = solver.conv_solve()
```

`lx/ly/angle/cx/cy/n_index` may be tensors (torch/jax) and are differentiable w.r.t. them.

## 4. Results (`studies/benchmark_analytic.py`)

**[1] Dielectric tilted rectangle** (30°, n=3.5, TM, subwavelength). Reference = analytic
fto=16. Same rectangle represented analytically vs. as a block staircase (n_split); `t0`
error:

| fto | analytic | staircase n=8 | staircase n=24 |
|---:|---:|---:|---:|
| 3 | 3.6e-2 | 7.4e-2 | 7.5e-2 |
| 5 | 1.0e-2 | 5.0e-2 | 4.9e-2 |
| 7 | 3.7e-3 | 4.0e-2 | 3.8e-2 |
| 9 | 2.4e-3 | 3.4e-2 | 3.1e-2 |
| 11 | **6.2e-4** | 3.0e-2 | 2.6e-2 |

Analytic at fto=5 (1e-2) already beats the staircase at fto=11 (n=24, 2.6e-2), and is ~40×
more accurate at fto=11. The staircase saturates at its geometric-error floor (~3e-2) no
matter the order.

**[2] Metallic circle** (n=0.05+3.5j, R=0.3, TM). Reference = analytic fto=16. `t0` error,
re-measured at `f537010` (2026-08-09):

| fto | analytic-NVF | raster-NVF (SDF normals) | raster-DFS |
|---:|---:|---:|---:|
| 3 | 1.56e-2 | 1.39e-2 | 4.18e-2 |
| 5 | 2.78e-2 | 3.42e-2 | 2.89e-2 |
| 7 | 1.52e-2 | 1.48e-2 | 1.24e-2 |
| 9 | 1.49e-3 | **1.39e-3** | 2.59e-2 |
| 11 | **1.37e-3** | 3.48e-3 | 1.75e-2 |

**This case no longer separates analytic-NVF from raster-NVF, and the earlier version of this
section claimed it did.** It read "the accuracy ranks analytic-NVF < raster-NVF < DFS (as
expected)" against a raster-NVF column of 1.1e-2 at fto=9 and 5.5e-3 at fto=11. Those numbers
were correct when measured; the baseline has since moved.

Retiring `enhanced_dfs` (see [nvf-upsampling.md](nvf-upsampling.md)) improved raster-NVF here by
about 8x. A/B on this exact case, extracting the old tree with `git archive` so no git state is
touched:

| raster-NVF error | fto=9 | fto=11 |
|---|---|---|
| `21ad539` (before) | 1.073e-2 | 5.487e-3 |
| `f537010` (after) | 1.385e-3 | 3.484e-3 |

Nothing regressed — this is one of the 27-of-32 combinations that write-up reports gaining
accuracy. What went away is the *margin*: at fto=9 raster-NVF is now marginally ahead of
analytic-NVF, and only at fto=11 does analytic lead again.

So this case no longer supports the claim it was written for. Metal also oscillates badly at
low order — the analytic error *rises* from 1.56e-2 at fto=3 to 2.78e-2 at fto=5 — and with the
two NVF columns now level, there is no clean trend left to read off it. Case [1], the tilted
dielectric, is the one that still separates the paths, because there the staircase carries a
geometric error floor that no Fourier order removes and no raster refinement reaches.

For a curved boundary the honest evidence is now the RETICOLO comparison, which has an external
reference rather than the smooth path's own fto=16: the analytic result is flat from fto=5
while two independent staircases are still climbing toward it at fto=20. See
`<the manual's working notes>`.

**[3] Lossless multi-shape / multi-layer** (ellipse + rectangle + circle, conical
incidence, non-square period): `|R+T−1|` = 6.7e-16 (fto5), 1.1e-14 (fto8) — machine-
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

`tests/test_analytic.py`: (1) laurent==dfs2d convention, (2) the three shape
formulas vs a fine DFT, (3) axis-aligned analytic==raster-NVF, (4) lossless energy
conservation, (5) backend agreement, (6) torch/jax differentiability, (7) analytic ≥5× more
accurate than staircase for a tilted dielectric. Existing `test_nvf*` and `test_fixes` all
keep passing.

## 7. Limitations and next steps

- Shapes are assumed **non-overlapping** (indicator addition). Overlap support needs
  inclusion–exclusion handling.
- The normal field for multiple shapes is a superposition of level-set gradients
  (heuristic). A single shape is exact.
- A multi-shape layer's position gradients keep the defect described in §2, in two halves.
  Translating such a layer bodily still moves `sum(de_ri)` by 7.1e−4, against 1.5e−14 for a
  single shape; and moving one shape *relative* to another is right for smooth shapes (4.7e−6
  against central differences) but 11–105% wrong for rectangles. Centring on the mean of the
  shape centres fixes the first half (1.2e−4 → 6.6e−15) and was rejected: it places two shapes
  exactly opposed about the cell centre, their unit fields cancel exactly on grid points, and
  the gradient there is `0*inf`. The second half needs a rectangle normal field that is
  differentiable in position to begin with, which is the `tanh` mollification already rejected
  for breaking smoothness through `angle = 0`.
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
  `_core.analytic.laurent_conv_2d` turns either into a block-Toeplitz matrix.

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

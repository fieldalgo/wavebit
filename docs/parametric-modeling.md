# Analytic (staircase-free) modeling — approach and results

Written: 2026-08. Paths updated after the single-source rewrite folded the three per-backend
copies into `_core`.

Targets:
- `wavebit/_core/parametric.py` — the analytic Fourier coefficients, and
  `normal_vector_field_analytical` for the normal field
- `wavebit/_core/solve_2d.py::layer_modes_2d` — the `epxy` / `epyx` off-diagonal slots the
  NVF form fills
- `wavebit/_core/engine.py::_eps_conv_matrices` — the modeling-type-2 dispatch, selected by the
  `structure` setter in `wavebit/_core/engine.py`

---

## TL;DR

The raster path approximates tilted or curved boundaries with a
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
- **On an axis-aligned boundary the default parametric NVF route is the slower one.** Li's rules along x and y
  are already optimal when the interfaces are axis-aligned, so NVF only costs there. Measured
  against RETICOLO: the block path matches at 1e-14 every order while the parametric path needs
  truncation_order≈16 to reach 1e-4.

Choose the path by the boundary, not on a general expectation that the parametric path wins.

---

## Complex NVF projection for rectangular contacts

Rectangular shapes can now use `factorization='nvf', projector='circular'`.
This remains normal-vector Fourier factorization with exact material transforms;
it changes the normal projector, not the factorization to Li. Rotations, skew
cells, overlapping paintings, and magnetic materials use the same construction.
Disjoint circles and ellipses also support circular projection, as described below.

```python
solver = wavebit.rcwa(
    backend=1, period=(700., 700.), wavelength=900., thickness=(300.,),
    truncation_order=(3, 3), normal_grid=128,
    factorization='nvf', projector='circular',
    structure=[dict(background=1., shapes=[
        dict(type='rectangle', cx=-150., lx=300., ly=300., material=4.),
        dict(type='rectangle', cx=150., lx=300., ly=300., material=9.)])])
geometry = solver.freeze_geometry()
result = solver.solve(geometry=geometry)
```

The construction sums positive normal dyads into `M`, forms the complex vector
`J = M @ (1, i)`, and uses `P = J J^H / |J|^2`. Opposite normals reinforce as
matrices rather than cancelling as vectors. The denominator stays positive,
including at the centre of a rectangle. On a regular interface the projector
is exactly the normal dyad. Both complex off-diagonal Fourier components are
passed to the existing NVF assembly; they are not assumed equal.

The tested rectangular period gap, unequal-material contact, and material-equality
limits have no sampled value step, on both odd and even grids. Frozen JIT geometry
and material derivatives agree with finite differences. A circular NVF geometry
record can be frozen at contact and replayed while the gap opens: it reads image
bounds, not the linear field's signed pair cancellation records. A new overlap,
changed shape count, or enlarged image set still requires refreezing. Traced calls
cannot enforce those checks, so keep those constraints when optimizing.

See [the runnable NVF contact example](../examples/nvf_contact.py), which evaluates
both contact and the one-sided opening derivative through the public JAX API.
`normal_grid` remains an integration-resolution control; it is not a smoothing
parameter for a gap. The default projector remains `linear` to preserve existing
results. Select `circular` explicitly for this NVF construction.

**Accuracy still needs Fourier-order convergence checks.** The circular extension
retains primitive faces when a material interface disappears. It therefore does not
inherit linear NVF's exact constant-normal reduction for a full-period bar, and
splitting a rectangle into equal-material primitives can change finite-order
answers. The latter also occurs with the existing linear field; it is not a newly
established 1e-10 equivalence guarantee. These are separate accuracy questions
from contact continuity. [The investigation record](nvf-contact-followup.md)
contains the measured order/grid ladders and the remaining representation test.

### Disjoint circles and ellipses

The same `factorization='nvf', projector='circular'` options support circles,
ellipses, and mixtures with rectangles whose interiors do not overlap. Tangency
is allowed, including contact with a periodic image. `square` and `disk` aliases
use their rectangle and circle constructions respectively.

For a curve's local level set `f = (u/a)^2 + (v/b)^2 - 1`, the construction sums
positive dyads of `min(a,b) * grad(f)` over periodic images, with inverse fourth
powers of `min(a,b) * abs(f)` and a compact taper on each image. An isotropic term
extends the tensor away from boundaries. It keeps the projector regular at a
primitive's centre, where its gradient vanishes (circular when no other image
contributes), while the pole makes
the projector exactly the geometric normal dyad on the boundary. The image taper
and isotropic term define the extension; they do not blur the material transform.
The support is bounded by the shape radius and reduced cell height. Switching the
fold's image indices only adds or removes zero contributions, so the periodic
field does not jump between two selected normals.

The curve implementation passes tangent circle, tangent ellipse, mixed
rectangle/circle and periodic circle contact checks on odd and even grids.
Frozen-contact JIT derivatives match one-sided finite differences. Its exact
boundary normals, periodicity, rotation covariance, image-ring independence,
centre regularity, and NumPy/JAX/Torch agreement are tested separately.

See [the curved-contact example](../examples/nvf_curved_contact.py). This extension
does not settle the rectangular split/merged accuracy target or the tilted
resonant-convergence investigation.

### Overlapping circles and ellipses

Curved paintings also accept `factorization='nvf', projector='circular'`.
The existing painting route approximates curves with polygon chords and resolves
their intersections. Circular NVF now follows those same chords: it retains the
analytic extension away from curves and blends in positive polygon-edge dyads
near them. The blend is exactly the facet normal on a regular curved polygon
boundary, and exactly the analytic normal on a rectangular boundary. Its width
is set by chord length and decreases under polygon refinement.

The original polygon edges remain extension geometry even when covered. Their
positions and lengths are live parameters; material equality never switches
their contribution on or off. The resolved painting supplies the material
coefficients, including complex permittivity and permeability. Both complex
off-diagonal projector components reach the NVF assembly. NumPy, JAX and Torch
use the same construction.

Use `freeze_geometry()` before JIT and stay within its recorded intersections
and periodic image bounds. Re-freeze when painting topology changes. See
[the curved painting example](../examples/nvf_curved_painting.py), which evaluates
geometry and material derivatives through the public solver.

**Polygon approximation is still an accuracy limit.** Crossing between analytic
disjoint curves and polygon-resolved overlaps changes the material approximation.
The local normal correction avoids the much larger jump caused by replacing
the entire normal extension, but does not make that approximation switch exact.
On the recorded tangent-ellipse fixture at order 2 and normal grid 64, the
256- and 512-vertex polygon solves differ from the analytic solve by 2.65e-7
and 6.64e-7. Intermediate total errors need not decrease monotonically because
material and projector quadrature errors can cancel. This is a fixed-order
approximation comparison, not an external Maxwell convergence guarantee.

## Cartesian rectangles with explicit Li factorization

For axis-aligned rectangular paintings, `factorization='li'` now integrates the
nonuniform Cartesian intervals at every shape edge exactly. This uses the shape
coordinates directly, including gaps far smaller than a pixel. Later shapes paint
over earlier shapes. Shape modeling still defaults to NVF; selecting Li is explicit.

```python
solver = wavebit.rcwa(
    backend=1, period=(700., 700.), wavelength=900., thickness=(300.,),
    truncation_order=(3, 3), factorization='li',
    structure=[dict(background=1., shapes=[
        dict(type='rectangle', cx=-150., lx=300., ly=300., material=4.),
        dict(type='rectangle', cx=150., lx=300., ly=300., material=9.)])])
geometry = solver.freeze_geometry()
result = solver.solve(geometry=geometry)
```

Supported cells are positive, axis-aligned rectangles. Shapes must fit inside the
cell, including its edges; a rectangle may span a full period. Rotated/curved shapes,
skew cells, and rectangles crossing a cell edge are refused. Omit `angle` when
tracing a shape dictionary: this route has a fixed zero angle, not an angle derivative.
`normal_grid`, `normal_method`, and `projector` are refused because no normal field
is sampled. Coefficients are integrated; requesting sampled coefficients is refused.

`freeze_geometry()` stores edge order and painting ownership. Lengths, centers,
permittivity, and permeability remain live. It retains zero-width intervals at
contact, so a gap can open continuously using the same plan. Eager replay refuses
crossed edges or a changed shape count. JIT cannot check those values: keep the
edge order fixed within a traced optimization step and refreeze on concrete
geometry when the ordering changes. At contact, the plan defines the one-sided
opening-gap derivative, not a claim that overlap has the same derivative.

See [the runnable contact example](../examples/rectilinear_contact.py) for JIT
transmission and gap/material gradients. Tests compare convolution matrices with
independently painted commensurate rasters, including overlaps, complex materials,
and magnetic layers, and compare derivatives with two finite-difference steps.
Exact geometry integration does not remove Fourier truncation error; still check
order convergence for the observable of interest. This route avoids the NVF contact
blend for the supported geometry; general-shape NVF contact issues remain open.

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
midpoint DFT (`tests/test_parametric.py`).

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
(`_core/parametric.py::normal_vector_field_analytical` — a level-set gradient per shape until
2026-08-17, a distance-product blend of per-shape surface normals since) on a fine grid
and transforming it (`normal_projector_coeffs`). Unlike the SDF normals of a staircased
boundary, these are the true-boundary normals, so the full NVF benefit is realized.

For a layer holding **one** shape the field is sampled with the shape at the centre of the
cell and the coefficients are then translated by `e^{-i k·c}`, where `c` is the displacement
from that centre — which since 2026-08-31 is just `(cx, cy)` as the caller wrote them, those
being measured from `0.5*(a1 + a2)`. It is the same phase `⟦ε⟧` already carries. Every one of
the five matrices is then `D X D⁻¹` for one diagonal
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
A layer of several shapes is rebuilt whenever any of them moves: the offsets from the first
shape enter the field itself, and even a rigid slide changes them by an ulp, which the key sees.
Holding the field fixed across a move instead would be
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

**A face the shape's own image covers is not an interface**, and that is a third thing the fold
does not settle. A rectangle spanning the cell — `ly = Py`, i.e. a 1-D grating written on a 2-D
cell — has y-faces that bound nothing, and the blend reported a surface normal across them
anyway, `max|n_y| = 1.000` over a third of the cell. Each face distance is now measured to the
nearest **image** of that face, so at the span the two opposite families are the same set of lines,
their distances are equal and their normals cancel identically. That is exact rather than
approximate: the field becomes `(±1, 0)`, and the parametric path then reproduces a raster of the
same grating under Li's 1-D rule to 1e−13, where it was 5e−4 to 9e−3 away. Folding the distance
rather than erasing the face is what keeps it continuous — an erasure has to decide *when*, and
any threshold puts a step of ~2.5e-3 at its own tolerance, on the edge of the feasible set where
an optimiser would read it as a gradient. An image slid *along* the face kills only part of it,
which no closed form over one box reaches — so since 2026-09-06 the field is not built over one
box: a box is four face **pieces**, each evaluated at every periodic instance that can be its
nearest, and a plan (`parametric.plan_gamma`) records which face shares a line with which image's
face, over what run, so that the two cancel exactly there. The sheared and tilted scenes that
used to be refused build at 0.16–0.19 deg against an exact interface oracle. Both halves, with
the convergence tables and what moved, are in
[`normal-field-known-limits.md`](normal-field-known-limits.md).

Several shapes are built the same way since 2026-09-11, with the **first** shape at the centre
of the cell and the others at their offsets from it, and carried by the same phase, so a rigid
slide of the whole layer is exact too (coefficients at 3e-17, `sum(T)` at 3e-13 over sub-pixel
shifts, against 1.1e-2 / 1.2e-3 before; `studies/nvf_placement_spread.py`). The first shape and
not a centroid, because a centroid is not invariant under naming one shape a period over and that
broke the layer's periodicity across the cell edge (3.2e-5 where 1e-9 is promised); one shape's
centre shifts by a lattice vector or not at all, which the fold and the grid cannot see. What
stays sampled is the *relative* geometry: moving one shape alone still leaves a sampling ripple
across a pixel of about five times the legitimate change over that pixel (1.2e-3 / 7.9e-4 /
1.6e-4 in the largest coefficient at `normal_grid` 64 / 256 / 1024, against 2.1e-4 / 1.7e-4 /
7.9e-5), which is what `normal_grid` is for. They are combined by **distance-product
weights** rather than the *normalized sum of unit vectors* this used to describe, and that
difference is not cosmetic: a sum weights a shape ten cells away exactly as heavily as the one
underfoot, which put the normal 40 deg off at a boundary with two shapes in the layer and 43 with
three. They are the ratio form now, `(d_min / d_j)^4`, direction-identical to the product and
linear in the count, and since 2026-09-06 it is **one sum over every piece of every shape** —
a box's four faces over their periodic images, a round shape's one piece — with no per-shape
stage in between: two shapes that come into contact share a face line over a run, the plan
(`parametric.plan_gamma`) records the pair, and their material slots decide whether the two
faces cancel there (equal: the material runs on) or co-orient (different: the seam is an
interface). The weights collapse to one piece's own normal exactly at its surface, so boundaries read
0.000 deg for any number of shapes, naming a shape a period over moves 5.3e−15, and the position
gradient survives for rectangles as well as circles. See
`parametric.normal_vector_field_analytical` and `_piece_field`. Why the `½`-symmetrization is
needed — the plain product `Δ C` is not Hermitian, which gives complex `q²` and gross
energy-conservation violation on curved boundaries — is written out in the
`nvf_conv_matrices` docstring in `_core/factorization.py`.

## 3. Usage (new modeling type)

Passing `structure` as a **list of layer dicts** dispatches to the parametric path (modeling type
2, grating type 2). The raster (ndarray) path keeps working; the other list spelling, a
per-layer `[background, [[shape, ...], ...]]`, was the block modeller and is refused by name
since 2026-08-26.

```python
structure = [dict(background=1.0, shapes=[
                 dict(type='circle', cx=0.0, cy=0.0, lx=0.6, material=12.25)]),
             dict(background=2.25, shapes=[
                 dict(type='rectangle', cx=-0.1, cy=0.0, lx=0.4, ly=0.5,
                      angle=np.deg2rad(25), material=1.0),
                 dict(type='ellipse', cx=0.3, cy=0.4, lx=0.3, ly=0.2,
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
reference rather than the parametric path's own truncation_order=16: the analytic result is flat from truncation_order=5
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
  (torch·jax, `test_parametric.py`).

## 6. Regression tests

`tests/test_parametric.py`: (1) conv_matrix_2d==fourier2d_pointwise convention, (2) the three shape
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
- The normal projectors still depend on `normal_grid` sampling (default 1024) -- the
  coefficients themselves are analytic, and the layer's *position* (one shape or several) is
  carried analytically, but the field's shape (size, rotation, and the relative placement of
  several shapes) retains grid error. Increase the grid if needed.
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
  takes a *shape list*, not a layer, and sums — so it refuses shapes that overlap rather than
  returning the summed structure under the name of the painted one (`<eps>` 1.99 against the
  1.90 the solver integrates, for two 300x300 rods of permittivity 4 and 9). An overlapping
  layer's own coefficients come from `_core.boundary`
  (see [boundary-modeling.md](boundary-modeling.md)).

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
  This path is that trivial case — `parametric.normal_vector_field_analytical` writes the normal
  down in closed form from the rectangle/ellipse parameters, so it follows the 2007 method and
  owes nothing to the 2008 algorithm. The raster path
  (`factorization.normal_vector_field`) is the one solving the 2008 paper's problem, by a
  signed-distance transform; whether that is the construction the paper itself uses has not been
  checked against the full text.

---

## A length is a size, and the transform carries its sign

`check_layer_shapes` refuses a length that is not positive, before it looks at anything else.

A negative `lx` does not make a smaller shape. It makes a *subtracted* one. `shape_ft`'s
coefficient at `G = 0` is the fraction of the cell the shape claims, and the sign rides straight
through it:

| `lx` on a 300 period | coefficient at `G = 0` | what the layer then is |
|---|---|---|
| +120 | +0.200000 | `eps` 4 in 20% of the cell |
| -120 | **-0.200000** | `eps` 0.4 — the background *minus* the contrast |
| -340 | -0.566667 | `eps` -0.7, a **metal**, out of a spec naming two dielectrics |

None of it was refused before 2026-08-28. A 120x150 rod of `eps` 4 on `eps` 1 came back with
`sum(T)` 0.444002054 against the 0.980652726 of the rod the dict names — identically on NumPy,
JAX and PyTorch, energy conserved, nothing said.

**It also disabled the refusal below.** `overlap.primitive` stores `lx / 2` as a half-width, so
a negative length is a negative projection radius and every separating-axis test comes back
clear: `overlapping_pairs` returns `[]` over a rod of `lx = -340` exactly as it does for one
that fits. `lx = +340` is refused as larger than the period; `lx = -340` was built, and answered
0.047049717. That is why the sign is checked first — not because the message would otherwise
differ (it would not; the period check simply finds nothing), but so that check is never asked a
question it has no way to answer.

**An optimiser reaches it by the most ordinary move it has** — shrinking a feature. There is no
cliff at zero to stop a descent: `T` is even there to second order (0.999998510 at `lx = +0.5`
against 0.999998461 at `-0.5`, only the derivative changing sign), so the walk continues and
starts growing anti-material on the far side.

Zero goes with the negatives. A `rectangle` of `lx = 0` is a shape that is not there — a layer
that builds, as a uniform slab of the background, which is the same argument `check_layer_spec`
makes one level up about an unread layer key. An `ellipse` of `lx = 0` reaches `LinAlgError`
from inside the solve with nothing said about the geometry. And an *omitted* length is the same case, since
`shape_geometry` defaults it to `0.0`.

The message names the key that actually supplied the value, following `shape_geometry`'s own
fallback: a dict saying `l` is told about `l`, never about an `ly` it did not write. That is
also what keeps an `ly` off a `circle`, which does not take one — its two lengths are equal, so
the first is always what raises.

Under `jax.jit` this is unchecked, like every other value check on this path — a jit tracer
carries no number to read. Five of the six doors are closed (NumPy, JAX eager, `jax.grad`,
`freeze_geometry`, and both PyTorch modes); the sixth is the standing trace limitation recorded
in [`boundary-modeling.md`](boundary-modeling.md) and [`backend-choice.md`](backend-choice.md).

## Overlapping shapes

This module **sums** its shapes, so an overlap would get both contributions and be neither
shape's material. It refuses one instead. A layer whose shapes actually intersect is built by
the boundary route — it resolves the painting into disjoint polygons, later shape over earlier,
and transforms those, keeping the tilted edges exact. See
[`boundary-modeling.md`](boundary-modeling.md).

While nothing intersects the two rules mean the same structure, and the layer stays here.
Nothing is asked of the recipe either way; `layer_needs_resolving` reads the geometry. A
`combine=` key named the rule until 2026-08-27 and is now refused as an unread layer key.

### A shape larger than the period

This one is refused outright, by `check_layer_shapes`. Nothing in either builder clips a shape
against a *translated copy of itself*, so the only reading of "later over earlier" that means
anything when the two are the same shape — their union — is out of reach, and the sum is not a
second meaning one could have documented either.

Why not, is worth recording, because the picture the removed block path drew looked reasonable.
A 1.3-wide rod of `eps` 5.76 at the centre of a unit cell over a background of 1.0 came out as
three materials — 5.76 / 10.52 / 20.04, i.e. `1 + n*4.76` for cover counts `n` = 1, 2, 4 — none
of them in the recipe. Blocks were gridded in *unwrapped* coordinates and `_fourier1d_integrated`
weights each segment by its **signed** width, which is exactly what draws the wrapped part of a
sub-period shape correctly; past one period the same arithmetic counts the covered strip twice,
so the image adds to the shape instead of covering it.

And only half of the layer was built that way. Li's rule transforms `1/eps` through the same
signed segments, so the two halves describe different functions: on that rod the `eps`
coefficients matched the superposition profile to quadrature accuracy while the `1/eps`
coefficients came back with a DC term of **-0.074** — a negative mean reciprocal permittivity on
a layer whose permittivity is between 5.76 and 20.04, against +0.150 for the profile itself. So
the number the solver returned was neither reading's, and the tell is that it does not settle on
one. `sum(T)` for that rod:

| `truncation_order` | block | smooth, summed | raster of the superposition | raster of the union |
|---|---|---|---|---|
| 6  | 0.4403 | 0.7635 | 0.8346 | 0.9722 |
| 10 | 0.4147 | 0.4239 | 0.7136 | 0.9722 |
| 15 | 0.4165 | 0.4971 | 0.7178 | 0.9722 |
| 20 | 0.3697 | 0.5890 | 0.7185 | 0.9722 |
| 25 | 0.4011 | 0.6508 | 0.7189 | 0.9722 |

Both readings are structures and both converge — the union is a uniform slab, so it is exact at
every order, and the superposition settles on 0.7189 by 25. The two shape paths converged on
neither and, over that range, on nothing. The same rod at 0.6 wide, where nothing self-overlaps,
matches its raster to every digit printed at every count.

**Where the line falls.** A shape *exactly* the period is touching, not overlapping, and is
built — it is the filled cell, and agrees with a uniform slab of the same material to 1e-12. A sub-period shape straddling the cell boundary is the case the signed-width reading
exists for and is unaffected by where it sits. And the bound follows the shape, not its bounding
box: a 400x40 bar at 45 degrees spans 311 across a 300 cell and still misses its own image in
every direction, so it is built.

# Non-orthogonal (oblique) unit cells — what `lattice` does and what it costs

Written: 2026-08-09. Measured on CPU, NumPy backend, float64.

Targets:
- `fieldwave/_core/lattice.py` — the basis, the reciprocal mapping, the Jacobian
- `fieldwave/_core/solve_2d.py::order_grid_kxy` — the two k-layouts a cell can have
- `fieldwave/_core/engine.py::CoreSolveMixin.lattice` — the entry point
- `fieldwave/_core/engine.py::_reject_unsupported_lattice` — what is still refused, and why

---

## TL;DR

A hexagonal array used to be spelled as an `a x a*sqrt(3)` rectangular supercell holding two
motifs. That doubles the cell area, the retained order count scales with area, and the solve is
`O(N^2.7)` — so it cost about **8x**. Pass the basis instead:

```python
import numpy as np, fieldwave

a = 400.0
a1, a2 = np.array([a, 0.0]), np.array([a / 2, a * np.sqrt(3) / 2])   # hex

solver = fieldwave.rcwa(
    backend=0, ucell=cell, fto=[10, 10], thickness=[300.0], wavelength=900.0,
    factorization='nvf',                       # required — see "What is refused"
    lattice=np.column_stack((a1, a2)),         # columns are a1, a2
)
```

`period=(Lx, Ly)` remains the default and means `diag(Lx, Ly)`. Setting `lattice` makes `period`
report the basis-vector lengths `(|a1|, |a2|)`.

## The one thing that will bite you

**The raster is read in lattice coordinates.** `ucell[layer][j, i]` is the permittivity at

    r = (i + 0.5)/Nx * a1  +  (j + 0.5)/Ny * a2

Axis 1 runs along `a1` (and pairs with `fto[0]`), axis 0 along `a2` (with `fto[1]`). Build the
raster on a **Cartesian** grid and hand it a skewed basis and you have described a *different
structure* — and nothing will tell you. There is no error, the transform is still a transform,
and energy still conserves to 1e-15 because the NVF assembly is Hermitian whatever the geometry
says. The way to write it:

```python
u = (np.arange(nx) + 0.5) / nx
v = (np.arange(ny) + 0.5) / ny
U, V = np.meshgrid(u, v)
pts = U[..., None] * a1 + V[..., None] * a2      # Cartesian position of every cell
```

then test membership in Cartesian space, with **minimum image over the lattice** (`i*a1 + j*a2`
for `i, j` in `-1, 0, 1`) so a feature straddling an edge wraps correctly.

Analytic shapes do **not** have this problem: their lengths are absolute Cartesian quantities, so
a rectangle stays a rectangle under a skew and `cx` / `cy` mean what they always did.

## What it buys

Hex array of Si pillars, pitch 400 nm, r = 90 nm, h = 300 nm, lambda = 900 nm, normal incidence.
The supercell and the native cell describe the same infinite array.

| | orders | wall clock |
|---|---|---|
| `a x a*sqrt(3)` supercell, `fto=(14, 24)` | 1421 | 38.9 s |
| native hex cell, `fto=(12, 12)` | 625 | 3.3 s |

The order-count ratio is exactly 2 (it is the cell-area ratio; verified numerically across
`fto`), and the measured cost exponent is `N^2.7` — not `N^3`, because the convolution build is
sub-cubic. At matched k-space coverage that is **~8x**, and better in practice because the
supercell converges more slowly than the native cell does.

Do not confuse this with a convergence fix. On the same structure, `factorization='li'` had not
converged at `fto=(16, 28)` after 72 s while `'nvf'` settled by `fto=(10, 17)` in 5 s. That gap
is the *factorization*, not the lattice, and an oblique cell does nothing about it. The two wins
are independent and multiply.

## What works and what is refused

| | status | note |
|---|---|---|
| `factorization='nvf'`, sampled or integrated | supported | — |
| analytic shapes, summed | supported | lengths are absolute Cartesian; nothing deforms |
| analytic shapes, painted **overlap** | supported | the clipper is Cartesian; only the periodic images and the transform needed the basis |
| field reconstruction | supported | output grid is indexed by `(v, u)`, not `(y, x)` — see below |
| vector modeler (`modeling_type=1`) | **refused** | rasterises onto a rectangular grid of its own |
| `factorization='li'` | **refused** | see below |

Everything except `li` and the vector modeler works on a skewed cell, and both of those have a
supported route to the same structure — a raster, or analytic shapes.

The two refusals raise rather than answering, and that matters: both of them *run* on a skewed
cell and return a plausible wrong number, with energy still conserving to 1e-15.

`li` applies the inverse rule along the **raster axes** (`epx = dfs2d(cell, 0, 1)`), and the solve
consumes the result as Cartesian `eps_xx` / `eps_yy`. On a skewed cell those axes are `a1` / `a2`,
not `x` / `y`. Repairing it means Li's covariant formulation (JOSA A **14**(10), 2758).

Before anyone spends that: **`nvf` already contains `li`.** With a constant axis-aligned normal
the projector collapses to `C_xx = I`, `C_xy = C_yy = 0`, and the assembly reduces to
`eps_xx = <1/eps>^-1`, `eps_yy = <eps>` — Li's rules exactly, not an approximation of them (see
`factorization.normal_vector_field`). A rectangle has a constant normal on each face, so `nvf`
reproduces `li` face by face there.

Measured, on the structure most favourable to `li` — a rectangle aligned to the axes, 512x512
raster, square lattice:

| fto | `li` | `nvf` |
|---|---|---|
| 12 | 0.86259051 | 0.86239874 |
| 15 | 0.86204124 | 0.86242349 |
| 18 | 0.86197717 | 0.86186546 |

Same limit to ~1e-4, same cost (30.0 s against 31.1 s at fto 18). On a *curved* boundary the two
part company completely: on a circular pillar `li` had not converged at `fto=(16, 28)` after 72 s
while `nvf` settled by `fto=(10, 17)` in 5 s. No measured case has `li` ahead. One structure, one
wavelength, normal incidence — but the containment above is structural and holds regardless.

## Fields on a skewed cell

`calculate_field` works, with one change you have to handle when plotting: **samples land on the
cell's parallelogram**, `r = u*a1 + v*a2` for `u, v` in `[0, 1]`, so the returned grid is indexed
by `(v, u)` and not by `(y, x)`. Convert before drawing:

```python
u = np.linspace(0, 1, res_x); v = np.linspace(0, 1, res_y)
U, V = np.meshgrid(u, v)
X = U * a1[0] + V * a2[0]
Y = U * a1[1] + V * a2[1]
plt.pcolormesh(X, Y, np.abs(field[0, z_index, :, :, 0]))    # not imshow
```

Separability survives the skew, which is why this was a few lines: `k.r = u*(k.a1) + v*(k.a2)`,
still an outer product. The incident offset rides inside `kx` / `ky` already, so nothing extra is
carried.

**One thing not to be surprised by.** Comparing a field between two bases that describe the same
lattice will *not* agree near a material boundary, and that is not an error. A change of basis
keeps the lattice but retains a different set of orders — a sheared parallelogram in `(m, n)`
where the other has a square. Converged quantities cannot tell (efficiencies agree to 2e-6), but
a field is a finite sum over exactly that set, and the evanescent tails the two sets disagree
about decay far too slowly to cancel at a discontinuity. Measured: 1.1e-1 at the interface, flat
from `fto=4` to `fto=12`; **2.0e-4** in a homogeneous region of the same run, where every
surviving order is propagating and both sets hold all of them. Compare fields away from
boundaries, or at matched order sets.

## How it was validated, and how to validate your own use

Energy conservation is **not** evidence here — `R + T == 1` holds to 1e-15 with the geometry read
wrongly. Three checks that are:

**A unimodular shear.** `a2' = a2 + a1` generates the *identical* point lattice: same cell area,
same reciprocal set. So the two runs must agree, and they do — 1.2e-4 on a raster, 2.0e-6 on
analytic shapes (tighter because there is no staircase in it). This is the cheapest correctness
check available and it is worth running on your own structure; `tests/test_oblique_lattice.py`
has it as `test_unimodular_shear_is_the_same_physics`. With the normal field's metric correction
removed the same comparison drifts to 8.5e-4 — that is the test having teeth.

**S4 as an outside oracle, against a square-lattice control.** S4 supports arbitrary lattice
vectors, so it checks the reciprocal mapping independently. It sits 6.60e-3 from FieldWave on a
square lattice — where none of this code runs — and 6.77e-3 on hex. The gap is S4's own slow
convergence, equal on both, so the oblique path adds **1.7e-4** over the already-validated
orthogonal one. (Use S4's default Laurent rule for this. `PolarizationDecomposition=True`
diverged on *both* lattices in our hands, 0.986 -> 0.933, and is not usable as an oracle.)

**The supercell it replaces.** A native hex cell reproduces its rectangular supercell; the
residual is the two rasters staircasing the feature differently, so it tightens with resolution
rather than with `fto`.

## Where the skew actually reaches the code

Almost nowhere, which is why this was small. Bloch's theorem gives the retained orders Cartesian
components `k_inc + lambda * A^-T @ (m, n)` and that is the whole of it: the eigenproblem reads
only diagonal `Kx` / `Ky`, and substituting `r = u*a1 + v*a2` leaves

    eps_hat[m, n] = \int_0^1 \int_0^1 eps(u, v) exp(-2i*pi*(m*u + n*v)) du dv

an ordinary DFT on the parametric grid for *any* basis — still indexed by integer differences,
still block-Toeplitz. `dfs2d`, the Toeplitz builds and `final_matching` are untouched. Truncation
stays rectangular in `(m, n)`; Li introduced circular truncation *in* the non-orthogonal setting
and still found the parallelogramic one superior, and it measured worse here too.

The one genuinely Cartesian-geometric quantity is NVF's normal field, which is built from a
distance transform and a gradient taken in **index** space and so needs the metric (`J^-T`). That
correction is why NVF rather than Li's nested rule is the factorization that ports: `_nvf_assemble`
is written in Cartesian tensor components, so once the normal is Cartesian nothing else in the
factorization learns the cell was skewed. The same correction fixes an orthogonal case that was
also wrong — a raster whose pixels are not square in real space; see
`tests/test_nvf_non_square_grid.py`.

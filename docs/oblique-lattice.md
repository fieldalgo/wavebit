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

## What is refused

Each of these **runs** on a skewed cell and returns a plausible wrong number, so each raises
instead of answering.

| | status | why |
|---|---|---|
| `factorization='nvf'`, sampled or integrated | supported | — |
| analytic shapes, summed | supported | lengths are absolute Cartesian; nothing deforms |
| analytic shapes, painted **overlap** | refused | resolving an overlap clips against a rectangular cell |
| vector modeler (`modeling_type=1`) | refused | rasterises onto a rectangular grid of its own |
| `factorization='li'` | refused | see below |
| field reconstruction | refused | the inverse transform is separable; a skew breaks it |

`li` applies the inverse rule along the **raster axes** (`epx = dfs2d(cell, 0, 1)`), and the solve
consumes the result as Cartesian `eps_xx` / `eps_yy`. On a skewed cell those axes are `a1` / `a2`,
not `x` / `y`. Repairing it means Li's covariant formulation (JOSA A **14**(10), 2758) — and it is
worth costing before doing, given `nvf` already converges faster on exactly the curved boundaries
this feature exists for.

Field reconstruction is deferred rather than blocked: on the *parametric* `(u, v)` grid the
lattice part is `exp(2i*pi*(m*u + n*v))`, separable again, with only the smooth incident phase
left to carry.

## How it was validated, and how to validate your own use

Energy conservation is **not** evidence here — `R + T == 1` holds to 1e-15 with the geometry read
wrongly. Three checks that are:

**A unimodular shear.** `a2' = a2 + a1` generates the *identical* point lattice: same cell area,
same reciprocal set. So the two runs must agree, and they do — 1.2e-4 on a raster, 2.0e-6 on
analytic shapes (tighter because there is no staircase in it). This is the cheapest correctness
check available and it is worth running on your own structure; `tests_verification/test_oblique_lattice.py`
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
`tests_verification/test_nvf_non_square_grid.py`.

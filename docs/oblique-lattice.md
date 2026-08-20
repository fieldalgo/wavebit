# Non-orthogonal (oblique) unit cells — what `lattice` does and what it costs

Written: 2026-08-09. Measured on CPU, NumPy backend, float64.

Targets:
- `wavebit/_core/lattice.py` — the basis, the reciprocal mapping, the Jacobian
- `wavebit/_core/solve_2d.py::order_grid_kxy` — the two k-layouts a cell can have
- `wavebit/_core/engine.py::CoreSolveMixin.lattice` — the entry point
- `wavebit/_core/engine.py::_reject_unsupported_lattice` — what is still refused, and why

---

## TL;DR

A hexagonal array used to be spelled as an `a x a*sqrt(3)` rectangular supercell holding two
motifs. That doubles the cell area, the retained order count scales with area, and the solve is
`O(N^2.7)` — so it cost about **8x**. Pass the basis instead:

```python
import numpy as np, wavebit

a = 400.0
a1, a2 = np.array([a, 0.0]), np.array([a / 2, a * np.sqrt(3) / 2])   # hex

solver = wavebit.rcwa(
    backend=0, structure=cell, max_order=[10, 10], thickness=[300.0], wavelength=900.0,
    factorization='nvf',                       # required — see "What is refused"
    lattice=np.column_stack((a1, a2)),         # columns are a1, a2
)
```

`period=(Lx, Ly)` remains the default and means `diag(Lx, Ly)`. Setting `lattice` makes `period`
report the basis-vector lengths `(|a1|, |a2|)`.

## The one thing that will bite you

**The raster is read in lattice coordinates.** `structure[layer][j, i]` is the permittivity at

    r = (i + 0.5)/Nx * a1  +  (j + 0.5)/Ny * a2

Axis 1 runs along `a1` (and pairs with `max_order[0]`), axis 0 along `a2` (with `max_order[1]`). Build the
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
| `a x a*sqrt(3)` supercell, `max_order=(14, 24)` | 1421 | 38.9 s |
| native hex cell, `max_order=(12, 12)` | 625 | 3.3 s |

The order-count ratio is exactly 2 (it is the cell-area ratio; verified numerically across
`max_order`), and the measured cost exponent is `N^2.7` — not `N^3`, because the convolution build is
sub-cubic. At matched k-space coverage that is **~8x**, and better in practice because the
supercell converges more slowly than the native cell does.

Do not confuse this with a convergence fix. On the same structure, `factorization='li'` had not
converged at `max_order=(16, 28)` after 72 s while `'nvf'` settled by `max_order=(10, 17)` in 5 s. That gap
is the *factorization*, not the lattice, and an oblique cell does nothing about it. The two wins
are independent and multiply.

## What works and what is refused

| | status | note |
|---|---|---|
| `factorization='nvf'`, sampled or integrated | supported | — |
| analytic shapes, summed | supported | lengths are absolute Cartesian; nothing deforms |
| analytic shapes, painted **overlap** | supported | the clipper is Cartesian; only the periodic images and the transform needed the basis |
| field reconstruction | supported | output grid is indexed by `(v, u)`, not `(y, x)` — see below |
| block modeler (`modeling_type=1`) | **refused** | rasterises onto a rectangular grid of its own |
| `factorization='li'` on a raster | **refused** | see below |

Everything except `li` and the block modeler works on a skewed cell, and both of those have a
supported route to the same structure — a raster, or analytic shapes.

**Analytic shapes need no keyword at all.** `factorization` resolves per modeling path, and the
analytic path implements only `nvf` — it builds `eps` from the shapes' own analytic normal field,
so there is no raster axis for `li` to run along. Its default is therefore already `nvf`, and the
`li` refusal above never applied to it. Passing `factorization='li'` with analytic shapes raises
whether or not the cell is skewed, because that path cannot express it; the `factorization='nvf'`
in the example at the top of this page is required for the **raster** it uses.

The two refusals raise rather than answering, and that matters: both of them *run* on a skewed
cell and return a plausible wrong number, with energy still conserving to 1e-15.

`li` applies the inverse rule along the **raster axes**
(`epx = fourier2d_piecewise_constant(cell, 0, 1)`), and the solve
consumes the result as Cartesian `eps_xx` / `eps_yy`. On a skewed cell those axes are `a1` / `a2`,
not `x` / `y`. Repairing it means Li's covariant formulation (JOSA A **14**(10), 2758).

Before anyone spends that: **`nvf` already contains `li`.** With a constant axis-aligned normal
the projector collapses to `C_xx = I`, `C_xy = C_yy = 0`, and the assembly reduces to
`eps_xx = <1/eps>^-1`, `eps_yy = <eps>` — Li's rules exactly, not an approximation of them (see
`factorization.normal_vector_field`). A rectangle has a constant normal on each face, so `nvf`
reproduces `li` face by face there.

Measured, on the structure most favourable to `li` — a rectangle aligned to the axes, 512x512
raster, square lattice:

| max_order | `li` | `nvf` |
|---|---|---|
| 12 | 0.86259051 | 0.86239874 |
| 15 | 0.86204124 | 0.86242349 |
| 18 | 0.86197717 | 0.86186546 |

Same limit to ~1e-4, same cost (30.0 s against 31.1 s at max_order 18). On a *curved* boundary the two
part company completely: on a circular pillar `li` had not converged at `max_order=(16, 28)` after 72 s
while `nvf` settled by `max_order=(10, 17)` in 5 s. No measured case has `li` ahead. One structure, one
wavelength, normal incidence — but the containment above is structural and holds regardless.

## Fields on a skewed cell

`reconstruct_field` works, with one change you have to handle when plotting: **samples land on the
cell's parallelogram**, `r = u*a1 + v*a2` for `u, v` in `[0, 1]`, so the returned grid is indexed
by `(v, u)` and not by `(y, x)`. Convert before drawing:

```python
u = np.linspace(0, 1, sample_x); v = np.linspace(0, 1, sample_y)
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
from `max_order=4` to `max_order=12`; **2.0e-4** in a homogeneous region of the same run, where every
surviving order is propagating and both sets hold all of them. Compare fields away from
boundaries, or at matched order sets.

## Reading the complex amplitude (not lattice-specific, but everyone hits it)

Efficiencies come out of `result.T` and are polarisation-summed, so they are always safe. The
**complex** amplitudes are split into s and p, and the one carrying the light depends on `psi`:

| `psi` | incidence | the amplitude is in |
|---|---|---|
| `0` | TM (p) | `T_p` — **`T_s` is identically zero** |
| `pi/2` | TE (s) | `T_s` — `T_p` is zero |

Reading the wrong one gives an array of zeros, and `np.angle` of a near-zero complex number is
noise that looks like a plausible phase. The safe route is the pure-TE / pure-TM variants the
solve computes anyway, where the assignment is unambiguous:

```python
res = solver.solve()
ts = np.asarray(res.s_inc.T_s)          # pure-TE incidence: T_s is the whole amplitude
c = (ts.shape[0] // 2, ts.shape[1] // 2)     # the zero order sits at the centre
t0 = complex(ts[c])
amplitude, phase = abs(t0), np.angle(t0)
```

Note that `|t0|**2` is not the transmitted efficiency: power into a substrate of index `n_sub`
carries a factor of `n_sub` (at normal incidence). On a metalens cell with `n_sub = 1.46`,
`|t0|**2 = 0.67` and `sum(T) = 0.98` are the same result — check that they agree before
trusting either.

## How it was validated, and how to validate your own use

Energy conservation is **not** evidence here — `R + T == 1` holds to 1e-15 with the geometry read
wrongly. Four checks that are, the first of them against numbers from the literature:

**Li's checkerboard, against published values.** The benchmark from Li, JOSA A **14**(10), 2758
(1997), as tabulated by Demésy et al. in *Gratings: Theory and Numeric Applications* (2012),
Table 5.4. Parameters: `phi0 = theta0 = 0`, `psi0 = 45 deg`, `dx = dy = 5*lambda*sqrt(2)/4`,
`h = lambda`, `eps+ = 2.25`, `eps- = 1`.

This is the ideal case for the feature, because a checkerboard **is** an oblique-cell problem: on
a square cell it needs two blocks, and on its own primitive cell — a square rotated 45 degrees,
which `is_axis_aligned` correctly reports as *not* axis-aligned — it needs one. The paper's
`dx = dy = 5*sqrt(2)/4 = 1.767767` is exactly that primitive cell's edge.

| order | Li FMM | FEM | Wavebit | vs FEM |
|---|---|---|---|---|
| T(-1,-1) | 0.04308 | 0.04333 | 0.04318 | -1.5e-4 |
| T(-1,0) | 0.12860 | 0.12845 | 0.12850 | +5e-5 |
| T(-1,+1) | 0.06196 | 0.06176 | 0.06177 | +1e-5 |
| T(0,-1) | 0.12860 | 0.12838 | 0.12850 | +1.2e-4 |
| **T(0,0)** | 0.17486 | 0.17577 | **0.17556** | -2.1e-4 |
| T(0,+1) | 0.12860 | 0.12839 | 0.12850 | +1.1e-4 |
| T(+1,-1) | 0.06196 | 0.06177 | 0.06177 | 0 |
| T(+1,0) | 0.12860 | 0.12843 | 0.12850 | +7e-5 |
| T(+1,+1) | 0.04308 | 0.04332 | 0.04317 | -1.5e-4 |

Worst 2.06e-4 over all nine, at `max_order=(16,16)`; sum of reflected orders 0.10057 against the
paper's 0.10040. For scale, the paper's own FMM and FEM differ by 9.1e-4 on T(0,0) — this sits
closer to FEM than those two sit to each other. `tests/test_oblique_lattice.py` has it as
`test_li_1997_checkerboard_matches_the_published_table`, at `max_order=(12,12)` so it stays a test
rather than a study.

**Watch the polarisation frame when reproducing it.** `psi0 = 45 deg` in the paper is measured
from *its* cell axes, and those axes are the rotated basis, so in Wavebit's Cartesian frame it
is `psi = pi/2`. Getting this wrong is not subtle if you look at the right thing: the wrong frame
swaps which pairs of orders are degenerate — `psi=0` makes T(-1,-1) equal T(-1,+1) where the
paper has them split — and the error grows to 1.9e-2 while T(0,0) barely moves, because T(0,0) is
polarisation-independent on a C4v cell. **Compare the whole order table, not the zeroth order.**

Three more:

**A unimodular shear.** `a2' = a2 + a1` generates the *identical* point lattice: same cell area,
same reciprocal set. So the two runs must agree, and they do — 1.2e-4 on a raster, 2.0e-6 on
analytic shapes (tighter because there is no staircase in it). This is the cheapest correctness
check available and it is worth running on your own structure; `tests/test_oblique_lattice.py`
has it as `test_unimodular_shear_is_the_same_physics`. With the normal field's metric correction
removed the same comparison drifts to 8.5e-4 — that is the test having teeth.

**S4 as an outside oracle, against a square-lattice control.** S4 supports arbitrary lattice
vectors, so it checks the reciprocal mapping independently. It sits 6.60e-3 from Wavebit on a
square lattice — where none of this code runs — and 6.77e-3 on hex. The gap is S4's own slow
convergence, equal on both, so the oblique path adds **1.7e-4** over the already-validated
orthogonal one. (Use S4's default Laurent rule for this. `PolarizationDecomposition=True`
diverged on *both* lattices in our hands, 0.986 -> 0.933, and is not usable as an oracle.)

**The supercell it replaces.** A native hex cell reproduces its rectangular supercell; the
residual is the two rasters staircasing the feature differently, so it tightens with resolution
rather than with `max_order`.

## A worked case: a visible metalens phase library

TiO2 pillars (n = 2.43) on fused silica (n = 1.46), hex pitch 350 nm, height 600 nm, at 532 nm —
the geometry family of Khorasaninejad et al., Science **352**, 1190 (2016). Note the pitch check
this design depends on: `|G|_min = 0.02073 /nm` against `k0*n_sub = 0.01724 /nm`, so nothing but
the zero order propagates into the substrate.

Sweeping the pillar diameter from 80 to 260 nm gives a usable library — phase spans 6.648 rad
(1.06 x 2*pi), **monotonic**, so the phase-to-diameter map inverts. Three diameters (180, 210,
220 nm) sit on resonances and drop to T = 0.50-0.79; excluding them leaves 16 of 19 points with
a worst transmission of 0.936. The whole sweep takes 14 s.

Cross-checked against S4 on the same structure, which is the useful part: at D = 150 nm S4
converges to 0.96675 against Wavebit's 0.96643 (3.5e-4), and at D = 240 nm to 0.99694 against
0.99905 (2.1e-3) while still climbing. Both gaps are S4's own convergence — and Wavebit is
already at its final value with 289 orders, where S4 has not settled at 595. That is the
staircase-free shape transform and NVF together, not the lattice.

The same structure as a rectangular supercell agrees to 6.0e-4 and takes 2.9x as long
(`max_order=(8,14)`, 493 orders, 2.57 s against 289 orders, 0.89 s). Note that is **~3x, not the ~8x**
quoted above: 8x is the asymptotic ratio on a converged high-contrast structure, and a metalens
library at `max_order=8` is not in that regime — the order ratio there is 1.71, not 2. Expect ~3x on
this kind of sweep and ~8x when you genuinely need the harmonics.

## Where the skew actually reaches the code

Almost nowhere, which is why this was small. Bloch's theorem gives the retained orders Cartesian
components `k_inc + lambda * A^-T @ (m, n)` and that is the whole of it: the eigenproblem reads
only diagonal `Kx` / `Ky`, and substituting `r = u*a1 + v*a2` leaves

    eps_hat[m, n] = \int_0^1 \int_0^1 eps(u, v) exp(-2i*pi*(m*u + n*v)) du dv

an ordinary DFT on the parametric grid for *any* basis — still indexed by integer differences,
still block-Toeplitz. The `dfs2d_*` builders, the Toeplitz builds and `final_matching` are
untouched. Truncation
stays rectangular in `(m, n)`; Li introduced circular truncation *in* the non-orthogonal setting
and still found the parallelogramic one superior, and it measured worse here too.

The one genuinely Cartesian-geometric quantity is NVF's normal field, which is built from a
distance transform and a gradient taken in **index** space and so needs the metric (`J^-T`). That
correction is why NVF rather than Li's nested rule is the factorization that ports: `_nvf_assemble`
is written in Cartesian tensor components, so once the normal is Cartesian nothing else in the
factorization learns the cell was skewed. The same correction fixes an orthogonal case that was
also wrong — a raster whose pixels are not square in real space; see
`tests/test_nvf_non_square_grid.py`.

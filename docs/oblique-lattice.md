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
motifs. That doubles the cell area, the retained order count scales with area, and the measured
cost exponent is 2.86 -- so it cost **7.3x**. Pass the basis instead:

```python
import numpy as np, wavebit

solver = wavebit.rcwa(
    backend=0, structure=cell, truncation_order=dict(x=10, y=10), thickness=[300.0], wavelength=900.0,
    factorization='nvf',                       # required — see "What is refused"
    lattice=wavebit.hexagonal(400.0),          # or lattice=np.column_stack((a1, a2))
)
```

`period=(Lx, Ly)` remains the default and means `diag(Lx, Ly)`. Setting `lattice` makes `period`
report the basis-vector lengths `(|a1|, |a2|)`.

## Spelling the cell

`lattice` is always a 2x2 whose **columns** are `a1` and `a2`, and you can write it by hand.
Three constructors spell the usual ones:

```python
wavebit.rectangular(700.0, 1100.0)          # diag(700, 1100); `ly` defaults to `lx`
wavebit.hexagonal(400.0)                    # 120-degree rhombus, the crystallographic setting
wavebit.hexagonal(400.0, np.radians(60))    # the 60-degree alternative basis
wavebit.oblique(400.0, 520.0, np.radians(105))
```

`angle` is the angle between `a1` and `a2`, in radians, like every other angle in this API.
These are spellings and not a new kind of cell — each returns the same object `lattice=` takes,
and nothing stores an angle afterwards. Which is what makes the angle differentiable: build the
basis *inside* the function you differentiate and it is a design variable, for the ordinary
reason any expression is. See "Where the skew actually reaches the code".

**Building the structure.** `wavebit.grid` gives the Cartesian coordinates of the raster's pixel
centres, so a layer can be written in coordinates rather than indices:

```python
A = wavebit.hexagonal(400.0)
X, Y = wavebit.grid((256, 256), lattice=A)
layer = np.where(X ** 2 + Y ** 2 < 90.0 ** 2, 12.25, 1.0)
```

**The cell is centred on the origin**, so the pillar above needs no arithmetic to be centred.
That matters more than it looks here: a pillar sits at the *lattice point* in most
descriptions, and the lattice point is a **corner** of the cell — put it there and the raster
holds one wedge of it, with the other wedges belonging to the neighbouring copies. Correct as
a periodic structure only if you draw all of them; putting the pillar in the middle of the
cell is the same array, described once, and now it is also the shortest thing to write. Until
2026-08-31 this example opened by computing `0.5*(a1 + a2)` by hand.

On a skewed cell this is the step that is easy to get wrong by hand — the samples are at
`u*a1 + v*a2`, not on a rectangle, and `np.meshgrid` alone will not put them there. An
expression written in `X` and `Y` also cannot get the axis order wrong, since `where` is
elementwise and the layout simply follows; one written in indices (`layer[:2] = ...`) can, and
does so silently on a non-square raster.

## The one thing that will bite you

**The raster is read in lattice coordinates.** `structure[layer][j, i]` sits at the parametric
point `u = (i + 0.5)/Nx`, `v = (j + 0.5)/Ny`, and the cell is centred on the origin:

    r = (u - 0.5) * a1  +  (v - 0.5) * a2

Axis 1 runs along `a1` (and pairs with `truncation_order[0]`), axis 0 along `a2` (with `truncation_order[1]`). Build the
raster on a **Cartesian** grid and hand it a skewed basis and you have described a *different
structure* — and nothing will tell you. There is no error, the transform is still a transform,
and energy still conserves to 1e-15 because the NVF assembly is Hermitian whatever the geometry
says. The way to write it:

```python
X, Y = wavebit.grid((ny, nx), lattice=A)         # Cartesian position of every pixel
```

then test membership in Cartesian space, with **minimum image over the lattice** (`i*a1 + j*a2`
for `i, j` in `-1, 0, 1`) so a feature straddling an edge wraps correctly. Written out by hand
it is `((np.arange(n) + 0.5)/n - 0.5)` per axis, then `U*a1 + V*a2` — the `- 0.5` being the
centred cell. `wavebit.grid` exists so that neither half has to be remembered.

Analytic shapes do not have this problem *in their lengths*: those are absolute Cartesian
quantities, so a rectangle stays a rectangle under a skew. `cx` / `cy` are Cartesian too, and
they are measured from the same origin the raster grid above uses — the centre of the cell —
so `cx = cy = 0` is a centred shape on every cell, skewed or not.
Their **periodic fold** is a separate thing and it was wrong until 2026-08-23. The nearest
image of a shape sits at `m*a1 + n*a2`; the fold was written `d % Px`, and `period` on a skewed
cell reports `(|a1|, |a2|)`, so it reduced towards a point where no image is. On a 60-degree
cell it looked straight up while the neighbour was up and to the right. It only shows once a
feature is large enough that a neighbour's surface reaches into the sampled cell -- measured on
a disc at `r/a = 0.45`, 21% of the surface band was more than 0.05 out and the solve was
1.1e-3 from the raster path, which the fix brings to 5e-6. Energy conserved to 1e-12 either
way, and an axis-aligned cell is bit-identical, which is why nothing caught it.

## What it buys

Hex array of Si pillars, pitch 400 nm, r = 90 nm, h = 300 nm, lambda = 900 nm, normal incidence.
The supercell and the native cell describe the same infinite array.

| | orders | wall clock |
|---|---|---|
| `a x a*sqrt(3)` supercell, `truncation_order=(14, 24)` | 1421 | 38.9 s |
| native hex cell, `truncation_order=(12, 12)` | 625 | 3.3 s |

**Those two rows are not matched in accuracy**, and the 11.8x they invite is not the number:
1421 against 625 is an order ratio of 2.27, and the supercell's truncation covers *less* of
k-space than the hex row's (0.281 against 0.329 in the same units). They are here because they
are the two spellings a user actually writes.

The ratio that is measured, and that holds:

* **Order density is the cell-area ratio, 2.0.** Retained orders against the k-space area each
  truncation covers: 3719 per unit area on the native cell, 7351 on the supercell -- a ratio of
  **1.98** against a cell-area ratio of 2.00. Half of the supercell's orders are **fictitious**:
  they sit on the coset the finer reciprocal lattice adds, where the structure factor of the
  two-motif basis vanishes identically, so `eps` cannot couple to them and the source never
  excites them. They are carried at `O(N^p)` anyway.
* **The cost exponent is 2.86**, fitted over 529 / 841 / 1225 orders on the native cell
  (4.31 / 15.79 / 47.63 s); per-step exponents 2.80 and 2.94. Sub-cubic because the convolution
  build is.

Together: **2.0x the orders, 4x the eigenproblem array, and `2^2.86 = 7.3x` the time** at
matched k-space coverage.

This page used to add "and better in practice because the supercell converges more slowly than
the native cell does". **That is withdrawn: it was never measured, and it does not resolve.**
Both arms converge oscillatorily with an amplitude around 2e-4 on this structure, so
first-run-below-tolerance gives 7.6x / 2.6x / 2.6x / 33.2x as the tolerance moves from 1e-3 to
2e-4 -- noise, not a measurement. The order ratio is the area ratio and this page claims nothing
beyond it.

Do not confuse this with a convergence fix. On the same structure, `factorization='li'` had not
converged at `truncation_order=(16, 28)` after 72 s while `'nvf'` settled by `truncation_order=(10, 17)` in 5 s. That gap
is the *factorization*, not the lattice, and an oblique cell does nothing about it. The two wins
are independent and multiply.

## What works and what is refused

| | status | note |
|---|---|---|
| `factorization='nvf'`, sampled or integrated | supported | — |
| analytic shapes, summed | supported | lengths are absolute Cartesian; nothing deforms |
| analytic shapes, painted **overlap** | supported | the clipper is Cartesian; only the periodic images and the transform needed the basis |
| every `normal_method` on a raster | supported | since 2026-08-21; **use `'contour'`** for accuracy, **not** where a feature overlaps its own images — see below |
| field reconstruction | supported | output grid is indexed by `(v, u)`, not `(y, x)` — see below |
| `factorization='li'` on a raster | **refused** | see below |

Everything except `li` works on a skewed cell, and `li` has a supported route to the same
structure — analytic shapes, which carry the metric. The block modeller was refused here too,
on the same grounds (it rasterised onto a rectangular grid of its own); that whole path was
removed on 2026-08-26.

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

| truncation_order | `li` | `nvf` |
|---|---|---|
| 12 | 0.86259051 | 0.86239874 |
| 15 | 0.86204124 | 0.86242349 |
| 18 | 0.86197717 | 0.86186546 |

Same limit to ~1e-4, same cost (30.0 s against 31.1 s at truncation_order 18). On a *curved* boundary the two
part company completely: on a circular pillar `li` had not converged at `truncation_order=(16, 28)` after 72 s
while `nvf` settled by `truncation_order=(10, 17)` in 5 s. No measured case has `li` ahead. One structure, one
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
from `truncation_order=4` to `truncation_order=12`; **2.0e-4** in a homogeneous region of the same run, where every
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

(Run through the *analytic shape* path, where the geometry is exact. Rasterized instead, the
same benchmark lands at 3.1e-3 at N=256 whatever the `normal_method` — that is the staircase,
and it is why this table cannot arbitrate a normal field.)

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

Worst 2.06e-4 over all nine, at `truncation_order=(16,16)`; sum of reflected orders 0.10057 against the
paper's 0.10040. For scale, the paper's own FMM and FEM differ by 9.1e-4 on T(0,0) — this sits
closer to FEM than those two sit to each other. `tests/test_oblique_lattice.py` has it as
`test_li_1997_checkerboard_matches_the_published_table`, at `truncation_order=(12,12)` so it stays a test
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
rather than with `truncation_order`.

**Which hex spelling.** `wavebit.hexagonal(a)` is the 120-degree cell, the crystallographic
setting: `a1 = (a, 0)`, `a2 = (-a/2, a*sqrt(3)/2)`. 60 degrees spans the same lattice and is one
argument away (`hexagonal(a, np.radians(60))`), and it is what the numbers on this page were
originally measured on. They carry over: on the pillar array above the two spellings agree to
**1e-13** at `truncation_order` 6, 9 and 12 — machine precision, because a pillar at the lattice
point has the lattice's own six-fold symmetry, so the orders each truncation keeps and the other
drops are related by it and contribute alike. **That is a property of this structure, not of the
spelling.** Break the symmetry and the two differ by their truncation error, since
`truncation_order` cuts a parallelogram in `(m, n)` and a different basis maps that onto a
different set of `G` — 81 orders either way at `truncation_order=4`, 61 of them shared. And an
*omitted* shape centre moves outright, since the fundamental cell is a different parallelogram:
`(100, 173)` against `(300, 173)` at `a = 400`.

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
(`truncation_order=(8,14)`, 493 orders, 2.57 s against 289 orders, 0.89 s). Note that is **~3x, not the ~7x**
quoted above: 7.3x is the asymptotic ratio on a converged high-contrast structure, and a metalens
library at `truncation_order=8` is not in that regime — the order ratio there is 1.71, not 2. Expect ~3x on
this kind of sweep and ~7x when you genuinely need the harmonics.

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

## Which `normal_method` to use, and what the default costs here

> **`'contour'` is the one raster construction that is not fully periodic, and a skewed cell is
> where what remains of that bites.** Rolling a raster -- the same structure written from a
> different corner, which every answer must be invariant under -- moved `sum(T)` by 3.9e-4 on a
> square cell and 9.7e-4 on a 2:1 cell, where `'distance'`, `'spectral'` and `'smooth'` held at
> 4e-14. Two of the three causes are fixed; read the rest of this block for which, because the
> version of it that shipped before 2026-09-01 was both stale and wrong about its own scene. The cause was `_two_phase_indicator` reading the background off the
> *corner pixel*, which a translation can move inside the feature; it marks the smaller phase
> now. It was **not** `_resample`'s phase, which is what this caveat said until then, and it
> was not a large-feature problem either.
>
> **The skewed-cell half of this caveat was stale, and its scene was not what it said.** It
> read: "a disc of radius 30 in a 64-px hex cell still moves by 2.3e-4 over ten rolls", blamed
> the **medial axis**, and told you to prefer `'spectral'` on a skewed cell. Two things were
> wrong with it.
>
> *The number does not reproduce.* On that scene, today: `'contour'` **8.1e-15 on 0 of 63
> rolls**, the annulus 6.1e-15 on 0 of 63, against the 2.3e-4 and 2.2e-3 the text quoted.
> `9a84f60` (the tile ring is read from the cell) and `876c1c4` (the fold landed the
> representative on the origin rather than on the cell) fixed it and this block was never
> re-taken. There is no skewed-cell preference for `'spectral'` left to state.
>
> *The scene was an ellipse.* "A disc of radius 30" was `_disc(X, Y)` in
> `tests/test_nvf_lattice_translation.py`, whose `X` and `Y` were raster **indices**. The raster
> is read at `(i + 0.5)/Nx * a1 + (j + 0.5)/Ny * a2`, so that mask is the image of a disc under
> the basis: **1.73:1 at 60 degrees, 3.73:1 at 30, 5.67:1 at 20.** The helper is `_index_disc(U, V)`
> since `c59d221` and its docstring says so. That does not invalidate what the tests measure --
> periodicity is a property of the raster and of the code that reads it, not of the shape being
> round -- but it does invalidate calling the scene a disc.
>
> **The accuracy tables below are not affected**, and that was checked rather than assumed: they
> come from `tests/test_nvf_normal_methods.py::skewed_disc`, which builds through `wavebit.grid`
> with a minimum-image membership test and is a genuine Cartesian circle. Every printed digit
> reproduces.
>
> **What is still open** is the *open-curve* path, not the medial axis: a level set that crosses
> the cell is closed on the torus, but `_contour_polylines` hands contourpy a finite 3x3
> rectangle and the cut fragments get neither the fold nor the dedup that closed loops get.
> Measured on a Cartesian circle large enough to touch its neighbours at 20 degrees:
> `'contour'` **1.0e-01 on 63 of 63 rolls**, where `'distance'`, `'spectral'` and `'smooth'`
> hold at 4e-15. Pinned by `test_a_percolating_level_set_is_periodic` and
> `test_a_closed_loop_does_not_delete_the_percolating_interface_beside_it`, both
> `xfail(strict=True)`. **Prefer `'distance'`, `'spectral'` or `'smooth'` where a feature
> overlaps its own periodic images** -- on a strongly skewed cell that is the ordinary case, not
> an exotic one: at 20 degrees with `|a1| = |a2| = 640` any circle of radius over 109 is in it.

**Use `normal_method='contour'`.** The paragraph above describes the *default* construction,
`'distance'`, and it is the least accurate of the four the raster path offers. A skew makes it
worse rather than better: the handful of directions a pair of integer central differences can
spell stays put while the cell they are read in shears away from them.

| gamma | `distance` | `contour` | `spectral` | `smooth` |
|---|---|---|---|---|
| 90 | 15.6 | 0.96 | 0.75 | 0.28 |
| 60 | 18.2 | 0.98 | 0.89 | 0.16 |
| 30 | 23.7 | 1.31 | 0.96 | 0.48 |

Mean angular error in degrees against a disc's exact normal (r=60 in a 400-unit cell, N=128;
`gamma` is the angle between `a1` and `a2`). At the solve, on a hex rod against a
staircase-free analytic reference, the three replacements beat `'distance'` by 2.0x to 4.3x at
every grid and truncation measured.

All three were **refused** on a skewed cell until 2026-08-21, on the grounds that only
`'distance'` carried the `J^-T` pullback. That was a fact about the code rather than about the
constructions: `'contour'` measures distances between Cartesian points and `'spectral'`
multiplies by a Cartesian wavevector, so neither takes an index-space difference at all and
neither needs a pullback — what they needed was the coordinates. `'smooth'` did need work and
got it, since scipy's distance transform and its Gaussian are both per-axis and a skewed metric
is not diagonal. [`nvf-normal-methods.md`](nvf-normal-methods.md) has all of it: the
per-construction table, the solver A/B, and the two axis-aligned cells that changed with it.

Two things not to expect. `'contour'` needs the optional `contourpy`
(`pip install wavebit[contour]`), and Li's checkerboard above will not show you the difference:
it is straight-edged, so a raster of it carries a permittivity staircase that all four
constructions share to within 5%. Pick a curved feature to see the gain.

# The NVF normal field and the grid it is taken on — measurements

Written: 2026-08-07. Measured on CPU, NumPy backend, float64.

Targets:
- `fieldwave/_core/factorization.py::_normal_field_floor` — the floor that replaced the knob
- `fieldwave/_core/factorization.py::normal_vector_field` — the distance-transform construction
- the `enhanced_dfs` parameter — removed outright, not deprecated: nothing read it, so
  leaving it to be set would have been a lie rather than a compatibility shim

---

## TL;DR

`enhanced_dfs=True` repeated every cell `4*fto+1` times before taking the normal field. Across
32 combinations of shape, material, polarization and truncation order it **lost accuracy in 27
and gained in none that stood up**. It is gone — the parameter with it, from the solver
constructors and from `Numerics`, so passing it now raises rather than being quietly ignored.
The normal field is taken on the raster as given, lifted only to the `4*fto+1`-point floor its
transform structurally requires.

## What the knob did

```python
if enhanced_dfs:  min = (4*fto + 1) * N     # per cell
else:             min = 4*fto + 1           # total
```

Verbatim from `meent`. What changed underneath it is where it lands:

| | in `meent` | now |
|---|---|---|
| sampled-path permittivity coefficients | its main purpose | **ignored** — obtained in closed form since `072710d` |
| NVF normal field | (NVF did not exist) | its only remaining reader |

So a flag whose reason for existing was the permittivity survived into a codebase where the
permittivity no longer reads it.

At `fto=6`, `N=240` the two settings differ by a factor of 26 per axis — a 6240x6240 grid
against 240x240, 676x the cells, with a Euclidean distance transform run over it.

## Why refining it does not help

The normal field is not a property the raster *has*. Asked honestly, a raster's boundary is a
staircase and its normals point along the axes — which is exactly the case where NVF collapses
onto Li's rules and buys nothing. So `normal_vector_field` does something else: it takes the
distance transform of the boundary set and reads its gradient, which on a staircase comes out
roughly radial. It is a **guess at the smooth boundary the staircase stands for**.

Refining the grid does not improve the guess, because upsampling adds no information — the
staircase is the same staircase, spelled with more pixels. What it does is resolve that
staircase more sharply, so the guess tracks the steps rather than the shape behind them.

The measurement separates the two cleanly. Building the **same final grid** two ways, from a
real 480x480 raster of a circle, and from a 60x60 raster repeated 8x:

| | `fto=6` | `fto=10` |
|---|---|---|
| N=480, no repeat (real resolution) | **3.65e-04** | **1.89e-03** |
| N=60 repeated 8x (same grid) | 1.28e-02 | 1.18e-02 |
| N=60, no repeat | 5.03e-03 | 5.53e-03 |

Same number of grid points, 35x apart. Refinement that carries information converges;
refinement that duplicates it does not, and costs accuracy.

Raising the real raster resolution behaves as it should:

| N | 30 | 60 | 120 | 240 | 480 |
|---|---|---|---|---|---|
| NVF error (dielectric circle, `fto=6`) | 7.63e-03 | 5.03e-03 | 1.43e-03 | 1.33e-03 | 3.65e-04 |

## The sweep

Base raster 240x240 (fine enough that the staircase error is small), truth from the analytic
shape path at `fto=16`, error on the specular transmitted order.

| case | `fto` | sampled | R=1 | R=2 | R=4 |
|---|---|---|---|---|---|
| dielectric circle, TM | 4 | 1.60e-02 | **3.73e-03** | 5.34e-03 | 6.43e-03 |
| | 6 | 1.25e-02 | **1.33e-03** | 2.86e-03 | 3.90e-03 |
| | 8 | 9.92e-03 | **2.22e-03** | 3.72e-03 | 4.71e-03 |
| | 10 | 8.35e-03 | **2.71e-03** | 4.17e-03 | 5.11e-03 |
| dielectric rect 30°, TM | 4 | 2.38e-02 | **3.43e-03** | 5.13e-03 | 6.07e-03 |
| | 6 | 1.81e-02 | **1.67e-03** | 3.29e-03 | 4.23e-03 |
| | 8 | 1.43e-02 | **1.88e-03** | 3.51e-03 | 4.48e-03 |
| | 10 | 1.16e-02 | **2.32e-03** | 3.95e-03 | 4.90e-03 |
| dielectric rect 10°, TM | 4 | 4.31e-03 | 5.29e-03 | 6.57e-03 | 7.13e-03 |
| | 6 | 3.77e-03 | **2.76e-03** | 3.85e-03 | 4.33e-03 |
| | 8 | 3.02e-03 | **2.67e-03** | 3.60e-03 | 4.02e-03 |
| | 10 | 2.41e-03 | **1.84e-03** | 2.67e-03 | 3.05e-03 |

Over all 32 rows (adding metallic circle and rect, and both polarizations): **27 worse with
upsampling, 4 unchanged, 1 better** — and that one is a case whose analytic reference had not
converged (`fto=12` residual 8.7e-03), so it does not count for anything.

Two controls make the separation exact:

- **The permittivity coefficients are invariant under upsampling** — measured at 1e-16, so
  everything above is the normal field and nothing else.
- **Repeating the `ucell` manually and repeating it via `enhanced_dfs=True` are bit-identical**
  (0.00e+00), so the manual sweep measures the real setting.

## Why the floor stays

The floor is not an accuracy preference, it is structural. The normal field has no closed-form
transform (unlike the permittivity, which `_pixel_correction` handles for any order), so
`fourier2d_pointwise` reads it off the grid, and the circulant asks for orders up to `±2*fto`. On a grid
shorter than `4*fto+1`, `k % n` folds distinct orders onto one bin.

The consequence is not gradual. A **constant** normal field must transform to the identity for
NVF to collapse onto Li's rules on a one-cell axis. On a short grid it instead picks up
off-diagonals wherever `(r - c) % n == 0`, and the collapse fails outright — a 1D grating of 8
cells at `fto=10` returned 0.1236 against Li's 0.0324. That is what
`test_nvf_on_a_one_cell_axis_reduces_to_li` pins.

So: lift to the floor, never past it.

## What this cost

`2d_nvf` in `golden_legacy.npz` was recomputed. It is a random 5x5 raster, where the grid goes
from 50x50 to 10x10 and the answer moves a long way (transmitted efficiency 0.684 to 0.352
under TM incidence; energy conserved to 1.000000000000 either way). There is no independent
oracle for that case and there cannot really be one — NVF's premise is that a staircase stands
for a smooth boundary, and a random raster has none. The justification is this document.

## Reproduce

The sweeps here are not checked in as a benchmark; they are one-off measurements against the
analytic shape path. Two neighbouring write-ups cover the questions either side of this one:
[`analytic-modeling.md`](analytic-modeling.md) on how the smooth path compares to the raster
and staircase representations, and [`backend-choice.md`](backend-choice.md) on what the change
here did to the cost of NVF relative to the other two strategies (it removed the difference).

### References

- L. Li, *Use of Fourier series in the analysis of discontinuous periodic structures*, JOSA A
  **13**(9), 1870 (1996). [doi:10.1364/JOSAA.13.001870](https://doi.org/10.1364/JOSAA.13.001870)
  — the rules NVF must collapse onto when the normal field is constant, which is what the
  `4*fto+1` floor protects.
- T. Schuster, J. Ruoff, N. Kerwien, S. Rafler, W. Osten, *Normal vector method for convergence
  improvement using the RCWA for crossed gratings*, JOSA A **24**(9), 2880 (2007).
  [doi:10.1364/JOSAA.24.002880](https://doi.org/10.1364/JOSAA.24.002880) — the method itself.
- P. Götz, T. Schuster, K. Frenner, S. Rafler, W. Osten, *Normal vector method for the RCWA with
  automated vector field generation*, Opt. Express **16**(22), 17295 (2008).
  [doi:10.1364/OE.16.017295](https://doi.org/10.1364/OE.16.017295) — the paper that addresses
  generating the field for an arbitrary shape, which is the problem the distance transform here
  solves. Whether it is the *same* construction has not been checked against the full text; see
  the same caveat in [`analytic-modeling.md`](analytic-modeling.md).

# Polynomial body fields in the rectangular NVF research model

`studies/rcwa/nvf_polynomial_body/nvf_polynomial_body.py` implements a higher-degree conforming potential
inside the body of the rectangular gap model. At the same 5039 scalar unknowns,
the degree-4 body narrows the relative static enclosure by about 37% compared
with the preceding refined linear body. At outer order 16, its s transmission
discrepancy against the independently fitted auxiliary field decreases by about
82%. A degree-3 alternative uses 3887 unknowns and also improves that comparison.

This is an implemented and tested **research model**, not an update to the
production `wavebit.rcwa` NVF dispatch. General geometry and outer optical
convergence remain open. The fitted-cell comparison is not an independent
3D Maxwell error certificate.

## Enriching the forward field while preserving its traces

The preceding [continuous-flux model](nvf-continuous-flux.md) represents the
body potential with periodic P1 functions in normalized y. In each body interval,
let r run from 0 to 1. Retain the two endpoint functions and add

```
phi_left(r) = 1-r,     phi_right(r) = r
beta_k(r) = integral_0^r P_k(2*u-1) du,     k = 1,...,body_degree-1
beta_k(0) = beta_k(1) = 0
```

P_k is the Legendre polynomial. Each interval gets its own interior bubbles;
neighboring intervals still share endpoint unknowns. The potential is continuous
through every body interface and remains periodic. The added functions vanish
at the two gap faces, so the existing gap trace and contact coupling need no
new constraint, cutoff or topology change.

For E body intervals and degree p there are E*p periodic body functions.
Writing the normalized body coordinate as t in [-1/2,1/2], the same trial
potential is now

```
chi_body(x,t) = v(x,t) - g*t*a(x)
v(x,t) = sum_A Phi_A(t)*v_A(x)
physical y = (1-g)*t
partial_y chi_body = (partial_t v - g*a)/(1-g)
```

The affine x lift, material-integrated x flux basis and finite-gap bubbles
remain those of the preceding formulation. The body stiffness, mass, load and
ramp coupling are assembled from the enriched Phi_A. The existing block cell
solve uses these matrices, with the same analytic zero-gap row scaling and
live diagonal equilibration. Its acceptance residual remains the residual of
the original equations.

The local derivative integrals use Legendre orthogonality:

```
integral_0^1 beta_k'(r)*beta_l'(r) dr = delta_kl/(2*k+1)
integral_0^1 phi_left/right'(r)*beta_k'(r) dr = 0
```

Keeping those zero couplings exactly zero avoids dividing monomial cancellation
error by the tiny contact interval width. The remaining local mass and load
terms use the corresponding analytic Legendre integrals. This is numerical
assembly of the same variational equations, with no regularization or fitted
optical term.

`body_degree=1` reproduces the preceding `MaterialGradedGapModel` coefficients
exactly. The opt-in class defaults to degree 2 and supports degrees 1 through
4. The convergence instrument keeps degree 1 as its default. The physical
material function still defines the x grading; body bubbles introduce no
material-equality branch and do not make primitive partitions affect the mesh.

## Integrating the actual energy

Within each body interval the potential is a degree-p polynomial in r and its
y derivative is degree p-1. The projection integrates both the physical basis
B and its energy `B.H epsilon B`, using exact Fourier moments through degree
2*p in y and the existing material-jump moments in x. The final eight
coefficient arrays feed the same retained energy closure. Body and gap
projection are now separate methods so the original model can retain its
linear body implementation as an exact control.

Independent tests evaluate the local basis using Legendre function values and
Gauss quadrature, without the polynomial assembly formulas. A second tensor
quadrature computes the full field and weighted energy at physical x/y points.
Both checks cover degrees 2, 3 and 4. Nested degree refinements are also checked
for narrowing primal/complementary energy bounds and preserving unit mean B.

The degree-2 model passes the inherited redundant-partition, translated shape,
independent contact, closed-laminate, material and real-seam tests. Additional
degree-4 JIT checks cover gap, seam and both material derivatives. The final
degree-4 resolution also passes the independent p/s contact criterion with
conditioning warnings promoted to errors. None of those physical tolerances
was relaxed.

For the final degree-4 setting, the order-7 epsilon-12.25 contact check gives
p/s derivative discrepancies of -2.55e-11 / 2.01e-8 per nm against independent
Cartesian secants extrapolated from steps 1e-4 and 5e-5 nm. Both satisfy the
existing 1e-7 criterion. The degree-3 alternative also passes this independent
contact check; the raw outputs retain the values and true cell residuals.

## Same-dimensional and smaller-dimensional comparisons

The fixture has period 700 nm, width 350 nm, gap 17 nm, seam 31 nm and materials
4/9 in background 1. All configurations below use base rows 8, x subdivisions 2,
gap degree 6, edge fraction 1e-5, grading strength 8 and radius .01.

| body degree | body y subdivisions | scalar unknowns | relative static enclosure |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 5039 | 0.000182637470 |
| 2 | 1 | 2735 | 0.000193621778 |
| 2 | 2 | 5039 | 0.000126463083 |
| 3 | 1 | 3887 | 0.000124629888 |
| 4 | 1 | 5039 | 0.000115356596 |

At the same dimension, degree 4 narrows the enclosure by about 37%. Degree 3
uses about 23% fewer unknowns than the preceding 5039-unknown model while
also narrowing its enclosure. These are measured allocation choices for this
fixture, not universal optimality claims.

The degree-4 diagonal interval is

```
1.670771622439439 <= Uxx <= 1.670783983361078
3.184220811164662 <= Uyy <= 3.184588132038305
```

The recorded primal/reciprocal residuals are below 2.5e-12. The independently
fitted FEM interval lies inside every candidate interval in the table. Static
cell uncertainty remains; it is not an optical transmission bound.

Use wavelength 900 nm, thickness 300 nm, incidence angle .2 radians, incident
permittivity 1 and substrate permittivity 2.25. At outer order 16:

| auxiliary field | p T | s T | s difference from preceding FEM control |
| --- | ---: | ---: | ---: |
| preceding fitted FEM, h=.04 | 0.907165915611 | 0.941480649651 | — |
| preceding degree-1 body, 5039 unknowns | 0.907167633867 | 0.941498495500 | 1.78458e-5 |
| degree-3 body, 3887 unknowns | 0.907167186862 | 0.941485573526 | 4.92387e-6 |
| degree-4 body, 5039 unknowns | 0.907166994307 | 0.941483923649 | 3.27400e-6 |

Degree 4 reduces the s discrepancy by about 82%, and the p discrepancy from
1.71826e-6 to 1.07870e-6. Degree 3 reduces the s discrepancy by about 72% while
using fewer unknowns.

At outer order 24, degree 4 gives p/s T of
0.906794978336 / 0.941570869123. The preceding fitted-cell projection gives
0.906794061748 / 0.941568404449. The corresponding discrepancies are
9.16588e-7 / 2.46467e-6. The s discrepancy of the preceding linear-body model
was 1.67667e-5 at this order, so the improvement persists at a higher order.

The [directional comparison](nvf-directional-convergence.md) still applies:
outer x/y truncation errors can partially cancel, and increasing both orders
from 16 to 24 still changes transmission substantially. This implementation
reduces cell approximation error; it does not remove outer truncation error.

## Refining the independent control

Because the candidate-to-control differences are now smaller, the independent
2D FEM reference was also refined from h=.04 to .03 micrometers at degree 8.
Its scalar unknowns increase from 43649 to 59969, and the relative static
enclosure narrows from 7.93665e-7 to 4.62166e-7. This is a fitted triangle mesh
separate from the candidate's tensor-product space.

At the new mesh, cap-48 quadrature 24 to 28 changes the coefficient arrays by
at most 2.67e-15. At outer order 16 the refined p/s T values are
0.907165907146 / 0.941480627757. Their changes from the preceding reference are
only about 8.47e-9 / 2.19e-8, much smaller than the candidate discrepancies
reported above. The improvement is therefore retained under this independent
cell refinement. This does not turn the shared Maxwell instrument into an
independent optical reference.

The refined order-24 p/s T values are 0.906794053264 / 0.941568382861. Changes
from the preceding reference are about 8.48e-9 / 2.16e-8. The new degree-4
candidate differs from this finer control by 9.25e-7 / 2.49e-6, confirming the
same conclusion at the higher optical order.

## Reproduction and validation

Run from the repository root. The separate reference export requires NGSolve;
JIT validation requires JAX x64. The optional class is importable as
`studies.nvf_polynomial_body.PolynomialBodyGapModel`.

Set `NVF_CACHE` to an existing scratch directory for generated projection files.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_convergence --orders 16 24 --meshes 2:8:1 --gap-degrees 6 --body-degree 4 --grading 8:.01 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_convergence --orders 0 --meshes 2:8:1 2:8:2 --gap-degrees 6 --body-degree 2 --grading 8:.01 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_polynomial_body --order 2 --gaps 0 17 --seams 0 --body-degree 4 --gap-degree 6 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_reference --cap 48 --h .03 --degree 8 --quadrature 28 --output "$NVF_CACHE/wavebit-gap-reference-h003-cap48.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_convergence --orders 16 24 --meshes --projection "$NVF_CACHE/wavebit-gap-reference-h003-cap48.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_polynomial_body.py tests/rcwa/test_nvf_directional_convergence.py tests/rcwa/test_nvf_interface_resolution.py tests/rcwa/test_nvf_gap_convergence.py tests/rcwa/test_nvf_gap_reference.py tests/rcwa/test_nvf_continuous_flux.py tests/rcwa/test_nvf_harmonic_coordinates.py tests/rcwa/test_nvf_partitioned_gap_energy.py tests/rcwa/test_nvf_thin_gap_energy.py tests/rcwa/test_nvf_cell_bounds.py tests/rcwa/test_nvf_cell_corrector.py tests/rcwa/test_nvf_harmonic_energy.py tests/rcwa/test_nvf_harmonic_energy_jax.py tests/rcwa/test_resonant_nvf_probe.py tests/rcwa/test_contact_jones_probe.py::test_equal_material_split_is_equivalent_to_one_rectangle
```

Raw output is in `studies/rcwa/nvf_polynomial_body/records/nvf_polynomial_body_static.log`,
`nvf_polynomial_body_powers.log`, `nvf_polynomial_body_jax.log`,
`nvf_polynomial_body_contact.log`, `nvf_polynomial_body_reference.log` and
`nvf_polynomial_body_reference_powers.log`.

Validation on 2026-09-08: **213 passed, 8 skipped, 5 xfailed** in the full
selection above. This adds ten dedicated polynomial tests, 25 shared model
parameterizations and one final-resolution contact test. The eight skips
require optional FEM dependencies; the separate FEM refinements and quadrature
check above were run in the NGSolve environment. The five strict expected
failures remain scoped to production or older models. Ruff, whitespace and
documentation-link checks pass. No production dispatch, physical tolerance or
eigenvalue derivative regularization was changed.

## Subsequent x-resolution comparison

The [polynomial-flux follow-up](nvf-polynomial-flux.md) implements and tests
higher-degree continuous x flux. At equal scalar dimension, its tested spaces
have wider static enclosures than refining the original linear flux. The new
space is therefore retained as an experimental option without changing this
model's defaults.

Increasing x subdivisions from 2 to 3 while keeping body degree 4 gives 7559
unknowns instead of 5039. It narrows the static enclosure by another 57.7% and
reduces the order-16 p/s discrepancy against the finer fitted-cell field by
55.1% / 49.3%. This costs 50% more scalar unknowns. A lower-cost body-degree-3
allocation narrows the static enclosure but does not improve the s comparison,
so static accuracy alone does not select the optical resolution. These are
research resolution choices; outer optical convergence remains unresolved.

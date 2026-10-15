# Resolving the finite gap and separating convergence errors

The continuous-flux study now resolves variation inside a finite gap with
hierarchical potential bubbles. `ContinuousFluxGapModel` defaults to
`gap_degree=4`; `gap_degree=1` reproduces the previous linear-gap model.
The independent contact derivative, real-seam derivative and material-partition
checks continue to pass. Explicit nested body refinement is also available
through `y_subdivisions`.

The new convergence instrument separates the static cell approximation from
outer Fourier truncation and compares both with independent controls. The
rectangular auxiliary field now has an independently fitted 2D FEM reference
with a relative static enclosure below 8e-7. Optical transmission still changes
at outer order 16. This work does not establish a converged 3D optical answer,
resolve the tilted fixture, or integrate the model into production NVF.

The subsequent [material-defined mesh study](nvf-interface-resolution.md)
reduces the static enclosure by about 64% at the same dimension and narrows
the order-16 s discrepancy against the fitted-cell control. Its independent
comparisons extend to outer order 20, where transmission still changes.
The tables below retain the preceding fixed-mesh measurements.

The newer [direction-separated comparison](nvf-directional-convergence.md)
accepts `--orders NX:NY` as well as square orders. It isolates the optical
effect of each Fourier direction and compares alternative allocations of the
auxiliary cell's x/y unknowns.

## An interior gap field that retains the contact limit

Write the normalized gap coordinate as s in [-1/2,1/2]. The previous gap
potential was `v(x,-1/2) + g*s*a(x)`, where g is gap divided by period. Extend it by

```
chi_gap = v(x,-1/2) + g*s*a(x) + g*sum_k beta_k(s)*b_k(x)
beta_k'(s) = P_k(2*s),       k = 1, ..., gap_degree-1
beta_k(-1/2) = beta_k(1/2) = 0
```

P_k is the Legendre polynomial. The bubbles change only the gap interior;
the potentials at its two faces still match the body. The x affine lift and
the periodic x basis are those of the
[integrated continuous-flux model](nvf-continuous-flux.md).

Divide each bubble test equation by g analytically, as for the existing ramp
enrichment. Let M and K be the unweighted x mass and stiffness matrices. The
scaled bubble block is

```
A_bb/g = D kron M + g^2*S kron K
D_kl = integral beta_k'(s)*beta_l'(s) ds = delta_kl/(2*k+1)
S_kl = integral beta_k(s)*beta_l(s) ds
```

The v-to-b coupling before row scaling is proportional to `g^2*integral(beta)`;
the a-to-b coupling is proportional to `g^3*integral(s*beta)`. The x forcing
in a scaled bubble equation is `-g*integral(beta)*integral(L*psi')`, where L is
the affine-lift derivative. The y forcing vanishes because the bubble's
normal derivative integrates to zero.

At g=0 the bubble block is positive definite and its right-hand side is zero.
Consequently b=0 there and the original first-order gap energy is retained.
There is no small-gap cutoff, numerical division by zero or prescribed optical
gradient. Ordinary and bubble DOFs retain the same residual/domain guards.

The gap potential is polynomial in s. Its products use exact Fourier moments
through degree `2*gap_degree`; NumPy/JAX share the formula. Degrees 2, 4 and 6
are independently checked against a tensor Gauss integration of the full field
and energy. Degree 4 is the default; the supported explicit range is 1 through 6.
The body y grid can be subdivided within every original graded interval,
which refines the near-gap region as well as the interior. Both refinements
preserve the fixed x topology and equal-material cancellation.

## Independent static reference

`studies/rcwa/nvf_gap_reference/nvf_gap_reference.py` builds a separate periodic triangular mesh fitted
to every material region and solves the primal and reciprocal scalar cell
problems with NGSolve. It checks material areas, both periodic identifications,
mean field, true residuals and Fourier projection quadrature. Its result is an
auxiliary-field reference, not an independent Maxwell solve. The resource-heavy
3D tilted FEM run was not restarted.

The fixture is period 700 nm, width 350 nm, gap 17 nm, seam 31 nm and materials
4/9 in background 1. The reference refinements give:

| maximum mesh size, micrometers | polynomial degree | scalar unknowns | relative static enclosure |
| ---: | ---: | ---: | ---: |
| 0.06 | 6 | 15481 | 3.22659e-6 |
| 0.04 | 8 | 43649 | 7.93665e-7 |

At the finer level the diagonal bounds are

```
1.670777582297659 <= Uxx <= 1.670777804152657
3.184544887602813 <= Uyy <= 3.184547414999149
```

These are floating-point variational bounds, not interval-arithmetic certificates.
Increasing the fine projection quadrature from 20 to 24 points changes the
cap-32 coefficient arrays by at most 1.69e-15. The coarse 16-to-20 quadrature
change was 6.14e-9 and was kept separate from mesh convergence. Closed fitted
laminates recover the analytic energy tensor and unit mean field. Incomplete
material mappings and invalid geometry are rejected.

The continuous-flux model's own primal/reciprocal solves show the improvement
from each refinement separately:

| x subdivisions | base body rows | body y subdivisions | gap degree | relative static enclosure |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 8 | 1 | 1 | 0.001954201692 |
| 2 | 8 | 1 | 4 | 0.001541216595 |
| 4 | 8 | 1 | 1 | 0.001705394515 |
| 4 | 8 | 1 | 4 | 0.001291912228 |
| 4 | 8 | 2 | 4 | 0.000771780974 |

The last model has 5183 scalar unknowns. Its cell approximation remains less
accurate than the fitted FEM reference; a small algebraic residual does not
remove that difference. Passing the seam derivative gate also does not imply
that the cell approximation is converged.

## Optical comparison at fixed and increasing outer order

`studies/rcwa/nvf_gap_convergence/nvf_gap_convergence.py` projects once at the largest requested Fourier
cap, then forms each smaller retained system from those coefficients. A direct
lower-cap calculation checks this reuse. A separate Cartesian instrument
integrates the material intervals exactly, including a zero-width gap. The
optional FEM export is accepted only when its geometry and materials match.
All three paths share the Maxwell scattering instrument, so this comparison
isolates factorization/cell effects; it is not three independent Maxwell solvers.

At outer order 16 on the same fixture:

| factorization / auxiliary field | p T | s T |
| --- | ---: | ---: |
| exact Cartesian intervals | 0.907193571579 | 0.941468107350 |
| fitted FEM field, same energy closure | 0.907165915611 | 0.941480649651 |
| continuous flux, x4, y1, linear gap | 0.907203629098 | 0.941560169727 |
| continuous flux, x4, y1, degree-4 gap | 0.907170121909 | 0.941553172975 |
| continuous flux, x4, y2, degree-4 gap | 0.907170049872 | 0.941540519833 |

The degree-4 gap reduces the p discrepancy against the independently fitted
cell projection from 3.77e-5 to 4.21e-6 at the same body mesh. The s discrepancy
is reduced more modestly and remains 5.99e-5 after the listed body refinement.
These are differences against a finite-order projected-cell control, not
certified errors in physical transmission.

Even with the fitted FEM field held fixed, outer refinement remains visible:

| outer order | p T | s T |
| ---: | ---: | ---: |
| 12 | 0.907643855516 | 0.941181620193 |
| 16 | 0.907165915611 | 0.941480649651 |

The approximately 4.78e-4 / 2.99e-4 changes exceed the remaining differences
between several of the order-16 formulations. Thus improving the static cell
alone cannot establish optical convergence. No numerical plateau or extrapolated
optical reference is accepted from this sweep.

Both incident polarizations now share one layer eigensolve. The gap polynomial
projection also contracts the x basis before multiplying by y Fourier moments,
avoiding repeated contractions without changing the field or energy. Independent
slab formulas, full gap quadrature and JIT/finite-difference checks validate
these computation changes.

## Reproduction and validation

Run from the repository root. The first commands require NumPy/SciPy; the JIT
checks additionally require JAX x64. The fitted reference uses a separate
environment with NGSolve installed.

Set `NVF_CACHE` to an existing scratch directory for generated projection files.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_continuous_flux.nvf_continuous_flux --order 2 --gaps 0 17 --seams 0 --gap-degree 4
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 4 8 12 16 --meshes 2:8 4:8 --gap-degrees 1 4
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 16 --meshes 4:8:2 --gap-degrees 4
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_reference.nvf_gap_reference --cap 32 --h .04 --degree 8 --quadrature 20 --output "$NVF_CACHE/wavebit-gap-reference.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 4 8 12 16 --meshes --projection "$NVF_CACHE/wavebit-gap-reference.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_gap_convergence.py tests/rcwa/test_nvf_gap_reference.py tests/rcwa/test_nvf_continuous_flux.py tests/rcwa/test_nvf_harmonic_coordinates.py tests/rcwa/test_nvf_partitioned_gap_energy.py tests/rcwa/test_nvf_thin_gap_energy.py tests/rcwa/test_nvf_cell_bounds.py tests/rcwa/test_nvf_cell_corrector.py tests/rcwa/test_nvf_harmonic_energy.py tests/rcwa/test_nvf_harmonic_energy_jax.py tests/rcwa/test_resonant_nvf_probe.py tests/rcwa/test_contact_jones_probe.py::test_equal_material_split_is_equivalent_to_one_rectangle
```

Raw measurements are in `studies/rcwa/nvf_gap_convergence/records/nvf_gap_convergence.log`,
`nvf_gap_body_refinement.log`, `nvf_gap_refined_powers.log`,
`nvf_gap_reference.log`, `nvf_gap_reference_powers.log` and
`nvf_gap_enriched_jax.log` in the same directory.

Validation on 2026-09-08: **136 passed, 8 skipped, 5 xfailed** in the full
selection. The eight optional FEM tests also pass in their own environment
(five existing cell tests and three new profile tests). The existing expected
failures remain scoped to the original production/older-study limitations.
No physics tolerance was relaxed. Production dispatch and eigenvalue
derivative regularization are unchanged.

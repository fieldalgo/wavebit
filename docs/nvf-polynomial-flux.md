# Testing polynomial x flux and cell resolution allocation

The x enrichment is implemented and independently checked, but it is **not a
replacement for the preceding research setting**. At equal scalar dimension,
the tested degree-2 and degree-3 flux spaces have wider static enclosures than
refining the original continuous linear flux. A more balanced allocation of
the original x mesh and polynomial y body gives a useful accuracy/cost tradeoff.
This remains research in `studies/`, not a change to production `wavebit.rcwa`.

## Implemented space

`PolynomialFluxGapModel` extends `PolynomialBodyGapModel`. Each x interval has
the existing periodic linear endpoint hats plus the interior Bernstein fluxes

```text
b_k(r) = binomial(p,k) r^k (1-r)^(p-k),  k=1,...,p-1.
```

These flux functions are continuous across artificial nodes because their
interior traces vanish. They are strictly positive inside their interval.
For every flux function f_j, including the bubbles, define

```text
rho = 1 / epsilon
A_j = integral(rho f_j dx)
F_j(x) = integral from -1/2 to x of rho f_j du
R = sum of A_j over the original nodal hats only
L(x) = sum of F_j(x) over those nodal hats / R
psi_j(x) = F_j(x)/A_j - L(x).
```

Only the original nodal hats sum to one; adding bubble integrals to R would
change the physical affine lift. Every A_j is positive for the supported
positive real materials, so no sign-changing bubble integral is used as a
denominator. Omit the same fixed nodal index as before and prepend the
constant potential. With E intervals, the potential space has E*p functions.
The integrated potential is periodic and continuous, and epsilon*psi'_j is
continuous at artificial nodes. Its physical field jump follows the material
interface inside each interval.

The code integrates material-jump suffixes exactly, with potential degree p+1
and energy moments through degree 2(p+1). Equal-material jumps cancel without
an equality predicate. Geometry, material values, grading, normalization and
the balanced cell solve remain live under JAX. Flux degrees 1 through 3 are
supported; degree 1 delegates bit-for-bit to the previous implementation.

The physical x mesh still comes from `fractions` and `per_region`. Its interval
count is distinct from `nx`, which counts enriched potential functions.
`ContinuousFluxGapModel._x_integrals` separates that assembly from the existing
y-body and gap solve. Contact scaling, the gap bubbles and the optical closure
are unchanged.

## Static comparison

The fixture is the same 700 nm period, 350 nm envelope, 17 nm gap, real seam at
31 nm and relative permittivities 4/9 in background 1. All rows below use base
body rows 8, y subdivisions 1, gap degree 6, edge fraction 1e-5 and material
grading strength/radius 8/.01. The enclosure uses independent primal and
reciprocal-material cell solves.

| Flux degree | x subdivisions | Body degree | Scalar unknowns | Relative static enclosure |
| --- | --- | --- | --- | --- |
| 1 | 2 | 4 | 5039 | 1.153565960e-4 |
| 2 | 1 | 4 | 5039 | 1.866865042e-4 |
| 1 | 3 | 2 | 4103 | 1.271253268e-4 |
| 3 | 1 | 2 | 4103 | 1.568834636e-4 |
| 1 | 4 | 2 | 5471 | 1.072995465e-4 |
| 2 | 2 | 2 | 5471 | 1.134143150e-4 |
| 1 | 3 | 3 | 5831 | 5.810270957e-5 |
| 3 | 1 | 3 | 5831 | 8.791814062e-5 |
| 1 | 3 | 4 | 7559 | 4.880540352e-5 |
| 3 | 1 | 4 | 7559 | 7.863237192e-5 |

Higher-degree flux narrows the enclosure when the mesh is held fixed, as
expected for nested conforming spaces. That does not establish efficiency:
the direct comparisons at equal dimension all favor the original flux with
more intervals. No study or production default is changed to the new space.

Keeping linear flux and using x subdivisions 3/body degree 3 narrows the
preceding 5039-unknown enclosure by **49.6%**, using **15.7% more unknowns**.
Body degree 4 at the same x resolution narrows it by **57.7%**, using **50.0%
more unknowns**. These are dimension comparisons, not runtime speedups; the
dense solve cost and measured timings need not scale linearly with dimension.
All measured primal/reciprocal residuals are below 2.3e-11.

The independent h=.03 micrometer, degree-8 fitted FEM cell from the
[body study](nvf-polynomial-body.md) provides a much tighter static enclosure.
Its full 2-by-2 interval is contained inside all ten intervals above. Static
enclosures bound effective cell energy, not optical transmission.

## Optical comparison at a fixed outer order

At outer order 16, the same layer and incidence from the preceding body study
give the following transmissions. The independent fitted-cell field uses
h=.03 micrometers, degree 8 and 59969 scalar unknowns; every row uses the same
Maxwell scattering instrument.

| Flux degree / x subdivisions / body degree | Scalar unknowns | p T | s T |
| --- | --- | --- | --- |
| 1 / 2 / 4, preceding setting | 5039 | 0.907166994307 | 0.941483923649 |
| 3 / 1 / 3 | 5831 | 0.907166549923 | 0.941483968917 |
| 1 / 3 / 3 | 5831 | 0.907166604781 | 0.941484000955 |
| 1 / 3 / 4 | 7559 | 0.907166395579 | 0.941482300435 |
| Fitted FEM auxiliary field | 59969 | 0.907165907146 | 0.941480627757 |

The 5831-unknown alternatives reduce the p discrepancy but slightly increase
the s discrepancy relative to the preceding setting. In particular, halving
the static enclosure does **not** halve the optical discrepancy. The small
difference between the two 5831-unknown candidates does not establish an
optical advantage for either flux space.

The 7559-unknown linear-flux/body-degree-4 setting reduces the p discrepancy
from 1.08716e-6 to 4.88433e-7 and the s discrepancy from 3.29589e-6 to
1.67268e-6, about **55.1% / 49.3%**, for **50.0% more scalar unknowns**.
Lossless power residuals in all three new optical runs are below 2.4e-12.
This is a higher-resolution research option, not a gain at equal cost.
It does not settle the previously observed outer-order error or cancellation.

## Reproduction and validation

The two CLI controls `--body-degree` and `--flux-degree` are independent. The
convergence driver defaults to body/flux degree 1, preserving its existing
behavior. Without `--grading`, enabling enrichment alone retains the fixed
mesh. The direct polynomial-flux CLI defaults to body/flux degree 2 and the
material-graded mesh, matching its model constructor.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_convergence --orders 0 --meshes 1:8 --gap-degrees 6 --body-degree 4 --flux-degree 2 --grading 8:.01 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_convergence --orders 16 --meshes 1:8 --gap-degrees 6 --body-degree 3 --flux-degree 3 --grading 8:.01 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_convergence --orders 16 --meshes 3:8 --gap-degrees 6 --body-degree 3 --grading 8:.01 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_convergence --orders 16 --meshes 3:8 --gap-degrees 6 --body-degree 4 --grading 8:.01 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_polynomial_flux --order 2 --gaps 0 17 --seams 0 --body-degree 2 --flux-degree 3 --gap-degree 4 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_polynomial_flux --static --seams 0 31 --rows 4 --body-degree 2 --flux-degree 3 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_nvf_polynomial_flux.py tests/test_nvf_polynomial_body.py tests/test_nvf_directional_convergence.py tests/test_nvf_interface_resolution.py tests/test_nvf_gap_convergence.py tests/test_nvf_gap_reference.py tests/test_nvf_continuous_flux.py tests/test_nvf_harmonic_coordinates.py tests/test_nvf_partitioned_gap_energy.py tests/test_nvf_thin_gap_energy.py tests/test_nvf_cell_bounds.py tests/test_nvf_cell_corrector.py tests/test_nvf_harmonic_energy.py tests/test_nvf_harmonic_energy_jax.py tests/test_resonant_nvf_probe.py tests/test_contact_jones_probe.py::test_equal_material_split_is_equivalent_to_one_rectangle
```

Raw measurements are in
[`nvf_polynomial_flux_static.log`](../studies/results/nvf_polynomial_flux_static.log),
[`nvf_polynomial_flux_powers.log`](../studies/results/nvf_polynomial_flux_powers.log)
and [`nvf_polynomial_flux_jax_static.log`](../studies/results/nvf_polynomial_flux_jax_static.log).
Projection caches written outside the repository are disposable; the commands regenerate
the same physical comparison without depending on those files.

The dedicated tests independently integrate physical flux with nested Gauss
rules, then compare every weighted/unweighted basis, field and energy pack.
They also check nested primal/dual bounds, the unit mean, redundant real
material partitions, degree-1 bit equality and live JIT gap/seam/material/width
derivatives for both supported enriched degrees and both polarizations. The
driver is checked against explicitly identical physical and mesh settings.

Validation on 2026-09-08: targeted checks and the complementary regression
selection cover **253 distinct passing tests, 8 skipped and 5 strict expected
failures**. This adds 11 dedicated tests, 25 shared model parameterizations and
four final-resolution contact configurations to the preceding body study.
The final-resolution contact checks promote `LinAlgWarning` to an error and
retain the independent p/s one-sided derivative tolerance of 1e-7 per nm,
including both the 5831- and 7559-unknown original-flux settings and the tested
degree-2/3 flux alternatives. The skipped tests need optional FEM dependencies;
the five expected failures remain scoped to production or older models.

The direct degree-3 CLI additionally matches the independent closed-laminate
energy at seams 0 and 31 nm to 9e-16 and its JIT seam derivative to 4e-19.
Ruff, whitespace, local documentation links and the JSON result logs pass.
No existing physical tolerance or eigenvalue derivative regularization is
changed.

Production dispatch, the unresolved outer Fourier sequence and tilted/curved
geometry support are outside this change. A closer comparison to a fitted
auxiliary field at one fixed outer order is not independent Maxwell validation
or evidence that the outer optical series has converged.

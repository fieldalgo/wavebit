# Separating x, y and auxiliary-field resolution

The rectangular NVF instruments now accept independent retained Fourier half
widths `(x, y)`. This identifies which direction limits the optical calculation
while keeping the physical cell and the auxiliary field fixed. The resulting
comparison also identifies a more effective allocation of cell unknowns:
with 5039 unknowns, the refined material-graded model reduces the order-16 s
discrepancy against the fitted-cell control by about 60% compared with the
preceding 5183-unknown model.

These are research instruments. Differences against the fitted static cell
are not independent 3D optical error bounds, and increasing the outer order
still changes transmission. Production NVF integration and general geometry
remain open.

The subsequent [polynomial body implementation](nvf-polynomial-body.md)
replaces additional interval subdivision with conforming interior y bubbles.
At the same 5039 unknowns it narrows the static enclosure and reduces the
order-16 s discrepancy against the fitted-cell control by about another 82%.
It is a research implementation; the measurements below retain the preceding
linear body as the control.

The subsequent [mirror-sector implementation](nvf-mirror-convergence.md)
accelerates the same finite NVF closure and extends independent x/y checks
through order 40. The s square increment falls slightly below 1e-5 there,
while its y increment remains above 1e-5 and opposes the x increment. The
directional evidence still excludes a convergence claim.

## Independent directional orders

`studies/nvf_fourier_orders.py` defines a single explicit order convention:

```
order = N       -> m in [-N,N],   n in [-N,N]
order = (X,Y)   -> m in [-X,X],   n in [-Y,Y]
number of spatial orders = (2*X+1)*(2*Y+1)
```

The existing y-major ordering is retained. The CLI spelling is `X:Y`, for
example `--orders 16:20 20:16 20`. Integers preserve the square-order behavior.
The shared energy closure, Cartesian interval comparison and p/s scattering
instrument accept the same convention. Both polarizations continue to share
one layer eigensolve.

The auxiliary coefficient export still covers a square cap of at least
`2*max(X,Y)`. It is reused without changing the cell solve. Each requested
retained space forms its own change-of-field and energy Gram matrices before
inversion; cropping the inverse of a larger closure would not give this model.
Only truncation changes: the material profile, incidence, period and thickness
remain fixed.

Eight new checks cover scalar compatibility and invalid orders, exact uniform
slab transmission for three directional sets including a zero half width,
physical axis swapping of the energy tensor, mixed-cell transmission through
the public scattering path, and a closed laminate's independence from unused
transverse orders. The tests compare physical properties and independent
controls rather than simply comparing a directional run with its own secants.
An additional refined-cell test checks both contact derivatives and treats
linear-system conditioning warnings as errors.

## What the direction-separated comparison shows

The fixture is period 700 nm, width 350 nm, gap 17 nm, seam 31 nm and materials
4/9 in background 1. Wavelength is 900 nm, thickness 300 nm, incidence angle
.2 radians, incident permittivity 1 and substrate permittivity 2.25.

Hold the independently fitted FEM auxiliary field fixed:

| x half width | y half width | p T | s T |
| ---: | ---: | ---: | ---: |
| 16 | 16 | 0.907165915611 | 0.941480649651 |
| 20 | 16 | 0.906985573128 | 0.941530315339 |
| 16 | 20 | 0.907095833081 | 0.941512327389 |
| 20 | 20 | 0.906915509832 | 0.941563020477 |

At the first corner, increasing x alone changes p by -1.8034e-4; increasing y
alone changes it by -7.0083e-5. Their mixed increment is only 1.92e-8. Thus x
accounts for most of this p change, but the y contribution is still material.
For s the corresponding increments are +4.9666e-5 and +3.1678e-5, with a mixed
increment of +1.0274e-6. Both directions remain consequential.

The Cartesian interval instrument is kept as a separate comparison. Highly
unequal sets such as (12,20) and (20,12) differ from the fitted-cell s values
by about 1.16e-4 and 1.31e-4 respectively, larger than the square order-20
formulation difference. Agreement at one square order cannot be turned into
an error certificate for other retained spaces. No Li fallback is used inside
the NVF candidate.

The next four corners demonstrate an actual cancellation of directional errors:

| x half width | y half width | fitted-cell p T | fitted-cell s T |
| ---: | ---: | ---: | ---: |
| 20 | 20 | 0.906915509832 | 0.941563020477 |
| 24 | 20 | 0.906847699373 | 0.941582575266 |
| 20 | 24 | 0.906861847734 | 0.941548779999 |
| 24 | 24 | 0.906794061748 | 0.941568404449 |

Here the s change from increasing x alone is +1.9555e-5, while increasing y
alone gives -1.4240e-5. The combined square-order change is only +5.3840e-6,
including a mixed increment of 6.97e-8. The small combined change therefore
hides two larger, opposing contributions. The p square-order change remains
-1.2145e-4. Neither polarization is certified converged by this sweep.

At square order 24, Cartesian intervals give p/s T of
0.906805437058 / 0.941572093232. The fitted-cell differences are 1.14e-5 / 3.69e-6.
All these optical paths share the same Maxwell instrument. The cap-64 fitted
cell export retains its 7.93665e-7 static enclosure. Quadrature 32 to 36 changes
its coefficients by at most 3.94e-11; the retained cap-40 part differs from the
earlier cap-40 export by 5.96e-12. These are measured quadrature/roundoff checks,
not optical error bounds.

## Better allocation within the auxiliary cell

The [preceding material-defined grading](nvf-interface-resolution.md) used
strength 8 and radius .01. Keep those settings, base body rows 8 and edge
fraction 1e-4. Refine the body y intervals independently of x and the gap
polynomial degree:

| x subdivisions | y subdivisions | gap degree | scalar unknowns | relative static enclosure |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 4 | 2591 | 0.000367876905 |
| 4 | 2 | 4 | 5183 | 0.000280984807 |
| 2 | 4 | 4 | 4895 | 0.000191572895 |
| 2 | 2 | 6 | 2735 | 0.000359813921 |
| 2 | 4 | 6 | 5039 | 0.000183416060 |

Refining y gives a greater improvement here than spending a similar number of
unknowns on x. Raising the gap degree from 4 to 6 has a smaller effect. The
5039-unknown setting narrows the enclosure by about 35% compared with the
previous 5183-unknown setting. These are explicit resolution choices for this
fixture; the model's defaults have not been changed or fitted to a transmission
target.

At outer order 16:

| auxiliary field | p T | s T | absolute s difference from fitted cell |
| --- | ---: | ---: | ---: |
| fitted FEM | 0.907165915611 | 0.941480649651 | — |
| graded x4, y2, gap degree 4 | 0.907168707460 | 0.941525111539 | 4.44619e-5 |
| graded x2, y4, gap degree 4 | 0.907167915217 | 0.941499505030 | 1.88554e-5 |
| graded x2, y4, gap degree 6 | 0.907167708863 | 0.941498509428 | 1.78598e-5 |

The final s discrepancy decreases by about 60%; the p discrepancy decreases
from 2.7918e-6 to 1.7933e-6. These improvements follow the static refinement
result, but remain differences against a same-order projected-cell control.
The fitted reference's relative static enclosure is 7.93665e-7, much narrower
than the remaining candidate enclosure.

## Keeping the contact criterion with the refined setting

The preceding table uses `edge_fraction=1e-4` for comparison with the older
positive-gap sweeps. A separate order-7, epsilon-12.25 closed-bar check finds
that this coarse contact-edge setting misses the existing 1e-7 s derivative
criterion. Reducing the physical secant steps from (1e-3,5e-4) nm to
(1e-4,5e-5) nm leaves the discrepancy near 2.1e-7; it is not just a large-step
extrapolation artifact.

Use `edge_fraction=1e-5` with x subdivisions 2, y subdivisions 4 and gap degree
6 to resolve that edge. The number of unknowns remains 5039 and the mesh setting
is fixed across zero and positive gaps. It does not depend on a gap cutoff.
The p/s contact derivative discrepancies then fall to 3.84e-9 / 2.07e-8 per nm,
below the original criterion. Closed transmissions agree within 6e-14. This
check uses independent Cartesian contact values and one-sided extrapolation;
there is no replacement backward rule. The coarser measurements are retained
as evidence of why finite-gap improvement alone is insufficient.

The finer edge also exposes very different scales in the linear system. Before
equilibration, SciPy reports reciprocal condition estimates near 3e-17 for the
contact system. The continuous-flux solve now makes a diagonal change of
coordinates before solving:

```
D_ii = 1/sqrt(abs(A_ii))
(D A D) z = D b
u = D z
```

This solves the same equations. It adds no regularization, removes no modes,
and differentiates through D in JAX. The acceptance residual is still computed
using the original `A u - b`. At gap 17 nm the primal residual decreases from
7.28e-11 to 1.44e-11, and the cap-48 coefficients change by at most 1.57e-11.
Conditioning warnings are treated as errors in the refined contact test and
the final measurement run; both succeed. The original gap-degree-1 control
uses its previous solve path.

With this contact-resolved, equilibrated setting the relative static enclosure
is 0.000182637470, about 35% smaller than the preceding 5183-unknown model.
The independent fitted-cell interval remains inside it. Positive-gap optical
measurements are recorded alongside the contact checks in
`studies/results/nvf_directional_balanced.log`.

| square outer order | final model p T | final model s T | p difference from fitted cell | s difference from fitted cell |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.907167633867 | 0.941498495500 | 1.71826e-6 | 1.78458e-5 |
| 24 | 0.906795368094 | 0.941585171198 | 1.30635e-6 | 1.67667e-5 |

The contact refinement and equilibration preserve the finite-gap improvement.
However, the remaining s cell discrepancy persists from order 16 to 24 and is
comparable with the separate x/y outer increments. Both the cell and outer
spaces still need convergence checks; refining only one cannot close the other.

## Reproduction

Run from the repository root. FEM exports require the optional NGSolve
environment. Existing square-order calls retain their previous behavior.

Set `NVF_CACHE` to an existing scratch directory for generated projection files.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_reference --cap 64 --h .04 --degree 8 --quadrature 32 --output "$NVF_CACHE/wavebit-gap-reference-cap64.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_convergence --orders 12:20 20:12 16:20 20:16 --meshes --projection "$NVF_CACHE/wavebit-gap-reference-cap64.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_convergence --orders 20:24 24:20 24 --meshes --projection "$NVF_CACHE/wavebit-gap-reference-cap64.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_convergence --orders 0 --meshes 2:8:2 2:8:4 --gap-degrees 4 6 --grading 8:.01
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_convergence --orders 16 24 --meshes 2:8:4 --gap-degrees 6 --grading 8:.01 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_nvf_directional_convergence.py tests/test_nvf_interface_resolution.py tests/test_nvf_gap_convergence.py tests/test_nvf_gap_reference.py tests/test_nvf_continuous_flux.py tests/test_nvf_harmonic_coordinates.py tests/test_nvf_partitioned_gap_energy.py tests/test_nvf_thin_gap_energy.py tests/test_nvf_cell_bounds.py tests/test_nvf_cell_corrector.py tests/test_nvf_harmonic_energy.py tests/test_nvf_harmonic_energy_jax.py tests/test_resonant_nvf_probe.py tests/test_contact_jones_probe.py::test_equal_material_split_is_equivalent_to_one_rectangle
```

Measurements are in `studies/results/nvf_directional_convergence.log`,
`nvf_directional_order24.log`, `nvf_directional_reference.log`,
`nvf_directional_cell_refinement.log` and `nvf_directional_graded_powers.log`.
Earlier order-16/20 and mesh controls remain in the preceding studies' logs.
The contact-step investigation is in `nvf_directional_contact.log`; the finer
edge before system equilibration is in `nvf_directional_contact_resolved.log`,
and the final equilibrated setting is in `nvf_directional_balanced.log`.

Validation after equilibration on 2026-09-08: **177 passed, 8 skipped,
5 xfailed** in the full selection above. The nine dedicated checks pass,
including the refined contact test with conditioning warnings promoted to
errors. Shared partition, JIT geometry/material, contact and independent
closed-laminate gates pass. The eight skips require optional FEM dependencies;
the five strict expected failures remain attached to production or older
models. Ruff, whitespace and documentation-link checks pass. No physical
tolerance, production dispatch or eigenvalue regularization was changed.

# Resolving real material interfaces with a differentiable mesh

This page records the independent-jump density used through revision
`532489b` (landed as `908f60b`), including its historical measurements and reproduction commands.
The current [separated-rectangle construction](nvf-separated-rectangles.md)
uses a signed log-jump density that merges coincident interfaces. Its finite
cell results differ from the tables below. The former CDF remains available
as `independent_graded_fractions` in the shared material-grading module for
direct comparisons; reproducing the complete historical cells requires the
earlier revision.

`studies/rcwa/nvf_interface_resolution/nvf_interface_resolution.py` adds an opt-in, material-defined mesh to
the [continuous-flux rectangular model](nvf-continuous-flux.md). It keeps the
number and connectivity of unknowns fixed while concentrating interior x nodes
near actual permittivity jumps. On the mixed rectangle below, the relative
static energy enclosure narrows by about 64% at the same number of unknowns.
The remaining order-16 s transmission difference against an independently
fitted auxiliary field decreases by about 26% at 5183 unknowns.

This is a rectangular research extension. The order-20 comparison still shows
outer Fourier truncation error. General geometry, independently converged
optical transmission and production integration remain open.

The subsequent [directional convergence study](nvf-directional-convergence.md)
varies x and y Fourier orders separately. It also finds a more effective cell
allocation: body y refinement reduces the listed order-16 s discrepancy by
about another 60%, using fewer unknowns than the preceding x4 model.

## A mesh determined by material jumps

Normalize the rectangular envelope to r in [0,1]. For its real or redundant
interior seams a_j, define

```
w_j = strength/pi * ((epsilon[j+1]-epsilon[j]) /
                      (epsilon[j+1]+epsilon[j]))^2
C(r) = r + sum_j w_j * (atan((r-a_j)/radius) - atan(-a_j/radius))
C'(r) = 1 + sum_j w_j*radius / ((r-a_j)^2 + radius^2)
```

For every fixed original fraction q_i, the new node r_i solves
`C(r_i) = q_i*C(1)`. The density is strictly positive for the supported positive
materials. Thus the map preserves the ordering and endpoints without creating
or deleting nodes. Only the nodes within the envelope move; its existing
exterior mesh remains in place. No seam is promoted to a node, and there is no
sorting or topology switch as a seam moves through the mesh.

An equal-material seam contributes exactly zero algebraically. Adding such a
seam leaves the mesh and field coefficients unchanged, including when other
real jumps are present. The implementation accumulates the contributions in
order so that adding zero terms also preserves floating-point grouping of the
nonzero terms. It does not branch on material equality. Materials remain live
in both the mesh and the underlying PDE, so a zero mesh derivative at zero
contrast does not discard the physical material derivative.

The squared contrast also makes the mesh density invariant under reciprocal
permittivity in exact arithmetic. Both primal and complementary calculations
use the same resolution settings. Their computed energies remain separate
variational upper/lower controls; neither one is an optical error bound.

This extends the earlier fixed-mesh construction: node positions may now depend
on the physical material function, but an arbitrary primitive partition cannot
change them. Continuous body flux and material-jump integration are retained.
No Cartesian Li calculation is used inside this model.

## Root solve and derivatives

Eighteen bisection steps bracket each root, followed by four ordinary Newton
steps. NumPy and JAX evaluate the same forward equation. Differentiation passes
through the Newton updates, including their material and seam dependence;
there is no stopped geometry gradient or custom backward correction. The
root residual must be below 2e-12 and nodes must remain strictly ordered.
An invalid NumPy mesh raises; an invalid traced update is rejected through
the existing nonfinite-result guard.

`grading_strength=8` and `grading_radius=.01` are explicit study settings.
Radius is a fraction of envelope width, so here it is 3.5 nm. They were chosen
using the static enclosure, without fitting an optical target; they are not
claimed optimal across structures. The supported setting ranges are strength
0 through 32 and radius .005 through .1 for the original global flux basis.
The later [localized basis](nvf-local-flux.md) supports radii down to .0001
with the same trial space and unchanged residual/physical criteria. It
addresses the conditioning encountered when further resolving mixed-material
contact; coarse settings are not certified by that finer result.
Strength zero reproduces the original
fixed-mesh model exactly. The original `ContinuousFluxGapModel` remains
available unchanged as a control.

The independent mesh check uses SciPy's bracketed Brent solver. JIT mesh
Jacobians agree with secants of those independently solved roots, including
zero contrast. The full model also passes the shared independent closed-bar
static and optical checks, both physical contact derivatives, moving real-seam
derivatives, translated partition invariance, live material derivatives and
invalid-input checks. Multiple real jumps with redundant inserted partitions
produce bit-identical coefficients at gaps 0, 17 and 400 nm.

## Same-dimensional static comparison

The fixture has period 700 nm, width 350 nm, gap 17 nm, seam 31 nm and materials
4/9 in background 1. All listed continuous-flux models use base body rows 8,
body y subdivisions 2, gap degree 4 and edge fraction 1e-4.

| x subdivisions | scalar unknowns | fixed mesh relative enclosure | material-graded relative enclosure |
| ---: | ---: | ---: | ---: |
| 2 | 2591 | 0.001024637861 | 0.000367876905 |
| 4 | 5183 | 0.000771780974 | 0.000280984807 |

The reductions are about 64% in both cases. At x2 the new diagonal bounds are

```
1.670761658413817 <= Uxx <= 1.670793592106591
3.183817928828364 <= Uyy <= 3.184989181912976
```

The [independently fitted FEM cell](nvf-gap-convergence.md#independent-static-reference)
has a much narrower enclosure, relative width 7.93665e-7. Its diagonal interval
lies inside both of these candidate intervals. The improved mesh therefore
reduces auxiliary-field uncertainty but has not eliminated it. The graded
primal/reciprocal residuals are below 1.5e-12 in the recorded runs.

## Optical comparison and the remaining outer error

Use wavelength 900 nm, thickness 300 nm, incidence angle .2 radians, incident
permittivity 1 and substrate permittivity 2.25. At outer order 16:

| auxiliary field / factorization | scalar cell unknowns | p T | s T |
| --- | ---: | ---: | ---: |
| fitted FEM field, energy closure | 43649 | 0.907165915611 | 0.941480649651 |
| fixed mesh, x2 | 2591 | 0.907171160845 | 0.941544331905 |
| fixed mesh, x4 | 5183 | 0.907170049872 | 0.941540519833 |
| material-graded mesh, x2 | 2591 | 0.907169465216 | 0.941527092395 |
| material-graded mesh, x4 | 5183 | 0.907168707460 | 0.941525111539 |
| independent Cartesian intervals | — | 0.907193571579 | 0.941468107350 |

At x4, the s discrepancy against the fitted-cell projection decreases from
5.9870e-5 to 4.4462e-5, about 26%. The p discrepancy also decreases, from
4.1343e-6 to 2.7918e-6. Even the graded x2 model is closer in both polarizations
than the fixed x4 model, using half as many cell unknowns. These are differences
against a same-order auxiliary-field control, not certified transmission errors.
All paths use the same Maxwell scattering instrument.

The independent controls were also extended to outer order 20:

| outer order | fitted-cell p T | fitted-cell s T | Cartesian p T | Cartesian s T |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.907165915611 | 0.941480649651 | 0.907193571579 | 0.941468107350 |
| 20 | 0.906915509832 | 0.941563020477 | 0.906934177474 | 0.941561099135 |

Their order-20 differences are 1.87e-5 in p and 1.92e-6 in s. However, the
fitted-cell transmission still changes by 2.50e-4 / 8.24e-5 from order 16 to 20.
Agreement between formulations at one order does not establish convergence.
The cap-40 FEM projection changes by at most 2.98e-15 when quadrature increases
from 24 to 28 points, separating this projection check from outer truncation.
No extrapolated optical reference is accepted from these measurements.

## Reproduction and validation

Run from the repository root; the optional FEM export requires NGSolve.

Set `NVF_CACHE` to an existing scratch directory for generated projection files.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_interface_resolution --order 2 --gaps 0 17 --seams 0 31 --grading-strength 8 --grading-radius .01
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_convergence --orders 16 --meshes 2:8:2 4:8:2 --gap-degrees 4 --grading 8:.01
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_convergence --orders 16 --meshes 2:8:2 --gap-degrees 4
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_reference --cap 40 --h .04 --degree 8 --quadrature 24 --output "$NVF_CACHE/wavebit-gap-reference-cap40.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_gap_convergence --orders 20 --meshes --projection "$NVF_CACHE/wavebit-gap-reference-cap40.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_interface_resolution.py tests/rcwa/test_nvf_gap_convergence.py tests/rcwa/test_nvf_gap_reference.py tests/rcwa/test_nvf_continuous_flux.py tests/rcwa/test_nvf_harmonic_coordinates.py tests/rcwa/test_nvf_partitioned_gap_energy.py tests/rcwa/test_nvf_thin_gap_energy.py tests/rcwa/test_nvf_cell_bounds.py tests/rcwa/test_nvf_cell_corrector.py tests/rcwa/test_nvf_harmonic_energy.py tests/rcwa/test_nvf_harmonic_energy_jax.py tests/rcwa/test_resonant_nvf_probe.py tests/rcwa/test_contact_jones_probe.py::test_equal_material_split_is_equivalent_to_one_rectangle
```

Raw measurements are in `studies/rcwa/nvf_interface_resolution/records/nvf_interface_resolution.log`,
`nvf_interface_resolution_jax.log`, `nvf_gap_reference_cap40.log` and
`nvf_gap_order20.log`. Previous fixed-mesh and fitted-cell order-16 measurements
are retained in the [preceding study's logs](nvf-gap-convergence.md#reproduction-and-validation).

Validation on 2026-09-08: **168 passed, 8 skipped, 5 xfailed** in the combined
selection. This includes seven new dedicated tests and 25 new shared
parameterizations for the graded model. The eight skips require the optional
FEM environment; those tests passed in the preceding reference work. The five
strict expected failures remain attached to production or older models. No
existing physical tolerance, production dispatch or eigenvalue derivative
regularization was changed.

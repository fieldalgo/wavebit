# Continuous body flux through a moving material seam

`studies/nvf_continuous_flux.py` resolves the tested positive-gap seam
derivative discrepancy left by the local harmonic-coordinate model. Its
periodic potential is obtained by integrating continuous flux hats against
reciprocal permittivity. This preserves the physical field jump inside an
element and keeps the body displacement continuous at artificial mesh nodes.
The existing partition, contact, closed-laminate and material derivative gates
continue to pass. The preceding models and their strict failure tests remain
available as controls.

This is still a rectangular research model. Passing these interface checks
does not establish converged optical transmission for general structures or
integrate the method into production NVF. Tilted/curved geometry, independent
optical convergence and production backend integration remain open.

The subsequent [finite-gap convergence work](nvf-gap-convergence.md) adds
interior gap bubbles and nested body refinement. The current default is
`gap_degree=4`; the initial measurements below used `gap_degree=1`.
Both share the continuous x flux described here. The independent static FEM
reference and optical-order comparisons are recorded in the follow-up.

An additional [material-defined mesh extension](nvf-interface-resolution.md)
concentrates the same number of x nodes around real material jumps while
retaining partition invariance and live geometry/material derivatives. It is
an opt-in subclass; the original mesh remains the control used below.

The [directional follow-up](nvf-directional-convergence.md) additionally
equilibrates the enriched cell's linear system by a live diagonal change of
coordinates. It solves the same equations and checks their original residual,
allowing the refined contact-edge setting to pass without conditioning warnings.

An opt-in [polynomial body model](nvf-polynomial-body.md) now enriches the
periodic y potential with interior bubbles while preserving its traces at the
gap. Body degree 1 reproduces this model exactly; degrees 2 through 4 have
independent integral and interface checks. The x flux construction below is
shared by both implementations.

## Integrating a continuous flux space

Let f_j be the ordinary periodic P1 hats on the envelope mesh, and define

```
rho(x) = 1/epsilon(x)
A_j = integral_period rho(x)*f_j(x) dx
R = sum_j A_j = integral_period rho(x) dx
F_j(x) = integral_{-1/2}^x rho(u)*f_j(u) du
ell(x) = sum_j F_j(x)/R
psi_j(x) = F_j(x)/A_j - ell(x)
```

Each psi_j is periodic. The relation `sum_j A_j*psi_j = 0` permits omitting one
hat at a fixed topology index; an additional constant basis function spans
the gauge. The x trial space has the same scalar dimension as the preceding
model. The mesh and omitted index do not depend on the number or equality of
material pieces. All materials must be positive real numbers.

Use the affine lift ell for the x potential and y for the y potential, with
periodic corrections expanded in the psi basis. Their x derivatives satisfy

```
ell' = rho/R
psi_j' = rho*(f_j/A_j - 1/R)
epsilon*ell' = 1/R
epsilon*psi_j' = f_j/A_j - 1/R
```

The last two expressions are continuous in x inside the material body,
including at both real material seams and artificial element boundaries.
The preceding local harmonic basis had constant flux within each element
but permitted a flux jump between elements. Removing that artificial jump
removes the tested seam-crossing sensitivity defect without moving the mesh
or changing the reported gradient independently of the forward solve.

The body/gap potential construction, analytically scaled gap test equation
and mean-field energy closure are inherited from the
[conforming-gap study](nvf-thin-gap-energy.md). As in the
[local harmonic-coordinate model](nvf-harmonic-coordinates.md), the actual
affine lift must be included in both the body/gap forcing and projected
energy. The same x basis is extended into the gap; its finite approximation
error there is still subject to refinement.

## Exact moments and material cancellation

Within an element and a material piece, F_j is quadratic in physical local x.
Basis products have degree at most four. A new Fourier moment helper uses
centered integration by parts away from zero frequency and a convergent
series near zero. Zero-width intersections remain differentiable under JAX.
An independent 80-point Gauss rule checks degrees zero through four,
including negative/near-zero frequencies and zero-width intervals. Their
JIT endpoint derivatives are checked separately.

Element totals and cumulative integrals are assembled from reciprocal material
jumps. Weighted basis and lift polynomials are then integrated by their signed
changes on suffix intervals. Equal-material seams contribute exactly zero
without an equality predicate or removal of independent material parameters.
The merged/split projected arrays, including redundant seams within a
nonuniform profile, remain bit-identical in the tested NumPy calls. Independent
48-point piecewise quadrature checks the complete weighted mass, stiffness,
load, derivative and lift packs at absolute tolerance 2e-12.

Shared shape tests cover translations through periodic boundaries and live
envelope, gap, material and seam parameters under JIT. Invalid ordinary
profiles raise errors; invalid traced updates return nonfinite data. The
original study's residual guard remains active. No production solver,
Cartesian Li dispatch or eigenvalue regularization was changed.

## Interface measurements

For width 350 nm, period 700 nm, gap 17 nm, seam zero, materials 4/9, outer
order 2, s polarization, `rows=4` and `edge_fraction=1e-4`, the seam secants
at step 1e-4 nm are:

| model | left derivative, /nm | right derivative, /nm | absolute difference |
| --- | ---: | ---: | ---: |
| local harmonic coordinate | -0.001382052641 | -0.001383810998 | 1.75836e-6 |
| integrated continuous flux | -0.001387386046 | -0.001387388302 | 2.25597e-9 |

The existing 1e-7 continuity gate now passes for the new model. It remains a
strict expected failure for the old local model. New p/s checks also compare
the JIT seam derivative with both one-sided NumPy secants at the mesh node.
The default order-3 CLI log independently shows the discrepancy reduced from
about 1.72e-6 to 8.13e-9. These finite-difference differences include ordinary
curvature and floating-point error; they are not claimed to be identically zero.

Closed mixed-bar static energy and seam/material/width derivatives retain
their independent analytic agreement, including unfitted interfaces. Closed
constitutive matrices and p/s optical seam/material derivatives pass the
same independent Cartesian controls. At outer order 7, uniform material
12.25, width 350 nm, `rows=8` and `edge_fraction=1e-5`, contact results are:

| polarization | closed T | JIT dT/dgap, /nm |
| --- | ---: | ---: |
| p | 0.626061531896529 | 0.001891546100057 |
| s | 0.878242289574428 | -0.000778519822153 |

Both pass the preceding 1e-7 absolute derivative gate against independently
extrapolated Cartesian positive-gap secants. The artificial-seam derivative
vanishes and both material derivatives remain live. Cartesian Li is only a
test instrument, not a call made by this model.

## Cell refinement remains separate from optical convergence

For a real seam at 31 nm with materials 4/9, gap 17 nm and outer order 2:

| x subdivisions | scalar unknowns | static Uxx | static Uyy | p T | s T |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 467 | 1.671011843868 | 3.187630224161 | 0.939108136894 | 0.906437006083 |
| 2 | 935 | 1.671001225919 | 3.186369469373 | 0.939108105679 | 0.906236254717 |
| 4 | 1871 | 1.670998589002 | 3.186197687187 | 0.939107991586 | 0.906224852820 |

The integrated spaces are nested under this x refinement for fixed material.
The static primal energies decrease as expected. Their changes, and the
remaining change in transmission, require further convergence work. A small
linear residual (below 2e-13 here), unit mean B, Hermiticity and lossless energy
conservation do not provide an optical error bound. The tilted reference and
outer Fourier-order uncertainties described in the earlier studies remain.

## Reproduction and validation

Run from the repository root with NumPy, SciPy and JAX installed:

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_continuous_flux --gap-degree 1
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_continuous_flux --static --gap-degree 1
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_continuous_flux --order 2 --gaps 17 --seams 0 --gap-degree 1
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_continuous_flux --order 7 --gaps 0 --seams 31 --values 12.25 12.25 --rows 8 --edge-fraction 0.00001 --gap-degree 1
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_nvf_continuous_flux.py tests/test_nvf_harmonic_coordinates.py tests/test_nvf_partitioned_gap_energy.py tests/test_nvf_thin_gap_energy.py tests/test_nvf_cell_bounds.py tests/test_nvf_cell_corrector.py tests/test_nvf_harmonic_energy.py tests/test_nvf_harmonic_energy_jax.py tests/test_resonant_nvf_probe.py tests/test_contact_jones_probe.py::test_equal_material_split_is_equivalent_to_one_rectangle
```

`ContinuousFluxGapModel` inherits the scalar profile and translated horizontal
shape-list API. The CLI shares the local harmonic study's reporting format,
enables JAX x64, and logs NumPy/JAX values, seam secants and live derivatives.
Measured optical, static, refinement and contact data are in
`studies/results/nvf_continuous_flux.log`.

Initial validation on 2026-09-08: **122 passed, 5 skipped, 5 xfailed** in the full
selection above. The new model passes 23 shared parameterized checks and five
dedicated checks. All preceding strict expected failures remain scoped to
their original models; none of the existing physics or partition tolerances
was relaxed. The skips are optional NGSolve controls.

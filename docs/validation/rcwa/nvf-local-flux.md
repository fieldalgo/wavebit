# Local coordinates for the integrated-flux cell space

The optional `flux_basis='local'` in `ContinuousFluxGapModel` and its
polynomial-body descendants replaces globally supported correction functions
with localized functions spanning the same finite space. It enables sharper
material grading without the ill-conditioned solves encountered by the
original basis. Defaults and production NVF remain unchanged.

## The same space before integration

Let f_j be the periodic continuous linear flux hats and define

```text
F_j(x) = integral from the periodic cut to x of rho f_j
A_j = integral over the period of rho f_j
H(x) = sum_j F_j(x), R = sum_j A_j.
```

The existing potential functions are `F_j/A_j - H/R`, plus a constant, with
one redundant function omitted. In the new coordinates, use adjacent cyclic
differences `F_(j+1)/A_(j+1) - F_j/A_j`. The common harmonic lift cancels.
The two functions crossing the periodic integration cut have constant tails.
If c is the first element's contribution to `A_0`, divided by `A_0`, subtract
`1-c` from the first difference and `c` from the last. The resulting
functions are localized across the periodic boundary as well.

Their sum is the constant -1. Keeping the separate constant and omitting one
fixed pivot therefore retains the original span. The local model omits an
exterior function, whose support is unaffected by interior material grading.
Omitting a sharply concentrated interior function instead makes the constant
nearly redundant on stronger, finer material profiles. The exterior pivot is
a fixed topology index, not a data-dependent choice. Material and
geometry dependence, including the constants at the periodic cut, remains
live in NumPy and JAX. No material equality predicate, node insertion,
regularization or backward correction is introduced.

Differences and constant removal happen in the local polynomial coefficients
**before** forming mass/energy products and Fourier integrals. Transforming
already integrated global mass matrices would subtract large nearly equal
entries to recover the tiny support energies. Likewise, merely taking
adjacent differences without removing the periodic constant tails still
triggered conditioning warnings on the finer cell. Both steps are necessary.

The local path also contracts the polynomial factors directly into each
Fourier matrix. It does not allocate an element-by-basis-by-basis quadratic
tensor. Weighted material differences use a polarized quadratic identity
before integration, so a small change is not recovered by subtracting two
completed Fourier matrices. At zero Fourier frequency, three positive Gauss
weights integrate the degree-four products exactly. This reduces cancellation
in the mean matrices that determine the cell solve. Zero-width intersections
retain their ordinary geometry derivatives.

The same diagonal balancing, original-equation residual limit 1e-8, NVF
Hermiticity and power/tail checks remain in force. The local basis extends
the permitted grading-radius range to [.0001, .1]; the original global basis
keeps [.005, .1]. Radius is a fraction of envelope width. A permitted radius
is not a convergence guarantee for every choice of mesh and material.
Degree-2/3 polynomial **x flux** retains the global basis; unsupported local
requests raise. Degree-1 x flux uses the same localized implementation.

## Measurements and physical checks

At the same mesh, field and energy Fourier coefficients agree with the
global representation within 3e-11 at gaps 0, 17 and 400 nm, including two
real seams. Adding redundant material partitions produces bit-identical
coefficients. Independent piecewise Gauss integration checks every weighted
and unweighted mass, stiffness, load, derivative and lift integral for both
bases. Root values and JIT mesh derivatives at radii .01, .001 and .0001 are
checked against an independent bracketed root solver.

For materials 4/9 at seam 31 nm in a 350 nm envelope, a trial extension of
the **global** basis to radius .001 issued a SciPy conditioning warning
(`rcond` about 2e-16). Radius .0001 reached about 3e-20 and failed the
downstream propagation check. That unsupported range is rejected by the
global constructor. The local basis computes the recorded finer sequence
with conditioning warnings treated as errors. No propagation tolerance or
slice count was changed to conceal the upstream failure.

With base rows 8, body degree 4, x subdivisions 3, gap degree 6 and edge
fraction 1e-5, decreasing the radius from .01 to .0001 lowers the order-2
mixed-material contact slope discrepancy from 3.79e-7 / 2.61e-6 to
4.96e-9 / 2.92e-8 per nm for p/s. The [pose study](nvf-posed-gap.md) records
the full sequence at a nonzero local incident azimuth. These are finite-order
differences from an independent Cartesian factorization, not certified
optical errors or a Cartesian fallback.

The smaller base-rows-4/body-degree-1 cell at the same x resolution also
passes both JIT contact derivatives at two azimuths under the unchanged
1e-7 per nm criterion. A two-seam profile requires x subdivisions 6;
subdivisions 3 are insufficient despite the sharp radius. Its finer cell
passes nine angle/order contact controls. At positive gap the JIT gap,
real-seam and both material derivatives use two modal Richardson references
under the existing `atol=2e-9, rtol=3e-6` criterion. These controls qualify the new coordinate
representation and the stated contact fixture, not general geometry or
positive-gap cell convergence.

The sharp-mesh seam comparison needs a resolved finite-difference reference.
For seam 0 nm, the p JIT derivative is -0.0007312648345504 per nm. Raw modal
secants at steps 1e-4 and 5e-5 nm are -0.0007312631361556 and
-0.0007312602556819: reducing the step amplifies value roundoff. Independent
Richardson pairs from steps .001/.0005 and .0005/.00025 nm give
-0.0007312646850277 and -0.0007312648965992. Both agree with each other under
`atol=1e-9, rtol=1e-6` before comparison with JIT under the original criterion.
The test also retains both one-sided mesh-node controls, and exercises zero
material contrast. `nvf_local_flux_secant_resolution.log` records all steps
and parameters; no derivative or physical acceptance tolerance was relaxed.

The final fixed-exterior-pivot model can have a different roundoff pattern
despite the same physical space. `nvf_derivative_reference.py` therefore
checks agreement of the two references without consulting JIT, enlarging the
initial window by at most three doublings if necessary. It fails explicitly
if no window meets the same reference criterion. Initial largest steps are
.01 nm for gap and .001 for seam/material; every bounded window is within
the tested physical domain. The earlier fixed-window numbers above are
retained as evidence of reference resolution, not hard-coded targets.

The final `nvf_contact_sweep.py` uses x subdivisions 6 for all three profiles,
azimuths 0/.37/1.1 and outer orders 2/4/7. All **27** combinations pass the
unchanged 1e-7 per nm gate. Their largest p/s discrepancies are:

| profile | seams (nm) | permittivities | largest absolute discrepancy / nm |
| --- | --- | --- | ---: |
| mixed | 31 | 4, 9 | 9.96e-9 |
| stronger contrast | -73 | 2.25, 12.25 | 9.52e-9 |
| multiple seams | -73, 31 | 4, 9, 6.25 | 2.04e-8 |

Every outer order uses the same maximal cell projection, and conditioning
warnings are treated as failures. The coarse x3 sweep, including its failed
multiple-seam cases, remains in `nvf_contact_sweep.log`; the passing final
sequence is `nvf_contact_sweep_refined.log`. This finite set does not certify
arbitrary contrast, geometry or optical truncation.

## Memory, cost and positive-gap limits

A separate-process measurement of the x integral assembly uses 216 basis
functions, nine Fourier modes, materials 4/9/6.25, seams -73/31 nm, width
350 nm and the local radius .0001. The former expanded local integrals took
6.55 seconds with 3.77 GB peak process RSS. The factored, polarized assembly
with the exact mean rule takes .75 seconds with .37 GB peak RSS. These are
single measurements of this component on this host, about 8.7 times faster
and 90% less peak memory; they exclude the cell solve, closure and propagation.
All returned integral matrices agree with the saved earlier assembly under
`atol=2e-12, rtol=1e-12`; normalized maximum differences are below 3.5e-15.
An intermediate .63-second implementation is retained in the timing log as
development evidence, not the final implementation.

Sharper grading is not universally more accurate. At gap 17 nm, order 8,
base rows 8, body degree 4, x subdivisions 3 and gap degree 6, compare the
same-order independently fitted FEM auxiliary field:

| radius / width | relative static enclosure | p T difference from fitted cell | s T difference from fitted cell |
| ---: | ---: | ---: | ---: |
| .01 | 4.881e-5 | 4.198e-7 | 1.713e-6 |
| .001 | 2.126e-5 | 2.963e-7 | 2.842e-6 |
| .0001 | 2.664e-5 | 3.676e-7 | 4.496e-6 |

The static enclosure improves while the s optical discrepancy grows. The
fitted control is itself a finite cell projection, and the comparison is at
one outer order. These results do not establish an optical error bound or
justify replacing every positive-gap setting with the contact setting.
The default grading remains unchanged. The convergence CLI now accepts
`--flux-basis local` so cell, grading and outer-order effects can be separated.

Use `PolynomialBodyGapModel(..., flux_basis='local', grading_radius=.0001)`
or the pose CLI's `--flux-basis local --grading-radius .0001`. Tests are in
`tests/rcwa/test_nvf_local_flux.py`, `test_nvf_continuous_flux.py` and
`test_nvf_interface_resolution.py`. Raw refinement records are in
`studies/rcwa/nvf_local_flux/records/nvf_posed_contact_local_flux.log`; the earlier failed-global
experiment's one completed record is retained in
`nvf_posed_contact_grading.log` as diagnostic evidence.

Additional records are `nvf_local_flux_integration_timing.log` and
`nvf_local_flux_positive_gap.log`; `nvf_local_flux_positive_gap_verified.log`
repeats the latter with the final mean integration and exterior pivot.
Reproduce the positive-gap comparison with
the fitted projection generated in [the independent cell study](nvf-gap-convergence.md):

Set `NVF_CACHE` to an existing scratch directory for generated projection files.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 8 --meshes 3:8 --gap-degrees 6 --body-degree 4 --edge-fraction 1e-5 --flux-basis local --grading 8:.0001 --projection "$NVF_CACHE/wavebit-gap-reference-h003-cap48.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_contact_sweep.nvf_contact_sweep
```

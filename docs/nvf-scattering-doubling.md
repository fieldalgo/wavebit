# Eigenvector-free propagation for the NVF research model

`studies/nvf_scattering_doubling.py` implements a short-slice Maxwell transfer
followed by scattering-matrix doubling. On the recorded rotated-rectangle
material-derivative fixture, it removes the approximately 6 ppm s-polarized
bias of the existing regularized eigenvector derivative: the new difference
from the first independent NumPy secant is approximately 0.028 ppm. The
original two-step derivative criterion passes without relaxing its tolerance.

The implementation is usable directly with constitutive matrices, or through
`ScatteringDoublingGapModel`, an explicit alternative to the polynomial-body
research model. The cell equations, live material grading and NVF Gram closure
are unchanged. Production NVF dispatch and its existing eigensolver are not
changed. This fixes a propagation-derivative issue in the new path; it does
not repair the old Fourier auxiliary field's separate s contact error.

## Propagation without modal derivatives

In the existing scaled electric/magnetic coordinates, write the homogeneous
layer generator using its first-order Maxwell factors L and R:

```text
F(tau) = exp(-tau [[0, L], [R, 0]])
tau = (2 pi / wavelength) thickness / 2^doublings
P = tau^2 L R
Q = tau^2 R L
```

For a short slice, evaluate the even and odd series

```text
C(P) = sum P^k / (2k)!
S(P) = sum P^k / (2k+1)!
F00 = C(P)       F01 = -tau L S(Q)
F10 = -tau R S(P)  F11 = C(Q).
```

Both LR and RL use expanded expressions to avoid subtracting terms of order
k^4 to recover a result of order k^2. Tests compare these expressions against
the first-order products with complex Hermitian cross blocks. The method
forms no matrix square root, selects no modal branch and divides by no
eigenvalue or eigenvalue separation. A zero layer propagation eigenvalue is
therefore evaluated by the same entire series.

The reference admittance is `Y = i [[0,-I],[I,0]]`, with `Y^2=I`. Writing
`E=a+b`, `H=Y(a-b)` gives reference ports with flux proportional to
`|a|^2-|b|^2`. Convert only the short transfer to scattering coordinates, then
cascade it with itself `doublings` times using linear solves. Growing
evanescent transfer factors are never formed across the full layer thickness.
Finally match these reference ports to the physical exterior p/s waves and
evaluate their outgoing fluxes. The exterior normalization matches the
existing research instrument.

NumPy uses SciPy solves; JAX differentiates ordinary matrix products and
solves in float64. There is no custom eigenvector derivative, frozen material
field or stopped geometry derivative in this path.

## Numerical checks and limits

For K retained Taylor terms, let alpha be the maximum infinity norm of P and
Q. The tail estimate uses the first omitted even term and a geometric bound
with ratio `alpha / ((2K+3)(2K+4))`, also covering the odd blocks through the
factor `1 + |tau| max(||L||,||R||)`.

The ratio must be below one and the estimated local tail times the slice
count must be below 1e-13. An unresolved slice raises in NumPy and yields
NaNs under JIT; increasing the number of doublings or terms is explicit.
This is a conservative local-series check, not a proven global error bound
through arbitrarily ill-conditioned boundary matching. Separate slice
refinement and independent optical comparisons are still necessary.

The current lossless instrument also requires finite T/R and power imbalance
below 1e-9. That guard cannot certify optical truncation accuracy. Absorbing
materials are outside this lossless power contract. The fixture retains its
fixed period, wavelength and exterior media; thickness and live polar/azimuth
incidence angles can be supplied to `doubling_powers` or the model transmission
methods. The [pose extension](nvf-posed-gap.md) documents the polarization
frame through normal incidence and the remaining exterior-graze exclusion.
`doubling_response` also returns complex outgoing amplitudes. Orders may be
scalar or directional in the matrix-level function; the inherited cell
adapter currently takes a scalar order.

The independent uniform-slab control extends to thickness 30000 nm and a
directional order `(2,3)`. A later [mirror-sector integration](nvf-mirror-doubling.md)
supports both NumPy and JAX, compares values through order 24 and measures
lower forward/derivative cost within the doubling method. The shared kernel
also uses the homogeneous layer's z symmetry to avoid duplicate scattering
blocks. This has not been performance-qualified at the modal instrument's
order 40, generalized to multilayer production geometry or used to declare
outer optical convergence.

## Derivative measurements and physical controls

The original material-bias fixture has order 2, Fourier auxiliary order 7,
rectangle width 440 nm, height 270 nm, angle .31 radians, center `(17,-21)` nm
and relative permittivity 6.25. Reusing that exact auxiliary model isolates the
change in propagation:

| quantity | new derivative | independent modal secant, step 1e-5 |
| --- | ---: | ---: |
| dT_p / d epsilon | 0.00846134966851 | 0.00846134614041 |
| dT_s / d epsilon | 0.00619272391621 | 0.00619272374491 |

The s relative difference is about 0.028 ppm at this step; roundoff in the
secant remains, so this is not an absolute derivative-error certificate.
Width, height, angle and material derivatives pass the original
`atol=2e-9, rtol=3e-6` comparison at both original step sizes. The old
regularized-path expected failure remains a control for that unchanged path.

Further tests exercise the current polynomial-body NVF adapter at order 2,
rows 4, body degree 4, gap degree 6 and edge fraction 1e-5:

- JIT gap, moving real seam and both material derivatives against independent
  modal NumPy secants, including equal materials and zero contrast.
- Both single-material JIT contact derivatives against a separately
  factorized Cartesian one-sided contact limit, retaining the existing
  1e-7 per nm criterion. Mixed-material contact requires additional cell
  refinement, as recorded in the [localized flux study](nvf-local-flux.md).
- Translated split/merged rectangle equality and zero translation derivatives.
- Uniform and repeated-mode material derivatives, including an exactly
  grazing layer, against the independent slab formula.

For the contact control with permittivity 12.25, the new JIT gap derivatives
are 0.0009240734238 and 0.0233077786485 per nm for p and s. The separately
factorized, two-step extrapolated reference gives 0.0009240733023 and
0.0233077783351. Absolute discrepancies of 1.21e-10 and 3.13e-10 per nm are
below the unchanged 1e-7 criterion at this stated low-order fixture.

At exact layer graze, the field value is checked against an independent
Jordan-limit 2-by-2 transfer. For its material derivative the ordinary Airy
secant at step 1e-5 has a visible O(h^2) error: the p values are
2.1351583380, 2.1351547468 and 2.1351538493 at steps 1e-5, 5e-6 and 2.5e-6.
Richardson extrapolation from two successive pairs agrees with the JIT value
2.1351535499 under the unchanged tolerance. The test checks agreement of the
two extrapolated references before comparing the derivative.

Additional controls compare mixed-material p/s powers against the previous
modal solve at gaps 0, 17 and 400 nm while independently varying the number
of slices. These controls isolate propagation; they do not replace the cell
or outer Fourier convergence studies.

## Use and reproduction

```python
from studies.nvf_scattering_doubling import ScatteringDoublingGapModel

model = ScatteringDoublingGapModel(
    2, rows=4, body_degree=4, gap_degree=6, edge_fraction=1e-5,
)
result = model.transmission(
    17., dict(seams=[31.], values=[4., 9.]), polarization='both',
)
# result['T'] and result['R'] are vectors in p, s order.
# Pass xp=jax.numpy for live JIT derivatives, with JAX x64 enabled.
```

The inherited `shape_transmission` interface supports the same ordered,
translated horizontal rectangle tilings. It does not become a general CSG
adapter. `polarization='p'` or `'s'` returns scalars, matching the older study
interface. Direct use of `doubling_powers(order, slots, xp)` accepts any
compatible finite constitutive matrices under the lossless contract.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_scattering_doubling --jax
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_scattering_doubling --mirror --jax
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_scattering_doubling --gap 0 --seams --values 12.25 --jax
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_nvf_scattering_doubling.py
```

The full-matrix CLI reports cell residual, Hermiticity, the local Taylor-tail
estimate, power balance and optionally the JIT gap/seam/material Jacobian.
With `--mirror`, the same internal guards apply, while the returned diagnostics
contain the cell information and power balance; the high-order mirror CLI's
`--progress` output also exposes the per-sector Taylor bounds. Raw outputs
are in `studies/results/nvf_scattering_doubling_material.log` and
`nvf_scattering_doubling_gap.log`.

Validation on 2026-09-08: **18 passed**. No production failure is relabeled as
passing on the strength of a separate research implementation.

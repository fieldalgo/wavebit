# Conforming NVF with independent material loss

Written: 2026-09-09

Targets:
- `src/wavebit/_core/conforming/continuous_flux.py`
- `src/wavebit/_core/conforming/polynomial_body.py`
- `src/wavebit/_core/conforming/material_grading.py`
- `src/wavebit/_core/conforming/__init__.py`
- `tests/test_nvf_complex_variational.py`
- `tests/test_conforming_complex_material.py`

`ConformingNVF` now permits a different passive complex permittivity in
every rectangle and background. A lossy inclusion in a lossless background
is supported. Real parts must remain positive, and imaginary parts must be
nonnegative. Geometry restrictions are unchanged; see the
[manual](guide/conforming-nvf.md) for inputs and JAX replay.

The [common-factor construction](nvf-common-loss.md) remains an exact
control, but a shared loss ratio is no longer required. The auxiliary cell
now solves complex weak equations and projects complex fields, rather than
discarding imaginary coefficients or switching to a Li factorization.

## Complex cell and conjugate test functions

The public adapter normalizes each material by the layer background. For
physical materials in the open right half-plane with nonnegative imaginary
parts, these ratios also have positive real parts, although their imaginary
parts can have either sign. The physical background factor is restored in
both the energy matrix and longitudinal material matrix before closure.

The integrated flux basis uses the full complex reciprocal material.
Its periodic potentials, affine lift, and normal-displacement traces are
therefore complex. The cell equations use

```text
integral grad(test)^H epsilon grad(potential) = 0
```

The x mass and stiffness integrals contain conjugate test factors. The
affine x forcing contains `conj(derivative_basis)*lift`, while the y
forcing contains `conj(potential_basis)`. Their zero Fourier modes remain
complex during assembly and solution. The aspect metric remains
`diag(1, aspect^2)`.

The material-weighted matrices need not be symmetric or Hermitian. Opposite
body/gap coupling blocks therefore exchange their real y factors while
retaining the x matrix orientation; transposing a completed block would
incorrectly transpose that x matrix too. Contact row scaling and diagonal
balancing retain the original equations and the existing 1e-8 residual
guard. No regularization or stopped derivative is added.

The historical `PolynomialBodyGapModel` still accepts only positive real
materials. Its `ComplexMaterialGapModel` subclass selects conjugate tests
and the full projection. This keeps existing real-cell study interfaces
and coefficient indices reproducible.

## Full energy and passivity

For the two physical corrector fields in B, projection forms

```text
Q = Fourier projection of B
G = Fourier projection of B^H epsilon B
Z = Fourier projection of epsilon
E = Q^(-H) G Q^(-1)
```

Both cross terms of G are retained independently. `G_xy` and `G_yx` cannot
share one coefficient, even though the material is scalar. The ninth
internal coefficient carries `G_yx`; the original eight indices remain
unchanged. Polynomial products conjugate the test field and its solution
coefficients, but never conjugate the Fourier phase or the material weight.
The affine/correction cross terms likewise use both orientations.

For any retained trial field, positive real epsilon makes the real
Hermitian part of G positive, and nonnegative imaginary epsilon makes its
dissipative Hermitian part nonnegative. Congruence by `Q^(-1)` preserves
these properties when Q is invertible. The adapter checks positive storage,
nonnegative dissipation within an absolute 1e-9 roundoff allowance, and
finite slots. For lossless input it also retains the existing absolute
1e-9 Hermitian-defect guard. It does not clip eigenvalues, symmetrize the
exported tensor or erase loss. Independent tests use tighter matrix
tolerances and verify absorption through the actual engine.

## Real mesh coordinates with complex contrast

The signed log-jump density extends to complex materials as

```text
j_k = (log(epsilon_(k+1)) - log(epsilon_k))/2
f(x) = sum_k j_k exp(-(x-s_k)^2/(2 radius^2))
rho(x) = 1 + strength/(sqrt(pi)*radius) * |f(x)|^2
```

The complex logarithm is continuous on these positive-real-part material
ratios. The density remains real and positive, includes phase contrast,
and agrees with the existing density for real materials. Gaussian-product
integrals use `Re(conj(j_i)*j_j)` and the same analytic erf primitive.
Coincident jumps still merge or cancel before forming the density. A
collapsed interval's unused complex material has no influence on the mesh,
field or energy. The 2e-12 CDF guard is unchanged.

## Independent controls and limits

Validation covers independent piecewise quadrature of complex x integrals,
tensor quadrature of polynomial energy, and a separate physical-quadrature
assembly of the weak equations. Public controls include analytic material
integrals, mixed-axis contact stacks on three engines, positive storage and
dissipation, and actual-engine absorption with independent material loss.
Fifteen live geometry, real-material and imaginary-material parameters are
compared with finite differences. Contact, zero loss and collapsed-interval
derivatives are checked separately.

The new perpendicular-contact control uses fourth-order one-sided
Richardson extrapolation at steps 1e-6 and 5e-7, in physical length units.
The contact enrichment has a narrow curvature scale: second-order
extrapolation at 1e-6 had errors 8.82e-6 and 1.83e-5 in the two axes.
Fourth-order extrapolation gives errors between 1.82e-8 and 1.20e-6 in the
two chosen windows, satisfying the unchanged absolute 3e-6 derivative
criterion. Smaller steps eventually amplify roundoff. Internal-gap and
loss-boundary controls retain their second-order comparisons.

The selected suites passed 494 distinct cases, with the one historical
fixed-P1 moving-interface expected failure unchanged. Manual examples,
lint and the strict documentation build also passed. Commands and
measurements are recorded in
`studies/results/nvf_complex_material_validation.log`. These qualify the
finite construction and integration, not optical convergence across all
geometries. The provisional 1e-5 optical-order increment is not a release
criterion. Curved interfaces, sheared cells, nonpositive real permittivities,
permeability cells and automatic projector selection remain outside this
builder's scope.

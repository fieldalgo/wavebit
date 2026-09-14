# Closed constitutive error from conforming electric and flux fields

Written: 2026-09-14

Targets:
- `wavebit/_core/conforming/galerkin_error.py`
- `wavebit/_core/conforming/equilibrated_flux.py`
- `tests/test_nvf_galerkin_error.py`
- `tests/test_nvf_equilibrated_flux.py`
- `studies/nvf_galerkin_error.py`
- `studies/results/nvf_galerkin_error.log`
- `studies/nvf_equilibrated_flux.py`
- `studies/results/nvf_equilibrated_flux.log`

The finite weak residual checks whether the selected nodal equations were
solved. It does not measure the field omitted by that mesh. The new closed-cell
diagnostic measures the physical mismatch between a conforming electric field
and an independently conforming displacement field. Coercivity turns this
mismatch into an error bound for the transverse constitutive cell at fixed
retained Fourier modes. A refinement driver uses that bound and reports failure
if the supported mesh range cannot meet the requested tolerance.

This supplies a stopping criterion for the closed auxiliary cell. It does not
certify a gap derivative, a Maxwell observable, convergence in Fourier order,
or ordinary NVF routing. The metallic optical results in the
[coupled-pair study](nvf-coupled-pair-seam-galerkin.md) remain unqualified.

## The two physical fields

Use cell averages as integrals, and let P be the orthogonal projection onto
the requested transverse Fourier modes. Phi synthesizes those modes in the
original Cartesian frame. The finite electric solution is

```text
E_h = Phi*x + grad(u_h),       P E_h = I.
```

Each column corrects one retained electric input. The potentials are continuous
periodic real nodal basis functions with complex solved coefficients. Thus
E_h has continuous tangential traces. Its normal displacement epsilon*E_h
need not be continuous pointwise at a material interface on a finite mesh.

A second solve uses the material 1/conj(epsilon), which stays in the same
passive rotated half-plane. Conjugating the result and reversing the Fourier
indices recovers the response for 1/epsilon. A real quarter-turn R maps the
potential gradients to divergence-free fields. Inverting the retained dual
response gives

```text
D_h = Phi*x_d + R grad(v_h).
```

D_h has continuous normal traces. Its high Fourier modes are divergence-free;
its retained part is free to represent the constitutive response. No complex
material-dependent trial frame is introduced. The dual tensor is returned as
a diagnostic and never substituted into the electric tensor.

### Minimizing the mismatch within the same flux space

The optional `flux='least_squares'` strategy keeps the original electric solve
and minimizes `integral(|epsilon*E_h - D_h|^2/a)` over all fields
`D_h = Phi*y + R grad(v)` in the same closed potential space. This is a
positive weighted projection, with complex coefficients in the original real
geometric basis. The default `flux='surrogate'` retains the reciprocal-material
method above.

Let G contain the closed common gradients, C(w) be the scalar Fourier
convolution matrix of w, Q(w) = integral(Phi^H w G), and
K(w) = integral(G^T w G). With w = 1/a, the full projection system is

```text
L = [ diag(C(w), C(w))       R Q(w) ]
    [ (R Q(w))^H             K(w)  ]

b = [ integral(Phi^H w epsilon E_h)       ]
    [ integral((R G)^T w epsilon E_h)     ]

L [y; v] = b.
```

The nodal gauge is fixed exactly as in the original closed cell. No potential
or retained mode is deleted. A separate geometry-identical cell at angle zero
assembles the positive metric materials, even when the physical material
sector uses an angle beyond pi/2. The forcing uses the original physical
sector. Finite, Hermitian, Cholesky positivity and original weighted
stationarity checks guard the projection before evaluating the residual Gram.
Neither this metric solve nor the returned flux modifies the electric tensor.

The surrogate flux is a feasible member of the same space. Weighted
stationarity makes the optimized residual orthogonal to every admissible flux
variation, for every retained input. Therefore, in exact arithmetic,

```text
M_surrogate - M_least_squares
    = integral((D_surrogate - D_least_squares)^H
               (D_surrogate - D_least_squares)/a) >= 0.
```

This reduces the bound for every input combination, not merely one sampled
polarization. Reported floating cancellation allowances still apply. For this
strategy `dual_tensor` denotes `P D_h`, rather than a separate constitutive
solution. For real positive materials at angle zero, the weighted optimum
agrees with the surrogate flux and preserves the hypercircle identity below.

## Coercivity gives the bound

Let E be the exact solution of the *infinite constrained cell problem at this
fixed Fourier set*. It satisfies P E = I and its high-mode displacement is
orthogonal to all high-mode potential gradients. Write

```text
e = E_h - E
r = epsilon*E_h - D_h
a = Re(exp(-i*angle)*epsilon) > 0.
```

Both electric fields have the same retained projection, so e is a high-mode
gradient. Both the exact displacement and D_h are orthogonal to that space.
For every retained input vector, the complex inner product therefore gives

```text
<e, epsilon*e> = <e, r>
||e||_a^2 = Re(exp(-i*angle)*<e, r>)
          <= ||e||_a * ||r/sqrt(a)||.
```

Hence the residual Gram M bounds the energy error for all input combinations:

```text
M = integral(r^H r / a)
||e||_a <= sqrt(lambda_max(M)) * ||input||.
```

The exported tensor is H_h = P(epsilon E_h). Weighted Cauchy-Schwarz for this
projection supplies the operator bound

```text
W = integral(Phi^H (|epsilon|^2/a) Phi)
||H_h - H||_2 <= sqrt(lambda_max(W) * lambda_max(M)) = B.
```

For real positive materials and angle zero, stationarity yields a separate
identity: M = H_h - H_dual, a positive-semidefinite matrix. Tests verify this
identity against the independently assembled two-field residual. No such
identity is assumed for complex materials.

These statements follow directly from the displayed field constraints and
coercivity argument. They make no assertion about a Maxwell propagation
operator's sensitivity to H.

## Integration and floating arithmetic

The residual on each regular triangle is the sum of retained plane waves and
a polynomial gradient of the selected supported degree. Polynomial products and mixed
Fourier moments use the original exact triangle integration routines. Pure
Fourier products use a separate exact rectangle indicator transform, including
the larger difference-frequency set. Collapsed triangles have zero physical
area at contact. No sample grid, gap cutoff, material smoothing, or new
third-party implementation is used.

The raw Gram is returned without a positivity repair. Its Hermitian defect
and minimum eigenvalue are reported. The spectral calculation uses its
Hermitian part only after the defect guard. A cancellation allowance of
64 * machine_epsilon * matrix_size * sum_of_contribution_norms is reported
and added to the largest residual eigenvalue when evaluating B. Negative
eigenvalues beyond that allowance reject the estimate. All original weak,
retained, flux, reciprocal and physical checks remain active with their
existing tolerances; the transformed dual additionally checks its inverse
and retained projection.

The allowance exposes floating cancellation; it is not an interval-arithmetic
proof, nor does it rigorously enclose errors in the linear solves. Accordingly
the API is a floating evaluation of an exact-arithmetic bound. Near a numerical
floor, stronger arithmetic verification is required before interpreting it as
a certificate. Uniform-medium residuals are integrated directly and remain
at roundoff instead of subtracting two large tensor energies.

## Automatic refinement and its limits

`estimate_closed_error(cell, flux='surrogate', **geometry)` accepts an explicit pair or periodic
rectangle Galerkin cell, degree one or two, or the explicit
[polynomial pair family](nvf-polynomial-galerkin.md), degree two through four.
Geometry keywords follow that
cell's `closure` call with gap and all tangents omitted. It returns the
unchanged electric tensor, dual tensor, raw residual Gram, energy/operator
bounds and diagnostics. Pair oblique geometry uses the original fixed plan.
Periodic rectangles retain their positive diagonal basis requirement.
Both estimator strategies support the same families. The refinement driver
forwards its `flux` keyword at every level; invalid strategy names are refused
before a solve. The selected strategy is recorded as `diagnostics['flux_method']`.

`refine_closed_error` begins at that cell's level and advances through the
supported global refinement range, at fixed order, degree and coercivity
angle. Its declared error budget is

```text
B <= absolute_tolerance + relative_tolerance * max(0, ||H_h||_2 - B).
```

The relative term uses a lower bound on the exact tensor norm, so it does not
mistake a large erroneous approximation for a permissive relative tolerance.
The reported relative error bound is infinite if that lower bound is zero.
There is no plateau acceptance rule. At least one requested tolerance must be
positive. Reaching `max_refinement` without meeting the budget returns
`converged=False`, reason `refinement_limit`, and every level's measurements.
Numerical or material guard failures raise instead of returning success.
Omitting `max_refinement` selects the family's supported cap: level 3 for
the original linear/quadratic families, level 2 for cubic/quartic pairs.
An explicitly requested level outside that family is refused.

```python
from wavebit._core.conforming.periodic_rectangle_galerkin import PeriodicRectangleGalerkinGapCell
from wavebit._core.conforming.galerkin_error import refine_closed_error

cell = PeriodicRectangleGalerkinGapCell((1, 0), degree=2)
result = refine_closed_error(
    cell, period=(700., 560.), width=310., epsilon=4.,
    absolute_tolerance=0.05, max_refinement=3,
)
print(result.converged, result.refinement, result.history)
```

The adaptive mesh decision is discrete and has no JVP. This explicit NumPy
diagnostic neither changes ordinary factorization selection nor widens the
scope of an opening derivative. The current dense mesh cap is a resource
limit, not an accuracy guarantee.

## Independent checks and measurements

Predetermined tests compare the residual Gram against physical Duffy integrals
at two quadrature orders. They use rational geometry, an independent periodic
quotient and Vandermonde gradients, for both polynomial degrees, both
geometries, dielectric and passive metal inputs. Real positive tests check the
energy identity across refinements. Exact one-dimensional laminated tensors
check that the operator bound contains the actual error, including metal
inputs. Uniform media, unit changes, translation, invalid inputs, successful
refinement and explicit exhaustion are separate controls.

The source-pinned measurement instrument compares orders 4 and 8, refinements
0 through 3, pair and periodic geometries, dielectric and metal materials.
The pair uses a 700 by 700 cell, two 300-wide, 300-high bodies with materials
(4, 4) or (12.25, -10+1j). The periodic rectangle is 350 wide and fills the
700-high cell at contact, with material 4 or -10+1j. Background is 1. The
dielectric angle is zero; the metal angle is 1.53 radians.

The completed run contains all 32 requested rows and records the measured
implementation's source hashes. The following table shows the first and last levels.
Intermediate levels, all residuals and cancellation allowances remain in the
raw log. B is an upper bound on tensor error, not a measured actual error.

| Geometry/material | Order | B, level 0 | B, level 3 | B / approximate tensor norm, level 3 |
| --- | ---: | ---: | ---: | ---: |
| Pair/dielectric | 4 | 1.9857569 | 0.5970002 | 0.1492543 |
| Pair/dielectric | 8 | 1.9488713 | 1.1686246 | 0.2921562 |
| Pair/metal | 4 | 4443.4265 | 2645.9727 | 17.7642165 |
| Pair/metal | 8 | 2871.8647 | 2260.9425 | 61.2533629 |
| Periodic/dielectric | 4 | 1.8973645 | 0.9563441 | 0.2390866 |
| Periodic/dielectric | 8 | 1.8973667 | 1.7736012 | 0.4434003 |
| Periodic/metal | 4 | 2687.4229 | 744.1672 | 13.4948972 |
| Periodic/metal | 8 | 982.5073 | 2160.6384 | 31.8017490 |

All original nodal residuals are below 1.04e-14 and assembled weak residuals
below 7.13e-14. Those small numbers therefore do not supply a small
constitutive error guarantee. The dielectric bounds improve with the tested
mesh sequence; the same meshes are less resolving at higher Fourier order.
The metal bounds are conservative, partly because the admitted half-plane
has minimum rotated material storage about 0.040785. They do not establish
that the actual tensor errors are thousands. They establish that this
estimator cannot certify the requested accuracy on these meshes.

The metal tensors also change in spectral norm by about 52 to 69 between
levels 2 and 3, independently of the error bound. There is no basis here for
accepting optical convergence or promoting the finite family. A finer or
better-resolved closed field remains necessary; increasing only the number
of opening traces leaves the closed tensor unchanged.

### Weighted flux comparison (2026-09-15)

The second source-pinned instrument compares both flux strategies on the same
order-4 closed pair, cubic/quartic spaces and refinement levels 0 and 1. Its
eight comparisons cover equal dielectric and dissimilar metal materials with
the same geometry and angles as above. All electric tensors are exactly
unchanged between strategies. Every projection passes its original weighted
stationarity guard; the largest residual is below 8.8e-14.

| Metal pair degree / level | Surrogate operator bound | Least-squares operator bound | Reduction |
| --- | ---: | ---: | ---: |
| 3 / 0 | 4253.0766 | 1904.4797 | 55.2% |
| 3 / 1 | 3225.5209 | 2261.4515 | 29.9% |
| 4 / 0 | 3954.7433 | 2001.8851 | 49.4% |
| 4 / 1 | 2883.5595 | 1970.5044 | 31.7% |

The dielectric estimates agree to floating precision, as the real energy
identity predicts. The minimum eigenvalue of each Gram improvement is
nonnegative within the combined reported cancellation allowances. Independent
tests assemble sampled physical flux columns with rational geometry, cardinal
gradients and direct Fourier fields, then solve their weighted QR projection.
Two quadrature orders agree with the implementation's exact-moment residual
Gram and retained flux for all supported degrees and both geometries. Tests
also cover exact laminate bounds, units, translation, uniform negative-real
material sectors, invalid strategies, corrupted solutions and refinement
exhaustion.

All measured metal bounds still exceed the approximate tensor norm, giving
an infinite relative bound. A smaller conservative bound is useful, but it
does not resolve the optical convergence failure or qualify ordinary NVF
routing. No new third-party code or dependency is used by this projection.

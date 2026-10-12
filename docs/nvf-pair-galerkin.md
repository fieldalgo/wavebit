# A real constrained-gradient finite contact response

Written: 2026-09-14.

Targets:
- `src/wavebit/_core/conforming/cartesian_pair_galerkin.py`
- `src/wavebit/_core/conforming/triangle_moments.py`
- `tests/test_nvf_pair_galerkin.py`
- `studies/nvf_pair_galerkin.py`
- `studies/results/nvf_pair_galerkin.log`
- `studies/results/nvf_pair_galerkin_quadratic.log`

The new explicit NumPy finite family solves a different potential correction
for every retained transverse Fourier column. Its real trial space and original
complex material equations preserve reciprocity, passivity and holomorphic
material dependence together. Analytical gap/material directional derivatives
include exact contact and arbitrarily small positive gaps. Independent original
nodal and Laurent equations verify this finite calculation. The measured metal
optics still change substantially with the potential space; this is not a
qualified default NVF replacement or a completed optical-convergence result.

## The retained field is a constraint

Let Phi be the retained orthonormal transverse plane waves, L the gradients of
real periodic nodal potentials, and Q their retained Fourier
projection. The constant potential gauge is removed. Write

```text
Z = L - Phi Q,
E = Phi c + Z a = Phi x + L a,
x + Q a = c.
```

Z has zero retained Fourier projection. Thus a corrector cannot change the
physical electric coordinates c passed to the Maxwell solver. The trial space
is closed under complex conjugation because the Fourier box includes both
signs and the nodal geometry is real. It depends on geometry, not material.
Unlike the two constant-incident correctors in the earlier pair energy model,
a has a separate column for each retained transverse electric field.

The original physical integrals are

```text
A = integral Phi.H epsilon Phi,
R = integral Phi.H epsilon L,
S = integral L.T epsilon Phi,
K = integral L.T epsilon L.
```

All integrals are divided by cell area. Complex epsilon is never conjugated in
S or K; S is not R.H for an absorbing material. For real L and Fourier reversal
J, S = R[J,:].T. Stationarity with the retained constraint gives

```text
x + Q a = c,
d = A x + R a,
K a + S x - Q.H d = 0.
```

The resulting d = H c is the constitutive response. At positive gap it is also
the Schur complement in the zero-projection gradient basis Z. The independent
positive-gap oracle uses that Schur form and separately integrates the physical
energy. The implementation eliminates x and d algebraically and solves one projected
potential system for all retained right-hand sides:

```text
B = R - A Q,
D = K - S Q - Q.H B,
a = solve(D, -(S - Q.H A)),
H = A + B a.
```

The implementation also eliminates the original weak potentials first when
their count exceeds the number of retained field/flux unknowns. With
`U = solve(K, Q.H)` and `V = solve(K, S)`, the same equations become

```text
[ I-Q V,   Q U ] [x] = [I],
[-A+R V, I-R U ] [H]   [0]
a = U H - V x.
```

At contact, Q.H in these expressions means the analytically scaled test
moments defined below. Both eliminations differentiate their original equations;
neither truncates a rank, changes the trial space or adjusts the material. The
choice depends only on matrix dimensions. Metadata names the `elimination` and
the selected matrix's `solve_condition`; the old `projected_condition` field
described only the potential elimination. Independent tests compare both forms
and reassemble the original nodal equations for each.

The contact version retains the different physical-column and divided-test-row
scalings described below. The projected matrix has the number of potential
unknowns, rather than four times the number of retained Fourier modes. For
example, refinement 3 has 575 open potentials, while order (8,8) has 289 modes.
The independent contact oracle retains x, d and the original contact potentials
in a separate uncondensed system.

## What the construction guarantees

At the stationary field, `c.H H c = integral E.H epsilon E`. Consequently passive
physical materials give nonnegative dissipation. A common rotated material
half-plane gives positive rotated storage. Since the retained projection is c,
Parseval also gives the scalar bounds

```text
HermitianPart(exp(-i alpha) H) >= min_j Re(exp(-i alpha) epsilon_j) I,
DissipativePart(H) >= min_j Im(epsilon_j) I.
```

These are properties of the exact finite construction, not interval-arithmetic
certificates for the floating-point assembly. The implementation checks physical
storage/dissipation, reciprocity, retained coordinates and original residuals.
Tests also check the scalar storage bound and the monotone decrease of lossless
energy under nested potential refinement. None is an optical error bound.

A real trial space gives Lorentz reciprocity `H = J H.T J` directly. No average
with the transposed response is taken. This differs from the explicit
[conjugate-paired frame energy](nvf-paired-frame.md), which averages the energies
of two material-dependent field maps.

A, R, S and K are linear in the physical scalar materials, while Q and the trial
space are material independent. On its nonsingular domain H is therefore
holomorphic in those materials. A directional derivative rotated by i gives i
times the original derivative. No material conjugation, absolute-value contrast,
reference permittivity, fitted loss or stopped derivative enters the equations.
The admission phase certifies the material sector only.

A uniform material gives exactly `H = epsilon I`; its first patterned material
variation is the direct Fourier transform of that variation. This is a stronger
control than changing every material by the same constant. General patterned
finite-order values need not equal Cartesian Li factorization.

## Exact contact and directional derivatives

The existing integer periodic-collapse graph is reused. Let s be gap divided by
`sqrt(det(basis))`, C inject contact potentials and Zc contain their independent
opening differences. The physical nodal potential is `C u + s Zc v`. After
analytically dividing the difference test rows by s, the weak matrix is

```text
Kbar = [[Kcc, s Kcz],
        [Kcz.T, Kzz]].
```

Physical Fourier columns carry the same s; divided test rows do not. This
matrix need not be Hermitian even for a lossless material: its row scaling must
be respected. At zero, its contact and difference equations retain the finite
right-opening response. No numerical inverse gap, small-gap cutoff, eigenvalue
clipping or material-dependent rank decision is used.

For a triangle with phases z_i, the mean plane wave is
`2 exp[z_0,z_1,z_2]`, an exponential divided difference. A moving phase appends
its node in the derivative. This evaluates a reference-simplex mean directly,
including when the physical triangle has zero area. Multiplication by the
signed area supplies the indicator transform; gradient-area cancellation is
performed analytically before multiplication. No physical area is divided out
of a computed small Fourier integral. The existing confluent divided-difference
implementation handles repeated and nearby phases. The larger Fourier difference
grid is integrated once in batches of at most 16 triangles; retained-mode means
are gathered from it. Stored means still scale with triangle count times Fourier
coefficient count. There is no whole-solver constant-memory claim.

Every solve and its derivative are checked against the original equations.
The code additionally reassembles every unprojected nodal equation, including
the gauge row and collapsed-strip reactions. Residual guards remain `1e-10`.
Two defect-correction steps solve the same equations without damping or added
loss. Invalid material, geometry, nonfinite output or failed original/physical
checks raises. A failed response is never repaired by symmetry projection.

## Independent finite verification

The 48 tests in `test_nvf_pair_galerkin.py` cover:

- Contact and positive-gap values for real dielectrics, absorbing dielectrics
  and metals, using two independent Duffy integration orders, 16 and 24.
- Contact derivatives from successive original Laurent equations. Geometry
  coefficients and Laurent stiffness are constructed with rational physical
  Vandermonde matrices; the candidate's rescaled blocks and triangle moments
  are not used. Value tolerance is `3e-10`, and derivative tolerance is `3e-9`
  after multiplication by the normalization length.
- All four frozen orientation charts, signed/skew lattice bases, unequal
  Fourier orders, and contact refinements 0 through 3.
- Uniform values, exact first patterned material variation, complex-linear
  material JVPs, simultaneous gap/material directions and physical-unit scaling.
- Fourth-order material secants at two windows, positive-gap central secants,
  second-order contact Taylor remainders and the gap `1e-9` limit.
- Nested lossless energy, scalar coercivity bounds, independent collapsed
  triangle means/derivatives, failure injection and invalid replay rejection.

The shared topology and rational geometry helpers were already independently
qualified. The new oracle solves original nodal constraints and successive
Laurent equations; it does not call the candidate assembly or closure. The
existing 61 quadratic-cell tests also pass after adding the triangle mean
helper. These tests verify a finite constitutive family, not the continuum
Maxwell solution or all geometry derivatives.

## Optical measurement and decision

The unchanged pair fixture uses period 700, two touching 300-square rectangles,
background 1, thickness 300, wavelength 900, top/bottom permittivity 1/2.25,
theta 0.2 and phi 0. The potential mesh is refined independently of the retained
Fourier box. Li integrates the same material painting independently, but its
finite-order values are not assumed converged references.

The log records both polarizations, reflections, gaps `0, .0005, .001, .002,
1e-9`, two one-sided slope windows, original residuals and source hashes.
For materials `12.25 | -10+i`, the closed transmission totals are:

| potential refinement | p, order 4 | p, order 8 | s, order 4 | s, order 8 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.318112424 | 0.319220974 | 0.618596377 | 0.442763611 |
| 1 | 0.378351202 | 0.377955281 | 0.617416836 | 0.408439897 |
| 2 | 0.296914881 | 0.406190556 | 0.632438079 | 0.429152015 |
| 3 | 0.417043536 | 0.433015544 | 0.461736741 | 0.418364180 |

Li gives p `0.483946395 / 0.480976241` and s
`0.480212354 / 0.402781608` at orders 4/8. At order 8/refinement 3,
the new p/s values `0.433015544 / 0.418364180` are closer to this finite-order
comparison than the earlier two-corrector P1 energy values
`0.618325309 / 0.518567397`. Those earlier values are recorded in the
[refinement study](nvf-pair-refinement.md). This improvement at one setting is
not a general convergence result: the order-4 p value becomes less accurate
relative to its Li comparison, and the new ladder remains nonmonotone.

At order 8/refinement 3, the new p/s contact slopes are about
`-0.0166665 / +0.0104858` per length unit; Li gives
`-0.00323219 / +0.0115477`. Both signs now agree on this setting, whereas the
earlier two-corrector P1 slopes had opposite signs. The large p discrepancy
still rejects a contact-accuracy claim. Agreement between the two slope windows
only checks a resolved derivative of the respective finite calculation.

For equal dielectric materials `4 | 4` at order 8/refinement 3, the new p/s totals
are `0.997722367 / 0.987966005`, against Li `0.997682393 / 0.988163075`.
These differences are smaller than in the earlier two-corrector P1 example,
but they too remain finite-order comparisons. Lossless power balance and the
original equation checks pass throughout the recorded ladder.

The result is retained as a real, reciprocal, passive and material-holomorphic
finite contact implementation. It does not resolve general metal optical
convergence, supply a continuum contact-gradient bound, or change default NVF
routing. Further progress requires resolving the potential space and Fourier
truncation together; the finite material and reciprocity guarantees alone do
not establish either optical limit.



## Quadratic potentials and local assembly

Degree two uses six nodal potentials on each affine triangle. Its gradient
is linear, specified at the three reference vertices. The physical weak
integrals therefore contain quadratic polynomials. Fourier loads use the exact
means `mean(lambda_i exp(-i G.r))`, computed as twice the exponential divided
difference with vertex i repeated. Differentiating that repeated phase counts
it twice. These reference means remain finite when a triangle collapses.
Their sum gives the constant plane-wave mean; no area division is performed.

The contact C/sZ scaling is unchanged. Its transverse polynomial annihilation
is certified in the integer topology before any physical geometry is evaluated.
The independent quadratic oracle uses a six-node Vandermonde, rational
geometric cofactors, numerical Duffy integrals, and successive original Laurent
equations. Refinement three retains the existing absolute 3e-10 value and 3e-9
normalized derivative checks. Computing the oracle's cofactors from rounded
inverse Jacobians initially exceeded the derivative guard; rational cofactors
resolve that reference roundoff without changing either tolerance.

Assembly now restricts a triangle to the at most three or six potentials it
actually touches, then scatters its blocks. It no longer constructs a global
nodal-width gradient or dense global-width local product for every triangle.
The global weak solve is still dense. This is a bound on local assembly storage,
not a sparse whole-solver or constant-memory claim.

On the skew, offset metal control at degree two/refinement three and order
(1,0), the potential elimination's condition number is about 6.74e7, compared
with 8.65e3 for the retained elimination. This is an observation for that
specified finite system, not a universal conditioning guarantee. Original
nodal residuals and the same independent derivative guards apply to either
elimination.

### Quadratic optical ladder

The quadratic log uses the same fixture, physical power checks and contact
windows as the linear study above, adding order 12. All 48 requested rows
passed their finite residual, reciprocity and physical guards. Metal closed
transmission is:

| Refinement | p, n=4 | p, n=8 | p, n=12 | s, n=4 | s, n=8 | s, n=12 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.378686023 | 0.361990161 | 0.414288503 | 0.631855974 | 0.496658999 | 0.433634811 |
| 1 | 0.428119068 | 0.457057276 | 0.429199030 | 0.446176516 | 0.415485036 | 0.387158657 |
| 2 | 0.454192751 | 0.466272060 | 0.442039501 | 0.398871799 | 0.391349471 | 0.395952814 |
| 3 | 0.467236471 | 0.467848935 | 0.461417724 | 0.417392569 | 0.375747030 | 0.392565033 |

Li at n=12 gives p 0.489085870 and s 0.407928008. Neither the geometric nor
Fourier ladder is settled by these data. The p contact slope at n=8 also
changes from -0.0216597 at refinement two to +0.0057894 at refinement three,
whereas Li gives -0.00323219. Thus the larger space does not qualify the metal
contact derivative, despite its exact finite algebraic properties.

The separate [periodic rectangle study](nvf-periodic-galerkin.md) applies the
same equations to the original closing-gap case. It improves that case's
p-polarized derivative substantially but retains an s-polarized trace error.
Neither study changes ordinary NVF dispatch.


## Explicit use

```python
import numpy as np
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.cartesian_pair_galerkin import CartesianPairGalerkinGapCell

geometry = dict(basis=np.diag([700., 700.]), widths=(300., 300.), height=300.)
plan = CartesianPairOpeningPlan.from_geometry(**geometry, refinement=1)
cell = CartesianPairGalerkinGapCell(plan, order=(2, 1), degree=2, coercivity_angle=1.53)
slots, gap_derivative, metadata = cell.closure(
    gap=0., epsilon=(12.25, -10.+1j), **geometry)

# Derivative along simultaneous changes of the two materials.
slots, material_derivative, metadata = cell.closure(
    gap=7., epsilon=(12.25, -10.+1j), gap_tangent=0.,
    epsilon_tangent=(.2, .3j), **geometry)
```

Slots include the layer axis and retain the existing five-slot ordering,
including negative transverse cross blocks. The default JVP direction opens
the gap by one caller length unit. `background_tangent` defaults to zero.
The other geometry parameters are fixed when differentiating. This module is
NumPy-only; no JAX tracing or automatic public NVF dispatch is supplied. Planning
and geometry admission follow the existing Cartesian pair family, with strict
containment and refinement 0 through 3.

`degree=1` remains the default and uses three linear potential nodes per
triangle. `degree=2` uses six quadratic potential nodes and a linear gradient.
Both permit geometric refinements 0 through 3. Degree two at refinement three
has 2,303 open and 1,791 contact free potentials; its edge nodes use the next
dyadic graph. This does not expand the separate P1 plan's refinement contract
or the earlier two-corrector quadratic cell's supported range.

No dependency, external solver, copied source or reference data was added. The
implementation uses the existing NumPy dependency and repository triangle and
periodic-topology code. Reproduce from the checkout:

```sh
PYTHONPATH=. python -m pytest -q tests/test_nvf_pair_galerkin.py tests/test_nvf_pair_quadratic.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_pair_galerkin
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_pair_galerkin --degree 2 --orders 4 8 12 --out studies/results/nvf_pair_galerkin_quadratic.log
```

# A constrained real field through periodic rectangle contact

Written: 2026-09-14.

Targets:
- `wavebit/_core/conforming/periodic_rectangle_galerkin.py`
- `wavebit/_core/conforming/cartesian_pair_galerkin.py`
- `tests/test_nvf_periodic_galerkin.py`
- `studies/nvf_periodic_galerkin.py`
- `studies/results/nvf_periodic_galerkin.log`

The explicit finite Galerkin family now includes a rectangle approaching its
own periodic image. Its gap value and analytical right derivative remain
finite at contact and at subpixel gaps. The p-polarized contact derivative
approaches the independent Cartesian Li sequence, unlike the large closing-gap
error in ordinary NVF. The s-polarized derivative still has a significant finite
trace-space error. This is an implemented and independently checked finite
model, not a qualified replacement for default NVF.

## One graph for an open and closed periodic seam

A positive diagonal lattice contains a centered rectangle of fixed width w
and height `period_y - gap`. Three x intervals cover the left background,
inclusion and right background. Three y intervals cover half the gap, the
body and half the gap. Every rectangle is split into two triangles and
subdivided dyadically. The topology is fixed as the gap tends to zero.

Opposite outer nodes are periodic. At zero gap, the lower and upper gap bands
collapse to the corresponding periodic boundary. C identifies the resulting
contact potentials; Z carries independent opening differences. The physical
potential is `C u + s Z v`, with `s = gap / sqrt(cell_area)`. As in the
[pair Galerkin method](nvf-pair-galerkin.md), analytical row scaling removes
the inverse gap from the weak equations before assembly. Thin triangles retain
their moving Fourier means even at zero area.

One detail differs from the separated pair chart: the open potential chosen
as gauge belongs to a group with several collapsing nodes. The other nodes of
that group still need opening-difference potentials. Removing the entire group
would clamp a physical gap voltage at the periodic corner. The implementation
includes those Z columns, and an independent periodic quotient verifies the
complete free potential space.

Each retained electric Fourier column gets its own correction. The real nodal
space and physical complex material equations give the same reciprocal, passive
and material-holomorphic finite response as the pair family. The shared kernel
now admits either horizontal or vertical affine opening; it still rejects a
two-axis changing Jacobian. There is no normal-projector blend, material switch
or automatic Li fallback in this family.

## Finite verification

The test reference constructs rational partition coordinates, material labels
from physical triangle interiors, and its own periodic quotient and C/Z
transformation. It then integrates independent nodal polynomials and solves
the original unscaled or successive Laurent equations. It does not call the
candidate gradient, Fourier moment, rescaled assembly or response solve.

The full-height contact triangles require a higher independent integration
order than the smaller separated-pair triangles: q=16 leaves about 2.5e-8
Fourier error in a coarse contact reference. The q=24 and q=32 integrals agree
under the unchanged 3e-10 value and 3e-9 normalized derivative guards.

The periodic tests cover both potential degrees, positive and zero gaps,
dielectric and passive metal materials, the finest admitted contact mesh,
the collapsing gauge group, uniform materials, complex-linear material JVPs,
simultaneous gap/material directions, physical units, nested lossless energy,
Taylor limits and gaps of 1e-9. Invalid geometry and material sectors are
refused. These finite checks do not assert continuum optical convergence.

## The original closing-gap fixture

The recorded NumPy float64 CPU study uses a 700-square cell, width 350,
inclusion permittivity 12.25, background/top/bottom 1/1/2.25, wavelength 900,
thickness 300, theta 0.2 and phi 0. Fourier truncation is `(n,n)`.
The quadratic potential mesh has geometric refinements 1, 2 and 3.

Each row records both polarizations at gaps 0, 0.0005, 0.001, 0.002 and 1e-9.
Two second-order opening secants use the first four values. Two additional
central differences perturb the constitutive matrices along the analytical
gap JVP, checking its propagation through the optical solve. All 24 rows
pass the original equation and lossless physical power guards. The largest
recorded residual is below 8e-14; the largest discrepancy between the narrower
geometric and constitutive-tangent slopes is about 2.2e-9.

All slopes below are per unit **opening gap**. They have the opposite sign
to the height derivative in the earlier
[closing-gap investigation](nvf-contact-followup.md).

| n | p slope, r=1 | p slope, r=2 | p slope, r=3 | Independent Li |
|---:|---:|---:|---:|---:|
| 3 | 0.0015525332 | 0.0014818610 | 0.0014768701 | 0.0014764148 |
| 5 | 0.0018527092 | 0.0018278128 | 0.0018186184 | 0.0018176706 |
| 7 | 0.0019371697 | 0.0019030230 | 0.0018926767 | 0.0018915460 |
| 9 | 0.0019224714 | 0.0019200563 | 0.0019176980 | 0.0019163814 |

At refinement three the p slope differs from the respective finite Li
comparison by about 0.031%, 0.052%, 0.060% and 0.069%. This comparison is not
an infinite-order error bound: the Fourier ladder itself still moves.
The gap-1e-9 transmission changes remain of the expected tiny size, about
1e-12 to 8e-12 across all rows. Ordinary NVF on its grid 256 keeps much larger
contact changes and opening secants; the log records that unchanged comparator.

The second polarization prevents a completion claim:

| n | s slope, r=1 | s slope, r=2 | s slope, r=3 | Independent Li |
|---:|---:|---:|---:|---:|
| 3 | 0.0080003908 | 0.0058075916 | 0.0051130464 | 0.0046515710 |
| 5 | 0.0042944929 | 0.0015764631 | 0.0007068003 | 0.0001864901 |
| 7 | 0.0034249282 | 0.0006549681 | -0.0002450946 | -0.0007785736 |
| 9 | 0.0031387670 | 0.0003492708 | -0.0005668281 | -0.0011066412 |

The s value at exact contact agrees with the respective Li value to numerical
roundoff, while its derivative does not. Thus matching a closed-bar solution,
lossless power, or the candidate's own secants is insufficient. Increasing the
potential space lowers the bias, but refinement three retains an absolute
slope discrepancy near 5.4e-4 at the larger Fourier orders. Resolving the
finite gap trace, including its ends, remains necessary. Metal pair optics and
the separate tilted resonant cases also remain unresolved.

## The normal trace error has an independent variational expression

In the retained `k_y=0` sector, a closed bar's y-directed electric field is
exactly its retained plane wave: epsilon varies only with x, so this field
has zero constitutive divergence. The closed transverse yy response is
therefore the direct material convolution for every potential refinement.

For positive real inclusion and background permittivities, opening the gap
introduces a normal electric-field correction. Let V denote
the finite leading normal gradients on the collapsed gap triangles, M their
background-weighted Gram matrix, and F the forcing from the contrast between
the body and background traces. In the normalized coordinates used by the
implementation, with `L = sqrt(cell_area)`, the finite yy contact derivative is

```text
H'_yy = C((epsilon_background - epsilon) indicator) / period_y
        - F.H solve(M, F) / L.
```

Here the convolution and its retained sector are one-dimensional in x. The
test constructs M and F from rational geometry, independent nodal polynomials
and gap integrals alone; it does not use the candidate matrix blocks or an
optical/Li result. The independently assembled expression reproduces the
candidate's yy derivative under the existing normalized 3e-9 guard.

If the normal gap trace could vary freely in L2, the same quadratic
minimization would instead give

```text
H'_yy,unrestricted = C((epsilon - epsilon**2 / epsilon_background)
                       indicator) / period_y.
```

For positive real materials, the finite derivative minus this unrestricted
response is a positive semidefinite projection-error Gram matrix. Twelve
tests verify the finite identity and that ordering for both potential degrees,
three refinements and two material/background pairs. The result identifies a
missing trace approximation term, rather than an optical differentiation
error. It does not imply a sign for the resulting transmission-derivative
error, since transmission is a separate nonlinear function of the tensor.

A valid forward improvement must supply the missing normal traces through a
consistent positive-gap field space. Substituting the unrestricted expression
only into the reported derivative would leave the old forward error intact.
This identity is a qualification target for that field-space development; it
does not change the production derivative or relax an acceptance criterion.

## Explicit use and limits

```python
from wavebit._core.conforming.periodic_rectangle_galerkin import PeriodicRectangleGalerkinGapCell

cell = PeriodicRectangleGalerkinGapCell(order=(3, 5), degree=2, refinement=2)
slots, derivative, metadata = cell.closure(
    gap=0., period=(700., 700.), width=350., epsilon=12.25)
```

The five slots include a layer axis and can be passed to `solve_conv`.
`gap_tangent=1` is the default; scalar `epsilon_tangent` and
`background_tangent` can vary simultaneously. At gap zero, the gap tangent
has right-derivative meaning. Period, width and Cartesian offset are fixed
when differentiating. Width must lie strictly inside the x period and gap
in `[0, period_y)`. Periods must be positive. Passive materials must share the
declared coercive sector; `coercivity_angle` certifies that sector without
changing the physical equations.

Potential degree is 1 or 2 and isotropic refinement is 0 through 3. At degree
two/refinement three there are 2,303 open and 767 contact free potentials.
The global weak and optical systems are dense. Local triangle assembly uses
only its actual nodal support. NumPy values and analytical first JVPs are
provided; this family has no JAX replay or ordinary NVF dispatch.

No dependency, copied external code or external solver data was added.
Reproduce with the existing numerical environment:

```sh
PYTHONPATH=. python -m pytest -q tests/test_nvf_periodic_galerkin.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.nvf_periodic_galerkin
```

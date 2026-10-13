# Two physical rectangles in an oblique auxiliary cell

Written: 2026-09-10

Targets:
- `src/wavebit/rcwa/_engine/conforming/cartesian_pair.py`
- `studies/rcwa/nvf_cartesian_pair/nvf_cartesian_pair.py`
- `tests/rcwa/test_nvf_cartesian_pair.py`
- `tests/rcwa/test_nvf_cartesian_rectangle_pair_api.py`
- `studies/rcwa/nvf_cartesian_pair/records/nvf_cartesian_pair.log`

The experimental `CartesianPairCell` supplies finite P1 field, energy and
material coefficients for two horizontal, equal-height physical rectangles
inside an oblique periodic cell. A positive gap and an exact full-height
shared edge each have a fixed conforming mesh. Independent weak equations,
triangle integration and NumPy/JAX derivatives verify these finite models.
Opening a gap is handled by a separate
[NumPy finite right-opening response](nvf-cartesian-pair-opening.md); it does
not change this cell's fixed-contact JAX tangents. More general pairs and
optical convergence remain unqualified. The explicit experimental
[`CartesianRectanglePairNVF`](../../guide/rcwa/cartesian-rectangle-pair-nvf.md) exposes this
bounded model through ordinary rectangle dictionaries and a mandatory plan.
The other public geometry adapters keep their current scope.

## Physical geometry and frozen charts

The columns of A are physical lattice vectors. Two rectangles of independent
positive widths share one positive height and lie left-to-right, with a
nonnegative gap. Their bounding rectangle is centered at
`A @ (.5, .5) + offset`. The offset translates the whole integration cell,
preserving the finite model's translation covariance. Changing A moves the
center and periodic boundary without shearing the Cartesian rectangle edges.

`CartesianPairPlan.from_geometry` selects one of four cyclic correspondences
between the outer parallelogram and the bounding rectangle. Each positive
inner strip has two triangles. Fans on the adjacent outer triangles include
every material endpoint, so interfaces have no hanging nodes. Refinement
splits each triangle into four using shared edge midpoints. Exact dyadic
weights give periodic node identities independently of live geometry.

| State | Refinement | Triangles | Vertices | Free potential unknowns | Periodic point pairs per axis |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-height contact | 0 | 14 | 10 | 6 | 2 |
| Full-height contact | 1 | 56 | 33 | 27 | 3 |
| Positive gap | 0 | 18 | 12 | 8 | 2 |
| Positive gap | 1 | 72 | 41 | 35 | 3 |

The immutable plan stores only contact state, cyclic choice and refinement.
Planning requires concrete geometry and is refused under JIT. Replay keeps
the selected topology fixed. Geometry must be real and finite, with positive
finite lattice determinant, strict containment using a `1e-10` fractional
margin, and positive finite triangle areas. Complex geometry is refused before
conversion to real values. Containment does not replace the triangle guard.

The bounded domain excludes unequal heights, vertical offsets, partial-edge
or corner contact, overlapping rectangles, arbitrary rotations and contact
with a periodic outer boundary. The preliminary geometry scan accepted 135
prescribed skew/aspect/gap cases, including 45 contacts; it is not a general
coverage result. Two forced cyclic choices fail even on the contained main
fixture, demonstrating the need for an explicit positive chart.

## Finite coefficients and closure

For each macroscopic Cartesian direction, the periodic P1 potential solves
the original weak equation with physical permittivity weights. One potential
value fixes the constant gauge. Diagonal balancing changes the linear solve's
coordinates; the residual is evaluated in the original full weak equation
and must not exceed `1e-10`.

The physical fields are B = I + grad(u) and G = epsilon BᴴB. Exact polygon
transforms project the nine piecewise-constant fields in the existing order
`Bxx, Bxy, Byx, Byy, Gxx, Gxy, Gyy, epsilon, Gyx`. Fourier phases use physical
reciprocal vectors and the original cell origin. The caller supplies a scalar
Fourier half-width or an `(x, y)` pair; exact requested differences are selected
before the energy closure's inverses. No lattice reduction or extra material,
phase or rotation factor is introduced.

`CartesianPairCell(plan, order=1)` exposes `coefficients` and `conv_matrices`.
Both return their arrays plus finite-cell metadata. Geometry lengths share
caller physical units. `epsilon` contains the two physical scalar inclusion
permittivities and `background` is a physical scalar, each finite with positive
real part and nonnegative imaginary part. NumPy and JAX require x64. Invalid
eager inputs raise; invalid traced updates poison every coefficient or slot.
Closure requires positive storage, dissipation eigenvalues above `-1e-9`,
and a lossless Hermitian discrepancy below `1e-9`, without clamping signed
results. These checks can reject a finite closure; they do not establish
arbitrary-contrast reliability.

## Independent verification

The main fixture uses nanometers: A = 700 [[1, .45], [0, .7]], widths
(140, 193), height 210, offset (49, -21), and gap 17 or 0. The permittivities
are (4 + .8i, 9 + .3i), with background 1 + .1i. Refinements 0 and 1 give four
small cells. Separate controls use skew -1.1, which selects a different
cyclic chart, and nonzero unequal Fourier orders `(1, 2)` and `(2, 1)`.

The oracle shares only the declared geometry. It separately constructs moving
vertices, accumulates original nodal K and F, and differentiates
K du = dF - dK u. Moving-triangle Duffy quadrature includes derivatives of
physical gradients, areas, reciprocal vectors, Fourier phases and material
weights. It does not call production assembly, balancing, polygon transforms
or closure. Closure derivatives independently differentiate explicit inverses.

All 16 dimensionless coordinates are live for a separated pair: four lattice
entries, two widths, height, gap, two offset entries and six real/imaginary
material entries. Contact controls keep `dgap = 0`, leaving 15 coordinates.
Mixed, whole-cell translation and common-scale directions bring the six-case
total to 111. Both coefficient and closure JVPs use the same `3e-9` absolute
guard per dimensionless direction. Value and reference-quadrature guards are
`3e-10`. The original four cases also retain 74 five-point finite-difference
controls at steps `2e-4` and `1e-4`.

| Preliminary finite control | Largest measured discrepancy | Guard |
| --- | ---: | ---: |
| Main four cells, q16/q24 values or derivatives | 2.603e-14 | 3e-10 |
| Main coefficients versus independent nodal/Duffy values | 2.887e-15 | 3e-10 |
| Main analytical derivative versus five-point step 2e-4 | 1.677e-10 | 3e-9 |
| Main analytical derivative versus five-point step 1e-4 | 2.174e-11 | 3e-9 |
| Different chart, gap/refinement 1/order (1, 2), q16/q24 | 4.146e-14 | 3e-10 |
| Different chart, contact/refinement 0/order (2, 1), q24/q32 | 8.755e-14 | 3e-10 |

The last case initially **failed q16/q24**: the height derivative of the
material coefficient at Fourier label (m, n) = (4, 2) differed by
`3.2018768211880356e-10`, above `3e-10`. The original failure and raw-array hashes
are retained in the result record. Increasing only reference quadrature to q24/q32
reduced the discrepancy to `8.754374308731933e-14` on the same contact mesh,
materials and Fourier set. The repeated q24 arrays are exactly identical.
This resolves reference integration for that finite case; it is not a q16
pass, a finite-space refinement or a relaxed tolerance.

Forty-eight zero/unequal-order lossy, lossless and homogeneous closure controls
verify slot dimensions, physical material coefficients, homogeneous B = I,
isotropic slots, storage and dissipation. Patterned lossless dissipation reaches
only `-4.737e-16` from roundoff, with Hermitian discrepancy below `4.653e-16`.
The tests recompute their independent references from source and need no
private result arrays. JAX tests skip when the optional backend is unavailable.

The integrated suite passed 25 tests in 12.47 s on an Apple M4 host with
16 GiB, NumPy 2.5.1 and JAX 0.11.0. A separate fresh process with JAX imports
blocked passed 17 tests and skipped its eight JAX tests in 5.25 s, without
changing the installation. The full suite's sampled peak RSS was 662,110,208
bytes; the child process resource high-water mark was 665,124,864 bytes.
Source and test hashes were unchanged across both runs. These operational
measurements describe these tiny controls, not the cost or accuracy of a
practical optical solve.

## Public physical-input qualification

The packaged kernel preserves every prior cell helper, planner and numerical
method's abstract syntax exactly; only module wording, exports and relative
imports change. The study module re-exports that implementation, and its
25-test source is unchanged. The public builder adds structural admission
and maps physical widths and centers directly into the existing finite cell.
It performs no background normalization, lattice reduction, extra Fourier
phase or rotation. Planning sorts concrete centers once while preserving
original input indices and material identity during replay.

Public tests cover both input orders, ordinary `l`/`lx`/`ly` precedence,
Material wrappers, uniform/patterned stacks, translations, common scale,
zero/unequal Fourier orders, strict constraints and invalid traced inputs.
The original nodal/Duffy oracle independently supplies explicit closure
variations for 111 public native-parameter directions across the same six
finite cases. The native coordinates are physical lattice entries, two
widths, one shared height, two horizontal centers, one shared vertical center
and six material components. Their tangents are mapped analytically from the
original dimensionless family. Contact directions preserve the shared edge;
arbitrary one-height, one-y or opening-gap tangents are outside that family.
The prior per-case quadrature choices and `3e-10` value/reference and `3e-9`
derivative guards are unchanged.

The API fixtures prescribe physical decimal dimensions and centers directly.
An initial test instead formed contact centers from dimensionless floating
values and produced a tiny negative gap, which admission correctly refused.
The computed contact gap was `-2.842170943040401e-14` nm. That failed fixture
record is preserved. The final physical primal is
specified explicitly; the adapter performs no snapping and separately refuses
a representable nonzero contact opening. This test-fixture correction changes
neither the finite model nor any tolerance. Mapped native geometry differs
from the dimensionless oracle coordinates by at most `1.137e-13` nm from
representation roundoff; the value and derivative comparisons pass their
original guards on those explicitly recorded primals.

The existing 25 and new 29 controls passed together: **54 tests in 15.89 s**.
A fresh process with JAX imports blocked passed 38 and skipped 16 in 5.54 s.
A package-only numerical smoke imports no study or JAX module. The combined
run sampled 759,201,792 bytes RSS, with a separate resource high-water mark of
759,513,088 bytes; source hashes stayed fixed. Three tiny `solve_conv` controls
at `(0, 0)`, `(1, 0)` and `(0, 1)` match independent uniform Airy powers and
check patterned lossless slot layout/energy. The patterned energy check and
manual example are integration controls, not independent optical references.
A further 156 existing single-cell, single-rectangle, ConformingNVF and
AffineProfileNVF controls pass in 56.83 s. The literal guide example runs
without importing study or JAX modules; its finite p/s transmissions are
0.9901311633000325 and 0.9366247460109753. These values document execution,
not optical convergence.

## Contact and accuracy limits

The exact-contact chart merges the two interface endpoints and removes
zero-width triangles. Opening the gap would add degrees of freedom and
degenerating triangles. An ordinary derivative of the contact chart cannot
recover that response; a limiting weak formulation remains necessary.
Contact-to-open and open-to-contact replay therefore raise or produce traced
NaNs. Within-chart contact widths, height, materials, lattice and offset remain
differentiable.

Equal-material contact has the same analytic material transform as one larger
rectangle. Its finite B/G need not equal the
[single-rectangle cell](nvf-cartesian-cell.md), because the pair mesh contains
a different potential space. No equality between those finite spaces is
enforced. These tests verify the stated finite equations and derivatives;
they supply no continuum error bound, patterned optical qualification or
change to the existing optical convergence criterion.

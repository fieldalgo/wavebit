# Independent affine-cell and directional optical comparisons

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


Written: 2026-09-09

The [full-metric affine kernel](nvf-affine-cells.md) now has an independent
fitted FEM reference and accepts distinct x/y Fourier orders directly.
On the tested oblique profile, refining x inside the auxiliary cell reduces
the order-12 transmission difference from the fitted reference by about 73%
for p and 83% for s. The remaining optical order change is larger, especially
in y for s. These measurements do not establish optical convergence.

The latest [candidate endpoint comparison](#candidate-comparison-at-y32)
reaches `(24,32)`, with p/s total-transmission differences of
`-1.52748e-6`/`-3.42893e-6` from the degree-10 fitted auxiliary field.
Independent 3D optical accuracy, oblique contact-opening derivative
qualification and a public general-geometry adapter remain open.

The implementation and instruments are:

- `src/wavebit/rcwa/_engine/conforming/affine_cell.py`: scalar or `(x,y)` retained orders.
- `studies/rcwa/nvf_cell_corrector/nvf_cell_corrector.py` and `studies/rcwa/nvf_gap_reference/nvf_gap_reference.py`: physical
  gradients on an affine image of an independently fitted periodic mesh.
- `studies/rcwa/nvf_affine_convergence/nvf_affine_convergence.py`: one maximal projection per cell,
  checked reference provenance, static enclosures and directional optics.

All shapes here are affine images of reference rectangles, hence physical
parallelograms. The public physical-rectangle adapter retains its existing
contract. Neither this reference nor the new order support silently shears
ordinary rectangle inputs.

## Fixed physical problem

The physical lattice, in nm, has columns

```text
A = [[700,315], [0,490]]
J = A/700 = [[1,.45], [0,.7]].
```

Reference widths use period 700. The envelope width is 350, perpendicular
gap is 17, and one seam at x=31 separates permittivities 4 and 9 in background
1. The physical shape is the image under J of this centered reference
profile. The lossless film has thickness 300 nm, wavelength 900 nm, incidence
theta=.2 and phi=0, and exterior permittivities 1 and 2.25. These parameters
are identical in every optical row below.

## An independent physical-gradient reference

The optional NGSolve instrument meshes the material boundaries conformingly
on the reference square. It uses periodic H1 polynomials with a mean-zero
constraint. For a reference derivative, the physical gradient is
`J^(-T) grad(u)`. The equation uses that physical gradient and a Cartesian
unit forcing directly. It does not import Wavebit, use the candidate's
integrated-flux basis, or use its laminated contact lift.

The constant area Jacobian cancels between physical integrals and cell-area
normalization. Fourier phases stay in reference coordinates because the
physical reciprocal vector is `2 pi A^(-T) k`. Both primal and reciprocal
material problems are solved in physical components. Their energies give
the existing two-dimensional variational enclosure. It concerns the static
scalar cell, with floating-point integration and algebra; it is neither
interval arithmetic nor an optical transmission bound.

FEM Fourier projection uses element-owned integration points. Five new
optional tests cover exact isotropic uniform fields, the exact oblique
laminate, physical rotation/common scale, independent material Fourier
integrals, quadrature refinement and invalid transforms. The eight existing
static FEM tests also pass.

| reference h | degree | vertex refinement | elements | scalar unknowns | relative static enclosure |
| ---: | ---: | ---: | ---: | ---: | ---: |
| .10 | 4 | 20 | 728 | 5825 | 2.01494e-5 |
| .06 | 6 | 20 | 860 | 15481 | 2.83491e-6 |
| .04 | 8 | 40 | 1622 | 51905 | 2.62367e-7 |

Here h is a reference mesh length in micrometers. The finest export covers
coefficient cap 32. Increasing its triangle quadrature from 28 to 32 changes
any coefficient by at most `2.74e-15`; the independently integrated mean
energy agrees within `1.37e-13`. The degree-6 to degree-8 refinement changes
some coefficients by `3.41e-6`, so the quadrature check is not a substitute
for the separate cell refinement.

Across the tested optical orders through 16, the degree-6 to degree-8
reference change is below `2.82e-7` for p transmission and `7.18e-7` for s.
Those are observed reference changes, not certified optical error bounds.
The FEM reference solves the static auxiliary problem, not independent
three-dimensional Maxwell scattering. Both optical paths consume their
fields through the same energy closure and Wavebit engine.

## Which auxiliary direction needs refinement?

All candidate rows use base body rows 4, material grading strength 8 and
radius .01, local flux functions, and edge fraction `1e-5`. x/y subdivisions
refer to the candidate mesh, not retained Fourier orders.

| x subdivisions | y subdivisions | body degree | gap degree | unknowns | relative static enclosure |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 2 | 4 | 1007 | 6.63190e-4 |
| 1 | 1 | 4 | 6 | 1943 | 5.69499e-4 |
| 1 | 2 | 4 | 6 | 3671 | 5.67375e-4 |
| 2 | 1 | 4 | 6 | 3887 | 1.42965e-4 |
| 4 | 1 | 2 | 4 | 4031 | 1.40300e-4 |
| 6 | 1 | 1 | 4 | 3455 | 9.74742e-4 |

Doubling y intervals has little effect on the degree-4 body cell here.
Doubling x intervals narrows its enclosure by about 75%. Spending a similar
number of unknowns on x=4 with a quadratic body gives a comparable enclosure;
reducing the body to linear polynomials loses that benefit. Thus increasing
x alone while reducing every other degree is not supported by this evidence.
No default mesh setting has changed.

The following comparison uses the degree-8 fitted FEM reference at the same
outer order 12. Differences are signed candidate minus reference:

| auxiliary field | p T | s T | p difference | s difference |
| --- | ---: | ---: | ---: | ---: |
| fitted FEM | .852527657039 | .731688518712 | — | — |
| x1, body 2, gap 4 | .852506675486 | .731631268484 | -2.09816e-5 | -5.72502e-5 |
| x1, body 4, gap 6 | .852511035065 | .731652308240 | -1.66220e-5 | -3.62105e-5 |
| x2, body 4, gap 6 | .852522051245 | .731678833971 | -5.60579e-6 | -9.68474e-6 |

The final difference decreases by about 73%/83% relative to the first
candidate row. This is improvement in the auxiliary-field comparison at a
fixed optical order. It is not the total error relative to continuum
Maxwell scattering. At order 16 the refined candidate still differs from
the fitted reference by `-5.50011e-6` / `-1.06650e-5` for p/s transmission.

## Keep the exact directional Fourier set

The two orders count spatial Fourier harmonics, not physical mesh cells or
polarizations. `(Nx,Ny)` retains integer labels `-Nx <= m <= Nx` and
`-Ny <= n <= Ny`, with physical reciprocal vector
`2 pi A^(-T) (m,n)`. Thus these are the supplied lattice-coordinate
directions, which need not be orthogonal in physical space. Increasing Nx
while fixing Ny resolves additional x-coordinate variation; increasing Ny
while fixing Nx resolves additional y-coordinate variation. Both orders
participate in both polarization solves.

This fixture has no x/y exchange symmetry. Its reference x material pieces
have widths 206 and 144, whereas its transverse background gap has width
17 in a period of 700. The exact material Fourier integral illustrates
the narrow-gap scale: with `delta = 17/700`, its nonzero-y factor is
`(1-delta) sinc(n (1-delta)) = (-1)^(n+1) sin(pi n delta)/(pi n)`.
For `abs(n) delta` small, its magnitude stays near delta, so many y harmonics
can remain relevant. This is a statement about the geometry's Fourier
content, not a required optical order or an optical error estimate.

At phi=0 the incident s electric field is along physical y, normal to the
gap faces in this particular affine map. The dielectric jump constrains the
normal displacement while allowing a jump in the normal electric field;
the p field and the slanted material seams couple differently. This supplies
a physical reason to expect different directional sensitivities, although
the optical tables do not isolate every contribution from geometry,
polarization and lattice coupling. In particular, an x refinement of the
auxiliary-cell mesh and a y refinement of the outer optical order address
different errors and can both be effective on the same structure.

`AffineCellGapModel(order=(x,y))` now records that pair as
`truncation_order`. The x/y directions are the original integer reciprocal
labels, even though their physical vectors are not orthogonal. The cell
projects the required coefficient envelope, then selects exactly those
modes in Q, G and the longitudinal matrix before closing them. It does not
extract a submatrix from a larger already-inverted constitutive matrix.
Zero half widths are supported. Invalid, negative or boolean orders raise.

Twenty checks cover the directional matrices, zero-width order directions,
contact/positive-gap engine energy balance, the noncommutation of inversion
and submatrix selection, and order validation. Scalar order behavior is
retained by the existing affine-cell suite.

For the independently fitted degree-8 field held fixed:

| x half width | y half width | p T | s T |
| ---: | ---: | ---: | ---: |
| 8 | 8 | .852727535317 | .732709929588 |
| 12 | 8 | .852597669148 | .732748852613 |
| 8 | 12 | .852656933773 | .731651099274 |
| 12 | 12 | .852527657039 | .731688518712 |
| 16 | 12 | .852475474946 | .731690420550 |
| 12 | 16 | .852495746055 | .731227967383 |
| 16 | 16 | .852443153987 | .731229889742 |

At the last four corners, increasing x alone changes p by `-5.21821e-5`
and s by `+1.90184e-6`. Increasing y alone changes p by `-3.19110e-5`
and s by `-4.60551e-4`. The mixed increments are `-4.09975e-7` and
`+2.05210e-8`. The s order change is overwhelmingly in y, whereas both
directions still matter for p. A single square-order increment cannot
identify this allocation.

The square 12-to-16 changes are `-8.45031e-5` and `-4.58629e-4`.
Optical truncation remains consequential. No value in these tables defines
a new release threshold or an automatic stopping criterion.

The comparison driver now accepts `--directional-changes`. After the raw
optical records, it reports changes on each complete adjacent rectangle of
x/y orders. Each polarization includes the x change at both y levels, the
y change at both x levels, their mixed difference, and the diagonal change.
Reporting the parallel edges prevents opposing directional changes from
being hidden by a small diagonal difference. Geometry, material, candidate
resolution and reference metadata define separate series; the driver does
not combine their corners. Missing corners produce no rectangle, and
duplicate orders within one series raise. These records contain no
convergence verdict or error-bound estimate.

## Higher optical orders on the same affine profile

The degree-8 fitted reference was exported through coefficient cap 64 on
the same 1,622-element mesh. Quadrature 32 and 40 differ by at most
`4.89e-15` in any coefficient; the shared cap-32 coefficients are unchanged
from the earlier quadrature-32 export. The analytic material-coefficient
error in the quadrature-40 export is below `1.91e-15`. This extends the
available Fourier data without changing the fitted auxiliary cell.

Holding that reference fixed gives:

| x half width | y half width | p T | s T |
| ---: | ---: | ---: | ---: |
| 16 | 16 | .852443153986 | .731229889742 |
| 20 | 16 | .852417073421 | .731227345524 |
| 16 | 20 | .852428151483 | .731039456471 |
| 20 | 20 | .852401539891 | .731036855555 |
| 24 | 20 | .852388526661 | .731034096403 |
| 20 | 24 | .852395205346 | .730980282619 |
| 24 | 24 | .852381909685 | .730977440340 |

The complete order rectangles expose these signed changes. x-only and
y-only changes start at the lower corner; mixed is the remaining joint
contribution, so their sum is the diagonal change.

| order rectangle | polarization | x only | y only | mixed | diagonal |
| --- | --- | ---: | ---: | ---: | ---: |
| 16 to 20 | p | -2.60806e-5 | -1.50025e-5 | -5.31027e-7 | -4.16141e-5 |
| 16 to 20 | s | -2.54422e-6 | -1.90433e-4 | -5.66967e-8 | -1.93034e-4 |
| 20 to 24 | p | -1.30132e-5 | -6.33454e-6 | -2.82432e-7 | -1.96302e-5 |
| 20 to 24 | s | -2.75915e-6 | -5.65729e-5 | -8.31265e-8 | -5.94152e-5 |

The square-order changes decrease over these two windows, but neither
window bounds the remaining optical error. Both directional changes have
the same sign here, so their small mixed term is not concealing cancellation.
At 20 to 24, s changes about twenty times as much when increasing y as
when increasing x. p remains more sensitive to x. The measurements support
continuing directional refinement; they do not justify declaring optical
convergence or choosing a universal default order.

A separate fitted reference increases the FEM polynomial degree from 8
to 10 on the same mesh, raising its scalar unknown count from 51,905 to
81,101. Its relative static enclosure narrows from `2.624e-7` to
`1.428e-7`. The largest coefficient change through cap 48 is `1.30e-7`.
At the same optical order `(24,24)`, this changes p transmission by
`+1.22e-8` and s by `+2.65e-8`, much less than the preceding optical
order increments. This is a separate observed reference-refinement check,
not a bound on optical error; both reference paths use Wavebit's Maxwell
solver. The degree-10 export uses quadrature 40 and the same affine map,
profile, mesh length and vertex refinement.

The refined candidate with x subdivisions 2, base body rows 4, body degree
4 and gap degree 6 gives `p T=.852377641159` and `s T=.730966937870`
at `(24,24)`. It retains 3,887 auxiliary unknowns and a relative static
enclosure of `1.42965e-4`. Candidate minus the degree-10 fitted reference
is `-4.28e-6` for p and `-1.05e-5` for s. These same-order differences
are smaller than the degree-8 reference's 20-to-24 optical increments.
The comparison supports further outer-order refinement on this fixture;
it does not remove the need to refine the candidate cell or establish
an independent three-dimensional optical error bound.

## The next y interval reverses the s change

The continuation uses the same degree-8 fitted auxiliary field and the
cap-64, quadrature-40 export. Its full-space energy closure now uses the
[lower-memory two-sided solve](nvf-energy-closure.md#full-space-projected-fields).
No retained modes, physical settings or optical acceptance criteria change.
An order-24 rerun agrees with the earlier quadrature-32 optical record
within `1.86e-12` in transmission; the separate matrix continuity check
holds the quadrature-40 export fixed.

| x half width | y half width | p T | s T |
| ---: | ---: | ---: | ---: |
| 24 | 24 | .852381909686 | .730977440342 |
| 20 | 28 | .852393633568 | .730986733073 |
| 24 | 28 | .852380193212 | .730983935373 |
| 20 | 32 | .852394179680 | .731015929291 |
| 24 | 32 | .852380698959 | .731013192091 |
| 28 | 24 | .852376352297 | .730975406939 |

At x=24, increasing y from 24 to 28 changes p transmission by `-1.72e-6`
and s by `+6.50e-6`. The s change reverses the preceding 20-to-24 decrease
of `-5.67e-5`. x=20 gives the same reversal (`+6.45e-6`), comparing the
earlier quadrature-32 row with the current quadrature-40 row. The export
quadrature comparison and the order-24 optical rerun separate that small
numerical change from the observed directional increments. These finite
windows do not support a monotone optical tail or an automatic stopping
rule based only on the latest small increment.

At x=20, the following y=28-to-32 change in s is `+2.92e-5`, larger
again than the preceding `+6.45e-6`. Thus the apparent flattening near
y=28 does not persist over the next measured window. Both current rows
use quadrature 40 and the same fixed auxiliary field.

x=24 confirms the same pattern: the y=28-to-32 s change is `+2.93e-5`.
Over that interval, the p changes at x=20 and x=24 are positive too,
`+5.46e-7` and `+5.06e-7`, but smaller than the x=20-to-24 p change
of about `-1.35e-5`. The complete rectangle therefore resolves both
directions and their coupling without assuming either one is converged.

For x=20-to-24 and y=28-to-32, the rectangle gives the following signed
increments from its lower corner:

| polarization | x only | y only | mixed | diagonal |
| --- | ---: | ---: | ---: | ---: |
| p | -1.34404e-5 | +5.46112e-7 | -4.03649e-8 | -1.29346e-5 |
| s | -2.79770e-6 | +2.91962e-5 | +6.04997e-8 | +2.64590e-5 |

The small mixed terms do not remove the separate directional changes.
Neither polarization has a demonstrated continuum optical error bound.
The candidate comparison for this sequence uses `(24,24)`; the later
[candidate endpoint](#candidate-comparison-at-y32) reaches `(24,32)`.
The separate
[reference-degree check](#reference-degree-refinement-at-y32) extends the
degree-8/10 comparison to y=32, without making it an error bound for these rows.

The separate x=24-to-28 edge at y=24 changes p by `-5.56e-6` and s
by `-2.03e-6`. The p change is smaller than the preceding x=20-to-24
edge at that y, but it supplies neither a bound on the remaining x tail
nor the missing rectangle corners at x=28. Across all six new optical
rows, the largest lossless energy error is `1.39e-11`, below the unchanged
algebraic guard. No optical convergence criterion has been selected.

## Completing the x=28 comparisons

The continuation retains the same cap-64, degree-8, quadrature-40 fitted
field and physical profile. It uses the production runtime at `ea4c85e` (landed as `3032cdf`),
including diagonal scaling and the [direct boundary solve](../../reference/common/what-a-solve-answers.md#11-solving-only-the-requested-boundary-columns).
The boundary equations, retained Fourier sets and optical guards are
unchanged. At order 20, per-order p/s R/T agrees with the saved
lifetime-only calculation within `5.54e-12`; no threshold was adjusted to
admit the higher orders. The older corners below precede the runtime
arithmetic changes, while sharing the identical projection export.

| x half width | y half width | p T | s T |
| ---: | ---: | ---: | ---: |
| 28 | 28 | .852374558731 | .730981857810 |
| 28 | 32 | .852375041494 | .731011083754 |

The first new corner completes x=24/28, y=24/28. Its signed changes from
the lower corner are:

| polarization | x only | y only | mixed | diagonal |
| --- | ---: | ---: | ---: | ---: |
| p | -5.55739e-6 | -1.71647e-6 | -7.70926e-8 | -7.35095e-6 |
| s | -2.03340e-6 | +6.49503e-6 | -4.41597e-8 | +4.41747e-6 |

At y=28 the x=24-to-28 edge changes p by `-5.63448e-6` and s by
`-2.07756e-6`. p remains more sensitive to x, while the s x/y changes
oppose each other. The small square-order s increment consequently includes
partial cancellation. At x=28 the y=24-to-28 s change is `+6.45087e-6`,
confirming the earlier sign reversal at a third x order. Neither the
smaller p x increment nor the small s diagonal supplies a tail bound.

The second new corner completes x=24/28, y=28/32:

| polarization | x only | y only | mixed | diagonal |
| --- | ---: | ---: | ---: | ---: |
| p | -5.63448e-6 | +5.05747e-7 | -2.29844e-8 | -5.15172e-6 |
| s | -2.07756e-6 | +2.92567e-5 | -3.07740e-8 | +2.71484e-5 |

At x=28, y=28-to-32 changes p by `+4.82763e-7` and s by
`+2.92259e-5`. The s change is about 4.53 times its preceding y-window
change. Across x=20, 24 and 28, the same y=28-to-32 s increment is
`+2.91962e-5`, `+2.92567e-5` and `+2.92259e-5`, respectively. Raising
x does not remove the observed y variation over these tested windows.
At y=32 the x=24-to-28 changes are `-5.65747e-6` for p and
`-2.10834e-6` for s. The small mixed terms and decreasing x increments
do not make the remaining y tail bounded or monotone.

The two fresh processes complete in 390.16 s and 453.82 s, with maximum
RSS of 6.395 GB and 6.477 GB, respectively (macOS bytes, including imports,
projection closure and native workspaces). These shared-host observations
are not an isolated speed comparison. The greatest new-row lossless balance
error is `7.20e-12`, below the unchanged algebraic guard. The fitted
reference remains degree 8. The later reference-degree check reaches
`(24,32)`, as does the later [candidate endpoint](#candidate-comparison-at-y32).
Neither the static field enclosure nor these finite optical
windows supplies an independent three-dimensional optical error bound.

## Refining the candidate at fixed optical orders

Increasing the candidate's x subdivisions from 2 to 4 while retaining
base rows 4, body degree 4 and gap degree 6 raises the auxiliary unknown
count from 3,887 to 7,775. The y subdivisions, local flux basis, material
grading, edge fraction and physical profile stay fixed. The new projection
uses the [bounded x batches](nvf-affine-cells.md#bounded-fourier-projection-intermediates)
and covers coefficient cap 48. The batching changes storage, while the
explicit x refinement changes the finite auxiliary space.

The relative static enclosure narrows from `1.42965e-4` to `3.35654e-5`,
a 76.5% reduction. Against the saved degree-10 FEM projection, the maximum
coefficient difference through cap 48 falls from `1.93657e-3` to
`6.31351e-4`. The analytic material-coefficient error in the new candidate
is below `4.00e-15`. Its primal and reciprocal cell residuals are
`5.95e-10` and `1.37e-10`, below the unchanged `1e-8` residual guard.

The fixed-order optical comparisons give:

| optical order | candidate x subdivisions | p T | s T | p difference from fitted field | s difference from fitted field |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 2 | .852522051245 | .731678833971 | -5.60579e-6 | -9.68474e-6 |
| 12 | 4 | .852525824322 | .731684219311 | -1.83272e-6 | -4.29940e-6 |
| 24 | 2 | .852377641159 | .730966937870 | -4.28069e-6 | -1.05289e-5 |
| 24 | 4 | .852380240145 | .730973827940 | -1.68170e-6 | -3.63887e-6 |

The order-12 reference is the earlier degree-8 fitted field, and the
order-24 reference is the degree-10 refinement. Each difference compares
the candidate and reference at the same optical order. The x=2 and fitted
optical rows are retained from the earlier measurements; the new x=4 rows
use the current production runtime. The separate arithmetic continuity
checks above and exact before/after candidate coefficient comparison
distinguish the storage change from this auxiliary refinement.

At order 24, the absolute p/s differences shrink by about 61%/65%.
At order 12, they shrink by about 67%/56%. This is evidence that the
candidate cell itself benefits from further x resolution while its body
degree is retained. It does not change a default resolution or convert
the static enclosure into an optical bound. The remaining fixed-order
differences are still measurable, and the y-order oscillation of the
fitted-field optics remains unresolved.

The new primal cap-48 projection takes 45.29 s; including the reciprocal
mean-energy solve takes 53.80 s. Maximum process RSS at that point is
4.231 GB, including imports, model construction and native workspaces.
The order-12 and order-24 optical calculations take 2.92 s and 120.41 s;
their greatest lossless balance error is `4.20e-12`. These are shared-host
observations, not controlled performance comparisons. No new FEM solve,
third-party code or dependency is used for these rows: the references
are the already-qualified numerical exports. The independent 3D Maxwell
uncertainty remains open; the subsequent
[candidate/reference endpoint](#candidate-comparison-at-y32) extends to `(24,32)`.

## Extending y to 36

The degree-8 fitted field is extended from coefficient cap 64 to 72 on
the same mesh, without changing the auxiliary solution. At quadrature 40,
all shared cap-64 coefficients are exactly identical. Increasing the new
export's quadrature from 40 to 48 changes any coefficient by at most
`5.27e-15`; its analytic material-coefficient error is below `1.91e-15`.
The degree, mesh, residuals and static enclosure remain unchanged.

Holding x at 24 and increasing y gives:

| x half width | y half width | p T | s T | coefficient cap |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 32 | .852380698959 | .731013192091 | 64 |
| 24 | 36 | .852381870038 | .731042637367 | 72 |

The y=32-to-36 changes are `+1.17108e-6` for p and `+2.94453e-5` for s.
The latter is slightly larger than the preceding y=28-to-32 increment
of `+2.92567e-5`. Thus the renewed s variation persists over another
window; the latest increment is not decreasing here. Neither the small
earlier interval near y=28 nor the new positive sequence supplies a
remaining-error bound or a stopping criterion.

The old row retains its cap-64 provenance. The new row retains its cap-72
provenance; their exact shared coefficients and identical finite field
justify this explicit comparison. The directional-reporting helper still
groups by complete metadata and does not merge these exports automatically.
This is a single y edge, not a newly completed x/y rectangle. Both rows
use the exact retained Fourier space before matrix closure.

The new optical solve takes 412.28 s, with peak process RSS 6.681 GB and
lossless balance error `8.20e-12`, below the unchanged algebraic guard.
These are shared-host observations, not controlled performance timings.
This continuation checks the next y increment of the fixed degree-8 field.
It does not check degree-10 reference refinement or candidate-field error
at y=36; the later reference-degree and candidate endpoint checks both
reach y=32. Independent three-dimensional optical convergence qualification
also remains open. Its
[Rayleigh-data helper](nvf-independent-fem.md#rayleigh-data-for-a-physical-primitive-lattice)
handles the physical affine reciprocal lattice, and its separate
[geometry helper](nvf-independent-fem.md#physical-affine-geometry-and-periodic-pairs)
passes material-volume and periodic-pair controls. The new
[affine Maxwell entry point](nvf-independent-fem.md#affine-maxwell-connection-and-independent-controls)
connects both polarization sources, boundary traces and physical-area
power normalization and passes independent slab controls. Its patterned
discretization and outgoing-boundary uncertainties remain open.
The first [patterned FEM refinement](nvf-independent-fem.md#patterned-affine-boundary-and-degree-refinement)
now reaches Rayleigh cut 4 and three-dimensional FEM degree 6 at two
mesh sizes. At h=.14, changing degree 5 to 6 still changes p/s transmission
by `+8.46e-5`/`+1.43e-4`, despite a nearly closed energy budget.
The [physical per-order comparison](nvf-independent-fem.md#comparing-the-physical-diffraction-orders-with-wavebit)
checks the shared `+G` convention and shows that total agreement can hide
larger cancelling errors in individual diffraction orders.
The optional [physical-edge mesh control](nvf-independent-fem.md#refining-the-physical-dielectric-edges)
subsequently reduces the measured degree-4-to-5 transmission change by
60.8%/88.4% for p/s; the maximum individual R/T change falls by
89.0%/95.0%. This is one finite degree interval on a different fitted
mesh, with larger cost at fixed degree, and still supplies no optical
error bound.

The independent Maxwell instrument's
[fixed-mesh degree-5-to-6 comparison](nvf-independent-fem.md#raising-the-film-degree-on-a-fixed-mesh)
on an edge-refined physical mesh still changes p/s transmission by
`+1.09803e-4`/`+1.14131e-4`. Raising only the film neighborhood hides
further individual-order changes behind smaller total changes. The
independent optical uncertainty remains open despite much smaller
energy-budget remainders; these FEM degrees are separate from the
two-dimensional auxiliary-field degree comparison below.

The completed [vertex/degree rectangle](nvf-independent-fem.md#separating-vertex-and-degree-refinement)
still gives p/s degree-5-to-6 changes of `+9.00952e-5`/`+7.22772e-5`
on the vertex-target-.070 mesh. At degree 6, adding that target changes
T by `+5.07196e-5`/`+9.79657e-5`, with larger cancelling changes in
individual orders. Both refinement axes remain relevant; a tiny energy
budget does not qualify an independent optical reference.

The subsequent [.070/.065 vertex interval](nvf-independent-fem.md#checking-the-nearer-vertex-interval-at-both-degrees)
also retains degree sensitivity. At .065, raising global degree 5 to 6
changes p/s T by `+9.50082e-5`/`+5.39352e-5`. The p degree change
grows compared with .070 while s shrinks, and total changes conceal
larger individual-order changes. Further degree verification remains
necessary before qualifying an independent reference.

The later [degree-6/7 vertex rectangle](nvf-independent-fem.md#vertex-and-degree-refinement-at-degrees-6-and-7)
now contains all four finite results after the separately audited
vertex-.070 degree-7 control completes. On that mesh, p/s degree changes
are `+5.23387e-5`/`+4.23733e-5`; adding the vertex target at degree 7
changes T by `+3.74748e-5`/`+7.14124e-5`. The corresponding mixed
differences are `-1.32448e-5`/`-2.65533e-5`. Individual s transmission
changes still reach `1.14586e-4`, with 64.24% cancellation in the spatial
total. Completion removes the missing finite corner without providing a
remaining optical-error bound or qualifying either finite Maxwell result
as a converged oracle.

## Reference degree refinement at y=32

The degree-10 fitted field is now exported through coefficient cap 64,
retaining h=.04, vertex refinement 40 and the same 1,622-element mesh.
The 81,101-unknown solution and its relative static enclosure `1.42771e-7`
are unchanged from the cap-48 export. At quadrature 40, the complete
shared cap-48 coefficient region is exactly identical, and the metadata
differ only in cap. This is an export extension, not another field solve
resolution change.

Increasing export quadrature from 40 to 48 changes any coefficient by
at most `5.27e-15`. The quadrature-40 analytic material-coefficient error
is below `1.91e-15`. Comparing degree 8 with degree 10 through cap 64
gives a maximum coefficient difference of `1.30e-7`. Independent checks
using explicit interval exponential integrals confirm the material
coefficients, real-field conjugacy and identity mean. The separate
reference-degree comparison retains both original metadata records.

Both degrees are solved at `(24,32)` with the same current production
runtime and cap-64, quadrature-40 exports. Each field's own retained
space is closed before its Maxwell solve:

| fitted FEM degree | p T | s T | lossless balance error |
| ---: | ---: | ---: | ---: |
| 8 | .852380698959 | .731013192091 | 1.39e-11 |
| 10 | .852380710977 | .731013218636 | 6.91e-12 |

Degree 8 to 10 changes p transmission by `+1.20173e-8` and s by
`+2.65450e-8`. These are about 2.38% and 0.0907% of the previously
measured degree-8 y=28-to-32 increments. The largest difference across
all individual diffraction-order R/T powers is `5.37e-8`; the comparison
therefore also checks redistribution among the outgoing orders. The
fresh degree-8 totals agree with the historical row within `6.56e-15`,
separating earlier runtime arithmetic changes from this field refinement.

The observed reference-degree effect at the y=32 endpoint is much smaller
than the earlier optical-order changes, particularly for s. This extends
the reference-field check beyond `(24,24)`. It does not reproduce the
whole y interval at degree 10 or provide a remaining-error bound: degree
10 has not been run at y=28 or 36. The subsequent candidate endpoint
comparison is recorded below. Both fitted fields still use Wavebit's Maxwell engine;
the independent 3D affine optical comparison remains open.

The degree-10 optical calculation takes 292.40 s with peak process RSS
5.367 GB; the degree-8 control takes 283.72 s and 6.290 GB. RSS includes
imports, closure and native workspaces. The two degree-10 scalar FEM
exports take 74.05 s and 105.53 s for quadratures 40 and 48. These are
shared-host observations and do not establish a performance comparison.

## Candidate comparison at y=32

The candidate with x subdivisions 4, base rows 4, body degree 4 and
gap degree 6 is now exported through coefficient cap 64 for optical
order `(24,32)`. Its 7,775-unknown auxiliary space is unchanged:
y subdivisions 1, local flux basis, edge fraction `1e-5`, grading
strength 8, grading radius .01 and quadrature 12. The shared cap-48
coefficients are exactly equal to the earlier export. Extending the
Fourier representation therefore does not refine this auxiliary field.

The analytic material-coefficient error is `4.00e-15`; real-field
conjugacy is exact, identity-mean error is `8.88e-16` and the independently
computed xy/yx Gram terms differ by at most `4.81e-12`. Primal and
reciprocal residuals are `5.95e-10`/`1.36e-10`, below the existing
`1e-8` guard. The relative static enclosure remains `3.35654e-5`.
These controls establish projection continuity and finite static
consistency, not optical accuracy.

The optical comparison reuses the saved degree-10, quadrature-40 fitted
projection and its `(24,32)` powers. The Wavebit runtime and projection
closure sources are unchanged between that calculation and this candidate
run, as checked against the recorded source revision and current file
hashes. Both calculations use the same physical profile and p/s incident
power normalization, with a separate closure at the selected optical order.

| polarization | candidate T | fitted-field T | candidate minus fitted T | candidate minus fitted R |
| --- | ---: | ---: | ---: | ---: |
| p | .852379183496 | .852380710977 | -1.52748e-6 | +1.52750e-6 |
| s | .731009789706 | .731013218636 | -3.42893e-6 | +3.42892e-6 |

An independent array audit evaluates the physical `+G` reciprocal vectors
and maps `(m,n)` to `[n+32,m+24]`. Reflection propagates only `(0,0)`;
transmission propagates the following three orders:

| transmitting order | candidate minus fitted, p | candidate minus fitted, s |
| --- | ---: | ---: |
| (-1,-1) | -4.60836e-7 | -9.37715e-7 |
| (-1,0) | +6.09474e-8 | +2.42107e-6 |
| (0,0) | -1.12759e-6 | -4.91228e-6 |

The largest individual R/T differences are `1.52750e-6`/`4.91228e-6`
for p/s. In particular, the s transmitting orders partly cancel, leaving
a smaller total-T difference. Candidate lossless balance error is
`7.93e-12`, below the unchanged `1e-9` algebraic guard. The maximum
coefficient difference from the fitted field through cap 64 is
`6.31351e-4`. This extends the fixed-order auxiliary-field comparison
beyond `(24,24)`; it does not establish a Fourier tail bound, a
continuum optical error bound or a release tolerance. Both fields
still use the same Maxwell engine, and the independent 3D FEM
discretization uncertainty remains open.

The recorded candidate totals change from the historical `(24,24)` row
by `-1.05665e-6`/`+3.59618e-5`. The auxiliary profile, settings and shared
coefficients match, but the older optical runtime hash and per-order
arrays were not saved. These are historical total differences, not a
newly controlled same-runtime directional refinement or an interval
reproduced with both candidate and fitted fields.

The primal cap-64 export takes 59.82 s and the reciprocal mean check
8.68 s, with peak process RSS 4.425 GB. The optical calculation takes
279.65 s, including a 16.60 s closure, with peak RSS 6.228 GB.
Primal coefficients are saved before the reciprocal stage, and all p/s
diffraction powers are saved with physical settings and input/source
hashes before the final comparison. Each stage completed; timings are
shared-host observations and do not establish a speed comparison.

## Reproduction and guards

The reference runs used separately installed `ngsolve==6.2.2606` and
`netgen-mesher==6.2.2606`. On 2026-09-09, their installed package metadata and
official [NGSolve](https://github.com/NGSolve/ngsolve/blob/master/pyproject.toml)
and [Netgen](https://github.com/NGSolve/netgen/blob/master/pyproject.toml)
manifests identified both as `LGPL-2.1-only`.
These are optional reference-generation tools: neither is imported by the
`wavebit/` runtime or declared as a Wavebit installation dependency, and their
implementations are not vendored here. The comparison consumes numerical
projection exports. Wavebit does not
distribute either package; see the running/output provisions in section 0
of the [upstream license](https://github.com/NGSolve/ngsolve/blob/master/LICENSE).

This records the scope of the reference use, not clearance to redistribute
the FEM environment. Bundling or integrating the libraries requires a new
check of notices, corresponding source and replacement/relinking obligations,
as applicable, together with the licenses of their bundled dependencies.

Run the reference export with an environment containing NGSolve. Set
`NVF_FEM_PYTHON` to that Python executable and `NVF_AFFINE_REF` to an output
NPZ path outside the checkout:

```bash
PYTHONPATH=$PWD "$NVF_FEM_PYTHON" studies/rcwa/nvf_gap_reference/nvf_gap_reference.py --cap 32 --h .04 --degree 8 --quadrature 28 --refine 40 --transform 1 .45 0 .7 --output "$NVF_AFFINE_REF"
PYTHONPATH=$PWD python studies/rcwa/nvf_affine_convergence/nvf_affine_convergence.py --orders 8 12:8 8:12 12 16:12 12:16 16 --projections "$NVF_AFFINE_REF" --cells 2:4:4:6
PYTHONPATH=$PWD "$NVF_FEM_PYTHON" -m unittest tests.rcwa.test_nvf_affine_reference tests.rcwa.test_nvf_cell_corrector tests.rcwa.test_nvf_gap_reference
```

The comparison loader requires the identical transform, profile and
coefficient cap. It checks the longitudinal coefficients against exact
material integrals, so matching metadata alone cannot admit an export
of another material distribution. It rejects non-real field data,
unequal xy/yx Gram terms and a non-identity field mean. Every retained
space receives its own closure. Lossless power balance is checked as an
algebraic diagnostic, not used as a convergence certificate.

Append `--directional-changes` to either a reference or candidate comparison
to emit the complete order-rectangle diagnostics. The existing `NX:NY`
spelling still selects retained orders; this option analyzes their changes
without altering the physical problem or the closure.

To reproduce the higher-order rectangle sequence, export the fitted
reference with `--cap 64 --h .04 --degree 8 --quadrature 32 --refine 40`
and the same transform, then run:

```bash
PYTHONPATH=$PWD python studies/rcwa/nvf_affine_convergence/nvf_affine_convergence.py --orders 16 20:16 16:20 20 24:20 20:24 24 --projections "$NVF_AFFINE_REF" --cells --directional-changes
```

For the separate reference-refinement check, export cap 48 with degree 10
and quadrature 40, keeping the same geometry and mesh settings. Compare
that export and the candidate at `--orders 24 --cells 2:4:4:6`.

For the additional candidate refinement, use
`--orders 12 24 --cells 4:4:4:6`. This projects the x=4 field once through
cap 48, checks its reciprocal mean-energy enclosure, and closes each
retained optical space independently. Keep the fitted degree explicit
when comparing with the earlier x=2 and reference rows above.

For the next directional intervals, use cap 64, degree 8 and quadrature 40
on the same mesh, then run:

```bash
PYTHONPATH=$PWD python studies/rcwa/nvf_affine_convergence/nvf_affine_convergence.py --orders 24 20:28 24:28 20:32 24:32 28:24 --projections "$NVF_AFFINE_REF" --cells --directional-changes
```

That sequence includes an order-24 continuity rerun, a complete rectangle
on x=20/24 and y=28/32, and a separate x=24-to-28 edge at y=24. It does
not contain the additional corners needed for a rectangle reaching x=28;
the reporting tool does not invent them or merge different quadratures.

For the x=28 follow-up, the additional optical orders are `28:28` and
`28:32`. Earlier `(24,24)`, `(24,28)`, `(24,32)` and `(28,24)` corners
can be reused after checking that their complete projection metadata is
identical. To regenerate all six corners and both adjacent rectangles in
one invocation, use the same cap-64, degree-8, quadrature-40 export:

```bash
PYTHONPATH=$PWD python studies/rcwa/nvf_affine_convergence/nvf_affine_convergence.py --orders 24 28:24 24:28 28 24:32 28:32 --projections "$NVF_AFFINE_REF" --cells --directional-changes
```

For the next single y edge, extend the same degree-8 mesh to cap 72
with quadrature 40 and independently check quadrature 48. Validate the
shared cap-64 coefficients against the previous export, then use
`--orders 24:36 --cells --projections "$NVF_AFFINE_REF"` with cap 72.
The older `(24,32)` row is reusable only with that explicit provenance
check; this does not make the two projection metadata records identical.

For the reference-degree comparison at `(24,32)`, export degree 10 with
`--cap 64 --h .04 --quadrature 40 --refine 40` and the same transform.
Repeat with quadrature 48 as a projection control, check the shared
cap-48 coefficients against the earlier degree-10 export, and use
`--orders 24:32 --cells --projections "$NVF_AFFINE_REF"` for optics.
Run the degree-8 cap-64, quadrature-40 export at the same optical order
with the same runtime before comparing the two degrees. Keep the full
projection metadata distinct; this is a fixed-order comparison rather
than a directional rectangle.

For the candidate endpoint comparison, retain that degree-10, cap-64,
quadrature-40 fitted export and use `--orders 24:32 --cells 4:4:4:6`.
Check the candidate's common cap-48 coefficients against its earlier
export before interpreting the new optical row. The candidate and
fitted projections must each receive the `(24,32)` closure; padding
the old cap-48 coefficients with zeros does not extend the field.

The measurements and validation details are retained in
[the study log](https://github.com/fieldalgo/wavebit/blob/main/studies/rcwa/nvf_affine_convergence/records/nvf_affine_convergence.log).
Independent optical accuracy, complete oblique contact-opening derivative
qualification, and a public general-geometry adapter remain open.

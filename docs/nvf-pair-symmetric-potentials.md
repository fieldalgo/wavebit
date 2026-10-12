# Reflection-preserving pair potential spaces

Since 2026-09-28 the exported public matrices are restated about the cell centre at the
API boundary. The private construction described below is unchanged; see
[the Fourier-origin record](amplitude-sp-frame.md#8-the-fourier-space-doors-move-to-the-centre-2026-09-28).

Written: 2026-09-15

Targets:

- `src/wavebit/_core/conforming/pair_symmetric_mesh.py`
- `src/wavebit/_core/conforming/pair_corner_prefactors.py`
- `src/wavebit/_core/conforming/pair_corner_coordinates.py`
- `src/wavebit/_core/conforming/pair_contact_refinement.py`
- `src/wavebit/_core/conforming/cartesian_pair_corner_galerkin.py`
- `tests/test_nvf_pair_symmetric_mesh.py`
- `tests/test_nvf_pair_centered_origin.py`

The explicit corner family accepts `mesh='symmetric'` on a shift-zero
Cartesian opening plan. The mesh and the complete potential span preserve
the physical pair's reflection symmetries before the weak problem is solved.
The default `mesh='original'` preserves the established family. Neither
selection changes ordinary NVF dispatch or certifies optical convergence.

## The symmetry being preserved

Two horizontal rectangles with a shared height have a reflection symmetry
about their common horizontal center line, including when their widths and
materials differ. The whole periodic cell translates with `offset`.
At retained Fourier order zero, the transverse tensor and its gap derivative
must therefore have zero `xy` and `yx` components. For nonzero orders, the
same statement becomes a Fourier-index reflection, translation phase and
vector-component parity identity. It does not require every off-diagonal
Fourier matrix entry to vanish.

The original triangle mesh need not share this symmetry. Its finite trial
space can consequently have a small asymmetric response, even though its
own weak equations and energy checks pass. That error can be much larger
in a contact derivative. Enforcing a zero response entry after the solve
would not correct the underlying approximation space.

## Mesh and exact contact quotient

Each body rectangle and each side-background quadrilateral uses its center
and four edge midpoints to form eight triangles. The lower background fan
is reflected to construct the upper one. Uniform red refinement preserves
both reflections. Reflection in the horizontal coordinate exchanges the
two material labels; reflection in the vertical coordinate preserves them.

| Geometry refinement | Open triangles | Contact triangles | Collapsing triangles |
| --- | ---: | ---: | ---: |
| 0 | 60 | 48 | 12 |
| 1 | 240 | 192 | 48 |
| 2 | 960 | 768 | 192 |

The dyadic weight denominator is `4*2**refinement`. Every collapsing child
has exactly one repeated contact-endpoint pair. The mesh uses the original
checks for oriented material triangles, periodic identification, the fixed
gauge and the connected pole graph. Cubic or quartic Bernstein potentials
use this same graph. Reconstructing the polynomial base for the real trace
projection retains the selected mesh.

This is an alternative mesh family. It is not claimed to contain the entire
original mesh space. Within the symmetric family, uniform refinements retain
the coarser piecewise polynomials.

## Exact redundant coordinates

New control-hat relations can make enrichment coordinates identical or
linearly dependent. Testing only the names of the corners misses some of
these identities: distinct global hats can be identical on a triangle.

Write a raw corner potential as `h*P_k*s**(alpha-k)`, with `s=1-h`.
Multiplication of `P_k` by `s**(K-k)` puts all spatial prefactors at the same
homogeneous degree `K`. Their coefficients are integers in the triangle's
barycentric variables. A signature records the triangle, the actual integer
linear form defining `h`, and those polynomial coefficients. Fraction-free
integer elimination then retains an independent set of signatures.

No material, sampled rank or numerical tolerance selects a field. Every
omitted prefactor has an exact polynomial identity in the retained span.
Subtracting the radial Taylor terms changes these identities only by a
continuous piecewise polynomial already represented by the nodal `Pp`
space. The complete enriched potential space is therefore preserved.

With two radial exponents and junctions enabled, angular orders zero, one
and two retain 168, 260 and 356 coordinates from 192, 308 and 424 candidates.
The original mesh retains every one of its 112, 168 and 224 coordinates.
The symmetric extension at angular orders three and four retains 448 and
544 coordinates. Exact integer-span checks retain every lower-order prefactor;
the increase resolves angular dependence without deleting a radial direction.

## Stable radial coordinates

The four-exponent symmetric space also needs stable coordinates before its
physical weak matrices are assembled. Orthogonalizing only the later solve
does not recover digits already lost in those matrices. A degree-three,
order-zero metal contact control had a common-coordinate condition of
`2.53e8` and an original contact-derivative residual of `4.20e-8`, above the
unchanged `1e-10` limit, despite passing its independent physical-energy
checks.

For every nonempty symmetric radial subset, a real nodal interpolation subtraction first
removes an element of the same complete common `Pp` space. Each common node
has one shared Bernstein coefficient, including at contact. Reference-field
scales and a complete per-corner QR then define invertible real coordinates
before any material integration. The preliminary norm only chooses units
for the coordinates; it is not an energy check or a field-selection rule.
The QR uses at least `root_order*max(p,4)` reference integration points.

Values, spatial gradients and moving-field gradients use the same fixed
coordinate change. These coordinates depend on the mesh and field choice,
not on materials or physical length units. Every coordinate survives.
The original weak, retained-field, flux, reciprocity, passivity and independent
energy checks are applied to the resulting complete equations.

The original two-exponent subset needs this preparation as well. In a
degree-three, Fourier-order-four, angular-order-two metal case at normalized
gap `0.003`, the unrefined symmetric mesh had a common-coordinate condition
of `5.88e6` and a common coefficient norm of `1.78e6`. Full-state pointwise
refinement alone left energy and energy-derivative residuals of `2.67e-10`
and `6.91e-10`, both above the original `1e-10` guard.

Preparing the same two-exponent space lowers the condition to `6.28e3`
and the coefficient norm to `3.44e3`. At integration order 64, the complete
calculation with pointwise refinement and ordinary constraint-residual
arithmetic passes every original physical guard:
its energy and energy-derivative residuals are `1.02e-11` and `3.08e-11`.
The value reflection discrepancy is `3.83e-11`, below `3e-10`. The complete
gap-derivative reflection discrepancy is still `1.46e-8`, above `3e-9`;
that earlier high-order derivative was therefore not reflection-qualified. These
measurements use unit period and otherwise the normalized metal geometry
given below. The preparation changes no radial exponent, quadrature rule,
potential count or original-mesh default. The subsequent
[accurate retained/flux residual evaluator](nvf-pair-corner-potentials.md)
addresses additional arithmetic residue amplified by large coefficients;
the measurements above isolate the coordinate preparation itself.

The earlier full-space regression at integration order 48 also passed every
original physical guard. It took about 12 minutes 23 seconds on the stated
runtime. The centered-origin regression described below now checks the
complete high-order reflection identity as well.

The coordinate recovery tests undo all real transforms and add independently
evaluated Bernstein interpolation values and gradients. They recover every
original potential and its spatial and moving-field derivatives on all 60
triangles. Controls cover degrees three and four with the original one- and
two-exponent subsets at angular orders zero through two, the two-exponent
extension at orders three and four, and cubic and quartic four-exponent cases.
The quartic four-exponent, angular-order-four reference inverts its full
triangular coordinate map independently with 80-digit decimal arithmetic:
an ordinary double-precision inverse solve can amplify its own rounding in
the original coordinates. This changes only the reference calculation; the
sampled fields and recovery tolerances stay fixed. The full and gradient-only
samplers agree bitwise.

Assembly requests gradients directly, avoiding dense coordinate transforms
of potential values that the integrals do not use. A bounded cache stores
only fixed interpolation-gradient coefficients for up to 128 triangles;
eviction recomputes the same coefficients. The full value sampler remains
available. Across all 60 unrefined triangles, values, gradients and moving
gradients are bitwise identical to the former full-value arithmetic, including
repeated calls with changing integration points and geometry tangents.

Each configured corner cell keeps the one field space prepared during its
construction. Both contact and positive assembly reuse those coordinates,
including the shared interpolation and reference QR. Preparation depends on
the static mesh and field choices; physical gaps, length units, materials and
their directions remain live inputs to every weak integral. A supplied field
space must match its owning mesh and all field options. Standalone assembly
without a prepared space constructs one as before.

Coordinate transforms are applied only to corner groups whose defining hat
is present on the triangle. An absent hat and its nodal interpolant vanish
identically there, including their spatial and moving gradients. Every
column remains in the complete returned space; no magnitude or numerical
rank determines this exact support check.

The retained storage is one configured geometry space per cell, rather than a
cache indexed by physical settings. Its gradient-coefficient cache retains the
same 128-triangle bound. Tests prevent re-preparation during a live contact /
positive sweep and check exact simultaneous material scaling, unit scaling and
gap/material derivative identities after those inputs change.

In that same contact control, the common-coordinate condition dropped to
`1.06e4`. The original contact-derivative residual became `6.48e-11`, and
the independent energy-derivative residual was `9.88e-15`. These measurements
use NumPy 2.5.1 with Apple Accelerate on macOS arm64. They qualify this finite
calculation, not all resolution or material choices.

Increasing the contact integration order to 96 exposed a remaining original
variation residual of `1.24e-10`. The full tensor and its derivative changed
by only `5.68e-14` and `2.36e-12`, respectively. The expanded symmetric contact
solve therefore also refines the original weak, retained and flux derivative
equations together, using the same fixed real metric as a preconditioner.
The independent stationary-energy verifier does not enter this correction.
All original checks remain in force after it.

The final original contact-derivative residuals at integration orders 80
and 96 are `1.07e-12` and `7.77e-13`; both complete responses pass all original
checks. The derivative difference between the orders is `1.22e-11`.

The symmetric Cartesian chart also computes its diagonal area directly,
before taking the normalization length. A generic logarithmic determinant
returned `699.9999999999998` for the length of the period-700 square. This
avoidable rounding changed the normalized geometry under a unit conversion:
the complete gap-derivative comparison missed its `3e-9` criterion at
`3.83e-9`. The direct diagonal product gives exactly 700, makes the normalized
contact value bitwise identical to the unit-period result, and reduces that
derivative discrepancy to `5.09e-11`. The same normalization applies at zero
and positive gap. The original mesh retains its existing normalization.

## Centering the physical coordinates

The pair and its fundamental cell translate together. Changing their common
origin therefore preserves the entire finite trial space, including every
nodal, corner, disappearing and continuous-trace field. Evaluating the mesh
at the cell center avoids repeatedly subtracting a common translation from
nearby physical vertices and Fourier phases. Large potential derivatives can
amplify the resulting rounding even when the original equations pass.

The symmetric family validates the caller's original geometry, solves at
`offset = -basis @ (1/2, 1/2)`, then restores the caller's Fourier origin.
For an integer Fourier mode `m` and fractional caller offset
`o = solve(basis, offset)`, define

```text
P[m] = (-1)**(m_x + m_y) * exp(-2*pi*i*m.o)
H_caller = P @ H_center @ P.conj().T
DH_caller = P @ DH_center @ P.conj().T
```

Here `P` is diagonal. The half-cell factor is evaluated as an exact sign.
Every complete transverse and longitudinal slot uses this same invertible
translation. Its phase is fixed in the supported gap/material JVP because
the basis and offset have no tangent in this API. Physical guards are applied
before and after translation; their thresholds and the potential span remain
unchanged. The original mesh retains its existing origin and arithmetic.

In the unit-period, degree-three, unrefined symmetric metal fixture at gap
`.003`, Fourier order four, angular order two and radial/angular integration
orders 64/24, choosing caller offset zero still left a maximum gap-derivative
reflection defect of `3.08e-9`, just outside the existing `3e-9` limit. Placing
the **cell center** at zero reduced it to `1.08e-9`. Translating the complete
result back to caller offset `(.01,-.03)` preserved that bound. The maximum
value defect was `4.42e-12`, below `3e-10`, and independent energy and
energy-derivative residuals were `5.33e-12` and `6.59e-12`, below `1e-10`.
This resolves that finite high-order regression; it does not establish optical
or mesh convergence.

Separate translated physical assemblies check complete values and joint
gap/material JVPs on a non-square cell with basis `diag(1.2,.9)` and offset
`(.17,-.08)`. Controls include unequal Fourier orders, cubic and quartic
potentials, empty and expanded real radial subsets, and the complete
oscillatory space. They retain absolute `3e-10` / `3e-9` comparison criteria.
The oscillatory positive-gap comparison passed in 1,277 seconds for its two
complete solves, exceeding the 1,200-second per-test CI limit. Its CI
counterpart uses exact contact; the separate positive calculation keeps every
field and its joint derivative.

Centering alone did not qualify every finite space. On that non-square cell,
the four-real-exponent metal contact at orders `(1,1)`, cubic degree and
radial/angular integration orders 80/12 had a joint-derivative reflection
defect of `2.13e-7` despite passing the original relative physical guards.
Increasing the angular integration order to 24 did not resolve it.

Symmetric contacts with real root-20 fields now assemble their complete weak
equations in compensated coordinates. Exact radial/angular coefficients and
the affine unit-simplex relation preserve the combined polynomial/corner
potential. Each triangle contributes every function with nonzero topological
support to high/low common, disappearing and trace matrices. The entire real
common transform is then applied to those matrices and their Fourier moments,
before forming the closed projection and its full variation. This avoids
evaluating dense global coordinates at every quadrature point. All global
columns survive the integration and coordinate change.

The value, contact derivative and joint material derivative come from that
same complete system. Every linear-solve column receives compensated residual
corrections. The result still passes the original nodal, retained, flux and
independent physical-energy checks; the original derivative-only correction
is not subsequently applied to the less accurate rounded assembly. Metadata
identifies this path as `common_coordinate_precision='compensated'`.
Original-mesh, real fifth-root, oscillatory and positive-gap responses keep
their respective evaluation paths.

The unprojected common and normal matrices are retained before forming the
closed equations or solving a response. Restoring the Fourier terms removed
from the normal test functions supplies the same original weak, retained and
flux equations at the improved assembly precision. These checks retain their
original tolerances. The independent pointwise energy verifier still
reconstructs the physical fields from the potential coefficients.

This distinction matters even at Fourier order zero. In the period-700,
four-exponent contact with radial/angular integration orders 80/12, a
compensated state compared against the older rounded matrices had a contact
variation residual of `2.33e-9`, above the original `1e-10` limit. Two terms
of norm about `.590` almost cancel. Directly integrating the same physical
matrices reduced that residual to `1.24e-12`; the independent energy and
energy-variation residuals remained below `4.2e-14` and `9.9e-15`.

The singular integration map also preserves high/low components. Gaussian
nodes and weights are computed from the Legendre recurrence with 90-digit
standard-library arithmetic, then used in the same twentieth-power radial
substitution. The affine partition, weights and complete Fourier phases keep
both components during assembly. Exact polynomial moments through degree
`2q-1`, triangle moments and an independent Decimal phase/variation reference
check this arithmetic. The finite space, integration orders and physical
criteria do not change. Cached scalar rules are bounded to 16 orders; local
field integration uses batches of at most 512 points.

The non-square four-exponent control uses widths `(.3,.4)`, height `.45`,
materials `(12.25,-10+i)`, background 1, gap direction `.7`, material directions
`(.2+.1i,-.4+.2i)` and background direction `.1-.03i`. Its full public response
with compensated common equations passed every original physical guard and
the unchanged `3e-10` / `3e-9` reflection criteria. Maximum value and derivative
defects were `5.33e-15` and `1.78e-9`; independent energy and energy-derivative
residuals were `1.56e-13` and `1.77e-13`. This resolves that finite joint metal
derivative regression; it does not establish optical convergence.

Compensated fields and matrices alone did not qualify angular integration:
the full joint derivative changed by `5.06e-9` between angular orders 12 and
24, and by `3.13e-9` between 24 and 32. Preserving the integration coordinates,
weights and phases reduced those differences to `1.93e-9` and `2.30e-13`,
respectively, within the unchanged `3e-9` criterion. The corresponding full
value differences were `1.06e-12` and `1.19e-25`, below `3e-10`. These are
complete finite-space integration comparisons, not optical convergence bounds.

The default radial/angular 80/80 rule also passed the complete material-scale
and length-unit identities with all four real exponents. Its two physical
solves took 1,058 seconds, close to the 1,200-second CI limit for one test.
Repeated representation checks therefore retain all those fields and use
angular order 12, whose complete 12/24 comparison is checked separately.
The radial 80/96 comparison, reversed exponent order and length-unit change
keep their original value/variation tolerances. The solver's default
integration orders are unchanged.

## Finite controls and limits

The metal fixture uses period 700, widths `(210,280)`, height 315, offset
`(7,-21)`, materials `(12.25,-10+1j)` and background 1. Degree-three,
order-zero controls with the original two exponents preserve reflection
at contact and gap 2.1 while passing the original physical guards. The
`xy`/`yx` value errors are below `1e-14`; their gap derivatives are below
`9e-15` per caller length unit. Degree-four contact and degree-three
angular-order-two contact with retained Fourier order one also pass their
full reflection checks.

The same positive-gap metal fixture also checks the full analytical gap
derivative against independent fourth-order physical-gap differences:

| Stencil step in caller length units | Maximum complete derivative discrepancy |
| --- | ---: |
| `0.04` | `1.2575e-7` |
| `0.01` | `4.9158e-10` |
| `0.005` | `3.0205e-11` |

The two smaller steps pass the unchanged `3e-9` criterion. Their discrepancy
ratio is about 16.3, consistent with fourth-order stencil truncation. Every
sampled state passes the original physical guards; no response entries are
symmetrized or replaced for this comparison.

With all four real radial exponents, the same fixture also passes every
original guard at gap `1e-7` using integration order 80. With the final
diagonal normalization, its tensor and gap-JVP norm differences from contact
are `7.9695e-9` and `7.8062e-6`, respectively. Its independent energy-derivative
residual is `3.88e-15`, and its original weak-derivative residual is `2.19e-12`.
The complete positive-gap response preserves reflection. These are finite
contact-approach measurements, not bounds on optical or continuum error.

That complete four-exponent positive-gap control took about 41 minutes on
the stated runtime, exceeding the 20-minute limit for one CI test. The
repeated subpixel-gap regression therefore uses the most singular allowed
exponent, `(0.55,)`, while keeping the same root-20 integration, full physical
guards and reflection criteria. It passed in about 11 minutes, with energy
and energy-derivative residuals of `3.96e-15` and `1.04e-15`. The four-exponent
contact representation and material-direction tests remain in the suite.
This bounds the cost of that regression fixture; the solver does not remove
fields or change the requested radial subset.

The symmetric mesh is an explicit resolution choice. It corrects a known
finite-space symmetry defect, but the mesh, angular, radial and retained
Fourier orders must still be qualified for each optical observable. In
particular, a small tensor increment alone does not certify a contact
transmission slope.

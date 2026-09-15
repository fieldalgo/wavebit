# Reflection-preserving pair potential spaces

Written: 2026-09-15

Targets:

- `wavebit/_core/conforming/pair_symmetric_mesh.py`
- `wavebit/_core/conforming/pair_corner_prefactors.py`
- `wavebit/_core/conforming/pair_corner_coordinates.py`
- `wavebit/_core/conforming/pair_contact_refinement.py`
- `wavebit/_core/conforming/cartesian_pair_corner_galerkin.py`
- `tests/test_nvf_pair_symmetric_mesh.py`

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

## Stable expanded radial coordinates

The four-exponent symmetric space also needs stable coordinates before its
physical weak matrices are assembled. Orthogonalizing only the later solve
does not recover digits already lost in those matrices. A degree-three,
order-zero metal contact control had a common-coordinate condition of
`2.53e8` and an original contact-derivative residual of `4.20e-8`, above the
unchanged `1e-10` limit, despite passing its independent physical-energy
checks.

For expanded radial subsets, a real nodal interpolation subtraction first
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

Assembly requests gradients directly, avoiding dense coordinate transforms
of potential values that the integrals do not use. A bounded cache stores
only fixed interpolation-gradient coefficients for up to 128 triangles;
eviction recomputes the same coefficients. The full value sampler remains
available. Across all 60 unrefined triangles, values, gradients and moving
gradients are bitwise identical to the former full-value arithmetic, including
repeated calls with changing integration points and geometry tangents.

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

The symmetric mesh is an explicit resolution choice. It corrects a known
finite-space symmetry defect, but the mesh, angular, radial and retained
Fourier orders must still be qualified for each optical observable. In
particular, a small tensor increment alone does not certify a contact
transmission slope.

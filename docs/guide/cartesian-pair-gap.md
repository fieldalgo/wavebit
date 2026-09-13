# A finite pair across contact and positive gaps

The experimental NumPy-only `CartesianPairGapCell` evaluates one finite pair
at an explicit nonnegative physical gap, including exact contact. It returns
coefficients or a selected energy closure together with their analytical gap
derivative. Import it from `wavebit._core.conforming.cartesian_pair_gap` and
supply the existing `CartesianPairOpeningPlan`.

```python
import numpy as np
from wavebit._core.conforming.cartesian_pair_gap import CartesianPairGapCell
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan

geometry = dict(
    basis=np.array([[700.0, 315.0], [0.0, 490.0]]),
    widths=(140.0, 193.0),
    height=210.0,
    offset=(49.0, -21.0),
)
plan = CartesianPairOpeningPlan.from_geometry(**geometry, refinement=1)
cell = CartesianPairGapCell(plan, order=(1, 2))
settings = dict(geometry, epsilon=(4.0 + 0.8j, 9.0 + 0.3j), background=1.0 + 0.1j)
length = np.sqrt(np.linalg.det(geometry["basis"]))

for gap in (0.0, length / 64):
    coefficients, derivative, metadata, details = cell.coefficients(gap=gap, **settings)
    slots, slot_derivative, closure_metadata, _ = cell.closure(gap=gap, **settings)
    assert coefficients.shape == derivative.shape == (9, 9, 5)
    assert all(slot.shape == (1, 15, 15) for slot in slots)
    assert all(slot.shape == (1, 15, 15) for slot in slot_derivative)
    assert metadata["derivative_units"] == "per caller length unit"
```

## The physical family

All supplied lengths use one caller unit, such as nanometers. `basis` is a
finite real 2-by-2 matrix with lattice vectors in its columns and finite
positive determinant. `widths=(left_width, right_width)` are positive; one
positive `height` is shared by the horizontal rectangles. `epsilon` is the
corresponding left/right pair. The rectangles keep physical Cartesian edges.

Let `center = basis @ (.5, .5) + offset`. Their centers have that y coordinate
and x coordinates `center.x - (right_width + gap)/2` and
`center.x + (left_width + gap)/2`. Changing gap holds the outer cell fixed and
moves the left rectangle by half the gap change to the left, and the right
rectangle by half to the right. `offset` translates the whole integration cell.

`gap` is a finite real scalar at least zero. At zero, `derivative` is the
right derivative. At positive gap, it is the local derivative with every
other setting held fixed. Both are returned **per caller length unit**.
This explicit family does not change the contact-preserving JAX contract of
[`CartesianRectanglePairNVF`](cartesian-rectangle-pair-nvf.md). There is no
`xp`, JAX or automatic differentiation mode in this interface.

## Plan and admission

Create the concrete plan at contact with `CartesianPairOpeningPlan.from_geometry`.
It selects one of four cyclic charts; `shift=0`, `1`, `2` or `3` requests a
specific chart. `refinement` defaults to 0 and accepts integers 0 through 3.
The [refinement note](../nvf-pair-refinement.md) records independent checks,
resource costs and optical limitations of the finer spaces. The plan
stores only those immutable choices and the cell reuses its fixed topology.
All physical geometry and materials are supplied when evaluating the cell.

Every evaluation checks contact-chart validity, strict envelope containment,
positive regular triangles at the requested gap and positive opening slopes
of collapsed triangles. Replanning is explicit if the chart becomes invalid.
A gap that fits inside the parallelogram can still fail the triangle guards.
A valid geometric interval is not a uniform conditioning or accuracy guarantee.

By default, `epsilon` and `background` require finite strictly positive real
parts and nonnegative imaginary parts. An explicit constructor
`coercivity_angle` also admits passive metals in a common certified half-plane;
see [material sectors](../nvf-pair-material-sector.md) for conditions and optical limits.
Complex geometry is refused even when its
imaginary part is zero. Negative, boolean, complex or nonfinite gap values,
invalid geometry/materials, unsupported orders and extra keywords raise.
Unequal heights, vertical shifts, rotated rectangles, overlap, multiple pairs
and tensor materials are outside this family.

## Returned coefficients and diagnostics

`order` defaults to 1 and accepts a nonnegative integer or `(x, y)` pair,
including zero. The nine fields are `Bxx, Bxy, Byx, Byy, Gxx, Gxy, Gyy,
epsilon, Gyx`. Shape is `(9, 4*y+1, 4*x+1)`, with y and x labels spanning
the required coefficient differences. The phase is
`exp(-2*pi*1j * [m,n] @ inv(basis) @ position)`, normalized by physical cell area.

Both methods return `(value, derivative, metadata, details)`. Closure value
and derivative are five-slot `ConvMatrices` tuples with one layer axis.
Requested differences are selected before inverses. Closure recomputes the
coefficients from its current settings and accepts no cached override.

Metadata records physical gap, `normalized_gap=gap/sqrt(det(basis))`, area,
normalization length, order, chart, counts, original and projected residuals,
and conditioning. Closure also reports storage, signed dissipation and inverse
residual checks. Details expose the normalized matrix and its derivative,
normalized vertices, fields and Fourier weights; suffix `_ds` means derivative
with respect to normalized gap. These diagnostic arrays do not use physical
length units. Returned coefficient and closure derivatives do.

The [method note](../nvf-cartesian-pair-gap.md) describes the exact finite
rescaling and its independent qualification. It gives no continuum or patterned
optical accuracy certificate. The separate [endpoint response](cartesian-pair-opening.md)
remains available as an independently qualified reference.

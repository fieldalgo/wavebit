# A finite right-opening response for two rectangles

The experimental `CartesianPairOpeningPlan` and `CartesianPairOpeningCell`
provide a separate NumPy-only contact value and analytical right derivative
as a physical gap opens between two horizontal equal-height rectangles.
Import them from `wavebit.rcwa._engine.conforming.cartesian_pair_opening`; the plan is
also exported by `wavebit.rcwa.experimental`. This interface does not alter the
contact-preserving JAX contract of
[`CartesianRectanglePairNVF`](cartesian-rectangle-pair-nvf.md).

```python
import numpy as np
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import (
    CartesianPairOpeningCell,
    CartesianPairOpeningPlan,
)

geometry = dict(
    basis=np.array([[700.0, 315.0], [0.0, 490.0]]),
    widths=(140.0, 193.0),
    height=210.0,
    offset=(49.0, -21.0),
)
plan = CartesianPairOpeningPlan.from_geometry(**geometry, refinement=1)
cell = CartesianPairOpeningCell(plan, order=(1, 2))
settings = dict(geometry, epsilon=(4.0 + 0.8j, 9.0 + 0.3j), background=1.0 + 0.1j)
coefficients, derivative, metadata, details = cell.coefficients(**settings)
slots, slot_derivative, closure_metadata, closure_details = cell.closure(**settings)

assert coefficients.shape == derivative.shape == (9, 9, 5)
assert all(slot.shape == (1, 15, 15) for slot in slots)
assert all(slot.shape == (1, 15, 15) for slot in slot_derivative)
```

## Geometry and plan

All lengths must use the same physical unit; the example can be read in
nanometers. `basis` is a finite real 2-by-2 matrix whose columns are lattice
vectors, with finite positive determinant. The two positive `widths` and
`epsilon` values are left/right pairs. One positive `height` is shared.
`offset` is a Cartesian displacement from `basis @ (.5, .5)` and translates
the whole integration cell. The rectangles retain physical Cartesian edges.

The response is evaluated at exact contact. Gap g is the derivative variable,
not an input: the left rectangle moves by `(-g/2, 0)` and the right rectangle
by `(g/2, 0)`, holding every supplied setting fixed. The right derivative is
returned **per caller length unit**. It is not an ordinary unconstrained
geometry JVP and there is no `xp` or JAX mode.

`from_geometry` selects one of four positive cyclic mesh charts. `shift=0`,
`1`, `2` or `3` requests a concrete chart explicitly. `refinement` is 0 by
default and accepts integers 0 through 3. See the [refinement controls](../../validation/rcwa/nvf-pair-refinement.md)
for the enlarged finite scope and optical limits. The frozen plan stores only shift and
refinement. Replay uses the current physical settings and requires strict
containment, positive regular triangles and positive opening-area slopes.
Replanning is explicit when a different chart is needed.

## Coefficients and closure

`order` accepts a nonnegative integer or `(x, y)` pair, including zero. The
nine coefficient fields are ordered as `Bxx, Bxy, Byx, Byy, Gxx, Gxy, Gyy,
epsilon, Gyx`. Their array shape is `(9, 4*y+1, 4*x+1)`; axes are component,
y label and x label. Labels run from `-2*y` to `2*y` and `-2*x` to `2*x`.
The Fourier phase is `exp(-2*pi*1j * [m,n] @ inv(basis) @ position)` with
physical area normalization.

Both methods return `(value, derivative, metadata, details)`. The closure's
value and derivative are each a `ConvMatrices` tuple in the usual five-slot
order, with one layer axis. Closure selects the requested field differences
before inversion and always computes coefficients from its current settings;
it accepts no cached-coefficient override. Metadata includes original weak
residuals, coercivity and conditioning information, counts and derivative
units. Closure adds storage, signed dissipation and inverse residual checks.
Details contain diagnostic fields, transforms and original Laurent operators;
no continuation or optical error bound is implied by them.

`epsilon` and `background` must be finite numeric scalar permittivities with
strictly positive real parts and nonnegative imaginary parts. Geometry is
real numeric data; complex geometry is refused even when its imaginary part
is zero. Invalid geometry/materials, unsupported orders, refinements and
extra keywords raise. Exact contact, one common height and horizontal
rectangles are intrinsic to this interface; angle, vertical shifts, overlap,
multiple pairs and tensors are not supported.

The independent finite-space qualification and the weak-limit derivation are
in [the method note](../../validation/rcwa/nvf-cartesian-pair-opening.md). Neither the finite
coefficients nor their right derivative certifies patterned optical accuracy.

The optional constructor `coercivity_angle` also admits passive metals in a
common complex half-plane. The certificate affects admission and diagnostics,
while the physical equations retain the original materials. See
[material sectors](../../validation/rcwa/nvf-pair-material-sector.md) for a runnable example,
independent metal gap checks and the unresolved optical limits.

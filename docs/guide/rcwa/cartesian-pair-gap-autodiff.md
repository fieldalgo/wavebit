# Differentiating the finite pair gap

:::{note}
This detailed guide documents the retained `wavebit.compat.legacy` interface and its measured scope. For the current problem/numerics lifecycle, start with [the common guide](../common/lifecycle.md). Advanced capabilities are preserved here with explicit compatibility calls; their limits are unchanged.
:::

The experimental `CartesianPairGapEvaluator` evaluates the exact finite pair
family using NumPy or JAX x64. It computes values only; ordinary JAX
operations provide the gap derivative, including the physical right derivative
at exact contact. Import the evaluator from its submodule and supply a concrete
`CartesianPairOpeningPlan` before tracing.

```python
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from wavebit.rcwa._engine.conforming.cartesian_pair_gap_autodiff import CartesianPairGapEvaluator
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan

geometry = dict(
    basis=np.array([[700.0, 315.0], [0.0, 490.0]]),
    widths=(140.0, 193.0),
    height=210.0,
    offset=(49.0, -21.0),
)
plan = CartesianPairOpeningPlan.from_geometry(**geometry, refinement=1)
cell = CartesianPairGapEvaluator(plan, order=(1, 2))
settings = dict(geometry, epsilon=(4.0 + 0.8j, 9.0 + 0.3j), background=1.0 + 0.1j)

@jax.jit
def response(gap):
    coefficients, slots, _, _ = cell.evaluate(gap=gap, **settings, xp=jnp)
    return coefficients, jnp.stack(slots)

length = np.sqrt(np.linalg.det(geometry["basis"]))
for gap in (0.0, length / 64):
    values, derivatives = jax.jvp(
        response, (jnp.asarray(gap),), (jnp.asarray(1.0),)
    )
    assert values[0].shape == derivatives[0].shape == (9, 9, 5)
    assert values[1].shape == derivatives[1].shape == (5, 1, 15, 15)
    assert all(np.isfinite(np.asarray(value)).all() for value in (*values, *derivatives))
```

JAX is optional. `cell.evaluate(gap=gap, **settings)` uses NumPy and returns
values without importing JAX. The separate
[`CartesianPairGapCell`](cartesian-pair-gap.md) remains the NumPy interface
that also returns an analytical derivative.

## Geometry and derivative meaning

All lengths use one caller unit. `basis` is a finite real 2-by-2 matrix whose
columns are lattice vectors, with finite positive determinant. `widths` gives
positive left/right widths; the horizontal rectangles share one positive
`height`. Their physical Cartesian edges are preserved on the oblique lattice.

With `center = basis @ (.5, .5) + offset`, their centers are
`center + (-(right_width + gap)/2, 0)` and
`center + ((left_width + gap)/2, 0)`. Increasing physical gap moves each rectangle
outward by half that change while holding the outer cell fixed. `offset`
translates the whole integration cell. `epsilon` gives the corresponding
left/right materials; by default it and `background` require finite positive
real parts and nonnegative imaginary parts. An explicit constructor
`coercivity_angle` also admits passive metals in a common certified half-plane;
see [material sectors](../../validation/rcwa/nvf-pair-material-sector.md). The angle is static
before tracing, and does not alter the physical equations or material values.

The evaluator accepts finite real `gap >= 0`. At zero, a positive gap tangent
is the physical right derivative. A negative tangent at zero is only the
formal local algebraic extension; negative-gap evaluations are refused.
A unit gap tangent returns derivatives **per caller length unit**. Internal
normalization by `sqrt(det(basis))` does not change that meaning.

This increment qualifies gap derivatives only. Geometry, pose and materials
remain live arguments and can be replayed on the fixed chart; their arbitrary
gradients, higher derivatives and mixed derivatives are not qualified here.
The ordinary [`CartesianRectanglePairNVF`](cartesian-rectangle-pair-nvf.md)
continues to require contact-preserving tangents on its contact chart.

## Fixed chart and returned values

`CartesianPairOpeningPlan.from_geometry` plans concrete contact geometry, with
`refinement` from 0 through 3 and an optional cyclic `shift` from 0 through 3.
The [refinement note](../../validation/rcwa/nvf-pair-refinement.md) covers the larger finite spaces,
JAX gap controls, reduced assembly storage and unresolved optical limits.
The evaluator stores that immutable topology and its nonnegative Fourier
half-widths, supplied as an integer or `(x, y)` tuple. Zero widths are allowed
for Fourier orders. There is no dynamic planning during tracing, physical
coefficient cache, or automatic chart change.

Every replay checks strict envelope containment, positive regular triangle
areas and positive collapsed-triangle opening slopes. A contained rectangle
pair can still fail its chart's triangle guards. Unsupported geometry includes
unequal heights, vertical shifts, rotation, overlap, extra pairs and tensors.
Boolean or complex geometry is rejected even if a complex imaginary part is
zero. Unsupported namespaces, x64-disabled JAX, incompatible shapes and extra
keywords are refused. Invalid numerical values raise eagerly; under JIT the
coefficient and closure outputs become all NaN after safe intermediate
calculations. Invalid-domain derivatives have no validity guarantee.

`evaluate` returns `(coefficients, slots, metadata, details)`. Coefficients
have shape `(9, 4*y+1, 4*x+1)` and field order `Bxx, Bxy, Byx, Byy, Gxx, Gxy,
Gyy, epsilon, Gyx`. The phase is
`exp(-2*pi*1j * [m,n] @ inv(basis) @ position)` with physical area normalization.
The five `ConvMatrices` slots each have shape `(1, H, H)`, where
`H=(2*x+1)*(2*y+1)`. The exact requested differences are selected before
inverses, including unequal and zero orders.

Metadata contains physical/normalized gap, normalization length, space counts,
projected and original weak residuals, diagonal-block energy checks, condition
number, and closure storage, signed dissipation and inverse checks. Details
contain the normalized matrix, load, solution, vertices, fields, Fourier
weights and original weak flux/load. These diagnostics are values rather than
an additional analytical derivative API. Use a function returning the outputs
needed for differentiation, as in the example.

The [method and qualification note](../../validation/rcwa/nvf-cartesian-pair-gap-autodiff.md)
records independent references, resource observations and limits. This finite
constitutive evaluation provides no continuum or patterned optical accuracy
certificate.

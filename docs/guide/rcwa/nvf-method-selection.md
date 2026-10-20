# Selecting NVF and the auxiliary solver

Select the matrix construction explicitly. Ordinary NVF does not call the
experimental transmission cell, and a failed auxiliary calculation does not
fall back to another construction. The auxiliary cell supplies a different
constitutive matrix; it is not a correction appended to an ordinary NVF solve.

| Choice | How to select it | Scope |
| --- | --- | --- |
| Ordinary NVF; auxiliary off | `rcwa.run(problem, numerics=rcwa.Numerics(factorization='nvf'))` | The existing raster or supported shape NVF path and its selected projector. |
| NVF off; explicit Cartesian Li | `rcwa.run(problem, numerics=rcwa.Numerics(factorization='li'))` | Existing raster support or admitted axis-aligned rectangle shapes. This is a user choice, never an automatic NVF fallback. |
| Experimental auxiliary construction | Build a `rcwa.experimental.CartesianPairTransmissionGalerkinGapCell`, put its matrices in place of the operators from `rcwa.operators.assemble`, then call `rcwa.operators.solve` | A NumPy finite cell for one horizontal, equal-height rectangle pair, with an analytical gap/material JVP. |

The third choice is experimental. It retains all internal physical checks,
but optical convergence and practical runtime remain unresolved. Its supported
geometry and derivatives are narrower than those of ordinary NVF. A selected
degree or mesh does not certify accuracy; see the
[measurement and qualification note](../../validation/rcwa/nvf-pair-transmission-potentials.md).

## An explicit auxiliary call

```python
import numpy as np
from wavebit import Material, Units, rcwa
from wavebit.rcwa import experimental, operators

basis = np.diag([700., 700.])
orders = (1, 1)
cell = experimental.CartesianPairTransmissionGalerkinGapCell(
    experimental.CartesianPairOpeningPlan(0, refinement=0), orders,
    reference=(12.25, -10.+1j, 1.), degree=3, branches=2,
    quadrature=80, angular_quadrature=48, coercivity_angle=1.53,
)
matrices, gap_jvp, diagnostics = cell.closure(
    gap=0., basis=basis, widths=(300., 300.), height=300.,
    epsilon=(12.25, -10.+1j), background=1., gap_tangent=1.,
)
problem = rcwa.Problem(
    units=Units(length='nm'),
    stack=rcwa.Stack(
        lattice=basis, top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
        # A background-only layer fixes the thickness; the matrices carry the pair.
        layers=(rcwa.Layer(thickness=300., background=Material(epsilon=1.)),)),
    excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(
        side='top', polarization='s', theta=.2, phi=0.),)))
assembled = operators.assemble(problem, numerics=rcwa.Numerics(orders=orders),
                               backend='numpy').replace(epsilon=(matrices,))
answer = operators.solve(problem, assembled)
```

The example's low Fourier and polynomial orders demonstrate the call, not
converged optics. `gap_jvp` is the derivative of the constitutive matrices,
not a transmission or reflection derivative. `operators.solve` consumes the
supplied matrices directly: the factorization in the assembly's
`rcwa.Numerics` does not rebuild or disable them. To turn the auxiliary
construction off, declare the pair as layer regions and run the problem with
`rcwa.run` and an explicit factorization instead.

The cell accepts a positive diagonal lattice, strictly contained equal-height
horizontal rectangles, nonnegative gap and admitted passive scalar materials.
Unequal heights, arbitrary shapes, overlap, multiple pairs and tensor materials
are outside this auxiliary family. Cell construction is NumPy-only; there is
no JAX tracing or general shape-gradient interface for this family. The
[pair guide](cartesian-pair-gap.md#fixed-reference-transmission-potentials)
defines the reference materials, supported JVP directions, degree, quadrature
and resource bounds. Its runtime checks remain active when explicitly selected.

## Switching the same example

The runnable [selection example](../../../examples/rcwa/nvf_method_selection.py) offers
an explicit method argument and prints the selected method before calculating:

```shell
python examples/rcwa/nvf_method_selection.py --method nvf
python examples/rcwa/nvf_method_selection.py --method li
python examples/rcwa/nvf_method_selection.py --method auxiliary --degree 3
```

These are example-script options, not new keywords on `rcwa.Numerics`;
the script uses the current physical Problem API and explicit `rcwa.operators` for the auxiliary matrices.
The script defaults to ordinary circular NVF. Degree and mesh-refinement
arguments are accepted only for the auxiliary choice; it refuses ignored
settings or unknown methods. All three runs use the same physical pair and
optical settings. The printed elapsed time includes construction and the
selected path's internal checks. A single low-order run is neither a matched
accuracy benchmark nor a certificate for contact optics.

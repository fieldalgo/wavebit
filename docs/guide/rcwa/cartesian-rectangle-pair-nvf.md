# Two Cartesian rectangles on an oblique lattice

`rcwa.experimental.CartesianRectanglePairNVF` builds finite NVF energy
matrices for two horizontal, equal-height physical rectangles per patterned
layer. It supports a positive gap or exact full-height contact on a required
frozen plan. The rectangles retain their Cartesian edges on an oblique
lattice. This explicit experimental builder supplies `ConvMatrices` for
`rcwa.operators.solve`; it does not certify optical convergence or change
the other geometry builders.

```python
import numpy as np
from wavebit import Material, Units, rcwa
from wavebit.rcwa import experimental, operators

lattice = np.array([[700., 315.], [0., 490.]])
structure = [dict(background=1., shapes=[
    dict(lx=140., ly=210., cx=-56., cy=-21., material=4.),
    dict(lx=193., ly=210., cx=127.5, cy=-21., material=9.),
])]
builder = experimental.CartesianRectanglePairNVF((1, 2), refinement=1)
plan = builder.plan(structure, lattice)
conv = builder.conv_matrices(structure, lattice, plan=plan)
numerics = rcwa.Numerics(orders=builder.orders)
transmitted = []
for polarization in ('p', 's'):
    problem = rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(
            lattice=lattice, top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
            # A background-only layer fixes the thickness; conv carries the pair.
            layers=(rcwa.Layer(thickness=300., background=Material(epsilon=1.)),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(
            side='top', polarization=polarization, theta=.2, phi=0.),)))
    assembled = operators.assemble(problem, numerics=numerics).replace(epsilon=(conv,))
    transmitted.append(operators.solve(problem, assembled).T.sum())
print(*transmitted)  # p, then s
```

The example has a 17-unit gap and bounding-box offset `(49, -21)`. All
lengths, wavelength and thickness must use the same units. Declare the
original lattice and the same Fourier orders in the consuming problem; its
assembled layer operators are replaced by the builder's matrices, as described
for [`ConformingNVF`](conforming-nvf.md). Auxiliary mesh
refinement and optical Fourier order are distinct resolutions; the
[finite-cell controls](../../validation/rcwa/nvf-cartesian-pair.md) describe their present limits.

## Accepted inputs

| Input | Contract |
| --- | --- |
| `orders` | Nonnegative integer or `(x, y)` pair, default 1. Either or both axes may be zero. The exact original difference-mode set is selected before closure. |
| `refinement` | Integer 0 through 3, default 0. Separated pairs have 8/35/143/575 free potential unknowns; contacts have 6/27/111/447. See the [refinement controls and optical limits](../../validation/rcwa/nvf-pair-refinement.md). |
| `lattice` | A `Lattice` or finite real 2-by-2 matrix whose columns are physical basis vectors, with positive finite determinant. No lattice reduction is performed. |
| `structure` | Nonempty list or tuple of dictionaries with only `background` and `shapes`. Background defaults to vacuum. |
| Uniform layer | Omit `shapes` or supply an empty list or tuple. Its construction still requires a plan. |
| Patterned layer | Exactly two `rectangle` shapes; `rectangle` is also the default type. Other names, including `square`, are refused. |
| Lengths | Positive physical `lx` and `ly`; each separately falls back to `l` if absent. Explicit `lx`/`ly` takes precedence. The two heights must be exactly equal. |
| Pose | `angle` must equal zero. Physical Cartesian `cx` and `cy` default to zero. The two `cy` values must be exactly equal, with distinct left and right centers. |
| Materials | Scalar physical permittivities or `Material(epsilon=...)`, finite, positive in real part and nonnegative in imaginary part. Declared permeability, tensors and gain are refused. |

Planning identifies the physical left and right rectangles without changing
the input sequence. For widths `wL`, `wR` and centers `xL < xR`, it uses

```text
gap = (xR - xL) - (wL + wR) / 2
offset_x = (xL + xR) / 2 + (wR - wL) / 4
offset_y = common cy
```

The bounding rectangle must fit strictly inside a fundamental cell centered
on that pair, with positive finite triangles on the chosen chart. The whole
cell translates with the pair offset, which can cross the caller's cell
boundary without changing the containment test. Returned matrices are about the
cell centre, the frame `rcwa.operators` uses; the offset is the pair's public centre.
Do not add another
phase, material normalization or rotation to the returned matrices.

The **computed** gap must be strictly positive or exactly zero. Nothing is
snapped to contact: an intended shared edge whose floating-point center
arithmetic leaves a tiny positive gap is separated, and a negative gap is
refused. Exact contact can be specified by widths `(140, 193)` and centers
`(-47.5, 119)` in the example's units. Unequal heights, unequal vertical
centers, nonzero angles, overlap, partial-edge/corner contact, additional
shapes and unknown keys are refused. No general pair coverage is claimed.

## Planning and live derivatives

Call `plan(structure, lattice)` with concrete inputs before JAX tracing. The
immutable plan stores layer kinds, original input indices in physical
left-to-right order, contact state, cyclic mesh choice and refinement. It
stores no live geometry or material values. Pass it explicitly to every
`conv_matrices(..., plan=plan)` call. Changing the input ordering, layer kinds,
contact state or refinement requires an appropriate new plan. Replay never
sorts shapes or runs the planner inside a differentiated calculation.

With NumPy or JAX x64, lattice entries, widths, common height, common vertical
center, pair offset and physical materials remain live. Positive-gap pairs
also admit gap changes while the gap stays positive. Contact derivatives
must preserve the shared edge. Use one shared height and one shared vertical
center parameter, and center tangents satisfying
`d(xR - xL) = (dwL + dwR) / 2` at contact. An arbitrary tangent moving only one
height, vertical center or contact gap is outside this constrained family,
even if its primal point is valid. Differentiation through a rounding or
snapping operation is not provided.

Invalid eager inputs raise. Invalid traced numerical values poison the
affected layer's matrices with NaNs; an invalid lattice poisons every layer,
including uniform layers. Planning inside JIT is refused. Containment,
positive triangles, the original weak residual and finite storage/dissipation
checks remain active. A positive replay mesh does not prove continuity
across a cyclic planning choice. These are finite-space derivatives, not
contact-opening limits or a patterned optical accuracy guarantee.

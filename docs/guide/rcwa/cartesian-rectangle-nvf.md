# Cartesian rectangle NVF on an oblique lattice

`rcwa.experimental.CartesianRectangleNVF` builds finite NVF energy matrices
for one separated physical rectangle per patterned layer. Rectangle edges
retain their physical lengths and angle on an oblique lattice. This
experimental builder returns `ConvMatrices` for `rcwa.operators.solve`; its
finite auxiliary resolution does not certify optical convergence.

```python
import numpy as np
from wavebit import Material, Units, rcwa
from wavebit.rcwa import experimental, operators

lattice = np.array([[700., 315.], [0., 490.]])
structure = [dict(background=1., shapes=[dict(
    type='rectangle', lx=350., ly=300., angle=3.45,
    cx=49., cy=-21., material=4.,
)])]
builder = experimental.CartesianRectangleNVF((1, 2), refinement=1)
plan = builder.plan(structure, lattice)
conv = builder.conv_matrices(structure, lattice, plan=plan)
numerics = rcwa.Numerics(orders=builder.orders)
transmitted = []
for polarization in ('p', 's'):
    problem = rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(
            lattice=lattice, top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
            # A background-only layer fixes the thickness; conv carries the rectangle.
            layers=(rcwa.Layer(thickness=300., background=Material(epsilon=1.)),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(
            side='top', polarization=polarization, theta=.2, phi=0.),)))
    assembled = operators.assemble(problem, numerics=numerics).replace(epsilon=(conv,))
    transmitted.append(operators.solve(problem, assembled).T.sum())
print(*transmitted)  # p, then s
```

Use the same original lattice and Fourier orders in the builder and the
consuming problem; its assembled layer operators are replaced by the
builder's matrices, as described for [`ConformingNVF`](conforming-nvf.md).
The example illustrates matrix use. Refine both the
auxiliary cell and the optical Fourier orders for the quantity being
calculated; the [finite-cell controls](../../validation/rcwa/nvf-cartesian-cell.md) describe
what has been verified.

## Accepted inputs

| Input | Contract |
| --- | --- |
| `orders` | Nonnegative integer or `(x, y)` pair, default 1. Zero along either axis is allowed. The exact original difference-mode set is selected before closure. |
| `refinement` | 0, 1 or 2, default 0. These give 4, 19 or 79 free scalar potential unknowns. This is independent of the optical Fourier order. |
| `lattice` | A `Lattice` or finite real 2-by-2 array with positive finite determinant, with physical basis vectors in columns. No lattice reduction is performed. |
| `structure` | Nonempty list or tuple of layer dictionaries containing only `background` and `shapes`. Background defaults to vacuum. |
| Uniform layer | Omit `shapes` or supply an empty list or tuple. A plan is still required when constructing its matrices. |
| Patterned layer | Exactly one shape of type `rectangle`, which is also the default type. Other type spellings, including `square`, are refused. |
| Rectangle lengths | Positive physical `lx` and `ly`. Each separately falls back to `l` when its own key is absent. Explicit `lx` or `ly` takes precedence. |
| Pose | `angle` in laboratory radians, default 0; physical Cartesian `cx` and `cy` offsets from the caller's cell center, each default 0. |
| Materials | Physical scalar permittivities or `Material(epsilon=...)`. Background and inclusion must be finite, nonmagnetic, positive in real part and nonnegative in imaginary part. Independent loss ratios are allowed. |

The rectangle is centered inside a fundamental cell that translates with
its offset. It must fit strictly inside that rectangle-centered cell and
have positive finite triangles on the selected mesh. Offsets may translate
the whole periodic structure freely: crossing the caller's chosen cell
edge by translation does not enlarge the admitted rectangle dimensions.
The returned matrices are about the cell centre, the frame `rcwa.operators` uses;
the offset is the rectangle's public centre.
Do not add another phase, material scaling or Cartesian rotation to the
returned matrices.

Unknown keys, additional shapes, unsupported shape kinds, tensor or gain
materials and declared permeability are refused. Contact, overlapping
periodic images and a general mesh for multiple rectangles are outside
this builder's contract. The existing
[`ConformingNVF`](conforming-nvf.md) and
[`AffineProfileNVF`](affine-profile-nvf.md) retain their respective geometry
contracts; this explicit builder does not change their admission rules.

## Planning and live derivatives

Create the builder and call `plan(structure, lattice)` on concrete geometry
before JAX tracing. The immutable plan stores each layer's uniform or
patterned status and the patterned layer's cyclic vertex choice. Supply
it explicitly to every `conv_matrices(..., plan=plan)` call. A changed layer
count or uniform/patterned status requires a new plan.

Replay reads the lattice, rectangle lengths, pose and material values
again. These remain live under `jax.jit` with
`conv_matrices(..., plan=plan, xp=jax.numpy)` and JAX x64. NumPy construction
also requires 64-bit precision. The builder does not run a geometry
planner inside the differentiated calculation. Derivatives refer to the
fixed finite space and do not include selection between vertex choices.

Replan explicitly when changing the parameter representation, such as
adding pi to the rectangle angle. A new plan can represent the same
physical rectangle with a different raw vertex ordering. Replay checks
containment and triangle validity, but cannot detect every representation
change that still leaves a positive mesh. No continuity or derivative
across a planning tie is claimed.

Invalid eager inputs raise. Invalid traced numerical values produce NaNs
in the affected layer's matrices; an invalid lattice affects every layer,
including uniform layers. The independent weak-system and derivative
controls qualify this finite construction, not continuum contact behavior
or Maxwell transmission accuracy.

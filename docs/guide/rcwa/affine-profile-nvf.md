# Conforming NVF from affine profiles

`rcwa.experimental.AffineProfileNVF` builds energy matrices for an explicit
affine image of a fractional strip profile. The lattice vectors may be
genuinely nonorthogonal. It returns `ConvMatrices` for `rcwa.operators.solve`,
using the same lattice and retained Fourier orders as the consuming problem.

```python
import numpy as np
from wavebit import Material, Units, rcwa
from wavebit.rcwa import experimental, operators

lattice = np.array([[700., 315.], [0., 490.]])
structure = [dict(background=1., profile=dict(
    width=.5,
    height=1.-17./700.,
    seams=[31./700.],
    values=[4., 9.],
    offset=(0., 0.),
))]
builder = experimental.AffineProfileNVF(
    (2, 1), rows=2, body_degree=2, gap_degree=2, grade_y=False,
)
conv = builder.conv_matrices(structure, lattice)
numerics = rcwa.Numerics(orders=builder.orders)
transmitted = []
for polarization in ('p', 's'):
    problem = rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(
            lattice=lattice, top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
            # A background-only layer fixes the thickness; conv carries the profile.
            layers=(rcwa.Layer(thickness=300., background=Material(epsilon=1.)),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(
            side='top', polarization=polarization, theta=.2, phi=0.),)))
    assembled = operators.assemble(problem, numerics=numerics).replace(epsilon=(conv,))
    transmitted.append(operators.solve(problem, assembled).T.sum())
print(*transmitted)  # p, then s
```

The builder's matrices replace the assembled layer operators, as described
for [`ConformingNVF`](conforming-nvf.md); each polarization is assembled
separately.

The example specifies a finite model. Its auxiliary cell resolution and
optical Fourier orders are illustrative; each needs refinement for the
observable being calculated. The
[independent optical comparison](../../validation/rcwa/nvf-affine-convergence.md) records the
remaining accuracy limits.

## Coordinates and inputs

Write the supplied lattice as A, with vectors in columns. A point with
centered fractional coordinates `(u, v)` has public Cartesian position
`A @ [u, v] + offset`, measured from the cell center. Returned matrices
also use that centre, the frame `rcwa.operators` uses. Do not add a half-cell
phase to them or a half-cell shift to public field-sample coordinates. The profile occupies
`-width/2 < u < width/2` and `-height/2 < v < height/2`.
Its material intervals run in increasing u, separated by `seams`.
The rest of the periodic cell contains the layer background.

| Input | Contract |
| --- | --- |
| `orders` | Nonnegative integer or `(u, v)` integer pair. The exact requested modes are selected before matrix closure, including zero order along either axis. |
| `lattice` | A `Lattice` or finite real 2-by-2 array with positive determinant. Its vectors and `offset` use the consuming problem's length unit. No integer basis reduction is performed. |
| `structure` | Nonempty sequence of layer dictionaries with `background` and `profile`. An absent or `None` profile is uniform. Background defaults to 1. |
| `width` | Required positive fractional span strictly less than 1. |
| `height` | Required positive fractional span at most 1. Height 1 is exact contact across the periodic boundary. Only the existing contact roundoff allowance is accepted beyond it. |
| `seams` | Required ordered one-dimensional sequence of fractional u positions inside the width envelope. Use an empty sequence for one material. Coincident seams retain zero-width intervals and their opening derivatives. |
| `values` | Required sequence of physical scalar materials, one more than the number of seams. Each may be a permittivity or `Material(epsilon=...)`. |
| `offset` | Optional real Cartesian displacement from the cell center, default `(0., 0.)`. Translation is unrestricted. |
| Materials | Finite scalar, nonmagnetic, positive real permittivity with nonnegative imaginary part. The background and each interval may have different loss ratios. |

Widths, heights and seams are fractions, while offset is a physical
Cartesian displacement. In the example, the fractional gap `17/700`
has physical normal width 11.9 in the lattice's length units. In general,
that width is `(1-height) / norm(inv(A).T @ [0, 1])`.
Reference rectangles become physical parallelograms under shear.
For ordinary physical rectangle dictionaries, use
[`ConformingNVF`](conforming-nvf.md) and its supported lattice contract.
The affine-profile builder refuses unknown keys rather than interpreting
ordinary shape dictionaries as fractional profiles.

## Differentiation and matrix use

Create the builder outside `jax.jit`. Geometry, offsets, the lattice and
material values can then be passed live to
`builder.conv_matrices(structure, lattice, xp=jax.numpy)` with JAX x64.
The explicit ordered intervals need no sorting plan. Keep layer counts
and profile array shapes fixed within a compiled call. Invalid eager
geometry raises; invalid traced numerical values produce NaN matrices.
NumPy construction also uses 64-bit precision. Exported matrices can be
converted to Torch tensors for operators assembled with `backend='torch'`.

Returned Fourier matrices use the cell-centre origin, including the physical offset.
The cell already returns fields in the laboratory frame. The adapter
restores the physical background in its energy and longitudinal material
coefficients before closing the matrices. Lossless storage and passive
loss are checked without changing the exported matrices.

Exact laminated contact values and finite-cell opening derivatives have
[independent controls](../../validation/rcwa/nvf-affine-cells.md). Those checks do not certify
continuum opening-derivative convergence or optical accuracy. Arbitrary
physical rectangles on oblique lattices, curves, overlaps, tensor or
magnetic materials, gain, and nonpositive real permittivity are outside
this adapter's contract.

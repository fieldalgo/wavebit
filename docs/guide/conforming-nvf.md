# Conforming NVF from rectangle layers

`wavebit.ConformingNVF` constructs a material-dependent conforming cell and
exports its energy closure as `ConvMatrices` for `solve_conv()`. It accepts
ordinary `background` / `shapes` dictionaries. An installed Wavebit package
is sufficient; research scripts are not runtime dependencies.

The current geometry is a horizontal sequence of touching rectangles in
the lattice's local axes. All rectangles in a layer have the same height
and vertical center. Materials may differ across their shared edges.

```python
import numpy as np
import wavebit

lattice = np.diag([700., 430.])
structure = [dict(background=1.7, shapes=[
    dict(type='rectangle', lx=206., ly=401., cx=-72., material=4.),
    dict(type='rectangle', lx=144., ly=401., cx=103., material=9.),
])]
builder = wavebit.ConformingNVF((2, 1), rows=4, body_degree=2, gap_degree=4)
conv = builder.conv_matrices(structure, lattice)
solver = wavebit.rcwa(
    lattice=lattice, truncation_order=builder.truncation_order,
    wavelength=900., thickness=[300.], theta=.2, phi=.37,
    top=1., bot=2.25,
)
result = solver.solve_conv(*conv)
print(result.p_inc.T.sum(), result.s_inc.T.sum())
```

This is a finite-model example, not an accuracy prescription. Cell
resolution and retained Fourier orders require separate refinement for the
observable of interest. A small last-order increment alone does not bound
the optical error. The [energy and convergence studies](../nvf-gap-convergence.md)
explain the distinction.

## Inputs and supported geometry

| Input | Contract |
| --- | --- |
| `truncation_order` | Nonnegative integer or `(u, v)` integer pair, matching the consuming solver. Projection uses this order set before closing the energy matrices. |
| `lattice` | A `wavebit.Lattice` or 2-by-2 array with basis vectors in columns. Right-handed orthogonal cells with two independently positive periods and rigid rotation are supported. |
| `structure` | Nonempty list or tuple of layer dictionaries, each with `background` and/or `shapes`. A layer with no shapes is uniform. |
| `cx`, `cy` | Cartesian offsets from the cell center, in the lattice's length units. Default zero. Periodic translation is unrestricted; coordinates must still describe one contiguous tiling, rather than separately wrapped pieces. |
| `lx`, `ly` | Positive physical lengths. Equal height and local y center within a tiling; total local x width strictly less than the first period. Height may reach the second period, including exact contact. |
| `angle` | Radians from laboratory x, default zero. Rectangle axes align with the lattice's first axis modulo pi. Rotate the lattice, centers, and shape angles together. |
| Materials | Positive, finite, real scalar permittivities, as numbers or `Material(epsilon=...)`. Each layer may have a different background. A declared `mu`, including `mu=1`, is unsupported. |

The shape list can arrive in any order; planning sorts by local x. Shared
boundaries between different materials remain live. Subdividing a rectangle
into equal-material neighbors does not add an interface constraint.
Uniform and patterned layers may share one stack, in the same order as the
solver's thickness list.

Curves, overlaps, internal gaps, unequal heights, independently rotated
objects, sheared cells, tensors, and complex or nonpositive cell
permittivities are refused. This builder does not select a different NVF or
Li factorization for unsupported input. The existing circular shape NVF
remains available through the ordinary solver API.

## Resolution controls

Constructor keywords configure the fixed cell approximation and remain
fixed during a JAX trace.

| Keyword | Default | Meaning |
| --- | --- | --- |
| `rows` | `4` | Initial y subdivisions, at least two, before grading and polynomial enrichment. |
| `body_degree` | `2` | Body potential degree, from one through four. |
| `gap_degree` | `4` | Gap potential degree, from one through six. |
| `flux_basis` | `'local'` | Local integrated flux basis; `'global'` is available for numerical comparisons. |
| `edge_fraction` | `1e-5` | Smallest reference edge interval, strictly between zero and `.001`. A resolution setting, not a gap cutoff or convergence tolerance. |
| `grade_y` | `True` | Add fixed edge grading to the body y intervals. |
| `x_subdivisions`, `y_subdivisions` | `1` | Positive integer subdivisions of each reference interval. |
| `grading_strength` | `8.` | Material-jump grading strength, from zero through 32. |
| `grading_radius` | `.01` | Normalized grading radius: `.0001` through `.1` for local flux, `.005` through `.1` for global flux. |
| `quadrature` | `12` | Base quadrature setup, at least two. Polynomial projections use analytic moments. |

Coefficient fields and energy are rotated to the solver's Cartesian frame
and translated to its Fourier origin together. Each axis is normalized
separately; the physical aspect ratio enters the cell equations and energy
integrals. There is no physical 700-unit period restriction. The
[rectangular-cell derivation](../nvf-rectangular-cells.md) explains why
stretching square-cell coefficients alone would give a different result.

## JAX planning and replay

NumPy and JAX with x64 construct the cell; output is complex128. The returned
matrices can also be transferred to a Torch solver, but the cell builder
does not supply Torch geometry derivatives. JAX derivatives concern cell
construction and the energy closure. The consuming eigensolver retains its
own [autodiff behavior](autodiff.md).

Build an immutable plan from concrete geometry before `jit`. It stores only
each layer's input-index ordering. Coordinates, dimensions, materials,
backgrounds, both lattice periods, and common rotation remain live on replay.

```python
import jax
import jax.numpy as jnp
import numpy as np
import wavebit

wavebit.enable_x64()
lattice = np.eye(2)*700.
builder = wavebit.ConformingNVF(1, rows=2, body_degree=2, gap_degree=2)

def layers(gap, epsilon):
    return [dict(background=1., shapes=[
        dict(type='rectangle', lx=350., ly=700.-gap, material=epsilon),
    ])]

plan = builder.plan(layers(17., 4.), lattice)

@jax.jit
def material_matrix(gap, epsilon):
    return builder.conv_matrices(
        layers(gap, epsilon), jnp.asarray(lattice), plan=plan, xp=jnp,
    ).epx

derivative = jax.jacfwd(material_matrix, argnums=0)(17., 4.)
print(derivative.shape)
```

Keep the number and ordering of shapes and layers fixed during replay.
Changing a shared edge requires moving both adjacent rectangles so they
still touch. Re-plan when changing topology. Invalid concrete geometry
raises; invalid traced geometry or material produces NaN matrices for that
layer. At full-height contact, the gap derivative is the one-sided limit
from positive gaps. Gaps are not snapped to zero. Within 32 machine epsilons
of normalized contact, the same analytic equations cover negative rounding
error from a lattice norm or height subtraction. This keeps contact
derivatives live; a larger height overrun is refused. This allowance is a
floating-point geometry tolerance, not an optical accuracy criterion.

Use a solver with the same lattice and orders. A fresh solver without a
raster structure selects the 2D operator even for `(u, 0)`, `(0, v)`, or
`(0, 0)`. If reusing a solver previously routed to a 1D raster calculation,
set `solver.formulation = '4ff'` to retain all in-plane coupling.

The [integration record](../nvf-energy-engine.md) identifies the independent
controls and remaining geometry and accuracy limits.

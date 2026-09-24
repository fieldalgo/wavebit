# Conforming NVF from rectangle layers

`wavebit.ConformingNVF` constructs a material-dependent conforming cell and
exports its energy closure as `ConvMatrices` for `solve_conv()`. It accepts
ordinary `background` / `shapes` dictionaries. An installed Wavebit package
is sufficient; research scripts are not runtime dependencies.

Each patterned layer is a sequence of nonoverlapping rectangles along either
axis of the orthogonal reduced lattice. The rectangles share their extent and center perpendicular
to that axis. Materials may differ across their shared edges, and the
tiling direction may differ between layers. Internal gaps contain the layer's
background material; they can open and close during JAX replay.

```python
import numpy as np
import wavebit

lattice = np.diag([700., 430.])
# Layers are listed from the bottom to the top.
structure = [
                dict(background=1.7, shapes=[
        dict(type='rectangle', lx=671., ly=155., cy=-97.5, material=4.),
        dict(type='rectangle', lx=671., ly=155., cy=97.5, material=9.),
    ]),
                dict(background=2.3),
                dict(background=1.7, shapes=[
        dict(type='rectangle', lx=186., ly=401., cx=-82., material=4.),
        dict(type='rectangle', lx=124., ly=401., cx=113., material=9.),
    ]),
            ]
builder = wavebit.ConformingNVF((2, 1), rows=4, body_degree=2, gap_degree=4)
conv = builder.conv_matrices(structure, lattice)
solver = wavebit.rcwa(
    lattice=lattice, truncation_order=builder.truncation_order,
    wavelength=900., thickness=[150., 60., 150.], theta=.2, phi=.37,
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
| `lattice` | A `wavebit.Lattice` or 2-by-2 array with basis vectors in columns. A right-handed basis that reduces to an orthogonal basis by an integer unimodular change is supported. The two reduced periods are positive and may vary independently. |
| `structure` | Nonempty list or tuple of layer dictionaries, each with `background` and/or `shapes`. A layer with no shapes is uniform. |
| `cx`, `cy` | Cartesian offsets from the original supplied cell center, in the lattice's length units. Default zero. Periodic translation is unrestricted; coordinates must describe one ordered profile with span below one period, rather than separately wrapped pieces. |
| `lx`, `ly` | Positive physical lengths along the rectangle's own axes. A tiling has equal perpendicular extent and center. Its span along the tiling axis is strictly less than that period; the perpendicular extent may reach the other period, including exact contact. |
| `angle` | Radians from laboratory x, default zero. Rectangle edges align with the orthogonal reduced lattice axes. Adding pi/2 and exchanging `lx` and `ly` describes the same rectangle. Arbitrary angles within a fixed lattice are unsupported. |
| Materials | Finite scalar permittivities with positive real part and nonnegative imaginary part, as numbers or `Material(epsilon=...)`. Every shape and background may have an independent loss ratio. A declared `mu`, including `mu=1`, is unsupported. |

The shape list can arrive in any order; planning sorts along the chosen
tiling axis. Shared boundaries between different materials remain live.
Subdividing a rectangle
into equal-material neighbors does not add an interface constraint.
Uniform and patterned layers may share one stack, in the same order as the
solver's thickness list.

Curves, overlaps, unequal perpendicular extents or centers,
edges not aligned with the reduced lattice, genuinely nonorthogonal point
lattices, tensors, gain and nonpositive real permittivity are refused.
This builder does not select a different NVF or
Li factorization for unsupported input. The existing circular shape NVF
remains available through the ordinary solver API.

For explicit affine images of fractional profiles on nonorthogonal lattices,
use [`AffineProfileNVF`](affine-profile-nvf.md). That builder takes fractional
widths, heights and material intervals; it does not reinterpret the physical
rectangle inputs described here.

## Equivalent lattice bases

A skew basis can describe an orthogonal point lattice. For example,
`a1=(700,0)` and `a2=(2100,430)` generate the same translations as
`(700,0)` and `(0,430)`. Planning records the integer matrix T for which
`lattice @ T` is orthogonal. Rectangle lengths and angles keep their physical
meaning; the adapter does not turn them into parallelograms.

```python
import numpy as np
import wavebit

lattice = np.array([[700., 2100.], [0., 430.]])
structure = [dict(background=1.7, shapes=[
    dict(type='rectangle', lx=350., ly=401., material=4.+.4j),
])]
builder = wavebit.ConformingNVF((2, 1), rows=2, body_degree=2, gap_degree=2)
plan = builder.plan(structure, lattice)
conv = builder.conv_matrices(structure, lattice, plan=plan)
print(plan.reduction, conv.epx.shape)
```

The consuming solver still receives the original `lattice` and
`truncation_order`. Those define its physical wavevectors. The adapter
selects the transformed version of that exact mode set before energy
closure; the output matrix dimensions do not grow. Computing the required
Fourier coefficients can cost more for large integer shears.

Centers remain offsets from the original supplied cell center. When
changing from lattice A to another basis Anew while keeping absolute shape
positions fixed, add `sum(A-Anew, axis=1)/2` to every `(cx,cy)` offset.
The [rebasing derivation](../nvf-lattice-rebasing.md) explains the phase and
mode mapping. A genuine shear such as `(700,0),(50,430)` remains unsupported.

## Passive complex materials

Every shape and background accepts its own passive complex permittivity.
For example, materials `4+.4j` and `9+1.8j` can share a lossless background
`1.7`, or an absorbing background `1.7+.2j`. Both real and imaginary parts
remain live during JAX replay. Gain and nonpositive real parts are refused.

The cell uses complex trial fields and conjugate test fields. It retains
both transverse cross terms and projects the full complex material energy
into the engine's constitutive matrices. The
[complex-material derivation](../nvf-complex-material.md) explains this
construction and its independent controls. The common-factor case described
in the [earlier derivation](../nvf-common-loss.md) remains a useful exact
control, but is no longer an input restriction.

Vary each material's imaginary part independently, as in the JAX example
below. At zero imaginary part, a passive loss derivative is one-sided from
positive loss. Invalid traced materials produce NaN matrices; eager calls
raise. The mesh follows the magnitude and phase of material contrast and
keeps the same topology when loss changes.

## Resolution controls

Constructor keywords configure the fixed cell approximation and remain
fixed during a JAX trace.

| Keyword | Default | Meaning |
| --- | --- | --- |
| `rows` | `4` | Initial subdivisions perpendicular to the chosen tiling axis, at least two, before grading and polynomial enrichment. |
| `body_degree` | `2` | Body potential degree, from one through four. |
| `gap_degree` | `4` | Gap potential degree, from one through six. |
| `flux_basis` | `'local'` | Local integrated flux basis; `'global'` is available for numerical comparisons. |
| `edge_fraction` | `1e-5` | Smallest reference edge interval, strictly between zero and `.001`. A resolution setting, not a gap cutoff or convergence tolerance. |
| `grade_y` | `True` | Add fixed edge grading to the body intervals perpendicular to the tiling axis. |
| `x_subdivisions`, `y_subdivisions` | `1` | Positive integer subdivisions of each reference interval along and perpendicular to the tiling axis, respectively. |
| `grading_strength` | `8.` | Material-jump grading strength, from zero through 32. |
| `grading_radius` | `.01` | Normalized grading radius: `.0001` through `.1` for local flux, `.005` through `.1` for global flux. |
| `quadrature` | `12` | Base quadrature setup, at least two. Polynomial projections use analytic moments. |

Material grading smooths signed log-permittivity jumps before squaring their
sum. Thus the two boundaries of a closing interval merge or cancel, and the
mesh does not retain a feature from a vanished background interval. This
replaces the earlier independent-jump density and changes finite-resolution
values when grading is enabled. The [separated-rectangle derivation](../nvf-separated-rectangles.md)
describes the formula and its independent controls. Strength zero keeps the
fixed mesh; no grading setting certifies optical accuracy.

Coefficient fields and energy are rotated to the solver's Cartesian frame
and translated to its Fourier origin together. Each axis is normalized
separately; the physical aspect ratio enters the cell equations and energy
integrals. There is no physical 700-unit period restriction. The
[rectangular-cell derivation](../nvf-rectangular-cells.md) explains why
stretching square-cell coefficients alone would give a different result.
For a second-axis tiling, the cell uses the reduced right-handed frame `(a2, -a1)`
and the reciprocal aspect ratio. Each layer is exported in the original
solver's common Cartesian frame and Fourier order layout. The
[layer-axis construction](../nvf-layer-axes.md) describes this mapping.

## JAX planning and replay

NumPy and JAX with x64 construct the cell; output is complex128. The returned
matrices can also be transferred to a Torch solver, but the cell builder
does not supply Torch geometry derivatives. JAX derivatives concern cell
construction and the energy closure. The consuming eigensolver retains its
own [autodiff behavior](autodiff.md).

Build an immutable plan from concrete geometry before `jit`. It stores
each layer's input-index ordering and tiling axis, plus the integer lattice
reduction. Planning prefers the first reduced axis when both are valid,
as for many single rectangles.
Coordinates, dimensions, materials, backgrounds, both lattice periods,
and common rotation remain live on replay. The recorded reduction must keep
the live lattice orthogonal; independent continuous shear leaves this
supported family and is refused eagerly or produces NaNs under tracing.

```python
import jax
import jax.numpy as jnp
import numpy as np
import wavebit

wavebit.enable_x64()
lattice = np.eye(2)*700.
builder = wavebit.ConformingNVF(1, rows=2, body_degree=2, gap_degree=2)

def layers(gap, epsilon, loss):
    return [dict(background=1., shapes=[
        dict(type='rectangle', lx=350., ly=700.-gap, material=epsilon+1j*loss),
    ])]

plan = builder.plan(layers(17., 4., .1), lattice)

@jax.jit
def material_matrix(gap, epsilon, loss):
    return builder.conv_matrices(
        layers(gap, epsilon, loss), jnp.asarray(lattice), plan=plan, xp=jnp,
    ).epx

gap_derivative, loss_derivative = jax.jacfwd(material_matrix, argnums=(0, 2))(17., 4., .1)
print(gap_derivative.shape, loss_derivative.shape)
```

Keep the number and ordering of shapes and layers fixed during replay.
Adjacent edges may move independently to open or close an internal background
interval, provided the rectangles do not overlap. Replay retains its chosen
axis even if both axes later become
valid; it does not switch the finite approximation during differentiation.
Re-plan when changing those counts or ordering, or when the chosen axis becomes invalid.
Invalid concrete geometry raises; invalid traced geometry or material
produces NaN matrices for that layer. At full perpendicular-period contact,
the gap derivative is the one-sided limit from positive gaps. The same applies
when an internal background interval closes. Gaps are not
snapped to zero. Within 32 machine epsilons
of normalized contact, the same analytic equations cover negative rounding
error from a lattice norm or extent subtraction. This keeps contact
derivatives live; a larger extent overrun is refused. This allowance is a
floating-point geometry tolerance, not an optical accuracy criterion.

Use a solver with the same lattice and orders. A fresh solver without a
raster structure selects the 2D operator even for `(u, 0)`, `(0, v)`, or
`(0, 0)`. If reusing a solver previously routed to a 1D raster calculation,
set `solver.formulation = '4ff'` to retain all in-plane coupling.

The [integration record](../nvf-energy-engine.md) identifies the independent
controls and remaining geometry and accuracy limits.

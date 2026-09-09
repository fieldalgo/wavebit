# Conforming NVF from rectangle layers

`wavebit.ConformingNVF` constructs a material-dependent conforming cell and
exports its energy closure as `ConvMatrices` for `solve_conv()`. It accepts
ordinary `background` / `shapes` dictionaries. An installed Wavebit package
is sufficient; research scripts are not runtime dependencies.

Each patterned layer is a sequence of nonoverlapping rectangles along either
lattice axis. The rectangles share their extent and center perpendicular
to that axis. Materials may differ across their shared edges, and the
tiling direction may differ between layers. Internal gaps contain the layer's
background material; they can open and close during JAX replay.

```python
import numpy as np
import wavebit

lattice = np.diag([700., 430.])
structure = [
    dict(background=1.7, shapes=[
        dict(type='rectangle', lx=186., ly=401., cx=-82., material=4.),
        dict(type='rectangle', lx=124., ly=401., cx=113., material=9.),
    ]),
    dict(background=2.3),
    dict(background=1.7, shapes=[
        dict(type='rectangle', lx=671., ly=155., cy=-97.5, material=4.),
        dict(type='rectangle', lx=671., ly=155., cy=97.5, material=9.),
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
| `lattice` | A `wavebit.Lattice` or 2-by-2 array with basis vectors in columns. Right-handed orthogonal cells with two independently positive periods and rigid rotation are supported. |
| `structure` | Nonempty list or tuple of layer dictionaries, each with `background` and/or `shapes`. A layer with no shapes is uniform. |
| `cx`, `cy` | Cartesian offsets from the cell center, in the lattice's length units. Default zero. Periodic translation is unrestricted; coordinates must describe one ordered profile with span below one period, rather than separately wrapped pieces. |
| `lx`, `ly` | Positive physical lengths along the rectangle's own axes. A tiling has equal perpendicular extent and center. Its span along the tiling axis is strictly less than that period; the perpendicular extent may reach the other period, including exact contact. |
| `angle` | Radians from laboratory x, default zero. Rectangle edges align with the lattice axes. Adding pi/2 and exchanging `lx` and `ly` describes the same rectangle. Arbitrary angles within a fixed lattice are unsupported. |
| Materials | Finite scalar permittivities with positive real part and nonnegative imaginary part, as numbers or `Material(epsilon=...)`. Every material in a patterned layer, including its background, shares `Im(epsilon)/Re(epsilon)`. Each layer may use a different ratio. A declared `mu`, including `mu=1`, is unsupported. |

The shape list can arrive in any order; planning sorts along the chosen
tiling axis. Shared boundaries between different materials remain live.
Subdividing a rectangle
into equal-material neighbors does not add an interface constraint.
Uniform and patterned layers may share one stack, in the same order as the
solver's thickness list.

Curves, overlaps, unequal perpendicular extents or centers,
edges not aligned with the lattice, sheared cells, tensors, gain, nonpositive
real permittivity, and unequal loss ratios within a patterned layer are
refused. This builder does not select a different NVF or
Li factorization for unsupported input. The existing circular shape NVF
remains available through the ordinary solver API.

## Passive materials with a common loss ratio

Within a patterned layer, supply `epsilon_j = (1 + 1j*eta)*storage_j`, with
positive real `storage_j` and nonnegative `eta`. The background carries the
same factor. For example, `4+.4j`, `9+.9j` and background `1.7+.17j` share
`eta=.1`. Each patterned or uniform layer can have its own loss ratio.

The common factor cancels from the cell equations. It multiplies the
transverse constitutive matrices and divides the inverse longitudinal slot,
so the loss reaches the actual engine. The real storage cell retains its
Hermitian check. The [common-loss derivation](../nvf-common-loss.md) gives
the passivity argument and independent absorbing-layer controls.

On JAX replay, vary the shared `eta` together for all materials in that
patterned layer, as in the example below. Derivatives are supported along
this common-ratio constraint; at `eta=0`, the loss derivative is one-sided
from positive loss. Unequal ratios raise eagerly and produce NaN matrices
when traced. Ratio agreement permits only floating-point roundoff:
`128*machine_epsilon*(1+abs(eta_background))`. A lossy inclusion in a lossless
background still requires the separate complex-corrector extension.

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
For a second-axis tiling, the cell uses the right-handed frame `(a2, -a1)`
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
each layer's input-index ordering and tiling axis. Planning prefers the
first lattice axis when both are valid, as for many single rectangles.
Coordinates, dimensions, materials, backgrounds, both lattice periods,
and common rotation remain live on replay.

```python
import jax
import jax.numpy as jnp
import numpy as np
import wavebit

wavebit.enable_x64()
lattice = np.eye(2)*700.
builder = wavebit.ConformingNVF(1, rows=2, body_degree=2, gap_degree=2)

def layers(gap, epsilon, loss):
    factor = 1+1j*loss
    return [dict(background=factor, shapes=[
        dict(type='rectangle', lx=350., ly=700.-gap, material=factor*epsilon),
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

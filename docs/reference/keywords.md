# Keyword reference

:::{note}
This page lists the fields of the current RCWA declarations in `wavebit.rcwa` and of the
`wavebit.Execution` that runs them. It used to list the keywords of the earlier
`rcwa(...)` constructor; [Earlier spellings](#earlier-spellings) maps each of those to its
current home and names the ones that have none. The [earlier API reference](../migration/legacy-api.md)
keeps the complete constructor.
:::

Every field each declaration takes, what it accepts, what it defaults to and what it refuses.
Grouped by the question it answers -- the problem and its outputs, the stack, the light, the
numerics, and where the run happens -- with one section per declaration. The
[RCWA API reference](rcwa/index.md) has the signatures and docstrings.

The FDTD module has its own declarations (`fdtd.Problem`, `fdtd.Scene`, `fdtd.Numerics` and
the rest); they are in the [FDTD API reference](fdtd/index.md) and
[the FDTD model guide](../guide/fdtd/model.md).

This page is the lookup. For what the choices *mean* and how to pick between them, the user
guide takes them one subject at a time: [the light](../guide/rcwa/light.md),
[the scene](../guide/rcwa/scene.md), [the numerics](../guide/rcwa/numerics.md),
[near fields](../guide/rcwa/fields.md) and [backends](../guide/common/backends.md).

A few rules cover the whole surface and are worth stating once:

* **An unknown keyword is refused, not swallowed.** Every `rcwa` declaration is keyword-only
  and names the nearest real field: `rcwa.Stack(bot=...)` raises a `TypeError` that suggests
  `bottom`, and `rcwa.Numerics(truncation_order=(2, 2))` one that lists `orders`. The earlier
  constructor's keywords are refused the same way, not read as aliases. `Units` and
  `Execution` refuse an unknown keyword with Python's own `TypeError`.
* **Declarations are immutable.** `.replace(...)` builds a new one and runs its validation
  again, and `prepared.run(problem)` takes the complete new problem.
* **A refusal is named by its code and by when it is found.** A code such as `stack.period` is
  the `.code` of the `ValidationError` or `UnsupportedFeatureError` raised; every code has the
  one `area.item` form listed in [Refusal codes](#refusal-codes). *At construction*
  is the declaration's own constructor; *at check* means `rcwa.check` reports it and
  `rcwa.prepare` and `rcwa.run` raise it; *at prepare* and *at run* mean nothing earlier sees
  it.
* **Every material is a material model** -- `Material`, `ConductiveMaterial`,
  `DispersiveMaterial` or `TabulatedMaterial` -- never a bare number. Its `epsilon` is a
  relative permittivity, not a refractive index (`eps_from_refractive_index` converts), and a
  bare `3.5` is refused at check with `material.type` rather than read as either.
* **Lengths share one unit**, the problem's `Units(length=...)`: periods, thicknesses, shape
  dimensions, source positions and vacuum wavelengths. `kx`, `ky` and `bloch_k` are in its
  inverse. Angles are radians.

## The problem

### `rcwa.Problem`

| Field | Default | Accepts | Refuses |
| --- | --- | --- | --- |
| `stack` | required | an `rcwa.Stack` | anything else: `problem.type`, at construction |
| `excitation` | required | an `rcwa.Excitation` | anything else: `problem.type` |
| `units` | `Units()`, metres | a `wavebit.Units`. `Units(length=...)` takes `'m'`, `'mm'`, `'um'` or `'nm'` | a bare string such as `'nm'`: `problem.type`; any other unit name: `units.length` |
| `outputs` | `Outputs()` | an `rcwa.Outputs` | anything else: `problem.type` |

### `rcwa.Outputs`

What the run keeps. Field reconstruction and the scattering matrix are opt-in because the state
they need is kept per layer. Explained in [Near fields](../guide/rcwa/fields.md).

| Field | Default | Accepts | Refuses |
| --- | --- | --- | --- |
| `diffraction` | `True` | a `bool`: per-order efficiencies, s/p amplitudes and powers at both ports (`result.R`, `result.T`, `result.reflection`, `result.transmission`). Only a single `PlaneWave` has them | not a `bool` (`1`): `outputs.boolean`, at construction; `True` beside any other source, or beside a second source: `rcwa.coherent_diffraction`, at check |
| `fields` | `False` | a `bool`: keep the state `result.field(x=..., y=..., z=...)` and `result.boundary_fields(...)` read | not a `bool`: `outputs.boolean` |
| `scattering` | `False` | a `bool`: keep the stack's scattering matrix, one per wavelength, as `result.scattering` | not a `bool`: `outputs.boolean`; `True` under `algorithm='etm'`: `rcwa.scattering_algorithm`; `True` with a coherent source: `rcwa.coherent_scattering`, both at check |
| `power_region` | `None`, read as `'cell'` | `'cell'`, `'infinite'`, or `((xmin, xmax), (ymin, ymax))` in the length unit: the area a coherent-source run averages its boundary power over. A single `PlaneWave` does not read it | any other string: `rcwa.invalid`, at run. A `GaussianBeam` or an isolated `ElectricDipole` without the finite rectangle: `rcwa.invalid`, at run |

## The stack

Explained in [Declaring an RCWA stack](../guide/rcwa/scene.md).

### `rcwa.Stack`

`period` and `lattice` are two spellings of one cell: naming both is refused, and naming neither
gives the unit square.

| Field | Default | Accepts | Refuses |
| --- | --- | --- | --- |
| `layers` | `()` | a tuple of `rcwa.Layer`, **bottom to top**; the bottom interface is `z = 0`. An empty stack is one interface | anything but `Layer`s: `stack.layers`, at construction; raster layers beside region layers: `rcwa.mixed_layer_encoding`, at check |
| `top` | `Material(epsilon=1.)` | the material model of the upper half-space. A passive double negative (`Re(eps) < 0`, `Re(mu) < 0`) is solved on either side, lit or not; see [negative-index media](../theory/rcwa/negative-index-media.md) | a bare number: `material.type`; an active double negative that propagates (`Re(eps*mu) > 0`, and `Im(eps) < 0` or `Im(mu) < 0`): `rcwa.invalid`, both at check |
| `bottom` | `Material(epsilon=1.)` | the lower half-space, read the same way | as `top` |
| `period` | `(1., 1.)` when `lattice` is not given | `(Lx, Ly)`: two positive, finite, real lengths. The axis-aligned spelling of the cell | beside `lattice`: `stack.cell_ambiguous`, at construction; a scalar, a `dict` or a length that is not positive: `stack.period`, at check |
| `lattice` | `None` | the two primitive vectors as the columns of a `(2, 2)` array, or a `wavebit.Lattice`. The only spelling of a non-rectangular cell | beside `period`: `stack.cell_ambiguous` |

Region geometry and `period` differentiate on PyTorch and under eager `jax.grad`; under
`jax.jit` or `jax.vmap` the cell and every region geometry are refused with `prepared.host_topology`
([automatic differentiation](../guide/common/autodiff.md#rcwa-which-inputs-differentiate-on-which-backend)).

### `rcwa.Layer`

A layer is either painted with regions over a background or sampled as a raster, never both.

| Field | Default | Accepts | Refuses |
| --- | --- | --- | --- |
| `thickness` | required | a finite, real, nonnegative length | negative, complex or not finite: `layer.thickness`, at check |
| `background` | `Material(epsilon=1.)` on a region layer | the material model where no region paints | beside `epsilon`: `layer.ambiguous`, at construction; a bare number: `material.type`, at check |
| `regions` | `()` | a tuple of `rcwa.Region`; a later region paints over an earlier one | anything but `Region`s: `layer.regions`; beside `epsilon`: `layer.ambiguous`, both at construction |
| `epsilon` | `None` | a sampled relative-permittivity raster of shape `(Ny, Nx)`, complex allowed; one row is a 1D grating, and `(Ny, Nx, 3, 3)` holds a tensor per pixel. Every raster layer of a stack has the same shape | `Material` objects inside the array, or raster layers of different shapes: `rcwa.invalid`, at check |
| `mu` | `None` | the relative-permeability raster beside `epsilon`, the same shape. On a region layer a permeability belongs to each `Material` | without `epsilon`: `layer.mu_without_raster`, at construction |

### `rcwa.Region`

| Field | Default | Accepts | Refuses |
| --- | --- | --- | --- |
| `geometry` | required | an `rcwa.Rectangle`, `rcwa.Circle`, `rcwa.Ellipse` or `rcwa.Polygon` | anything else, a recipe `dict` included: `shape.type`, at construction |
| `material` | required | a material model, scalar or tensor | a bare number: `material.type`, at check |

### The geometries

Lengths are absolute, in the problem's unit, and `center` and `vertices` are measured from the
centre of the cell. A shape larger than the period, which runs into its own periodic image, is
refused at prepare with `rcwa.invalid`.

| Declaration | Field | Default | Accepts | Refuses |
| --- | --- | --- | --- | --- |
| `rcwa.Rectangle` | `size` | required | full widths `(wx, wy)` | not two components: `shape.dimension`, at construction |
| | `center` | `(0., 0.)` | `(x, y)` | not two components: `shape.dimension` |
| | `angle` | `0.` | counterclockwise rotation, radians | |
| `rcwa.Circle` | `radius` | required | a length | |
| | `center` | `(0., 0.)` | `(x, y)` | not two components: `shape.dimension` |
| `rcwa.Ellipse` | `radii` | required | principal semi-axes `(rx, ry)` | not two components: `shape.dimension` |
| | `center`, `angle` | `(0., 0.)`, `0.` | as on `Rectangle` | as on `Rectangle` |
| `rcwa.Polygon` | `vertices` | required | `N >= 3` pairs `(x, y)`, or an `(N, 2)` array, in order around the boundary; either winding | fewer than three: `shape.vertices`; an outline that is not simple (a zero-length edge, two edges crossing or touching): `shape.outline`, both at construction; under `factorization='li'` or `nvf_projector='circular'`, or with a tensor material: `rcwa.unsupported`, at check |

`factorization='li'` builds axis-aligned rectangles only; any other region under it is refused at
check with `rcwa.unsupported`.

## The light

Explained in [Declaring RCWA illumination](../guide/rcwa/light.md) and
[Coherent illumination and internal sources](../guide/rcwa/coherent-illumination.md).

### `rcwa.Excitation`

| Field | Default | Accepts | Refuses |
| --- | --- | --- | --- |
| `wavelength` | required | the vacuum wavelength: a positive, finite, real scalar, or a nonempty one-dimensional array or tuple of them. Results keep a leading wavelength axis either way | complex or not positive: `wavelength.value`; two-dimensional: `wavelength.shape`; empty: `wavelength.empty`, all at check |
| `sources` | required | a nonempty tuple of `PlaneWave`, `IncidentOrders`, `GaussianBeam` and `ElectricDipole`. One `PlaneWave` is the diffraction problem; anything else is a coherent-source run, which reports boundary power and requested fields | empty, or anything else (a side-keyed `dict` included): `excitation.sources`, at construction |

A coherent-source run takes `Outputs(diffraction=False)`, `algorithm='smm'`, no `detuning`, no
`scattering`, and values PyTorch or eager `jax.grad` can read: an input traced under `jax.jit`
or `jax.vmap` is refused with `rcwa.coherent_transform`.

### `rcwa.PlaneWave`

| Field | Default | Accepts | Refuses |
| --- | --- | --- | --- |
| `side` | `'top'` | `'top'` or `'bottom'`: the half-space the light arrives from. Scene labels never move; results are incidence-relative, and `theta` and `phi` are measured in the incidence medium | anything else (`'bot'`): `source.side`, at construction |
| `polarization` | `'s'` | `'s'`, `'p'`, or a complex Jones pair `(a_s, a_p)`. The pair is normalized to unit length and keeps its overall phase | another string (`'te'`): `source.polarization`, at construction; a pair that is zero, not finite or not two components: `source.polarization_value`, at check |
| `amplitude` | `1.` | the incident electric amplitude, V/m. Fields and powers scale with it; efficiencies and amplitude ratios do not | zero, not finite, or an array: `source.amplitude`, at check |
| `theta` | `None`, normal incidence | polar angle of the incident light's direction from the inward normal, radians: the direction its power travels in a lossless medium of either sign of index (in a negative-index medium the in-plane wavevector then points the other way). A nonzero imaginary part selects the formal complex-angle parameterization; see [real-angle incidence](common/conventions.md#real-angle-incidence) and [negative-index media](../theory/rcwa/negative-index-media.md) | beside `kx` or `ky`: `source.direction_ambiguous`, at construction |
| `phi` | `None`, read as `0.` | azimuth of the incident light's direction, radians | beside `kx` or `ky`: `source.direction_ambiguous` |
| `kx`, `ky` | `None` | the transverse angular wavenumbers in the inverse length unit, an alternative to `theta`/`phi`. These are physical values, not divided by `2π/λ`; an omitted component is zero | beside `theta` or `phi`: `source.direction_ambiguous` |

### `rcwa.IncidentOrders`

Several diffraction orders lit coherently around one reference direction. It is a coherent-source
run, so it reports boundary power rather than per-order efficiencies.

| Field | Default | Accepts | Refuses |
| --- | --- | --- | --- |
| `weights` | required | `(((m, n), weight), ...)`: integer order pairs, each with a complex weight, its relative electric amplitude at `x = y = 0` on the lit boundary. A 1D order `m` is written `(m, 0)` | a `dict`, an empty tuple, or a label that is not two Python integers: `source.weights` (a bare integer label raises a plain `TypeError`); a repeated label: `source.weights_duplicate`, all at construction; an order outside `Numerics(orders=...)`, or an evanescent or grazing one: `rcwa.invalid`, at run |
| `side`, `polarization`, `amplitude`, `theta`, `phi`, `kx`, `ky` | as `PlaneWave` | the reference plane wave whose orders are lit | as `PlaneWave` |

### `rcwa.GaussianBeam`

A propagating Gaussian angular spectrum. It needs a finite rectangle
`Outputs(power_region=((xmin, xmax), (ymin, ymax)))`.

| Field | Default | Accepts | Refuses |
| --- | --- | --- | --- |
| `waist` | required | the 1/e amplitude radius of the envelope, a positive length | not positive: `rcwa.invalid`, at check |
| `focus` | `(0., 0., 0.)` | `(x, y, z)` of the focus | not three components: `source.dimension`, at construction |
| `sampling` | `(9, 9)` | midpoint-quadrature points on the two axes transverse to the central direction | not two components: `source.dimension` |
| `angular_spectrum_extent` | `4.` | the half-width of that quadrature, in units of `1/waist` | not a positive, finite, real number: `source.angular_spectrum_extent`, at check |
| `side`, `polarization`, `amplitude`, `theta`, `phi`, `kx`, `ky` | as `PlaneWave` | the central plane wave; `amplitude` is the field at the focus | as `PlaneWave` |

### `rcwa.ElectricDipole`

A prescribed dipole inside a uniform layer, with `exp(-iωt)`. An isolated dipole needs a finite
rectangle `power_region`, as a Gaussian beam does.

| Field | Default | Accepts | Refuses |
| --- | --- | --- | --- |
| `position` | required | `(x, y, z)`, inside a uniform layer: no regions, or a raster of one material | not three components: `source.dimension`, at construction; a host layer that is not uniform: `rcwa.invalid`, at run |
| `moment` | required | `(px, py, pz)`, complex, in C·m | not three components: `source.dimension` |
| `repetition` | `'periodic'` | `'periodic'`, one dipole per cell; or `'isolated'`, one dipole, by Brillouin-zone quadrature | anything else: `source.repetition`, at construction |
| `bloch_k` | `None`, zero | periodic only: `(kx, ky)`, the phase between copies | beside `repetition='isolated'`: `rcwa.invalid`, at check |
| `sampling` | `(5, 5)` | isolated only: the quadrature grid. A finite grid is a larger repeated supercell and must be converged | |

## The numerics

Explained in [Choosing RCWA numerical settings](../guide/rcwa/numerics.md).

### `rcwa.Numerics`

The `None` defaults resolve per problem; `rcwa.check(...).resolved` reports what they resolved
to. An option set where nothing reads it is refused at check with `rcwa.unsupported`, rather
than ignored.

| Field | Default | Accepts | Refuses |
| --- | --- | --- | --- |
| `orders` | `(0, 0)` | the retained Fourier window `(Mu, Mv)`, the half-widths along the two lattice vectors, u first: `2*Mu + 1` orders along u and `2*Mv + 1` along v. An explicit tuple or list of two nonnegative integers (`int` or a NumPy integer) | anything else -- a bare integer, a `bool`, a `dict`, a set, a range or an iterator included: `numerics.orders`, at construction |
| `algorithm` | `'smm'` | `'smm'` (scattering matrix) or `'etm'` (enhanced transmittance matrix), lowercase. The same modes and answers; only `'smm'` forms a scattering matrix and runs coherent sources | anything else, `'ETM'` included: `numerics.choice`, at construction |
| `fourier_coefficients` | `None`: `'sampled'` on a raster, `'integrated'` on regions | `'sampled'` (series off the pixel grid) or `'integrated'` (closed-form transform of the raster's rectangles) | anything else: `numerics.choice`; `'sampled'` on a region layer: `rcwa.unsupported`, at check |
| `factorization` | `None`: `'li'` on a raster, `'nvf'` on regions | `'li'` (successive rules along v then u, with axes aligned to y and x, in the default `li_nesting`), `'nvf'` (normal vector field, factorizing along material interfaces) or `'laurent'` (the direct rule for every product -- opt-in, for comparing with solvers that use it); tensor support is described in [the material guide](../migration/materials-v1.md#anisotropic-materials) | anything else: `numerics.choice`; `'li'` on a region that is not an axis-aligned rectangle, and `'laurent'` on a tensor material: `rcwa.unsupported`, at check |
| `formulation` | `'auto'` | `'auto'` reads the problem: a material with a z row or column takes `'first_order'`, everything else `'second_order'`. Forcing `'first_order'` on a problem that does not need it is how the two are cross-checked | anything else (`'2ff'`): `numerics.choice`; `'second_order'` on a z-coupled tensor: `rcwa.invalid`, at run, because dropping the row answers a different problem while conserving energy to machine precision |
| `raster_normal_method` | `None`: `'distance'` | raster NVF only: `'distance'`, `'contour'`, `'spectral'` or `'smooth'` | anything else: `numerics.choice`; set on regions or under `'li'`: `rcwa.unsupported`, at check |
| `nvf_projector` | `None`: `'linear'` | raster and shape NVF: `'linear'` or `'circular'`; independent of incident polarization | anything else: `numerics.choice`; set without `'nvf'`, or `'circular'` on a polygon: `rcwa.unsupported`, at check |
| `shape_nvf_grid_size` | `None`: `max(1024, 4*Mu + 1, 4*Mv + 1)` | shape NVF only: projector sampling points per cell axis, a positive `int`; `128` means a 128×128 grid | not a positive `int`: `numerics.grid`, at construction; set where shape NVF does not read it (a raster, `'li'`, `'laurent'`): `rcwa.unsupported`, at check |
| `li_nesting` | `None`: `'fixed'` | Li only: the order Li's rules are nested in. `'fixed'` (`v` inside `u` for every slot) or `'symmetric'` (each slot's inverse-rule axis inside -- covariant under swapping x and y). Read on rasters and on axis-aligned rectangles | anything else: `numerics.choice`; set under `'nvf'` or `'laurent'`, or `'symmetric'` on a tensor material (tensor Li builds `'fixed'` only): `rcwa.unsupported`, at check |
| `detuning` | `None`: the requested point, exactly | a positive relative step. Where the requested point is numerically singular (an order on a light line), the solve is retried that far off -- the wavelength first, then the angle -- and `result.diagnostics['effective_wavelengths']` records what was solved | `0`, `True`, or a value that is negative or not finite: `numerics.detuning`, at construction; under `jax.jit` or `jax.vmap` (eager `jax.grad` runs it as PyTorch does): `prepared.detuning_transform`; with a coherent source: `rcwa.coherent_detuning`, at check |
| `half_space_merge_tolerance` | `0.0`: bit-identical materials only | how near a layer's material must be to the half-space beside it to be merged into it, a relative tolerance in `[0, 1)`. `0` merges only a layer whose every permittivity and permeability value is the half-space's bit for bit, and only at a complex in-plane wavevector, where solving it as a layer can be O(1) wrong; real angles keep their bits. A positive value takes a material within it **to be** the half-space's, at every angle, and the answer is that snapped scene's; `result.diagnostics['half_space_merge']` names the layers. See [connecting algorithms §10](../theory/rcwa/connecting-algorithms.md#10-etms-labels-at-a-complex-in-plane-wavevector) | negative, not below 1, or not finite: `numerics.half_space_merge_tolerance`, at construction; positive with a coherent source: `rcwa.coherent_half_space_merge`, at check; positive with an `IncidentOrders`, with a material that `jax.jit` or `jax.vmap` traces, or with a material derivative: refused at run, since the merge is decided from concrete materials (a traced angle alone is fine, and so are materials written as constants inside the jitted function) |

### NVF names changed on 2026-09-21

| Previous keyword | Current keyword | Reason |
| --- | --- | --- |
| `normal_method` | `raster_normal_method` | Identifies the raster-only field construction. |
| `normal_grid` | `shape_nvf_grid_size` | Identifies shape NVF and a number of samples, rather than a material raster or a Fourier order. |
| `projector` | `nvf_projector` | Identifies the NVF projection, separate from the incident polarization. |

The current names are the `rcwa.Numerics` fields above, and the earlier interface's `rcwa(...)`,
solver attributes and per-call settings such as `solve(...)` took them too. The previous names
are rejected; they are not aliases. Defaults, accepted option values and numerical algorithms
are unchanged. `linear` / `circular` remain projection choices, and are not renamed to
`real` / `complex`: the linear shape construction can also carry complex material-dependent
components.

:::{admonition} Earlier interface
:class: earlier-interface

The earlier `JobSpec` (`wavebit.compat.legacy`, retired on 2026-10-15) wrote schema version **10**. For a version-7
document, rename these keys in `numerics` when present and keep their option values. Also
reverse `stack.layers` and review the coordinate migration in [Conventions](common/conventions.md)
before setting version 10. The capability manifest has used the new option keys since version
**10**. Earlier JobSpec versions also require their own material and coordinate migrations.
:::

## Where the run happens

Explained in [Backends, devices and precision](../guide/common/backends.md).

### `wavebit.Execution`

`rcwa.check`, `rcwa.prepare` and `rcwa.run` take these three fields as keywords; an `Execution`
is how a `wavebit.Job` carries them. It is validated when a run uses it, not when it is built.

| Field | Default | Accepts | Refuses |
| --- | --- | --- | --- |
| `backend` | `'numpy'` | `'numpy'`, `'jax'` or `'torch'`, exactly, or a `wavebit.Backend` member | anything else (`'JAX'`, `1`): `execution.backend`; an extra that is not installed: `execution.backend_unavailable` |
| `device` | `'cpu'` | a string: `'cpu'`; on PyTorch also `'cuda'` or `'cuda:N'`; on JAX also `'gpu'` or `'tpu'`, optionally with `:N` | not a string (`0`): `execution.device`; NumPy off the CPU, PyTorch's `'mps'` or JAX's `'cuda'`: `execution.device`; a device that is not present: `execution.device_unavailable` |
| `precision` | `'double'` | `'double'` (complex128) or `'single'` (complex64) | anything else: `execution.precision`; JAX `'double'` before `jax.config.update("jax_enable_x64", True)`: `execution.jax_x64` |

## Two axis-order rules, and where they meet


**Every array axis runs `v` first. Every hand-written pair runs `u` (`x`) first.** No exceptions
either way. They meet at `Numerics(orders=...)`, which is a written pair, `(Mu, Mv)`:

```python
numerics = rcwa.Numerics(orders=(4, 1))        # (Mu, Mv): u first
result.R.shape                                  # (1, 3, 9) -- wavelength, then v, then u;
                                                # the leading 4 sized the TRAILING axis
result.reflection.orders_u, result.reflection.orders_v  # the order each column and each row is
```

`result.R.shape[1:] == (len(orders_v), len(orders_u))`, so zipping them the wrong way round is a
length mismatch rather than a plausible number. `u` runs along the cell's first basis vector and
`v` along the second — x and y on a rectangular cell. `result.reflection.order(m, n)` indexes by
order directly and needs none of this. `orders` takes the pair only; the named
`dict(u=..., v=...)` spelling of the earlier `truncation_order` has no current form.

The same rule holds for every public `orders=`: `rcwa.Numerics` and `fdtd.periodic.Numerics`
(`numerics.orders`), `fdtd.DiffractionMonitor` (`monitor.orders`, at construction) and the
`rcwa.experimental` builders (`experimental.orders`). Each takes an explicit pair of nonnegative
integers, NumPy integers included, and refuses a bare integer and a `bool`: a bare `M` meant
`(M, M)` to the FDTD monitor and the experimental builders but `(M, 0)` to `fdtd.periodic`, so
write the pair you mean (a one-dimensional period takes `(M, 0)`). `None`, where a field
defaults to it (`fdtd.DiffractionMonitor`, `fdtd.periodic.Numerics`), keeps that field's default
window.

## Named constants


`wavebit.Backend` is the one named constant the current modules export. Its members equal their
strings (`Backend.TORCH == 'torch'`), and it is accepted wherever a backend is:

```python
from wavebit import Backend, rcwa

result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(2, 2), factorization='nvf'),
                  backend=Backend.TORCH)
```

Write the string in a `wavebit.Job`: `Job.to_json()` and `Job.key` refuse an
`Execution(backend=Backend.TORCH)` with "Backend is not portable job data". The `rcwa.Numerics`
fields take their string literals. The earlier interface's other constants were importable only
from `wavebit.compat.legacy`, retired on 2026-10-15. `Answers` and `Tetm` name choices with no current field; the other
six -- `FourierCoefficients`, `Factorization`, `NormalMethod`, `Projector`, `Nesting` and
`Formulation` -- equal their strings, so `rcwa.Numerics` accepts them, but a `Job` refuses them
the same way.

## Refusal codes

Every structured failure -- a `ValidationError`, `UnsupportedFeatureError`,
`IncompatibleProblemError` or `NumericalError`, and the `Issue` a `check` report holds -- carries a
`.code` of one form, `area.item`: two or more lowercase dotted parts, each a snake-case word
(`[a-z][a-z0-9_]*`). The area is the declaration or subsystem the refusal's `.path` names
(`numerics`, `source`, `monitor`, `execution`, `prepared`, ...); the item says what is wrong with
it. A code is matched exactly; the `.path` says where, and the message says why in words that may
change.

Each solver module has a catch-all pair for an engine refusal no specific code describes, with
the engine's message kept: `rcwa.invalid` / `rcwa.unsupported`, `fdtd.invalid` /
`fdtd.unsupported`, `native.invalid` / `native.unsupported` (`fdtd.native`) and
`periodic.invalid` / `periodic.unsupported` (`fdtd.periodic`). The `.unsupported` code is an
`UnsupportedFeatureError` (an engine `NotImplementedError`); the `.invalid` code a
`ValidationError`. `check` reports the same code that `prepare`, `run` and `Prepared.run` raise.
A failure constructed without a code takes its class's default: `input.invalid`
(`ValidationError`), `feature.unsupported` (`UnsupportedFeatureError`), `problem.incompatible`
(`IncompatibleProblemError`) or `numerical.failure` (`NumericalError`); no Wavebit refusal relies
on one.

`prepared.structure` is the one code, in every solver, for a problem whose fixed layout differs
from the one a preparation fixed; its `.path` names the declaration that changed
(`scene.domain`, `sources.<name>`, `initial.E`, `stack.layers[1].regions`, `units.length`, ...).

| Area | Codes | Raised by |
| --- | --- | --- |
| `boundaries` | `boundaries.face` | `fdtd` |
| `check`, `validate`, `prepare` | `check.concrete`, `validate.concrete`, `prepare.concrete` | `rcwa` |
| `decay_check` | `decay_check.threshold`, `decay_check.window_seconds` | `fdtd.native`, `fdtd.periodic` |
| `dispersion` | `dispersion.poles` | `fdtd.periodic` |
| `excitation` | `excitation.sources` | `rcwa`, `fdtd.periodic` |
| `execution` | `execution.backend`, `execution.backend_unavailable`, `execution.device`, `execution.device_unavailable`, `execution.precision`, `execution.jax_x64`; `execution.cuda_scope` (outside the qualified FDTD CUDA route) | every solver; `fdtd` |
| `experimental` | `experimental.orders` | `rcwa.experimental` |
| `fdtd` | `fdtd.invalid`, `fdtd.unsupported`, `fdtd.padding_continuation` | `fdtd` |
| `field` | `field.coordinates` | `rcwa` |
| `geometry` | `geometry.cell_snapped` (a warning) | `fdtd` |
| `initial` | `initial.type` | `fdtd` |
| `job`, `record` | `job.encoding`, `job.schema`, `record.format` | `wavebit.Job`, run records |
| `layer` | `layer.ambiguous`, `layer.mu_without_raster`, `layer.regions`, `layer.thickness` | `rcwa` |
| `material` | `material.type`; `material.scalar`, `material.shape`, `material.poles`, `material.wavelength_range`, `material.unknown_field`; `material.engine_model`, `material.causal_model`, `material.pole_type`, `material.fdtd` (lowering a material to time domain) | `rcwa`, `fdtd.periodic`; `wavebit.materials`; `fdtd` |
| `medium` | `medium.poles` | `fdtd.native` |
| `migration` | `migration.shape`, `migration.shape_fields`, `migration.layer_fields`, `migration.layer_mu`, `migration.solver` | `wavebit.migration` |
| `monitor` | `monitor.type`, `monitor.geometry`, `monitor.plane`, `monitor.plane_span`, `monitor.extent`, `monitor.outside_domain`, `monitor.source`, `monitor.orders` | `fdtd` |
| `native` | `native.invalid`, `native.unsupported` | `fdtd.native` |
| `numerics` | `numerics.type`, `numerics.choice`, `numerics.orders`, `numerics.grid`, `numerics.detuning`, `numerics.half_space_merge_tolerance` | `rcwa` (`numerics.type`, `numerics.choice` and `numerics.orders` also FDTD) |
| `numerics` (FDTD) | `numerics.absorber`, `numerics.checkpoint_steps`, `numerics.checkpoint_scope`, `numerics.discretization`, `numerics.duration`, `numerics.geometry_ranges`, `numerics.junction`, `numerics.memory_bound`, `numerics.stopping`; `numerics.time_step_seconds`; `numerics.spacing`, `numerics.duration_seconds`, `numerics.transverse_spacing`, `numerics.reference_scope` | `fdtd`; `fdtd.native`; `fdtd.periodic` |
| `observation` | `observation.empty`, `observation.order_labels`, `observation.shape`, `observation.wavelength_axis` | `wavebit.DiffractionOrders` |
| `operators` | `operators.type`, `operators.shape`, `operators.slots`, `operators.spectral_shape`, `operators.source_groups`, `operators.detuning`, `operators.execution`, `operators.physical_contract`, `operators.physical_transform` | `rcwa.operators` |
| `outputs` | `outputs.boolean`, `outputs.unavailable` | `rcwa` |
| `periodic` | `periodic.invalid`, `periodic.unsupported` | `fdtd.periodic` |
| `prepared` | `prepared.structure`; `prepared.static_input`, `prepared.dependencies`; `prepared.topology`, `prepared.host_topology`, `prepared.host_topology_scope`, `prepared.source_grouping`, `prepared.detuning_transform`, `prepared.shape_topology`, `prepared.rectilinear_topology`, and the runtime guards `prepared.finite_inputs`, `prepared.positive_wavelength`, `prepared.nonnegative_thickness`, `prepared.nonzero_polarization`, `prepared.raster_partition` | every solver; `fdtd`; `rcwa` |
| `problem` | `problem.type`, `problem.leaf` | every solver; `rcwa` |
| `pulse` | `pulse.frequency_sigma_hz` | `fdtd.periodic` |
| `rcwa` | `rcwa.invalid`, `rcwa.unsupported`, `rcwa.singular`, `rcwa.nonfinite`, `rcwa.host_geometry`, `rcwa.mixed_layer_encoding`, `rcwa.raster_partition_transform`, `rcwa.scattering_algorithm`, `rcwa.coherent_detuning`, `rcwa.coherent_diffraction`, `rcwa.coherent_half_space_merge`, `rcwa.coherent_scattering`, `rcwa.coherent_transform` | `rcwa` |
| `scene` | `scene.type`, `scene.structure_name`, `scene.structure_name_duplicate` | `fdtd` |
| `shape` | `shape.type`, `shape.dimension`, `shape.outline`, `shape.vertices` | `rcwa`, `fdtd` (`shape.type`) |
| `snapshots` | `snapshots.every`, `snapshots.components`, `snapshots.slices` | `fdtd.native`, `fdtd.periodic` |
| `source` | `source.type`, `source.name`, `source.name_duplicate`, `source.pulse`, `source.plane`, `source.plane_span`, `source.box`, `source.outside_domain`, `source.waveform_range`; `source.amplitude`, `source.direction`, `source.polarization_value`; `source.side`, `source.repetition`, `source.dimension`, `source.direction_ambiguous`, `source.polarization`, `source.polarization_shape`, `source.weights`, `source.weights_duplicate`, `source.angular_spectrum_extent` | `fdtd`; `rcwa` and `fdtd.periodic`; `rcwa` |
| `stack` | `stack.layers`, `stack.period`, `stack.cell_ambiguous`, `stack.mixed_layer_encoding` | `rcwa`, `fdtd.periodic` |
| `structure` | `structure.concrete`, `structure.layer`, `structure.nonfinite`, `structure.quantity`, `structure.samples`, `structure.wavelength` | `rcwa.sample_structure` |
| `units` | `units.type`, `units.length` | every solver |
| `wavelength` | `wavelength.value`, `wavelength.empty`, `wavelength.shape` | `rcwa`, `fdtd.periodic`, `wavebit.materials` |

## Earlier spellings

Each keyword of the earlier `rcwa(...)` constructor, and the solver attributes and per-call
settings that took its place in a run, against the declaration field that holds it now. The
scene, light and numerics rows were checked by solving one problem both ways: efficiencies,
amplitudes and scattering matrices agree to round-off (3e-13 or better). The
[earlier API reference](../migration/legacy-api.md) documents the constructor itself.

| Earlier keyword | Current home | Notes |
| --- | --- | --- |
| `backend` | `Execution(backend=...)`, or `backend=` on `rcwa.check` / `prepare` / `run` | the same three names |
| `device` | `Execution(device=...)` | a string only; the integers `0` / `1` and a backend's own device object are refused |
| `complex_dtype` | `Execution(precision=...)` | `'double'` / `'single'`; `0` / `1` and the namespaces' dtypes are refused |
| `wavelength` | `Excitation(wavelength=...)` | may now be a spectrum |
| `period` | `Stack(period=...)` | the pair only; `dict(x=..., y=...)` is refused |
| `lattice` | `Stack(lattice=...)` | |
| `structure`, a raster | `Layer(epsilon=...)`, one `(Ny, Nx)` raster per layer | |
| `structure`, a list of shape recipes | `Layer(background=..., regions=(Region(geometry=..., material=...), ...))` | `cx`, `cy` → `center=(cx, cy)`; a `'rectangle'` → `Rectangle(size=(lx, ly), angle=...)`, a `'circle'` → `Circle(radius=lx/2)`, an `'ellipse'` → `Ellipse(radii=(lx/2, ly/2), angle=...)`, a `'polygon'` → `Polygon(vertices=...)` |
| `thickness` | `Layer(thickness=...)`, one per layer | |
| `mu` | `Layer(mu=...)`, beside `epsilon` | |
| `top` | `Stack(top=...)` | a material model; the bare-permittivity shorthand is refused |
| `bot` | `Stack(bottom=...)` | as `top` |
| `incidence_side` | `PlaneWave(side=...)` | |
| `theta`, `phi` | `PlaneWave(theta=..., phi=...)` | |
| `kx`, `ky` | `PlaneWave(kx=..., ky=...)` | |
| `polarization` | `PlaneWave(polarization=...)` | also `'s'` and `'p'` |
| `psi`, `chi` | **none** | the same light is the Jones pair `polarization=(sin ψ cos χ + i cos ψ sin χ, cos ψ cos χ − i sin ψ sin χ)` (`chi = 0` is `(sin ψ, cos ψ)`), which agrees to round-off; the two-angle chart itself, without the pair's redundant scale and phase, has no field |
| `tetm` | **none** | no field declares the polarization away, so a one-row raster always solves on the conical path. The numbers are the same (to 3e-13), but in one measurement at `orders=(10, 0)` to `(30, 0)` the run was 2.2 to 2.9 times slower than the earlier pure TE/TM path |
| `excitation`, an order map | `IncidentOrders(weights=(((m, n), weight), ...))` | `{0: 1.0}` on 1D becomes `((0, 0), 1.0)`. A coherent-source run: its total boundary power agrees with the earlier per-order sums, but **per-order efficiencies have no current form** (`rcwa.coherent_diffraction`) |
| `excitation`, a side-keyed map or a source list | `Excitation(sources=(...))` | |
| `stack=`, `incidence=` | `rcwa.Stack`, `rcwa.PlaneWave` | |
| `truncation_order` | `Numerics(orders=...)` | the pair `(Mu, Mv)` of nonnegative integers only; a bare integer and the `dict` spellings are refused |
| `layer_algorithm` | `Numerics(algorithm=...)` | lowercase. **The default moved** from ETM to `'smm'` |
| `fourier_coefficients`, `factorization`, `raster_normal_method`, `nvf_projector`, `li_nesting`, `shape_nvf_grid_size` | `Numerics(...)`, the same names | the same values and defaults |
| `formulation`, a solver attribute | `Numerics(formulation=...)` | |
| `solve(auto_detune=...)` | `Numerics(detuning=...)` | **The default moved**: the earlier `solve()` retried at a relative step of `1e-12`; `Numerics()` solves the requested point. `detuning=1e-12` asks for the earlier default step |
| `answers` | `Outputs(fields=...)` | `'field'` is `fields=True`; `'amplitude'` and `'efficiency'` are both `fields=False`, and the result carries amplitudes either way |
| `solver.scattering` | `Outputs(scattering=True)`, then `result.scattering` | one matrix per wavelength |
| `power_region` | `Outputs(power_region=...)` | |
| `length_unit` | `Problem(units=Units(length=...))` | one of `'m'`, `'mm'`, `'um'`, `'nm'` rather than a free factor |

Three earlier keywords have no current home: `psi`, `chi` and `tetm`. The order-map
`excitation` has one, without its per-order efficiencies. A problem that needs one of those
keeps the [earlier interface](../migration/legacy-api.md) for now.

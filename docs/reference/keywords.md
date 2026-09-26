# Keyword reference

Every keyword `wavebit.rcwa()` takes, what it accepts and what it defaults to. Grouped by the
question it answers rather than by the constructor's order, which is chronological.

This page is the lookup. For what the choices *mean* and how to pick between them, the user
guide takes them one subject at a time: [the light](../guide/light.md),
[the scene](../guide/scene.md), [the numerics](../guide/numerics.md),
[near fields](../guide/fields.md) and [backends](../guide/backends.md).

Two rules cover the whole surface and are worth stating once:

* **An unknown keyword is refused, not swallowed.** `rcwa(thicknes=300.)` raises and names the
  nearest real keyword. So does assignment after construction -- `solver.thicknes = 300.` is a
  refusal, not a new attribute.
* **A retired spelling is refused by name**, with the refusal saying that the *quantity* moved
  too. `ucell`, `n_top`, `n_bot`, `n_base` and `n_index` all held a refractive index; the
  fields that replaced them hold a relative permittivity, and `3.5` reread as a permittivity is
  a legal, energy-conserving, different medium.

## The light

Explained in [Saying what the light is](../guide/light.md).

| Keyword | Default | What it takes |
| --- | --- | --- |
| `wavelength` | `1.` | free-space wavelength, in whatever unit `period` and `thickness` share |
| `theta` | `0.` in angle mode; `None` with direct k | phase polar angle from the inward normal, radians. Real angles give uniform surface illumination, including in absorbing incident media. A nonzero imaginary part selects the formal complex-angle parameterization; see [phase incidence](../conventions.md#phase-incidence) |
| `phi` | `None` | azimuth of the plane of incidence, radians. `None` is "not named", which the pure 1D path requires |
| `kx`, `ky` | `None` | direct real Cartesian transverse angular wavenumbers, in inverse length units; an omitted component is zero. Alternative to `theta`/`phi`; these are physical values, not divided by k0. See [direction inputs](../guide/light.md#two-ways-to-specify-direction) |
| `tetm` | `None` | `'te'` / `'tm'`. A **declaration** that polarization and azimuth are not degrees of freedom here; 1D structures only, and it is what buys the pure TE/TM formulation |
| `psi` | `pi/2` (s/TE) | polarization angle, radians. `pi/2` is s/TE, `0` is p/TM |
| `chi` | `None` | ellipticity angle. With `psi` it is the canonical chart of any state: `0` linear, `±pi/4` circular |
| `polarization` | `None` | the Jones pair `(a_s, a_p)` itself, complex. The only spelling that carries an overall phase; magnitude is normalized away |
| `excitation` | `None` | which diffraction *orders* the light arrives on: `{0: 1.0, 1: 0.6-0.8j}` on 1D, `{(m, n): amp}` on 2D. `None` is one plane wave on the specular order. A side-keyed map, `{"top": wavebit.PlaneWave(...), "bottom": wavebit.PlaneWave(...)}`, selects [coherent illumination](../coherent-illumination.md) |
| `incidence_side` | `'top'` | `'top'` / `'bottom'` — which half-space the light arrives from. Scene labels never move; results are incidence-relative, and `theta` / `phi` are measured in the incidence medium |

## The stack

Explained in [Describing a scene](../guide/scene.md).

| Keyword | Default | What it takes |
| --- | --- | --- |
| `structure` | `None` | the layers. An `ndarray` of shape `(n_layers, Ny, Nx)` is a **raster** of relative permittivity; a `list` of `dict(background=..., shapes=[...])` is the **parametric** path |
| `thickness` | `(0.,)` | one length per layer, same unit as `period` |
| `top` | `1.` | superstrate material — a `Material`, or a bare relative permittivity |
| `bot` | `1.` | substrate, read the same way |
| `mu` | `None` | **raster path only**: the permeability field, `(n_layers, Ny, Nx)`, beside `structure`. On the shape paths a permeability belongs to each `Material`, so naming it here is refused |

## The cell

Explained in [Describing a scene](../guide/scene.md), "The cell".
`period` and `lattice` are two spellings of one cell: naming **both** raises, and naming
**neither** while giving a structure raises too.

| Keyword | Default | What it takes |
| --- | --- | --- |
| `period` | unset | `(Lx, Ly)`, a list or `dict(x=..., y=...)`. The axis-aligned **spelling** of a cell |
| `lattice` | `None` | the two primitive vectors as the columns of a `(2, 2)`, or a `Lattice` object. The only spelling that can describe a non-rectangular cell |

## The numerics

Explained in [Choosing the numerics](../guide/numerics.md), except
`backend`, `device` and `complex_dtype`, which are
[Backends, devices and precision](../guide/backends.md), and `answers`, which is
[Near fields](../guide/fields.md).

| Keyword | Default | What it takes |
| --- | --- | --- |
| `truncation_order` | `(0, 0)` | retained Fourier orders per axis. A pair, `dict(x=..., y=...)` or `dict(u=..., v=...)` |
| `backend` | `0` | `0` NumPy, `1` JAX, `2` PyTorch — see [Backends and precision](../guide/backends.md) |
| `device` | `0` | `0`/`'cpu'`, `1`/`'gpu'`/`'cuda'`, or the backend's own device object |
| `complex_dtype` | `0` (double) | `0` double, `1` single, or any of the three namespaces' complex spellings |
| `layer_algorithm` | `'ETM'` | `'ETM'` (enhanced transmittance matrix) or `'SMM'` (scattering matrix) |
| `fourier_coefficients` | resolves | `'sampled'` (series off the pixel grid) or `'integrated'` (closed-form transform of the raster's rectangles) |
| `factorization` | resolves | `'li'` (successive rules along v then u, with axes aligned to y and x, in the default `li_nesting`) or `'nvf'` (normal vector field, factorizing along material interfaces); tensor support is described in [the material guide](../guide/materials.md#anisotropic-materials) |
| `raster_normal_method` | `'distance'` | raster NVF only: `'distance'`, `'contour'`, `'spectral'`, `'smooth'` |
| `nvf_projector` | `'linear'` | raster and shape NVF projection: `'linear'` or `'circular'`; independent of incident polarization |
| `li_nesting` | `'fixed'` | Li only: the order Li's rules are nested in. `'fixed'` (`v` inside `u` for every slot) or `'symmetric'` (each slot's inverse-rule axis inside -- covariant under swapping x and y). Read on rasters and on axis-aligned rectangles; tensor Li builds `'fixed'` only |
| `shape_nvf_grid_size` | `max(1024, 4*Nx+1, 4*Ny+1)` | shape NVF only: projector sampling points per cell axis; `128` means a 128×128 grid |
| `answers` | `'amplitude'` | what the solve keeps the state to answer — see [Near fields](../guide/fields.md) |

### NVF names changed on 2026-09-21

| Previous keyword | Current keyword | Reason |
| --- | --- | --- |
| `normal_method` | `raster_normal_method` | Identifies the raster-only field construction. |
| `normal_grid` | `shape_nvf_grid_size` | Identifies shape NVF and a number of samples, rather than a material raster or a Fourier order. |
| `projector` | `nvf_projector` | Identifies the NVF projection, separate from the incident polarization. |

Use the current names in `rcwa(...)`, solver attributes, per-call settings such as
`solve(...)`, and `Numerics`. The previous names are rejected; they are not aliases.
Defaults, accepted option values and numerical algorithms are unchanged.
`linear` / `circular` remain projection choices, and are not renamed to `real` / `complex`:
the linear shape construction can also carry complex material-dependent components.

JobSpec now writes schema version **10**. For a version-7 document, rename these keys in
`numerics` when present and keep their option values. Also reverse `stack.layers` and
review the coordinate migration in [Conventions](../conventions.md) before setting version 10. The capability
manifest has used the new option keys since version **10**. Earlier JobSpec versions also
require their own material and coordinate migrations.

## Two axis-order rules, and where they meet


**Every array axis runs `v` first. Every hand-written pair runs `x` first.** No exceptions
either way. They meet at `truncation_order`, which is a written pair:

```python
truncation_order = [4, 1]             # x first
result.R.shape                        # (3, 9) -- v first; the leading 4 sized the TRAILING axis
truncation_order = dict(u=4, v=1)     # named, so there is no position to misread
result.orders_u, result.orders_v      # the order each column and each row is
```

`R.shape == (len(r.orders_v), len(r.orders_u))`, so zipping them the wrong way round is a length
mismatch rather than a plausible number. `u` runs along the cell's first basis vector and `v`
along the second — x and y on a rectangular cell. `result.order(m, n)` indexes by order
directly and needs none of this.

## Named constants


The integer and string codes have enum equivalents, if you prefer them to the literals:

```python
from wavebit import (Answers, Backend, Tetm, FourierCoefficients, Factorization, NormalMethod,
                     Projector, Nesting)

solver = wavebit.rcwa(backend=Backend.JAX, tetm=Tetm.TM, factorization=Factorization.NVF, ...)
```

`Formulation` is the ninth and is not a constructor keyword: it is a settable attribute
defaulting to `'auto'`, which reads the problem — a solve handed any of the four z-row
convolution slots takes `'4ff'`, everything else `'2ff'`. Forcing `'4ff'` on a problem that
does not need it is how the two are cross-checked; forcing `'2ff'` on a z-coupled tensor raises
rather than dropping the row, because dropping it answers a different problem while conserving
energy to machine precision.

For side-keyed excitation, `power_region` chooses `"cell"`, `"infinite"`, or
`((xmin, xmax), (ymin, ymax))`. It is required when the beams occupy different
Bloch sectors. PlaneWave direction and polarization arguments are documented in
[coherent illumination](../coherent-illumination.md).

Typed source lists use `excitation=[PlaneWave(...), GaussianBeam(...), ElectricDipole(...)]`.
An internal dipole requires `length_unit`, the metres per scene length.
Gaussian and isolated-dipole sources require a finite rectangular `power_region`.
See [source objects and examples](../coherent-illumination.md#source-objects).

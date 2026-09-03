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
| `theta` | `0.` | polar angle from the normal, radians. May be complex — evanescent incidence, float64 only |
| `phi` | `None` | azimuth of the plane of incidence, radians. `None` is "not named", which the pure 1D path requires |
| `tetm` | `None` | `'te'` / `'tm'`. A **declaration** that polarization and azimuth are not degrees of freedom here; 1D structures only, and it is what buys the pure TE/TM formulation |
| `psi` | `pi/2` (s/TE) | polarization angle, radians. `pi/2` is s/TE, `0` is p/TM |
| `chi` | `None` | ellipticity angle. With `psi` it is the canonical chart of any state: `0` linear, `±pi/4` circular |
| `polarization` | `None` | the Jones pair `(a_s, a_p)` itself, complex. The only spelling that carries an overall phase; magnitude is normalized away |
| `excitation` | `None` | which diffraction *orders* the light arrives on: `{0: 1.0, 1: 0.6-0.8j}` on 1D, `{(m, n): amp}` on 2D. `None` is one plane wave on the specular order |
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
| `factorization` | resolves | `'li'` (Li's rules along x and y) or `'nvf'` (normal vector field, factorizing along the material interfaces) |
| `normal_method` | `'distance'` | raster NVF only: `'distance'`, `'contour'`, `'spectral'`, `'smooth'` |
| `projector` | `'linear'` | how `P` is formed from that field: `'linear'` (`P = n n^T`) or `'circular'` |
| `normal_grid` | `max(1024, 4*to+1)` | normal-field sampling, per axis |
| `answers` | `'amplitude'` | what the solve keeps the state to answer — see [Near fields](../guide/fields.md) |

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
from wavebit import Answers, Backend, Tetm, FourierCoefficients, Factorization, NormalMethod, Projector

solver = wavebit.rcwa(backend=Backend.JAX, tetm=Tetm.TM, factorization=Factorization.NVF, ...)
```

`Formulation` is the eighth and is not a constructor keyword: it is a settable attribute
defaulting to `'auto'`, which reads the problem — a solve handed any of the four z-row
convolution slots takes `'4ff'`, everything else `'2ff'`. Forcing `'4ff'` on a problem that
does not need it is how the two are cross-checked; forcing `'2ff'` on a z-coupled tensor raises
rather than dropping the row, because dropping it answers a different problem while conserving
energy to machine precision.

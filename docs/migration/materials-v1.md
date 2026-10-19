# Materials and wavelength sweeps

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


:::{note}
This is the complete compatibility reference for the earlier interface. Use `wavebit.compat.legacy` explicitly. Start new applications with the [current lifecycle](../guide/common/lifecycle.md) and [current API reference](../reference/api.md). Numerical evidence quoted here retains its original scope.
:::

Assign a material to a region once. Wavebit evaluates its relative permittivity
and optional relative permeability at each RCWA solve wavelength.

## Select a material

```python
from wavebit.compat import legacy as wavebit

silicon = wavebit.materials.catalog.load('Si')
solver = wavebit.rcwa(
    wavelength=900., period=(600., 600.), thickness=[120.],
    structure=[dict(background=silicon, shapes=[])],
    truncation_order=(0, 0), polarization=(1., 0.), factorization='li',
)
for wavelength in (800., 900., 1000.):
    result = solver.solve(wavelength=wavelength)
    print(wavelength, result.R.sum(), result.T.sum())
```

`materials.load` always returns an immutable `TabulatedMaterial`. It does not take
a wavelength. Only the selected data file is loaded, and repeated selections use
a cache. Table loading does not depend on the numerical backend or precision.

```python
wavebit.materials.catalog.available()             # source-qualified catalog identifiers
silicon = wavebit.materials.catalog.load('Green2008-Li1980/Si')
value = silicon.at(900.)                   # numerical Material at one wavelength
spectrum = silicon.at([800., 900., 1000.]) # Material containing epsilon arrays
print(value.epsilon, silicon.wavelength_range)
```

Names are case-insensitive. Short names such as `Si` work when unambiguous;
qualified names identify a specific dataset. The catalog contains
`Green2008-Li1980/Si`, `Luke2015/Si3N4`, `Malitson1962/Al2O3` and `Malitson1965/SiO2`,
each named after the publications its numbers come from (2026-10-19). The earlier
`filmetrics/...` tables carried no recorded source or terms and were removed; loading
those names raises an error that lists the replacements.
A formula does not specify all deposition conditions or forms of a material;
choose data appropriate to the sample. An older `p_Si.mat` asset that declared neither
its wavelength unit nor its data source was removed from the package on 2026-10-14.

## Choose the scene's length unit

The default is **nm**. For a scene expressed in micrometres:

```python
silicon = wavebit.materials.catalog.load('Si', wavelength_unit='um')
value = silicon.at(0.9)
# Use wavelength=0.9, period=0.6, thickness=[0.12], etc. in that scene.
```

`wavelength_unit` accepts `nm`, `um`, `mm` or `m`. It converts the table's
wavelength axis to the receiving scene's unit. It never changes your geometry or
solver wavelength. Numeric scene inputs carry no units; all lengths must agree.
Custom tables likewise use the same length unit as the scene.

## Definitions and evaluated values

| Input | Meaning when the solve wavelength changes |
| --- | --- |
| `Material(12.25)` or `12.25` | Constant relative permittivity |
| `materials.load('Si')` | Reevaluate the retained material definition |
| `silicon.at(900.)` | Fixed numerical `Material` evaluated at 900 nm |
| `TabulatedMaterial(wavelengths, epsilon)` | Reevaluate a custom permittivity table |

`Material(epsilon, mu=None)` is the numerical value container. `mu=None` selects
the non-magnetic path, physically relative permeability one. Supplying `mu=1`
explicitly selects the magnetic path. `Material.at(wavelength)` returns the same
constant object; it does not broadcast a constant to a wavelength array.

A `Material` whose permittivity and permeability both have negative real parts, such as
`Material(-2 + 0.1j, mu=-1 + 0.1j)`, is a double negative: a negative-index medium. It works as
either half-space, lit or not, and in any layer; an active one (gain) that propagates is refused.
Its index, as `solver.n_top` / `n_bot` report it, is `sqrt(eps)*sqrt(mu)` with `Re(n) < 0`, and
`theta` from such a side still names the direction the light travels. [Negative-index
media](../theory/rcwa/negative-index-media.md) explains what that changes and why.

Models work in shape `material`, layer `background`, and `top` / `bot` slots.
`solver.top`, `solver.bot` and `solver.structure` expose their evaluated numerical
values at the current wavelength. To change a region, assign `top`, `bot` or a
new `structure`; do not edit those evaluated dictionaries in place.

Raster `structure` and `mu` arrays are numerical fields. They do not retain a
material identity per pixel. For a raster sweep, evaluate materials with
`.at(wavelength)` and rebuild the arrays at each wavelength. Similarly,
`solve_conv` consumes already-built matrices and does not rebuild them for you.

## Anisotropic materials

Use the same `Material` assignment for scalar, diagonal and full tensor data.
Tensor components refer to Wavebit's x, y and z axes. The constitutive relation
is `D = epsilon_0 * epsilon_r @ E`.

Anisotropy describes a material's response to different field directions at one
point. Spatial variation is a separate property: a uniform anisotropic layer
has the same material tensor everywhere in its xy plane. An anisotropic cylinder
in air is a patterned structure containing an anisotropic material, because the
material changes between the cylinder and its surroundings. The examples below
show these two cases in that order.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

crystal = wavebit.Material(epsilon=[2.25, 4., 2.25])
rotated = crystal.rotated(np.deg2rad(30.), axis='z')
# Equivalent full tensor spelling, with physical signs:
full = wavebit.Material(epsilon=[[2.25, 0., 0.],
                                [0., 4., 0.],
                                [0., 0., 2.25]])
solver = wavebit.rcwa(
    wavelength=600., period=(600., 600.), thickness=[220.],
    structure=[{'background': crystal, 'shapes': []}],
    truncation_order=(0, 0), phi=0.,
)
result = solver.solve()
print(float(result.s_inc.R.sum()))  # 0.3574726625
print(float(result.p_inc.R.sum()))  # 0.0163080252
```

Three values are the xx, yy and zz diagonal entries. A 3×3 array may include
independent off-diagonal entries; it is not automatically symmetrized.
`.rotated(angle, axis='x'|'y'|'z')` applies an active right-handed rotation in
radians, `R @ tensor @ R.T`, to both epsilon and optional mu. The result is a
new material; the shape and lattice do not rotate. Several calls apply rotations
in their written order. A scalar remains scalar. This method preserves live
JAX/PyTorch material and angle derivatives.

A pattern uses the same assignment:

```python
solver = wavebit.rcwa(
    wavelength=700., period=(600., 600.), thickness=[170.],
    structure=[{'background': 1., 'shapes': [
        {'type': 'circle', 'lx': 270., 'material': rotated},
    ]}],
    truncation_order=(3, 3), shape_nvf_grid_size=128,
)
result = solver.solve()
solver.plot_structure(component='xx')
```

The same shape `material` and layer `background` slots accept scalar or tensor
materials. This example places a 270 nm diameter, 170 nm tall anisotropic cylinder
in a background with relative permittivity one. Rotating the material axes does
not change the cylinder's shape. `component='xx'` displays the xx component of
its relative permittivity; use another component to inspect a different entry.

`shape_nvf_grid_size=128` controls NVF sampling. This grid size and the Fourier
order are starting values, not a convergence result. See
[choosing the numerics](../guide/rcwa/numerics.md) for coefficient construction and sampling.

A tensor raster stores a 3×3 material matrix in each spatial pixel. Its shape is
**`(layers, Ny, Nx, 3, 3)`**, with physical constitutive components in the final
two axes. `mu` can be another tensor raster
or a scalar field on the same spatial grid. A scalar epsilon field can likewise
be paired with tensor mu. Raster arrays are evaluated values, not material
models, and layers remain ordered bottom to top.

| Material and geometry | Supported route |
| --- | --- |
| Uniform diagonal or full epsilon/mu tensor | Assign `Material`; `solve()` builds exact uniform blocks |
| Uniform epsilon or mu with xz/yz/zx/zy coupling | Shape-free background or constant tensor raster; `formulation='auto'` selects `'first_order'` |
| Full epsilon and/or mu patterns | Linear NVF by default; Cartesian Li for rasters and axis-aligned rectangle paintings |
| Tensor wavelength table | `TabulatedMaterial` with a diagonal or 3×3 tensor at each wavelength |
| Pattern with z coupling, including explicit mu | Full three-component factorization and the first-order formulation; ETM or SMM layer algorithm |
| Anisotropic `top` or `bot` half-space | Not supported; half-spaces remain scalar |

Patterned structures containing anisotropic materials default to
`factorization='nvf'` and `nvf_projector='linear'`.
These are selected by default for tensor inputs. NVF uses the material-interface
normal to construct the material–field products; `linear` chooses the projector
construction, not the incident polarization. An explicit `factorization='li'`
supports tensor rasters and axis-aligned rectangle paintings inside an
axis-aligned rectangular cell, with `li_nesting='fixed'` only. Both rules retain all nine components of epsilon
and mu. `factorization='laurent'` is refused on any tensor material, uniform tensors included:
no tensor Laurent build is written, and a uniform layer is the same under every rule, so leave
the keyword unset there. A z coupling selects the first-order solver automatically. See
[full tensor patterns](../guide/rcwa/full-tensor-patterns.md) for examples and validation.
See [tensor Li construction and usage](../theory/rcwa/tensor-li-factorization.md).
For tensor shapes under NVF, `fourier_coefficients='sampled'` also supports periodic painting
that the integrated route currently refuses: shapes overlapping their own images,
or each other through a cell edge. See
[tensor shape integration](../guide/rcwa/tensor-shape-integration.md) for these limits and
the numerical construction. Incompatible explicit options raise.
A uniform anisotropic layer can be assigned as the `background` of a layer with
`shapes=[]`. To represent it as a raster, store the same material matrix in every
spatial pixel. Uniform layers with nonzero xz/yz/zx/zy entries are supported as
a shape-free background or a 1×1 spatial raster.

`plot_structure` and `slice_structure` require a tensor
`component`, such as `'xx'` or `'xy'`. They do not assign a single refractive
index to an anisotropic medium or display a band-limited tensor approximation.

For JAX tracing of shape patterns or raster NVF, call `freeze_geometry()` once with concrete data
and reuse that record through `solve(geometry=...)`. Integrated NVF shapes support
material, position, size, angle and lattice derivatives while the stored
intersection and periodic-interface topology remains valid. Eager replay checks
that topology; JIT cannot replan it, so re-freeze outside a trace when moving
through a contact or overlap change, or when an ellipse grows out of the
Wigner-Seitz cell of its centre. The sampled normal extension still steps where
it switches -- at ties between periodic instances of a curve, between shapes in
the contrast-weighted blend, and at corners -- and those steps bias a gradient
by an amount grid refinement does not reliably remove. Check a gradient against
the loss over a window of the design parameter: see [autodiff](../guide/common/autodiff.md).

The explicit sampled shape route and tensor rasters under NVF retain fixed geometry:
material derivatives are supported, shape and lattice derivatives are refused.
Their geometry, normal method and material-interface changes require a new
record. Changing Fourier orders, shape grid or coefficient mode also requires
a new record for either shape route. Tensor Li rasters need no frozen normals
and support material and rectangular-period derivatives. Li rectangles support
material, position, size and period derivatives with the frozen edge order;
their orientation stays axis-aligned. Table samples and table
rotations are fixed data; use `.at(wavelength).rotated(angle)` when optimizing
an angle. Uniform material/angle, thickness and wavelength derivatives remain
available within the selected solver's differentiation support. A traced full
tensor keeps all nine slots, including the derivatives of currently zero
couplings. See [full tensor patterns](../guide/rcwa/full-tensor-patterns.md).

`solve_conv` remains available for callers supplying their own matrices. Its
**off-diagonal slots carry minus the physical component**; `Material` and
rasters always receive the physical component itself. See the
[slot convention](../theory/rcwa/off-diagonal-slot-convention.md),
[tensor factorization](../theory/rcwa/patterned-anisotropic-factorization.md) and
[first-order formulation](../theory/rcwa/first-order-formulation.md) for the underlying arithmetic.

## Include permeability when measured

A table can carry both epsilon and mu:

```python
magnetic = wavebit.TabulatedMaterial(
    wavelengths=[800., 900., 1000.],
    epsilon=[4., 4.2, 4.5],
    mu=[1.1, 1.2, 1.3],
)
value = magnetic.at(850.)  # epsilon=4.1, mu=1.15
```

Both quantities are reevaluated during a wavelength sweep and retained in stored
jobs. The bundled catalog has **n and k only**, so it assumes a non-magnetic
material (`mu=None`, physically mu_r=1). Index data alone gives the product
`epsilon * mu`, not the two separately. If isotropic index and relative
permeability are both known at the same wavelength, convert explicitly:

```python
n_complex, mu_r = 2.+0.1j, 1.25
material = wavebit.Material(
    epsilon=wavebit.eps_from_refractive_index(n_complex, mu=mu_r),
    mu=mu_r,
)
```

`eps_from_refractive_index(n, mu=...)` computes `n**2 / mu`; omitting `mu` retains the
non-magnetic `n**2` conversion. This is a scalar isotropic relation, also applied
elementwise to batches. It does not recover anisotropic tensors from an index.
Permeability must be finite and nonzero: invalid values raise eagerly, or become
NaN at invalid entries of a traced JAX array.

For wavelength-dependent index data, provide permeability on the same grid:

```python
magnetic = wavebit.TabulatedMaterial.from_refractive_index(
    wavelengths=[800., 1000.],
    index=[2., 3.],
    mu=[1., 2.],
)
value = magnetic.at(900.)  # index=2.5, mu=1.5, epsilon=2.5**2/1.5
# mu=1.25 is also accepted for constant permeability over the table range.
```

Index and permeability are interpolated separately, then the evaluated
permittivity is `index**2 / mu`. Squaring/dividing the endpoints first and then
interpolating epsilon defines a different material between samples. Supplied
permeability is retained in the evaluated `Material` and in stored jobs.
If interpolation reaches zero permeability, conversion raises eagerly; under
JAX JIT the affected epsilon entries become NaN. A scalar index cannot determine
epsilon at zero mu, so this case requires independent constitutive data.

## Import your own data

```python
material = wavebit.TabulatedMaterial(
    wavelengths=[800., 900., 1000.],
    epsilon=[4.+0.1j, 4.2+0.12j, 4.5+0.15j],
)
print(material.at(850.).epsilon)   # 4.1 + 0.11j
```

Samples must be finite scalars, three diagonal entries, or 3×3 tensors at two
or more positive, strictly increasing wavelengths. All samples of one quantity
must have the same shape. Complex permittivity is interpolated component by component. Optional `mu` is a
table of equal length, interpolated in the same way. Input arrays are copied into
immutable tuples. For example:

```python
crystal = wavebit.TabulatedMaterial(
    wavelengths=[500., 700.],
    epsilon=[[2., 3., 2.5], [3., 4., 2.9]],
    mu=[[1.1, 1.2, 1.3], [1.2, 1.3, 1.4]],
).rotated(0.3, axis='z')
# Assign crystal to a background or shape; it is evaluated at each solve wavelength.
```

`from_refractive_index` is an isotropic scalar import, optionally including scalar mu.
Tensor permeability is rejected on this path. A general tensor cannot be
specified by one refractive index. For tensor data, `.at(wavelengths).epsilon`
appends `(3,)` or `(3, 3)` to the wavelength-array shape. Such a spectrum is for
analysis; assign the table itself or evaluate one wavelength for a region.

For external data published as refractive index, import it explicitly:

```python
material = wavebit.TabulatedMaterial.from_refractive_index(
    wavelengths=[800., 1000.], index=[2.+0.1j, 3.+0.3j],
)
```

This non-magnetic model interpolates n and k before computing `(n + 1j*k)**2`.
The bundled catalog follows this convention. Interpolating epsilon directly is
a different model between samples. With `mu=...`, `from_refractive_index` also interpolates
permeability and divides the squared index by that value. Solver inputs remain epsilon and mu.
`refractive_index_from_eps(value.epsilon)` is available for derived index plots; no separate
index lookup or name suffix is needed. Removing absorption changes the physical
model and should be explicit in the supplied data.

## Evaluate between sample wavelengths

The `wavelengths` constructor argument identifies where material samples are
known. It does not enumerate allowed solve wavelengths. A table at 800, 900 and
1000 can be evaluated at 850 or 873.2; epsilon and optional mu are interpolated
between adjacent samples. A solve outside 800–1000 needs a table covering that
range. The bundled Si table covers 250–14000 nm. Interpolation cannot recover
spectral features absent from the source data, so choose adequate sample spacing.

## Valid ranges, derivatives and stored jobs

Out-of-range or non-finite evaluation raises `ValueError` eagerly. Under JAX JIT,
invalid elements become NaN; valid elements in the same array remain valid.
No evaluation silently clamps endpoints or extrapolates.

NumPy, JAX and PyTorch share the interpolation. `.at` preserves the wavelength
array's shape and backend. JAX and PyTorch differentiate with respect to wavelength,
including its effect on the material during a solve. Table samples are static,
not optimizer variables. At interior knots the derivative uses the right-hand
segment, and the final endpoint uses the last segment. This is the derivative of
the interpolation, not a fitted physical dispersion law. Usual solver constraints
and frozen-geometry requirements still apply.

`JobSpec.to_json()` embeds the material's table in a `dispersion` object, in the
scene's length unit. No catalog or original file is needed to reconstruct it.
Table values and the interpolation quantity contribute to `spec.key()`.
`JobSpec.expand()` and `wavebit.run(point)` evaluate each sweep point. Schema 9 introduced this capability. The current schema is **10**, which also changes
layer order to bottom-to-top and z to upward-positive height. Old documents must follow
the coordinate migration in [Conventions](../reference/common/conventions.md); changing only their version
number is not sufficient.

These tables supply frequency-domain constitutive data for RCWA. They do not
provide a causal time-domain realization for FDTD, which needs a separate model.

# 07. Materials: permittivity, absorption and wavelength dependence

:::{note}
The runnable examples in this imported course explicitly use `wavebit.compat.legacy` to preserve the workflow behind its historical numerical records. For new work, use the [current getting-started guide](../../getting-started/index.md) and [API migration guide](../../migration/index.md). Historical measurements were not regenerated during this documentation import.
:::

**The material value supplied to the calculation is relative permittivity εᵣ.** The subscript r means relative: εᵣ = ε/ε<sub>0</sub>, the ratio to vacuum permittivity ε<sub>0</sub>.

Enter a constant material directly as a number such as `12.25`. For a wavelength-dependent material, assign an object containing its dispersion data once. The solver evaluates its value at the current wavelength. There is no need to manage refractive index as a separate calculation input.

```{figure} ../../figures/learn/learn/07-material.svg
:alt: Assign a material once; evaluate its relative permittivity at each wavelength and pass the result to the RCWA calculation.
:figclass: learning-figure

Figure 7-1. Assign a material once; evaluate its relative permittivity at each wavelength and pass the result to the RCWA calculation.
```

(material-inputs)=
## First choose how to supply the material

Consider two independent questions: **does the material depend on direction (isotropic or anisotropic), and does it depend on wavelength (constant or dispersive)?** An anisotropic material can also be dispersive. The table connects these physical classifications to the available Wavebit inputs.

| Available information | Example input | Value used in the calculation |
| --- | --- | --- |
| Constant isotropic permittivity | `12.25` | The same εᵣ in every direction |
| Constant permittivity including absorption | `4. + 0.1j` | Complex εᵣ |
| Constant ε and μ | `wavebit.Material(epsilon=4., mu=1.2)` | εᵣ and μᵣ supplied together |
| Bundled material table | `wavebit.materials.load('Si')` | εᵣ evaluated automatically at the calculation wavelength |
| Your wavelength-dependent ε and μ data | `wavebit.TabulatedMaterial(...)` | Scalar or tensor εᵣ and μᵣ interpolated from your table |
| External n+ik table | `wavebit.TabulatedMaterial.from_index(..., mu=...)` | Permittivity calculated from index and optional permeability; nonmagnetic if mu is omitted |
| Anisotropic permittivity tensor | `Material(epsilon=[2.25, 4., 2.25])` or a 3×3 array | Directional response and coupling between components; see the examples below |

Constants and user tables can represent both isotropic and anisotropic materials. Bundled n·k data are evaluated as nonmagnetic isotropic materials. `from_index` applies to isotropic materials and also converts magnetic materials when `mu` is supplied. A raster contains evaluated numerical arrays. Assign a scalar, supported tensor material or dispersive object to a shape’s `material` or a layer’s `background`. The `top` and `bot` half-spaces accept only scalar materials, with real or complex values. See the {ref}`half-space discussion <material-half-spaces>` below for the input details.

## Assign silicon once and change only the wavelength

```python
silicon = wavebit.materials.load('Si')
# Do not fix the wavelength here. silicon retains its dispersion table.
```

Assign this object to a shape’s `material`, a layer’s `background`, or the upper and lower media `top` and `bot`. The following example **keeps the same silicon object on the same circle and changes only the wavelength.** It does not recreate the shape or manually replace its permittivity at each wavelength.

Runnable file: {download}`09_material_sweep.py <../../../examples/rcwa/learning/learn/09_material_sweep.py>`

```{literalinclude} ../../../examples/rcwa/learning/learn/09_material_sweep.py
:language: python
:linenos:
```

| Wavelength (nm) | Circle relative permittivity εᵣ | Total R | Total T |
| --- | --- | --- | --- |
| 800.0 | 13.634191 + 0.047966j | 0.057296958 | 0.940747954 |
| 900.0 | 13.240457 + 0.054569j | 0.050530329 | 0.945208958 |
| 1000.0 | 12.903107 + 0.029383j | 0.015798325 | 0.983445526 |

The table shows that the permittivity inside the circle changes along with wavelength. Orders `(1, 1)` and 64 NVF samples are chosen to demonstrate the workflow briefly; they do not establish optical convergence.

## How do constant and dispersive materials differ?

**Both `TabulatedMaterial` and `Material` are objects representing materials.** `TabulatedMaterial` holds wavelength-dependent data, whereas `Material` holds permittivity and permeability values. When the calculation wavelength changes, `TabulatedMaterial` evaluates the corresponding values and `Material` continues to use its stored values.

| Specification | Value type | After changing wavelength |
| --- | --- | --- |
| Number `12.25` | Number | Permittivity stays at 12.25; no object construction required |
| `wavebit.Material(epsilon=4., mu=1.2)` | `Material` | Permittivity 4 and permeability 1.2 remain fixed |
| `wavebit.materials.load('Si')` | `TabulatedMaterial` | Evaluates the silicon table at the new wavelength |
| `silicon.at(900.)` | `Material` | Keeps the permittivity and permeability evaluated at 900 nm |

`materials.load('Si')` loads silicon’s **wavelength-dependent material data**. Assigning it to a shape or layer automatically uses values appropriate to the calculation wavelength.

`silicon.at(900.)` **evaluates permittivity and permeability at 900 nm and places them in a `Material` object.** Assign that result to a shape or layer and it keeps those 900 nm values even if the calculation wavelength changes.

### Both objects can be assigned to the same material positions

Both object types can be used for a shape’s `material`, a layer’s `background`, and the upper and lower media `top` and `bot`. This example assigns silicon to a layer while choosing whether its values should follow wavelength or stay fixed.

```python
silicon = wavebit.materials.load('Si')       # TabulatedMaterial: wavelength-dependent data
silicon_900 = silicon.at(900.)              # Material: ε and μ at 900 nm

# A layer whose silicon values follow the calculation wavelength
layer = dict(background=silicon, shapes=[])

# A layer that retains the 900 nm values even when calculation wavelength changes
fixed_layer = dict(background=silicon_900, shapes=[])
```

At a calculation wavelength of 1,000 nm, `layer` uses silicon’s values at 1,000 nm, while `fixed_layer` uses its values at 900 nm. **To include silicon’s dispersion in a wavelength sweep, assign `silicon` directly.** The solver evaluates the appropriate values, so there is no need to call `.at()` yourself.

Calling `.at(900.)` leaves the original `silicon` wavelength data intact. `silicon_900` is a separate constant-valued material.

## Available materials and length units

```python
wavebit.materials.available()
# ('filmetrics/Al2O3', 'filmetrics/Si', 'filmetrics/Si3N4', 'filmetrics/SiO2')
silicon = wavebit.materials.load('filmetrics/Si')  # Specify the source as well
print(silicon.wavelength_range)                  # (190.7, 2500.0), nm by default
silicon_um = wavebit.materials.load('Si', wavelength_unit='um')
value = silicon_um.at(0.9)                       # 0.9 μm = 900 nm
```

Short names such as `Si` are case-insensitive. If several sources make a name ambiguous, one is not chosen arbitrarily: include the source, as in `filmetrics/Si`. **The same chemical formula can have different optical properties under different fabrication conditions, so choosing data is part of defining the physical model.** Only the requested file is read, and its immutable object is reused; you do not need to preload the entire material library.

The default length unit is `nm`. Setting `wavelength_unit='um'` converts the table’s wavelength axis to μm. Supply the calculation wavelength, period, shape dimensions and layer thicknesses in μm as well. Supported units are `nm`, `um`, `mm` and `m`; existing geometry is not silently rescaled. The previously bundled `p_Si.mat` lacks unit and original-source metadata, so it is not included in the new public catalog.

Material values read from `solver.structure`, `solver.top` and `solver.bot` are evaluated at the current wavelength. Reassign `structure`, `top` or `bot` when supplying a new material or shape. Avoid directly editing part of a retrieved evaluation dictionary: doing so does not change the original dispersion definition.

## Using your own permittivity table

```python
material = wavebit.TabulatedMaterial(
    wavelengths=[800., 900., 1000.],
    epsilon=[4.+0.1j, 4.2+0.12j, 4.5+0.15j],
)
value = material.at(850.)
# value.epsilon = 4.1 + 0.11j
```

Supply positive, increasing sample wavelengths and one complex permittivity per wavelength. At least two samples are required; complex permittivity is interpolated linearly between them. An optional `mu` table can have the same length. The data are copied, so later edits to the original array do not silently change the material definition. At each wavelength, supply a scalar, three diagonal components or a 3×3 tensor. All samples for one physical quantity must have the same shape; tensors are interpolated component by component.

## Sample wavelengths and calculation wavelengths are different

`wavelengths=[800., 900., 1000.]` lists **the sample points at which material data are available**. A calculation can use any wavelength within that interval.

```python
value = material.at(850.)              # Linearly interpolate ε between 800 and 900
result = solver.solve(wavelength=873.2) # Evaluate assigned dispersive materials here too
```

When both ε and μ tables are supplied, both are interpolated. The magnetic example below gives ε=4.1 and μ=1.15 at 850 nm, without adding a new sample wavelength. However, a table spanning 800–1000 nm raises an error at 1100 nm; broader data are required. The default `materials.load('Si')` data cover 190.7–2500 nm. Being able to interpolate within a range does not establish that the table represents the actual material accurately enough. Check sample spacing and the original data where dispersion changes rapidly.

## Supply permeability μ data together with permittivity

```python
magnetic = wavebit.TabulatedMaterial(
    wavelengths=[800., 900., 1000.],
    epsilon=[4., 4.2, 4.5],
    mu=[1.1, 1.2, 1.3],
)
value = magnetic.at(850.)
# value.epsilon = 4.1, value.mu = 1.15
```

Assigning this object as a material makes ε and μ change together with wavelength. **The current bundled tables contain only n·k data, not measured μ data.** They assume nonmagnetic material, μᵣ=1. In a general isotropic medium, `n²=εᵣ μᵣ`, so n·k alone cannot determine ε and μ separately. For magnetic materials, supply both measured or modeled quantities as above.

## Convert refractive index when importing external data

External optical data are often provided as `n, k` tables. For a material selected with `materials.load`, n and k are interpolated first, then converted to permittivity using `(n+ik)²` under the nonmagnetic assumption μᵣ=1. This conversion is internal. It is unnecessary when using a permittivity table directly. The two workflows also interpolate different quantities: interpolating index and then squaring it need not equal direct permittivity interpolation between samples.

Import your own n+ik data explicitly with `TabulatedMaterial.from_index(wavelengths, index)`. To plot index from permittivity under the nonmagnetic assumption, use `wavebit.index_from_eps(silicon.at(wavelengths).epsilon)`. `.at` also accepts a wavelength array and returns a permittivity array with the corresponding shape. There is no separate index lookup function or hidden suffix on material names. Wavebit uses the `exp(−iωt)` time convention; ordinary passive media express loss through a positive imaginary part of scalar permittivity.


## Converting a magnetic material’s refractive index to permittivity

For an isotropic material with complex index `n_complex`, `n_complex² = εᵣ μᵣ`. Thus **for a magnetic material, square the index and divide by relative permeability.** Index and permeability must refer to the same wavelength.

```text
εᵣ = n_complex² / μᵣ
```

Passing `mu` to `eps_from_index` performs this conversion. For example, index 2 and relative permeability 1.25 give relative permittivity 3.2. Include the same permeability in the resulting `Material`.

```python
n_complex = 2.0
mu_r = 1.25
magnetic = wavebit.Material(
    epsilon=wavebit.eps_from_index(n_complex, mu=mu_r),
    mu=mu_r,
)
# magnetic.epsilon = 3.2, magnetic.mu = 1.25
```

Permeability can also accompany a wavelength-dependent index table.

```python
magnetic = wavebit.TabulatedMaterial.from_index(
    wavelengths=[800., 1000.],
    index=[2., 3.],
    mu=[1., 2.],
)
value = magnetic.at(900.)
# Interpolate to index 2.5 and relative permeability 1.5
# value.epsilon = 2.5**2 / 1.5, value.mu = 1.5
```

`index` can contain complex values including absorption, such as `2.+0.1j`. Supply `mu` as an **array at the same sample wavelengths**, as above, or a **constant throughout the table range**, such as `mu=1.25`. Omitting `mu` retains the nonmagnetic assumption `μᵣ=1`.

This model **interpolates refractive index and permeability separately, then calculates εᵣ=n_complex²/μᵣ**. Values between samples can differ from those obtained by first converting each sample to permittivity and then interpolating permittivity.

Index alone cannot determine permittivity and permeability separately; permeability needs an independent measurement or model. The permeability used in conversion must be finite and nonzero. If interpolation reaches zero permeability, ordinary execution raises an error, while JAX JIT produces NaN for the corresponding permittivity. **This conversion is for isotropic materials.** Supply `epsilon` and `mu` tensors directly for anisotropic materials.

## Wavelength sweeps and differentiation

Use the same length unit for the table and all geometry. Every dispersive object rejects out-of-range wavelengths with an error in ordinary execution. Inside JAX JIT, out-of-range elements become NaN. Values are not silently clamped to endpoints or extrapolated, so keep a sweep inside `wavelength_range`.

When a `TabulatedMaterial` is assigned to a shape, layer background or surrounding medium, changing wavelength through `solver.solve(wavelength=...)` automatically evaluates its values at that wavelength.

**A numerical raster is an already evaluated permittivity distribution.** It does not retain per-pixel material names or dispersion definitions. For raster sweeps, evaluate materials with `.at(wavelength)` and update the array. Likewise, matrices passed directly to `solve_conv` must be rebuilt for each wavelength by the caller.

In JAX and PyTorch, wavelength derivatives include the change in material values. This differentiates the table’s piecewise-linear interpolation; it does not imply that a physical dispersion model has been fitted. At sample points, derivatives use the interval to the right, except at the final sample, which uses the left interval. Sample values themselves are fixed data, not differentiable design variables. An RCWA dispersion table is also not automatically an FDTD time-domain material model.

(material-half-spaces)=
## top and bot: the half-spaces outside the stack

Relative to the finite-thickness layers, `top` fills the region above the uppermost layer and `bot` fills the region below the lowest layer. For a film on a glass substrate in air, for example, set `top` to air and `bot` to glass.

`top` extends infinitely upward and `bot` infinitely downward. These regions are **half-spaces**, and they have no `thickness` input.

Wavebit’s `top` and `bot` take isotropic materials: scalar permittivity, a `Material` with scalar ε and μ, or a dispersive material object. Anisotropic half-spaces are unsupported. Anisotropic materials can instead be assigned to the `background` of a finite layer or a shape’s `material`.

```python
top = 1.                                           # Nonmagnetic air
bot = 4. + 0.3j                                    # Absorbing nonmagnetic medium
magnetic = wavebit.Material(epsilon=4.+0.3j, mu=1.2) # Supply ε and μ together
```

`incidence_side` selects the illuminated side. With the default `"top"`, `top` is the incident medium; with `"bottom"`, `bot` is the incident medium. **Real `theta` and `phi` specify the direction of phase propagation inside that selected incident medium.** `theta` is measured from the inward normal and `phi` is the azimuth in the fixed xy coordinates, both in radians.

Direct direction input using wavevectors `kx` and `ky` is explained in {ref}`Chapter 8’s two input forms <incident-direction-inputs>`. ε and μ of `top` and `bot` remain necessary. See {ref}`Chapter 8 <absorbing-incidence>` for wavevectors and examples with lossy incident or transmitted media, and {ref}`Chapter 13 <half-space-power>` for the reference planes of reflectance and transmittance and absorption in finite layers.

(material-anisotropy)=
## Isotropy and anisotropy: direction-dependent material response

Imagine changing the electric-field direction at one point in a material. **A material described by the same permittivity in every direction is isotropic.** **A material whose permittivity must distinguish directions is anisotropic.** Direction here describes the relation between the electric field and the material.

### One number or three values

An isotropic material uses one relative permittivity. If the reference directions of an anisotropic material align with the x, y and z axes, specify the three directional values.

```python
from wavebit.compat import legacy as wavebit

isotropic = 2.25
crystal = wavebit.Material(epsilon=[2.25, 4., 2.25])
```

| Material | x-direction value | y-direction value | z-direction value |
| --- | --- | --- | --- |
| `isotropic` | 2.25 | 2.25 | 2.25 |
| `crystal` | 2.25 | 4.0 | 2.25 |

For `crystal`, the response to a y-directed electric field differs from that in x and z. Supplying three values **specifies the diagonal components in x, y, z order**, with the remaining matrix components discussed below set to zero.

### Why write permittivity as a 3×3 matrix?

Write the relation between electric field **E** and electric displacement **D** as follows. Electric displacement includes the material’s polarization response to the field; ε<sub>0</sub> is vacuum permittivity.

```{raw} html
<div class="power-equations">
<p><b>D</b> = ε<sub>0</sub> ε<sub>r</sub> <b>E</b></p>
</div>
```

For an isotropic material, εᵣ is a scalar. For a general anisotropic material, we must distinguish how the x, y and z components of electric field contribute to each component of electric displacement, so εᵣ is a 3×3 matrix. This material quantity is the **permittivity tensor**. Each ε<sub>ij</sub> below is a relative-permittivity tensor component, with ij as subscripts.

```{raw} html
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" style="font-size:1.2rem" aria-label="Three-by-three relative permittivity tensor">
<mrow><msub><mi>ε</mi><mi>r</mi></msub><mo>=</mo><mo>[</mo><mtable>
<mtr><mtd><msub><mi>ε</mi><mrow><mi>x</mi><mi>x</mi></mrow></msub></mtd><mtd><msub><mi>ε</mi><mrow><mi>x</mi><mi>y</mi></mrow></msub></mtd><mtd><msub><mi>ε</mi><mrow><mi>x</mi><mi>z</mi></mrow></msub></mtd></mtr>
<mtr><mtd><msub><mi>ε</mi><mrow><mi>y</mi><mi>x</mi></mrow></msub></mtd><mtd><msub><mi>ε</mi><mrow><mi>y</mi><mi>y</mi></mrow></msub></mtd><mtd><msub><mi>ε</mi><mrow><mi>y</mi><mi>z</mi></mrow></msub></mtd></mtr>
<mtr><mtd><msub><mi>ε</mi><mrow><mi>z</mi><mi>x</mi></mrow></msub></mtd><mtd><msub><mi>ε</mi><mrow><mi>z</mi><mi>y</mi></mrow></msub></mtd><mtd><msub><mi>ε</mi><mrow><mi>z</mi><mi>z</mi></mrow></msub></mtd></mtr>
</mtable><mo>]</mo></mrow>
</math>
```

**Rows specify the direction of D; columns specify the direction of E.** Expanding the first row gives:

```{raw} html
<div class="power-equations">
<p>D<sub>x</sub> = ε<sub>0</sub> (ε<sub>xx</sub>E<sub>x</sub> + ε<sub>xy</sub>E<sub>y</sub> + ε<sub>xz</sub>E<sub>z</sub>)</p>
</div>
```

| Component | Meaning |
| --- | --- |
| ε<sub>xx</sub> | x-directed electric field E<sub>x</sub> contributes to x-directed electric displacement D<sub>x</sub> |
| ε<sub>xy</sub> | y-directed electric field E<sub>y</sub> contributes to x-directed electric displacement D<sub>x</sub> |
| ε<sub>xz</sub> | z-directed electric field E<sub>z</sub> contributes to x-directed electric displacement D<sub>x</sub> |

ε<sub>xx</sub>, ε<sub>yy</sub> and ε<sub>zz</sub>, whose row and column indices match, are **diagonal components**; the rest are **off-diagonal components**. For example, a nonzero ε<sub>xy</sub> can produce x-directed electric displacement from a purely y-directed electric field.

If the full matrix is known, enter its rows and columns in x, y, z order. Express all components in Wavebit’s x, y and z coordinates.

```python
material = wavebit.Material(epsilon=[
    [3.0, 0.2, 0.0],   # Row contributing to the x component of D
    [0.2, 4.0, 0.0],   # Row contributing to the y component of D
    [0.0, 0.0, 2.5],   # Row contributing to the z component of D
])
```

This material has ε<sub>xy</sub> = ε<sub>yx</sub> = 0.2. Wavebit uses the input matrix as supplied; it does not arbitrarily equate two unequal components. Components can also be complex.

Relative permeability μᵣ uses the same formats. Omitting `mu` selects the nonmagnetic condition μᵣ = 1.

```python
magnetic = wavebit.Material(
    epsilon=[2., 3., 4.],
    mu=[1.1, 1.2, 1.3],
)
```

(material-rotation)=
### How does rotating a material change its matrix?

First distinguish the **calculation axes fixed in space** (x·y·z) from the **principal axes rotating with the material** (x′·y′·z′). Principal axes are the reference directions in which the material’s permittivity is diagonal, as for `crystal` above. The axes initially coincide; rotating only the material leaves the calculation axes fixed and rotates the principal axes with the material.

#### Fixed calculation axes and rotated material axes

| Notation | Meaning | When only the material rotates |
| --- | --- | --- |
| x·y·z | Fixed calculation coordinates for electric-field, displacement and material-matrix components | Remain fixed |
| x′·y′·z′ | Principal axes attached to the material; the prime distinguishes material axes | Rotate with the material |

The xy view after rotating the material by +45° about z is shown below. Viewed toward the origin from +z, positive rotation is counterclockwise.

```{figure} ../../figures/learn/learn/07-material-axes.svg
:alt: The fixed x calculation axis points right and y points up. Rotating the material by 45 degrees sends its x′ axis up and right with permittivity 2.25, and its y′ axis up and left with permittivity 4. The calculation axes remain fixed.
:figclass: learning-figure

Figure 7-2. Gray denotes the fixed calculation axes, and blue denotes the material axes after a +45° rotation. z and z′ coincide and point out of the screen.
```

#### Which values stay fixed, and which change?

Consider rotating a material with relative permittivity `[2.25, 4., 2.25]`.

| Material axis | Direction and permittivity before rotation | Direction and permittivity after +45° rotation |
| --- | --- | --- |
| First principal axis | → along x, 2.25 | ↗ along x′, 2.25 |
| Second principal axis | ↑ along y, 4.0 | ↖ along y′, 4.0 |
| Third principal axis | +z direction, 2.25 | +z′ direction, 2.25 |

**The directions associated with values 2.25 and 4 rotate with the material.** In the rotated material coordinates x′·y′·z′, the permittivity matrix still has diagonal entries `[2.25, 4., 2.25]`. Expressing that same material in the fixed x·y·z coordinates introduces off-diagonal components.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

crystal = wavebit.Material(epsilon=[2.25, 4., 2.25])
rotated = crystal.rotated(np.deg2rad(45.), axis="z")
```

If Q is the matrix for a +45° rotation about z, `.rotated()` performs the following multiplication.

```{raw} html
<div class="power-equations">
<p>ε<sub>after</sub> = Q ε<sub>before</sub> Q<sup>T</sup></p>
</div>
```

**The matrix below is the final result after multiplying by both Q on the left and Qᵀ on the right.** It is stored in `rotated.epsilon`, with every component expressed in the fixed x·y·z axes.

```text
[ 3.125  -0.875   0.   ]
[-0.875   3.125   0.   ]
[ 0.      0.      2.25 ]
```

After rotation, ε<sub>xx</sub> and ε<sub>yy</sub> are 3.125, while ε<sub>xy</sub> and ε<sub>yx</sub> are −0.875. These are the fixed-coordinate representation of the same values 2.25 and 4 read along the rotated material axes in the table.

To rotate one vector, multiply it from the left by the rotation matrix, as in QE. A permittivity tensor is a **relation mapping electric field E to electric displacement D**. Rotating the material therefore accounts for Qᵀ on the input E side and Q on the output D side, giving Q ε<sub>before</sub> Qᵀ. Once that matrix has been formed, do not multiply by Q or Qᵀ again when applying it to the field.

#### Computing the rotated permittivity tensor

The response to field E after rotating a material by +45° can be computed equivalently as follows: **keep the original material tensor, rotate the field components by −45°, apply the tensor, then rotate the resulting displacement components by +45°.**

The two material matrices are labeled before and after rotation. **Both are relative-permittivity matrices expressed in the fixed spatial x·y·z coordinates.**

| Notation | Meaning | Value in this example |
| --- | --- | --- |
| ε<sub>before</sub> | Tensor before rotating the material | Diagonal entries 2.25·4·2.25; off-diagonal entries 0 |
| ε<sub>after</sub> | Tensor after rotating the material by +45° | The `rotated.epsilon` matrix above |

Q is the +45° rotation matrix. Its transpose Q<sup>T</sup> represents the inverse rotation, −45°. E′ is the intermediate field obtained by rotating E’s components by −45°, and D′ is the displacement obtained by applying the unrotated material tensor to E′.

| Step | Calculation | Meaning |
| --- | --- | --- |
| ① Rotate field components by −45° | E′ = Q<sup>T</sup>E | Obtain intermediate field components to apply to the original material |
| ② Apply the original permittivity tensor | D′ = ε<sub>0</sub> ε<sub>before</sub> E′ | Apply diagonal values 2.25·4·2.25 respectively and multiply by ε<sub>0</sub> |
| ③ Rotate the resulting displacement components by +45° | D = QD′ | Obtain the displacement of the rotated material under the original field E |

Because the original tensor applies different permittivities to different directions, both the intermediate inverse rotation and the final forward rotation are required. Combining the three equations gives:

```{raw} html
<div class="power-equations">
<p><b>D</b> = ε<sub>0</sub> Q ε<sub>before</sub> Q<sup>T</sup> <b>E</b></p>
<p>ε<sub>after</sub> = Q ε<sub>before</sub> Q<sup>T</sup></p>
</div>
```

This relation holds for any field E. Consequently, `.rotated()` precomputes ε<sub>after</sub> = Q ε<sub>before</sub> Q<sup>T</sup> from the material and rotation angle alone. The solver subsequently applies that rotated tensor directly to E in fixed xyz coordinates.

#### A hand calculation: an electric field pointing right

Evaluate the same +45° example in the xy plane. Set E = (1, 0) V/m with zero z component. The matrices below are the xy blocks of the three-dimensional rotation matrices.

```{raw} html
<div class="rotation-matrices">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" style="font-size:1.15rem" aria-label="The 45-degree rotation matrix Q is one over square root two times the matrix 1, minus 1, 1, 1">
<mrow><mi>Q</mi><mo>=</mo><mfrac><mn>1</mn><msqrt><mn>2</mn></msqrt></mfrac><mo>[</mo><mtable rowspacing="0.35em" columnspacing="0.8em"><mtr><mtd><mn>1</mn></mtd><mtd><mo>−</mo><mn>1</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow>
</math>
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" style="font-size:1.15rem" aria-label="The inverse rotation, transpose of Q, is one over square root two times the matrix 1, 1, minus 1, 1">
<mrow><msup><mi>Q</mi><mi>T</mi></msup><mo>=</mo><mfrac><mn>1</mn><msqrt><mn>2</mn></msqrt></mfrac><mo>[</mo><mtable rowspacing="0.35em" columnspacing="0.8em"><mtr><mtd><mn>1</mn></mtd><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mo>−</mo><mn>1</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow>
</math>
</div>
```

Write each vector as an ordered pair. Factoring out the common ε<sub>0</sub> means that we calculate D/ε<sub>0</sub>; every numerical entry in the table therefore has units V/m.

| Step | Calculation performed | x and y components |
| --- | --- | --- |
| Input E | Electric field pointing right | (1, 0) |
| E′ = Q<sup>T</sup>E | Rotate field components by −45° | (1/√2, −1/√2) |
| D′/ε<sub>0</sub> = ε<sub>before</sub>E′ | Apply the original tensor | (2.25/√2, −4/√2) |
| D/ε<sub>0</sub> = Q(D′/ε<sub>0</sub>) | Rotate the resulting displacement components by +45° | (3.125, −0.875) |

The intermediate field E′ points ↘. Applying the original tensor multiplies its x component by 2.25 and its y component by 4. Rotating that result by +45° gives the displacement D of the +45°-rotated material under the original right-pointing field E.

Expanding the two components in the final step:

```{raw} html
<div class="power-equations">
<p>D<sub>x</sub>/ε<sub>0</sub> = (2.25 + 4)/2 = 3.125 V/m</p>
<p>D<sub>y</sub>/ε<sub>0</sub> = (2.25 − 4)/2 = −0.875 V/m</p>
</div>
```

Directly multiplying the previously computed `rotated.epsilon` by E = (1, 0, 0) gives the same D/ε<sub>0</sub> = (3.125, −0.875, 0).

#### Why is an isotropic material unchanged by rotation?

For an isotropic material with relative permittivity ε<sub>s</sub>, the matrix is ε<sub>s</sub>I. I is the **identity matrix**, with diagonal entries 1 and all other entries 0.

```{raw} html
<div class="power-equations">
<p>Q (ε<sub>s</sub>I) Q<sup>T</sup> = ε<sub>s</sub> QQ<sup>T</sup> = ε<sub>s</sub>I</p>
</div>
```

Multiplying every component by the same value allows the −45° and +45° rotations to cancel. In the anisotropic example, the components are multiplied by different values, 2.25 and 4, so rotation changes the matrix representing the material in fixed calculation coordinates.

#### Specifying rotations in Wavebit

`.rotated(angle, axis=...)` returns a new object representing the rotated material. Angles are in radians and follow the right-hand rule. Use `"x"`, `"y"` or `"z"` for `axis`. Both specified ε and μ are rotated, and successive calls apply rotations in their call order. Shape position, shape geometry and lattice remain unchanged.

Tilting a material about x or y can introduce ε<sub>xz</sub>, ε<sub>yz</sub>, ε<sub>zx</sub> and ε<sub>zy</sub> components. Assign a full 3×3 matrix directly or a material produced with `.rotated(...)` in the same way.

### Filling a whole layer with an anisotropic material

Anisotropic materials also go in a layer’s `background` or a shape’s `material`. To fill the entire layer with one material, specify `background` and use `shapes=[]`.

Here, an anisotropic film 220 nm thick lies between air half-spaces and is illuminated normally from above. All lengths are in nm.

<!-- uniform-anisotropic-example:start -->
```python
from wavebit.compat import legacy as wavebit

crystal = wavebit.Material(epsilon=[2.25, 4., 2.25])
solver = wavebit.rcwa(
    wavelength=600., thickness=[220.],
    period=(600., 600.), top=1., bot=1.,
    structure=[{"background": crystal, "shapes": []}],
    theta=0., phi=0., truncation_order=(0, 0),
)
result = solver.solve()
print("y-polarized reflectance:", float(result.s_inc.R.sum()))
print("x-polarized reflectance:", float(result.p_inc.R.sum()))
```
<!-- uniform-anisotropic-example:end -->

| Incident electric-field direction | Relative permittivity in that direction | Reflectance |
| --- | --- | --- |
| y direction | 4.0 | 0.357473 |
| x direction | 2.25 | 0.016308 |

At normal incidence with `phi=0`, the s-polarized electric field aligns with y and the p-polarized field with x. Different directional permittivities produce different reflectances. Because the film is uniform in xy, Fourier order 0 is sufficient. `period` specifies the calculation-cell size.

### Assigning an anisotropic material inside a shape

The following example assigns a rotated anisotropic material to a cylinder in air. The material is uniform inside the cylinder, and the exterior is air.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

crystal = wavebit.Material(epsilon=[2.25, 4., 2.25])
rotated = crystal.rotated(np.deg2rad(45.), axis="z")
solver = wavebit.rcwa(
    wavelength=700., period=(600., 600.), thickness=[170.],
    top=1., bot=1.,
    structure=[{"background": 1., "shapes": [
        {"type": "circle", "lx": 270., "material": rotated},
    ]}],
    truncation_order=(3, 3), shape_nvf_grid_size=128,
)
result = solver.solve()
solver.plot_structure(component="xx")
```

Its diameter is 270 nm and its height equals the layer thickness, 170 nm. `plot_structure(component="xx")` displays the spatial ε<sub>xx</sub> distribution; use `component="xy"` for ε<sub>xy</sub>. The Fourier order and NVF sample count above are example settings. Assess the required accuracy using [Chapter 19’s convergence checks](convergence.md).

**Anisotropy describes directional properties at one position; a material distribution describes differences between positions.** Specify the two independently.

| Structure | Directional properties | Material distribution in xy |
| --- | --- | --- |
| Film filled with one anisotropic material | Anisotropic | Uniform everywhere |
| Isotropic cylinder in air | Each material is isotropic | Different inside and outside the cylinder |
| Anisotropic cylinder in air | Cylinder material is anisotropic | Different inside and outside the cylinder |

### Supplying a raster array

A raster stores material values in spatial cells. When storing the full permittivity matrix, `structure` has shape `(layers, rows, columns, 3, 3)`. Its final two axes are **the row and column of the permittivity matrix in each cell**. Supply permeability in a separate `mu` array.

This array fills all 2×4 cells in one layer with the same material.

```python
import numpy as np

cell_tensor = np.diag([2.25, 4., 2.25])
structure = np.broadcast_to(cell_tensor, (1, 2, 4, 3, 3)).copy()
```

Using this array as `structure` in the film example represents the same uniform layer. Change individual cell matrices to assign different materials. For example, `structure[0, 0, 0] = np.eye(3)` changes row 0, column 0 of the first layer to an isotropic material of relative permittivity 1. The permeability array uses the same spatial grid; each cell can store a scalar or a 3×3 matrix.

### Wavelength-dependent anisotropic materials

Supplying three diagonal components or a 3×3 matrix at each wavelength to `TabulatedMaterial` interpolates every component at the calculation wavelength.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

crystal_table = wavebit.TabulatedMaterial(
    wavelengths=[500., 700.],
    epsilon=[[2., 3., 2.5], [3., 4., 2.9]],
    mu=[[1.1, 1.2, 1.3], [1.2, 1.3, 1.4]],
)
value = crystal_table.at(600.)
# Diagonal epsilon components: [2.5, 3.5, 2.7]
# Diagonal mu components: [1.15, 1.25, 1.35]

rotated_table = crystal_table.rotated(np.deg2rad(30.), axis="z")
```

Assign `rotated_table` to a shape or layer background once; each wavelength then uses the corresponding rotated material values. With a numerical raster, evaluate materials and update the array at each wavelength.

### Calculation options and supported scope

After assigning materials as above, call `solve()` and Wavebit constructs the required matrices. The available approaches depend on the structure representation.

| Structure | Calculation approach |
| --- | --- |
| Layer uniform in xy | Uniform-layer calculation; full 3×3 ε and μ supported |
| Raster in an orthogonal cell aligned with x and y | NVF or Li |
| Rectangular shapes with x- and y-aligned edges in the same orthogonal cell | NVF or Li |
| Circles, ellipses, rotated shapes or an oblique lattice | NVF |
| Upper and lower half-spaces `top` and `bot` | Isotropic scalar ε and μ supported; anisotropic tensors unsupported |

The defaults for patterns containing anisotropic materials are `factorization="nvf"` and `nvf_projector="linear"`. `linear` selects how the NVF projector is constructed; incident polarization is a separate input described in [Chapter 8](light.md). To select Li, set `factorization="li"` for a structure satisfying the conditions in the table.

If ε or μ contains xz, yz, zx or zy components, the first-order formulation (`first_order`, formerly 4ff), which handles those components together, is selected automatically. Both the default `layer_algorithm="ETM"` and `layer_algorithm="SMM"` can connect layers. Both calculate reflection, transmission, and electric and magnetic fields within the structure. [Chapter 15](layer-algorithms.md) explains the connection algorithms and scattering-matrix use. The [full-tensor example](../../guide/rcwa/full-tensor-patterns.md) gives runnable code assigning a tilted material to a rectangular shape.

[Chapter 10](fourier-orders.md) explains Li, NVF and Fourier-coefficient calculation, while [Chapter 11](nvf.md) covers NVF normal fields and projectors. Use `fourier_coefficients="sampled"` when periodic replicas of anisotropic NVF shapes overlap or shapes cover one another across a cell boundary. The geometry conditions for `"integrated"` are listed in the [shape-integration guide](../../guide/rcwa/tensor-shape-integration.md).

At finite NVF order, R+T may differ from exactly 1 even for lossless materials. Assess accuracy by checking convergence of the quantity of interest as in [Chapter 19](convergence.md).

```{dropdown} Conditions for automatic differentiation
NumPy, JAX and PyTorch accept the same material inputs. See [Chapter 17](autodiff.md) for the basic differentiation workflow.

- The integrated NVF shape path supports derivatives of material values, shape position, dimensions, rotation angle and lattice. Before JAX tracing, record shape intersection and contact relationships with `freeze_geometry()` and pass them to `solve(geometry=...)`. If overlap or periodic-contact relationships change, record them again outside tracing. Also check grid convergence of derivatives near points where the normal field is equidistant from several boundaries.
- NVF `"sampled"` shapes and tensor rasters support material-value derivatives with fixed boundaries. Shape and lattice derivatives are unsupported on these paths.
- Li tensor rasters support material-value and orthogonal-period derivatives. Li rectangular shapes support material, position, size and period derivatives while boundary ordering is preserved. Changing order, normal-field grid or coefficient method requires refreshing the stored preparation information.
- Dispersion-table samples and angles rotating an entire table are fixed data. To differentiate a material rotation angle as a design variable, evaluate at the current wavelength before rotating, as in `crystal_table.at(wavelength).rotated(angle)`. Values returned by `.at()` for a wavelength array are for inspecting a spectrum; assign a material object or a value at one wavelength to an individual material region.
- During differentiation or JAX tracing, off-diagonal components that currently equal zero are retained. A first-order path may therefore be used even when xz, yz, zx and zy happen to be zero at the current values.
```

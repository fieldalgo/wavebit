# 03. Conventions to establish first: coordinates, units, and materials

:::{note}
The figures in this chapter still label the media and layers with the earlier constructor's names:
`bot`, `structure[i]`, `thickness=[d0, d1]` and the circle's diameter `lx`. In the current API these
are `Stack(bottom=...)`, `Stack(layers=(...))` with one `Layer(thickness=...)` per layer, bottom first,
and `Circle(radius=...)`. The text and code use the current names.
:::

Wavebit cannot infer from a number whether you mean nm or µm. Declare **one length unit** with `Units(length="nm")` and use it for `wavelength`, `period`, `thickness`, and shape positions and sizes. The executable examples in this book use nm. Angles are in radians, so 30 degrees is `np.deg2rad(30.)`.

## Start with the whole structure

Read the diagram below in the order **one repeating cell → stacked layers → incoming light**. Define a material pattern within one cell, and that pattern repeats along x and y. Stack such layers along z, then specify the direction and wavelength of the incoming light.

```{figure} ../../figures/learn/learn/03-stack-overview.svg
:alt: A three-dimensional conceptual diagram of a two-layer periodic structure. The cell inside the yellow outline repeats at spacings Px and Py, and structure 0 and 1 are stacked from bottom to top with thicknesses d0 and d1. The propagation direction k, incident angle theta, and azimuth phi are shown for light arriving from the upper medium.
:figclass: learning-figure

Figure 3-1. **An example of oblique incidence from above.** Read `period` as the lateral repetition spacing, `thickness` as the layer thickness in the vertical direction, and the incoming light above the structure. The pale plane of incidence is an **imaginary plane** containing the propagation direction and the vertical normal. It is not a material layer.
```

The yellow outline encloses **an entire cell, including the background**, not the boundary of a single shape. `Px` and `Py` are the cell's repetition spacings, not the object's width. Only 3×3 cells are drawn, but the calculation treats a structure that repeats indefinitely in the plane. Teal indicates the repeated shapes and the gray planes indicate layer divisions; these colors do not correspond to particular permittivity values.

`layers[0]` is the bottommost layer, and `layers[1]` is the layer above it; each `rcwa.Layer` carries its own `thickness`, here `d0` and `d1`. With the bottom at `z=0`, the first layer occupies `0≤z≤d0` and the second occupies `d0≤z≤d0+d1`. Writing the total height as `h_total`, the sum of the layer thicknesses, the top surface is at `z=h_total`. For the two layers in the diagram, `h_total=d0+d1`.

For incidence from above, light travels downward along `−z`, and `theta` is measured from the inward normal `−z`. For incidence from below, light travels upward along `+z`, and `theta` is measured from that side's inward normal `+z`. In both cases, `phi` is the azimuth of the xy component, measured from `+x` toward `+y`. Changing the incident side does not change the layer list or coordinates. [Chapter 8](light.md) explains direction and polarization in detail.

The two-layer structure in the diagram is a conceptual example connecting the input quantities. The executable example below is a pillar structure with **one patterned layer**.

## Coordinates and layer order shared by both input representations

**Raster and shape input share the same coordinate axes, cell-centered origin, and layer order.** The axes follow the right-hand rule `x̂ × ŷ = ẑ`. **`+z` points upward, and the bottom of the structure is at `z=0`.** In the three-dimensional diagram, `+x` points toward the lower right of the screen, `+y` toward the upper right along the surface, and `+z` vertically upward.

In the xy diagram on the left below, `+x` points right, `+y` points up, and `+z` points out of the screen (`⊙`). This is consistent with looking down at the structure from above. The right-hand diagram shows the layer order in a side view of the same structure.

```{figure} ../../figures/learn/learn/03-coordinates.svg
:alt: The left panel is a right-handed xy coordinate diagram with x to the right, y upward, and z out of the screen. A raster array with four rows and four columns is drawn as a monochrome grid, omitting material values. The gray divisions on the right show the order and thickness of two layers, omitting their material patterns.
:figclass: learning-figure

Figure 3-2. The left panel shows the xy coordinates of one layer; the right shows the layer order along z. Both are **shared by raster and shape input**. The overlaid grid and row/column numbers on the left are **an example of a raster array**. `[layer, row y, column x]` is likewise raster-array notation. Shape input specifies centers, sizes, and materials in the same xy coordinates.
```

`Stack.bottom` is the semi-infinite medium below the bottom, and `Stack.top` is the semi-infinite medium above the complete structure. Neither is included in the layer list.

| Input or array | Meaning |
| --- | --- |
| `Layer(epsilon=array[y, x])` | Raster: row, column. For an oblique lattice: v, u |
| `period=(Px, Py)` | Explicit pairs put x first |
| `Numerics(orders=(Mu, Mv))` | The retained-order half-widths along the lattice vectors, u first (x first on a rectangular cell); always an explicit pair |
| `result.T[row, column]` | In result arrays, v-direction orders occupy rows and u-direction orders occupy columns |
| `layers=(Layer(thickness=d0, ...), Layer(thickness=d1, ...))` | Layers listed from bottom to top |
| `top`, `bottom` | Upper and lower semi-infinite media; these names remain fixed when the incident direction changes |

The xy cell is centered at `(0, 0)`. In a rectangular raster cell, **row 0 is on the −y side**, and the first pixel center is at `−Py/2 + Py/(2·number of rows)`. The same rule applies to x. `+z` points from the bottom upward, and the index in `layers` increases in that order. Some older descriptions say “the first row is at y=0”; that is not the current convention.

### Stacking layers in code

```python
from wavebit import Material, rcwa

stack = rcwa.Stack(
    bottom=Material(epsilon=2.25),       # Substrate: z < 0
    layers=(
        rcwa.Layer(thickness=100., background=Material(epsilon=4.)),   # z: 0 → 100
        rcwa.Layer(thickness=200., background=Material(epsilon=9.)),   # z: 100 → 300
        rcwa.Layer(thickness=50., background=Material(epsilon=2.)),    # z: 300 → 350
    ),
    top=Material(epsilon=1.),            # Upper air: z > 350
    period=(600., 600.),
)
light = rcwa.PlaneWave(side="top", polarization="s")
```

To illuminate from below, change only `side="bottom"`. Do not reverse the layers. `R` is the light returning to the incident side; `T` is the light leaving through the opposite side.

## Finding the right material input

This chapter starts with **assigning constant materials to regions**. [Chapter 7: Materials](materials.md) covers the full set of input methods and anisotropy.

| Material to represent | Input method | Details |
| --- | --- | --- |
| Constant isotropic permittivity | `Material(epsilon=12.25)`; a complex value if there is absorption | The constant-material example below and {ref}`Chapter 7 <material-inputs>` |
| Permittivity and permeability together | `Material(epsilon=4., mu=1.2)` | The ε–μ example below and [Chapter 7](materials.md) |
| Silicon or another material from a table | `materials.load('Si')` | [Chapter 7: material selection and wavelength sweeps](materials.md) |
| Your own wavelength-dependent ε–μ data | `TabulatedMaterial(...)` | [Chapter 7: user-supplied material tables](materials.md) |
| An anisotropic material whose response depends on direction | `Material(epsilon=[εx, εy, εz])` or a 3×3 tensor | {ref}`Chapter 7: isotropy and anisotropy <material-anisotropy>` |

**Anisotropic materials are also assigned to layers or shapes as `Material` objects.** Wavebit constructs the computational matrices for the tensor. For anisotropic shapes, integration over material boundaries is the default, while the normal-vector field is still computed numerically. Layers and shapes accept full 3×3 material tensors. The upper and lower semi-infinite media must be isotropic. [Chapter 7](materials.md) gives examples and the support limits.

## Give a constant permittivity as a `Material`

The number inside `Material(epsilon=...)` is **relative permittivity εr**. For an example material whose permittivity is fixed at 12.25, this is sufficient:

```python
from wavebit import Material

silicon_like = Material(epsilon=12.25)
```

Use it as a region's `material`, a layer's `background`, or the stack's `top` or `bottom`. **Every material slot takes a `Material`; a bare number such as `12.25` is refused there** (code `material.type`), because it would not say whether it is a permittivity or a refractive index. The variable above is an example with an assumed constant permittivity, not loaded silicon data.

### Example: pillars in air on a substrate

The code below can be run as written. Each 600×600 nm cell contains a repeated pillar with a diameter of 240 nm and a height of 180 nm. The pillar's relative permittivity is 12.25, the surrounding air's is 1, and the substrate below has relative permittivity 2.25.

```{figure} ../../figures/learn/learn/03-cylinder-materials.svg
:alt: A three-dimensional view and a top view of one cell containing a pillar on a substrate. The cell measures 600×600 nm; the centered pillar has a diameter of 240 nm and a height of 180 nm. Below z=0 is the semi-infinite substrate bot with permittivity 2.25, and above z=180 is air top with permittivity 1. Within the patterned layer, the pillar has material 12.25 and the surrounding air has background 1. Light at a wavelength of 900 nm is normally incident from above.
:figclass: learning-figure

Figure 3-3. **One cell matching the code.** The dashed box marks the extent of the patterned layer; it is not an object's wall. The pillars repeat every 600 nm along x and y. The substrate extends indefinitely downward; the finite substrate thickness in the drawing is only a visual abbreviation. Every medium in this example has μr=1.
```


<!-- material-assignment-example:start -->
```python
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(
    units=Units(length="nm"),            # every length in nm
    stack=rcwa.Stack(
        period=(600., 600.),             # Repetition spacings along x and y
        top=Material(epsilon=1.),        # Upper air: relative permittivity 1
        bottom=Material(epsilon=2.25),   # Lower substrate: relative permittivity 2.25
        layers=(rcwa.Layer(
            thickness=180.,              # Thickness of the patterned layer
            background=Material(epsilon=1.),   # Air outside the shape in the patterned layer
            regions=(rcwa.Region(
                geometry=rcwa.Circle(radius=120.),   # Centred in the cell; radius 120 nm
                material=Material(epsilon=12.25)),)),)),   # Inside the circle: relative permittivity 12.25
    excitation=rcwa.Excitation(
        wavelength=900.,
        sources=(rcwa.PlaneWave(side="top", theta=0., polarization="s"),)),   # Normal incidence from above
)

result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(1, 1)))   # Low order for a quick example run
print(f"Total reflectance: {float(result.R.sum()):.6f}")
print(f"Total transmittance: {float(result.T.sum()):.6f}")
```
<!-- material-assignment-example:end -->

Output from running the code above:

```text
Total reflectance: 0.044164
Total transmittance: 0.955836
```

The reflectance is about **4.42%**, and the transmittance is about **95.58%**. Both are fractions of the incident energy, and their sum is 1 within rounding error. **These are example execution values calculated with `orders=(1, 1)`, not final predictions with verified convergence.**

**There are four places to assign materials.** `top` is the space above the pattern, `bottom` is the substrate below it, `background` is the space outside the shapes within the patterned layer, and `material` is the interior of a shape. The upper air and the layer background both use 1 here, but they are distinct regions. To change the pillar material, change only the `Material` assigned to the region.

`rcwa.Circle(radius=120.)` is centred on the cell by default (`center=(0., 0.)`), and `radius` is a **radius**: the pillar is 240 nm across. Because every length uses nm, read `wavelength`, `period`, `thickness`, and `radius` on the same scale. The outputs are reflectance and transmittance summed over all diffraction orders.

The low Fourier order here is intended to make it quick to check the inputs and execution flow. Accuracy is addressed in [Chapter 10's discussion of orders](fourier-orders.md) and [Chapter 19's convergence checks](convergence.md). [Chapter 7](materials.md) compares the material input methods.

### Specifying ε and μ together

To specify relative permittivity εr and relative permeability μr together, pass both values to a `Material` object. The following standalone example uses **the same pillar structure, changing only the pillar material to εr=4 and μr=1.2**. This is a hypothetical material specified directly by those two values, not loaded experimental material data.

<!-- magnetic-material-example:start -->
```python
from wavebit import Material, Units, rcwa

pillar_material = Material(epsilon=4., mu=1.2)

problem = rcwa.Problem(
    units=Units(length="nm"),            # every length in nm
    stack=rcwa.Stack(
        period=(600., 600.),             # Repetition spacings along x and y
        top=Material(epsilon=1.),        # Upper air: relative permittivity 1
        bottom=Material(epsilon=2.25),   # Lower substrate: relative permittivity 2.25
        layers=(rcwa.Layer(
            thickness=180.,              # Thickness of the patterned layer
            background=Material(epsilon=1.),   # Air outside the shape in the patterned layer
            regions=(rcwa.Region(
                geometry=rcwa.Circle(radius=120.),   # Centred in the cell; radius 120 nm
                material=pillar_material),)),)),   # Assign both εr and μr to this pillar
    excitation=rcwa.Excitation(
        wavelength=900.,
        sources=(rcwa.PlaneWave(side="top", theta=0., polarization="s"),)),   # Normal incidence from above
)

result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(1, 1)))   # Low order for a quick example run
print(f"Total reflectance: {float(result.R.sum()):.6f}")
print(f"Total transmittance: {float(result.T.sum()):.6f}")
```
<!-- magnetic-material-example:end -->

Output from running the code above:

```text
Total reflectance: 0.007886
Total transmittance: 0.992114
```

These outputs are also example execution values checked at the low order `(1, 1)`.

Assigning `pillar_material` to a shape's `material` applies **both εr=4 and μr=1.2 to that shape**. You can assign `Material` objects to `top`, `bottom`, and `background` in the same way. `Material(epsilon=4.)`, with μ omitted, is a nonmagnetic material with μr=1. Its material differs from the nonmagnetic pillar above, so the calculated response differs as well.

## When external data are given as refractive index

Refractive index and permittivity are different quantities. If external data for a nonmagnetic material give `n=3.5`, convert this to `εr=n²=12.25`. In code, write `Material(epsilon=3.5**2)` or `Material(epsilon=12.25)`. If you already know the permittivity, do not convert it. The general isotropic relationship is `n²=εr μr`, so for a magnetic material you must not enter `n²` alone as the permittivity.

To use actual wavelength-dependent silicon data, select the material with `wavebit.materials.load('Si')`. The [materials chapter](materials.md) distinguishes constants, dispersive tables, and refractive-index data.

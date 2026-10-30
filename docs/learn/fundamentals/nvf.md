# 11. Where does NVF enter the calculation?

:::{note}
This page's text and code use the current `rcwa` API. Its figures may still show the earlier
constructor's names (`bot`, `structure`, `lx` for a circle's diameter); see
[conventions](conventions.md) for their current spellings.
:::

NVF is **a way of constructing Fourier matrices for products of materials and fields while respecting material boundary conditions**. These matrices determine each layer’s eigenmodes; matching boundary conditions between layers then gives reflection, transmission, and electric and magnetic fields. This chapter follows the need for NVF through its directional representation, settings and calculation sequence.

## Components continuous across a material boundary

Directions **parallel to a material interface are tangential**; the **perpendicular direction is normal**. Label the materials on either side 1 and 2. At an ordinary dielectric boundary with no free surface charge, both conditions below hold. **Continuous** means that the limiting values immediately on either side are equal.

- **Tangential electric field {math}`E` is continuous:** {math}`E_{1,\parallel}=E_{2,\parallel}`.
- **Normal electric displacement {math}`D` is continuous:** {math}`D_{1,\perp}=D_{2,\perp}`.

For isotropic materials, {math}`D=\varepsilon_0\varepsilon_r E`. At an interface between different permittivities, **tangential electric displacement can differ even when tangential electric field matches, and normal electric field can differ even when normal electric displacement matches.** For example, adjoining relative permittivities 2 and 4 require {math}`2E_{1,\perp}=4E_{2,\perp}`, so the second material’s normal electric field is half the first’s.

NVF uses the boundary normal to distinguish tangential and normal components and apply Fourier-product rules consistent with their continuity. Curved boundaries especially require local directional information because their tangent and normal directions vary with position.

```{figure} ../../figures/learn/learn/10-nvf.svg
:alt: Raster and shape inputs pass through NVF matrix construction before joining the common solver
:figclass: learning-figure

Figure 11-1. Raster and shape inputs pass through NVF matrix construction before joining the common solver
```

## Including boundary directions in the matrices

Geometry determines the boundary normal `n`. An auxiliary field `J`, extended throughout the cell, supplies a projector `P` used in constitutive-matrix assembly. A real unit vector gives `P=nnᵀ`; a suitably normalized complex auxiliary vector gives `P=JJᴴ`. As a matrix acting on vectors, `P` enters the operation selecting normal components. Do not interpret its extension away from boundaries or its complex construction as one literal geometric normal everywhere.

## NVF calculation options

| Option | What it selects |
| --- | --- |
| `raster_normal_method` | Construction of a raster auxiliary direction field: distance / contour / spectral / smooth |
| `shape_nvf_grid_size` | Shape NVF samples per axis; 128 means 128×128 |
| `nvf_projector` | NVF projector construction: linear / circular |

For shapes with `fourier_coefficients='integrated'`, **material Fourier coefficients come from integrating the shape regions**. `shape_nvf_grid_size` instead controls **the grid for NVF auxiliary fields and projectors representing boundary-normal directions**.

For a circular shape with `shape_nvf_grid_size=128`, material coefficients come from integrating the circle defined by its center and radius, while NVF uses a 128×128 grid. Increasing this to 256 refines the NVF grid to 256×256. The integrated material coefficients are unchanged, but different NVF matrices can change final reflectance and transmittance.

Here, `linear/circular` **does not mean incident polarization**. Circularly polarized incidence can use linear NVF, and linearly polarized incidence can use circular NVF. The circular construction offers another treatment of locations where the extended field direction is ambiguous; it is not always more accurate or faster. Raster and shape construction details also differ.

The default shape NVF grid is `max(1024, 4*Nx+1, 4*Ny+1)`. If reducing it to speed up calculations, check convergence of both the requested outputs and their gradients.

## Selecting NVF and running a calculation

Describe the structure and light in `rcwa.Problem` and select `factorization="nvf"` in `rcwa.Numerics`. `rcwa.run(problem, numerics=numerics)` builds the NVF matrices and calculates reflection and transmission.

This example calculates a 180 nm layer containing a circle of radius 160 nm (diameter 320 nm). Lengths are in nm; `Material(epsilon=4.)` and `bottom=Material(epsilon=2.25)` are the relative permittivities of the shape and the lower medium.

```python
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(
    units=Units(length="nm"),
    stack=rcwa.Stack(
        period=(700., 700.),
        top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(
            thickness=180., background=Material(epsilon=1.),
            regions=(rcwa.Region(geometry=rcwa.Circle(radius=160.),
                                 material=Material(epsilon=4.)),)),)),
    excitation=rcwa.Excitation(wavelength=900., sources=(
        rcwa.PlaneWave(side="top", theta=0.2, phi=0.3, polarization=(0., 1.)),)))
numerics = rcwa.Numerics(
    orders=(2, 2),
    fourier_coefficients="integrated",
    factorization="nvf",
    nvf_projector="linear",
    shape_nvf_grid_size=128,
)
result = rcwa.run(problem, numerics=numerics)
print("Total reflectance:", result.R.sum())
print("Total transmittance:", result.T.sum())
```

`integrated` and `nvf` are already shape-input defaults, but are written explicitly to show the methods. `polarization=(0., 1.)` selects p incidence. `result.R` and `result.T` are arrays of diffraction-order reflectance and transmittance; summing each gives its total.

## Checking convergence of the result

Convergence checking **tests whether the quantity of interest stays sufficiently stable as calculation settings are refined**. The Fourier order and NVF grid in the example are starting settings. Compare the following according to your required accuracy.

1. Hold `orders` fixed and increase `shape_nvf_grid_size`, for example through 128, 256 and 512. Compare the requested reflectance and transmittance.
2. Hold NVF grid size fixed and increase `orders`, for example through `(2, 2)`, `(3, 3)` and `(4, 4)`.
3. Refine both together and check that changes remain within the required tolerance. If optimization uses derivatives, compare those as well.

All materials here are lossless, so also check whether total reflectance plus transmittance is near 1. However, individual reflection and transmission values can still vary with order or grid size despite conservation; convergence comparisons remain necessary.

For a fuller treatment of normal fields and projectors, continue with the [NVF principles, implementation and usage course](../nvf/index.md). The [NVF convergence example](../nvf/convergence.md) compares results as settings change.

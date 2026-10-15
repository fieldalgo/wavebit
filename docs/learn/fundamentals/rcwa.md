# 02. RCWA: solving complex light as a sum of waves

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

RCWA stands for Rigorous Coupled-Wave Analysis. Here, “splitting” does not mean dividing space into a fine grid for time stepping. It means expressing the transverse distribution within a layer as **a sum of waves with different spatial frequencies**. The periodic material couples these waves to one another.

```text
E(x, y, z) ≈ Σ E_mn(z) · exp[i (k_parallel + m b₁ + n b₂) · (x, y)]
                         ↑ spatial oscillation of each Fourier order
```

`b₁` and `b₂` are the reciprocal-lattice vectors. In a rectangular cell, the spacing of x components is `2π/Px`, and the spacing of y components is `2π/Py`. `m` and `n` are the order indices, and finding each wave's coefficient `E_mn(z)` is the heart of the calculation. We cannot retain infinitely many waves, so the expansion is truncated to a finite set of orders.

```{figure} ../../figures/learn/learn/02-rcwa.svg
:alt: The roles of the periodic cell, Fourier waves, modes in each layer, and connections between layers
:figclass: learning-figure

Figure 2-1. The roles of the periodic cell, Fourier waves, modes in each layer, and connections between layers
```

Each layer has a cross-section that is constant along z. The solver calculates the propagating and decaying modes available in that layer and their z dependence, then matches the boundary conditions between layers. A structure with sloping sidewalls and a cross-section that varies with z must be approximated by several thin layers; you also need to check how the result changes with this layer subdivision.

**Diffraction orders and layer eigenmodes are not the same thing.** In a homogeneous exterior medium, each order has an identifiable propagation direction. Inside a patterned layer, an eigenmode is a coupled combination of several Fourier components. “Retaining 25 orders” does not mean “there are 25 independent rays inside the pillar.”

## From structure and materials to the RCWA calculation

In Wavebit, this becomes the following calculation sequence. **Construct computational matrices from the structure and materials, find the eigenmodes of each layer, then match the boundary conditions between layers to calculate reflection, transmission, and electromagnetic fields.** Li and NVF in the figure are two ways to represent products of material properties and fields as matrices in the first stage; `factorization` selects between them.

```{figure} ../../figures/learn/learn/02-factorization-flow.svg
:alt: Geometry is processed with Li or NVF, and material values are supplied to both paths to construct material convolution matrices before they join the common RCWA calculation
:name: factorization-flow
:figclass: learning-figure

Figure 2-2. From structure and materials to the RCWA calculation. This conceptual diagram omits details such as wavelength and incident conditions and the computational shortcut for uniform layers.
```

Solid lines show the calculation flow; dotted lines show **the transfer of material values ε and μ**. Both Li and NVF use those material values. On the NVF path, boundary directions define a normal-vector field and projection matrices, which are then combined with the material values to construct convolution matrices. Material boundaries are the locations where different materials meet.

The **material convolution matrices** in the figure are computational matrices describing relationships between retained Fourier orders. The figure's `conv_mat` bundle means the convolution matrices constructed for the individual components. Their sizes and roles differ from those of a 3×3 permittivity or permeability tensor describing the material at a single position.

[Chapter 10](fourier-orders.md) explains Fourier orders and the conditions for using each matrix construction method. [Chapter 11](nvf.md) explains NVF's normal-vector field and projection matrices in more detail.

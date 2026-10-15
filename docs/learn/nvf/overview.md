# 01. What does NVF do inside RCWA?

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

RCWA calculates light scattering by periodic structures. It represents the material and fields in a layer's xy cross section with Fourier components, finds modes that propagate along z, and connects the boundary conditions between layers. The final outputs include reflectance and transmittance for each diffraction order.

```{figure} ../../figures/learn/nvf/01-rcwa.svg
:alt: A repeating xy cell and a stack of layers along z
:figclass: learning-figure

Figure 1-1. A repeating xy cell and a stack of layers along z
```

Fourier orders are labeled by integer pairs `(m,n)`. If the primitive reciprocal-lattice vectors are `b₁` and `b₂`, the wavevector for an order is `m b₁ + n b₂`. This is the notation used in [Fundamentals Chapter 2](../fundamentals/rcwa.md) and [Chapter 6](../fundamentals/lattice.md). In a rectangular cell of size `(Lx, Ly)`, `b₁=(2π/Lx, 0)` and `b₂=(0, 2π/Ly)`, giving `m b₁ + n b₂ = 2π(m/Lx, n/Ly)`. The electric field's transverse wavevector also includes the incident light's parallel wavevector. An actual calculation retains only a subset of the infinitely many `(m,n)` pairs.

```text
E(x,y,z) ≈ Σ E_mn(z) exp[i(k_parallel + m b₁ + n b₂) · (x,y)]
           m = −Nx … Nx,  n = −Ny … Ny

Number of retained spatial modes M = (2Nx + 1)(2Ny + 1)
```

When ε and an electric-field component are both discontinuous at a material boundary, simply multiplying their truncated Fourier series can represent the boundary conditions poorly. NVF **uses material-boundary normals to help apply the appropriate Fourier product rule to each component.** It constructs material operators within RCWA; it is not a separate optical solver that replaces RCWA as a whole.

NVF was not invented by Wavebit. The foundations for Fourier products of discontinuous functions appear in [Li's 1996 paper](https://doi.org/10.1364/JOSAA.13.001870), and applying normal vector fields to RCWA for crossed gratings is described by [Schuster et al. in 2007](https://doi.org/10.1364/JOSAA.24.002880). Against that background, this book explains **the finite-matrix assembly and API actually chosen by the Wavebit implementation covered here**. It does not imply that every implementation detail below is identical to those two papers.

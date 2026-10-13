# 05. How Wavebit assembles NVF

:::{note}
The runnable examples in this imported course explicitly use `wavebit.compat.legacy` to preserve the workflow behind its historical numerical records. For new work, use the [current getting-started guide](../../getting-started/index.md) and [API migration guide](../../migration/index.md). Historical measurements were not regenerated during this documentation import.
:::

We now connect the principles to the code. The equations in this chapter are readable restatements of [`_nvf_assemble`](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/factorization.py) and [`nvf_from_coeffs`](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/parametric.py) in the reference implementation. Every matrix uses the same ordering of retained modes.

```{figure} ../../figures/learn/nvf/06-assembly.svg
:alt: Combining permittivity and projector coefficients produces five slots
:figclass: learning-figure

Figure 5-1. Combining permittivity and projector coefficients produces five slots
```

```text
A = C(ε),  B = C(1/ε),  Δ = A − B⁻¹
Cij = C(Pij)

Kxx = A − ½(Δ Cxx + Cxx Δ)
Kyy = A − ½(Δ Cyy + Cyy Δ)
Kxy =   − ½(Δ Cxy + Cxy Δ)
Kyx =   − ½(Δ Cyx + Cyx Δ)
```

The K blocks describe transverse material response in the finite Fourier space. In the figure, `δij` is 1 for i=j and 0 otherwise. The simple spatial intuition is “A for tangential components, B⁻¹ for normal components.” Truncated Fourier matrices, however, generally change their product when the multiplication order changes. Wavebit uses **the average of both orders**, rather than only `Δ Cij`.

For lossless real materials and appropriate Hermitian projectors, this symmetrization preserves the Hermitian structure of the full block matrix. It does not force the response of lossy materials to be Hermitian. Complex loss must remain present.

## Two hand calculations that explain the code

If the normal points along x throughout the cell, `Cxx=I` and `Cyy=Cxy=Cyx=0`. The equations immediately give `Kxx=B⁻¹`, `Kyy=A`, and `Kxy=Kyx=0`. The normal x direction receives the inverse rule, and the tangential y direction receives the direct rule.

For a homogeneous material, `A=εI` and `B=(1/ε)I`, so `Δ=0`. The correction vanishes regardless of how the direction field is described. This agrees with NVF's purpose of handling discontinuities at material boundaries.

## The five slots and their minus signs

Internally, the following `ConvMatrices` slots are passed to the optical engine in this order:

| Slot | Symbol in this chapter | Interpretation |
| --- | --- | --- |
| `epx` | Kxx | x-x material response |
| `epy` | Kyy | y-y material response |
| `epz_i` | A⁻¹ | Inverse of the direct matrix used for the z component |
| `epxy` | −Kxy | Cross component with the engine's sign convention |
| `epyx` | −Kyx | Cross component in the opposite direction |

Here `epz_i=A⁻¹` differs from the normal component's `B⁻¹`. Recall the geometry: boundaries change direction within the xy plane, while a layer retains the same cross section along z. The minus signs in the cross slots are also conventions of the engine's variables and equations, not additional negative material values.

Ordinary users do not need to construct these matrices themselves. `wavebit.rcwa(..., factorization='nvf').solve()` connects their construction to the optical calculation. A research path with material matrices already assembled can pass them to `solve_conv()`, but changing the factorization setting in that case does not reassemble the supplied matrices.

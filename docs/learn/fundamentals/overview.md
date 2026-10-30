# 01. What questions does Wavebit answer?

:::{note}
The downloadable teaching scripts, the text and the snippets of every chapter use the current
`rcwa.Problem` / `rcwa.run` API. A few passages describe features the current API does not have
yet, such as the `psi`/`chi` polarization angles and the pure 1D `tetm` path; they sit in boxes
titled "Earlier interface" (retired with `wavebit.compat` on 2026-10-15). Some figures keep the
earlier labels, and the chapters that show them name the current equivalents. Compare the
[current getting-started guide](../../getting-started/index.md) and
[migration guide](../../migration/index.md). Historical measurement files and their source
provenance remain unchanged.
:::

“When light illuminates a periodically repeated nanostructure, where does it go, and how much goes in each direction?” This is the central question addressed by the RCWA calculations in this course. Given the wavelength, materials, geometry, layer thicknesses, incident direction, and polarization, Wavebit calculates the reflected and transmitted intensities by direction, complex amplitudes, and electromagnetic fields inside the structure. With JAX or PyTorch, you can also find how much a result changes when a design variable changes slightly.

```{figure} ../../figures/learn/learn/01-journey.svg
:alt: The complete flow from an input scene through calculation and results to a change in the design
:figclass: learning-figure

Figure 1-1. The complete flow from an input scene through calculation and results to a change in the design
```

For example, suppose you want to “place a periodic array of silicon pillars on glass and increase the zeroth-order transmission of 900 nm light.” First define the structure and the light. Run a calculation, then read `result.order(0, 0).T`. Changing a pillar's diameter or height is a design change; increasing the Fourier order calculates the same design more accurately. Keeping these two activities distinct helps reduce the chance that optimization exploits numerical error.

## Scope of this course and the role of FDTD

The solver taught in this course is **RCWA**. It represents temporal oscillation with complex numbers and solves the problem at a specified wavelength. It is suited to layered structures that are periodic in the transverse directions. FDTD is a separate computational method that advances electromagnetic fields on a grid through time. The September 2026 working copy used for this book's measurements did not yet have an integrated FDTD user API. The current repository provides the public `wavebit.fdtd` module with `fdtd.Problem`, `fdtd.prepare` and `fdtd.run`; see the [current FDTD guide](../../guide/fdtd/general-fdtd.md). Earlier periodic-stack and general-scene factories were retired from `wavebit.compat.legacy`; their [periodic FDTD manual](../../guide/fdtd/time-domain.md) and [general FDTD manual](../../migration/general-fdtd-v1.md) state the compatibility scope. These APIs are not part of the historical RCWA measurement record reproduced here.

| Question | Approach |
| --- | --- |
| Reflection/transmission of periodic gratings, metasurfaces, or multilayer films | Build a layered structure that RCWA can represent |
| Local electric fields inside a structure | `answers='field'` and field reconstruction |
| Sensitivity to material or geometry variables | JAX or PyTorch automatic differentiation, after checking support for the chosen path |
| Scattering from a single finite object | A large periodic cell may be used, with the influence of periodic boundaries examined; distinguish this from an isolated-object solution |
| Using integrated FDTD | Follow the current FDTD guide for the periodic and general APIs and their respective support limits |

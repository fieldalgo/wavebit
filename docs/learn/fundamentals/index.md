# A book about the whole of Wavebit

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

**From drawing a structure to calculating its optical response and changing its design.** Rather than list Wavebit's features one by one, this book follows a calculation through its stages. If you have already studied NVF, you can now see where it fits into the complete solver.

The examples and recorded measurements in this book were prepared on September 21, 2026, using a **local working copy with API renaming and dispersive-material improvements**, based on Wavebit 0.1.0 / develop `a1d23fc`. The examples below were run with that working copy. FDTD had not yet been integrated into that snapshot, so the worked examples in this course cover RCWA. The current repository provides the public `wavebit.fdtd` module with `fdtd.Problem`, `fdtd.prepare` and `fdtd.run`; see the [current FDTD guide](../../guide/fdtd/general-fdtd.md). Earlier periodic-stack and general-scene factories remain under `wavebit.compat.legacy`; their [periodic FDTD manual](../../guide/fdtd/time-domain.md) and [general FDTD manual](../../migration/general-fdtd-v1.md) state the compatibility scope. These historical execution records are not a declaration of a public release or of completion of the full regression suite.

## How to read this book

| Goal | Reading order |
| --- | --- |
| Understand the overall picture | Chapters 1–3 → the first calculation in Chapter 4 → results in Chapter 12 |
| Understand efficiency and absorption | Incident conditions in Chapter 8 → results and power in Chapters 12–13 |
| Understand back-surface reflection and measurement averaging in a thick substrate | Connecting layers in Chapter 15 → Appendix C (thick substrates) |
| Calculate my own structure | Chapters 4–8 → Chapters 10–14 → accuracy checks in Chapter 19 |
| Calculate several incident waves or illumination from both sides | Basic inputs in Chapter 8 → multiple incident waves in Chapter 9 → fields in Chapter 14 |
| Perform inverse design and optimization | Chapters 16–18, especially differentiation and constraints in Chapter 17 |
| Study the internal code | Chapters 2, 10, 11, and 15 → the code map in Chapter 20 |
| Understand complex angles, gain media, or differences from other solvers | Complex transverse wavevectors in Chapter 8 → Appendix A (lossless media) → Appendix B (absorbing and gain media) |

**20 chapters and three appendices, with figures, interactive concepts, and 10 runnable Python examples.** Figures marked “conceptual diagram” are explanatory. Figures showing numerical results identify the associated executable file. You can read the code in the documentation or download the executable files. Click a figure to enlarge it.

## Contents of this course

- [1. What questions does Wavebit answer?](overview.md)
- [2. RCWA: solving complex light as a sum of waves](rcwa.md)
- [3. Conventions to establish first: coordinates, units, and materials](conventions.md)
- [4. Installation and a complete first calculation](quickstart.md)
- [5. Two languages for building structures: rasters and shapes](geometry.md)
- [6. Beyond square cells: lattices and two coordinate systems](lattice.md)
- [7. Materials: permittivity, absorption, and wavelength dependence](materials.md)
- [8. Choosing the incident light: direction and polarization](light.md)
- [9. Multiple incident waves and internal sources](multiple-beams.md)
- [10. Fourier orders determine accuracy and cost](fourier-orders.md)
- [11. Where does NVF fit into the full calculation?](nvf.md)
- [12. Reading results: uppercase R/T and lowercase r/t](results.md)
- [13. Power flow and diffraction efficiency](power-and-efficiency.md)
- [14. Obtaining electric and magnetic fields at each position](fields.md)
- [15. Connecting layers: ETM and SMM](layer-algorithms.md)
- [16. NumPy, JAX, and PyTorch: the same physics, different computational tools](backends.md)
- [17. Automatic differentiation and optimization: which way should a design change?](autodiff.md)
- [18. From one calculation to an experiment: JobSpec and sweeps](jobs-and-sweeps.md)
- [19. Judging a result: energy conservation is not enough](convergence.md)
- [20. A code map and further study](code-map.md)

Appendices

- [A. Choosing kz in lossless media: two roots, branch points, and branch cuts](complex-kx-roots.md)
- [B. Extending to absorbing and gain media: absorbing incident media, displaced branch points, and gain media](complex-kx-lossy-gain.md)
- [C. Thick substrates: back-surface reflection and averaging interference](incoherent-substrate.md)

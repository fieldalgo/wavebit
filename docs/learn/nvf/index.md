# Learning NVF: principles, implementation, and use in Wavebit

:::{note}
The runnable examples in this imported course explicitly use `wavebit.compat.legacy` to preserve the workflow behind its historical numerical records. For new work, use the [current getting-started guide](../../getting-started/index.md) and [API migration guide](../../migration/index.md). Historical measurements were not regenerated during this documentation import.
:::

This course explains how electric-field components behave at material boundaries and follows those principles through to Wavebit's material matrices and executable code.

**Baseline: a Wavebit 0.1.0 development working copy.** The discussion focuses on nonmagnetic materials and scalar relative permittivity. Consult the English user guide as well for the scope of tensor materials and advanced auxiliary cells.

**This book has three goals:** to help you explain why NVF is needed, read how Wavebit turns a normal field into a material matrix, and apply NVF to your own structure and check the accuracy of the result.

The earlier [development-history tutorial](../../validation/rcwa/nvf-contact-followup.md) describes problems that were fixed and the checks that supported those fixes. This book is a self-contained course on principles and use. One possible route is to learn the principles in Chapters 1–4, run the example in Chapter 7, then return to Chapters 5–6 to read about the implementation.

The core course includes explanatory figures, interactive concept demonstrations, and five runnable Python examples. Concept diagrams and browser interactions are not RCWA calculation results; actual results are identified separately in tables and figures with their provenance.

**Read the metal case study in Chapters 13–15 in order.** First identify suspicious behavior in the original plots, then understand the small denominators in the inverse rule, and finally assess the benefits and limitations of imposing a singular-value floor. These three chapters explain saved experimental results and do not run new calculations.

## Course contents

- [1. What does NVF do inside RCWA?](overview.md)
- [2. Start with boundary conditions: the whole E field is not continuous](boundary-conditions.md)
- [3. Direct and inverse rules: which matrix is inverted?](fourier-rules.md)
- [4. Boundary normal n → auxiliary vector J → projector P](projectors.md)
- [5. How Wavebit assembles NVF](implementation.md)
- [6. Two input types and the role of each setting](settings.md)
- [7. A first calculation: NVF reflection and transmission for a circular inclusion](shape-example.md)
- [8. Using pixel input and selecting NVF](raster-example.md)
- [9. Before trusting a result: how to check convergence](convergence.md)
- [10. Differentiating with JAX: start with material values](autodiff.md)
- [11. Multiple shapes, auxiliary solvers, and common misconceptions](multiple-shapes.md)
- [12. A code-reading map and review](code-map.md)
- [13. Finding suspicious metal results in the plots](metal-original-plots.md)
- [14. Small denominators, large responses: metals and the inverse rule](metal-inverse-rule.md)
- [15. Placing a floor on small singular values: correction and validation](metal-regularization.md)

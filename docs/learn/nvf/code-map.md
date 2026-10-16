# 12. A code-reading map and review

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

This course recommends starting with the public API. If you read internal functions, the following order connects them most directly to the principles. The file and function locations describe the reference implementation and may change later.

| Question | Where to read | Related chapter |
| --- | --- | --- |
| Defaults, supported combinations, and rejection of unused settings | [`engine.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/engine.py), `_resolve_normal_grid`, `_reject_unread_*` | 6 |
| Raster normal estimation | [`normal_methods.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/normal_methods.py), `normal_vector_field` in [`factorization.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/factorization.py) | 4, 6 |
| Shape direction fields and projector coefficients | [`parametric.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/parametric.py), `normal_vector_field_analytical`, `normal_projector_coeffs` | 4 |
| Symmetrized NVF assembly | `_nvf_assemble` in [`factorization.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/factorization.py), `nvf_from_coeffs` in [`parametric.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/parametric.py) | 5 |
| Slot names and return conventions | `ConvMatrices` in [`factorization.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/factorization.py) | 5 |
| Inputs, units, polarization, and result axes | [Keywords](../../reference/keywords.md), [Conventions](../../reference/common/conventions.md) | 7, 8 |

## Exercises

1. Change `material=4.` to 1 in Example 1. The material contrast between the patterned region and background disappears. Check whether changing the circle size also stops changing the result. The substrate remains, so total reflectance need not become zero.
2. Change Example 1's `truncation_order=(2,2)` to `(3,1)`. Check that the output array shape is `(3,7)`, and explain why using the mode-count formula.
3. In Chapter 3's 50:50 example, set both permittivities to 4. Calculate whether A and B⁻¹ become equal and Δ becomes zero.
4. Explain how n and J differ at the right boundary of a circle and at its center. In the interaction, compare θ=30° with −150° and check why opposite arrows give the same P.
5. In the JAX example, increase only shape_nvf_grid_size and compare both T and dT/dε. Then increase only the truncation order and compare again. Record why agreement with finite differences and convergence of a physical gradient are separate questions.

## A short glossary

| Term | Meaning in this book |
| --- | --- |
| NVF | A method extending boundary-normal information into a domain-wide decomposition reference and incorporating it into Fourier product rules |
| Boundary normal n | The actual unit vector perpendicular to a smooth shape boundary |
| Auxiliary vector J | A computational vector matched to boundary normals and extended into interiors and the background |
| factorization | Here, a rule for representing products of discontinuous functions in Fourier space; a different context from matrix decompositions such as LU |
| Projector P | The matrix performing the component decomposition defined by J: P=JJᴴ/(JᴴJ) |
| raster_normal_method | Method for constructing the direction field in raster NVF |
| shape_nvf_grid_size | Number of samples per axis used to evaluate auxiliary and projector fields in shape NVF |
| nvf_projector | API option selecting the NVF projector construction: linear / circular |
| convolution matrix | Matrix C(f) representing multiplication through convolution of Fourier coefficients |
| truncation | Retaining a finite subset of infinitely many Fourier modes |
| raster | A structure representation assigning a material value to each pixel |
| shape / parametric | A shape representation defined by continuous parameters such as center, width, and angle |
| freeze_geometry | Preparation step recording discrete geometry choices before tracing |
| convergence | Checking that the change in a quantity of interest falls within the required tolerance as numerical resolution increases |

## References and the scope of this course's validation

[L. Li, “Use of Fourier series in the analysis of discontinuous periodic structures,” JOSA A 13, 1870–1876 (1996)](https://doi.org/10.1364/JOSAA.13.001870) provides the background for Fourier factorization of products of discontinuous periodic functions.

[T. Schuster, J. Ruoff, N. Kerwien, S. Rafler, W. Osten, “Normal vector method for convergence improvement using the RCWA for crossed gratings,” JOSA A 24, 2880–2890 (2007)](https://doi.org/10.1364/JOSAA.24.002880) is prior work on the normal-vector method in RCWA.

The original course's authors checked the official publishers' bibliographic entries and abstracts. This book is neither a translation nor a reproduction of the full papers. Its implementation formulas and API descriptions were checked against local Wavebit source; the diagrams and hand-calculation examples were made for this course. No external implementation code or figures were copied.

The five examples were executed on the API-revision working copy described by the original course, with raw numerical outputs and environments preserved in {download}`measurements.json <../../validation/learning-records/nvf-measurements.json>`. These are execution and consistency checks for introductory examples. They do not certify the full regression suite, optical convergence for every structure, or the completeness of advanced auxiliary cells. Source-file hashes are listed in [Build baseline and provenance](../../development/learning-provenance.md), and the documentation-review scope appears in [Review scope](../../development/learning-provenance.md).

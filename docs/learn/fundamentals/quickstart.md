# 04. Installation and a complete first calculation

:::{note}
The runnable examples in this imported course explicitly use `wavebit.compat.legacy` to preserve the workflow behind its historical numerical records. For new work, use the [current getting-started guide](../../getting-started/index.md) and [API migration guide](../../migration/index.md). Historical measurements were not regenerated during this documentation import.
:::

The source requires **Python 3.12 or later**. The commands below install Wavebit from its source directory. The released alpha is also on PyPI (`pip install wavebit`); see [Installation](../../getting-started/installation.md). You do not need to install every backend if you will not use them all.

```sh
python -m pip install .
# In an environment where you will use the JAX examples:
python -m pip install '.[jax]'
# In an environment where you will use the PyTorch examples:
python -m pip install '.[pytorch]'
```

An existing environment was used to verify the examples in this book: NumPy 2.5.1, SciPy 1.18.0, JAX 0.11.0, and PyTorch 2.13.0. The environment and individual execution records are in {download}`measurements.json <../../validation/learning-records/learn-measurements.json>`. No new external code or dependencies were added for this course.

```{figure} ../../figures/learn/learn/04-first-grating.svg
:alt: The first example's half-and-half grating, with a 700 nm period and 300 nm height, and its incident light
:figclass: learning-figure

Figure 4-1. The first example's half-and-half grating, with a 700 nm period and 300 nm height, and its incident light
```

Air and a material with `n=3.5` each occupy half the grating's width. The structure does not vary along y. Below it is glass with `n=1.5`; above it is air. A single row of four pixels describes an **exact half-and-half geometry**, with two cells on the left and two on the right. Even for this simple geometry, a separate Fourier truncation is needed to represent the fields.

Executable file: {download}`01_first_grating.py <../../../examples/rcwa/learning/learn/01_first_grating.py>`

Both the figure and code show **relative permittivity directly**: `1.` for air, `12.25` for the patterned material, and `2.25` for the glass medium below.

```{literalinclude} ../../../examples/rcwa/learning/learn/01_first_grating.py
:language: python
:linenos:
```

| Measured execution result | Value |
| --- | --- |
| Total R | 0.205331628258 |
| Total T | 0.794668371742 |
| Zeroth-order T | 0.353101123079 |
| Result array shape (`result.T.shape`) | (1, 21) |

With the incident energy normalized to 1, about 20.53% is reflected and 79.47% is transmitted. Zeroth-order transmission is about 35.31%; the rest of the transmitted power leaves through other propagating orders. **Total transmission and zeroth-order transmission are different quantities.** In this example, `R+T≈1` checks lossless energy balance, but it does not by itself establish that enough Fourier orders have been retained.

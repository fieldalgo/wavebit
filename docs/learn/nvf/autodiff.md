# 10. Differentiating with JAX: start with material values

:::{note}
The runnable examples in this imported course explicitly use `wavebit.compat.legacy` to preserve the workflow behind its historical numerical records. For new work, use the [current getting-started guide](../../getting-started/index.md) and [API migration guide](../../migration/index.md). Historical measurements were not regenerated during this documentation import.
:::

One of Wavebit's benefits is the ability to differentiate the selected finite calculation. Begin with an example that changes one material value ε while keeping the shape arrangement fixed. This makes the differentiation flow easier to understand without simultaneously handling changes in shape contacts or overlaps.

```{figure} ../../figures/learn/nvf/11-gradient.svg
:alt: Geometry preparation, automatic differentiation, and an independent centered difference
:figclass: learning-figure

Figure 10-1. Geometry preparation, automatic differentiation, and an independent centered difference
```

If your environment does not include JAX, install the documented optional dependency from the Wavebit source directory before running the example. The original course was developed using an existing environment, without adding libraries.

```bash
# From the Wavebit source directory, only if needed
python -m pip install '.[jax]'

# From the course example directory
python examples/04_jax_gradient.py
```

The {download}`complete Example 4 file <../../../examples/rcwa/learning/nvf/04_jax_gradient.py>` uses a rotated rectangle and a circular projector. Its structure differs from the earlier circle example.

```{literalinclude} ../../../examples/rcwa/learning/nvf/04_jax_gradient.py
:language: python
:linenos:
```

`freeze_geometry()` reads a concrete structure and records discrete geometry decisions before tracing. Passing that record through `solve(geometry=geometry)` lets JAX follow the continuous calculation with respect to material values. “Preparing geometry” does not mean cutting off derivatives of every input.

This example reuses its prepared geometry because **only material values change within the same shape arrangement**. Do not blindly reuse a record when shape counts, overlap relations, periodic-image requirements, or similar structure changes. Not every invalid reuse can be detected inside JIT. New geometric relationships must be prepared and checked again from concrete inputs.

## Measured derivative and finite-difference comparison

| Quantity | Measured value |
| --- | --- |
| T_p under JIT | 0.996179995709902 |
| T_p from a fresh eager calculation | 0.996179995709905 |
| Automatic derivative dT/dε | 0.002181443388964 |
| Centered difference, h=0.001 | 0.002181443461702 |
| Centered difference, h=0.0005 | 0.002181443408689 |
| Absolute difference between the last finite difference and autodiff | 1.97×10⁻¹¹ |

The finite differences create fresh solvers at ε+h and ε−h rather than reusing a frozen geometry record. Agreement was checked again after halving h. Smaller h is not always better: subtracting two nearly equal values eventually amplifies roundoff error.

This result shows agreement between the material-derivative implementation and finite differences for this structure and these finite settings. It does not prove convergence to the same physical gradient at higher Fourier order. For example, if optimizing n for a lossless real material with ε=n², then `dT/dn = 2n·dT/dε`; that derivative has a different target variable from the value reported here.

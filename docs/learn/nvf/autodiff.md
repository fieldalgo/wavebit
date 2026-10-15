# 10. Differentiating with JAX: start with material values

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
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

# From the same directory
python examples/rcwa/learning/nvf/04_jax_gradient.py
```

The {download}`complete Example 4 file <../../../examples/rcwa/learning/nvf/04_jax_gradient.py>` uses a rotated rectangle and a circular projector. Its structure differs from the earlier circle example.

```{literalinclude} ../../../examples/rcwa/learning/nvf/04_jax_gradient.py
:language: python
:linenos:
```

`rcwa.prepare()` records discrete geometry decisions from a concrete problem. Calling `prepared.run(updated_problem)` inside JIT keeps the updated material calculation and its JAX derivatives live. “Preparing geometry” does not mean cutting off derivatives of every input.

This example reuses its prepared geometry because **only material values change within the same shape arrangement**. Do not blindly reuse a record when shape counts, overlap relations, periodic-image requirements, or similar structure changes. This example keeps geometry fixed. To vary general shapes inside JIT, select the explicit host topology policy and inspect result validity. New geometric relationships must be prepared and checked again from concrete inputs.

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

# 17. Automatic differentiation and optimization: which change helps?

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

A forward calculation maps design variables to outputs. A derivative asks how much a target quantity changes under a small design variation. Automatic differentiation gives the **derivative of the finite-order numerical program that was executed**. Whether the physical solution is sufficiently converged is a separate question.

```{figure} ../../figures/learn/learn/16-autodiff.svg
:alt: Differentiation path from thickness through matrices and optical response to the objective
:figclass: learning-figure

Figure 17-1. Differentiation path from thickness through matrices and optical response to the objective
```

Set the thickness of a uniform `n=2` film to `d=900q` nm and try to reduce reflectance. Using `q` makes this design variable dimensionless. The returned gradient is `dR/dq`, numerically different from `dR/dd`; here `dR/dd=(dR/dq)/900`.

Runnable file: {download}`05_jax_gradient.py <../../../examples/rcwa/learning/learn/05_jax_gradient.py>`

```{literalinclude} ../../../examples/rcwa/learning/learn/05_jax_gradient.py
:language: python
:linenos:
```

| Recorded calculation at q = 0.1 | Value |
| --- | --- |
| R | 0.193241233492 |
| JAX dR/dq | 1.051632496046 |
| PyTorch dR/dq | 1.051632496046 |
| Central-difference dR/dq | 1.051632489893 |
| Absolute JAX–central-difference difference | 6.153e-09 |

The small difference between central differences and automatic differentiation checks the implemented derivative **at this point for this uniform film**. It does not validate every geometry or variable. Compare several finite-difference spacings and also vary precision and order.

```{figure} ../../figures/learn/learn/17-optimization.svg
:alt: Reflectance decrease and arrival at the thickness lower bound recorded by the teaching example
:figclass: learning-figure

Figure 17-2. Reflectance decrease and arrival at the thickness lower bound recorded by the teaching example
```

The iteration reaches the lower bound imposed by `q≥0.03`. Repeated values afterward do not mean continuing improvement, nor establish a global optimum. A real objective should include manufacturable dimensions, minimum gaps, material constraints and performance across relevant wavelengths and angles.

{download}`06_torch_gradient.py <../../../examples/rcwa/learning/learn/06_torch_gradient.py>` calculates the same gradient with PyTorch. Read `q.grad` after `.backward()`; iterative optimization must reset accumulated gradients as intended. The JAX example in this course uses eager execution, not JIT.

## Why JIT can require freeze_geometry

Some geometry stages branch on concrete values to determine boundary connectivity or containment. Traced values do not always permit these decisions. `freeze_geometry()` computes this information from a concrete scene and supplies it as `StaticGeometry`.

```python
# Conceptual JIT preparation after assigning a pattern with concrete values
geometry = solver.freeze_geometry()
# Pass geometry=geometry to a supported solve path afterward.
```

This is a preparation sketch, not a complete JIT optimization example. Fixed geometry whose material values alone change can reuse the record. Refresh it when geometry, contact or covering relationships change. Shape and raster paths record different information, so the same criterion does not cover every motion. Validity comparisons may be unavailable inside JIT, making updates outside tracing essential to avoid silently using stale records. See the [compatibility differentiation manual](../../migration/autodiff-v1.md) and [NVF course](../nvf/index.md) for complete patterned JIT workflows.

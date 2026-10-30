# 16. NumPy, JAX and PyTorch: the same physics with different computation tools

A backend provides arrays, linear algebra and automatic differentiation. Changing backend does not replace RCWA with a different physical theory. The standard solvers share a core, while backend adapters handle differences in arrays, eigensolvers and related operations.

```{figure} ../../figures/learn/learn/15-backends.svg
:alt: Shared physics core, three array backends and optional automatic differentiation
:figclass: learning-figure

Figure 16-1. Shared physics core, three array backends and optional automatic differentiation
```

| Choice | Main use | What to check |
| --- | --- | --- |
| `backend='numpy'` / `Backend.NUMPY` | First calculations, result inspection, ordinary NumPy workflows | Not an automatic-differentiation backend |
| `backend='jax'` / `Backend.JAX` | JAX differentiation, JIT and batching on supported paths | 64-bit settings and static geometry information |
| `backend='torch'` / `Backend.TORCH` | PyTorch eager differentiation | Under `torch.compile` Wavebit runs eagerly, so compiling buys it nothing; `fullgraph=True`, `torch.export` and `torch.jit.trace` are refused |

The default is `'numpy'`. Names match the libraries’ import names; `Backend` is a string enum containing the same three values, so `Backend.JAX == 'jax'`. Before 2026-10-11, integers `0`/`1`/`2` selected backends. Integers, and alternative spellings such as `'JAX'` or `'pytorch'`, are now rejected with a list of accepted names.

This does not mean every experimental auxiliary solver has identical features on all three backends. GPU availability also depends on the installed backend and operation support. Transfer and setup costs can dominate small problems, so a GPU is not automatically faster. Measure initial JIT compilation separately from repeated execution.

## Distinguish precision from execution tools

`precision='double'` (complex128, the default) or `'single'` (complex64) selects the arithmetic; pass it to `rcwa.run` or `rcwa.prepare`, or put it in a `wavebit.Execution`.

Importing wavebit does not enable JAX x64. The course's JAX examples call
`jax.config.update("jax_enable_x64", True)` **before constructing arrays**; a JAX run at
`precision='double'` without it is refused (`execution.jax_x64`). Casting a previously low-precision value to a higher dtype cannot recover lost digits. This setting controls 64-bit permission across the JAX process. For high contrast, resonances and decaying modes, verify outputs and gradients separately when choosing lower precision.

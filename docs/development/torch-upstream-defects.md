# PyTorch defects wavebit works around — what each is, where it reaches wavebit, and when the workaround can go

Written: 2026-10-12. Measured on macOS arm64 (CPU only), Python 3.12.9, NumPy 2.5.1, with
pure-torch reproducers (no wavebit, no `array_api_compat`) on torch 2.13.0, 2.14.1, 2.15.0
(the test channel's release candidate) and the 2.16.0.dev20261011 nightly.

Targets:
- `src/wavebit/_common/backend.py::linear_solve` — the solve every derivative-sensitive site goes through
- `src/wavebit/_common/backend.py::eager_under_torch_compile` — the doors that keep wavebit out of a compiled graph
- `pyproject.toml` — the torch floor (`torch>=2.13`), which decides which of these defects a user can meet
- `tests/common/test_linear_solve_derivatives.py` — `test_pytorchs_own_rule_is_still_wrong`, the canary for defect 2
- `tests/api/test_torch_compile.py` — `test_canary_inductor_still_reorders_the_eye_write`, the canary for defect 1

---

## TL;DR

Four torch defects returned wrong numbers with no error through wavebit. Each is routed around
inside wavebit, and none reaches an ordinary eager `backward()`, JAX, `hessian` or
`create_graph=True`. Two are fixed upstream in a release, and two are not fixed anywhere yet.
`torch.jit.trace` is listed separately: it is refused for a reason that no torch version will
change.

| # | Defect (torch) | 2.13.0 | 2.14.1 | 2.15.0 | 2.16 nightly | Upstream | Reaches wavebit when | Workaround | Can go when |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Inductor emits a complex identity's in-place diagonal write after the kernels that read it, when the input requires grad | wrong | right | right | right | none found; fixed by 2.14.1 | `torch.compile` (Inductor) around any solve | doors: `eager_under_torch_compile` | floor ≥ 2.14.1 **and** wavebit compiled without the doors measured exact (not measured yet) |
| 2 | `linalg.solve`'s JVP differentiated again loses a term (half the second derivative here) | wrong | wrong | right | right | [#192540](https://github.com/pytorch/pytorch/issues/192540) (fixed by [PR #192667](https://github.com/pytorch/pytorch/pull/192667)); [#196695](https://github.com/pytorch/pytorch/issues/196695) open for `left=False` | forward mode inside another level, through any solve site | `linear_solve` routes to LU | floor ≥ 2.15 (the routing for this case can then be narrowed) |
| 3 | CPU `lu_solve` reads LU at `LU.stride(-3)`, wrong when that axis has size 1 and a non-canonical stride | wrong | wrong | wrong | wrong | [#200599](https://github.com/pytorch/pytorch/issues/200599), open | indirectly, through defect 2's JVP under `vmap(jacfwd)` (2.13–2.14) | `linear_solve` never hands `lu_solve` a re-strided LU | #200599 fixed in a release the floor includes |
| 4 | `linalg.solve`'s backward drops the matrix's forward tangent when run without `create_graph` | wrong | wrong | wrong | wrong | [#200600](https://github.com/pytorch/pytorch/issues/200600), open | a `forward_ad` dual plus a backward without `create_graph` (forward-over-reverse) | `linear_solve` routes to LU | #200600 fixed in a release the floor includes |

The two canaries above fail on purpose when torch fixes their defect. The Inductor canary is
asserted only on macOS arm64, where it was measured, and skips elsewhere. On torch ≥ 2.14.1 it
therefore fails on macOS arm64. That is the signal to measure a door-free compile (row 1). It is
not a regression.

---

## 1. The Inductor identity reorder

The modal exponential builds its identity the way `array_api_compat`'s torch `eye` does: zeros
plus an in-place diagonal `fill_`. Compiled by Inductor, with a complex input that requires
grad, the write is scheduled after the kernels that read the identity, so they read zeros. The
reproducer, which needs no wavebit:

```python
q = torch.tensor([[1 + 1j, 0], [0, 4 + 4j]], dtype=torch.complex128, requires_grad=True)
def f(q):
    eye = torch.zeros(2, 2, dtype=q.dtype); eye.diagonal().fill_(1)
    return eye * torch.exp(q) + (q - eye * q)
torch.compile(f)(q)   # on 2.13.0 differs from eager f(q) by max|diff| 60.24; 0.0 on 2.14.1
```

Each of the following makes it exact on 2.13.0:

- float64 instead of complex;
- no grad;
- `torch.eye` in place of the in-place fill;
- `backend='aot_eager'`;
- `torch._inductor.config.reorder_for_peak_memory=False`.

On 2.14.1, 2.15.0 and the nightly it is exact as written. Through wavebit on 2.13.0, a 1D TE
loss read -4.43 for an eager -0.895, and its gradient -0.118 for 0.272.

The doors make wavebit run eagerly inside a compiled caller ([`torch-compile.md`](torch-compile.md)
§6). They can come out only after a compile of wavebit *without* them is measured exact on the
floor version. Compiling buys wavebit's part no speed either way (`torch-compile.md` §4). The
`torch.jit.trace` refusal at the same doors stays regardless (§5).

## 2. `linalg.solve`'s JVP, differentiated again

With `A(x)` affine and `B(x)` quadratic, the exact second derivative of `X = A⁻¹B` satisfies
`A X'' = B'' − 2 A' X'`. On 2.13.0 and 2.14.1, `jvp(jvp)`, `jacfwd(jacfwd)`, `jacrev(jacfwd)`,
`grad(jvp)`, a backward pass over `forward_ad`, and `forward_ad` over a backward pass without
`create_graph` each return half of it on the reproducer (0.125 for 0.25). `hessian`,
`jvp(grad)`, `grad(grad)` and `create_graph=True` are exact.

The cause is that `linalg_solve_jvp` solved with the LU saved from the primal, which carries no
derivative. On 2.15 it re-solves with `A` whenever the JVP can be differentiated again
(`(GradMode::is_enabled() && A.requires_grad()) || areAnyTensorSubclassLike({A, dA, dB})`).

Through wavebit on 2.13.0:

- a 1D TM grating's thickness second derivative read 1.1801616522e-04 for 1.1001006774e-04;
- a 2D raster's read 760.746214 for 812.595272 on ETM, and 725.018988 on SMM with field values.

The full list is in [`docs/guide/common/autodiff.md`](../guide/common/autodiff.md). A related reverse-over-forward
route (`torch.func.jvp`, then `backward()` to a parameter captured with `requires_grad`) has the
same before-and-after: wrong on 2.13.0 and 2.14.1, right on 2.15.0 and the nightly.

## 3. CPU `lu_solve` with a re-strided batch axis — #200599

`apply_lu_solve` (`aten/src/ATen/native/BatchLinearAlgebraKernel.cpp`) takes the distance
between consecutive LU matrices from `LU.stride(-3)` and indexes with a flattened batch index.
`LU[:, None]` (or LU factors batched by an outer `vmap` with the right side batched by an inner
one) puts a size-1 axis there whose stride is not the matrix stride, so a later lane reads
another lane's factors. On the issue's 2×2 example the second solution came back `[5, −1]` for
`[1/6, 1/3]`, on all four versions. `.contiguous()` cures it.

wavebit never called `lu_solve` itself. The defect reached it through defect 2: on 2.13–2.14,
`linalg.solve`'s JVP solves with the saved LU through this kernel, so `vmap(jacfwd)` over lanes
of a thickness read 196.4 for 0.011455 (1D TM) and NaN on conical and 2D.

`linear_solve` avoids the layout in two ways:

- outside `vmap`, its LU comes straight from `lu_factor`, so the strides are canonical;
- under any `vmap`, it solves with `lu_unpack` and two `solve_triangular` calls instead of
  `lu_solve` (`backend.py::_differentiable_solve`).

## 4. `linalg.solve`'s backward under forward mode — #200600

`linalg_solve_backward` re-solves with `A` only when grad mode is on. A backward without
`create_graph` runs with grad mode off, so it uses the saved LU and pivots, which carry no
forward tangent. Forward AD does not depend on grad mode, so a live tangent on `A` is dropped.
On the issue's 1×1 example, the tangent of `∂X/∂B` came back `None` (for −0.25) and that of
`∂X/∂A` came back 0.125 (for 0.25), on all four versions.

`lu_factor` + `lu_solve`, `inv(A) @ B`, `create_graph=True` and `torch.func.jvp(torch.func.grad)`
are exact. The JVP fix of defect 2 did not touch this branch.

In wavebit it reached the conical and 2D ETM final matching (`solve_1d.py::final_matching`) and
SMM's field state (`scattering.py::_below_reflections`, `scattering.py::_field_state`): a 2D
raster's thickness second derivative by `forward_ad` over a backward pass read 1.284 for 11.18.
The 1D TE/TM ETM cascade inverts with `linalg.inv` and was never affected.

## 5. `torch.jit.trace` — refused, and not a torch defect

Tracing records each value-dependent decision as a constant and returns an output with no
`grad_fn`. Through wavebit on develop before the refusal:

- a traced solve emitted 66 `TracerWarning`s;
- a loss mixing it with another term read wavebit's share of the gradient as 0.0, where eager
  reads 0.2721483665790;
- a solved result's `reconstruct_field` replayed one layer's formula in another.

This is what tracing is, so no torch release changes it. Every door refuses it by name, and the
refusal stays.

## What is not affected

- An eager `backward()` (the ordinary optimisation loop), any JAX transform, `hessian`,
  `create_graph=True`, and the values of a `torch.func.vmap`: none of these touches defects 1–4.
- The derivative routes that do touch them (forward mode inside another level, forward-over-reverse
  without `create_graph`, `vmap(jacfwd)`, `torch.compile`) are exact through wavebit since
  `4938026b` (defect 1) and `1d47957e` (defects 2–4), on every version measured above.

## Re-checking on a torch upgrade

1. Run `tests/common/test_linear_solve_derivatives.py` and `tests/api/test_torch_compile.py`. A failing canary
   means torch fixed its defect.
2. Re-run the upstream issues' reproducers (#200599, #200600) on the new version.
3. Only then narrow `linear_solve`'s gate, or measure a door-free compile, and raise the floor in
   `pyproject.toml` to the version that was measured. A workaround stays until the floor
   excludes every version with the defect.

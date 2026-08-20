# `torch.compile` — why it failed, and why it is not worth chasing

Written: 2026-08. Measured on macOS arm64 (CPU only) against `develop` at `8257e2c`, with
NumPy 2.5.1 / SciPy 1.18.0 / jax 0.11.0.

Targets:
- `wavebit/_core/modeler.py::Modeling.period` — where the first failure is raised
- `wavebit/_core/backend.py::eig` — the primitive the remaining failures are about
- `pyproject.toml` — the torch floor, raised from 2.7 to 2.13 as a result; see §3

---

## TL;DR

`torch.compile` failed on **every** Wavebit path with `InternalTorchDynamoError`. None of
the four causes was in Wavebit: all four are PyTorch defects, present in **torch 2.7.0** (the
version the floor used to allow) and **all fixed by torch 2.13.0**, where the failing snippet
runs unmodified and returns bit-identical values.

So there is nothing to make work. And no reason to: with everything working on 2.13.0,
compiling buys **0.92–0.99x on the forward pass and 0.85–1.02x on forward+backward** — nothing,
either way. The reason is structural, not a bug: at `fto=10`, **98% of the runtime is inside
LAPACK/BLAS calls that Inductor forwards verbatim**, and Inductor declines to generate code for
complex operators at all, which is essentially the entire remainder.

**Recommendation: do not add `torch.compile` support.** The torch floor was nonetheless raised
to 2.13 — not to enable compiling, but because this project's stated rule is that floors are the
versions the suite is verified against (`pyproject.toml`), and 2.13 is what it is now verified
against. Eager torch and `.backward()` work fine on 2.7.0; only `torch.compile` does not.

---

## 1. The failure, and why it hit every path

```python
total = lambda x: torch.sum(wavebit.rcwa(backend=2, structure=x, **cfg).solve().de_ti)
torch.compile(total)(u)      # InternalTorchDynamoError on torch 2.7
```

```
torch._dynamo.exc.InternalTorchDynamoError: AttributeError: 'str' object has no attribute 'IF_NEEDED'
from user code:
   File ".../array_api_compat/numpy/_aliases.py", line 94, in asarray
    return np.array(obj, copy=copy, dtype=dtype, **kwargs)
```

The chain, innermost first:

1. `Modeling.__init__` defaults `self._xp` to the **eager-NumPy** namespace, then runs the
   `period` setter. With `period=[300., 300.]` — a plain Python list, no arrays in sight — the
   setter takes the `xp.asarray(vals, dtype=F, device=dev)` branch on the NumPy namespace.
   This is ordinary setup bookkeeping; no tensor math is involved.
2. Because that code sits inside the compiled region, Dynamo traces it. Dynamo's `trace_numpy`
   feature substitutes `torch._numpy` for `numpy` while tracing.
3. `array_api_compat`'s NumPy `asarray` reaches for `np._CopyMode.IF_NEEDED`, a NumPy-private
   enum. `torch._numpy` does not implement it (`hasattr(torch._numpy, '_CopyMode')` is
   `False`), Dynamo models the missing attribute as a `str`, and `getattr(str, 'IF_NEEDED')`
   raises **inside Dynamo** — hence an internal error rather than the graph break Dynamo would
   normally use for something it cannot trace.

That is why every path failed identically: the failure is in object construction, upstream of
anything path-specific. Every path builds a `Modeling` with a list-valued `period`.

Minimal repro, no Wavebit (fails on 2.7.0, passes on 2.13.0):

```python
import torch, array_api_compat.numpy as anp
torch.compile(lambda t: t + float(anp.asarray([300.0, 300.0])[0]))(torch.zeros(1))
```

## 2. Three more blockers behind it

Clearing each one exposes the next. All four had to be cleared to get a single number out.

| # | Failure | Cleared by | Whose bug |
|---|---|---|---|
| 1 | `AttributeError: 'str' object has no attribute 'IF_NEEDED'` | `torch._dynamo.config.trace_numpy = False` | PyTorch (`torch._numpy`) |
| 2 | `AssertionError: expected size 98==98, stride 1==98 ... 98==1` | `torch._inductor.config.size_asserts = False` | PyTorch (Inductor) |
| 3 | `InternalTorchDynamoError: KeyError: i` | one-char source edit, see below | PyTorch (Dynamo) |
| 4 | `RuntimeError: self.stride(-1) must be 1 to view ComplexDouble as Double` | nothing found | PyTorch (Inductor/AOTAutograd) |

**#2 is about `linalg.eig`, the one primitive RCWA cannot do without.** It reduces to three
lines with no Wavebit involved:

```python
A = torch.randn(8, 8, dtype=torch.complex128)
torch.compile(lambda A: torch.linalg.eig(A)[1])(A)   # AssertionError on 2.7.0
```

Eager `torch.linalg.eig` returns eigenvectors with stride `(1, 8)` — **column-major**, LAPACK's
native layout. Inductor's meta kernel declares `(8, 1)`, and `assert_size_stride` catches the
mismatch. The eigen*values* compile fine (1-D, no layout to get wrong), and disabling
`size_asserts` yields bit-exact results — so it was a false alarm about layout, not wrong
numbers. `linalg.eigh` failed too; `linalg.svd` was fine.

**#3** is Dynamo failing to generate resume bytecode for a frame that has cell variables, when
a graph break lands in it. In `_core/solve_1d.py::final_matching` the trigger is the generator
expression in `variants = [_pack(..., *(e[i] for e in eff2)) for i in (0, 1)]` — a genexp still
creates a real frame under PEP 709, so it captures `i` and makes it a cellvar of
`final_matching`. Rewriting `*(...)` as `*[...]` removes the cellvar and the error. **That edit
was made only to get past the blocker and was reverted**; it is the only Wavebit-side change
any of this would have needed, it was behaviour-neutral (results bit-identical), and it is
unnecessary on 2.13.0.

**#4** only appears in the backward pass — the half that matters for optimization. It is the
same complex-layout problem as #2, this time viewing a non-contiguous `ComplexDouble` as
`Double` during autograd. No workaround was found on 2.7.0.

## 3. All four are fixed upstream

In a throwaway venv with torch 2.13.0, the failing snippet runs **unmodified** — no config
flags, no source edits — and returns the same value as eager.

| | torch 2.7.0 | torch 2.13.0 |
|---|---|---|
| blocker 1 (`np._CopyMode`) | fails | passes |
| blocker 2 (`linalg.eig` stride) | fails | passes |
| blocker 4 (backward complex view) | fails | passes |
| full Wavebit solve, compiled | fails | passes, `diff = 0.0` |
| compiled `.backward()` | fails | passes, `max|Δgrad| ≤ 8.5e-20` |

Blocker 3 has no standalone repro, so it is only known to be absent from the full 2.13.0 run.
Which release fixed each one was not bisected — the useful fact is that 2.7 fails and 2.13 does
not. Both halves were re-checked on `8257e2c` after the same-material short-circuit was removed
(`5c73bda`); that removal deleted one graph-break source and changed nothing else here.

**The floor moved to `torch>=2.13` because of this.** Not to enable compiling — §4 shows that
buys nothing — but because `pyproject.toml`'s own rule is that floors are the versions the suite is
verified against. The full suite was run on that combination (torch 2.13.0, jax 0.11.0, NumPy
2.5.1, SciPy 1.18.0, array-api-compat 1.15.0): green, nothing skipped, no regressions against
the frozen goldens.

Versions 2.8–2.12 are untested and therefore excluded; the floor can be relaxed later if
someone verifies one, which is cheaper than tightening it after a release. At the time of
writing `wavebit` was not yet on PyPI and `develop` was unpushed, so this excluded no
existing user.

Auditing this also turned up that the floors were not being set by one rule — `numpy>=2.3.1`
and `scipy>=1.15` were the oldest versions without a known defect, while the suite was actually
verified on 2.5.1 and 1.18.0. They were aligned to the stated rule at the same time. `tqdm` and
`optax` stay loose on purpose: neither touches a number the goldens cover.

## 4. The prize, measured

Raster path, 64×64 permittivity `structure`, one CPU, best-of-3..7 per cell, **torch 2.13.0** where everything
works:

| `fto` | forward eager | forward compiled | | fwd+bwd eager | fwd+bwd compiled | |
|---|---|---|---|---|---|---|
| 3 | 0.0126 s | 0.0137 s | **0.92x** | 0.0186 s | 0.0219 s | **0.85x** |
| 6 | 0.1810 s | 0.1905 s | **0.95x** | 0.2992 s | 0.2940 s | **1.02x** |
| 10 | 1.9800 s | 1.9996 s | **0.99x** | 3.0386 s | 3.0172 s | **1.01x** |

Values were bit-identical (`diff = 0.0`) and gradients agreed to `8.5e-20`.

For scale, `jax.jit` — the analogue that does work — bought roughly 1.9x at `fto=3` falling to
1.3x at `fto=10` when last measured -- the same collapsing trend, measured the same way.
Compiled torch does not reach that, and does not reach plain NumPy either.

## 5. Why the prize is zero, and what would change that

Not a bug, and not fixable by clearing blockers. Where the time goes (torch profiler, aten self
time, raster path):

| | `fto=3` | `fto=10` |
|---|---|---|
| BLAS / LAPACK | 83.7% | **97.9%** |
| everything else | 16.3% | 2.1% |

At `fto=10` the top four ops are `linalg_eig` (41.5%), `linalg_lu_solve` (25.1%), `mm` (17.5%)
and `linalg_lu_factor_ex` (13.8%). Inductor does not implement any of these; it emits a fallback
to the same ATen kernel eager calls. The remaining ~2% is complex-valued elementwise work, which
Inductor announces it will not compile:

```
UserWarning: Torchinductor does not support code generation for complex operators.
Performance may be worse than eager.
```

The share grew during this investigation, for a reason worth recording. An earlier profile on
`c992b72` put BLAS/LAPACK at 89% at `fto=10`, with `_fft_c2c` taking 5% there and 30% at
`fto=3`. `072710d perf(factorization): compute the raster's Fourier coefficients exactly` then
landed. It did not remove the FFT — the same six transforms still run — but it removed the
**upsampling** in front of them: the sampled path used to repeat pixels until the grid resolved
`4*fto+1` harmonics and transform *that*, where the exact coefficients are the N-point DFT of
the original grid times `sinc(k/N) * exp(-i*pi*k/N)`. Transforming the small grid instead took
the FFT from 8.4 ms to 0.05 ms at `fto=3` (0.34% of the work) and from 157.9 ms to 0.27 ms at
`fto=10` (0.01%).

Either way it was a library call Inductor forwards untouched, so this did not change the
conclusion — it only concentrated what remains into LAPACK.

The graph is also broken repeatedly by decisions Wavebit makes on purpose —
`array_namespace` dispatch chief among them — so what reaches the compiler is fragments of glue
between library calls.

This would be worth revisiting if any of the following changed:

- **A GPU.** All of the above is CPU-only, because this machine has no GPU. Kernel-launch
  overhead and memory traffic are a much larger share on GPU, and the ranking could differ.
- **Inductor learns to codegen complex operators.** That is what would put the non-LAPACK
  fraction in play — and it is worth most at low `fto`, where that fraction is largest.
- **The cost stops being dense eigendecomposition** — a different mode solver, or batching many
  small solves so per-call overhead dominates again.

Until then, `jax.jit` is the working answer for a traced solve; see "Differentiating a
simulation" in the README.

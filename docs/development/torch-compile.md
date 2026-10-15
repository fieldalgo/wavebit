# `torch.compile` — why it failed, why it is not worth chasing, and why wavebit runs eagerly under it

Written: 2026-08. Measured on macOS arm64 (CPU only) against `develop` at `8257e2c`, with
NumPy 2.5.1 / SciPy 1.18.0 / jax 0.11.0. **Updated 2026-10 (§6)**, on torch 2.13.0.

Targets:
- `src/wavebit/rcwa/_engine/engine.py::CoreSolveMixin.period` — where the first failure was raised (it was the block modeller's own `period` setter until that path was removed)
- `src/wavebit/_common/backend.py::eig` — the primitive the remaining failures are about
- `pyproject.toml` — the torch floor, raised from 2.7 to 2.13 as a result; see §3
- `src/wavebit/_common/backend.py::eager_under_torch_compile` — the doors that keep wavebit out of the
  compiled graph since 2026-10; see §6

---

## TL;DR

**Since 2026-10 a compiled caller is exact, because wavebit is never compiled (§6).** Every
public wavebit callable is a `torch.compiler.disable` door, so wavebit runs eagerly inside a
`torch.compile`d function and its values and gradients are eager torch's; only the caller's own
arithmetic on wavebit's outputs is compiled. `fullgraph=True` and `torch.export` are refused by
name, and so is `torch.jit.trace`, at every public call (a solved result's read-outs included).
Until then, default-mode `torch.compile` on torch 2.13 returned **wrong values and wrong
gradients with no error** (a 1D TE loss of -4.43 for an eager -0.895, its gradient -0.118 for
0.272): Inductor emits the in-place diagonal write of `array_api_compat`'s complex `eye` after
the kernels that read it. That defect arrived with the modal exponential (`modal_gradient.py`,
2026-09-29), after everything §1–§5 measured, so §3's and §4's "bit-identical" are history.

The 2026-08 investigation, §1–§5, stands as written below: `torch.compile` failed on **every** Wavebit path with `InternalTorchDynamoError`. None of
the four causes was in Wavebit: all four are PyTorch defects, present in **torch 2.7.0** (the
version the floor used to allow) and **all fixed by torch 2.13.0**, where the failing snippet
runs unmodified and returns bit-identical values.

So there is nothing to make work. And no reason to: with everything working on 2.13.0,
compiling buys **0.92–0.99x on the forward pass and 0.85–1.02x on forward+backward** — nothing,
either way. The reason is structural, not a bug: at `truncation_order=10`, **98% of the runtime is inside
LAPACK/BLAS calls that Inductor forwards verbatim**, and Inductor declines to generate code for
complex operators at all, which is essentially the entire remainder.

**Recommendation (2026-08): do not add `torch.compile` support.** It still holds for speed —
§6 keeps wavebit eager under a compiler, which by construction buys wavebit's part nothing. The torch floor was nonetheless raised
to 2.13 — not to enable compiling, but because this project's stated rule is that floors are the
versions the suite is verified against (`pyproject.toml`), and 2.13 is what it is now verified
against. Eager torch and `.backward()` work fine on 2.7.0; only `torch.compile` does not.

---

## 1. The failure, and why it hit every path

```python
total = lambda x: torch.sum(wavebit.rcwa(backend='torch', structure=x, **cfg).solve().T)
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
a graph break lands in it. In `src/wavebit/rcwa/_engine/solve_1d.py::final_matching` the trigger is the generator
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

Those two "passes" were measured at `8257e2c`, before `modal_gradient.py` existed (it was created
by `8ac44120`, 2026-09-29). They are no longer true of a compiled wavebit: once the modal
exponential's complex `eye` reached the graph, Inductor got values and gradients wrong (§6), and
the cure was to stop compiling wavebit.

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

| `truncation_order` | forward eager | forward compiled | | fwd+bwd eager | fwd+bwd compiled | |
|---|---|---|---|---|---|---|
| 3 | 0.0126 s | 0.0137 s | **0.92x** | 0.0186 s | 0.0219 s | **0.85x** |
| 6 | 0.1810 s | 0.1905 s | **0.95x** | 0.2992 s | 0.2940 s | **1.02x** |
| 10 | 1.9800 s | 1.9996 s | **0.99x** | 3.0386 s | 3.0172 s | **1.01x** |

Values were bit-identical (`diff = 0.0`) and gradients agreed to `8.5e-20` — at `8257e2c`, before
`modal_gradient.py` existed; not true of a compiled wavebit since (§6). The timings are history
too: wavebit now runs eagerly under `torch.compile`, so the "compiled" columns would read as eager.

For scale, `jax.jit` — the analogue that does work — bought roughly 1.9x at `truncation_order=3` falling to
1.3x at `truncation_order=10` when last measured -- the same collapsing trend, measured the same way.
Compiled torch does not reach that, and does not reach plain NumPy either.

## 5. Why the prize is zero, and what would change that

Not a bug, and not fixable by clearing blockers. Where the time goes (torch profiler, aten self
time, raster path):

| | `truncation_order=3` | `truncation_order=10` |
|---|---|---|
| BLAS / LAPACK | 83.7% | **97.9%** |
| everything else | 16.3% | 2.1% |

At `truncation_order=10` the top four ops are `linalg_eig` (41.5%), `linalg_lu_solve` (25.1%), `mm` (17.5%)
and `linalg_lu_factor_ex` (13.8%). Inductor does not implement any of these; it emits a fallback
to the same ATen kernel eager calls. The remaining ~2% is complex-valued elementwise work, which
Inductor announces it will not compile:

```
UserWarning: Torchinductor does not support code generation for complex operators.
Performance may be worse than eager.
```

The share grew during this investigation, for a reason worth recording. An earlier profile on
`c992b72` put BLAS/LAPACK at 89% at `truncation_order=10`, with `_fft_c2c` taking 5% there and 30% at
`truncation_order=3`. `072710d perf(factorization): compute the raster's Fourier coefficients exactly` then
landed. It did not remove the FFT — the same six transforms still run — but it removed the
**upsampling** in front of them: the sampled path used to repeat pixels until the grid resolved
`4*truncation_order+1` harmonics and transform *that*, where the exact coefficients are the N-point DFT of
the original grid times `sinc(k/N) * exp(-i*pi*k/N)`. Transforming the small grid instead took
the FFT from 8.4 ms to 0.05 ms at `truncation_order=3` (0.34% of the work) and from 157.9 ms to 0.27 ms at
`truncation_order=10` (0.01%).

Either way it was a library call Inductor forwards untouched, so this did not change the
conclusion — it only concentrated what remains into LAPACK.

The graph is also broken repeatedly by decisions Wavebit makes on purpose —
`array_namespace` dispatch chief among them — so what reaches the compiler is fragments of glue
between library calls.

This would be worth revisiting if any of the following changed:

- **A GPU.** All of the above is CPU-only, because this machine has no GPU. Kernel-launch
  overhead and memory traffic are a much larger share on GPU, and the ranking could differ.
- **Inductor learns to codegen complex operators.** That is what would put the non-LAPACK
  fraction in play — and it is worth most at low `truncation_order`, where that fraction is largest.
- **The cost stops being dense eigendecomposition** — a different mode solver, or batching many
  small solves so per-call overhead dominates again.

Until then, `jax.jit` is the working answer for a traced solve; see
[differentiating a solve](../migration/autodiff-v1.md) (the README's "Differentiating a
simulation" when this was written).

## 6. 2026-10: Inductor miscompiles the complex identity, so wavebit runs eagerly

Measured 2026-10-12 on torch 2.13.0, Python 3.12.9, macOS arm64 (CPU), on `develop` at
`01815aa0` and on this change.

### The defect

`array_api_compat`'s torch `eye` is a zeros tensor whose diagonal is then written in place
(`diagonal().fill_(1)`). When that tensor is complex and the input requires grad, Inductor emits
the in-place write *after* the kernels that read the tensor, so they read zeros. The modal
exponential (`src/wavebit/rcwa/_engine/modal_gradient.py`, created 2026-09-29) is built exactly so:
`eye*phases + (q - eye*values)*divided`. No wavebit is needed to see it:

```python
import torch, array_api_compat.torch as act

def f(q):
    eye = act.eye(q.shape[0], dtype=q.dtype)
    return eye * torch.exp(q) + (q - eye * q)

q = torch.randn(4, 4, dtype=torch.complex128, generator=torch.Generator().manual_seed(11))
print((torch.compile(f)(q.clone().requires_grad_()) - f(q)).abs().max())   # 1.381; eagerly 0
```

| `f` compiled with Inductor | max \|compiled − eager\| |
|---|---|
| complex128, input requires grad | **1.381** |
| complex64, input requires grad | **5.709** |
| complex128 without grad; float64 with or without grad | 0 |
| `backend='aot_eager'` or `backend='eager'` | 0 |
| `torch.eye` in place of `array_api_compat`'s `eye` | 0 |
| `torch._inductor.config.reorder_for_peak_memory = False` | 0 |

So Dynamo and AOTAutograd are exact and Inductor's peak-memory reordering moves the write;
turning that pass off cures this repro (a fact, not a recommendation: nothing here depends on
it). A search of PyTorch's issues on 2026-10-11 found none that matches.

In wavebit it was silent in value **and** gradient. One solve and a backward of a weighted
read-out (`tests/rcwa/test_forward_mode.py`'s scenes), default mode, before this change:

| scene | compiled loss | eager loss | compiled gradient | eager gradient |
|---|---|---|---|---|
| 1D TE, permittivity, ETM | **-4.434983049888** | -0.8953932051167 | **-0.1182433628003** | 0.2721483665790 |
| 2D Li raster with fields | **-5.512585863438** | -0.8677333191407 | **0.02838407891956** | 0.01971995024377 |

The same 1D scene was exact under `backend='aot_eager'` (gradient rel 6.1e-16) and
`backend='eager'` (0). A TM thickness and a theta setter on a solver built outside raised
`InternalTorchDynamoError: TypeError: Cannot interpret 'torch.complex128' as a data type`, and
`fullgraph=True` raised `Unsupported` at `cmath.sqrt`. The two wrong rows raised nothing.

### The doors

A refusal cannot be raised from inside the trace: Dynamo turns an exception raised while tracing
into a graph break and reruns the frame natively, where it is swallowed (a toy with no wavebit
returned 5.0 with no error). And the arithmetic is not wrong; the compiler is. So wavebit keeps
its frames out of the graph. `src/wavebit/_common/backend.py::eager_under_torch_compile` decorates
every class in `wavebit.__all__`, every type a public call returns and every wavebit base of
either — each public method, property accessor, `classmethod`/`staticmethod` and dunder of the
class's own namespace — and the public functions: 65 sites. Since 2026-10-13 the `wavebit.rcwa`
and `wavebit.fdtd` lifecycle functions and their declaration and result types are doors as well. A class only ever built and used
inside a door, and never handed to a caller, needs none: everything a door calls is already
kept out. A door

- calls straight through while `torch._dynamo` is not in `sys.modules`. Nothing can be compiling
  then, since `torch.compile` imports it, and an eager solve and backward never import it;
- otherwise calls through a cached `torch.compiler.disable(fn, reason=...)` wrapper, PyTorch's
  documented way to keep a function and everything it calls out of `torch.compile`
  (docs.pytorch.org/docs/2.13/generated/torch.compiler.disable.html). Building it lazily on
  the first call is the idea of PyTorch's own `torch._compile._disable_dynamo`.

Two shortcuts that look equivalent are wrong. Choosing the route by
`torch.compiler.is_compiling()` gets all five tripwire scenes of `tests/api/test_torch_compile.py`
wrong: Dynamo stops tracing a wrapper whose frame compiled to nothing and runs it natively,
with the door's own frames still compiled. Skipping the wrapper when already inside a door
lets Dynamo trace wavebit again whenever a user callable compiled with `torch.compile` is called
from inside it, because that callable sets the frame hook back: `solver.grad` with a compiled
`forward` then returned a permittivity gradient up to 0.8165 off (eager's largest entry is
0.1539), with the loss right and no error.

**What it gives.** Values within 2.6e-16 of eager and gradients within 9.5e-16 — the remainder
is the caller's own compiled weighted sum — in default mode, `mode='reduce-overhead'` and
`dynamic=True`, for a solver built inside or outside the compiled function, property setters,
`solve(...)` keywords, a result's read-outs, `ScatteringMatrix`, `Lattice`, `Material` and
`TabulatedMaterial`, `CoherentSolver` and `excitation=`. `torch.compile` around `torch.func`
routes and double backward returns the eager derivative: around `create_graph=True` twice it
reads 1.1001006774e-04 against a NumPy second difference of 1.1001006765e-04, where it raised
`InternalTorchDynamoError` before. Around `jvp(jvp)` and `vmap(jacfwd)` it is the eager
derivative too, and so exact: these two differentiate a linear solve's forward rule again or
run it under two `vmap` levels, which `torch.linalg.solve`'s own rule gets wrong (torch 2.13)
and `backend.linear_solve`, which every solve goes through, gets right (the autodiff guide's
[Forward mode](../migration/autodiff-v1.md#forward-mode)). `tests/api/test_torch_compile.py` checks all
three against NumPy.

**What is refused, by name.**

- `fullgraph=True`, and a strict `torch.export`, raise `torch._dynamo.exc.Unsupported: Skip
  calling torch.compiler.disable()d function ... (reason: wavebit runs eagerly under
  torch.compile: Inductor (torch 2.13) silently miscompiles its complex arithmetic -- see
  docs/development/torch-compile.md)`. A door carries the reason only once it has run with Dynamo
  loaded: the first `fullgraph` compile or strict export in a process names
  `torch.compiler.disable` without the reason, even after eager wavebit calls (those take the
  fast path and cache nothing) -- still loud, `Unsupported`.
- A non-strict `torch.export` runs the door natively, where `disable` does not apply, so the
  door refuses it with `NotImplementedError` ("wavebit cannot be exported: ...") while
  `torch.compiler.is_exporting()` is true. Without that, the exported module returned eager
  values and no `grad_fn`.
- `torch.jit.trace` raises `NotImplementedError` ("wavebit cannot be traced by
  torch.jit.trace: ...") at every public wavebit call. Dynamo is not involved and the values
  are concrete, so `disable` does not catch it; each door refuses while the public
  `torch.jit.is_tracing()` is true. Before that it was silent wherever it reached:
  - a solve traced with 66 `TracerWarning`s and returned an output with no `grad_fn`, so a
    loss mixing the traced solve with another term lost wavebit's share of its gradient: 0.0
    at p = 0 where eager reads 0.2721483665790, and 0.1 at p = 0.05 (that is, 2p alone) where
    eager reads 0.3733920308334;
  - a solved two-layer grating's `reconstruct_field`, traced at a height in one layer and
    replayed in the other, read d/dz -0.004063805947 where eager and a central difference read
    0.003442581948, and replayed outside the stack it answered 0.9016 where eager refuses;
  - `Lattice.hexagonal(a)` from a traced length became a NumPy constant, so
    `meshgrid_xy(lattice=...)` replayed the example's cell: +2.922e5 for an eager +3.393e5,
    with no gradient for eager's 1574.35.

  A first version refused only where a solve reads a value at a decision
  (`backend._concrete`); the last two never reach it, so the check is in the door, which every
  public call passes.

**Not covered.**

- `make_fx` on real tensors returns values with no `grad_fn`; it has no public flag to refuse on.
- Compiled autograd (`torch._dynamo.config.compiled_autograd = True`) compiles wavebit's custom
  `autograd.Function` backward rules, which are not doors. It was measured within 7.2e-13 of
  eager (2026-10-11); that is not a contract.
- A `forward_ad` tangent through any compiled function comes back `None`. It does without
  wavebit too, so it is torch's.

### What it costs, and what it relies on

About 90 ns per door call while Dynamo is not loaded and about 490 ns once it is, at 374–548
door calls per forward (counted 2026-10-11, with 67 sites). Minimum of three alternating rounds, one
thread, `develop` → this change: a 1D TE (3,0) value-only solve 2.14 → 2.17 ms without Dynamo
and 2.17 → 2.40 ms with it; a 2D (2,2) solve with fields 7.66 → 8.04 ms and 7.64 → 7.96 ms. A
forward-mode dual, `torch.func.grad` and every `torch.optim` step import Dynamo, so an
optimisation loop pays the second figure after its first step: a fixed ~0.2–0.3 ms per forward,
under 0.1% at orders (15,15) and above, where a solve takes hundreds of milliseconds. The
`torch.jit.trace` check adds about 100 ns to every door call on either route: a door around a
no-op went from 61 to 158 ns without Dynamo and from 346 to 449 ns with it (2026-10-12, the mean
of two runs, each the minimum of seven million-call repeats), at 373, 390 and 392 door calls
per forward on the 1D TE, conical and 2D scenes — about 0.04 ms per forward.

Two implementation facts are relied on, and both would fail silently if they changed:

1. `torch.compile` and a non-strict `torch.export` import a module named `torch._dynamo` before
   compiling anything — the fast path's premise.
2. `torch.compiler.disable` is recursive (its documented default `recursive=True`), so
   everything a door calls stays eager.

`tests/api/test_torch_compile.py` catches either (its tripwire compares a compiled solve with eager
autograd at 1e-12), so run it on every torch upgrade. Its canary,
`test_canary_inductor_still_reorders_the_eye_write`, fails once a torch compiles the five-line
`f` above correctly. Then update this section; the doors stay, because they keep wavebit out of
every future Inductor defect at the cost above.

Everything in this section was measured on macOS arm64 (CPU) with torch 2.13.0 only. CI runs on
Linux x86-64 with the CPU index's latest torch, where nobody has measured the reorder: Inductor
writes its own C++ per platform. So the canary asserts only on macOS arm64, and elsewhere a gap
that does not reproduce skips with the gap and the torch version it read. Record what such a
platform does here.

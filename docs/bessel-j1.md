# The Bessel $J_1$ every backend uses, and why its two constants are what they are — measurements

Written: 2026-08-11. Measured against `mpmath` at 40 digits, CPU, float64 unless stated.

Targets:
- `wavebit/_core/backend.py::j1` — the dispatch
- `_J1_THR`, `_J1_THR_F32`, `_J1_ASYM_TERMS` — the constants this document exists to justify
- `wavebit/_core/backend.py::jinc` — `2 J_1(z)/z`, the only caller that matters: it is the
  Fourier coefficient of a circle or an ellipse

---

## TL;DR

Two backends had two different inaccurate implementations, and neither was visible from inside
because they had only ever been compared to each other.

| | before | after |
|---|---:|---:|
| NumPy, `scipy.special.j1` | 1.7e-16 | 2.2e-16 *(unchanged)* |
| JAX | 7.2e-11 | **7.8e-13** |
| PyTorch | 4.1e-07 | **1.1e-12** |

Both traced backends now share one series/asymptotic hybrid. The effect is on curved shapes
only, because a rectangle needs no `jinc`: PyTorch's disagreement with NumPy on a
rectangle-plus-circle layer went **5.5e-08 → 4.0e-13**.

## Why an external reference was necessary

Every check that existed compared backends to each other, or to NumPy. Nothing established that
NumPy was right, so the comparison was circular — and it stayed circular right up until `mpmath`
was brought in. That is the whole reason two defects survived: agreement between implementations
is sensitive to a difference and blind to a fault they share.

`scipy.special.j1` turns out to be reference-grade (2.2e-16, essentially double precision), so
using NumPy as the reference elsewhere is legitimate. That is now a measurement rather than an
assumption about the library's reputation.

## The two defects

**JAX: the crossover picked the worse branch on both sides of itself.** It was 12.0. At
`x = 12` the series gives 2.5e-13 and the asymptotic expansion 7.4e-11 — and the code took the
asymptotic above 12. The measured worst case landed at `z ≈ 12`, exactly there.

**PyTorch: `torch.special.bessel_j1`,** described in this repository as "accurate". It is the
least accurate of the three by four orders. It also has no autograd backward, which is why it
was wrapped in a custom `autograd.Function` supplying `J_1' = J_0 - J_1/x`. Sharing the hybrid
removed that wrapper — 32 lines — because the hybrid is plain arithmetic and differentiates
natively.

## Why the fix is a compromise and not "more terms"

The two branches fail in different ways, which is the whole reason a crossover exists:

- **The series is limited by cancellation, not truncation.** Its largest intermediate term
  dwarfs the result once `x` grows, so past `x ≈ 12` it loses digits no term count can restore.
  Measured: 30 terms and 80 terms give the *same* 4.8e-13 at `x ≤ 12`.
- **The asymptotic expansion is divergent.** Past its optimal order the error grows again. At
  `x ≥ 12.5` it improves out to 12 terms and worsens after.

Scanning both constants against `mpmath`, 6000 points on `(0, 95]`, taking the max error of the
*combined* function:

| crossover | 10 terms | 12 terms | 14 terms |
|---|---:|---:|---:|
| 11.0 | 2.34e-12 | 1.40e-12 | 1.65e-12 |
| 12.0 | 1.58e-12 | 8.82e-13 | 9.77e-13 |
| **12.5** | 7.61e-13 | **6.99e-13** | 6.99e-13 |
| 13.0 | 1.13e-12 | 1.13e-12 | 1.13e-12 |
| 14.0 | 3.85e-12 | 3.85e-12 | 3.85e-12 |

Moving either constant away from `(12.5, 12)` makes the worst case worse.

## The crossover depends on the precision

Sharing the hybrid with PyTorch fixed double precision and **regressed single**, which the first
version of the change did not check.

| | float64 | float32 |
|---|---:|---:|
| `torch.special.bessel_j1` | 4.1e-07 | 4.7e-07 |
| hybrid, one crossover for both | 1.1e-12 | **2.7e-04** |
| hybrid, a crossover per precision | 5.6e-13 | **2.1e-06** |

The cause follows from the paragraph above: cancellation costs a fixed number of *digits*, and
that is a far larger share of single precision's budget than of double's. So the optimum is a
function of the dtype, and one hard-coded value was the defect. Fitted separately: **12.5 in
double, 7.0 in single.**

| float32 crossover | 5 | 6 | **7** | 8 | 10 | 12.5 |
|---|---:|---:|---:|---:|---:|---:|
| max abs error | 8.2e-05 | 1.3e-05 | **2.1e-06** | 8.2e-06 | 3.1e-05 | 4.5e-04 |

Single precision is still ~4x short of what the native routine gave it, and ~17x short of
float32 epsilon (1.2e-07). That is accepted rather than hidden — 32-bit is documented as
hazardous — but 2.1e-06 and 2.7e-04 are different kinds of answer.

## No `abs(x)`

The shared branch does **not** take `abs(x)` first, and the omission is deliberate. On a domain
of `x >= 0` the guard looks free. It is not:

- it is *even* where `J_1` is odd, so it would be wrong for negative argument rather than merely
  redundant;
- `d|x|/dx` is `0` at `0` in both frameworks, so it silently zeroes `dJ_1/dx` at the origin.

The series branch is odd by construction and gives the correct `1/2` there. Pinned by
`test_j1_derivative_is_right_including_at_zero`.

## Verified across the argument range

float64, both traced backends, against `mpmath`:

| `z` | max abs error |
|---|---:|
| 1e-12 … 1e-3 | 1.1e-19 |
| 1e-3 … 1 | 1.1e-16 |
| 1 … 12.5 *(series)* | 1.0e-12 |
| 12.5 … 95 *(asymptotic)* | 3.9e-13 |
| 95 … 500 | 9.9e-16 |
| 500 … 5000 | 3.6e-15 |

Large argument is the easy end: the asymptotic expansion improves as `z` grows.

## Verified in the solve

58 combinations — circle, 3:1 ellipse, rotated ellipse, a 0.04-diameter circle, a
0.85-diameter circle, rectangle + circle, lossy circle × normal TE, oblique TM, conical ×
`truncation_order` 3, 8, 15 — comparing every backend's `R` and `T` against NumPy:

| | worst over all 58 |
|---|---:|
| cross-backend difference | **4.7e-12** |
| energy, `\|R+T-1\|`, lossless cases | **3.8e-12** |

Most rows are below 1e-14; the larger values occur only at `truncation_order = 15` and are accumulated
round-off in bigger matrices, not a property of `J_1`. The curved-shape penalty on PyTorch does
not reappear in any combination.

One combination was **refused** rather than solved: a 0.85-diameter circle in a 0.8 period runs
into its own image, and the layer-combination check said so and asked which rule was meant. The
case was ill-posed; the guard worked.

## What it cost, and the honest ledger

The hybrid is **29-170x slower than `torch.special.bessel_j1`** as a function. It evaluates
30 series terms and 12 expansion terms as separate tensor operations, and `where` computes
*both* branches everywhere, against one fused kernel:

| points | hybrid | builtin | ratio |
|---:|---:|---:|---:|
| 169 | 210 us | 1.2 us | 170x |
| 961 | 404 us | 6.3 us | 64x |
| 3721 | 1102 us | 26 us | 43x |
| 40401 | 10.7 ms | 373 us | 29x |

It does not matter, and that is measured rather than assumed. `jinc` is called once per layer
to build the permittivity coefficients, on an array of `(4*truncation_order+1)^2` points, against a solve
dominated by a dense complex eigendecomposition:

| `truncation_order` | total solve | J1 | share |
|---:|---:|---:|---:|
| 5 | 88.5 ms | 0.26 ms | **0.30%** |
| 9 | 1184 ms | 0.51 ms | **0.04%** |
| 13 | 8357 ms | 0.93 ms | **0.01%** |

So the ledger:

| | hybrid | `torch.special.bessel_j1` |
|---|---|---|
| float64 accuracy | **1.1e-12** | 4.1e-07 |
| float32 accuracy | 2.1e-06 | **4.7e-07** |
| autograd | native, plain arithmetic | **none** — needed a custom `Function` |
| speed, as a function | 29-170x slower | faster |
| speed, in a solve | 0.01-0.3% | indistinguishable |
| maintenance | ours | PyTorch's |

**Single precision is the one place the builtin is still better**, by 4x. Since neither wins at
both precisions, the choice is exposed rather than decided for you.

## The switch

```python
import wavebit
wavebit.set_j1_implementation('hybrid')     # returns the previous setting
wavebit.get_j1_implementation()
```

| setting | torch float64 | torch float32 | jax |
|---|---:|---:|---|
| **`'auto'`** *(default)* | **9.0e-13** | **4.6e-07** | hybrid |
| `'hybrid'` | 9.0e-13 | 3.6e-06 | hybrid |
| `'native'` | 4.1e-07 | 4.6e-07 | hybrid |

- **`'auto'`** takes the accurate one for the precision in hand — the hybrid at 64-bit, the
  framework's own below it. It is not a tie-break: neither implementation wins at both, so the
  default does not pretend one does.
- **`'hybrid'`** always, when you want the traced backends comparable regardless of dtype, or
  one fewer variable while profiling something else.
- **`'native'`** `torch.special.bessel_j1`, wrapped for the derivative it does not ship, in reverse
  and forward mode (`backward`, `jvp`) and under `torch.func`.
  Faster as a function; that is 0.01-0.3% of a solve, so speed is rarely the reason.

**JAX ignores the setting**, in every mode, and `'native'` deliberately does not raise for it:
`bessel_jn` returns NaN for small arguments, so there is nothing to switch to, and a global
switch should not break one backend because another lacks an option. Pinned by
`test_j1_switch_leaves_jax_alone`.

The setting is module-global because `j1` is reached from inside the transform with no keyword
channel to thread. That makes it process-wide state: set it once at start-up, not per call,
and be aware of it if two things share an interpreter.

## Tests

- `test_j1_matches_scipy_on_every_backend` — against SciPy, not against the other backend. The
  two traced backends now share an implementation, so comparing them to each other would pass
  vacuously.
- `test_j1_single_precision_uses_its_own_crossover` — the regression lived entirely in float32
  and every other check runs in double.
- `test_j1_derivative_is_right_including_at_zero` — including the origin, for the `abs` reason
  above.

---

See also: [`backend-choice.md`](backend-choice.md) for which backend to use, which is now a
performance question rather than an accuracy one on every path; and
[`parametric-modeling.md`](parametric-modeling.md) for the closed-form shape transforms that call
`jinc`.

### References

The two expansions are textbook, not novel here.

- M. Abramowitz, I. A. Stegun, *Handbook of Mathematical Functions*, §9.1.10 (ascending series
  for $J_\nu$) and §9.2.5–9.2.10 (Hankel asymptotic expansion, and the $P$/$Q$ coefficient
  recurrence used in `_J1_A`). National Bureau of Standards, 1964.
- The reference values are `mpmath`'s `besselj`, at 40 decimal digits.

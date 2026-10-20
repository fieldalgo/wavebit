# Branch cuts — which square root, and the two ways it went wrong

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


Written: 2026-08-11. Measured on CPU (Apple Accelerate), NumPy / JAX / PyTorch, float64 and
float32.

**Half-space rule changed (2026-10-02).** `boundary_kz` now takes Maystre's cut, the root with
`Re(kz) + Im(kz) > 0`. §8 gives the reasons and the measurements, and it supersedes two claims
below. First, failure two (§4) is not a limit of single precision for the half-space root. The
old rule's cut lay along the real evanescent axis, and complex64 gave `kx` a genuine imaginary
part of ~1e-7. Under the new rule the float32 SMM solve is within 3.4e-6 of float64 at the point
that was 4.5 off. What remained at single precision was a layer-label effect on the ETM
algorithm, fixed the same day: ETM's layers now take the half-space's root too (`etm_mode_q`,
[`connecting-algorithms.md`](connecting-algorithms.md) §10), and the same point is 1.9e-6 off.
Second, the trade in §4's "And offers a scoped override" belonged to the old half-space rule.
With neither rule reading a cutoff any more, `branch_tolerance` overrode nothing that moved an
answer, and it was removed.

**Double negatives take the mirror of that cut (2026-10-03).** A half-space with
`Re(eps) < 0` and `Re(mu) < 0` takes the root with `Im(kz) - Re(kz) > 0`, the limiting absorption
of its own loss, which lowers `kz**2` where an ordinary medium's raises it. Maystre's cut, read
off `eps*mu` alone, took the growing root there (glass onto `eps = -2+0.1i`, `mu = -1+0.1i`:
`R = 928.78` against `0.0010767`). §9 states the rule; the whole subject -- the signed index, what
`theta` means in such a medium, the layers, the evidence -- is
[`negative-index-media.md`](negative-index-media.md).

Targets:
- `src/wavebit/_common/conventions.py` — `mode_q`, `etm_mode_q`, `boundary_kz`, `incident_kz`,
  `_check_resolvable`
- `src/wavebit/rcwa/_engine/solve_1d.py` — `solve_1d` stage 3, `final_matching`
- `tests/references/test_reticolo_conical_sweep.py` — the RETICOLO oracle
- `tests/rcwa/test_fp_warnings.py` — the three tiers of "this result is not trustworthy"

---

## TL;DR

Every `sqrt` of a complex number in this solver has **two answers**, and picking one is a
physical decision, not an arithmetic one. Two separate things went wrong there — one a bug,
one a limit — and neither was visible at a real incidence angle:

| | Symptom | Cause | Fixed |
| --- | --- | --- | --- |
| **Respelled root** | complex `theta` returned `R = -delta`, `T = 0` — a grating that scatters nothing | the incident wave's `kz` was written two ways that agree only for real angles | `incident_kz` |
| **Unresolvable branch** | complex64 returns \|R\| ~ 4 on all three backends, each wrong differently | at that precision `Re(v)` *is* round-off, so the branch is taken from the sign of noise | it warned, and offered a scoped override. **Both parts were fixed on 2026-10-02**: the half-space by Maystre's cut (§8), ETM's layer labels by `etm_mode_q`; the override was removed |

The first was invisible because **a real incidence angle makes it an identity**; it is fixed and
double precision is now exact to 1e-14 everywhere measured. The second was taken, until
2026-10-02, for a limit that cannot be fixed — single precision does not carry the information
a decision on `Re(v)` alone needs — so the solver detected it and said so. It turned out to be a
property of the two rules rather than of single precision: decisions that read `Re(kz) + Im(kz)`
and `Re(q) - Im(q)` are answerable at complex64 (§8). The oracle that found both is a 2024 sweep
over complex angles.

---

## 1. The one idea

A wave's out-of-plane wavevector comes from

```
kz = sqrt(n^2 - k_par^2)
```

and `sqrt` on the complex plane has two roots, `+v` and `-v`. They are *both* mathematically
valid. Which one is physical depends on what the wave is doing:

```
                    Im(kz)
                      ^
            +i*s  *   |                  Im > 0 : decays upward   -> a REFLECTED order
                      |
      ----------------+----------------> Re(kz)
              Re > 0  |  ...carries energy downward -> a TRANSMITTED / incident order
                      |
            -i*s  *   |                  Im < 0 : grows upward
```

So the rule is *ordered*: prefer `Re(kz) > 0` (energy going down); only when the root is
**purely imaginary** — no real part at all — does the sign of `Im` decide. That ordering is
what `boundary_kz` implements, and `mode_q` implements its `q = -i*kz` counterpart, which is
why their tie-breaks come out opposite.

"Down" and "up" here are the propagation kernels' words, as in `src/wavebit/_common/conventions.py`: `z` runs
from the kernel's entry face into the stack, the way its incident wave travels. That was the
public frame when this page was written. Since 2026-09-21 the public `+z` points up from `z = 0`
at the bottom interface, and the kernels solve public top incidence as their z-mirrored scene,
so "down" below is public `-z` under top incidence and public `+z` under bottom incidence
(see [`conventions.md`](../../reference/common/conventions.md#geometry-and-propagation)).

> **Scope of that counterpart claim** (added 2026-08-12, after the time-convention audit
> measured it): `q = -i*kz` is an identity **only for a real in-plane wavevector**. With a
> complex `k_par` (an absorbing superstrate at oblique incidence) the two functions return
> *unrelated* roots — measured, `mode_q` hands back `+i*kz`, the other one — and that is
> correct rather than a defect: a half-space's root is a boundary condition with one right
> answer, while a layer's `±q` pair enters the stack algebra in full, so its label is free
> and `mode_q` spends the freedom on keeping `exp(-k0*q*d)` bounded (flip any subset of a
> layer's roots and the solve moves by ~1e-13). The "Re-first picks a growing wave in the
> substrate" worry resolves the same way: for a passive medium fed by a lossy superstrate,
> `k_r . k_i = 0` — the wave is constant along its own ray, and the growth in `z` is the
> transverse bookkeeping of an `x`-decay the incident field shares, so boundedness is not
> an available criterion there; continuation from normal incidence is, and `boundary_kz`
> now carries its closed form (`_off_principal_branch`). Derivations and measurements:
> `docs/validation/rcwa/time-convention-audit.md` §2-§5; pinned in `tests/rcwa/test_branch_selection.py`
> (`test_the_two_selectors_disagree_at_complex_k_par_by_design` is the exact case above).
> Everything below this box concerns the regimes where the identity does hold, and stands
> unchanged.
>
> **Superseded for the half-space (2026-10-02):** `_off_principal_branch` is gone, and
> `boundary_kz` takes Maystre's cut (§8). The layer half of this box stands for SMM only. ETM
> inverts a transmission into the medium below each layer, so its labels are not free: a
> mislabelled mode crossing an unreflecting interface lost 12 digits, and ETM now takes the
> half-space's root, capped where it would grow (`etm_mode_q`,
> [`connecting-algorithms.md`](connecting-algorithms.md) §10). Since 2026-09 a real angle in an
> absorbing superstrate fixes a real `k_par` ([`conventions.md`](../../reference/common/conventions.md#real-angle-incidence)),
> so only a complex `theta` reaches this regime.

Everything below is a consequence of two questions that rule leaves open:

1. Which root does the **incident** wave take? It is the one order travelling *downward* into
   the stack, so it is not a "reflected order" and the rule above does not name it.
2. What counts as **"no real part at all"**? That is a numerical question with a
   dtype-dependent answer.

---

## 2. Where the angle goes complex

The 2024 oracle sweeps `(kx, ky)` over `[-2, 2]^2` and sets

```python
theta = arcsin(sqrt(kx**2 + ky**2 + 1e-4))
```

Wherever `kx^2 + ky^2 > 1` that is `arcsin` of a real number greater than one, which is
**complex**. Physically it is evanescent illumination — light arriving as a decaying field
rather than a propagating beam, the way a prism's frustrated total internal reflection feeds a
nearby surface.

```
   ky
  +2 +-------------------------------------+
     | x x x x x x x x x x x x x x x x x x |   x = theta COMPLEX (evanescent)
     | x x x x x . . . . . . . x x x x x x |       372 of 441 points
     | x x x . .               . . x x x x |
     | x x .        _____        . . x x x |
   0 | x x .       /  o  \         . x x x |   . = theta REAL (propagating)
     | x x .       \_____/         . x x x |       69 of 441 points
     | x x x . .               . . x x x x |
     | x x x x x . . . . . . . x x x x x x |
     | x x x x x x x x x x x x x x x x x x |
  -2 +-------------------------------------+
    -2                 0                  +2  kx
                  light line: kx^2 + ky^2 = 1
```

Two facts about that outer region matter for everything that follows:

* It is **84% of the grid**. The 2024 study plotted it as ordinary data.
* `arcsin` of a real `> 1` always returns `pi/2 + i*arccosh(...)`, so every one of those 372
  points sits at `Re(theta) == pi/2` **exactly**. One special value, not a spread — which is
  why one of the two bugs produced a suspiciously tidy answer there.

---

## 3. Failure one — the same number, spelled twice

The boundary match solves, at stage 3 of `solve_1d`:

```python
inc = -1j * n_top * cos_t * delta                        # the incident field
T1  = inv(G - 1j*Kz_top @ F) @ (-1j * Kz_top @ delta + inc)
```

`Kz_top` comes from `boundary_kz`. `inc` recomputes the same physical quantity as
`n_top * cos(theta)`. Watch what the right-hand side does at the incident order:

```
  RHS  =  -i * kz_top[c] * delta   +   -i * n_top*cos(theta) * delta

  real theta      kz_top[c] = n_top*cos(theta) = +kz
                  RHS = -i*kz + -i*kz = -2i*kz            the two terms ADD

  complex theta   cos(pi/2 + i*b) = -i*sinh(b)            negative-imaginary
                  boundary_kz forces Im > 0  ->  +i*sinh(b)
                  RHS = -i*(+i*s) + -i*(-i*s) = (+s) + (-s) = 0     they CANCEL
```

The right-hand side vanishes identically at the one order that carries the illumination, so
`T1 = 0`, `T = 0`, and `R = F @ 0 - delta = -delta`. The solver reports total reflection with
no diffraction whatsoever — and because that is a smooth constant, it **plots as a clean flat
region** rather than as noise:

```
   wavebit BEFORE                wavebit AFTER
   +---------------+               +---------------+
   |###############|               |  ..:::===:::..|
   |####,-----,####|               |..::==     ==::|
   |###/       \###|               |.::=  (real)  =|
   |###\       /###|               |..::==     ==::|
   |####'-----'####|               |  ..:::===:::..|
   +---------------+               +---------------+
    # = R is exactly -1             the pattern continues across the light line
```

**The fix is to stop spelling it twice.** `incident_kz(kz_top, index)` returns `kz_top[index]`
— read from the vector the other side of the match already uses:

```python
kz_inc = incident_kz(kz_top, n_orders_u // 2)
inc = -1j * kz_inc * delta
```

For a real angle this is an identity and nothing moves; the 69 propagating points went
`1.97e-14 -> 1.95e-14` against RETICOLO. Three call sites needed it: `solve_1d`'s TE and TM
branches, and **both** `final_B` builds in `final_matching` — the second makes the pure-TE/TM
incidence variants, and missing it leaves conical broken while 1D looks fixed. `solve_2d`
reuses `final_matching` and came along.

### Why not fix `boundary_kz` instead

Two other ways to make the two sides agree were tried and are wrong:

| Attempt | Result |
| --- | --- |
| Use meent's plain `(eps - k_par_sq)**0.5`, no branch enforcement | \|R\| up to **6.9** |
| Force `kz_top[incident]` to match `cos(theta)` | \|R\| = **4.3** |

The tell for the second was that the answer came out as `s * conj(analytic)` for a real scale
`s` — a *flipped branch*, not a scale error. `boundary_kz` was right all along.

### The independent witness

`smm_1d` and `_smm_sp` never respelled anything: they build the incident field from a
`boundary_kz`-derived `V_top`. **SMM was already correct at complex angles**, and after the fix
TMM lands on SMM's answer. Two independent formulations agreeing is stronger evidence than one
matching a reference.

| | real theta | `0.5+0.3j` | `pi/2+1.3j` |
| --- | --- | --- | --- |
| TMM 1D | 9.3e-17 | 4.5e-16 | 1.2e-16 |
| TMM conical | 3.9e-16 | 2.8e-16 | 2.0e-16 |
| SMM 1D | 5.5e-16 | 2.3e-16 | 3.2e-16 |
| SMM conical | 1.0e-15 | 5.6e-16 | 2.0e-16 |

(vs the analytic Airy reference in `tests/rcwa/test_lossy_media.py`, uniform slab. TMM here is
the enhanced recursion, named `'ETM'` since 2026-08-11; `'TMM'` now names only the plain,
unstabilised transfer of [`connecting-algorithms.md`](connecting-algorithms.md).)

---

## 4. Failure two — a question single precision cannot answer

Both branch rules asked the same thing, `is Re(v) zero?`, and answered it against a fixed
number. Since 2026-10-02 only `mode_q` does, for SMM's layers, which do not depend on the
answer; the half-space rule reads the sign of `Re(v) + Im(v)` and ETM's layers that of
`Re(q) - Im(q)`, which no real `k_par` brings near zero (§8):

```python
_FLAT = 1e-12
flip = xp.where(xp.abs(xp.real(v)) > _FLAT, xp.real(v) < 0, xp.imag(v) < 0)
```

For a root that is *mathematically* pure-imaginary, `Re(v)` is round-off — and round-off has a
size set entirely by the working precision:

```
   |Re(v)| for a root that should be exactly imaginary     measured: ~1e-16 (f64), ~1e-7 (f32)

   1e-20      1e-16          1e-12              1e-7               1e-3
     |----------|--------------|------------------|------------------|
             eps(f64)        _FLAT             eps(f32)
                            (fixed)

   float64:  noise sits BELOW _FLAT   -> the Im tie-break runs        correct
   float32:  noise sits ABOVE _FLAT   -> the tie-break is SKIPPED
                                         and `Re(v) < 0` reads the sign of noise
```

Measured at one evanescent point, `truncation_order=40` (162 orders):

| | `boundary_kz` decided on round-off | `mode_q` |
| --- | --- | --- |
| complex128 | 0 / 162 | 0 / 162 |
| **complex64** | **140 / 162** | **28 / 162** |

All three backends fail, and **each is wrong differently** — precision loss would put them near
each other:

| | float64 (all three identical) | numpy f32 | jax f32 | torch f32 |
| --- | --- | --- | --- | --- |
| propagating | `-0.58376357-0.12888271j` | 1.1e-05 | 3.0e-04 | 3.4e-05 |
| evanescent | `-0.46138379+0.48956102j` | **3.94** | **2.96** | **3.82** |

### There is no threshold that fixes it

The obvious repair is to scale the cutoff to the dtype. It does not work, and the measurement
is worth keeping so nobody re-derives it. Nine physically meaningful cases at `truncation_order=30`, float32
against float64 as ground truth, sweeping the cutoff:

| cutoff | lossless, complex angle | weakly absorbing superstrate | worst of nine |
| --- | --- | --- | --- |
| `1e-12` (shipped) | **4.51** | 1.9e-04 | 7.34 |
| `1e-6` | 1.3e-06 | **7.4e-01** | — |
| `1e-5` | 3.4e-07 | **1.75** | 1.75 |

Raising it fixes the complex angle and breaks the weak absorber; lowering it does the reverse.
They cannot both be served because **the genuine `Re(kz)` of a superstrate with `Im(n) = 1e-5`
is smaller than the round-off of a lossless layer at a complex angle** — signal and noise
occupy the same decade, and no cutoff separates overlapping distributions.

Widening the decision to complex128 does not help either: promoting a value that was *computed*
in complex64 cannot recover information the rounding already discarded. Measured — identical
numbers to the fixed cutoff, to every digit.

### So it reports instead

`_check_resolvable` warns (`UnresolvedBranchWarning`) when a decision is taken from a real part
inside its own dtype's round-off. It is a check on the precision, not on the physics:

| | complex128 | complex64 |
| --- | --- | --- |
| real angle, lossless | silent | warns (12 / 244) |
| complex angle | silent | warns (128 / 244) |
| weakly absorbing superstrate | silent | warns (103 / 244) |

Zero false positives at double precision across all three regimes.

**Unlike the other diagnostics this one is not NumPy-only**, and that matters: the backends it
would have skipped return the worst answers. At the same complex64 point NumPy gives `|R| = 1.80`
and warns; JAX gives 4.12 and PyTorch 5.68, silently — on the paths people take for gradients
and GPUs. What blocks the check is an *abstract* value, not a namespace, so the reduction runs
in the caller's own namespace and only the scalar is read; when that read raises, the value was
a tracer and there was nothing to inspect. `jax.jit(jax.grad(...))` is pinned in
`test_the_check_does_not_break_tracing`.

Cost, measured in isolation against the `eig` of the layer it belongs to (truncation_order=60):

| | one check | `eig(2N)` | share |
| --- | --- | --- | --- |
| numpy | 3.5 us | 57.3 ms | **0.006%** |
| torch | 13.8 us | 69.7 ms | **0.020%** |
| jax | 74.8 us | 44.9 ms | **0.166%** |

JAX is the expensive one because reading the scalar forces a device sync; it is still under a
fifth of a percent of the dominant operation. Whole-solve wall-clock could not resolve it above
run-to-run noise on this machine — JAX's async dispatch and per-call retracing swamp it — which
is why the isolated measurement is the one quoted.

**Under `jit` the check cannot run at all**, and silence there would mean two different things.
So a traced *single-precision* solve reports that it went unchecked; a traced double-precision
one stays quiet. The dtype is known at trace time even when the values are not, which is the
only thing there is to say — and it is worth saying, because tracing and single precision are
both reached for speed and their combination is where a wrong branch is least likely to be
caught:

| | eager | under `jit` |
| --- | --- | --- |
| complex128 | silent, \|R\| = 0.672 | silent, \|R\| = 0.672 |
| complex64 | **warns** ("a branch was decided from…"), \|R\| = 3.974 | **warns** ("this solve is traced…"), \|R\| = 3.929 |

Note that the `integrated` path is not traceable at all — `factorization.py::_as_bools`
calls `bool()` on a tracer — so a jit'd solve is on the sampled path regardless. That predates this
work.

### The x64 ordering trap, and why it now warns

This subsection records the historical import-order failure and the fix to the factory API.
Before that fix, the JAX backend enabled `jax_enable_x64` only when first *used*, so an array
built after package import but before the first solve was silently complex64. The snippet
below uses the **pre-fix historical API**; its `wavebit.rcwa(...)` factory call and import-time
behavior are not instructions for the current public modules:

```python
import jax.numpy as jnp, wavebit
theta = jnp.asarray(0.4 + 0.3j)                               # complex64 — x64 is not on yet
wavebit.rcwa(backend='jax', theta=theta, complex_dtype=0, ...)   # ...but double was asked for
```

The setter's `jnp.array(theta, dtype=complex128)` then widens it without recovering anything,
and the solve runs with ~8 significant digits missing from its incidence angle. Measured
`|R| = 3.94` against a true 0.67, with every dtype in sight reading complex128 — **this was
diagnosed as a solver bug before it was diagnosed as an import ordering artifact**, which is
the whole argument for reporting it.

Two things address it, because neither is enough alone.

**The compatibility API lifts the cap earlier.** Today, an explicit
`from wavebit.compat import legacy as wavebit` enables x64 when JAX is *already* loaded
(`_enable_jax_x64_if_already_imported`). A plain `import wavebit` for the new public API does not
change that global setting. The historical factory rationale was that its JAX backend would
flip the setting on first use anyway; enabling it at compatibility import makes that earlier
ordering safe without importing JAX for NumPy-only callers. Note what x64 actually is:

| | x64 off | x64 on |
| --- | --- | --- |
| `jnp.asarray(0.4)` | float32 | float64 |
| `jnp.asarray(0.4, jnp.float64)` | **float32** — cannot opt in | float64 |
| `jnp.asarray(0.4, jnp.float32)` | float32 | float32 — can still opt out |

It is a **cap**, not a default, and `complex_dtype` keeps selecting the solve's precision either
way (`complex_dtype=1` still returns complex64 with x64 on). The half of the old rationale that
survives is that a NumPy-only caller should pay nothing, so this never *imports* JAX — pinned
by `test_importing_wavebit_does_not_drag_jax_in`.

**What that cannot reach still warns.** Importing `wavebit.compat.legacy` before `import jax`
leaves no config to change. With `from wavebit.compat import legacy as wavebit`, call
`wavebit.enable_x64()` before creating JAX arrays to close that gap. Otherwise,
`check_input_precision` reports (`PrecisionLossWarning`) whenever `theta` or `phi`
arrives narrower than the working precision. Not JAX-specific — any array handed in below the
solve's precision loses the same way — and quiet for a Python float, a matching dtype, and
single data given to a single-precision solve.

### And offers a scoped override

The two regimes turn out to be mutually exclusive in practice, which makes the choice
*statable* rather than a guess — a superstrate with any absorption never produces the
degeneracy, because its roots are genuinely complex. A solve containing both wants the default:

| cutoff | complex angle only | weak absorber only | both in one solve |
| --- | --- | --- | --- |
| `1e-12` | 4.51 | 1.9e-04 | 1.4e-04 |
| `1e-6` | 1.3e-06 | 7.4e-01 | 1.4e-04 |
| `1e-5` | 3.4e-07 | 1.75 | 2.21 |

Hence `wavebit.branch_tolerance`, a context manager rather than a constructor keyword: it
overrides a convention for a block, which is what it honestly is, and it reaches all 15 call
sites across four modules without adding a permanent argument to the public API.

```python
with wavebit.branch_tolerance(1e-5):      # lossless superstrate, complex angle, float32
    res = solver.solve()
```

**Raise it for a lossless superstrate at a complex incidence angle; leave it alone if the
superstrate absorbs at all.** At complex128 it applies to nothing — the default already stands
four orders clear of round-off, and moving it can only hurt.

> **Superseded (2026-10-02).** The weak-absorber column measured the old half-space rule, which
> read the same cutoff. Since the half-space moved to Maystre's cut, the override moves only
> `mode_q`'s layer labels. In the weak absorber's configuration (re-created with the formal
> `theta = 0.6 + 1e-12j`) a raised cutoff no longer costs anything measured: SMM 4.4e-7 at
> `1e-12` and 3.8e-7 at `1e-5`, where the old rule gave 0.99. ETM's layers stopped reading it the
> same day (`etm_mode_q`), after which the override moved nothing but SMM's layer labels, which
> SMM does not depend on, and it was removed. This section is kept as the record of why it
> existed; §8 has the table.

## 5. What it bought

Max `|solver - RETICOLO|` on `Re(R_s)` of the (0,0) order, over all 441 points, `truncation_order=80`:

| | propagating (69) | evanescent (372) | points off by > 0.05 |
| --- | --- | --- | --- |
| before the `incident_kz` fix, float64 | 1.97e-14 | **7.85e-01** | 356 / 441 |
| after, float64 | 1.95e-14 | **3.91e-14** | 0 |
| float32, either way | 1.09e-03 | **1.55e+01** | 358 / 441 |

Float32 is unchanged and stays unusable across the light line — that is section 4's subject, and
it now announces itself rather than being silent. Float64 is what this table is about.

---

## 6. Why the suite could not see either one

This is the part worth keeping.

* **Both bugs are identities at a real incidence angle.** The first respells a number that is
  the same number; the second compares round-off against a threshold that round-off never
  reaches. No amount of tightening a tolerance on a real angle finds either.
* **Neither breaks energy conservation.** The degenerate `R = -delta` is perfectly unitary.
* **The wrong answer looked *tidier* than the right one.** A flat disc complement reads as
  "total reflection region", not as a failure.
* **No test in the repo set a complex `theta`** before 2026-08-11. The regime had never been
  exercised, so it rotted with nothing objecting — the same blind-spot shape as the normal-field
  goldens being all 1D or square-pixel.

What pins it now, in `tests/references/test_reticolo_conical_sweep.py` (the first three rows)
and `tests/rcwa/test_fp_warnings.py` (the last three):

| Test | Catches |
| --- | --- |
| `test_propagating_incidence_matches_reticolo` | the ordinary regime, 2e-13 |
| `test_evanescent_incidence_matches_reticolo` | failure one, from the 2024 golden |
| `test_uniform_stack_does_not_diffract_at_a_complex_angle` | failure one *away* from `Re(theta)=pi/2`, where it diverged to ~7e3 instead of going quiet |
| `test_double_precision_resolves_every_branch[3 regimes]` | that the warning is silent where the answer is trustworthy |
| `test_single_precision_reads_no_branch_from_round_off[3 regimes, ETM and SMM]` | that complex64 now resolves these regimes, silently and within 1e-5 of double (since 2026-10-02; until then it fired here) |
| `test_a_decision_on_its_cut_still_reports_at_single_precision` | that it still fires where a root sits within round-off of its cut |

The float64/float32 pair is the load-bearing one: zero false positives at double precision is
what makes the warning worth emitting at all.

Note that `|r_s| <= 1` is **not** available as an assertion under evanescent illumination — the
incident wave carries no power, and the analytic Airy amplitude for a lossless stack
legitimately reaches 1.01. Nor are the efficiencies: with no incident power there is nothing to
divide by, and the solve returns `R`, `T` and `power_entering` as NaN at every
`Re(theta) = pi/2` point of the sweep, keeping the amplitudes these tests compare
([#84](https://github.com/fieldalgo/wavebit/issues/84);
[`conventions.md`](../../reference/common/conventions.md#real-angle-incidence)). Orders that
must be identically zero are the honest invariant, which is what the uniform-stack test uses.

---

## 7. Where the oracle lives

The golden is `tests/references/golden_reticolo_conical_sweep.npz`, cut from a 2024-09-09
RETICOLO run by `gen_goldens_reticolo_conical_sweep.py`. Everything the tests need is in the
repository; the study it was cut from — notebooks, the raw `.mat` files, before/after data and
the figures — is retained privately by the authors and is not required to run anything here.
Regenerating the golden needs those `.mat` files and no RETICOLO installation, since the
generator only cuts its slice out of them; rerunning the sweep itself is what needs RETICOLO.

---

## 8. Maystre's cut (2026-10-02)

### The rule

```
kz = sqrt(eps - k_par**2),   the root with Re(kz) + Im(kz) > 0
```

The cut runs along the negative imaginary axis of `kz**2`. A passive medium at real `k_par`
has `kz**2` in the closed upper half plane. Its principal root is in the closed first
quadrant, where `Re(kz) + Im(kz) >= |kz|`, so on every real-`k_par` input the rule is the
principal root, bit for bit. That held over 4e5 random samples on each of the lossless and
absorbing branches, and every real-angle solve in the suite is unchanged. The rule differs
from the principal root only where `kz**2` falls below the real axis, which takes a complex
`theta` or a gain medium. RETICOLO takes the same cut (`retsqrt(x, -1)`, its "coupure de
Maystre"; J. P. Hugonin and P. Lalanne, *RETICOLO software for grating analysis*,
arXiv:2101.00901).

**The name.** "Maystre's cut" is RETICOLO's name, from its source code (`retsqrt.m`); its manual
does not use it. In print the convention is written `Re(sqrt(x)) + Im(sqrt(x)) > 0`: P. Lalanne
and T. Wu (arXiv:2508.10742 (2025)) attribute it to R. Petit (ed.), *Electromagnetic Theory of
Gratings* (Springer, 1980), and P. Lalanne and J.-P. Hugonin (arXiv:2609.10076 (2026)) cite
H. Benisty, J.-J. Greffet and P. Lalanne, *Introduction to Nanophotonics* (Oxford University
Press, 2022), §18.4.3, for it. Neither names Maystre. Together with §9's mirror, the rule is *the
limiting-absorption root, with the loss applied to `eps` and `mu` separately*; why that name and
not "the causal root" or "Petit's cut" is in [`negative-index-media.md`](negative-index-media.md)
§5, "What to call it".

**Not for a double negative.** The cut is decided on `eps*mu`, and for a medium with both real
parts negative that product sits where a gain medium's does, so the rule above takes the root that
grows. Such a half-space takes this cut's mirror image, §9.

### Why this cut

| | what it is | measured |
| --- | --- | --- |
| **limiting absorption** | Add `i*L` to the half-space's `eps` until `kz**2` is in the upper half plane, then remove it at fixed `k_par`, tracking the root by continuity. `kz**2` moves straight down, so it can reach the branch point only on the negative imaginary axis. That axis is this cut | 3000/3000 random `kz**2` |
| **switching gain on** | For `Im(eps) < 0` the same walk passes `Im(eps) = 0`, so it gives the root reached by turning the gain on from zero. At real `k_par` that is the causal answer for ordinary weak gain away from the critical angle (J. Skaar, Phys. Rev. E 73, 026605, 2006; J. O. Grepstad and J. Skaar, Opt. Express 19, 21404, 2011) | glass onto `1 - 0.05i`, 45 deg, TE: `R = 1.2646` |
| **continuity across the real axis** | The cut meets the real `k_par` axis only at the Rayleigh point, so nothing jumps as `Im(k_par) -> 0` from either side | 2e5 random orders at `Im(kx) = +-1e-9`: all within 1e-6 of the real-axis root |
| **an outside oracle** | RETICOLO takes the same cut | lossless-top grating at `theta = 0.2 +- 0.1i`: 2e-15 |

### What the old rule got wrong

From 2026-08-12 the half-space took the principal root, flipped where the straight segment
from normal incidence, `kz2(t) = eps - t*k_par**2`, crossed the negative real axis
(`_off_principal_branch`). Over a lossless half-space the flip never fires. The rule was then
the bare principal root, whose cut is the negative real axis of `kz**2`, which in `k_par` is
the real evanescent axis. That had three measured consequences.

* **Any imaginary part, however small, flipped evanescent orders.** Every evanescent order
  with `Re(kx)*Im(kx) > 0` took the growing root: 6858 of 6858 at `Im(kx) = 1e-3`, and the
  same at `1e-9`. The grating above moved `r0` by 0.22 at `Im(theta) = +1e-9` and by 0.36 at
  `-1e-9`. RETICOLO did not move.
* **Failure two in §4 was mostly this.** complex64 rounds `Re(theta) = pi/2` up, and
  `_incident_kxy` clamps it to just below, which gives `kx` a genuine imaginary part of
  ~1e-7. The old cut then flipped half of the evanescent orders. At the §4 point
  (`truncation_order=30`, NumPy), float32 against float64:

  | | old rule | Maystre's cut | and `etm_mode_q` |
  | --- | --- | --- | --- |
  | SMM | 8.2 | 3.4e-6 | 3.4e-6 |
  | ETM (the default) | 5.2 | 5.1e-2 | 1.9e-6 |

  The ETM residue in the middle column came from layer labels. `mode_q` reads `Re(q)` at
  round-off for exactly propagating modes, and at a complex `k_par` ETM is not label-free the
  way SMM is. A raised branch cutoff (`branch_tolerance(1e-5)`, since removed) sent those modes
  to the tie-break and gave 1.9e-6; `etm_mode_q` reads `Re(q) - Im(q)` and gets there with no
  cutoff ([`connecting-algorithms.md`](connecting-algorithms.md) §10).
* **Gain read as loss.** Under total internal reflection into a gain medium, the principal
  root is the lossy medium's root conjugated. So `R(1 - 0.05i) = R(1 + 0.05i) = 0.7907` at
  45 deg.

### The alternatives

| rule | its cut | verdict |
| --- | --- | --- |
| principal root, continued from normal incidence (2026-08-12) | the real evanescent axis | the three defects above |
| `Im(kz) > 0` (fmmax; RETICOLO's "coupure de Petit") | the real propagating axis | the mirror image: a propagating order takes the incoming wave at any `Im(kx)`. Gain at normal incidence (glass onto `1 - 0.05i`) gives `R = 24.98` |
| continuation from the nearest real `kx` (the "path D" that `time-convention-audit.md` §3 mandated) | vertical lines through the Rayleigh points | equals Maystre's cut except in a sliver of width `Im(kx)**2 / (2*sqrt(eps))` just past each Rayleigh point (0 of 20000 orders differ at `Im(kx) = 0.01`; all 121 orders of a 2D grating agree to 4e-15), and far from the axis in metallic cases. It needs `Im(kx)` and `Im(ky)` passed through every kernel, and no outside code takes it |
| Maystre's cut | the negative imaginary axis of `kz**2` | adopted |

In a lossy half-space every continuation agrees inside `|Im(kx)| < Im(sqrt(eps))` (2000 of
2000 random cases). Outside that band a complex-`kx` plane wave is not the steady state of
any real source, and the rules are conventions. Of those conventions, Maystre's equals
limiting absorption and is what an outside code uses.

### Gain near grazing

A gain half-space (`Im(eps) < 0` or `Im(mu) < 0`) takes Maystre's root, the root of switching
the gain on from zero at this frequency. Two caveats stay: which physical answer that root is,
which no single-frequency solve can settle (next), and near grazing, where a plane wave stops
describing a beam, which the solve reports with `GainCutoffWarning` (after it).

**Which answer this is.** Maystre's root is the answer for *weak, conventional* gain, and the
literature does not settle more than that; the solver's behaviour is unchanged by what follows.

* J. O. Grepstad and J. Skaar (*Opt. Express* **19**, 21404 (2011)), in this time convention,
  obtain evanescent gain beyond the critical angle (`R > 1`) for weak conventional gain by
  continuing from high frequency. Their `kz` changes sign only at the critical point, where
  `Re(kz**2) = 0` -- where Maystre's cut jumps in a gain medium, and where the band below sits.
* It is contested. T. S. Mansuripur and M. Mansuripur (*Appl. Phys. Lett.* **104**, 121106 (2014))
  argue that a semi-infinite gain medium gives no single-surface amplified total internal
  reflection, which is the `0.7907` the old rule gave (§8 above); A. E. Siegman (*Opt. Photon.
  News* **21**(1), 38 (2010), a magazine article, known here from the abstract of a related talk)
  argues against the amplification too.
* It can be the wrong root for a strongly dispersive active medium. Y.-F. Chen, P. Fischer and
  F. W. Wise (*Phys. Rev. Lett.* **95**, 067402 (2005)) propose an active, non-magnetic medium that
  refracts negatively where Maystre's cut takes the root of positive refraction, and B. Nistad
  and J. Skaar (*Opt. Express* **15**, 10935 (2007)) find a negative index in an active
  right-handed medium. Which root such a medium takes is fixed by its whole dispersion; at one
  frequency even weak gain at oblique incidence can leave `kz` without a meaning (J. Skaar, URSI
  EMTS 2010, arXiv:1008.2874).

[`negative-index-media.md`](negative-index-media.md) §18 places this among the rest of that
literature.

**A plane wave is not what a measurement sees here.** For a single Lorentz gain line every
nonzero `kx` has a branch point in the upper half of complex frequency, so an infinitely wide
plane wave over a gain medium has no true steady state at any oblique angle (J. Skaar, Phys. Rev. E 73,
026605, 2006). A finite beam does have one. It follows from pushing the in-plane wavevector
contour below the gain medium's branch point as `Im(w)` falls to zero (R. J. Briggs,
*Electron-Stream Interaction with Plasmas*, MIT Press, 1964). `studies/rcwa/gain_cutoff_band/gain_cutoff_band.py`
computes it for Gaussian beams of width `W`, from glass onto a Lorentz line pinned to
`eps = 1 - 0.05i`, TE. The natural length is the gain length
`L_g = 2*sqrt(Re eps)/(k0*abs(Im eps))`, here `40/k0`. The ratio below is
`abs(Re kz**2) / abs(Im kz**2)`, and the deviation is the beam's reflected field at its centre
against the same beam under Maystre's root, relative to the incident field:

| angle | ratio | `W/L_g = 2.5` | `5` | `10` | `20` |
| --- | --- | --- | --- | --- | --- |
| 42.0 deg | 0.15 | `abs(E_r/E_inc)**2 = 2.13` | 181 | — | — |
| 43.3 deg | 1.17 | deviation 4.4e-2 | 1.9e-3 | 2.0e-6 | 1.5e-16 |
| 43.8 deg | 1.56 | 5.1e-3 | 2.7e-6 | 1.8e-17 | 2.2e-60 |
| 44.5 deg | 2.11 | 6.0e-5 | 1.6e-11 | 6.2e-38 | 8.2e-142 |

* **Below a ratio of about 1** (`abs(kx - Re sqrt(eps)) < abs(Im sqrt(eps))`), the field at
  the beam centre grows with the beam's width without bound: light skimming the interface is amplified along
  it, and the field at the beam centre is a lateral wave fed from upstream (it grows downstream
  at about `abs(Im sqrt(eps))*k0`, 0.021 to 0.024 measured). No plane-wave value describes it.
* **Beyond 1**, the beam converges to Maystre's value, whatever the gain's dispersion, but only
  once it is a few gain lengths wide. "Whatever the dispersion" assumes the branch point
  crosses the real `kx` axis once as `Im(w)` falls to zero; see "What it cannot see". The tip of the contour weighs about
  `exp(-(ratio**2 - 1)*(W/L_g)**2/4)`.

**The band.** The solve warns per order where `abs(Re kz**2) < 2*abs(Im kz**2)`, with
`kz**2 = eps*mu - k_par**2` in units of `k0**2`. For weak gain that is an in-plane wavevector
within about `2*k0*abs(Im n)` of grazing in the gain medium. It holds the window where a beam's
response grows with its width, plus the zone where a beam a few gain lengths wide has not yet
converged. At the edge itself a beam needs `W/L_g` of 2.0 to 2.6 to come within 1e-3 of
Maystre's value, and 3.4 to 4.3 for 1e-6, on the slower side of the band and across
`abs(Im eps)` from 0.005 to 0.2.

At `abs(Im eps) = 0.05` it fires on 51 of the 900 angles from 0 to 89.9 deg; at 0.005 on 5,
at 0.2 on 210. In a grating every order has its own grazing condition in the gain medium, so
the band recurs near each order's Rayleigh condition. `solver.gain_cutoff_band()` returns the
per-order verdict for each side, and it traces, so a `jax.jit` or `vmap` caller can read it
where the warning cannot run. `CoherentSolver` runs the same test on its own ports for the
sectors a `PlaneWave`, `GaussianBeam` or `IncidentOrders` lights. It skips dipole sources
(see below).

**The fixed-kx flip does not set it.** Continue the root in complex frequency at fixed real
`kx` and it flips at an angle that moves with the gain's dispersion: 41.2462 / 41.8212 /
42.4828 deg for three line centres here, and up to a ratio of 4.9 on a fine grid of lines. A
finite beam does not see that flip. Its answer depends on `eps` at this frequency and on the
branch point crossing the real axis once. Beams centred at ratios 3 and 4 below a flip at 4.93
go to Maystre's value (0.4146, 0.3648), not to the fixed-kx one (2.4118, 2.7412).

**Why a warning, not a refusal.** The value is the exact answer to a stated problem, the band
has no provable edge, and a refusal would stop a sweep and NaN a traced solve. That is the
history of the Rayleigh-band refusals, two of which were reverted
([`rayleigh-points.md`](rayleigh-points.md)). The text is constant, so the default filter
shows it once per sweep, and it is not an `auto_detune` trigger, since a one-ULP nudge cannot
leave a band this wide.

**What it cannot see.**
* Sharp-edged illumination. A top-hat aperture seeds a growing lateral wave at its edges at any
  angle: at ratio 2.11 it deviates by 6.3e-2 to 34 for apertures of 100 to 800 `1/k0`, and at
  ratio 6.41 by 3.3e-2 to 13. The band is a property of smooth beams.
* A medium whose branch point does not cross the real `kx` axis exactly once as `Im(w)` falls.
  Every stable line in the study crosses once, but such a medium needs a different contour, and
  no single-frequency check can tell.
* A gain *layer* between passive half-spaces chooses no half-space root, but its amplified
  guided modes can make a beam's response depend on its width in the same way. The warning
  does not check layers. `ScatteringMatrix.cascade` does not re-check a joined stack.
* Dipole sources in `CoherentSolver`. A point source is sharp-edged, the case the band does
  not describe.

### What it does not settle

* Gain near grazing: see the previous section. The warning marks where the plane-wave answer
  stops describing a smooth beam; sharp-edged illumination of a gain half-space has no such
  band.
* Far from the real axis, where the rules differ, the choice is a convention. A study that
  needs a particular sheet for each order, as a leaky-mode pole search does, needs more than
  one default.
* At a complex `kx`, an order whose root decays with `Re(kz) < 0` has a negative formal
  efficiency. RETICOLO reports the magnitude, and
  `tests/references/test_reticolo_complex_kx_golden.py` applies that sign when it compares.

Pinned in `tests/rcwa/test_branch_selection.py` (the limiting-absorption walk, continuity across the
real axis, the grating, gain under total internal reflection), `tests/rcwa/test_gain_cutoff.py`
(the near-grazing band), `tests/rcwa/test_fp_warnings.py`
(single precision) and `tests/references/test_reticolo_complex_kx_golden.py` (RETICOLO, on both
sides of the old split).

---

## 9. Double negatives: the mirror of Maystre's cut (2026-10-03)

### The rule

```
Re(eps) < 0 and Re(mu) < 0:   kz = sqrt(eps*mu - k_par**2),   the root with Im(kz) - Re(kz) > 0
                               (on the cut, Im(kz) - Re(kz) == 0, the tie keeps -v)
otherwise:                     Maystre's root, Re(kz) + Im(kz) > 0
```

The tie is Maystre's mirrored, so the double-negative rule is `-conj(maystre(conj(kz**2)))` to
the bit, signed zeros included.

`boundary_kz` takes the half-space's `(eps, mu)` as `medium` and decides on the material, not
on `eps*mu`. The product cannot tell a passive double negative from a gain medium: `eps =
-2+0.1i`, `mu = -1+0.1i` multiplies to `1.99-0.3i`. The kernels pass the pair whenever the
scene is magnetic (`conventions.half_space_medium`), and a non-magnetic solve passes nothing,
so its arithmetic does not change.

### Why this cut

It is the same limiting absorption as §8, run for the medium's own loss.
`Im(eps*mu) = Re(eps) Im(mu) + Im(eps) Re(mu)`, so loss *lowers* `kz**2` when both real parts
are negative, where it raises it in an ordinary medium. Give a double negative enough loss to
push `kz**2` into the lower half plane, take the decaying root (the second quadrant), and remove
the loss at fixed `k_par`. `kz**2` moves straight up, so it can reach the branch point only on
the positive imaginary axis, which is this cut. It is Maystre's cut under `kz -> -conj(kz)`,
and at every real `k_par` of a passive double negative it gives `Im(kz) >= 0` and, on a
lossless propagating order, `Re(kz) < 0`, so `Re(kz/mu) > 0`: power leaves the interface while
the phase comes toward it (V. G. Veselago, Sov. Phys. Usp. 10, 509 (1968)). The decision is at
least `|kz|` from zero there.

The rule is the published one, though the cut is not. M. Cassier, C. Hazard and P. Joly
(*Commun. PDE* **42**, 1707 (2017); **47**, 1217 (2022)) prove it for a lossless Drude
half-space whose `eps` and `mu` are both dispersive: the limiting-absorption root, with the loss
on each factor, changes sign exactly in the bands where both are negative. H.-M. Nguyen and S. Sil
(*Commun. Math. Phys.* **379**, 145 (2020)) put the loss on `eps` and `mu` separately for a
bounded negative inclusion. M. Cassier, P. Joly and M. Kachanovska (*Comput. Math. Appl.* **74**,
2792 (2017), Theorem 3.34) show that a passive, non-dissipative local material carries backward
waves exactly where `eps < 0` and `mu < 0` -- the trigger, in the lossless limit. R. A. Depine and
A. Lakhtakia (*Optik* **116**, 31 (2005)) state the lossless mirror for the eigenvalues of a modal
(C) method over a negative-phase-velocity half-space, and M. Lequime et al. (arXiv:1312.6288
(2013)) and PyMoosh's code (D. Langevin et al., *J. Opt. Soc. Am. B* **41**, A67 (2024)) state the
same switch independently. None of them writes it as a cut on the positive imaginary axis of
`kz**2`, and the literature has no name for that cut: "the mirror of Maystre's cut" is a
description ([`negative-index-media.md`](negative-index-media.md) §5 and §18).

The two rules take the same root wherever `Re(kz**2) < 0`. They part only on
`Re(kz**2) >= 0` -- every propagating order of a double negative, where Maystre's root grew --
and on the negative imaginary axis itself, Maystre's cut, where his tie-break keeps the growing
root of a passive double negative (`eps = mu = -1+1i` at normal incidence). A double negative
too lossy to propagate anywhere (`Re(eps*mu) <= 0`) therefore keeps its `kz` to the bit.

The half-space's index follows: `n = sqrt(eps)*sqrt(mu)`, each factor's root on Maystre's cut --
the principal root for every passive factor (`conventions.signed_index`; A. A. Govyadinov,
V. A. Podolskiy and M. A. Noginov, Appl. Phys. Lett. 91, 191103 (2007)) -- which for a passive
double negative is the negative of the principal `sqrt(eps*mu)` and the root above at
`k_par = 0`. The same product negates the
principal root of every negative-phase-velocity medium (M. W. McCall, A. Lakhtakia and
W. S. Weiglhofer, Eur. J. Phys. 23, 353 (2002)), which includes passive magnetic metals that
are not double negatives, and of no other. Only the sign moves, so `n**2` and every `kz` are
unchanged; what it sets is `n/mu`, the admittance with which the kernels quote `t_p` and the
field state expands it, and on the lit side the direction of the incident `k_par`. With the
principal root the far side's `t_p` was `-t_s` at normal incidence. A factor exactly on the
negative imaginary axis is the cut's tie and keeps its principal root, decided on the factor's
real part rather than on the round-off of the root's `Re + Im`. Which media it negates, the signed
zeros, that tie and the class boundary: [`negative-index-media.md`](negative-index-media.md) §6.

### What it was measured against

Glass onto a double negative, closed-form Fresnel `r` and `t` for s and p, 0/30/60/80 degrees,
lit from either side, on the pure TE and TM, conical, 2D and first-order paths, ETM and SMM:
`3.2e-15` at worst over `(-2+0.1i, -1+0.1i)`, `(-3+0.1i, -0.5+0.05i)`, the lossless `(-2, -1)` and
`(-6+0.3i, -1.5+0.05i)`. A grating on the half-space against the same grating on a 16 um lossy
slab of the medium, which as a layer keeps both roots and needs no rule: every order's `R`
agrees to `4.0e-16`, and the power the slab takes in is the half-space's `T` to `1.7e-15`, on
every path and both algorithms. The lossless half-space is the limit of that slab as its loss
goes to zero: on the conical grating `1e-5` of loss on a 128 mm slab is within `1.9e-5` of it,
and a Richardson step on `1e-4` and `1e-5` within `2.3e-8`. `studies/rcwa/negative_index_media/negative_index_media.py`
prints all of these (section 8 of its log).

### Layers of a double negative

A layer keeps both roots, so its label is a choice of basis rather than a boundary condition --
but ETM's recursion is not free to make it (`connecting-algorithms.md` §10). A uniform
double-negative layer takes this section's mirror for its labels, exactly as its half-space
does, and a patterned layer holding one is labelled mode by mode by the direction of its power;
SMM's tie is mirrored the same way. Measured, and the perfect-lens pair that ETM cannot write,
in [`connecting-algorithms.md`](connecting-algorithms.md) §11.

### A lit double negative

Solved since 2026-10-04. `theta` and `phi` name the direction the incident light travels, as
they always did, so the in-plane wavevector of a double negative's incident wave points the
other way: `k_par = s*N*sin(theta)*(cos(phi), sin(phi))` with `s = -1`, and the incident `kz`
is this section's root at that `k_par`, decaying toward the structure. Until that meaning was
settled the lit side was refused: phase and power run opposite ways in it, and an angle had to
say which it followed. The rule, the frame at `theta = 0` and the measurements are in
[`negative-index-media.md`](negative-index-media.md) §7; fmmax agrees through the `theta` door
to 2.7e-14 per order (`tests/references/test_fmmax_dng_golden.py`).

### What it does not cover

* **An active double negative** that propagates. Refused where it is set, either side: the
  limiting absorption that fixes the passive root says nothing about gain, and no
  single-frequency rule can (J. Skaar, Phys. Rev. E 73, 026605 (2006); B. Nistad and J. Skaar,
  Phys. Rev. E 78, 036603 (2008)). Under `jax.jit`, where no refusal can read the value, the
  medium is solved with this section's root on every order, and `gain_cutoff_band` flags only
  the orders in its band near grazing: a double negative's gain band sits at `Im(kz**2) > 0`, the
  mirror of an ordinary gain medium's, and the traceable mask reads
  `where(double_negative, Im(kz**2) > 0, Im(kz**2) < 0)`. The other orders are solved unflagged,
  with no physical answer behind them either, so check such a material once eagerly, outside the
  trace ([`negative-index-media.md`](negative-index-media.md) §10).
* **A caller holding only `eps*mu`.** A kernel called with `mu_bot` and no `eps_bot` takes
  Maystre's root. The engine always passes both on a magnetic scene.
* **Two lossless half-spaces that are each other's negation**, `(eps, mu)` and `(-eps, -mu)`. No
  root rule helps: on every evanescent order both take the same decaying root, their admittances
  cancel, and the time-harmonic problem has no solution (B. Gralak and D. Maystre, C. R. Physique
  13, 786 (2012); M. Cassier and P. Joly, arXiv:2504.00235 (2025)). Refused, with uniform layers
  of the two media between them or none, wherever the truncation retains an evanescent order; on
  a propagating order the two cuts take opposite roots, the admittances are equal, and a
  truncation of propagating orders only is solved, `r = 0` and `t = 1`
  ([`negative-index-media.md`](negative-index-media.md) §9).
* **`Im(eps)` and `Im(mu)` of opposite signs.** The rule assumes both non-negative; the effective
  `Im(mu)` of a passive homogenised medium can be negative (B. Gralak, C. R. Physique 21, 343
  (2020)), and then neither cut is the outgoing wave. Such a double negative always propagates and
  is refused as active ([`negative-index-media.md`](negative-index-media.md) §10).
* **A tensor half-space.** Half-spaces are isotropic. A hyperbolic one would need its waves sorted
  by Poynting flux one by one, not a cut per medium
  ([`negative-index-media.md`](negative-index-media.md) §15).

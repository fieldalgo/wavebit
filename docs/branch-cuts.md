# Branch cuts — which square root, and the two ways it went wrong

Written: 2026-08-11. Measured on CPU (Apple Accelerate), NumPy / JAX / PyTorch, float64 and
float32.

**Half-space rule changed (2026-10-02).** `boundary_kz` now takes Maystre's cut, the root with
`Re(kz) + Im(kz) > 0`. §8 gives the reasons and the measurements, and it supersedes two claims
below. First, failure two (§4) is not a limit of single precision for the half-space root. The
old rule's cut lay along the real evanescent axis, and complex64 gave `kx` a genuine imaginary
part of ~1e-7. Under the new rule the float32 SMM solve is within 3.4e-6 of float64 at the point
that was 4.5 off. What remains at single precision is a layer-label effect on the ETM algorithm,
and `branch_tolerance` still addresses it. Second, the trade in §4's "And offers a scoped
override" belonged to the old half-space rule. Under the new one a raised cutoff costs nothing
measured.

Targets:
- `wavebit/_core/conventions.py` — `mode_q`, `boundary_kz`, `incident_kz`,
  `_check_resolvable`, `branch_tolerance`
- `wavebit/_core/solve_1d.py` — `solve_1d` stage 3, `final_matching`
- `tests/references/test_reticolo_conical_sweep.py` — the RETICOLO oracle
- `tests/test_fp_warnings.py` — the three tiers of "this result is not trustworthy"

---

## TL;DR

Every `sqrt` of a complex number in this solver has **two answers**, and picking one is a
physical decision, not an arithmetic one. Two separate things went wrong there — one a bug,
one a limit — and neither was visible at a real incidence angle:

| | Symptom | Cause | Fixed |
| --- | --- | --- | --- |
| **Respelled root** | complex `theta` returned `R = -delta`, `T = 0` — a grating that scatters nothing | the incident wave's `kz` was written two ways that agree only for real angles | `incident_kz` |
| **Unresolvable branch** | complex64 returns \|R\| ~ 4 on all three backends, each wrong differently | at that precision `Re(v)` *is* round-off, so the branch is taken from the sign of noise | it warns, and offers a scoped override. **The half-space part was fixed on 2026-10-02 (§8)**; a layer-label part remains on ETM |

The first was invisible because **a real incidence angle makes it an identity**; it is fixed and
double precision is now exact to 1e-14 everywhere measured. The second is not a bug that can be
fixed — single precision does not carry the information the decision needs — so the solver
detects it and says so. The oracle that found both is a 2024 sweep over complex angles.

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

"Down" and "up" here are the propagation kernels' words, as in `_core/conventions.py`: `z` runs
from the kernel's entry face into the stack, the way its incident wave travels. That was the
public frame when this page was written. Since 2026-09-21 the public `+z` points up from `z = 0`
at the bottom interface, and the kernels solve public top incidence as their z-mirrored scene,
so "down" below is public `-z` under top incidence and public `+z` under bottom incidence
(see [`conventions.md`](conventions.md#geometry-and-propagation)).

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
> `docs/time-convention-audit.md` §2-§5; pinned in `tests/test_branch_selection.py`
> (`test_the_two_selectors_disagree_at_complex_k_par_by_design` is the exact case above).
> Everything below this box concerns the regimes where the identity does hold, and stands
> unchanged.
>
> **Superseded for the half-space (2026-10-02):** `_off_principal_branch` is gone, and
> `boundary_kz` takes Maystre's cut (§8). The layer half of this box stands. Since 2026-09 a real
> angle in an absorbing superstrate is a phase angle with a real `k_par`, so only a complex
> `theta` reaches this regime.

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

(vs the analytic Airy reference in `tests/test_lossy_media.py`, uniform slab.)

---

## 4. Failure two — a question single precision cannot answer

Both branch rules asked the same thing, `is Re(v) zero?`, and answered it against a fixed
number. Since 2026-10-02 only `mode_q` does; the half-space rule reads the sign of
`Re(v) + Im(v)`, which no real `k_par` brings near zero (§8):

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

`wavebit/backends/jax_backend.py` enables `jax_enable_x64`, and it does so deliberately late: the
setting is process-wide and belongs to the caller, so it happens when the JAX backend is first
*used*, not on `import wavebit`. An array built in the gap between the two is silently
complex64:

```python
import jax.numpy as jnp, wavebit
theta = jnp.asarray(0.4 + 0.3j)                               # complex64 — x64 is not on yet
wavebit.rcwa(backend=1, theta=theta, complex_dtype=0, ...)   # ...but double was asked for
```

The setter's `jnp.array(theta, dtype=complex128)` then widens it without recovering anything,
and the solve runs with ~8 significant digits missing from its incidence angle. Measured
`|R| = 3.94` against a true 0.67, with every dtype in sight reading complex128 — **this was
diagnosed as a solver bug before it was diagnosed as an import ordering artifact**, which is
the whole argument for reporting it.

Two things address it, because neither is enough alone.

**The cap is lifted earlier.** `import wavebit` now enables x64 when JAX is *already* loaded
(`_enable_jax_x64_if_already_imported`). The original rationale was that the setting is
process-wide and belongs to the caller — but wavebit flips it either way, just later, so
deferring never spared the caller the change; it only made the moment depend on import order.
Note what x64 actually is:

| | x64 off | x64 on |
| --- | --- | --- |
| `jnp.asarray(0.4)` | float32 | float64 |
| `jnp.asarray(0.4, jnp.float64)` | **float32** — cannot opt in | float64 |
| `jnp.asarray(0.4, jnp.float32)` | float32 | float32 — can still opt out |

It is a **cap**, not a default, and `complex_dtype` keeps selecting the solve's precision either
way (`complex_dtype=1` still returns complex64 with x64 on). The half of the old rationale that
survives is that a NumPy-only caller should pay nothing, so this never *imports* JAX — pinned
by `test_importing_wavebit_does_not_drag_jax_in`.

**What that cannot reach still warns.** `import wavebit` before `import jax` leaves no config
to change, so `check_input_precision` reports (`PrecisionLossWarning`) whenever `theta` or `phi`
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
> `1e-12` and 3.8e-7 at `1e-5`, where the old rule gave 0.99. The rule above is now simply:
> raise it if `UnresolvedBranchWarning` fires on an ETM solve. §8 has the table.

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

What pins it now, in `tests/references/test_reticolo_conical_sweep.py`:

| Test | Catches |
| --- | --- |
| `test_propagating_incidence_matches_reticolo` | the ordinary regime, 2e-13 |
| `test_evanescent_incidence_matches_reticolo` | failure one, from the 2024 golden |
| `test_uniform_stack_does_not_diffract_at_a_complex_angle` | failure one *away* from `Re(theta)=pi/2`, where it diverged to ~7e3 instead of going quiet |
| `test_double_precision_resolves_every_branch[3 regimes]` | that the warning is silent where the answer is trustworthy |
| `test_single_precision_reports_an_unresolvable_branch[3 regimes]` | that it fires where it is not |
| `test_branch_tolerance_rescues_a_complex_angle…` | that the override actually moves the answer |

The float64/float32 pair is the load-bearing one: zero false positives at double precision is
what makes the warning worth emitting at all.

Note that `|R| <= 1` is **not** available as an assertion under evanescent illumination — the
incident wave carries no power, and the analytic Airy result for a lossless stack legitimately
reaches 1.01. Orders that must be identically zero are the honest invariant, which is what the
uniform-stack test uses.

---

## 7. Where the oracle lives

The golden is `tests/references/golden_reticolo_conical_sweep.npz`, cut from a 2024-09-09
RETICOLO run by `gen_goldens_reticolo_conical_sweep.py`. Everything the tests need is in the
repository; the study it was cut from — notebooks, the raw `.mat` files, before/after data and
the figures — is retained privately by the authors and is not required to reproduce anything
here. Regenerating the golden needs a RETICOLO installation and nothing else.

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

  | | old rule | Maystre's cut |
  | --- | --- | --- |
  | SMM | 8.2 | 3.4e-6 |
  | ETM (the default) | 5.2 | 5.1e-2 |
  | ETM, `branch_tolerance(1e-5)` | 1.9e-6 | 1.9e-6 |

  The ETM residue comes from layer labels. `mode_q` reads `Re(q)` at round-off for exactly
  propagating modes, and at a complex `k_par` ETM is not label-free the way SMM is. Raising
  the cutoff sends those modes to the tie-break.
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

### What it does not settle

* Within about a degree of the critical angle, a gain half-space has no right answer at a
  single frequency: the causal flip angle depends on the dispersion of `eps`.
* Far from the real axis, where the rules differ, the choice is a convention. A study that
  needs a particular sheet for each order, as a leaky-mode pole search does, needs more than
  one default.
* At a complex `kx`, an order whose root decays with `Re(kz) < 0` has a negative formal
  efficiency. RETICOLO reports the magnitude, and
  `tests/references/test_reticolo_complex_kx_golden.py` applies that sign when it compares.

Pinned in `tests/test_branch_selection.py` (the limiting-absorption walk, continuity across the
real axis, the grating, gain under total internal reflection), `tests/test_fp_warnings.py`
(single precision) and `tests/references/test_reticolo_complex_kx_golden.py` (RETICOLO, on both
sides of the old split).

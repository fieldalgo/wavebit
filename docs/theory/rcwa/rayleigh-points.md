# Rayleigh points — the exact graze, which paths solve it, and what `auto_detune` does

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


Written: 2026-08-16; §6 (the 2D coupled-cluster rebuild) added later the same day; §7 (the
1D conical x-cutoff at `ky != 0`, which is not a Rayleigh point) added 2026-10-10; §§8–9 record
the 2D and first-order exceptional-point design and judge measurements from 2026-10-12.
Measured on CPU (Apple Accelerate), NumPy / JAX / PyTorch, float64; the single-precision
caveat is stated where it applies. This file is the measurement record; for the
narrative walkthrough of how the pieces hand off to each other at solve time, read
[`rayleigh-handling-walkthrough.md`](../../guide/rcwa/rayleigh-handling-walkthrough.md).

Targets:
- `src/wavebit/rcwa/_engine/solve_1d.py` — `layer_modes_1d_conical` (the reformulation), `layer_modes_1d`,
  `_cutoff_cluster`, `_one_side` and `_regular_conical_modes` (§7)
- `src/wavebit/rcwa/_engine/solve_2d.py` — `layer_modes_2d`, `_rebuild_grazing_cluster` (§6),
  `_defective_pairs`, `_rebuild_defective_pairs`, `_unresolved_defective_cluster`,
  `_defective_clusters`, `_cluster_split`, `_rebuild_defective_clusters` (§8)
- `src/wavebit/rcwa/_engine/solve_2d_first_order.py` — `_pair_gate`, `_regular_pair`, `_layer_split` (§9)
- `src/wavebit/_common/backend.py` — `check_mode_cutoff`, `check_finite`,
  `RayleighPointWarning`, `AutoDetuneWarning`, `ExceptionalPointWarning`
- `src/wavebit/rcwa/_engine/engine.py` — `solve` (the `auto_detune` ladder)
- `tests/rcwa/test_rayleigh_guard.py` — every contract below, pinned; `tests/rcwa/test_conical_cutoff.py`
  — §7; `tests/rcwa/test_2d_exceptional_point.py`, `tests/rcwa/test_exceptional_point_clusters.py` — §8;
  `tests/rcwa/test_first_order_exceptional_point.py` — §9

---

## TL;DR

A **Rayleigh point** is a retained order sitting exactly on some medium's light line:
`kx_m == n`, so that order's `kz` is zero — neither propagating nor decaying. The failing
set is a **single floating-point value** per parameter: one ULP of wavelength, period or
angle to either side, the problem is ordinary, and the answers on the two sides agree to
eight digits. The physics is continuous across the point; the singularity is in the
coordinates, not in the structure.

What happens exactly on the point depends on the solve path, and on nothing else — the long
hunt for a *configuration* that separates failing from working cases found none, because the
"working" cases were either a different path or a round-off lottery win:

| Path | At the exact point | Why |
| --- | --- | --- |
| half-space graze (any path) | always fine | no mode matrix is built for a half-space |
| a layer of a half-space's own medium, or an empty stack, at that half-space's graze | **one plane wave: merged into the half-space and exact since 2026-11-09 (§10)**; a coherent multi-source solve refuses (`rcwa.singular`) | the layer holds the `q = 0` cluster no exponential basis represents; the interface it shares with its medium reflects nothing, so removing it is the exact answer |
| 1D pure TE/TM (`tetm=`) | always fine | `V = W*q` — nothing singular is ever inverted |
| 1D conical (`psi`/pair), `ky == 0` | **fine since the reformulation** | `V` blocks now come from the eigensystem; they used to pass through `inv(A)` of an exactly singular `A` |
| 1D conical at `kx_m**2 = eps*mu` with `ky != 0` (an x-cutoff, `kz = abs(ky)`) | **solved in a regular basis since 2026-10-10 (§7)**, eagerly and, since 2026-10-11, under every abstract trace (`jax.jit`, `jax.vmap`, a `jax.lax.map` or `scan` body, `jax.checkpoint`, `torch.func.vmap`) | the two mode families collapse onto one field there; the layer is rebuilt with an upper-triangular propagator, exact in a uniform layer and at a patterned layer's Jordan block alike; patterned double negatives at a complex `theta` too |
| 2D graze | **correct or loud since the rebuild (§6)** | the grazing cluster is re-derived from the coupled first-order problem: representable clusters solve exactly, numerically defective ones refuse with the detune advice |
| 2D or first order at a patterned EP (`q = abs(ky)`) | **rebuilt as a regular pair (§§8–9; design measured 2026-10-12)**; coincident and nearly coincident EPs **rebuilt as one cluster, eagerly and under traces**, on 2D (§8, 2026-11-08) and on the first-order formulation (§9, 2026-11-11) | larger or chained clusters warn eagerly; under a trace the same screens run as data and the layer answers NaN (§8, 2026-11-10) |
| any path, eigenvalue *exactly* `0.0` — and, on 2D, any cluster whose coupled rates split below ~1e-10 of the modal scale | refuses, unless the ladder clears it | the exponential mode basis is degenerate at the point itself (the limit basis is `{1, z}`) |

`solve` handles the practical side by default: `auto_detune=True` retries a solve that
refused or fired `RayleighPointWarning`, first at `wavelength*(1+1e-12)`, then also
`theta*(1+1e-12)` for the one graze wavelength cannot move — and reports every knob it
moved in an `AutoDetuneWarning`. Nothing is ever moved silently. (That default is the earlier
interface's, `wavebit.compat.legacy`. Since 2026-10-13 the public `wavebit.rcwa` module makes the
retry opt-in: `rcwa.Numerics(detuning=None)`, the default, solves the requested point exactly,
and a positive `detuning` is the relative step, recorded with the result.)

---

## 1. Why the paths differ — the nearby-problem criterion

`eig` is backward stable: the computed `(w, W)` is the exact eigensystem of `A + δA` with
`δA` at round-off scale. A layer whose `(W, V, X)` are all exact functions of that one
computed eigensystem is therefore the exact mode set of a *nearby physical layer* — and
since the answer is continuous across the Rayleigh point, the nearby layer's answer is the
right answer to ~`eps`-level. That criterion, not the accuracy of any individual matrix,
is what separates the paths. (The computed `q ~ 1e-8` at the graze is 100% wrong relative
to the true `q = 0` on *every* path — consistency is what matters, not closeness.)

- **Pure 1D**: `V = W * q` — products of the eigensystem. Consistent by construction.
- **Conical 1D, before**: `V_11 = (inv(A) @ W1) * q1` with `A` exactly singular
  (cond ~1e18). LU of a singular matrix returns garbage of arbitrary size with nothing
  raised, and the garbage is consistent with no problem at all. Measured on the defect
  case (D#81): `R` from 0.18 to 626 against a true 0.22, scattered randomly across `truncation_order`
  and connecting algorithm — one draw (`SMM`, `truncation_order=5`) returned `R+T = 0.981` with every
  guard green, reading as ordinary absorption on a lossless stack.
- **Conical 1D, after**: `Ky2 == ky^2 * I` on this path, so `W1` diagonalises `A` itself
  and `W2` does `B @ epx_conv`. The `V` blocks are rewritten as exact functions of the
  eigensystem (`inv(A) @ W1 == W1 * (1/(w1 - ky^2))`, and likewise for `B`), plus one
  inverse each of the well-conditioned `W1`/`W2` (cond ~2). Nothing singular is inverted.
- **2D, before the rebuild**: `V = (Omega_R @ W) / q`. The original diagnosis blamed
  cancellation in the product — measured later (§6), the truth is sharper: on the repro
  stack the computed `Omega_R @ W` agrees with 200-bit arithmetic to 1e-22, and the answer
  is *still* wrong by 1e-2. The cluster's eigvecs of the product `Omega2` are eps-unstable
  against the factor split, so the pair `(W_j, V_j, q_j)` satisfies `Omega_R W = V q`
  while violating `Omega_L V = W q` at order 1 — the layer is the mode set of *no* nearby
  problem, and no identity rescues it: `W` does not diagonalise `Omega_R` alone, and
  `Omega_L` is singular at the same point. The repair is §6's coupled-cluster rebuild.

## 2. What the conical fix measures as

| Claim | Measurement |
| --- | --- |
| exact point = one-ULP neighbour | 8 digits, on numpy/jax/torch × ETM/SMM × TE/TM/mixed `psi` |
| energy at the exact point | `R+T = 1` to 1e-10 (lossless stacks) |
| no behaviour change off the point | old vs new ≤ 8.2e-15 over 40 random conical solves; full suite green |
| fields | the old path's field reconstruction was wrong **over a band** — at one ULP off it erred at 2e18 scale and converged only ~1e4 ULP out, while efficiencies looked fine; the new path is smooth from the exact point (2.9e-8 to its neighbour, ETM) |
| cost | none — median solve times equal within noise at `truncation_order` 20/60/100 (the per-layer `eig` dominates both spellings) |

## 3. The 2D detection (now the fallback contract)

The 2D efficiency error of the *old product construction* follows
`~eps / (min|w| / max|w|)` — measured on four exact-graze stacks over a detune ladder:

| `min|w|/max|w|` | efficiency error |
| --- | --- |
| 1e-17 (exact) | 2e-3 … 8e0, some with **no warning of any kind** (5 of 48 solves) |
| 1e-15 | ~1e-3 |
| 1e-14 | ~1e-4 |
| 1e-13 | ~1e-5 |
| ≥ 1e-12 | ≤ 2e-6 |

Since §6 the rebuild claims the whole band below `solve_2d._COUPLED_REL = 1e-9`, so this
curve describes what ships only on the rebuild's fallback paths (E/H cluster-size
mismatch; concrete-but-unmaterializable values under grad). There `check_mode_cutoff`
still warns `RayleighPointWarning` below `1e-13` — where the error crosses ~1e-5 — a band
~1000 wavelength-ULPs wide. 60 random non-grazing 2D solves fired 0 false positives, and
no existing suite case fires it. It is a warning rather than an error on purpose: most of
the band is wrong at the 1e-5 level, not garbage, and the two earlier attempts at
*refusing* in this regime both took correct answers away and were reverted (the §6
refusal is narrower: it fires only where the basis provably does not exist).

## 4. `auto_detune` — the ladder, and its contract

Default on in the earlier interface's `solve()`; opt-in through `rcwa.Numerics(detuning=...)` in
the public module since 2026-10-13. When a solve refuses with a degeneracy (`FloatingPointError` /
`LinAlgError`) or fires `RayleighPointWarning`:

1. retry at `wavelength * (1 + 1e-12)` — moves every `m != 0` order;
2. if the condition survived (only `kx_0 = n_top*sin(theta)` is wavelength-free), retry
   with `theta * (1 + 1e-12)` as well — moves every order at once. No second rung exists
   at `theta = 0`, and none is needed: `kx_0 == 0` there, which grazes nothing;
3. if nothing helped, surface the asked-for point's own error or result, warnings and
   all, with the advice that remains (nudge an index or the period).

The contract, in order of importance:

- **Nothing moves silently.** Every rung that takes is named in an `AutoDetuneWarning`
  with the value actually used, and the solver keeps that value — `solver.wavelength`
  after the call *is* the wavelength of the returned result, so field reconstruction and
  everything downstream describe one consistent problem. This is the deliberate opposite
  of the removed `perturbation` knob, which nudged every solve, silently, by an amount
  too small to help.
- **Gradients survive.** The knobs are multiplied, not replaced, so a differentiable
  `wavelength`/`theta` keeps its autodiff graph through the retry (pinned:`jax.grad`
  through the refused point returns a finite, nonzero thickness gradient). The gradient
  returned is the nearby problem's — which is also the only one that exists, since the
  exact point is a cusp of `R(lambda)`.
- **Off the singular set it does nothing.** Bit-identical results, no warning; no new
  warning fires anywhere in the suite.
- `auto_detune=False` solves exactly the requested point; a float chooses the relative
  step. The default step is sized for float64 — complex64 resolves nothing below ~1e-7,
  so pass ~1e-5 there. A traced (`jax.jit`) solve skips every guard and therefore never
  retries.

The step itself is a measurement, not a choice: `1e-12` stands one order clear of where
the 2D warning band ends (~1e-13 relative) while remaining 5e-10 nm at 500 nm.

## 5. What still refuses, and why that is correct

An eigenvalue that is *exactly* `0.0` — a uniform layer whose FFT is noiseless on a
binary-exact grid — makes the mode partner 0/0. At that point the exponential basis
`e^{±k0 q z}` is genuinely degenerate (the correct limit basis is `{1, z}`), so no
spelling of this formulation produces the value; the refusal is a fact about the basis,
not a missing case. Two rungs of the ladder clear it in practice; if both fail, the
error message prescribes what is left.

The tempting shortcut was measured and rejected: lifting exactly-zero eigenvalues to
`eps`-scale ("it is within `eig`'s own backward error") returned 8-digit-correct answers
on a narrow test and failed 7 of 30 once broadened — the exact-zero point entangles the
layer eigenvalue with the `kz` of a grazing half-space, and regularising one of them
produces a problem whose parts disagree. A perturbation is only safe when it is applied
to the *inputs*, consistently, which is precisely what the ladder does and the shortcut
did not.

**Under `jax.jit` neither the refusal nor the ladder runs** (added 2026-10-20). `check_finite`
has no value to read there, so such a layer returns `NaN` -- value and gradient -- with no
notice. One family is answered: a layer of a half-space's own medium, which the half-space
merge deletes ([`connecting-algorithms.md`](connecting-algorithms.md) §10). Under a trace
whether `k_par` is real cannot be read, so the merge takes the layer at any angle, as long as
its materials can be read. Measured on a TE grating at normal incidence with
`wavelength == period`, so orders ±1 sit on the vacuum light line, under a 50 nm vacuum layer
beneath a vacuum cover: a jitted `value_and_grad` of `T` over the grating's thickness returned
`NaN` for both when the solver was built inside the jitted function, because its constant
materials were held as tracers and nothing merged; built before the trace, the same solver
merged and answered. Both now return the stack with that layer deleted (`T = 0.9943095000422952`,
ETM and SMM alike, and its gradient to the last bit), where the eager ladder answers its `1e-8`
neighbour, 3.2e-7 away. At `theta = pi/2` with the same layer the specular order grazes, and
the value and gradient of `|t_s|**2` were `NaN` the same way; they are now the deleted stack's.
`tests/rcwa/test_half_space_merge.py` pins both. A uniform layer whose medium is not the
half-space's beside it -- the same vacuum layer moved down onto the `eps = 2.25` substrate --
still returns `NaN` under `jit` where the eager ladder answers; that gap is open.

## 6. The 2D coupled-cluster rebuild — what solved it, and the wall that remains

`solve_2d._rebuild_grazing_cluster`, engaged by `layer_modes_2d` below
`min|w|/max|w| = 1e-9` (`_COUPLED_REL`). Everything in this section is measured; the
oracle throughout is the **two-sided sqrt extrapolation**: near a Rayleigh point
`R(λ(1±h))` varies as `R0 ± a·sqrt(h)` (a Wood anomaly, so *linear* Richardson silently
misconverges), and extrapolating both wavelength sides to `h → 0` brackets the exact
answer to ~6e-9 on the repro stack. The one-ULP neighbour is **not** a usable oracle for
the old construction: one ULP of wavelength moves the eigenvalue ratio only to ~1e-15,
still deep inside its broken band (the detuned "truth" itself returned `R+T = 1.0165`).

**Why no repair of `V = (Omega_R @ W)/q` could work.** Computing that product in 200-bit
arithmetic changed it by 1e-22 and the answer stayed wrong by 1e-2 — the failure was
never rounding. The grazing cluster's eigenvalues are separated by ~eps, so the product's
eigvecs are round-off-determined, and no `V` completes them into a mode set satisfying
both first-order relations `Omega_R W = V q` **and** `Omega_L V = W q`; whichever
relation breaks is amplified by the `1/q` conditioning of the interface inversions
(`solve_2d` inverts `big_W` and `big_V` separately).

**What does survive round-off** is the pair of cluster invariant *subspaces* — `S_E` of
the product `Omega2` and `S_H` of the swapped product `Omega_R @ Omega_L` (gap O(1) to
the rest of either spectrum). Restricting the factors to them,
`R_r = Q_H^H Omega_R Q_E` and `L_r = Q_E^H Omega_L Q_H` (k×k), and taking eigenpairs of
the small coupled matrix `[[0, L_r], [R_r, 0]]` gives per-column residuals ≤ eps·‖M_r‖ in
*both* relations at once — each column `(Q_E gE, Q_H gH, mu)` is a mode of a problem
within eps of the requested one. Two construction rules matter, both measured the hard
way:

- **Keep `eig`'s joint normalization** `‖[gE; gH]‖ = 1`. A mode may then carry a tiny
  E-part (a pure-H graze) or a tiny H-part (the classic E-graze) but never both.
  Normalizing the E-part to unit instead planted a `‖V‖ ~ 2.7e12` column whose
  eps-level equation residual (~5e-4 absolute) destroyed the solve.
- **Select the k modes (from 2k eigvecs) by joint E+H greedy independence**, not by
  ±eigenvalue pairing: eigvecs come in ±mu twins sharing one field pattern, and
  near-degenerate values make `eig` return arbitrary bases; greedy pairing stranded
  orthogonal leftovers and duplicated chains (condW 1e14, errors 1e-3).

**The wall: numerically defective clusters.** Every uniform-layer graze is a true
`q² = 0` double root; whether the *coupled* rates `mu` split usably is a round-off
lottery the one-sided formulations always win (their `q = sqrt(eig noise) ~ 1e-8` is
automatically self-consistent) and the two-sided one sometimes loses: stacks measured
with `mu ~ 5e-16` (theta-graze n=2.25; spacer n=1.5, where the restricted factors'
small singular values are ~1e-30) have **no exponential mode basis at working
precision** — the intrinsic error of any such basis is `~eps/(k0·d·|mu|)` (measured:
`mu = 2e-9` → 6.5e-9; `mu ~ 5e-16` → 1e-4..1e-3 across every basis tried). Clamping
`|mu|` up repeats §5's epsilon-floor dead end in miniature and was measured to do so
(spacer2.0 went from 6.5e-9 to 5.6e-3 when its natural `mu = 2e-9` was clamped): a `q`
inconsistent with its own mode pair, by any amount, is amplified by exactly the
conditioning that consistency lets cancel. So below `_DEFECTIVE_REL = 1e-10` (relative
to the largest modal rate) the rebuild **refuses loudly** with the detune advice — the
`{1, z}` fact of §5, now stated for a cluster instead of a single exact zero — and the
default ladder retries off the point. `_COUPLED_REL = 1e-9` is sized so the ladder's
`1e-12` step lands *inside* the rebuild's regime: with a narrower gate the rescued
solves fell back into the old construction's degraded band and returned 1e-6..8e-6.

**Validation** (all on the exact point unless said otherwise): five graze stacks
(spacer n=2.0; n=1.5 with and without the coincident bottom half-space graze; a
24-mode ky-graze at n=2.5, truncation_order=[5,5]; a theta-graze at n=2.25 with
`n_top·sin(theta) == 0.25` bitwise) × {ETM, SMM} × psi {pi/2, pi/4, 0} × {NumPy, JAX,
PyTorch}: every solve is **correct** (≤1e-6 of the extrapolated oracle, most ≤1e-8,
energy to ≤1e-9) **or loud**, and with `auto_detune` on (the default) every one of the
90 cells lands ≤1e-6. Exact vs one-ULP continuity ≤1.7e-9; field reconstruction exact
vs one-ULP ≤2.6e-8; `jax.grad` through the exact graze finite and nonzero (via the
fallback-then-ladder path — the rebuild itself steps aside for values it cannot
materialize). Off the condition: 40 random non-grazing solves, 0 rebuild engagements,
results bit-identical.

## 7. The conical x-cutoff (`ky != 0`) — not a Rayleigh point, and now solved exactly

`solve_1d.py::_cutoff_cluster` gates, `solve_1d.py::_regular_conical_modes` rebuilds,
`solve_1d.py::_regular_conical_modes_ad` differentiates. Tests: `tests/rcwa/test_conical_cutoff.py`.

**What it was.** On the conical path a layer's modes come in two families: `E_x = 0`
(normalised to `H_x`) and `H_x = 0` (normalised to `E_x`) — the decomposition of Moharam et
al. (JOSA A 12, 1068, 1995). L. Li proves it unique (J. Mod. Opt. 40, 553, 1993, appendix A)
except for a field meeting both conditions at once, which needs `kz**2 + lambda**2 = 0` in his
notation (his `kz` runs along the grooves, `exp(i lambda y)` across the layer): here
`q**2 = ky**2`, i.e. `wA = q1**2 - ky**2 = 0` — in a uniform layer, `kx_m**2 = eps*mu`. (`wA`
and `wB` are the eigenvalues of `A mu_x` and `B eps_x`, `A = Kx muz_i Kx - eps_y` and
`B = Kx epz_i Kx - mu_y`, the two families' matrices: family 1's `q1**2 = ky**2 + wA`,
family 2's `q2**2 = ky**2 + wB`, and `wB = 0` is the same condition met by family 2.) As a
mode approaches it, the component it is normalised to vanishes, so both families' columns grow
like `1/wA` and turn parallel, and the interface inverses amplify rounding by `~1/wA**2`. With
`ky == 0` the same zero is a Rayleigh point (§1–§5). With `ky != 0` it is not — the order's `kz`
is `abs(ky)`, nothing grazes — and the §1 nearby-problem argument does not rescue it: the error
band is wide (`C u / wA**2`, still 5e-5 at a relative 1e-8), so a nearby point is *not* the
answer.

**Uniform versus patterned.** Two different things share that one condition.

| | uniform layer | patterned layer |
| --- | --- | --- |
| the point | an ordinary double root: the two roots meet with equal slopes; only the *basis* collapses | a genuine 2x2 Jordan block: the roots cross `abs(ky)` with different slopes (ratio 0.92..1.36 measured), so no eigenvector basis exists there |
| `wA` at the point | often exactly `0.0` (noiseless FFT) | round-off, 1e-16..3e-15, never exactly zero |
| before | refused at an exact zero; silent one ULP off (2.4e-3..13 in `R` over eight scenes, `R + T - 1` up to +17), at rounded inputs (water `1.7689`: 2.8e-2, `R + T - 1` up to +0.8 on SMM), and the `auto_detune` retry landed 2.6e-2..0.1 off; SMM's `R + T` reached 2.4e3 over a census | never refused, never warned: `R` off by up to 11, `R + T - 1` up to 48 (68 offsets within relative 1e-4; ETM up to 2.0 in `R`), a lottery in the 13th digit of the wavelength; gradients off by 1e11..1e28 |

**The construction** (each layer whose gate engages; no `eig` beyond the two the layer already
had, no division by `wA` or `wB` anywhere):

- family-1 cluster modes re-normalised to their `E_y`: `b1_j = (wA_j/q1_j) f1_j`;
- family-2 cluster modes with the family-1 cluster's part *omitted*, not cancelled:
  `b2_k = f2_k - sum_j beta_jk f1_j`, `beta_jk = ky D_jk / q1_j`, `D = W1^-1 Kx muz_i W2`;
- the coupling this leaves, `L b2_k = q2_k b2_k + sum_j tau_jk b1_j`, is 0/0 as written and
  regular through the exact identity `wA_j N_jk = wB_k D_jk`:
  `tau_jk = ky (N_jk - D_jk)/(q1_j + q2_k)`, `N = W1^-1 mu_x^-1 epz_i Kx epx W2` (zero in a
  uniform layer, 5e-3..5e-2 in the patterned scenes);
- every other column through its *own* family (`V_21 = ky muz_i Kx mu_x W1/wA`,
  `V_12 = ky epz_i Kx epx W2/wB`), so no `1/wA_j` reaches a family-2 column — the family
  construction divided every family-2 column's rounding by every `wA_j`, which is how one cutoff
  order spoilt all of them.

The layer comes out in the general layout, `E = [S_x; S_y]`, `U = [U_x; U_y]`, with
`q = [[diag(q1), tau], [0, diag(q2)]]`. The coupling's rows and columns are disjoint, so
`modal_gradient.exponential`'s divided differences are its exact exponential (Parlett's
recurrence). It is an exact change of basis for any set of modes, so the gate decides only who
pays for it. The identities need `epz_i = inv(eps_y)` and `muz_i = inv(mu_y)`, the premise the
family split already rests on.

**Derivatives.** Under autodiff the value is the rebuild of the stopped inputs and the
derivative the first variation of the layer's first-order operator `L = [[0, P], [Q, 0]]` in that
fixed basis (`modal_gradient.py::_variation` on the `4n`-sized operator): the `++` block is `dq`,
the `-+` block tilts the `+` subspace through `T X + X T = S`, solved in four divided-difference
passes. "Every input" includes `wA` and `wB`: they are `w - ky**2` with a live `ky`, and left
live they gave the value a tangent of its own along any parameter that moves `ky` — the
`theta` derivative of `sum R` 0.50 off at a uniform cutoff, and 0.52 off inside the band where
the family construction had been right to 7e-13; stopped, the angle derivatives are the 2D
path's to 8.2e-14 on the tested scenes and to 2.9e-13 over nine uniform ones (magnetic and
double-negative spacers, backside, `theta = 1.4`, the incidence permittivity). Two consumers of a coupled `q` had to learn about it, both told
*structurally* by the code that rebuilt the layer, never by reading the value:

- `exponential(coupled=True)` takes the value from the formula and the tangent from an exact
  Fréchet derivative: eagerly the backend's matrix exponential's, under an abstract trace a
  closed form (below). The formula's own tangent misses the coupling chained behind a variation
  (6.4e-3 on `dR/dwl`). JAX's `expm` is pre-scaled by the ceiling of `log2(norm/theta_13)` and
  squared back: unscaled it was 4e-9 off at `norm = 10.5`.
- SMM's field state reads the rows that carry the coupling off the bottom face and adds, for
  every other kept row, the one term `two_face` would miss beside a coupling
  (`two_face(coupled=True)`). Plain `two_face` was 6.8e-3 off on a field gradient; dropping only
  the coupled rows without that term, 2.6e-2; reading the *whole* layer off its bottom face was
  exact at the patterned point but took the light-line protection away from every other mode of
  a rebuilt uniform layer — 1.5e-2 in the field at `theta = pi/3`, where order 1 grazes beside
  order 2's cutoff.

**The gate.** A mode is in when `abs(wA)` (or `abs(wB)`) is below
`g = min(0.04 * abs(ky)**1.5, abs(ky)**2 / 2)`, read once per layer through `_concrete` (under
an abstract trace there is nothing to read, and the masks travel as data: below).

- `0.04 * abs(ky)**1.5` (`_CUTOFF_REL`): the family construction's error follows
  `K0 u abs(ky)**3 / abs(w)**2` (`K0 <= 0.074` over twelve scenes), so at the gate's edge it is
  `K0 u / 0.04**2`, about 1e-14 whatever `ky` is. Measured just outside the edge, `ky` from 1e-4
  to 1.4: at most 9.1e-14 in amplitudes; inside it, the rebuild, at most 8.9e-14. A pure `ky**2`
  law left 2e-10 just outside it at `theta = 1e-4`; a fixed 1e-3 left 6.7e-13.
- `abs(ky)**2 / 2` takes over below `abs(ky) ~ 6e-3`. It keeps `abs(q) >= abs(ky)/sqrt(2)`, so a
  cluster never holds the `q -> 0` merge of an up- and a down-going mode, and it makes
  `ky == 0` (where `g = 0` and the comparison is strict) never engage: the Rayleigh contract,
  the TE/TM path and every `ky == 0` solve are untouched. A gate without it engaged at a
  `ky == 0` retry and scattered forward- and reverse-mode tangents by 4e-8.
- A cluster root under 1e-9 keeps the family construction. Members labelled up and down are
  not refused: the rebuild first puts every member on the deepest member's side
  (`solve_1d.py::_one_side`; *Patterned double negatives*, below).

Engaging where it was not needed costs time, never accuracy; failing to engage leaves the 1e-14
edge error. About 2% of random conical solves route; a solve with one rebuilt layer of two costs
1.03–1.06x.

**Measured** (oracles: the 2D path, a y-invariant raster, which never forms the families and is
exact at a uniform cutoff; and, at a patterned point, a degree-8 fit of the family construction
through 16 offsets at relative 1e-4..1e-3, where its own error is below 1e-12):

| | before | now |
| --- | --- | --- |
| five reported uniform scenes, exact point | refused; default 1.6e-8..0.10 off | `abs(dR) <= 2.9e-15`, amplitudes `<= 1.4e-14`, no warning, default = `auto_detune=False` |
| offset ladder, exact .. 1e-3 and one ULP | up to 0.78 in amplitudes | `<= 1.2e-14` in `R`, `<= 4.8e-14` in amplitudes |
| patterned PA616 / PB687 / PD574 / MU614 | `R` off by up to 11, `R + T - 1` up to 48 (SMM; ETM 2.0 in `R`) over 68 offsets within relative 1e-4 | amplitudes 1.4e-13 / 3.3e-13 / 1.4e-13 / 1.2e-12 (the last the fit's own accuracy); ETM = SMM |
| gradients, uniform (vs 2D `jax.grad`) | refused; default 1e4 off | raster 3.9e-14, wavelength 4.2e-13, thickness 4e-15; `jvp`, `jacfwd`, torch reverse and forward mode alike; `theta`, `phi` and the incidence permittivity 4.3e-15..2.9e-13 on `jax.grad`, `jvp` and torch, at the point and inside the band |
| `dR/dwl`, patterned (vs the fit's slope) | 1e11..1e28 off | 5.3e-11 .. 5.8e-9 (the fit), JAX = torch |
| fields, uniform (vs 2D) / patterned (SMM vs ETM) | 0.25..0.84 at the default | 1.2e-14 / 2.5e-15; field gradients SMM vs ETM 1.8e-14 |
| complex64, R1 (vs 2D complex64) | refused, and the 1e-12 retry cannot move a float32 | 3.8e-7 in amplitudes |
| `theta = pi/3`: order 2 at the cutoff, order 1 grazing | 0.30 in `R` | 3.8e-9 — a Rayleigh-type sensitivity of the grazing order, not the rebuild: 8.9e-13 at relative 1e-9 |

**Byte identity: eager only.** Every *eager* solve outside the gate is byte-identical. At the
fix, a 312-line `sha256` digest of `R`, `T`, `r_xyz`, `t_xyz` and fields over random conical,
magnetic, backside, `ky == 0`, TE/TM and 2D scenes on NumPy, JAX and PyTorch, with eager
`jax.grad`, field gradients, torch backward and forward mode, `jit` and `jit(grad)`, run twice
per tree, differed in 10 lines, all of them the three random scenes the gate routes. Two are a
lossy metal grating and a uniform layer inside the band: rebuilt, they sit 1.6e-14..2.1e-14 from
the 2D path, where the old construction sat 1.8e-14..5.1e-13 from it. The third, a field
gradient, moved 1.3e-14 and is 2.7e-13 from the 2D path's on either tree. The trace handling
below kept every eager bit: run twice per tree against the tree before it, all 212 eager lines
of a 232-line digest of that kind, and all 64 lines of an eager digest at the routed points
themselves (R1, PA616, MU614, glass over PA616 lit from below: NumPy, JAX and PyTorch solves and
fields, eager `jax.grad` in the wavelength, the raster and `theta`, a field gradient, torch
backward), are identical. Under a trace that holds for 2D, TE/TM and every other non-conical
solve, which the trace handling cannot reach — but no longer for a conical one: under any
abstract trace every conical layer takes the general layout, `ky == 0` included, so its outputs
are rounded differently, once (the 232-line digest's 20 jitted conical lines among them).
Re-baselined on a 378-line digest of 36 random off-band scenes (ETM and SMM; eager NumPy and JAX,
eager and jitted `grad`, `jit`, `jit(vmap)` over three wavelengths, eager and jitted fields,
`torch.func.vmap`), run twice per tree: every eager line and every 2D and TE/TM line is
identical (246 of 378), and every traced conical line moved — values by up to 7.0e-14 of their
largest entry
(conical + `mu`; 3.5e-14 patterned, 5.7e-15 backside, 4.4e-15 at `ky == 0`), fields 8.8e-14,
`jit(vmap)` 6.9e-14, `torch.func.vmap` 1.8e-14, `jit(grad)` 6.7e-13. They are as close to the
eager solve as they were: jitted patterned values 2.7e-14 from eager (5.6e-14 before),
`jit(grad)` 1.3e-13 (5.8e-13), fields 6.4e-14 (6.1e-14); conical + `mu` 8.9e-14 (3.0e-14).

**Under an abstract trace** — `jax.jit`, and just as much `jax.vmap` (the usual way to sweep a
wavelength), a `jax.lax.map` or `scan` body, `jax.checkpoint` (so `jax.grad(jax.checkpoint(f))`
too) and `torch.func.vmap` — the gate cannot read its values, and its masks decide the layer's
*structure*. So it does not decide there: `_cutoff_cluster` hands its masks on as traced data,
a refused cluster folded to empty masks, and *every* traced conical layer takes the rebuild in
the general layout. The rebuild is an exact change of basis for any masks, so an empty mask is
the family construction re-expressed. ETM, SMM's field state and field reconstruction already
read the layout structurally (`len(modes) == 3`, the record's `W.shape`), so no consumer, record
slot or pytree changed; a jit-built conical record is simply `2n x 2n`. `jax.grad`, `jvp`,
`jacfwd` and `jacrev` keep a concrete primal and are gated as an eager solve is, and so is a
solve under `torch.compile`, because wavebit runs eagerly inside a compiled caller
([`torch-compile.md`](../../development/torch-compile.md) §6) and the gate reads a real value. (This record first
read it as a graph break reading the value; that compile differentiated nothing, and a
value-only compile was right even before wavebit ran eagerly under it.)

The one piece that had to be new is the coupled exponential's tangent. Its eager rule, `expm`,
needs a squaring count; under a trace that count is traced, and undoing it takes a fixed scan of
conditional squarings, which `vmap` turns into selects that all run (with that tangent, a jitted
16-wavelength sweep cost 2.4–3.2x per call of what it did before, its gradient 3.4–5.9x). So
under an abstract norm
`exponential(coupled=True)` is a `jax.custom_jvp` (`modal_gradient.py::_jax_coupled_exponential`)
whose value is the divided-difference formula and whose tangent is a closed-form Fréchet
derivative of the triangular exponential (`modal_gradient.py::_coupled_frechet`). With
`A = s q = diag(a) + n`, the coupling `n` on disjoint rows `J` and columns `K`,
`exp(uA) = diag(e^{ua}) + n o exp_u[a_j, a_k]` exactly (the divided-difference form of a function
of a triangular matrix: G. Opitz, ZAMM 44, T52, 1964; C. Davis, Linear Algebra Appl. 6, 193,
1973), and `L(A, E) = int_0^1 exp((1-u)A) E exp(uA) du` (N. J. Higham, *Functions of Matrices*,
SIAM 2008, eq. 10.15) closes into

```
L_il = exp[a_i,a_l] E_il + sum_k n_ik exp[a_i,a_k,a_l] E_kl + sum_k E_ik n_kl exp[a_i,a_k,a_l]
     + sum_km n_ik E_km n_ml exp[a_i,a_k,a_m,a_l]
```

— the Daleckii–Krein kernel first. The divided differences are gathered one order from the one
below: by the recurrence over the widest pair, or, for a set within a diameter of 0.5, by the
Taylor series about its first point (19 terms, through degree 18; A. McCurdy, K. C. Ng and B. N. Parlett, Math.
Comp. 43, 501, 1984). Against a 40-digit `expm` of `[[A, E], [0, A]]` it is within 4e-16 of
`max|L|` across the series/recurrence switch, exact and 1e-12-near confluent pairs, a decaying
spread to `|s q| = 600` and complex `s` (SciPy's `expm_frechet`: 1.8e-16..2.8e-14). The scale
part of the tangent is `ds q exp(s q)`, which keeps thickness second derivatives and mixed
thickness × raster ones exact (1.1e-15 and 7.7e-14 from eager at PA616; without it every
thickness derivative through the layer is lost). Eagerly the tangent stays `expm`'s, so eager
gradients keep their bits; the two rules give eager gradients 2.4e-14 apart at the rebuilt test
points.

| under a trace (against the 2D path, or eager) | before | now |
| --- | --- | --- |
| R1 under `jit`, ETM / SMM: the point, one ULP and `1e-12` either side | NaN at the point; 0.27–0.34 / 0.52–0.78 | ≤ 3.7e-14 |
| R1 under `jax.vmap`, `jit(vmap)`, `jax.lax.map`, `jax.checkpoint` | NaN; 0.3 / 0.5–0.8 | ≤ 4.4e-14 |
| R1 under `torch.func.vmap` | NaN; 0.3 / 0.5–0.6 | ≤ 3.8e-14 |
| R1 `dR/dwl` through the point: `jit(grad)`, `jit(grad(sum vmap))`, `jit(vmap(grad))` | NaN | 1.7e-13–2.0e-13 |
| R1 / PA616 / MU614 `torch.autograd` and `forward_ad` through `torch.func.vmap`, wavelength and raster (against eager, per lane) | eigen-derivative missing (until `torch.autograd` through `vmap` carried it): 0.2–3.3 off in band, silent; `forward_ad` raised | ≤ 7.6e-13 |
| PA616 / MU614 values, `jit` and `jit(vmap)` | 3.9e-3–1.5 / 0.15–1.4 | ≤ 2.6e-15 / ≤ 3.6e-14 |
| PA616 `jit(grad)`, wavelength / raster / thickness | 1.0e12–2.4e22 / 2.9e12–3.2e24 / — | ≤ 1.6e-14 / ≤ 3.7e-14 / ≤ 1.2e-14 |
| MU614 `jit(grad)`, wavelength and raster / thickness; both under `grad(checkpoint)` | 1e12–2e25; 2e15–3e28 | ≤ 6.1e-13 / ≤ 2.0e-12; ≤ 4.7e-14 |
| fields under `jit` (five patterned scenes, R1) | 8.6e-3–2.5, NaN at R1 | ≤ 5.1e-14 (MU614; the others ≤ 8.6e-15) |
| field gradients under `jit`, raster and wavelength | 7e11–3e28, NaN | ≤ 1.2e-13 |
| jitted SMM field against jitted ETM field | 1.6–2.0 | ≤ 3.2e-15 |

Measured before, in other words: under `jit` NaN at an exact zero, 2.5e-2..7.3e-2 off in `R` one
ULP away, 3e-4..0.3 off at a patterned point, `jit(grad)` NaN or 1e12..1e24 off; under `vmap`
over the wavelength NaN at R1's point and 2.6e-2 off at `+-1e-12`; `jax.lax.map` and
`jax.checkpoint` alike (2.6e-2..2.7e-2), `torch.func.vmap` 2.5e-2; `jax.grad(jax.checkpoint(f))`
NaN at R1's point and 7.9e-7 off at relative 1e-6 — all silent. One trace now serves on- and
off-band wavelengths alike: a single compilation across the sweep.

**What it costs.** Every traced conical layer pays for the general layout, off the band too and
at `ky == 0`, which is not knowable under a trace. Per call against develop, off the band (R1 at
520 nm, and two 64-pixel patterned layers at truncation 10; ETM and SMM): `jit(solve)`
1.10–1.17x, `jit(grad)` 1.22–1.43x, a 16-wavelength `jit(vmap)` 1.01–1.20x and its gradient
1.12–1.46x, `jit(solve + fields)` 0.65–1.02x, a jitted field gradient 1.90–3.46x (the largest:
the closed form's second and third divided differences batched over `z`), `torch.func.vmap`
1.02–1.08x, compile 1.11–1.96x, XLA scratch memory 1.1–1.45x. At truncation 40 the ratios are
the same (an 8-wavelength `vmap` gradient 1.20–1.37x, 275 MiB of scratch against 228): the
eigensolves dominate, not the `4n`-sized solve of the modal variation.

**What stays as it was.**
- Nothing refuses under a trace: `check_finite` and `auto_detune` are skipped there, as before.
- A refused cluster (a root under the 1e-9 floor) gets empty masks: the
  family construction's arithmetic in the general layout, NaN at an exact zero included, silent.
- `ky == 0` at a Rayleigh point under a trace: NaN, unchanged (the `check_finite` gap).
- More than 4 coupled rows or columns in one layer (`_COUPLED_CAP`), or a coupled row that is
  also a coupled column, gives a NaN tangent, the value unaffected. A uniform layer couples
  none, every patterned point measured couples one, and a rebuilt layer's rows and columns are
  disjoint by construction.
- Second derivatives through the modal operator still raise, as do `torch.func.grad` and `jvp`;
  `torch.compile(fullgraph=True)` raises `Unsupported`, now at wavebit's first door
  ([`torch-compile.md`](../../development/torch-compile.md) §6).
- Not measured: GPU, `pmap` and `shard_map`.

**Why not a `lax.cond`.** A design that keeps the family construction off the band and the
rebuild on it, chosen at run time, was built and measured beside this one. Under `vmap` the
predicate is batched, the `cond` becomes a select and both branches run: 2.7–3.6x per call on a
16-wavelength sweep and 3.5–5.9x on its gradient with `expm` in the rebuilt branch, 1.25–1.45x
and 1.19–1.69x with the closed form, at 1.3–5x the memory (an 8-wavelength gradient at
truncation 40 needed 1.1 GiB of XLA scratch, against 228 MiB). The byte identity it would buy is
partial — its `jit(grad)` bits moved anyway, the `cond`'s presence in the graph alone moves them
— and it needs a traced record slot, so eager- and jit-built records differ in pytree structure.
And it is a trap for later: a `custom_jvp` whose tangent solves a system (a
`custom_linear_solve`, as `modal_gradient.py::_variation` does), placed inside a branch under
`vmap`, raises `TypeError` when differentiated outside the `vmap` (jax 0.11.0); that design had
to respell the modal variation to get past it.

A record built *eagerly* keeps its layout under any trace — `W`'s shape says so — and its
fields differentiate alike under `jax.jit`: passed in, its `q` is a tracer, so the coupled
exponential takes the closed-form tangent (9.9e-17 from eager on a field's `z` derivative at
PA616). Before, `_expm` formed jax's ceiling squaring count traced and ran a scan of conditional
squarings (4e-16..8e-16); before that, falling back to jax's own floor count put 6.4e-11 there.

**Patterned double negatives.** The following measurements are the 2026-10-12 DNG design
record against baseline `db37dbb8`. They have exceptional points of their own at a real angle.
Root-found on the layer's own matrices: six in four lossless layers (double-negative pillars in
silicon; glass pillars in a double-negative host; air beside `(-1, -3)`; glass beside `(-6, -2)`
at `phi = 1`), the pair's slopes in the ratio 0.76..1.0; none in 350..1600 nm in three milder
ones (glass or air beside `(-2, -1.5)` or `(-4, -1)`). Their flux labels
(`conventions.py::power_down_labels`) read the `1/(w - ky**2)` quotient, but at a real `ky` no
label rule can split a cluster. The gate holds `abs(w) <= ky**2/2`, so every root inside it lies
within `pi/12` of `+-abs(ky)`: decay-dominated, with a real part of at least `0.68 abs(ky)` --
with the 1e-9 root floor, far from the flux tie-break's 1e-12 -- and every rule (the decaying
root, Maystre's, its mirror, the power-down rule, the cap) takes the decaying root. A census
agrees: 380 random scenes with double-negative regions (lossless, lossy, beside a metal or a
gain region), 210 gate calls with a nonempty band on ETM and SMM, no split label. The rebuild is
exact at all six points: amplitudes 1.2e-13 .. 6.6e-12 from the fit at the four where the fit is
good to 2e-11, and 7.6e-11 and 1.9e-9 at the two where the fit itself is only 4.9e-10 and
1.4e-9; the family construction there was 0.18 .. 99 off in `R`.

A complex `theta` does split them. With `phi = pi/2` the cutoff does not move with `ky`, so the
pair sits at `q = +-ky`, and once `ky` is propagation-dominated the power-down rule labels each
mode by its flux: on ETM a band member that the flux sends the other way (at `abs(ky) ~ 1.5` the
band reaches the neighbouring crossing's mode); on both algorithms every root at a purely
imaginary `theta` (the tie `Re(q) = 0`, broken by the flux); and the cutoff pair itself, a
relative 1e-12 off the point. The gate refused each, and the family construction answered
silently (beyond the `ComplexIncidenceWarning` every complex angle carries): `R` 1.1..24 off at and beside the point (ETM), 0.48..1.4 at an imaginary `theta` on
SMM, 3.3e-5..7.4e-4 where only the pair split; under `jax.jit`, `jax.vmap` and
`torch.func.vmap` 3e-5..12, wavelength derivatives 9e6..2e26 relative, and the SMM field
4.7e-4..6.2e6 from the ETM field (relative to its largest entry). A split cluster is
now put on one side before the rebuild (`solve_1d.py::_one_side`): every member takes the root
on the side of the deepest member's own label. That is a change of basis, so it moves no exact
answer, and where no member disagrees `where` hands back the same bits: a 348-line digest
(eager NumPy, JAX and PyTorch, gradients, `jit`, `jit(grad)`, `jit(vmap)`, and this section's
routed points), run twice per tree, is identical to the tree before. At 16 split cases the
answer is within the fit's own residual (amplitudes at most 2.1e-8 where the fit is 1.5e-7,
elsewhere at most 2.2e-9) and ETM = SMM to 2.2e-11; at five of them NumPy, `jit`, `jit(vmap)`,
`torch.func.vmap`, `jax.grad`, `jit(grad)` and torch backward agree, the derivatives 3.6e-9 ..
7.8e-7 relative from the fit's slope (the fit's accuracy); the SMM and ETM fields agree to
2.1e-11; and at 3, 10 and 30 um of thickness (two scenes) the answer stays within the fit's
residual and ETM = SMM to 2.3e-10: the flipped roots cost ETM nothing measurable.

**Other paths at the same point.**

- The 2D path now rebuilds regular pairs (§8). Against baseline `db37dbb8`, the re-measured
  amplitude errors before the fix were 4.2e-8..5.4e-3 on ETM and 8.2e-7..3.6e-2 on SMM.
- First order now rebuilds regular pairs (§9). Baseline SMM errors were 3.0e-2..2.45,
  ETM 2.1e-7..2.5e-4, with `R + T - 1` up to +5.4. Neither path fails at a uniform cutoff.
- Patterned double negatives, with historical `R` errors up to 24, are covered by the
  one-side relabeling above.

## 8. The 2D path at an exceptional point — a near-defective pair, rebuilt

`solve_2d.py::_defective_pairs` gates, `solve_2d.py::_rebuild_defective_pairs` rebuilds,
`_pair_generator` / `_pair_inverse` / `_pair_sylvester` differentiate. Tests:
`tests/rcwa/test_2d_exceptional_point.py` and `tests/rcwa/test_exceptional_point_clusters.py`.

The measurements below are the 2026-10-12 prototype and judge record against baseline
`db37dbb8`, before the source-layout and FDTD integration changes. They describe those measured
trees, not a new run on the current checkout -- except the paragraph "Genuinely 2D EPs on both
formulations, re-measured", which is a run on `aadbc351`.

**What it was.** The 2D path diagonalises each layer's product `Omega2 = Omega_L Omega_R`. Where
two of its eigenvalues form a 2x2 Jordan block (an exceptional point, EP) `eig` returns two
columns that are each accurate to the unit round-off `u` but meet at a sine `s` of
1e-8..1e-6: `(Omega2 - w) ` has one singular value at 1e-16 and the next at 4e-3..7e-2, and the
eigenvalues split by `sqrt(u)`-sized amounts (1e-9..5e-8). Each column still satisfies both
first-order relations to 1e-15; what breaks is the basis: `cond(W)` 6e6..1.6e8, and the
interface inverses amplify rounding by about `1/s**2` -- measured `K u / s**2` with `K <= 1e-2`
(SMM; ETM 1e-5..1e-2) for `s` from 2e-5 to 1e-2. The grazing rebuild (§6) never engaged: it is
keyed to `min|w| / max|w| < 1e-9`, and these pairs sit at `|w| ~ 1e-1..7`.

Two kinds were measured, both silent before:

- **§7's exceptional points on a y-invariant raster.** `wA` does not depend on `ky`, so a 2D raster
  has one EP per order row at once, each at `q = abs(ky_v)` (three on truncation `(M, 1)`). Against
  the landed 1D rebuild, v = 0 row: amplitudes 4.2e-8..5.4e-3 (ETM) and 8.2e-7..3.6e-2 (SMM),
  `R + T - 1` to -2.1e-2 (MU614 at relative 1e-12); `|d sum R|` at the exact point, ETM, 1.2e-8..4.4e-8
  on PA616 / PB687 / PD574 and 2.1e-6..2.2e-6 on MU614 / PL952 (the "1e-8..4e-5" recorded in §7
  was `|d sum R|` at the point plus PL952's spread); gradients 2e-8..6e-2, fields 2.5e-7..5.1e-2;
  under `jit`, `vmap` and `torch.func.vmap` the same. On the lossless double-negative EP
  `Si 1558`, SMM amplitudes 5.2e-7 at relative 1e-5; at complex `theta` up to 9e-2 in `R`.
- **Genuinely 2D exceptional points.** Cutting the PA616 pillar to a fraction `fy` of the y period
  breaks the y-invariance and the families, and the layer's real `q**2` pairs turn complex at
  EPs of their own -- codimension one in the wavelength for a lossless layer: 8, 5 and 6 of
  them within +-3% of 616 nm at `fy` 0.95, 0.8 and 0.5, every one with `q**2 > 0`. Against an oracle
  that never runs the rebuild (below), the product construction was 4e-11..1.7e-6 off (ETM) and
  2.5e-9..6.4e-6 (SMM). Their error band is narrow -- `s ~ sqrt(|d wl|)`, so the error falls like
  `1/|d wl|` (6.4e-6 at the point, 2.2e-9 at 1e-12, 2.6e-12 at 1e-9) -- where §7's falls like
  `1/d**2`.

**The gate.** Two modes are a pair when each is the other's nearest eigenvalue of `Omega2`, their
eigenvector sine `s = |(I - v v^H) W_b|` is below `_EP_SINE = 1e-2`, and both roots are above
`_EP_ROOT_FLOOR = 1e-9` (`q -> 0` is §6's). Measured just outside the edge (`s` 1e-2..2e-1), the
product construction is at its floor, 4e-15..1.7e-13; inside, the rebuild is 2e-15..5e-14. Over
400 random 2D layers the gate engaged on one (smallest sine 6.6e-3), over 100 C4-symmetric layers
at normal incidence (exactly degenerate pairs) on none. Concrete decisions go through
`_concrete`; under a trace the masks remain data.
A pair is also skipped if an *unpaired* root is within `_EP_SEPARATION = 1e-7` times
`max(1, |w|)` of either member. The Newton step divides a plain row by that separation:
synthetic residuals were 2.0e-12 at 1e-8, 3.5e-13 at 2e-8, and at most 8.1e-14 from 5e-8 on.
Members of another gated pair are exempt because their rows use the coupled Sylvester solve;
that solve was accurate to 2.4e-14 down to a pair-to-pair spread of 1.5e-6 for two distinct
pairs. A 1e-4 guard on all third roots was rejected because it skipped pairs that could be
rebuilt exactly. The solve is not accurate for two pairs that are copies of one problem in
decoupled order rows (exceptional points that nearly coincide): their split is ill-posed, and
at spreads up to 2.2e-6 (PL952, truncation 20) they came out silently up to 1.0e-9 off. Those
layers were screened after the rebuild (*Nearly coincident pairs*, below), and are now rebuilt
as one cluster (*Defective clusters, rebuilt jointly*, below); the screen remains for clusters
the joint rebuild does not take.

**The construction** (values only, inputs stopped):

1. *Orthonormalise.* `v = W_a / |W_a|` -- an eigenvector to `u` -- and `x0 = (I - v v^H) W_b / s`,
   a direction accurate only to `u / s` (the span `eig` hands back is not backward stable: its
   invariant-subspace residual is `u / s`, about 1e-9 here; built on it the answer was 8e-10 off).
2. *One Newton step for the invariant subspace, every pair at once* (G. W. Stewart, SIAM J.
   Numer. Anal. 10, 752, 1973; J. W. Demmel, Linear Algebra Appl. 79, 163, 1986). In the basis
   `W0` -- the modes with each pair orthonormalised, well conditioned -- the operator is already
   (block) diagonal: `M = Q^H Omega2 Q` is upper triangular to `u` (`M_10 = x0^H Omega2 v`
   is dropped), the residual `R = Omega2 Q - Q M` expands as `R = W0 C` by one solve for every
   pair, and the correction `Q -> Q + W0 Y` solves `Lambda Y - Y M = -C` row by row: a plain row
   `y (lambda_i - M) = -c_i`, another pair's two rows the 2x2 upper-triangular Sylvester equation
   with that pair's `M`, its own rows zero. Measured: the eig span's 1e-9 lands at rounding
   (invariant-subspace residual <= 6e-16). Shifted inverse iteration was measured first and
   rejected: on a Jordan block it collapses the pair onto its eigenvector (1e-14 after two steps,
   1e-2 after three).
3. *The pair's modes.* Re-orthonormalised to `(v, x)`, `T = Q^H Omega2 Q` is upper triangular to
   `u` (`T_10` dropped), and `q = [[alpha, t], [0, beta]]`, `alpha**2 = T_00`, `beta**2 = T_11`,
   `t = T_01 / (alpha + beta)`, is its exact square root; `W = (v, x)` and `V = Omega_R W q^-1`
   (`V_a = Omega_R v / alpha`, `V_b = (Omega_R x - V_a t) / beta`) satisfy `Omega_R W = V q` and
   `Omega_L V = W q` to 6e-16, and `cond(W)` falls to 4..170. No division by a gap.
4. *One side.* Both roots take the side of the more decaying member's label -- each member's own
   wherever the layer's rule agrees with itself. Where it does not (a complex pair on Maystre's
   cut, a tie the flux or `Im(q) <= 0` breaks: purely imaginary `theta`, the double negatives of the
   patterned double negatives at complex `theta`), the other member turns, so the `+` set holds
   the whole Jordan block and stays complementary to the `-` set.

The layer comes out with `q = diag(q) + tau`, `tau[a, b] = t` per pair: coupled rows `{a}` and
columns `{b}`, disjoint, so `modal_gradient.exponential`'s divided differences are its exact
exponential (§7), and the ETM, SMM and field reconstruction consume it as they consume §7's
general layout. The consumers are told structurally: `layer_modes_2d(..., with_coupling=True)`
returns the word to the ETM and SMM loops; an ETM record gets a tenth slot (`None`, so a pytree
round trip keeps it) and SMM's records likewise.

**Derivatives.** In the fixed basis `W` the layer is the matrix `Q` with `Q**2 = W^-1 Omega2 W`,
and its first variation solves `Q dQ + dQ Q = W^-1 dOmega2 W` -- `modal_gradient.generator`'s
rule with a coupled `Q`: the four divided-difference passes of `_coupled_sylvester`, the
coupling's products taken by gathers (`_pair_sylvester`). `V = Omega_R W Q^-1` takes the closed
form's value and the exact tangent `-Q^-1 dQ Q^-1` (`_pair_inverse`: the formula's own tangent
drops `M dQ_off D^-1 + D^-1 dQ_off M - M dQ M`, `M = D^-1 tau D^-1`). The exponential is the
coupled one (`exponential(coupled=cap)`): eagerly `expm`'s Fréchet derivative, under a trace the
closed form of §7 with the pairs' cap. SMM's field state reads both members of a pair off the
bottom face (`two_face(coupled='pairs')`): a 2D pair's tangent `d_ba` is not tied to its gap
the way a conical cutoff pair's is.

**Under an abstract trace** every 2D layer takes the coupled layout with its pairs as data: the
`cap = max(n_orders_u, n_orders_v)` smallest sines (an invariant raster's EPs come one per order
row or column), masked; a lane whose pairs are masked keeps the product construction's columns
and a zero coupling. More qualifying pairs than the cap multiply the layer's `q` by NaN -- loud.

| measured (oracles: the 1D rebuild for §7's points; a Cauchy circle for the genuinely 2D ones) | before | now |
| --- | --- | --- |
| PA616 / PB687 / PD574 / MU614 / PL952 rasters, `rel` 0, +-1e-15, +-1e-12, +-1e-9 .. +-1e-3, amplitudes | 4.2e-8..5.4e-3 (ETM), 8.2e-7..3.6e-2 (SMM) | <= 4.5e-14 for `|rel|` <= 1e-9 and <= 1.0e-13 to 1e-3 (PB687 SMM at -1e-3, where develop gives the same 1.0e-13); `R + T - 1` <= 7.2e-15 and <= 4.1e-14 |
| genuinely 2D EPs, `fy` 0.95 / 0.8 / 0.5 (six points), amplitudes | 4e-11..1.7e-6 (ETM), 2.5e-9..6.4e-6 (SMM) | 5.8e-15..1.6e-13 |
| complex `theta`: patterned double negatives at 18 angles (fits of the 1D family construction) and PA616 at three | `R` up to 9e-2, amplitudes up to 3.4e-2 | within each fit's residual; PA616 <= 9.4e-13 |
| `jax.grad`, `jvp`, torch reverse, `forward_ad` (raster, wavelength, thickness, `theta`) and `jit(grad)`, vs `jax.grad` of the 1D rebuild, `rel` 0 .. 1e-9 | 4.7e-8..6.2e-2 | <= 2.2e-13 (MU614 <= 6.0e-12, the 1D path's own consistency there) |
| `jit`, `jit(vmap)`, `torch.func.vmap` values; `jit(grad(sum vmap))`, `jit(vmap(grad))`, torch `vmap` + backward and `forward_ad`, per lane | 2e-8..3e-4; 2e-6..4e-3 | <= 4e-14; <= 1.2e-12 |
| fields (six components) / field gradients, eager | 2.5e-7..5.1e-2 / 2.3e-6..1.9e-2 | <= 1.2e-13 / <= 2.6e-13 (the double negative `Si 1558`: 9.1e-13 / 3.1e-12) |
| fields / field gradients under `jit` | 8.1e-7..2.4e-4 / 3.1e-6..2.3e-3 | <= 5.8e-14 / <= 2.6e-13 |
| complex64, against the 1D rebuild at complex128 | 4e-6..9e-4 | 5.4e-7..2.0e-6, the 2D path's float32 floor off the band (5.8e-7..1.6e-6) |
| uniform cutoffs (`test_conical_cutoff.UNIFORM`, 13 scenes x exact, +1e-12, one ULP) | exact (2.3e-15..5.2e-14) | the same numbers: the gate never engaged, so the arithmetic is the product construction's |

The Cauchy oracle: the Cartesian amplitudes are analytic in `theta`, so their value at `theta*` is
their mean over `theta* + r exp(2 pi i k / K)` (the trapezoidal rule on a circle, geometrically
convergent in `K`: L. N. Trefethen and J. A. C. Weideman, SIAM Rev. 56, 385, 2014, Sec. 12); on the circle the pair is off its EP and the product
construction is accurate. It reproduced direct solves off the EP to 2e-14, and itself across
`r` 1e-4 / 2e-4, `K` 16 / 24 and ETM / SMM to 7e-14. (It cannot judge §7's points: there `wA`
does not depend on `theta`, so the whole circle sits on the EP.)

**Historical byte identity.** A 404-line digest (`sha256` of `R`, `T`, `r_xyz`, `t_xyz` and
fields) over 40 random scenes -- 2D Li rasters, with `mu`, backside, NVF, C4-symmetric at normal incidence, lossy and
metallic, first-order on 2D, conical with `mu` -- on NumPy, JAX and PyTorch eagerly, with eager
`jax.grad`, `jvp`, torch backward and `forward_ad`, eager field gradients, `jit`, `jit(grad)`,
`jit(vmap)` and `torch.func.vmap`, run twice per tree: each tree repeats itself, and develop and
the prototype differ in 22 lines -- the 12 routed EP lines and 10 `jit(grad)` lines. Every
*off-band* eager line and every *off-band* traced value line (`jit`, `jit(vmap)`, `torch.func.vmap`) is identical. The moved
`jit(grad)` lines are as accurate as before: against eager `jax.grad` on the same tree, 1.9e-16..
1.2e-10 on both trees, line for line.

**Historical prototype cost.** These timings precede the cluster screen below. Eagerly the gate
is `O(n**2)` per layer: off the band 1.00..1.01x (NumPy, two layers at truncation (5, 5) and (8, 8)). A rebuilt layer adds one solve with `W0` and a few
`2n x 2 * pairs` products: 1.06..1.15x on PA616's rasters. Under a trace every 2D layer pays the
rebuild (truncation (8, 8), two layers, two runs per tree): `jit(solve)` 1.22x, `jit(grad)` 1.07..1.11x,
an 8-wavelength `jit(vmap)` 0.93..1.17x and its gradient 0.97..1.08x, `torch.func.vmap` 0.95..1.14x;
compile 1.3..1.5x.

**The unresolved-cluster screen.** Nearly parallel *pairs* do not detect every defective
cluster. At `ky0 = Gy/2` a y-invariant raster has two coincident Jordan blocks: `eig` mixes
four vectors whose pairwise sines are 0.76..1.0, while the four-column smallest singular value
is about 1e-7. The pair gate then sees nothing. Eagerly, `_unresolved_defective_cluster`
checks the final basis after any rebuild, on stopped values:

1. Group roots connected by a distance at most `_EP_CLUSTER = 1e-4` times the larger
   `max(1, |w|)`. If `_concrete` cannot read the presence of a cluster of at least three modes,
   the screen returns without warning; if none exists, one quadratic comparison is all it does.
2. For each such cluster, normalise its columns and inspect their smallest singular value.
   A value at least `_EP_SINE` needs no further check.
3. Otherwise form an orthonormal basis `Q`, then `T = Q^H Omega2 Q`, and measure
   `nu = ||T T^H - T^H T||_F / max(1, |mean(diag(T))|)**2`. Warn with
   `ExceptionalPointWarning` when `nu > 1e-10 * (eps/eps64)**2`.

This measures non-normality, so a diagonalizable degenerate eigenspace is not confused with a
Jordan block just because `eig` returned a poor basis. Across the judge's 195 ordinary
clusters, `nu <= 1.9e-19`; across 429 exceptional clusters, `nu >= 1.4e-7`. Measuring
`||T - mean(diag(T)) I||` instead failed on two ordinary degenerate pairs whose roots were 4.5e-5
apart. The precision scaling suppresses the screen below double precision: a fixed float64
threshold gave four false warnings in 60 complex64 uniform first-order layers; the scaled
threshold gave none. The screen changes no arithmetic, and `auto_detune` does not respond to
`ExceptionalPointWarning`: its 1e-12 step cannot clear this wider band.

**Nearly coincident pairs.** Just off a coincidence the cluster splits into two gated pairs a
few 1e-8 .. 2e-6 apart. Both are rebuilt and their final columns are no longer nearly
dependent, so the screen above passes them. The rebuild is not exact there. The two pairs are
copies of one problem in two decoupled order rows, so how their joint four-dimensional subspace
splits into two pairs is ill-posed: `eig` hands back pairs that mix the rows, and the one Newton
step, whose Sylvester rows between the pairs divide by their gap, lands on a split chosen by
rounding. On the scenes below the answer was silently off by up to 1.7e-10 at truncation 5 and
1.0e-9 at truncation 20, in a band whose edges moved with the CPU's arithmetic path and widened
with the truncation (PL952 out to `|dtheta| = 1.5e-6` at 20). On one GitHub runner the band
reached `|dtheta| = 1e-7`, a point of the test grid.

Eagerly, for each cluster holding two or more rebuilt pairs, `_crowded_pairs_imprecise`
measures the backward error of the layer model its modes define,
`||(A W_c - W Q^2[:, c]) W^-1[c, :]||_2` over the cluster's columns `c`, against the cluster's
scale `max(1, |w|)`. It needs one solve with `W^T` for the rows `W^-1[c, :]` and a QR of that
block for the norm. Above `_EP_BACKWARD = 3e-12` (scaled by `eps/eps64`) it warns with
`ExceptionalPointWarning`. Where the cluster's propagation sensitivity
`S = k0 d / (2 min|q|)` exceeds `_EP_SENSITIVITY = 3.5`, the threshold is divided by
`S / 3.5`: a miss in `q**2` moves the phase across the layer by about `S` times as much. `k0 d`
is the layer's run optical thickness, the one the ETM labels read; SMM, which takes no labels
from it, computes the same value for the screen alone, so the two screen alike.

Measurements (NumPy 2.5.4 with OpenBLAS 0.3.34; CPU arithmetic paths selected with
`OPENBLAS_CORETYPE` and `NPY_DISABLE_CPU_FEATURES`):

- *Candidates, truncation 5.* PL952 and PA616, 2D, ETM and SMM, 32 offsets of either sign,
  seven paths; 710 two-pair layers. No candidate separates wrong from exact answers: each puts
  exact answers among the wrong ones (PA616 was exact at gaps where PL952 was 7e-12 off). A
  screen needs a quantity that bounds the wrong answers from below with room to spare, at one
  threshold. The pairs' gap does so one-sidedly at truncation 5, every silent wrong answer
  within 8.5e-7 (PA616's within 8.5e-8), but a threshold of 1e-6 leaves a margin of 1.2 and the
  bound moves with the truncation (2.2e-6 at 20). The Sylvester separation, the Newton
  correction, a second step's correction and the rebuilt cluster's smallest singular value each
  failed even that on at least one path. The residual `||A W_c - W_c Q^2||` without `W^-1`
  reached 7e-11 of `||A||_2` on an exact answer (`|dtheta| = 2e-6`). The backward error bounded
  the wrong answers from below on every path; exact answers reached it too, which is why the
  screen warns on many of them.
- *Its scale, truncations 5 .. 20.* Each scene at its own exceptional-point wavelength at
  truncations 5, 10, 15 and 20; 36 offsets, ETM and SMM, four paths; 2304 solves. Read
  relative to `||A||_2`, which grows with the truncation through the evanescent orders
  (11.6 .. 366 here), the threshold that separated them at truncation 5 left 14 answers up to
  3.4e-12 off silent at 15 and 20. Against the cluster's scale, every answer more than 1e-12 off
  that the cluster screen let through missed by at least 1.43e-11, at every truncation: a factor
  4.8 above the threshold.
- *Its sensitivity.* Those layers were 200 and 300 nm thick with cluster roots `|q| >= 0.44`,
  `S <= 3.47`. PA616 thickened to 1, 3 and 10 um with the coincidence moved to `ky0` = 0.02 ..
  0.1 (`|q|` down to 0.02, `S` up to 2.6e3) was up to 1.1e-11 off under a miss below 3e-12.
- *With the screen.* Those 2304 solves, and at truncation 5 seven paths over 34 offsets that
  include the test grid (952 solves): no answer more than 1e-12 off was silent, the worst quiet
  answer was 1.01e-13, nothing warned at `|dtheta| >= 3e-4`, and ETM and SMM never disagreed on
  loudness. The sensitivity scaling leaves every one of those decisions as it was.
- *Beyond those scenes.* 74 scenes on two arithmetic paths (11,720 solves), each at its own
  exceptional-point wavelength: PD574 at truncations 5, 6 and 10 and the magnetic MU614 at 5 and
  10; y periods 500 .. 1300 nm; cluster roots above 1 (scale up to 12.25); thicknesses 30 nm ..
  10 um; a two-layer stack; small roots (`ky0` 0.003 .. 0.3). No answer more than 1e-12 off was
  silent (worst quiet 3.7e-13), none crashed, and ETM and SMM never disagreed on loudness. The
  sensitivity scaling raised 696 of the 2904 new warnings on its own: 18 on answers more than
  1e-12 off, 526 on answers within 1e-13.

The price is warnings on exact answers: 512 of the 1060 new warnings in the truncation sweep
fell on answers within 1e-13. Both scenes warn out to `|dtheta| = 1e-5` (PA616 at truncation 5
to 2e-6). In the broad sweep's thick layers with small roots (`S` in the hundreds or
thousands) it warned on every cluster holding two rebuilt pairs within `|dtheta| <= 1e-5` of the
coincidence, exact or not, and on some out to `|dtheta| = 1e-3` (566 of 1186 such clusters over
offsets out to 3e-2). Its advice, to move the angle or wavelength by a relative 1e-4 and
compare, then still meets the warning on exact answers. The threshold is
conservative on purpose. In these scenes the part of the backward error that couples the two
order rows does not reach the answer, because the second row is not excited; a layer without
that symmetry would not be so forgiving, and none was available to measure. Rounding alone
contributes about `eps * ||A||_2` to the miss, so in a layer whose operator norm is many
thousands of times the cluster's scale the screen may warn on an exact rebuild (not measured). Rebuilding the two pairs' subspace jointly removes the gap division
(*Defective clusters, rebuilt jointly*, below); since then the screen reads only pairs outside
the clusters that rebuild takes, and the measurements above describe the screen as it stood
alone on main `ea57624b`.

On the test grid (0, 1e-9 .. 1e-3) at truncation 5, on all seven paths, the 2D path with the
screens alone was exact or warned. The cluster screen warned for `|dtheta| <= 1e-8` (PA616) and `<= 1e-7` (PL952). The
backward-error screen warned at PA616 1e-7, which was up to 2.0e-12 off and silent without it,
and at PL952 1e-6 and, on most paths, PA616 1e-6 and PL952 1e-5, which were exact. Every quiet
grid point was within 1.7e-14. The cluster screen's earlier measurement found no false warnings
on 120 degenerate scenes or on the 200-scene census. The backward-error screen reads only
clusters holding two or more rebuilt pairs; across the RCWA test suite's 4,451 eager 2D layer
solves, only the two exceptional-point test files held two or more gated pairs. Rebuilding
can still select a diagonalizable pair with a poor eigenvector basis: across both paths,
10 of 800 census solves moved by at most 1.2e-15 in `sum R` or `sum T`. This qualifies the
digest claim: byte identity applies outside the gates, not to every non-exceptional layer.

**Defective clusters, rebuilt jointly.** `solve_2d.py::_defective_clusters` gates,
`solve_2d.py::_cluster_split` splits, `solve_2d.py::_rebuild_defective_clusters` rebuilds (called
through `_rebuild_defective_pairs`); its core, the Newton and chord steps, the split and the check,
is `solve_2d.py::_joint_cluster_bases`, which the first-order split calls too (§9, since
2026-11-11). Tests: `tests/rcwa/test_exceptional_point_clusters.py`.

*The gate.* A cluster is a ball of `Omega2`'s eigenvalues: the modes within a radius (relative to
`max(1, |w|)`) of its lowest-index member, read at two radii. Balls of `_EP_JOINT_RADIUS = 3e-3`
come first; where none of those is taken, balls of `_EP_CLUSTER = 1e-4`, the unresolved screen's
radius, and a narrow ball inside a wide ball that is taken is left to the wide one. The wide level
takes the degenerate doublets off a symmetric coincidence (the C4v pillar in *Measured*, below),
which split by more than `_EP_CLUSTER`; the narrow level takes clusters closer than 3e-3 to other
clusters or pairs, whose wide ball holds more than four modes. A ball is taken when it is closed --
every member's own ball is the same set, so no chain runs out of it -- holds three or four modes
(`_EP_CLUSTER_SIZE = 4`) with roots above `_EP_ROOT_FLOOR`, its unit columns are nearly dependent
(smallest singular value below `_EP_SINE`, read from the Gram matrix's smallest eigenvalue
clamped at zero, since rounding leaves a rank-deficient Gram's below it), and its restriction
`T = Q^H Omega2 Q` is not normal (`nu > _DEFECT_NU`, the unresolved screen's measure). A ball
of exactly three holding one gated pair stays with the pair rebuild, which is exact for a pair
beside a root more than `_EP_SEPARATION` away. So the gate takes coincident exceptional points
(the pair gate sees none there), two gated pairs a few 1e-8 .. 1e-3 apart, one gated pair beside
two ungated modes, a pair with an unpaired root within `_EP_SEPARATION`, which the pair gate
refuses, and two degenerate doublets up to 3e-3 apart whose columns stay nearly dependent.
Eagerly a layer with no ball of three or four at either radius returns after one quadratic
table of its eigenvalues' distances, its arithmetic untouched. Under an abstract trace,
`max(1, min(pair_cap // 2, n2 // 3))` slots hold the qualifying balls of both radii with the
smallest singular values, their masks as data; more qualifying balls than slots multiply the layer's `q` by NaN,
as a pair overflow does. The defect measure is read only on the slots, so a degenerate ball in a
nearly dependent `eig` basis counts toward that overflow too: a trace can return NaN for a layer
from which the eager solve takes no cluster (loud; not met on any layer measured, and the
uniform layers' degenerate balls have orthonormal columns, which the singular-value test drops).

*The subspace.* One Newton step for each cluster's invariant subspace, in the basis `W0` with
every pair and every cluster orthonormalised (Stewart 1973, Demmel 1986, as for a pair). A
cluster's restriction `M_c` is a dense `k x k` block, and its rows in another block's correction
solve `k x k` systems: a plain row `y (lambda_i - M_c) = -g_i`, a pair's two rows the triangular
Sylvester system with that pair's `M`, another cluster's rows the Kronecker form of
`M_r Y - Y M_s = -G` (G. H. Golub and C. F. Van Loan, Matrix Computations, 4th ed., §7.6.3); its
own rows are zero. Inside a cluster nothing is divided, and between blocks the eigenvalues are at
least `_EP_CLUSTER` apart, since every cluster's ball is closed. The clusters' residual takes a
solve of its own with `W0`, so the pairs' solve keeps exactly its right-hand sides (a solve's
blocking, and with it its rounding, may depend on how many there are).

The blocks are not normal, so one step leaves a remainder of second order in the start's error
over the gap. A second step takes the first step's correction again with no new solve (the chord
method, C. T. Kelley, Iterative Methods for Linear and Nonlinear Equations, SIAM, 1995, ch. 5),
which leaves third order: in `W0`'s coordinates a cluster's corrected subspace is `E_c + Y`, and
`H = W0^-1 Omega2 W0` is known from the first step -- diagonal on the plain columns, the pairs'
and clusters' blocks plus their residual expansions on theirs -- so the residual
`H (E_c + Y) - (E_c + Y) T` costs products with those blocks only. Each cluster keeps the
two-step basis only where it misses the check below by at most `_EP_STEP_GAIN = 0.5` of the
one-step basis's miss (each term over its threshold; sufficient decrease, as a line search asks of
a step: Kelley 1995, ch. 8), and the one-step basis otherwise, bit for bit. The miss reads the
split as well as the residual: where the first step already reaches rounding the second moves the
basis by rounding, and at an exact coincidence `eig(T)`'s eigenvectors are sensitive to rounding in
`T` -- on two CPU arithmetic paths a basis with a smaller residual turned the split's `cond(X)`
from 1.9 to 2.6e3 and from 3.4 to 156, each loud. On a start whose subspace is off by 1e-6 ..
3e-6 (a hand-built model of `eig`'s nearly dependent columns), one step left 8.0e-14 .. 1.5e-12
of `||A||` and the check warned from 3e-6; two left <= 1.2e-14, quiet, on seven paths. Two
clusters 2e-4 apart in a basis that couples them, where one step left up to 4.3e-13 and the
check warned on one path, are within 1.8e-15 and quiet on seven paths and five bases.

*The split.* With `Q1` the corrected orthonormal basis and `T = Q1^H Omega2 Q1`, `_cluster_split`
takes `X = [E_J | C Y]`: `E_J` some of `eig(T)`'s unit eigenvectors, `C` the orthonormal complement
of their span, `Y` the eigenvectors of the quotient `C^H T C` (Golub and Van Loan, §7.1.4 and
§7.6). Then `X^-1 T X = L + B`, `L` diagonal and `B` coupling the rows `J` to the columns `K`,
disjoint. Every nonempty proper subset of member lanes is tried as `J` (14 for four modes) and the
smallest `cond(X)` kept: on the y-Littrow clusters `|J| = |K| = 2` and `cond(X) <= 8.14`, a
three-mode cluster splits one and two, and `cond(X)` reached 8.3 on the synthetic clusters and
12.2 on the genuinely 2D ones below. Where a
cluster holds a Jordan chain of length three no such split exists. A three-mode cluster pads its
fourth lane: decoupled at a stand-in eigenvalue far from every mode, put back on its lane by an
explicit permutation rather than by `eig`'s order, and given the identity's row and column in `X`.

*The modes.* `q = diag(r) + B / (r_j + r_k)` on `J x K` is the exact square root of `L + B`
(`B**2 = 0`), every `r = sqrt(L)` on the side of the member whose label decays most, as for a pair.
`W = Q1 X`, `V = Omega_R W q^-1`. The block is placed into the layer's `q`, so `q` is exactly zero
off each `J x K` and the pairs' entries, and the coupling is handed on as an *entry list*, one lane
per nonzero entry. The exponential, the inverse, `two_face('pairs')` and the derivative rules
(`_pair_sylvester`, `_pair_inverse`, `_pair_generator`) sum over lanes and take it unchanged: their
code is the pairs'. The traced Fréchet cap counts the coupled rows (the pairs' and each cluster's
`J`) and columns separately, as before.

*The check.* The corrected subspace's residual `||Omega2 Q1 - Q1 T||_F` must stay below the larger
of `_EP_BACKWARD` at the cluster's scale and `_EP_RESIDUAL = 8` units of `eps ||Omega2||_F` -- the
rounding a product with `Omega2` leaves, at most 0.26 of that unit on the y-Littrow layers below at
truncations 5 .. 20 (21 offsets each) -- and the part of `X^-1 T X` the split drops, times `cond(X)`, below
`_EP_BACKWARD` at the cluster's scale. Eagerly a failure warns (`ExceptionalPointWarning`); under
a trace it multiplies `q` by NaN. Three synthetic layers fail it: a Jordan pair and a plain root at
one eigenvalue whose three columns are nearly parallel (`eig` handed such a basis back on one CPU
arithmetic path of eleven, smallest singular value 3.6e-15, and one Newton step left a backward
error of 1.7e-5; built by hand, the start fails on every path, after the second step too, 9.7e-6 .. 1.9e-2
on four), a start off by 1e-4 in its subspace (4.6e-11 .. 4.3e-10 after two steps), and a length-3 Jordan chain handed to
the joint rebuild. The pair gate takes two of the chain's three modes itself, and the
unresolved screen then warns. A cluster the joint rebuild takes reaches neither screen: scaled by
the propagation sensitivity, the backward-error screen warned on exact answers there: 24 of 264
solves of the thick layers below (NumPy, JAX and PyTorch, 10 and 3 um, 22 offsets out to 1e-5),
every one within 2e-8 of the coincidence.

*Measured* (2026-11-08). Windows NumPy 2.5.1 with OpenBLAS 0.3.33, JAX 0.11.0, PyTorch 2.13.0;
and NumPy 2.5.4 with OpenBLAS 0.3.34 on the CPU arithmetic paths of *Nearly coincident pairs*
(seven at truncation 5, four beyond). Against the 1D rebuild, `v = 0` row, ETM and SMM, 39
offsets of either sign from 0 to 1e-3 per scene. With the wide radius and the second step
(2026-11-09) the Littrow rows at every truncation, the thick layers, the derivative, trace and
field rows and the synthetic rows were measured again on the same paths: no bound rose (the
thick layers' fell to 2.4e-13), and no answer warned that had not before.

| case | screens alone (`ea57624b`) | joint rebuild |
| --- | --- | --- |
| PL952 / PA616, truncation 5, seven paths (1092 solves) | loud to `|dtheta| = 1e-5`, up to 1.0e-3 off (SMM at the coincidence) | <= 1.7e-14, no warning; the gate engaged out to `|dtheta| = 1e-4` |
| the same at truncations 10, 15, 20, four paths (1872 solves) | up to 4.9e-4 off, loud | <= 7.1e-14, no warning |
| PA616 at 10 and 3 um, `|q| = 0.025`, 17 offsets down to +-1e-11, four paths (272) and NumPy / JAX / PyTorch (204) | 3.3e-12 off at -5.999e-6 | <= 3.1e-13 (the 1D rebuild's own spread there is ~2e-13), no warning |
| eager `jax.grad`, `jax.jvp`, torch reverse, `forward_ad` (raster, wavelength, thickness, `theta`) at PL952 0, 7e-8, 1e-7 and PA616 0, 1e-8 | 3.3e-8 .. 3.1e-3 | <= 1.8e-13 |
| at those points: `jit`, `jit(vmap)`, `torch.func.vmap` values; `jit(grad(sum vmap))`, `jit(vmap(grad))`, torch `vmap` + backward | silent, 1.1e-9 .. 1.9e-3; 3.2e-8 .. 7.5e-3 | <= 1.4e-14; <= 9.6e-14 |
| at those points: fields (six components, of `max|F|`) / `jax.grad` of `sum |E|**2`, eager and `jit` | 9.1e-9 .. 3.2 / 1.5e-8 .. 5.9e-2 | <= 1.3e-13 / <= 3.3e-12 |
| truncation `(5, 2)`, two clusters per layer, 9 offsets each | up to 2.5e-3 off | <= 8.1e-15 (both clusters' layer model <= 4.8e-15 of `||A||`), no warning |
| genuinely 2D: a C4v square pillar at normal incidence, 11 exact coincidences (NVF at truncations 4 and 7, Laurent at 4), against a Cauchy circle in the pillar's permittivity (spread <= 5.7e-15) | loud, 1.6e-12 .. 9.3e-5 off | <= 3.1e-14 (two truncation-7 points 1.5e-13 .. 3.2e-13, that layer's floor off the point), no warning; JAX and PyTorch eager and traced values <= 1.3e-14, `jax.jvp` <= 1.2e-13 |
| synthetic: Jordan pair beside a root at 0 .. 1e-7, six bases, eight paths (and 9e-8, 1e-7 on three more) | refused by the pair gate | <= 2.8e-15 of `||A||`; where `eig` collapsed one basis (one path, at 0) the check warned |
| synthetic: two clusters and a pair, 2e-4 .. 1e-2 apart, basis of condition 10, seven paths | -- | one step: <= 2.5e-15 of `||A||` at 5e-4 .. 1e-2, up to 4.3e-13 at 2e-4, where the check on one path read a miss and warned; with the second step <= 2.0e-15 at every gap (five bases), quiet on every path |
| off a C4v coincidence: the pillar above at relative offsets 1e-9 .. 1e-5 either side (and 1e-4, 1e-3 above) of five coincidences, Laurent 641.11 and 623.60 nm, NVF 603.50, 614.90 and 634.19 nm, seven paths (1330 solves), against the Cauchy circle | the narrow radius alone: silent, up to 9.1e-12 off where the doublets leave it | <= 3.3e-14 where rebuilt jointly, no warning; the wide ball took the four out to a relative 3e-6 at least on every path, and beyond it `eig`'s own basis was within 1.5e-13 |
| synthetic: two degenerate doublets `2 s` apart with nearly dependent columns, `s` 1e-6 .. 3e-4, seven paths | not taken beyond `s = 5e-5` | <= 2.4e-15 of `||A||`, quiet |
| synthetic: a start off by 1e-7 .. 1e-4 in a cluster's subspace, seven paths | one step: 8.0e-14 .. 1.5e-12 at 1e-6 .. 3e-6, loud from 3e-6 | two steps: <= 1.2e-14 to 3e-6, quiet; at 1e-5 up to 4.3e-13, loud on one basis of three; loud at 1e-4 |

*Byte identity.* 168 digests (`R`, `T`, `r_xyz`, `t_xyz`) on `test_conical_cutoff`'s uniform and
patterned scenes, the Littrow scenes at 3e-4 and 1e-3, two near-degenerate uniform layers and six
random 2D layers, NumPy, JAX and PyTorch eagerly: identical to `ea57624b`, the joint rebuild
engaged on none, before the wide radius. Under a trace every 2D layer runs the slots, dead where
nothing qualifies; there the layer keeps its bits: 60 digests (`jit` values, `jit(grad)` and
`torch.func.vmap` values) on ten of those scenes, ETM and SMM, identical to `ea57624b`. Layers the
gate takes that the pair rebuild already got right -- PA616 out to `|dtheta| = 1e-4`, PL952 to
5e-5, the thick layer to 1e-3 -- moved: exact either way, and no longer warned. The wide radius
and the second step moved exactly the Littrow digests at 3e-4 and 1e-3 (24 eager, 12 traced),
whose two pairs it now takes as one cluster (<= 1.3e-14 off the 1D rebuild there on seven
paths); the other 144 eager
and 48 traced digests are identical.

*The screens under a trace* (2026-11-10). Nothing can warn under `jit`, `vmap` or
`torch.func.vmap`, and the unresolved and backward-error screens used to be skipped there, so what
they warn on eagerly came back silently. On the first-order path at the y-Littrow coincidences
(§9), every solve that warns eagerly -- PA616 and PL952, ETM and SMM, `|dtheta| <= 1e-4`, 26 of 32
-- was 1.1e-6 .. 0.69 off under `jit`, with no NaN. Two pairs 1e-6 .. 1e-5 apart beside a root
within `_EP_CLUSTER` -- a ball of five, which the joint rebuild leaves to the pair rebuild -- kept
a backward error of 6.3e-13 .. 4.7e-10 under `jit`, where the backward-error screen warns
eagerly. Both screens now run as data under a trace, and a layer whose screen would warn answers
NaN: its `Q` (2D) or the split's eigenvalues (first order) are replaced by NaN in a selection,
not a product, so a quiet layer keeps its bits.

A trace cannot walk a connected cluster, so both screens read balls instead
(`solve_2d.py::_traced_balls`): each mode's at most `_EP_SCREEN_BALL = 8` nearest modes within
`_EP_CLUSTER`, read once per distinct ball (the whole set within the radius, so a degenerate
cluster larger than eight counts once). The unresolved screen (`_unresolved_cluster_traced`)
takes the balls of three or more modes whose unit columns' smallest singular value, from their
Gram matrix, is below the screen's sine; the candidates with the smallest values take the defect
measure on as many slots as the layer has, and the layer is loud where one is defective or more
candidates qualified than the slots hold -- the cluster gate's overflow rule, under which a
degenerate ball in a nearly dependent basis counts too. The backward-error screen
(`_crowded_pairs_traced`) takes the balls of a pair rebuilt one at a time that hold another such
pair, reads the eager screen's backward error on each slot's columns -- one solve with `W^T` for
every slot's rows of `W^-1` together -- against the same threshold, and reads the propagation
sensitivity from `k0 d` as data, which eagerly is read only where it is concrete. On 2D both take
the cluster gate's `pair_cap // 2` slots (at most `n2 // 3` and `n2 // 4`); the first-order split
takes `max(slots, n / 16)`, since its eigenvectors of a degenerate (normal) cluster are not
orthonormal: a uniform layer at normal incidence, truncation `(5, 5)`, holds 11 candidate balls,
and with the pair gate's four slots it came back NaN. A defective cluster's nearly dependent
columns are one Jordan block's and sit in one ball; on a chain, or a cluster of more than eight
modes, a ball is the eager measure on a part of it.

*Measured* (2026-11-10, Windows NumPy 2.5.1 with OpenBLAS 0.3.33, JAX 0.11.0, PyTorch 2.14.1):

| case | before (`4e5eba63`), traced | traced screens |
| --- | --- | --- |
| synthetic, unresolved screen: 16 named layers and the test's nine cases (chains, a length-3 chain after the pair rebuild, coincident clusters of five and six modes, clusters taken jointly, a degenerate ball in a parallel basis, a chain with its modes in separate groups, a lone Jordan pair in `eig`'s basis) and 200 random clustered layers, `jit` and `torch.func.vmap` | not read | the eager verdict on every one (129 of the random ones loud) |
| synthetic, backward-error screen: two pairs 1e-7 .. 1e-5 apart with and without a root at 5e-5, three bases; and the eager screen's own model | silent, up to 4.7e-10 | the eager verdict on every one |
| first order, y-Littrow, PA616 and PL952, ETM and SMM, `dtheta` 0 .. 3e-4 (32 solves) | the 26 the eager screen warns on 1.1e-6 .. 0.69 off | NaN on those 26; the six quiet ones within 2.0e-14 of the 1D rebuild |
| 2D with the joint gate closed (pairs and clusters as before §8's joint rebuild), PL952 and PA616, ETM and SMM, `dtheta` 0 .. 3e-5 (40 solves) | silent | NaN on every solve JAX's eager screens warn on, and on PA616 at 1e-7, quiet eagerly on JAX and loud on NumPy (a backward error of 4.41e-12 against 3.0e-12): a threshold reads rounding at its edge; the rest within 7.2e-15 |
| the thick PA616 layers, 10 and 3 um (sensitivity up to 2.0e3), ETM and SMM, 37 offsets out to 1e-3 (74 solves), where the backward-error screen once warned on exact answers | within 2.8e-13 eagerly, quiet | no NaN, within 2.3e-13 |

The first-order row describes the split before its joint rebuild: since 2026-11-11 it takes
those clusters as the 2D path does (§9, *Coincident exceptional points, rebuilt jointly*), the
26 solves are exact and quiet eagerly and under `jit`, and its screens read only what that
rebuild leaves.

*Bits.* On the byte-identity scenes above (the traced ones with the uniform layer at normal
incidence added), on the 2D path and again on the first-order split: 168 eager digests each (NumPy, JAX and
PyTorch, ETM and SMM) and 66 traced ones each (`jit` values, `jit(grad)`, `torch.func.vmap`
values), 468 in all, identical to `4e5eba63`. A quiet traced layer keeps its bits, and eager
solves run none of this.

*Cost.* Under a trace every 2D and first-order layer now forms the balls (`O(n2**2)` per lane),
the Gram matrices of their columns, the defect measure on its slots, and on 2D one more solve with
`W^T`; eager solves pay nothing.

*Limits.*

- Not taken: clusters of more than four modes, chains (a ball that is not closed), grazing layers
  (§6 runs first), and complex64 (the defect measure's `eps**2` scaling keeps the gate closed below
  double precision: PL952 and PA616 in complex64 on NumPy, JAX and PyTorch gave the pair
  rebuild's answers, and under `jit` its bytes). The first-order formulation takes the same
  rebuild in its own split since 2026-11-11 (§9, *Coincident exceptional points, rebuilt
  jointly*). Eagerly the unresolved and backward-error screens still warn on what is not
  taken, except in complex64, where every screen's threshold scales with `eps` and stays silent
  too (a coincident cluster there is 6.9e-6 .. 5.6e-4 off, silently, at the float32 level);
  under a trace they answer NaN where they would warn (*The screens under a trace*, above).
- Off a symmetric coincidence the wide ball holds the doublets while their columns stay
  dependent to within `_EP_SINE`; beyond that (relative offsets past about 3e-6 on the C4v
  pillar) `eig`'s basis is left as it is, and was within 1.5e-13 of the oracle there. Where such
  doublets sit within 3e-3 of another cluster or pair, their wide ball holds more than four modes
  and only the narrow radius reads them: silent, as before the wide radius (not met on any layer
  measured).
- A length-3 Jordan chain is loud, not exact.
- Where `eig`'s start is poorer than the second step can absorb, the check warns; the answer is
  not exact. Two steps leave a remainder of third order in the start's error over the gap to
  the nearest block, so how poor depends on that gap: off by about 1e-4 with every block 1e-2
  away, or collapsed; off by 1e-6 beside a pair 5e-4 .. 1e-3 away (synthetic, eight bases, one
  CPU arithmetic path: up to 3.7e-12, loud on 14 of 24; where quiet within 2.9e-13). Not met on
  any layer measured.
- Under a trace every 2D layer pays for its slots: one more solve with `W0`, a ranking of
  `n2 x n2` distances read at two radii (twice the gate's Gram and ranking work), small batched
  `eig`, QR and solves, and the second step's products with the blocks. Not timed. XLA's scratch memory
  for one patterned layer rose 13 .. 28% under `jit` and 2 .. 4% under `jit(grad)` at
  truncations (6, 6) and (10, 10).
- Not measured: GPU (batched `4 x 4` `eig` inside `jit` / `vmap`), `pmap`, anisotropic layers at
  a cluster, NVF layers under a trace, and genuinely 2D coincident clusters other than the C4v
  pillar's.

**Genuinely 2D EPs on both formulations, re-measured.** Run on `aadbc351` (2026-10-14; NumPy
2.5.1, JAX 0.11.0, PyTorch 2.13.0), not carried over from the record above. The six cut-pillar
points are still EPs there (eigenvector sine 1.2e-7..1.0e-6 at the point, growing like
`sqrt(|rel|)`), and the first-order operator of the same layer is built from the same convolution
blocks (to 4.4e-16) with eigenvalues `lam**2 = -q**2` (to 1.8e-14 relative to `max(1, |q**2|)`,
measured 1e-2 off the point): one Jordan pair in each direction at the same wavelength. Against the Cauchy circle, whose spread over radius, point count, algorithm and
formulation was at most 1.1e-13:

| six points, `rel` 0, +-1e-12, +-1e-9, +-1e-6, amplitudes | gates closed | `aadbc351` |
| --- | --- | --- |
| 2D product, ETM / SMM | to 1.7e-6 / 6.4e-6 | <= 1.6e-13 |
| first order, ETM / SMM | to 1.0e-5 / 0.32 (fy 0.95, 600.01 nm: `R + T - 1 = +0.3`, no warning) | <= 4.3e-13 (at fy 0.95 / 618.89 nm, whose first-order floor off the point is 4.1e-13; at 600.01 nm <= 3.9e-14) |

At fy 0.95 / 600.01, fy 0.8 / 620.39 and fy 0.5 / 630.56 nm, derivatives in theta, the
wavelength, the pillar permittivity and the thickness were compared with oracles that run neither
autodiff nor a rebuild: Cauchy's derivative formula on circles (the trapezoidal rule on a circle:
L. N. Trefethen and J. A. C. Weideman, SIAM Rev. 56, 385, 2014, Sec. 12; after J. N. Lyness and
C. B. Moler, SIAM J. Numer. Anal. 4, 202, 1967), and for `d` (which does not move
the EP, and which the solver refuses complex) a Chebyshev derivative in `d` averaged over a theta
circle; spread at most 1.6e-12. `jax.jacfwd`, `jax.grad`, `jit`, `jit(vmap)`, torch backward and
`torch.func.vmap` with backward, on both formulations with ETM and SMM: at most 3.2e-12. With the
gates closed the same derivatives were up to 44 off at 600.01 nm (first-order SMM's eigenvector
split happens to be accurate at the other two points). Fields inside the layer: at most 8.1e-14 of
`max|F|`, their theta and wavelength derivatives, eager and under `jit`, at most 1.8e-12, within
the field oracles' own spread of 2.4e-12 (gates closed: up to 0.88, first-order SMM). Rebuilt layers: modal residuals at most 4.7e-16, basis
condition 9..140 against `eig`'s 4.7e6..4.0e7, and the coupled exponential within 4e-18 of a
50-digit one (SciPy's `expm` of the same generator was 1.7e-10 off). Circles of 3 nm, or 0.15 in
the permittivity, disagree by up to 6.2e-3 at 600.01 and 620.39 nm: something non-analytic lies a
few nm away there, and the oracles use radii whose results converge. Tests:
`test_a_genuinely_2d_ep_on_the_first_order_path_matches_the_cauchy_oracle`,
`test_a_genuinely_2d_rebuilt_layer`, `test_genuinely_2d_ep_derivatives_match_cauchy_oracles`,
`test_a_genuinely_2d_ep_field_matches_cauchy_oracles`. These are three layers of one cut-pillar
family at truncation (5, 2), each a single regular pair per direction; they say nothing about
coincident, multi-pair, NVF, anisotropic or grazing EPs.

**Limits and unchanged behavior.**

- Eagerly a layer with no gated pair and no cluster keeps the product construction's
  arithmetic; the screen can add a warning. At unresolved coincidences, the old arithmetic was
  2.0e-6..5.8e-4 off.
- Under `jit`, `vmap` or `torch.func.vmap`, the screens run as data on balls of at most eight
  modes (*The screens under a trace*): a cluster the joint rebuild does not take, and pairs
  rebuilt one at a time that miss the operator, answer NaN. More candidate balls than the
  screens' slots answer NaN too, so a trace can return NaN where the eager solve only warns or
  is quiet (a degenerate ball in a nearly dependent basis, or a value at a threshold's edge).
  A `vmap` over the thickness alone leaves SMM's modes concrete: its screens then run eagerly,
  with the calibrated threshold only, without the thickness's sensitivity. More qualifying pairs
  or clusters than the gates' slots, and a joint rebuild that fails its check, produce NaN as
  before.
- A root below `_EP_ROOT_FLOOR = 1e-9` stays with the grazing machinery or product construction.
  A layer that both grazes and holds an EP takes the grazing path; its EP pair is not rebuilt
  (not measured). The grazing refusal, `RayleighPointWarning` and `_require_separated` remain.
- Every traced layer's `jit(grad)` uses the coupled exponential's closed-form tangent, even
  off the band. Its bits can move; the historical digest's moved gradients remained as accurate.
- In complex64 the rebuild reaches the float32 floor (5.4e-7..2.0e-6 in the measured cases),
  and the cluster screen is effectively inactive.
- Not measured: GPU, `pmap`, `shard_map`, `jax.lax.map` / `jax.checkpoint` bodies for this path,
  second derivatives through a rebuilt layer, NVF or anisotropic layers at an EP, and large
  clusters beyond the cases above. Historical prototype cost does not establish peak memory
  at large truncation or the cost of the added screen.


## 9. The first-order path at a patterned exceptional point

`solve_2d_first_order.py::_pair_gate` gates, `solve_2d_first_order.py::_regular_pair` rebuilds,
`modal_gradient.py::_split_sylvester` differentiates; defective clusters,
`solve_2d.py::_defective_clusters` and `solve_2d_first_order.py::_regular_clusters`. Tests:
`tests/rcwa/test_first_order_exceptional_point.py` and `tests/rcwa/test_exceptional_point_clusters.py`.

As in §8, the measurements in this section are the 2026-10-12 prototype and judge record
against baseline `db37dbb8`, except the genuinely 2D note after the table, which quotes §8's
re-measurement on `aadbc351`, and *Coincident exceptional points, rebuilt jointly*, which is
dated where it stands. The order-grid slot count and overflow handling below implement
the judge's follow-up specification; historical prototype timings and digests are identified
separately from that implementation.

**What it was.** On the first-order path the patterned x-cutoff of §7 is a 2x2 Jordan block of
the `4n x 4n` operator `M`, once among the forward modes (at `lambda = i abs(ky)`) and once among
the backward ones. Nothing there forms the conical families, so the §7 rebuild never applied: the
path took `eig`'s eigenvectors as its basis. Near the point the pair's roots cross linearly (the
sine `s` of the angle between their eigenvectors grows like `15 abs(rel)` at PA616), and at it
`eig` leaves them at `s` = 1.9e-7 .. 1.8e-6: an angle set by rounding, not by the layer. Through
that basis ETM rounds like `C_E u / s**2` and SMM like `C_S u / s**3` (fitted at relative 1e-4,
largest at PL952: `C_E = 0.087`, `C_S = 5.6e-3`), all of it silent. At the point and within relative 1e-9 of
it, SMM's amplitudes were 3.0e-2 .. 2.45 off the §7 rebuild (`R + T - 1` up to +5.4 on lossless
scenes) and ETM's 2.1e-7 .. 2.5e-4; their gradients 1.4e-2 .. 8.3e3 and up to 5.7e-2
relative; SMM's fields 1.3e-2 .. 7.3 of `max|F|`; the same under `jit`, `vmap` and
`torch.func.vmap`, in the coherent sources' cascade (output powers 2.2e-2 .. 0.58 off), and on a
y-invariant raster forced onto this path (to 7.9). The entry that stood here before, 0.07 .. 0.32
for SMM, came from a smaller sample.

**Where the digits went** (one layer, each SMM piece redone in 40-digit arithmetic from the same
float64 `eig` output; the metric is the stack S-matrix in the half-space ports, against the
regular basis below). SMM's Redheffer stars invert `I - A22 B11`: the round trip written in the
layer's own mode coordinates, where the forward and the backward pair are both near-parallel,
condition 5e10 .. 1e12. Those inverses alone in 40 digits leave 1.3e-4 .. 8e-4 of S-matrix error
(from 3.6 .. 12); the interface solves or the propagation alone in 40 digits leave it O(1). And
all of SMM in 40 digits is still 1e-5 .. 2e-4 off, because `Psi Lam Psi^-1` reproduces `M` only
to 4.6e-10 there: `eig`'s pair is not a basis worth keeping. ETM inverts the mode basis once per
layer (`[Psi_f, Psi_b]`, condition ~1e7) and loses less. The labels play no part: the pair is
decay-dominated, `Im lambda = abs(ky)`, and ETM and SMM label it alike.

**The construction.** Within each direction the defective pairs are taken greedily, most
parallel first (an `argmax` over the normalised eigenvectors' Gram matrix, taken modes masked),
while the sine is below 0.05 (`_PARALLEL_SIN`). A layer can hold several: the x-cutoff condition
does not depend on `ky`, so a y-invariant raster reaches it in every row of orders at the same
wavelength (three pairs per direction at `truncation_order = (M, 1)`). Per pair:

- `X`: the pair orthonormalised and turned by the closed-form 2x2 Schur rotation of `X^H M X`
  (`_schur_2x2`), so that `S = X^H M X` is upper triangular;
- one step of Newton's method for the invariant subspace through its Sylvester equation
  (G. W. Stewart, SIAM Rev. 15, 727, 1973; J. W. Demmel, Computing 38, 43, 1987): `Z`,
  `X^H Z = 0`, from `(I - X X^H) M Z - Z S = -(M X - X S)`, through the bordered matrix
  `[[M - sigma I, X], [X^H, 0]]`, both columns in one solve with `sigma` the mean of `S`'s
  diagonal. The exact step's two remaining terms are `O(u)` here and are left out: each column's
  own shift differs from `sigma` by `abs(S00 - S11)`, which multiplies `Z`, and the second
  column's `S01 z1` multiplies the first column's correction -- that column is `eig`'s own
  eigenvector, already in the pair's subspace to `O(u / sep)`;
- `B`: `X + Z` re-orthonormalised and turned triangular again; `T = B^H M B`, its `(2, 1)` entry
  `O(u)` and dropped.

The pair's columns become `B` and its block of the generator `[[T00, T01], [0, T11]]`: a
coupling on one row and one column, rows and columns disjoint across pairs, which
`modal_gradient.py::exponential` takes exactly (§7). One step puts both subspaces of every
measured point within 1.7e-14 of a sorted complex Schur decomposition's (SciPy's; that is its own
accuracy), from 1e-13 .. 1.8e-9 for the orthonormalised `eig` pair; the exact step's terms, or a
second step, change nothing at that level. `abs(M B - B T) <= 4e-16 abs(M)`.

**Skipped pairs and the cluster screen.** A gated pair stays in the eigenvector basis when a
third mode of its direction is as parallel to either member, or another root lies within
`_PAIR_SEPARATION = 1e-4` times `max(1, abs(root))` of the pair's mean root. Skipping is silent
by itself. The bordered Newton solve loses accuracy beside a third root: with the separation
guard disabled, its residual was 1.5e-7, 2.7e-9 and 6.0e-11 at distances 1e-7, 1e-6 and 1e-5,
and at most 1.2e-12 from 1e-4 on. Unlike §8's simultaneous Sylvester correction, this method
therefore keeps the 1e-4 guard for every third root. Since 2026-11-11 a pair skipped that way,
with the root beside it, is a defective cluster of three modes, which the joint rebuild below
takes.

After rebuilding the eligible pairs, `_layer_split` calls §8's non-normality screen on the
final basis, restricted to each direction (`group=forward`), using `_PARALLEL_SIN = 0.05`
as the singular-value threshold. It emits `ExceptionalPointWarning` only for an unresolved
defective cluster. The prototype warned merely because it skipped a candidate, producing
15 false warnings in 120 ordinary uniform/C4 scenes; the final screen produced none. Taking
both directions together would falsely warn at a uniform Rayleigh graze, where opposite-going
modes meet at zero, so the directional restriction is essential.

Until the joint rebuild below, at y-Littrow coincident EPs (`ky0 = Gy/2`), every measured offset
`|dtheta| <= 1e-4` warned; the old arithmetic there was up to 0.55 off. The prototype had silently missed PL952 at
`dtheta = 1e-8` (0.39 off) because none of the mixed eigenvectors looked like a parallel pair.
From `|dtheta| = 3e-4` the amended path was exact to 1.8e-14. The warning is eager only and
never triggers `auto_detune`; under a trace the screen runs as data and the layer answers NaN
instead (§8, *The screens under a trace*): every one of those warned solves is NaN under `jit`,
where it was 1.1e-6 .. 0.69 off, silently.

**Coincident exceptional points, rebuilt jointly** (2026-11-11). At a coincidence `eig` mixes
the four vectors of the two Jordan blocks so that no two look parallel: the pair gate takes none,
or refuses the one it takes, and the split above kept the four in `eig`'s nearly dependent basis,
warned, and was up to 1.1 off (SMM, PA616 at `dtheta = 1e-4`). The split now takes them as §8's
2D path does. Before the pair gate, `_layer_split` reads §8's cluster gate
(`solve_2d.py::_defective_clusters`) on `eig`'s eigenvalues and vectors within each direction
(`group=forward`, as the screen reads them), at both radii and with `_EP_SINE` and `_DEFECT_NU`,
and the pair gate then never sees a cluster's modes (`_pair_gate(exclude=)`). Read the other way
round, under a trace a refused candidate inside a cluster took a pair slot, and at truncation
`(5, 2)` the slots ran out before the pair beside two clusters. `_regular_clusters` then rebuilds
each cluster in the basis `W0`: the split's basis after the pair rebuild, with the cluster's unit
columns orthonormalised. There `W0^-1 M W0` is block diagonal to rounding off the clusters -- the
plain eigenvalues, and the rebuilt pairs' triangular blocks, whose columns are already an
invariant subspace -- which is §8's setting with the pairs corrected beforehand. So the Newton
step, the chord step, the split and the check are §8's own code
(`solve_2d.py::_joint_cluster_bases`, which both formulations call; on 2D the pairs' residual
expansion joins the chord step, here there is none). Only the generator differs: the first-order
operator's eigenvalue is the generator itself, so a cluster's block of `lam` is `L + B` on its
members, with no root taken, and its columns are `Q1 X`. `B`'s rows `J` and columns `K` are
disjoint from each other and from every pair's, which is all that `exponential`,
`split_generator` and `_split_sylvester` ask; they read the coupling entry by entry. A failed
check warns with §8's text, and under a trace it, or more clusters than slots, multiplies both
generators by NaN. The traced slots are the pair gate's `slots` (at most `n / 3`, `n` the split's
mode count). Since the gate reads first, a ball of three holding a pair is a cluster here, where
on 2D it stays with the pair rebuild: a pair beside a root within the wide radius, and the pair
the pair gate refuses beside a root within `_PAIR_SEPARATION`.

The exponential's traced rule (`modal_gradient._coupled_frechet`) gathers a direction's coupled
rows to a static count, four by default, and a cluster couples up to three. Where the split
rebuilt clusters, its structural word `coupled` is a count in place of `True`, `max(4, slots +
3 S)` with `S` the cluster slots, which ETM, SMM, the unstabilised transfer and the coherent
sources' cascade pass to the exponential; the layer record carries it as its length, so the
field reconstructions read it too (`len(record) - 8`, where a pair-only record says 1 and is
read as `True` was). At truncation `(5, 2)` each direction holds two clusters beside a pair, five
coupled rows.

*Measured* (2026-11-11, Windows NumPy 2.5.1 with OpenBLAS 0.3.33, JAX 0.11.0, PyTorch 2.14.1;
oracles: the §7 1D rebuild for the y-invariant scenes, §8's Cauchy circle for the C4v pillar):

| case | before | joint rebuild |
| --- | --- | --- |
| PL952 and PA616, truncation `(5, 1)`, ETM and SMM, 19 offsets `dtheta` 0 .. +-1e-3 (76 solves on each of NumPy, JAX and PyTorch) | loud for `abs(dtheta) <= 1e-4` (PL952 1e-5), up to 1.1 off | <= 2.0e-14, quiet; the gate engaged out to `abs(dtheta) = 3e-3` (PA616) and 2e-3 (PL952), and beyond them the pair rebuild was within 1.2e-14 |
| genuinely 2D: the C4v pillar's three coincidences of §8 (Laurent 641.11 and 623.60 nm, NVF 603.50 nm), relative offsets 0 and +-1e-9, and at two of them (641.11, 603.50) +-1e-8 .. 1e-5, ETM and SMM against the circle on this split | loud, up to 3.7e-4 off at the coincidence; with the gate closed, silent and up to 3.1e-13 off at +-1e-7 (the doublets, which the screen's narrow balls read two at a time) | <= 2.0e-14, quiet, one four-mode cluster a direction |
| eager `jax.grad` (raster, wavelength, thickness, `theta`), `jax.jvp`, torch reverse and `forward_ad` at PL952 0, 7e-8 and PA616 1e-8, against the 1D `jax.grad` | loud | <= 1.2e-13 |
| at PL952 0 and PA616 1e-8, lanes `rel` 0, +-1e-12 and 1e-2: `jit`, `jit(vmap)`, `torch.func.vmap` values; `jit(grad(sum vmap))`, `jit(vmap(grad))`, torch `vmap` + backward | NaN | <= 1.6e-14; <= 1.2e-13 |
| at those points: fields (six components, of `max|F|`) / `jax.grad` of `sum |E|**2` in the raster and wavelength, eagerly and under `jit` | loud | <= 3.5e-14 / <= 1.0e-12 |
| truncation `(5, 1)`, `jit` values and `jit(grad)` of `sum R`, five offsets 0 .. 1e-4 | NaN | <= 1.7e-14; <= 1.0e-13 |
| truncation `(5, 2)`, two clusters a direction beside a pair (five coupled rows): eager, `jit` values and `jit(grad)`, the same offsets | loud, up to 0.65 off; NaN | <= 1.9e-14, <= 1.3e-14; <= 1.2e-13 |
| synthetic: two coincident Jordan blocks (three bases); a Jordan pair beside a root at 0 .. 2.9e-3 (three bases each) | loud | an invariant subspace of `M` to <= 1.6e-15 of `abs(M)`, quiet |
| synthetic: two coincident Jordan blocks beside a Jordan pair the pair rebuild takes, in bases of condition 1, 10 and 100 that couple every block (three each) | loud | <= 8.2e-15 of `abs(M)`, quiet; with the pair's coupling left out of the cluster's rows in `W0`, 3.2e-10 .. 1.1e-8 and loud (the y-invariant scenes cannot see that term: their order rows never couple) |
| synthetic: a Jordan pair and a root at one eigenvalue with three nearly parallel columns built by hand, as on 2D (three bases) | -- | loud by the check alone ("joint rebuild failed"), NaN under `jit`; the screen after it reads the rebuilt columns as independent |

The `jit(grad)` rows are relative to the 1D path's `jax.grad` of `sum R`. With the coupled count
left at four, the `(5, 2)` layer's `jit(grad)` -- and its field's -- came back NaN.

*Bits.* The 2D path's 169 eager and 67 traced digests (§8's scenes) are identical to main
`2e4a1db4`: moving the core into `_joint_cluster_bases` changed no operation. On the first-order
split 145 of the 169 eager digests are identical; the 24 that moved are the Littrow scenes at
`dtheta` 3e-4 and 1e-3, whose two pairs the wide ball now takes as one cluster (exact either
way). Of the 67 traced ones (`jit` values, `jit(grad)`, `torch.func.vmap` values) 55 are
identical and the 12 that moved are those scenes again. Every other traced layer runs the
cluster slots dead and keeps its bits, `jit(grad)` too, although its exponential now gathers
more rows: a pair's coupling is one entry a row and a column, so each sum the closed form adds
holds one nonzero term.

*Mutants.* Thirteen one-line mutations of the new code, each against the tests that should
catch it: eleven are caught -- the gate removed, its direction restriction, the pair gate's
exclusion, a pair's coupling in the cluster's rows, the check's verdict eagerly and under a
trace, the slot overflow, the coupled count and the field's reading of it, and either placement
into the generator. Two survive: the rows' diagonal taken from `eig` instead of the rebuilt pairs
(they differ by about `sqrt(u)`, and the correction that moves is of second order), and §8's
pair terms in the chord step, which no measured model reaches (as when the chord step landed).

*Limits.* What the joint rebuild does not take keeps §8's limits: a cluster of more than four
modes, a chain, a length-3 Jordan chain (its check fails, and the screen warns too: loud),
complex64 (the gate stays closed). A ball of three holding a pair is a cluster here, unlike on
2D, since the gate reads before the pair gate; the synthetic rows above cover it. Under a trace
more clusters than slots answer NaN: a coincident cluster left over would be loud anyway (the
screen reads it), but the C4v doublets would not: with the gate closed they were up to 3.1e-13
off, silently. The traced exponential's closed-form tangent now gathers up to `slots + 3
S` rows a direction wherever clusters can be rebuilt, which is every traced first-order layer.

**Derivatives.** The rebuilt basis is built on stopped values, and the derivative is the modal
rule's (`modal_gradient.py::split_generator`) in that basis. Within a direction the generator's
tangent is `B^-1 dM B` as before; across the directions the tilt `X` solves
`X (D + N) - (D + N) X = R`, `N` the couplings, in four passes of plain quotients
(`_split_sylvester`, the mirror of §7's `_coupled_sylvester`). The split returns a structural
`coupled` word (`solve_2d_first_order.py::_layer_split`), which ETM, SMM, the unstabilised
transfer and the coherent sources' cascade pass to `exponential`; a rebuilt layer's field record
carries a ninth slot (`None`, no pytree leaf), and the field reconstructions read the word off the
record's length, under a trace too.

**Under an abstract trace** the masks travel as data, and every first-order layer takes the
regular-pair layout. A static keyword `slots` reaches `_pair_gate` and `_layer_split` through
the solve/cascade functions. The public solver derives it from the retained order grid:
`slots = min(max(n_orders_u, n_orders_v), modal_gradient._COUPLED_CAP)`, with
`_COUPLED_CAP = 4`. The coherent-source cascade uses the same grid when available, otherwise
four slots. Eager solves do not use this limit: they rebuild every eligible pair.

A lane that engages nothing keeps its eigenvector basis through `where` and uses an identity
bordered matrix, so an inactive slot forms no infinity. After the final slot in either
direction, one more greedy candidate checks for overflow. Another eligible pair multiplies
the layer's generator by NaN. A skipped, unsafe candidate does not count as overflow; it keeps
the unresolved-cluster limits described below.

The one-slot prototype silently missed additional order rows: at MU614 on a y-invariant raster,
SMM `jit` amplitudes were 2.0 off and `jit(grad)` 2.4e2 off. The historical three-slot experiment
reduced those errors to 7.4e-14 and 9.6e-13. Four slots cost a traced 2D first-order grating
7–11% more than one in the designer's timing experiment.

**Measured** (oracle: the §7 1D rebuild of the same structure, itself 1.4e-13 .. 3.6e-12 from the
§7 fit; scripts and outputs with the design record):

| historical measurement | baseline `db37dbb8` | prototype |
| --- | --- | --- |
| amplitudes at `rel` 0, +-1e-15, +-1e-12, +-1e-9, five scenes, ETM / SMM | 2.1e-7 .. 2.5e-4 / 3.0e-2 .. 2.45 | <= 1.1e-13 |
| offsets 1e-6 .. 1e-3 | to 1.4e-8 / 1.0e-4 | <= 1.0e-13 |
| `R + T - 1`, lossless | to +5.4 (SMM) | <= 1.8e-14 |
| `jax.grad`, `jvp`, torch reverse and `forward_ad`: raster, wavelength, thickness, `theta` | to 5.7e-2 / 1.4e-2 .. 8.3e3 | <= 7.1e-12; `dR/dwl` the 1D path's distance to the fit |
| `jit`, `vmap`, `jit(vmap)` through the point and off it, `lax.map`, `torch.func.vmap` | to 2.9e-4 / 0.21 .. 5.2 | <= 9.7e-14 |
| `jit(grad)`, `jit(vmap(grad))`, `grad(checkpoint)`, torch autograd through `vmap` | to 1.7e-2 / to 5.8e3 | <= 4.5e-12 |
| fields (eager, `jit`), eight scenes; field gradients (`jax.grad`, `jit(grad)`, torch) | 8.8e-7 .. 1.6e-4 / 1.3e-2 .. 7.3; to 5.3e2 | <= 2.0e-13; <= 7.1e-13; SMM = ETM to 3.3e-15 |
| coherent `PlaneWave` through the sources' cascade: output powers / fields | 2.2e-2 .. 0.58 / 0.2 .. 2.9 | <= 1.0e-14 / <= 6.0e-14 |
| y-invariant raster on this path, lit from either side | to 9e-5 / to 7.9 | <= 1.6e-13 |

Every scene in that table is a 1D grating or a y-invariant raster. Genuinely 2D EPs (a pillar
cut in y) reach this path as one Jordan pair per direction; §8's re-measurement on `aadbc351`
covers them: the eigenvector split was 0.32 (SMM) and 1.0e-5 (ETM) off at fy 0.95 / 600.01 nm,
silently, and the rebuilt split is within 4.3e-13 of the circle oracle in amplitudes, 3.2e-12 in
derivatives, 8.1e-14 in fields and 1.8e-12 in field derivatives (the field oracles' spread:
2.4e-12).

**Historical prototype byte identity.** Outside the pair gate, a 464-line `sha256` digest of
`R`, `T`, `r_xyz`, `t_xyz` and fields over 96 random first-order scenes (conical input, uniform,
2D rasters, tilted magnetic tensors; `mu`, lit from below; NumPy, JAX, PyTorch) with second-order
controls, eager `jax.grad` and torch backward, run twice per tree, is identical in every eager
line (368, none routed). Under a trace every first-order layer takes the regular-pair layout,
masks off: `jit`, `jit(vmap)` and `torch.func.vmap` values came out identical too (72 lines),
and `jit(grad)` moved in 22 of 24 lines -- the coupled exponential's closed-form tangent -- by at
most 3.3e-14 of the gradient's largest entry (median 1.9e-15), each tree as far from its own
eager gradient as before (3.1e-13 against 2.9e-13).

**Historical prototype cost**, with one traced slot and before the revised cluster screen,
per call against baseline `db37dbb8` (PA616 at truncation 5 and 20 and a 2D (6, 6) first-order
grating; medians of two runs). Eagerly off the band only the gate's read is added: 1.01 .. 1.05x
(NumPy solve), 1.02 .. 1.04x (`jax.grad`). At the point the rebuild costs 1.20x at truncation 5
and 1.00 .. 1.02x at 20. Under a trace every first-order layer pays for the layout and the
closed-form tangent: `jit(solve)` 1.08 .. 1.23x, `jit(grad)` 1.14 .. 1.41x, a 16-wavelength
`jit(vmap)` 1.22 .. 1.31x and its gradient 1.27 .. 1.44x, `torch.func.vmap` 1.09 .. 1.24x,
compilation 1.34 .. 1.86x.

**Complex incidence and double negatives.** The judge also ran five complex-angle and two
real-angle double-negative cases across both connecting algorithms and both formulations.
They agreed within the fit's accuracy at the split-label points. The largest first-order
error was 2.8e-8 at Si1397, `theta = 1.2j`, where the fit residual was 3.5e-9 and baseline ETM
was already 2.0e-8; no pair routed there. Baseline first-order SMM was 0.58..14 off in those
cases. This covers those measured cases, not every possible opposite-label pair.

**Limits and unchanged behavior.**

- Coincident EPs of three or four modes a direction are rebuilt jointly (above). What that
  rebuild does not take -- a cluster of more than four modes, a chain, a length-3 Jordan chain
  (its check fails) -- keeps the eigenvector arithmetic and warns eagerly; under `jit`, `vmap`
  or `torch.func.vmap` the screen runs as data and the layer answers NaN. The screen's slots
  there are `max(slots, n / 16)`, `n` the split's mode count, and more candidate balls than that
  answer NaN as well, as more clusters than the gate's slots do.
- More eligible pairs than the static slots poison the generator with NaN. A third parallel
  mode outside the screen's eigenvalue radius can instead cause a silent skip; this case
  was not observed in the measurements.
- Ordinary degenerate layers stay quiet. A nearly parallel but diagonalizable pair can still
  be rebuilt; the combined census changed 10 of 800 solves by at most 1.2e-15. Outside the
  gates eager arithmetic stays unchanged; traced `jit(grad)` may change rounding through the
  coupled exponential's closed-form tangent.
- Complex64 rebuilt values were 4.8e-7..2.0e-6 from complex128, at the float32 floor, while the
  precision-scaled cluster screen is effectively inactive. The separation floor was tuned
  in float64, not re-derived for single precision.
- The count-mismatch refusal, forward/backward collision check, `_require_separated`, grazing
  refusal and `RayleighPointWarning` are unchanged. A first-order Rayleigh graze is separate
  from the same-direction exceptional cluster and does not by itself emit this warning.
- Not measured: GPU, `pmap`, `shard_map`, second derivatives through a rebuilt layer, NVF or
  anisotropic layers at an EP, and source-field gradients at the point. The existing
  modal-operator restrictions remain.

## 10. A layer of a half-space's medium at that medium's graze, and a uniform layer `eig` cannot diagonalise

Recorded 2026-11-09, NumPy / OpenBLAS, float64.

**The graze.** An empty stack reaches the engine as a zero-width layer of the top medium, and a
stack may hold layers of either half-space's medium. At a real `k_par` those layers are solved
as layers so that their bits stay where they were (the merge of
[connecting algorithms §10](connecting-algorithms.md#10-etms-labels-at-a-complex-in-plane-wavevector)
runs at a complex `k_par` or under `half_space_merge_tolerance`). On the 2D path that left the
first row of the table above false in practice: at `wavelength == period`, normal incidence,
cover `eps = 1`, every such layer held the exact `q = 0` cluster and refused with
`rcwa.singular`, though the single interface has a closed-form answer. A run of layers equal to
a half-space is now merged into it whenever a retained order lies inside the band where the 2D
solve leaves its ordinary path (`min|kx**2 + ky**2 - eps*mu| < 1e-9 * max`, on that medium,
`solve_2d._COUPLED_REL`). The merged solve is the bare interface: `R` and `T` equal the
Fresnel values to 1e-12 for both polarizations, lit from either side, on SMM, ETM and the
first-order formulation, and agree with the stack one ULP of wavelength off the point to 1e-10
(`tests/rcwa/test_half_space_graze.py`). `result.diagnostics['half_space_merge']` names the
merged layers. Off that band nothing changes: 72 ordinary scenes (uniform, patterned and lossy
layers; layers of either half-space's medium at ordinary angles and 1e-6 off a graze; both
algorithms, both formulations, both polarizations and sides) returned byte-identical `R`, `T`
and amplitudes before and after. The 1D paths solve their graze in place and are not changed.

A coherent multi-source solve cannot merge a layer (its cascade carries every source, and the
face shift carries one incident order), so it still cannot solve a layer of a half-space's
medium exactly at the graze. It used to refuse with `rcwa.invalid` and advice to raise the
Fourier order, which cannot help; it now refuses with `rcwa.singular` and names the light-line
order. One step off the point solves.

**The non-converging `eig`.** A uniform isotropic layer's `Omega2` is
`diag(kx**2 + ky**2 - eps*mu)` twice over in exact arithmetic; computed, its off-diagonal
blocks keep a few ULPs of `kx*ky - ky*kx`. On a matrix of exactly repeated eigenvalues with that
noise LAPACK's `geev` can fail to converge: measured on an empty interface (cover `eps = 1.5`,
substrate `eps = 2`, period `(1.49414428555847, 1.775182937114707)`, `theta =
0.16079553455782242`, `orders=(3, 3)`), `info = 30` on the 98 x 98 operator, SciPy's driver too,
while one ULP of `theta` either side converges. Where `eig` fails and the operator is diagonal
to round-off (`max|off-diagonal| <= 4 n eps max|A|`), the layer takes `(diag(A), I)`: the exact
eigensystem of `A` without those entries, which is the nearby-problem criterion of §1. The
answer is the bare interface's Fresnel value to 1e-11 and is continuous in `theta`. A converged
`eig` is never replaced, and a matrix coupled beyond round-off still raises. PyTorch's `eig`
fails on the same operator (error code 30; on main it escaped the public API as a raw
`torch._C._LinAlgError`), and JAX's returns NaN without raising: an eager non-finite JAX
decomposition of a finite matrix is read as the same failure. Under a JAX trace nothing can
be read, and the NaN still reaches the solve's finiteness refusal.

# Rayleigh points — the exact graze, which paths solve it, and what `auto_detune` does

Written: 2026-08-16; §6 (the 2D coupled-cluster rebuild) added later the same day.
Measured on CPU (Apple Accelerate), NumPy / JAX / PyTorch, float64; the single-precision
caveat is stated where it applies. This file is the measurement record; for the
narrative walkthrough of how the pieces hand off to each other at solve time, read
`rayleigh-handling-walkthrough.md`.

Targets:
- `wavebit/_core/solve_1d.py` — `layer_modes_1d_conical` (the reformulation), `layer_modes_1d`
- `wavebit/_core/solve_2d.py` — `layer_modes_2d`, `_rebuild_grazing_cluster` (§6)
- `wavebit/_core/backend.py` — `check_mode_cutoff`, `check_finite`,
  `RayleighPointWarning`, `AutoDetuneWarning`
- `wavebit/_core/engine.py` — `solve` (the `auto_detune` ladder)
- `tests/test_rayleigh_guard.py` — every contract below, pinned

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
| 1D pure TE/TM (`tetm=`) | always fine | `V = W*q` — nothing singular is ever inverted |
| 1D conical (`psi`/pair) | **fine since the reformulation** | `V` blocks now come from the eigensystem; they used to pass through `inv(A)` of an exactly singular `A` |
| 2D | **correct or loud since the rebuild (§6)** | the grazing cluster is re-derived from the coupled first-order problem: representable clusters solve exactly, numerically defective ones refuse with the detune advice; quiet-wrong is gone |
| any path, eigenvalue *exactly* `0.0` — and, on 2D, any cluster whose coupled rates split below ~1e-10 of the modal scale | refuses, unless the ladder clears it | the exponential mode basis is degenerate at the point itself (the limit basis is `{1, z}`) |

`solve` handles the practical side by default: `auto_detune=True` retries a solve that
refused or fired `RayleighPointWarning`, first at `wavelength*(1+1e-12)`, then also
`theta*(1+1e-12)` for the one graze wavelength cannot move — and reports every knob it
moved in an `AutoDetuneWarning`. Nothing is ever moved silently.

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

Default on. When a solve refuses with a degeneracy (`FloatingPointError` /
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

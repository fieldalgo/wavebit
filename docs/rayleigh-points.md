# Rayleigh points — the exact graze, which paths solve it, and what `auto_detune` does

Written: 2026-08-16; §6 (the 2D coupled-cluster rebuild) added later the same day; §7 (the
1D conical x-cutoff at `ky != 0`, which is not a Rayleigh point) added 2026-10-10.
Measured on CPU (Apple Accelerate), NumPy / JAX / PyTorch, float64; the single-precision
caveat is stated where it applies. This file is the measurement record; for the
narrative walkthrough of how the pieces hand off to each other at solve time, read
`rayleigh-handling-walkthrough.md`.

Targets:
- `wavebit/_core/solve_1d.py` — `layer_modes_1d_conical` (the reformulation), `layer_modes_1d`,
  `_cutoff_cluster` and `_regular_conical_modes` (§7)
- `wavebit/_core/solve_2d.py` — `layer_modes_2d`, `_rebuild_grazing_cluster` (§6)
- `wavebit/_core/backend.py` — `check_mode_cutoff`, `check_finite`,
  `RayleighPointWarning`, `AutoDetuneWarning`
- `wavebit/_core/engine.py` — `solve` (the `auto_detune` ladder)
- `tests/test_rayleigh_guard.py` — every contract below, pinned; `tests/test_conical_cutoff.py`
  — §7

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
| 1D conical (`psi`/pair), `ky == 0` | **fine since the reformulation** | `V` blocks now come from the eigensystem; they used to pass through `inv(A)` of an exactly singular `A` |
| 1D conical at `kx_m**2 = eps*mu` with `ky != 0` (an x-cutoff, `kz = abs(ky)`) | **solved in a regular basis since 2026-10-10 (§7)**, eagerly and, since 2026-10-11, under every abstract trace (`jax.jit`, `jax.vmap`, a `jax.lax.map` or `scan` body, `jax.checkpoint`, `torch.func.vmap`) | the two mode families collapse onto one field there; the layer is rebuilt with an upper-triangular propagator, exact in a uniform layer and at a patterned layer's Jordan block alike |
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

## 7. The conical x-cutoff (`ky != 0`) — not a Rayleigh point, and now solved exactly

`solve_1d.py::_cutoff_cluster` gates, `solve_1d.py::_regular_conical_modes` rebuilds,
`solve_1d.py::_regular_conical_modes_ad` differentiates. Tests: `tests/test_conical_cutoff.py`.

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
- A cluster root under 1e-9, or a pair with opposite root labels, keeps the family
  construction.

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
`jacfwd` and `jacrev` keep a concrete primal and are gated as an eager solve is, and so is
`torch.compile` in its default mode, whose graph break reads the value.

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
- A refused cluster (a root under the 1e-9 floor, or opposite labels) gets empty masks: the
  family construction's arithmetic in the general layout, NaN at an exact zero included, silent.
- `ky == 0` at a Rayleigh point under a trace: NaN, unchanged (the `check_finite` gap).
- More than 4 coupled rows or columns in one layer (`_COUPLED_CAP`), or a coupled row that is
  also a coupled column, gives a NaN tangent, the value unaffected. A uniform layer couples
  none, every patterned point measured couples one, and a rebuilt layer's rows and columns are
  disjoint by construction.
- Second derivatives through the modal operator still raise, as do `torch.func.grad` and `jvp`;
  `torch.compile(fullgraph=True)` raises `Unsupported`.
- Not measured: GPU, `pmap` and `shard_map`, patterned double negatives under a trace.

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

**Still open, and separate.**

- The 2D path at a *patterned* exceptional point is 1e-8..4e-5 off, silently: its grazing
  rebuild is keyed to `q -> 0`, and this cluster sits at `q = abs(ky)`.
- First-order SMM at a patterned exceptional point is 0.07..0.32 off (`R + T - 1` up to
  +0.36), silently, and stays near 0.2 for every relative offset under 1e-8. First-order ETM
  there sits at the 2D path's level (3.5e-7); neither first-order path fails at a uniform
  cutoff.
- Patterned double negatives were not measured (no real crossing was found). Their flux labels
  read the same `1/(w - ky**2)` quotient; if a pair's labels disagree the gate keeps the family
  construction, silently.

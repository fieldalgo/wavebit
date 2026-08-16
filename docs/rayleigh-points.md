# Rayleigh points — the exact graze, which paths solve it, and what `auto_detune` does

Written: 2026-08-16. Measured on CPU (Apple Accelerate), NumPy / JAX / PyTorch, float64;
the single-precision caveat is stated where it applies.

Targets:
- `wavebit/_core/solve_1d.py` — `layer_modes_1d_conical` (the reformulation), `layer_modes_1d`
- `wavebit/_core/solve_2d.py` — `layer_modes_2d` (the documented limitation)
- `wavebit/_core/backend.py` — `check_mode_cutoff`, `check_finite`,
  `RayleighPointWarning`, `AutoDetuneWarning`
- `wavebit/_core/engine.py` — `conv_solve` (the `auto_detune` ladder)
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
| 2D | **unreliable — detected, not solved** | `V = (Omega_R @ W)/q` loses the grazing column to cancellation; fires `RayleighPointWarning` |
| any path, eigenvalue *exactly* `0.0` | refuses (0/0), unless the ladder clears it | the exponential mode basis is degenerate at the point itself |

`conv_solve` handles the practical side by default: `auto_detune=True` retries a solve that
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
  case (D#81): `R` from 0.18 to 626 against a true 0.22, scattered randomly across `fto`
  and connecting algorithm — one draw (`SMM`, `fto=5`) returned `R+T = 0.981` with every
  guard green, reading as ordinary absorption on a lossless stack.
- **Conical 1D, after**: `Ky2 == ky^2 * I` on this path, so `W1` diagonalises `A` itself
  and `W2` does `B @ epx_conv`. The `V` blocks are rewritten as exact functions of the
  eigensystem (`inv(A) @ W1 == W1 * (1/(w1 - ky^2))`, and likewise for `B`), plus one
  inverse each of the well-conditioned `W1`/`W2` (cond ~2). Nothing singular is inverted.
- **2D**: `V = (Omega_R @ W) / q`. The grazing column of `Omega_R @ W` is a true `O(q^2)`
  value produced by cancellation of `O(1)` entries, so it carries `O(eps)` *absolute*
  error — 100% relative — and no identity rescues it: `W` does not diagonalise `Omega_R`
  alone, and `Omega_L` is singular at the same point.

## 2. What the conical fix measures as

| Claim | Measurement |
| --- | --- |
| exact point = one-ULP neighbour | 8 digits, on numpy/jax/torch × ETM/SMM × TE/TM/mixed `psi` |
| energy at the exact point | `R+T = 1` to 1e-10 (lossless stacks) |
| no behaviour change off the point | old vs new ≤ 8.2e-15 over 40 random conical solves; full suite green |
| fields | the old path's field reconstruction was wrong **over a band** — at one ULP off it erred at 2e18 scale and converged only ~1e4 ULP out, while efficiencies looked fine; the new path is smooth from the exact point (2.9e-8 to its neighbour, ETM) |
| cost | none — median solve times equal within noise at `fto` 20/60/100 (the per-layer `eig` dominates both spellings) |

## 3. The 2D detection

The 2D efficiency error follows `~eps / (min|w| / max|w|)` — measured on four exact-graze
stacks over a detune ladder:

| `min|w|/max|w|` | efficiency error |
| --- | --- |
| 1e-17 (exact) | 2e-3 … 8e0, some with **no warning of any kind** (5 of 48 solves) |
| 1e-15 | ~1e-3 |
| 1e-14 | ~1e-4 |
| 1e-13 | ~1e-5 |
| ≥ 1e-12 | ≤ 2e-6 |

`check_mode_cutoff` therefore warns `RayleighPointWarning` below `1e-13` — where the error
crosses ~1e-5 — a band ~1000 wavelength-ULPs wide. 60 random non-grazing 2D solves fired 0
false positives, and no existing suite case fires it. It is a warning rather than an error
on purpose: most of the band is wrong at the 1e-5 level, not garbage, and the two earlier
attempts at *refusing* in this regime both took correct answers away and were reverted.

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

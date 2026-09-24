# The time convention, audited — which root, which direction, and what the flip left behind

Written: 2026-08-11. Measured on an Apple M4 (Accelerate), NumPy 2.5.1 / SciPy 1.18.0 /
mpmath 1.3.0, float64 throughout.

**Power-normalization correction (2026-09-22).** The historical interpretation of
the TM discrepancy in item 3 and §7 below is superseded. Once the reference plane
and incident-only power are specified, the pure-1D and coupled efficiencies must
agree. The real-transverse-wavevector power conversion is now shared across paths;
see [the half-space power audit](half-space-power-audit.md). This does not change
the amplitude tests or establish general complex-transverse-wavevector powers.

Targets:
- `wavebit/_core/conventions.py` — `boundary_kz`, `_off_principal_branch`, `mode_q`
- `wavebit/_core/boundary.py` — `conv_mats`
- `wavebit/_core/scattering.py` — the SMM incident amplitude (prose only)
- `wavebit/_core/solve_2d_4ff.py` — `split_forward_backward` (scope note only)
- `tests/test_branch_selection.py`, `tests/test_lossy_media.py`

---

## TL;DR

Wavebit was written under `e^{+iωt}` and now runs `e^{-iωt}` throughout. That flip is
complex conjugation of the entire problem, so anything holding a hard-coded sign, or a
root-selection priority chosen under the old convention, is inverted exactly where the two
conventions differ. This audit went looking. Four results:

| | Question | Answer | Moved? |
| --- | --- | --- | --- |
| **1** | Which priority is right for a half-space, `Re(kz) > 0` or `Im(kz) >= 0`? | **Neither, on its own.** `Re(kz) > 0` with an `Im` tie-break is right for every real `k_par` and for most complex ones; a third condition decides the rest | **yes** — `_off_principal_branch` |
| **2** | Is `mode_q` disagreeing with `boundary_kz` at complex `k_par` a defect? | **No** — a layer's `±q` is a label and a half-space's `kz` is a boundary condition. The label is measured free to 2.0e-15 | no |
| **3** | Is the TM discrepancy against the mpmath Airy reference a solver defect? | **No — and (corrected 2026-08-12) not a reference defect either**: the compared numbers were the coupled path's TM *efficiency* and the reference's `|r_p|²`, which for an absorbing superstrate differ by exactly `|n_top|²·Re(n_top⁻²)` — two conventions of a quantity the contract excludes. Solver matches both references to 4.3e-16 on amplitudes | no |
| **4** | Any other flip residue? | **One live**: the painted-polygon builder still wrote `conj(n)**2`, turning an absorbing painted layer into gain. Plus one sentence of prose left in the old convention | **yes** — `boundary.py` |

The load-bearing find is (1). The shipped rule put a **step discontinuity of 2.96 in the
reflected amplitude** into `r(θ)` at 71.565° for a lossy dielectric on gold — at an angle
where `kz**2` stands 9.03 clear of its only branch point and the physics is analytic. The
correction is bit-identical to what shipped for **every real `k_par`**, measured over 4e5
samples on each of the lossless and passive branches, so no released number can move because
of it. No existing test is in the regime it does move: `test_branch_selection.py` and
`test_lossy_media.py` are the only files in the repo that give a half-space a complex index at
all, and `test_lossy_media`'s one both-sides-absorbing case (`WET` on `AU` at 0.4 rad) sits
well clear of the crossing — `Im(eps_bot) = 1.08` against `Im(eps_top)·sin²θ = 0.009`. Full
suite: **1403 passed, 2 xfailed** — the 1374 that passed before this work, none moved, plus
the 29 tests it adds.

---

## 1. Why no amount of testing at a real angle could have found any of this

For a passive medium (`Im(eps) >= 0`) and a **real** in-plane wavevector,

```
Im(kz**2) = Im(eps - k_par**2) = Im(eps) >= 0
```

so `kz**2` lies in the closed upper half plane and its principal square root lies in the
closed **first quadrant**: `Re(kz) >= 0` *and* `Im(kz) >= 0`, simultaneously. "Carries energy
downward" and "does not grow downward" are different physical demands that happen to name the
same number. Measured over 2e5 random passive half-spaces at real `k_par`: the two candidate
rules disagree **0 times**, and `max |Re-first − Im-first| = 0.000e+00`.

They come apart only when `k_par**2` is complex — an absorbing superstrate at oblique
incidence, or a complex incidence angle. Over 2e5 random complex `k_par**2`:

| quadrant of `kz**2` | samples | rules disagree |
| --- | --- | --- |
| I (`Re>0, Im>0`) | 28048 | 0 |
| II (`Re<0, Im>0`) | 122011 | 0 |
| **III** (`Re<0, Im<0`) | 40457 | **40457** |
| **IV** (`Re>0, Im<0`) | 9484 | **9484** |

Every disagreement is a full sign flip, never a small one. The lower half plane is reachable
only when `Im(k_par**2) > Im(eps)`, which needs the superstrate to be lossier than the medium
being asked about.

**What the suite covered before this work.** Every absorbing-superstrate test in
`test_lossy_media.py` runs at `truncation_order=0` onto a *lossless* substrate. That is quadrant IV, where
the shipped rule is right (§3). The regime where it is not — quadrant III with an absorbing
substrate — had no test at all.

---

## 2. The growing wave, and why it is not the objection it looks like

Take the case from the brief: `n_top = 1.5 + 0.3i`, `theta = 0.6`, so
`kx = 0.846964 + 0.169393i`.

| region | `kz**2` | shipped root | `exp(+i k0 kz z)` going down |
| --- | --- | --- | --- |
| superstrate `(1.5+0.3i)^2` | `1.471346+0.613061j` | `1.238003+0.247601j` | decays |
| layer `eps = 2` | `1.311346−0.286939j` | `1.151894−0.124551j` | **grows** |
| substrate vacuum | `0.311346−0.286939j` | `0.606115−0.236704j` | **grows** |

A field growing without bound in `+z` looks inadmissible. It is not, and the measurement says
why: in a **lossless** medium a plane wave must have `k_r · k_i = 0` — its planes of constant
phase and constant amplitude are perpendicular. Measured:

| | `k_r` | `k_i` | `k_r · k_i` | `|k_r||k_i|` |
| --- | --- | --- | --- | --- |
| layer `eps=2` | `(0.84696, 1.15189)` | `(0.16939, −0.12455)` | **0.000e+00** | 3.006e-01 |
| substrate | `(0.84696, 0.60611)` | `(0.16939, −0.23670)` | **0.000e+00** | 3.032e-01 |

So the wave is **constant along its own ray**. It grows in `z` only because it decays in `x`,
and the incident field decays in `x` in exactly the same way — a plane wave at a real angle
inside an absorbing medium is already unbounded as `x → −∞`. Boundedness in `z` is therefore
not available as a selection criterion for this problem class, and the growth is bookkeeping
rather than physics.

That leaves the radiation condition: no energy may arrive from `z = +∞`. Over 4e5 random
complex-`k_par` half-spaces, restricted to the samples where the two rules disagree:

| | TE flux `S_z > 0` | TM flux `S_z > 0` |
| --- | --- | --- |
| `Re`-first | **100.00%** | 27.12% |
| `Im`-first | 0.00% | 72.88% |

`Re(kz) > 0` **is** downward TE flux, exactly. (The TM column is a separate question, taken up
in §7.)

---

## 3. Deriving the half-space rule

`kz = sqrt(eps − k_par**2)` has exactly **one** branch point, at `kz**2 = 0`. Away from it the
physical root is whatever is reached by deforming continuously from a configuration that is
not in doubt. Four such deformations were run to the same target points, tracking the root by
always taking whichever of `±sqrt` was nearer the previous step — no priority rule anywhere:

| path | what it varies | seed, and why it is certain |
| --- | --- | --- |
| **A** | `theta`, `0 → target` | normal incidence: `kz**2 = eps`, `Im(eps) >= 0`, first quadrant |
| **B** | `Im(n_top)`, `0 → target` | real `k_par`: both rules agree (§1) |
| **D** | `k_par`, `Re(k_par) → k_par` | real `k_par`, same reason |
| **C** | `Im(eps_bot)`, `+30 → 0` | limiting absorption: `kz**2` far up the imaginary axis |

Result, on seven configurations spanning lossless and absorbing substrates. `A = B = D` on
every row, so they are one column; `=` means "the same root as the shipped rule".

| case | shipped rule | **A = B = D** | C (limiting absorption) |
| --- | --- | --- | --- |
| lossless substrate, lossy prism 34° | `0.606115−0.236704j` | = | = |
| lossless substrate, lossy prism 52° | `0.397235−0.695107j` | = | `−0.397235+0.695107j` |
| lossless substrate, lossy prism 75° | `0.388775−1.080525j` | = | `−0.388775+1.080525j` |
| gold substrate, lossier dielectric top 74° | `0.005167−3.302738j` | **`−0.005167+3.302738j`** | = A |
| gold substrate, gold-like top 74° | `0.712961−0.076027j` | **`−0.712961+0.076027j`** | = shipped |
| weak-metal substrate, absorbing top 69° | `0.127859−1.659440j` | **`−0.127859+1.659440j`** | = A |
| `absorbing_on_both_sides` (a shipped test) | `0.175210+3.056050j` | = | = |

Two things to read off.

**The three physical paths always agree with each other.** A, B and D vary three different
things — the angle, the superstrate's absorption, and the Fourier variable — and land on the
same root every time. That is what makes the answer a property of the point rather than of
the path.

**Limiting absorption is the odd one out, and it is measurably the odd one out.** Closing the
loop (walk A forward, walk C backward) and counting turns about `kz**2 = 0` gives a
**winding number of −1.000000**, with the loop never approaching closer than 0.325 to the
branch point. The disagreement is a homotopy fact: adding loss to the substrate while holding
a complex `k_par` fixed drives `kz**2` straight up across the principal cut, which the
physical deformations do not do. Limiting absorption is the standard tool for fixing a
radiation condition and it is the wrong tool here.

**Which of A/B/D has the mandate.** Path D, on this argument: any field produced by an actual
source has a Fourier spectrum over **real** `k_x` — that is what a Fourier transform in `x`
is — and for real `k_x` the root is not in doubt. A complex `k_par` is not in that spectrum;
it is the idealisation "a plane wave at a real angle inside an absorbing medium", a field
unbounded in `−x`, so it is a continuation of the real-`k_x` family rather than a member of
it. A and B agreeing with D is then corroboration, not an independent claim.

### The rule, in closed form

Path D's segment is
```
kz2(t) = eps − t*k_par_sq,     t: 0 → 1
```
and `Im(kz2(t))` is **linear** in `t`, so the segment meets the real axis at most once, at
`t* = Im(eps)/Im(k_par_sq)`. The principal root (whose cut is the negative real axis) stops
being the continuous one exactly when that meeting happens on the negative side:

```
flip  ⟺  Im(k_par_sq) > 0   and   t* < 1   and   Re(eps) − t*·Re(k_par_sq) < 0
```

That is `_off_principal_branch`, and it is a function of the two arguments `boundary_kz`
already takes. Measured against 4000 brute-force walks over random `(eps, k_par_sq)`: **0
disagreements**. Measured against the shipped rule at real `k_par`: **bit-identical**, 0.0e+00
over 2e5 passive and 2e5 lossless samples.

The three conditions say, in words: *both half-spaces absorb, the superstrate is the lossier
one, and the substrate is metallic enough that the segment passes the branch point on the
left.*

It stays backend-agnostic and traceable: no Python `bool`, no boolean indexing, and `eps` is
broadcast against `k_par_sq` rather than assumed to be an array. Measured in the corrected
regime — NumPy, JAX and PyTorch return the same `kz` (max difference 5.2e-16, 0.0), `jax.jit`
gives the same, `jax.grad` through a `k_par` scale returns a finite `-0.2056`, and a full
`truncation_order=2` grating solve agrees to all ten printed digits on the three backends.

### The one construction that cannot arbitrate, and why it is worth recording

The obvious independent check is to delete the half-space: replace the substrate with a thick
slab of the same material on an arbitrary terminator, propagate with a matrix exponential (no
root is selected inside a layer at all), and take `D → ∞`. This was run, with four terminators
from vacuum to a near-mirror, `D` from 1e3 to 1e7 nm.

It agrees with the correction on every absorbing-substrate case. But it **cannot be used as
the ground truth**, because a truncation limit always selects the decaying root by
construction — that is what makes the terminator's contribution vanish. So it can only ever
answer `Im(kz) > 0`, and on the lossless-substrate cases (where the correct root grows in `z`)
it converges to the *wrong* answer while looking perfectly well-behaved:

| case | truncation limit | correct root |
| --- | --- | --- |
| lossless substrate, lossy prism 52° | the decaying root | the growing one (A = B = D) |
| gold substrate under a lossier top | the decaying root | the same (A = B = D) |
| `absorbing_on_both_sides` | the shipped root | the same |

A prerequisite for reading this doc's §3 correctly: **the half-space answer in the
lossless-substrate case is not the limit of any truncation**, so a "make it thick and see"
argument reaches the wrong conclusion there with no warning that it has.

---

## 4. The defect: a step in `r(θ)` where the physics is analytic

`kz_bot**2(θ) = eps_bot − eps_top·sin²(θ)` traces a straight segment as `θ` is swept. It
generically misses `kz**2 = 0`, so `r(θ)` is analytic and any step in it is the branch rule.

Run through the shipped solver, `truncation_order=0`, one 120 nm layer of `n = 2` between the half-spaces:

| case | pol | `θ` of the step | `r` below | `r` above | **step** | `Δθ` | `\|kz_bot²\|` there |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `n_top=1.5+0.4j`, `n_bot=` gold | TE | 1.249046 | `−1.5866−0.0379j` | `−4.5092−0.4777j` | **2.9555** | 6.7e-05 | 9.03 |
| same | TM | 1.249046 | `+3.5350−2.6329j` | `−1.9686+0.1298j` | **6.1581** | 6.7e-05 | 9.03 |
| `n_top=0.2+3.2j`, `n_bot=` gold | TE | 1.164419 | `−0.6944−0.1546j` | `−0.4731−0.1799j` | 0.2228 | 6.7e-05 | 0.36 |
| same | TM | 1.164419 | `−1.2604+0.0989j` | `−1.1866+0.0874j` | 0.0746 | 6.7e-05 | 0.36 |

After the correction, the same four rows read 9.9e-05, 2.6e-03, 8.7e-05 and 2.5e-05 — which
is just the finite-difference step times the local slope. Refining confirms it: shrinking `Δθ`
by a factor of ten, five times over, the increment falls by the same factor each time — ratio
**0.1000** at every step, at four widely separated spots including two picked because they
showed large slopes. A jump keeps its size; an analytic function does not.

**Swept, not spot-checked.** 4000 random stacks × 1200 angles, with every flagged spot refined
to separate a genuine jump from a sharp resonance:

| rule | spots flagged | **true jumps** | worst |
| --- | --- | --- | --- |
| shipped (principal root) | 536 | **3** | 3.33 |
| corrected | 255 | **0** | — |

Most flagged spots are surface-plasmon resonances, which are legitimately sharp; the refinement
is what tells them apart.

### What the correction does not fix

The corrected root still gives `Im(kz) < 0` — growth into the substrate — for **12.4%** of
reachable configurations with an absorbing substrate. That is not a residual defect; it is §2
again. Worst case found (`eps_top = 23.73+7.87j`, `eps_bot = 2.27+0.51j`, `θ = 89.7°`,
`kz = 0.784 − 4.698j`): the attenuation vector projected on the propagation direction is
`+0.0507`, so the wave **decays along its own ray**, and `Re(kz) = +0.784 > 0`, so no energy
arrives from below. The ray is nearly grazing, and the growth in `z` is the transverse
component of a decay.

---

## 5. `mode_q` versus `boundary_kz` — free, not broken

At complex `k_par` the two functions return roots that are not related by `q = -i*kz` at all.
For `eps = 2` under the §2 superstrate, `boundary_kz` gives `1.1519 − 0.1246j` and `mode_q`
gives `+0.1246 + 1.1519j`, which is `+i*kz` — the other root.

That is legitimate, because **a layer's root is a label**. The stack algebra uses the `±q` pair
in full, so relabelling which one is "down" maps the recursion to itself. Measured two ways:

| | measurement |
| --- | --- |
| Airy recursion, 4000 random stacks, every interior root flipped | max relative change **1.97e-15** |
| shipped RCWA solver, 40 nm layer, 6 random subsets of modes flipped | max `\|ΔR\|` **1.5e-13** (real angle), 1.0e-13 (absorbing superstrate), 6.0e-15 (on gold) |
| `mode_q`'s `Im(q) <= 0` tie-break inverted | `0.0` exactly on the complex-`k_par` cases |

A half-space's root is not a label — it is the boundary condition, and there is one right
answer. So the two functions are answering different questions and have no reason to agree.

**What `mode_q`'s rule is buying.** `Re(q) >= 0` keeps `|exp(-k0*q*d)| <= 1`. Driving a layer
with `boundary_kz`'s rule instead (via `q = -i*kz`) is *allowed* by the freedom above and
useless in floating point:

| layer thickness | `\|ΔR\|` between the two labellings | max `\|X\|`, `mode_q` | max `\|X\|`, boundary rule |
| --- | --- | --- | --- |
| 50 nm | 3.84e-15 | 0.9993 | 1.13e+03 |
| 200 nm | 1.98e-06 | 0.9972 | 1.61e+12 |
| 500 nm | 6.40e+01 | 0.9929 | 3.30e+30 |
| 2000 nm | 3.13e+17 | 0.9719 | 1.18e+122 |
| 4000 nm | `FloatingPointError` | 0.9446 | 1.39e+244 |

The algebra is free; the arithmetic is not. `mode_q` spends the freedom on conditioning, and
its tie-break — which fires only where `|exp(-k0*q*d)| == 1` either way — decides nothing at
all. That last point is stated because it is the one a reader is most likely to "fix" by
copying `boundary_kz`'s opposite tie-break, which would be a no-op dressed as a correction.

---

## 6. Flip residue found: the painted builder was still conjugating

`_core/boundary.py::conv_mats` wrote

```python
eps_bg = xp.conj(...(n_base)...) ** 2
eps_s  = xp.conj(...(n_index)...) ** 2
```

`_core/parametric.py::assemble_eps_coeffs` — of which this is a later copy — carried the
sentence *"`e^{-iwt}` throughout, with no conjugation on the way in"* and did `n ** 2`. The
copy kept the conjugate the original was moved off in 2026-08. An absorbing painted layer
therefore came out as a **gain medium**.

Both expressions are quoted as they stood. Neither survives: the material convention flip
(2026-08-20) took the squaring out of every layer site, so both builders now read a relative
permittivity straight off the layer's materials, and the sentence about conjugation is the
half that is still load-bearing — `Im(eps) > 0` is loss, as written
([`conventions.md`](conventions.md)). The keys are `background` and `material` now, and the
quantity in them is `eps`, not `n`. What the section is about is unchanged: a copy that keeps
a conversion its original has dropped, and the fact that a real-valued test set cannot see
it.

Measured on a uniform absorbing layer (two overlapping rectangles of the same index covering
the cell, so the painting resolves into polygons and the closed-form Airy answer still applies):

| layer | pol | `θ` | R painted, before | R exact | gap | `R + T` before |
| --- | --- | --- | --- | --- | --- | --- |
| gold `0.18+3j` | TE | 0.0 | 1.057624 | 0.908991 | 1.49e-01 | **1.0804** |
| gold | TM | 0.4 | 1.063050 | 0.902177 | 1.61e-01 | **1.0873** |
| `4+0.05j` | TE | 0.4 | 0.080126 | 0.103641 | 2.35e-02 | **1.1148** |

After: every row agrees with the oracle to **3.3e-16** and `R + T` sits below 1.

**Why nothing objected.** `test_boundary.py` built every shape with a real material (`n_index`,
as the key was spelled then), where `conj(n)**2 == n**2`. It cross-checks the two builders against each other to 5e-9 — and both
were on real indices, so agreeing proved nothing about this. The same blind-spot shape as the
parametric path had before `test_absorbing_layer_via_the_parametric_path` was written, and the
same shape as the NVF goldens all being 1D or square-pixel.

`check_physical` *does* fire here (`R + T > 1` on a real `n_top`), as a warning. It is the
only thing that did.

### And one sentence of residue in prose

`_core/scattering.py`'s module docstring said the SMM's incident p amplitude is
`H_p = i*n_top*cos(psi)`. The code writes `-1j * n_top * c_p`, and the code is right:
measured, `R_p / (−i·n_top)` reproduces the analytic amplitude to 1e-15 on both algorithms
and `+i` gives its negative. `conj(+i) = −i`, so the sentence was describing the retired
formulation. Fixed as prose; nothing numerical moved.

---

## 7. The TM discrepancy — resolved: two efficiency conventions, no defect anywhere

> **Correction, 2026-08-12.** This section originally concluded "the p-admittance of the
> earlier reference is what was wrong". The dev-paper-archive session then showed its
> reference had used the correct `n²/kz` all along, and supplied its exact stack — which
> settled the whole thing in one measurement, below. The reference was right, the solver
> was right, and the 1.7e-02 was **two legitimately different conventions of a quantity
> the contract declares undefined**: the reported "solver" number was the *coupled*
> (psi-specified) path's TM efficiency, the reference's was `|r_p|²`, and for an absorbing
> superstrate at a real angle those differ by exactly
>
> ```
> R_p(coupled) / |r_p|²  =  |n_top|² · Re(n_top⁻²)          (TE factor: exactly 1)
> ```
>
> Measured on their stack (`n_top = 1.5+0.3i`, layer `n = 2`, `d = 0.37`, `λ = 0.94`,
> `θ = 0.6`): `0.2207491559 × 0.9230769231 = 0.2037684516` — their two numbers, to all
> ten digits. The TE factor being exactly 1 (`kz_inc = n_top·cosθ` is an identity at real
> `θ`) is why TE never showed it, and the factor `→ 1` as `Im(n_top)² → 0` is the k²
> vanishing. The pure-1D path's TM efficiency equals `|r_p|²`; the coupled path weights by
> `Re(kz/n²)`. Both are defensible spellings of "incident power in an absorbing medium",
> which is precisely the quantity `docs/conventions.md` excludes from the contract — the
> amplitudes are the comparable surface there, and those agree across every path to 1e-15.
> The cross-path difference is now pinned as *known and deliberate-not-hidden* by
> `test_lossy_media.py::test_absorbing_superstrate_efficiency_is_convention_bound`.

What follows is the original investigation, kept because its measurements stand: the solver
agrees with two independent references to machine precision, and the spelling table remains
a useful catalogue of error families that produce this symptom's *shape* — even though, as
it turns out, none of them was the cause.

The open item was a TM disagreement of 1.7e-02 at `Im(n_top) = 0.3` against an mpmath
50-digit Airy reference in admittance form, falling as roughly `k²` to 8.0e-17 at `k = 0`,
where TE agreed to 6.7e-16. Smooth rather than a jump, so not a mis-selected branch.

Two references were built here to settle it.

- **R1** — mpmath at 50 digits, admittance/Airy recursion. Independent *arithmetic*.
- **R2** — Berreman 4×4 with `expm` and arbitrary complex half-spaces. Independent
  *formulation*: no Fourier orders, no eigenmodes, no s/p rotation, no transfer or scattering
  recursion, and a uniform layer is a matrix exponential, which needs no root at all. It never
  writes an admittance down.

**Validity gate, stated before use.** Both impose the half-space root rule that is itself
under audit, so they arbitrate the *admittance algebra* and not the root. Every case below is
a lossless substrate with `Re(eps) > 0`, where the corrected and shipped rules are
bit-identical (§3), so the root is not a free variable in this table.

`n_top = 1.5 + i·k`, layer `n = √2` (120 nm), vacuum substrate, `θ = 0.6`:

| `k` | pol | `\|solver − R1\|` | `\|solver − R2\|` | `\|R1 − R2\|` |
| --- | --- | --- | --- | --- |
| 0.30 | TE | 3.00e-16 | 2.78e-17 | 2.78e-16 |
| 0.30 | **TM** | **3.76e-16** | **4.31e-16** | 1.45e-16 |
| 0.10 | TM | 9.64e-17 | 1.82e-16 | 1.01e-16 |
| 0.00 | TM | 3.79e-16 | 4.48e-16 | 6.94e-17 |

The solver is right in TM at every absorption, against both. (The original conclusion drawn
here — that the earlier reference's p-admittance was wrong — did not survive contact with
that reference's actual code; see the correction box above.)

The correct one follows from the transverse pair that is continuous across an interface. For
TM under `e^{-iωt}`, with `H` carrying the free-space impedance, the first-order system is

```
d/dz (Hy, Ex) = +i k0 [[0, eps], [1 − k_par²/eps, 0]] (Hy, Ex)
```

so the downward wave has `Ex = (kz/eps)·Hy` and the admittance is `eps/kz` (equivalently
`n²/kz`) — up to the overall inversion, which only flips the sign of `r_p` and is the ordinary
TM sign convention. This repo's Airy reference writes it as `kz/eps` and so reports `−r_p` in
Macleod's sense; the magnitudes are identical. The same matrix is what `_berreman_M` in
`tests/test_anisotropic_berreman.py` reduces to for an isotropic tensor, which is the check
that it was derived rather than transcribed.

Spellings that are wrong, and how wrong, on the same stack (error against the solver):

| `eta_p` | `k = 0.3` | `k = 0.1` | `k = 0.03` | `k = 0` |
| --- | --- | --- | --- | --- |
| `n²/kz` **(correct)** | 3.8e-16 | 9.6e-17 | 1.3e-16 | 3.8e-16 |
| `kz/n²` (inverted; `−r_p`) | 3.8e-16 | 9.6e-17 | 1.3e-16 | 3.8e-16 |
| `conj(n)²/kz` | 2.20e-01 | 5.29e-02 | 3.99e-02 | 3.8e-16 |
| `\|n\|²/kz` | 1.98e-01 | 6.68e-02 | 2.00e-02 | 3.8e-16 |
| `n²/conj(kz)` | 1.41e-01 | 5.06e-02 | 1.38e-02 | 3.8e-16 |
| `n/cos(θ_real)` in every medium | 2.08e-01 | 1.56e-01 | 1.04e-01 | **9.9e-02** |
| `n/√(1−(Re k_par/n)²)` (real-part Snell) | 2.07e-01 | 6.82e-02 | 1.95e-02 | 3.8e-16 |

All but one vanish as the absorption does, which is the signature reported — and that shared
signature is exactly why the table could not identify the cause: the true explanation (the
efficiency-normalization factor in the correction box) vanishes as k² too. The table survives
as a catalogue of misspellings to check against, not as the resolution.

---

## 8. Swept and clean

**Six formulations, one physics.** `{ETM, SMM} × {1d, conical, 2d}` share `conventions.py` and
nothing else — the SMM builds its incident field from mode amplitudes and cascades Redheffer
stars, the ETM builds it from the boundary match and multiplies transfer blocks. (`'TMM'` was
the option's name when this audit ran; it was renamed the same week — the 2ff algorithm was
always the enhanced-transmittance form. `'TMM'` now names only the 4ff path's deliberately
plain transfer.) Compared on
the complex reflection *amplitude* (`de_ri` has no defined incident-power normalisation with
an absorbing superstrate), over ten configurations all with complex `k_par`, including the
three the correction moves:

max spread across the six paths: **≤ 1.5e-14**, worst case `lossier top on gold, 74°, TM`.
Everything else at 1e-15 or below. This rules out residue that differs between paths; it
cannot see residue they share, which is what §7's independent references are for.

**Field reconstruction.** Two checks no branch rule can slip past.

| check | result |
| --- | --- |
| does `\|E\|` shrink going down inside an absorbing layer? | yes, for gold, `1.5+0.05j` and `4+0.5j`, TE and TM |
| interior field of a uniform slab vs the closed form, amplitude **and phase** | **9.4e-17** (gold), 5.2e-16 (`1.5+0.05j`), overall scale `1.000000 − 0.000000j` |

A leftover conjugation would show up as a field that grows downward, or as a scale factor that
is not 1. Neither.

> **Both of those checks were blind to the sign defect found on 2026-08-14**, and the way they
> were blind is the reusable part. Every path was returning `Ez`, `Hx` and `Hy` in the frame
> whose `+z` points *out* of the stack (`docs/conventions.md` declares the opposite). The first
> check is a magnitude, and the second compares `E_t` — and `E_t` is exactly the part the
> parity leaves alone. So a check aimed at the *time* convention passed a *frame* defect
> through, because both of its probes happened to live in the intersection where the two
> conventions agree. The lesson is not that the checks were weak: it is that "the field is
> right" needs a probe that touches `H` and `E_z` **relative to** `E_t`, which is what a curl
> residual does and a closed-form amplitude comparison does not.
> Now covered by `tests/test_field_core.py::test_every_path_returns_a_maxwell_field_in_the_declared_frame`.

**`solve_2d_4ff`.** Its `split_forward_backward` splits modes by energy flux with a decay
fallback, and its docstring claimed that is "the same rule `boundary_kz` applies to a
half-space". After §3 that is true only for real `k_par` — which is all this path sees, since
its half-spaces are vacuum. The docstring now says so, and points here, so that growing an
absorbing superstrate onto that path does not silently inherit the pre-correction rule.

**The Fourier order-index sign** — the trap recorded from the earlier convention work — is
covered by the RETICOLO, S4 and fmmax goldens and by the field-phase check above; nothing new
was found and nothing was changed.

---

## 9. Where this stops being trustworthy

* **Single precision is out of scope and unaffected.** The branch decision asks whether
  `Re(v)` is zero against a fixed `1e-12`, which at complex64 is a question about round-off.
  Everything here is float64.

* **The lossless-substrate case rests on three continuations, not on a truncation.** §3's
  thick-slab construction gives the *other* root there and cannot be used to check it. What
  defends the shipped choice is that it is the one continuous in `θ` and in `Im(n_top)` —
  the two knobs a user turns — and that it satisfies the radiation condition (`Re(kz) > 0`,
  §2). If someone finds a fourth physical deformation that disagrees, this is where to look.

* ~~The exact TM discrepancy of 1.7e-02 was not reproduced~~ — **it now is, to all ten
  digits** (§7 correction box): the coupled path's TM efficiency versus `|r_p|²`, factor
  `|n_top|²·Re(n_top⁻²)`. Left struck rather than deleted because the original hedge is the
  reason the wrong attribution stayed retractable.

* **`Re(kz) > 0` and `Re(kz/eps) > 0` are not the same condition when both `eps` and
  `k_par**2` are complex** — they disagree in 29% of an unrestricted random sample. For real
  `k_par**2` they provably never disagree: `Re(kz·conj(eps)) = Re(kz)·(|kz|² + k_par²)`,
  verified to 1.0e-15 relative over 4e5 samples. The corrected rule is polarization-blind, and
  in the regime where it fires this was checked case by case rather than proved. A
  polarization-dependent half-space rule was not pursued and is the obvious next question if
  a TM result in that corner ever looks wrong.

* **Anisotropic half-spaces are not covered.** Every argument here is for a scalar `eps`.

---

## 10. What pins it

`tests/test_branch_selection.py` (21 cases). Every one of these fails on the pre-correction
rule — checked by stashing it, which is the only evidence that a test about a branch cut is
testing anything:

| test | catches |
| --- | --- |
| `test_the_rule_is_a_no_op_for_every_real_k_par` | the correction growing a reach it should not have |
| `test_both_conditions_hold_at_once_for_a_real_k_par` | the reason a real-angle suite is blind here |
| `test_reflection_is_analytic_across_the_branch_change` | the 2.96 step, by refining rather than by tolerance |
| `test_the_root_is_the_continuation_from_normal_incidence` | the closed form drifting from the walk it abbreviates |
| `test_a_layer_mode_root_is_free` | someone "fixing" `mode_q` to agree with `boundary_kz` |
| `test_the_mode_q_tie_break_is_worth_nothing` | the same, in its most tempting form |
| `test_the_two_selectors_disagree_at_complex_k_par_by_design` | the disagreement being read as a bug |

`tests/test_lossy_media.py::test_absorbing_layer_via_the_painted_path` (8 cases) covers §6.
All eight fail on the pre-fix builder.

Suite: **1403 passed, 2 xfailed** with both corrections and the new tests in place — the
1374 that passed before them, none moved, plus the 29 added.

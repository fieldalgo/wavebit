# How a solve handles a Rayleigh point — the end-to-end walkthrough

:::{note}
This detailed guide documents the retained `wavebit.compat.legacy` interface and its measured scope. For the current problem/numerics lifecycle, start with [the common guide](../common/lifecycle.md). Advanced capabilities are preserved here with explicit compatibility calls; their limits are unchanged.
:::

Written: 2026-08-16. Companion to `rayleigh-points.md`: that document carries the
measurements and the per-path history; this one explains the *process* — what actually
happens, in order, when a solve lands on or near the condition, which component owns each
decision, and why the pieces are tuned to hand off to each other. If you change any of the
thresholds named here, read the composition section first: they are sized against each
other, not independently.

The running example throughout is the repro stack: a patterned pillar layer over a uniform
spacer, period 1000 nm, wavelength 500 nm, normal incidence. The retained orders then have
`kx = 0.5 * m`, so a spacer of index `n = 2.0` puts order `m = +-4` at `kx = +-2.0` —
**exactly** on that layer's light line. That order neither propagates nor decays in the
layer; the layer eigenvalue for it is mathematically zero. One ULP of wavelength to either
side and the problem is ordinary: the singularity is in the coordinates, not the physics.

## The cast

| Component | Role |
| --- | --- |
| `solve_2d.layer_modes_2d` | builds each layer's modes; detects the graze and dispatches |
| `solve_2d._rebuild_grazing_cluster` | re-derives the grazing modes from the coupled first-order problem — the fix |
| `backend.check_mode_cutoff` / `RayleighPointWarning` | the fallback announcer, when the rebuild cannot run |
| `backend.check_finite` | the last-resort tripwire for non-finite results |
| `engine.solve(auto_detune=True)` | the default retry ladder — the "engineering mode" |

## The decision flow

Every 2D layer, every solve:

    eig(Omega2) -> (w, W);  q = sqrt(w);  V = (Omega_R @ W) / q      [the ordinary construction]

    is min|w| / max|w| < 1e-9 ?                                      [_COUPLED_REL]
    |
    +-- no ................. ordinary path. Bit-identical to what the solver always
    |                        returned; zero added cost. This is every solve that is not
    |                        within ~1e-9 of a graze — in practice, essentially all of them
    |                        (0 engagements over 40 random configs).
    |
    +-- unreadable ......... traced (jax.jit) or grad-carrying values: the rebuild cannot
    |                        materialize them, so the old detect-and-warn contract stands —
    |                        `check_mode_cutoff` warns below 1e-13 where it can read, and
    |                        the ladder (which runs eagerly around the trace) picks it up.
    |
    +-- yes ................ rebuild the grazing cluster:
                             restrict Omega_L / Omega_R to the cluster's invariant subspace
                             pair, eig the small coupled [[0, L_r], [R_r, 0]], and take one
                             mode per +-pair. Three exits:

                             (a) cluster representable  -> exact modes. No warning, no
                                 detune, the requested point solves. Done.
                             (b) coupled rates split below 1e-10 of the modal scale
                                 [_DEFECTIVE_REL] -> no exponential basis exists at working
                                 precision (the {1, z} regime). Raise FloatingPointError
                                 with the detune advice. LOUD, never quiet.
                             (c) E/H cluster sizes disagree (structure not understood)
                                 -> keep the ordinary construction + RayleighPointWarning.
                                 Detected, not guessed.

    solve, auto_detune=True (the default), catches (b) and (c):
        retry at wavelength * (1 + 1e-12)   [then theta * (1 + 1e-12) if needed]
        -> the retried solve has ratio ~1e-11, which is INSIDE the 1e-9 gate,
           so the rebuild solves the nearby point exactly -> AutoDetuneWarning
           names the knob and the value actually used; the solver keeps it.

## Three worked scenarios

**1. Representable graze (spacer n = 2.0, the common case).** The cluster is four modes
(orders +-4, two polarizations). The coupled rates come out at |mu| ~ 2e-9 — small, but
well clear of the defective line — and the rebuild returns modes satisfying both
first-order relations to eps. Result at exactly 500 nm: R = 0.272004739 against the
extrapolated truth 0.27200474, R+T-1 ~ 1e-10, no warning of any kind. The ordinary
construction returned R = 0.281 with R+T = 0.989 here — plausible-looking absorption on a
lossless stack. `auto_detune` never wakes up; this class has *graduated* from the ladder.

**2. Defective graze (spacer n = 1.5 on this platform).** The same geometry, but the
round-off happens to leave the coupled rates at |mu| ~ 5e-16 — the two exponentials of a
mode pair are numerically the same function, and no mode basis exists (measured: every
construction tried leaves 1e-4..1e-3). The rebuild refuses loudly; the ladder catches the
refusal, moves the wavelength by 1e-12 relative, and the retried point — now representable
— solves through the rebuild: R = 0.196001919 vs truth 0.19600190, with an
`AutoDetuneWarning` naming the wavelength used. Which stack falls in class 1 vs class 2 is
a *round-off lottery* (it can differ across LAPACK builds); what is not a lottery is the
contract: correct or loud, never quietly wrong.

**3. Everything else.** A solve at 520 nm, or any random structure: the gate is False,
the code path is the one that always ran, and the results are bit-identical to before the
fix. The rebuild costs nothing when it does not run, and ~one extra eig (~30% of a layer's
mode cost) only on the singular set.

## Why the thresholds sit where they do — the composition

These four numbers are a system, not four choices:

| Constant | Value | Sized by |
| --- | --- | --- |
| `solve_2d._COUPLED_REL` (rebuild gate) | 1e-9 | must contain the ladder's landing point (ratio ~1e-11 after a 1e-12 detune). With a 1e-12 gate the rescued retry fell back into the ordinary construction's degraded band and kept 1e-6..8e-6 errors — measured, then widened. |
| `solve_2d._DEFECTIVE_REL` (refusal line) | 1e-10 | the intrinsic error of an exponential basis is ~eps/(k0*d*mu); 1e-10 of the modal scale keeps the representable side at <= ~1e-6 for k0*d >~ 0.4. |
| `backend._CUTOFF_REL` (fallback warning) | 1e-13 | where the ordinary construction's error crosses ~1e-5 (err ~ eps/ratio); only meaningful on the rebuild's fallback paths now. |
| `engine._AUTO_DETUNE_REL` (ladder step) | 1e-12 | one order clear of the warning band, small enough to be physically invisible (5e-10 nm at 500 nm), and landing inside the rebuild gate — see row one. |

The dependency to respect: **ladder step -> landing ratio -> must be inside the rebuild
gate**. Shrink the gate or grow the step independently and rescued solves quietly lose
five digits.

## What this means for a user

- **Default (`auto_detune=True`)**: every point returns a correct answer. Exact Rayleigh
  points either solve exactly (silently) or come back as the 1e-12-detuned neighbour with
  an `AutoDetuneWarning` saying so. Nothing is ever moved silently.
- **`auto_detune=False`**: the requested point is solved exactly — and the outcome is
  *correct or loud*. A refusal means the exponential mode basis provably does not exist at
  that exact float; the message prescribes the one-ULP detune.
- **Measurement scripts probing this regime must pass `auto_detune=False`** — otherwise
  the library moves the exact point you meant to sample, and "exact vs detuned"
  comparisons come back identical (see `rayleigh-points.md` §4).
- **Oracles near the condition**: R(lambda(1+-h)) varies as R0 +- a*sqrt(h) (a Wood
  anomaly). Linear extrapolation misconverges, and the one-ULP neighbour computed with the
  *old* construction is itself inside the broken band. Use the two-sided sqrt
  extrapolation from far detunes (~1e-8 relative), as `rayleigh-points.md` §6 does.

## Not this process: the 1D conical x-cutoff

On the 1D conical path with `ky != 0`, an order with `kx_m**2 = eps*mu` in some layer is *not*
a Rayleigh point: its `kz` is `abs(ky)`, nothing grazes, and none of the machinery above is
involved. What breaks there is the conical path's two mode families, which collapse onto one
field; the layer is rebuilt in a regular basis inside the 1D builder
(`solve_1d.py::_regular_conical_modes`), exactly, with no warning and no detune, so the default
solve and `auto_detune=False` return the same answer — and under every trace the same:
`jax.jit`, `jax.vmap` (so a wavelength sweep written as a `vmap`), a `jax.lax.map` or `scan`
body, `jax.checkpoint` and `torch.func.vmap` rebuild the layer too, the gate's masks carried as
data. `rayleigh-points.md` §7 has the construction, the gate, how traces are handled and the
measurements.

On the 2D and first-order paths the same patterned point produces nearly parallel eigenvectors
at a nonzero root. Their builders replace each eligible pair with a regular basis and coupled
generator (`rayleigh-points.md` §§8–9). Coincident exceptional points can exceed that pair construction;
the eager cluster screen emits `ExceptionalPointWarning`, which does not trigger detuning.
The screen is silent under an abstract trace, so unresolved traced clusters can remain
inaccurate. A traced layer with more eligible pairs than its static slots returns NaN.

## Where the proofs live

- `docs/theory/rcwa/rayleigh-points.md` — every number above, measured, with the dead ends recorded.
- `tests/rcwa/test_rayleigh_guard.py` — the contracts pinned: correct-or-loud on the graze
  stacks, the exact solve on this platform, the refusal with its advice, the ladder's
  landing accuracy, silence off the condition, and the guard concreteness rules.

# FDTD — the time-domain solve beside the RCWA one

Landed 2026-08-31. Roadmap item 2: a second method reading the same `Stack` and `Incidence`,
so that one structure can be solved two ways and the answers compared without a translation
layer.

Target: `wavebit/_core/fdtd.py` — `solve_fdtd`, `component_property`, `cpml_coefficients`,
`_refuse_out_of_scope`; `wavebit/main.py` — `fdtd`; `tests/test_fdtd.py`.

**Scope.** Pure TE / TM at any polar angle in the classical mount (`phi = 0`), on a raster
structure periodic along one axis, with Drude/Lorentz dispersion and permeability —
deliberately the scope of
`wavebit/_core/solve_1d.py`, which this mirrors down to the amplitude conventions. Everything
outside it is refused by name; §6 lists the refusals and why each is a refusal rather than an
approximation.

---

## TL;DR

On the benchmark binary grating, at the same 5 nm grid Meep uses, the worst per-order gap
against the converged RCWA answer is **7.4e-4 (TE) and 6.2e-4 (TM)**, against Meep's 7.4e-4
and 6.9e-4 — the same number. Refine once more and it reaches 1.8e-4 / 1.9e-4, at a clean
second order throughout. The two methods agree in the limit, which is the only thing this
module was built to establish.

What the exercise actually taught, in the order the findings are worth reading:

1. **A dimensional constant in a scale-free solver is a bug with no symptom at the design
   wavelength.** The PML's frequency shift was written as a bare `0.05`, which is a reciprocal
   *length*. At 900 nm it happened to sit 7× above the operating frequency and merely cost an
   order of accuracy; at 7.2 µm, with the identical discretisation, it returned `R + T = 1.22`
   (§3).
2. **A wrong absorber looks exactly like a wrong discretisation, and I misdiagnosed it once.**
   Before the fix the error stopped falling at second order, and I concluded that a PML must be
   specified as a physical thickness rather than a cell count and wrote that down as a finding.
   It was not. A correctly graded CPML *is* dz-independent — 2.06e-6 → 1.37e-6 as `dz` goes
   20 → 2.5 nm — so cells is the right unit after all, and the thickness rule was a workaround
   for a broken layer that also made every run 4× slower than it needed to be (§3).
3. **`T[+1] == T[-1]` on a mirror-symmetric grating is a free, oracle-free error bar**, exact
   here at even cell counts. Meep does not pass it (8.7e-5 apart at its own res 200).
4. **Efficiencies converge second order while amplitudes did not**, twice, for two unrelated
   reasons — a half-cell monitor offset and a de-referencing that removed both legs of the
   phase instead of one. `|r|**2` cannot see either, so every efficiency test passed
   throughout (§5).

---

## 1. What makes the comparison worth making

**This is a solver to choose, not a test harness.** It is the better tool wherever a
broadband answer is wanted from one run — a spectrum costs one pass here and one RCWA solve
per wavelength there — or where a scene converges badly in a truncated Fourier series. Where
RCWA is better it is *much* better, and §2's cost column is the number to weigh.

That the two fail differently is a second dividend. RCWA truncates a Fourier series and
inherits every factorization choice that forces; FDTD discretises space and time and inherits
numerical dispersion, a staircased boundary and an imperfect absorber. Nothing is shared
except Maxwell's equations, so a defect surviving both is unlikely to belong to either — the
two paths cross-check each other for free, at no cost to either being a real solver.

Both readings need the same thing: the two must be comparable without a translation layer, and
they must be able to *disagree*. So the conventions are taken from wavebit rather than from
the FDTD literature, each one a place the two would otherwise silently part company:

| Convention | Taken from | What adopting the other one costs |
| --- | --- | --- |
| `+z` points **down**, into the stack | `docs/conventions.md`, `_core/fields.py` | Negates `E_z`, `H_x`, `H_y`; conserves energy, leaves every efficiency untouched. `fields.py` carried exactly this defect undetected for months |
| `e^{-iωt}` | `docs/formulation.md` | A phasor needs an `exp(+iωt)` kernel. `np.fft.fft` over the *time* axis applies the opposite one and conjugates every amplitude |
| `kx_m = n_inc sin θ cos φ + m·λ/Λ` | `_core/engine.py` `n_inc` | Reversing the order index mirrors `+1` onto `-1` **while conserving energy exactly** |
| TE all on `s`, TM all on `p`; `r_p` magnetic, `t_E = t_H·n_top/n_bot` | `_core/solve_1d.py:316-341` | A sign `\|r\|^2` cannot see |
| `H̃ = Z0 H`, `c = ε0 = μ0 = 1` | `docs/formulation.md` | A scale factor on every amplitude |

The amplitude conventions were not adopted, they were *derived and found to already agree*: a
single interface gives `Ey_r/Ey_i = (n1-n2)/(n1+n2)` and `Hy_r/Hy_i = (n2-n1)/(n1+n2)`, which
are `solve_1d`'s `r_s` and `r_p` outright. No sign patch sits between the two solvers.

---

## 2. The benchmark, and the three-way agreement

λ = 900 nm, period = 700 nm, one 300 nm layer, 50 % duty `n = 3.5` tooth in air, substrate
`n = 1.5`, normal incidence. In glass orders 0 and ±1 propagate; in air only order 0, so
`R[±1]` is identically zero. The same scene was solved by Meep (FDTD) and gyptis (FEM) in the
sibling lab repository on 2026-08-25; those numbers are quoted here, not recomputed.

Reference is the RCWA path at `truncation_order=80`, re-derived in this worktree. TM's `R0`
and `T0` still move by ~9e-6 out to 120, so the reference itself carries that much
uncertainty; TE is converged.

| | RCWA (ref) | this module, 5 nm | gap | Meep, res 200 | gap |
| --- | --- | --- | --- | --- | --- |
| TE `R0` | 0.204898 | 0.205633 | 7.4e-4 | 0.204162 | 7.4e-4 |
| TE `T0` | 0.352474 | 0.352724 | 2.5e-4 | 0.352583 | 1.1e-4 |
| TE `T±1` | 0.221314 | 0.220785 | 5.3e-4 | 0.221396 | 8.2e-5 |
| TM `R0` | 0.282393 | 0.283018 | 6.2e-4 | 0.282607 | 2.1e-4 |
| TM `T0` | 0.163805 | 0.164256 | 4.5e-4 | 0.164495 | 6.9e-4 |
| TM `T±1` | 0.276901 | 0.276292 | 6.1e-4 | 0.276318 / 0.276231 | 5.8e-4 / 6.7e-4 |
| Σ, TE / TM | 1 | 0.999928 / 0.999859 | | 0.999537 / 0.999650 | |

**Which Meep column.** Meep reports two — a mode decomposition and a raw flux — differing by
2.5e-4 (TE) and 3.4e-4 (TM) at res 200, the same order as the agreement being claimed. The
column above is the mode decomposition.

**Convergence**, at the shipped defaults (12 PML cells), worst per-order gap against RCWA:

| cell | TE | order | TM | order | Σ (TE / TM) | wall |
| --- | --- | --- | --- | --- | --- | --- |
| 20 nm | 1.16e-2 | | 8.89e-3 | | 0.99986 / 0.99834 | 0.3 s |
| 10 nm | 2.87e-3 | 2.02 | 2.32e-3 | 1.94 | 0.99997 / 0.99941 | 1.3 s |
| 5 nm | 7.35e-4 | 1.96 | 6.25e-4 | 1.89 | 0.99993 / 0.99986 | 8 s |
| 2.5 nm | 1.84e-4 | 2.00 | 1.90e-4 | 1.71 | 0.99991 / 1.00000 | 110 s |

Second order throughout, which is the point of the table — the number in any one row is a
property of the grid, but the *order* is a property of the method, and it is what says the
remaining error is discretisation and not a floor.

The cost column is the honest one, and it is the main thing to weigh when choosing between
the two solvers: 5 nm is 8 s a polarisation against the RCWA solve's fraction of a second, and
2.5 nm is two minutes.

**A band buys some of that back, but less than it sounds.** A whole spectrum comes out of one
FDTD run (§8) while RCWA re-solves per wavelength, so the two meet at
`N* = t_fdtd / t_rcwa_per_wavelength`. **Matched on accuracy rather than on settings** — the
cheapest `truncation_order` and the coarsest cell that each reach the same worst per-order
error, against an RCWA reference at `truncation_order=400`:

| scene | FDTD | its error | the cheapest RCWA that matches it | `N*` |
| --- | --- | --- | --- | --- |
| this grating, TE, 20 nm cell | 0.5 s | 1.2e-2 | fto 5, 0.51 ms | 1009 |
| this grating, TE, 10 nm cell | 2.0 s | 2.9e-3 | fto 10, 0.71 ms | 2850 |
| **60 nm metal grating, TM, 10 nm cell** | 2.4 s | 1.6e-3 | fto 60, 21.4 ms | **112** |

**Half of that came from one constant.** The clearance between the structure and each of the
TFSF plane, the monitors and the PML was a full wavelength, chosen so a grating's evanescent
orders would decay before the monitor. They do not need to — the spatial transform separates
the orders exactly, so an evanescent order sitting on the monitor plane cannot contaminate a
propagating one. Halving it is *more* accurate as well as twice as fast (7.37e-4 / 22.0 s
against 6.57e-4 / 13.9 s on the dielectric; 7.43e-4 / 31.9 s against 7.01e-4 / 15.9 s on the
metal), because a shorter outgoing leg carries less of the analytic-versus-discrete `kz`
mismatch. Below a quarter wavelength it stops paying. `pad_scale` moves it.

**Matching on accuracy is the whole point of the table**, and an earlier draft of this document
did not: it compared a 10 nm FDTD run against `truncation_order=80`, which was a *thousand
times* more accurate than it needed to be, and reported a crossover near a hundred. The real
figure on a lossless dielectric is nearer a thousand.

`N*` is a property of the scene. It moves toward FDTD as the RCWA side gets harder — the metal
row is 19x better than the dielectric at the same cell size, because RCWA in TM on a metal
converges slowly and non-monotonically (its error at `fto` 40 is *worse* than at 20) — and away
as the FDTD grid is refined, which is why the 5 nm rows are worse than the 10 nm ones.

---

## 3. Two constants, and a misdiagnosis

Both material-dependent quantities in the CPML were wrong, in the obvious direction, and
neither announced itself at the design wavelength.

**`σ_max` had `sqrt(eps_r)` upside down.** In a *stretched coordinate* `σ` is normalised by
the vacuum permittivity, so a wave accumulates `n·∫σ` and a fixed round-trip target needs
`σ_max ∝ 1/sqrt(eps_r)`. Taflove & Hagness eq. 7.61 carries the factor the other way up
because there it is a *material* conductivity acting against the local permittivity;
transplanting that spelling into the stretch overshoots by exactly `eps_r`. An independent
argmin over the conductivity scale settles it — the optimum tracks `1/n`, not `1/eps_r` and
not `n`:

| n | 1.0 | 1.5 | 2.0 | 3.0 | 3.5 |
| --- | --- | --- | --- | --- | --- |
| measured argmin | 0.80 | 0.60 | 0.44 | 0.35 | 0.25 |
| `1/n` | 1.000 | 0.667 | 0.500 | 0.333 | 0.286 |

The cost, on a lossless slab with an `n = 3.5` substrate: `T = 1.037` and `R` wrong by 27 %.
Nothing in the benchmark exceeds `n_bot = 1.5`, where the factor is 2.25 rather than 12.25.

**`α_max` was a bare `0.05`, which is a reciprocal length.** Everything else in the module is
dimensionless or scales with a length, so this one constant made the answer depend on whether
the caller wrote nanometres or microns. At 900 nm it works out to 7.16 ω; the measured optimum
is flat between 1 and 2 ω:

| `α/ω` | 0 | 0.5 | 1.0 | 2.0 | 4.0 | 7.16 (was shipped) |
| --- | --- | --- | --- | --- | --- | --- |
| air | 1.84e-6 | 1.76e-6 | 1.48e-6 | **1.15e-6** | 3.89e-6 | 2.13e-4 |
| glass | 1.88e-6 | 1.86e-6 | 1.63e-6 | **1.15e-6** | 2.42e-6 | 2.67e-4 |
| n = 3.5 | 2.12e-6 | 2.28e-6 | 2.12e-6 | **1.80e-6** | 4.73e-5 | 1.95e-3 |

Zero is *not* safe — the shifted pole is what absorbs the evanescent field a grating radiates
into the layer, and that part of the design was right; only its magnitude was ~7× oversized.
Scaled in the same nanometre convention the module ships with, the old constant returned
`R + T = 1.0124` at λ = 1800 nm and **1.2234** at λ = 7200 nm, on an identical discretisation.

**The misdiagnosis.** Before either fix, a sweep of PML depth at a fixed grid showed the error
falling to a plateau at about a quarter wavelength, and a convergence study showed the order
dropping from 2.04 to 1.31 on the last refinement. I concluded that a PML has to be specified
as a physical thickness rather than a cell count, since a cell count thins as the grid refines
— and wrote that into this document as the headline finding. **It was wrong.** The 1D probe
below is the instrument that settles it, and a correct layer is flat in `dz`:

| `dz` | 20 nm | 10 nm | 5 nm | 2.5 nm | effective order |
| --- | --- | --- | --- | --- | --- |
| before | 9.11e-3 | 2.24e-3 | 1.03e-3 | 6.85e-4 | ~`dz^1.25` |
| after | 2.06e-6 | 1.63e-6 | 1.42e-6 | 1.37e-6 | flat |

So cells is the right unit, the thickness rule was a workaround for a broken absorber, and it
cost 45 PML cells where 12 do better — a 4× slowdown bought nothing. The lesson generalises
past this module: **a floor under a convergence study is a hypothesis about the absorber, and
the way to test it is a direct reflection measurement, not a second convergence study.** The
1D probe (a short grid minus a long one at a common monitor) takes seconds and has ten orders
of resolving power; the convergence study that produced the wrong answer took an hour.

Two more constants moved with them: `m = 4` beats the textbook `m = 3` by ~140× once `α` is
right (9.12e-9 against 1.29e-6), and `κ_max` dropped from 5 to 2. `κ_max` is the one constant
here **not** settled by measurement — the 1D probe prefers 1, and the classical case for a
larger value is grazing and evanescent absorption that a 1D probe cannot see.

---

## 4. Subpixel smoothing, and what cannot arbitrate it

The material is averaged onto each Yee component's own cell, **harmonically along the axis the
component points along and arithmetically along the others**. Subpixel smoothing replaces the
stepped material with an effective tensor (Farjadpour 2006, Kottke 2008): arithmetic mean
tangential to the interface, harmonic mean along its normal. For an interface normal to `x`
that tensor is `diag(harmonic, arithmetic, arithmetic)`, and `E_x` — the only component
reading `eps_xx` — is exactly the one this rule averages harmonically along `x`. The other
components see a constant along `x` there, where the two means coincide. So the rule
**reproduces the full tensor exactly for axis-aligned interfaces**, and the geometry this
module accepts — a z-stack of piecewise-constant x rasters — has no other kind, so there is no
off-diagonal term being dropped.

The averages are exact rather than oversampled: the geometry is piecewise constant, so the
integral is a sum of segment overlaps and an oversampled estimate would only add a second
discretisation to converge.

Worth, on a slab whose interfaces genuinely cut cells, against point sampling (`subpixel=False`):

| `dz` | subpixel | point-sampled |
| --- | --- | --- |
| 38.9 nm | 8.07e-2 | 2.87e-1 |
| 20 nm | 1.04e-2 | 1.93e-2 |
| 10 nm | 1.49e-3 | 6.75e-3 |
| 5 nm | 2.73e-4 | 3.41e-3 |
| order | ~2.5 | ~1.0 |

**What cannot see it.** A planar stack at normal incidence is blind to both harmonic averages
— `E_z` is identically zero there, and a harmonic mean along `x` of a constant is that
constant. Worse, the `harmonic_axis='z'` branch is unreachable by any test whose layer
thickness is a whole number of cells, which the benchmark's 300 nm is at every round
resolution. Make one thickness non-commensurate and give it x structure and the branch becomes
load-bearing: 1.43e-3 with it against 6.55e-3 without, on a 317 nm layer at a 10 nm grid.
`tests/test_fdtd.py` carries that case for exactly this reason. Same shape of problem as
`docs/nvf-normal-methods.md`: the cheap test that validates everything else is precisely the
one that cannot judge the smoothing.

---

## 5. Four defects the efficiencies could not see

**The TM monitor's half cell.** `Hy` lives half a cell below the z node, and the amplitude was
de-referenced to the node's coordinate. The result is a `k0(kz_out + kz_inc)·dz/2` phase on
every TM amplitude — which cancels in `|r|**2`. So `R` and `T` converged at second order while
`r_p` fell as `dz**0.6`: 1.29e-1, 3.39e-2, 8.19e-3 became second order only after the fix, at
which point `r_p`'s error matched `r_s`'s to 1 % (8.19e-3 against 8.26e-3), as it must — at
normal incidence a planar stack has no plane of incidence and the two are degenerate.

**The de-referencing removed both legs.** Normalising by the incident spectrum sampled *at the
monitor* looks like it cancels the incident wave's numerical dispersion. It does — and then
adds the outgoing leg's, for a net factor of two rather than the cancellation it appears to
be. Sweeping only the monitor's distance, the phase residual tracks `2(k0 - κ_yee)·z_ref` at a
ratio of 0.96 and the one-leg form at 1.92. Reading the incident at `z = 0`, where the
amplitude is *referenced*, leaves exactly one leg. `|r|` is unaffected either way.

**The 1D auxiliary field's sign.** The incident wave is generated on a 1D grid sharing `dz`
and `dt` (Taflove & Hagness §5.8) so its numerical dispersion matches along z exactly. The
pair it integrates is `(Ey, Hx)` for TE but `(Ex, -Hy)` for TM, and that minus sign has **two**
consumers — the TFSF correction and the incident amplitude the result is normalised by.
Getting it wrong in only one flips the sign of `r_p` and leaves every efficiency intact.

**Evanescent orders were amplified without bound.** An order evanescent in the superstrate has
decayed by `exp(k0|Im kz||z_ref|)` before it reaches the monitor, and de-referencing undoes
that — amplifying whatever is left, which past some depth is float64 round-off. Measured
against the RCWA amplitudes on a duty-0.3 grating: a gain of 1.1e8 still gives 6 % accuracy,
4.5e12 is wrong by 5×, 1.4e17 by 3e5, and a caller who mirrored an RCWA `truncation_order=80`
order set got `inf` and `nan`. The efficiency is identically zero for these orders either way,
so nothing downstream of `R` could notice. They are now `nan` above a measured gain bound.

**The instrument for all of this needs an asymmetric structure, and almost nothing is one.**
A mirror-symmetric grating at normal incidence gives `T(+1) == T(-1)` exactly, so the two
orders are indistinguishable and no comparison can see the x origin, the sign of the order
index, or any amplitude phase — and the benchmark grating, a uniform slab, and even a duty-0.3
rectangle (symmetric about its own tooth centre) are all in that class. A two-step profile
`[3.5², 2.0², 1, 1]` is not: it gives `T(-1) = 0.7207` against `T(+1) = 0.0838`. On it the FDTD
and RCWA amplitudes agree to 2.1e-3 with `arg` differences of 0.007 rad, against the π a
half-period origin shift would put on every odd order. That is the check that says the two
solvers share a frame, and wavebit moved every user-facing coordinate to the centre of the
cell on 2026-08-31 — this module was unaffected because the raster's index convention and the
transform's `[0, 1]` integration did not move with it, but that is now pinned rather than
reasoned about.

The common instrument for the first two is **comparing amplitudes, not efficiencies**, against
a closed form. The common instrument for the last is asking what a quantity means when its
efficiency is identically zero.

---

## 5b. Oblique incidence

A periodic cell at oblique incidence is periodic only up to a phase: `E(x + Λ) = E(x)·exp(i kx Λ)`.
A real-valued time-domain field cannot carry that factor, which is why most periodic FDTD codes
reach for a split-field formulation. **This module does not need one**, for a reason specific
to what it is: it extracts at a single wavelength, so `kx` is a constant, and a complex field
with the Bloch phase applied at the one column that wraps is exact. The cost is a complex
dtype, and normal incidence keeps a real one — the phase is identically 1 there, and paying
for it would double the cost of most runs.

The auxiliary incident grid stays one-dimensional. The medium it lives in is uniform along x,
and `exp(i kx x)` is an eigenvector of the *discrete* shift, so every x derivative in it
collapses to one multiplier, `kx_op = 2i·sin(kx·dx/2)/dx`. Using the continuum `i·kx` instead
would put the two grids on different dispersion relations and degrade the TFSF cancellation
from round-off to the mismatch between them. The third field component (`Hz` for TE, `Ez` for
TM) appears only here: the `(E, H, ε, μ)` duality that lets one 1D system serve both
polarisations at normal incidence does not survive the transverse term, because `ε ≠ μ`.

Measured against RCWA on the benchmark grating, 10 nm grid, 120 optical cycles:

| θ | TE | TM |
| --- | --- | --- |
| 0° | 2.9e-3 | 2.3e-3 |
| 15° | 1.7e-3 | **1.4e-2** |
| 30° | 3.5e-3 | 3.7e-3 |
| 45° | 1.8e-3 | 3.6e-3 |

**Run length, not resolution, is what an oblique run costs.** The scattered wave leaves at an
angle and takes longer to clear. At 15° with 60 cycles the residual is 5.9e-2 and the answer
is wrong by 3.4e-2; at 120 it is 4.5e-3. `UnsettledFieldWarning` fires above a 1e-2 residual,
because a truncated running DFT returns a number shaped exactly like an efficiency.

**15° is the one bad row and it is not an accident.** There `kx_{-1} = sin 15° − 900/700 =
−1.027`, just past the light line in air — a nearly-grazing order 1.6° away from the Rayleigh
condition at 16.6°, and a nearly-grazing order is precisely what a PML graded along `z` alone
absorbs worst. Expect the near-Rayleigh band to be the worst case at any angle, and check the
residual there rather than trusting the number.

---

## 6. The refusals

Eleven, each refused by name. The common thread: every one returns a *plausible* number if it
is quietly ignored, and a plausible wrong number from the cross-check solver is worse than no
cross-check at all.

| Refused | Why not just implement it |
| --- | --- |
| `psi` / `chi` / `polarization` incidence | A different problem, not a different spelling — same reason `solve_1d` is separate from `solve_1d_conical` |
| complex `theta` | Evanescent incidence has no propagating incident wave to normalise against, and the TFSF plane injects one by construction. The RCWA path answers this — see `docs/branch-cuts.md` |
| `phi != 0` (conical mount) | It couples the two polarisations, and this is the pure TE/TM formulation — the same split `solve_1d` and `solve_1d_conical` make. The classical mount leaves them decoupled at every `theta`, which is what makes §5b answerable at all |
| `incidence_side='bottom'` | The mirror the ETM applies has no counterpart here yet |
| complex permittivity | One complex number describes a material at **one** frequency, and a run carries every frequency in its band. Give the dispersion as poles instead (§8b) — that *is* the frequency dependence, rather than one sample of it |
| `Re(ε) ≤ 0` | A metal below its plasma frequency — the same dispersion problem, and no real wave speed, so no stable time step exists |
| analytic shape structure | Rasterising it would put a second geometry builder beside `draw.slice_structure`, which is NumPy-only, square-only and refuses a traced shape parameter. The two would disagree about one scene |
| `(n_layers, Ny>1, Nx)` | A crossed grating needs a 3D FDTD |
| oblique `lattice` | The Yee grid is Cartesian; a skewed cell needs the metric in every update |
| a run shorter than the source | The source is not finished emitting until 48 optical cycles. A 20-cycle run returns 0.033 where the answer is 0.543 — an efficiency-shaped number, so it is refused rather than reported with a residual nobody reads |
| `\|m\| ≥ nx/2` | Past Nyquist the x transform aliases one order onto another and returns a plausible amplitude — with a flipped sign on TM, from the half-cell sample offset |
| `backend` | §7 |

Beyond the refusals, the solve runs the same `check_physical` the RCWA path runs. Every defect
this module has had announced itself there first, as `R + T` between 1.1 and 106.

---

## 7. Why NumPy only

The time loop is a Python `for` over ~10⁴ steps. Under `jax.jit` it unrolls to a trace of that
length; under `jax.grad` or torch autograd it materialises a tape of O(steps × grid). Crossing
either needs a per-backend `scan` — `jax.lax.scan`, an eager loop elsewhere — and there is no
such primitive in `wavebit/_core/backend.py`, nor any loop construct in `_core` that is not a
Python `for` over a small static layer count.

The field updates themselves would port unchanged: they are slice-shifted differences and
`xp.concat`, needing no index assignment (which is the one array operation with no spelling
common to all three backends). It is only the loop that has nowhere to live. Adding that
primitive is a design decision rather than a continuation of this work, so `backend` is
refused by name instead of half-supported.

---

## 8. A spectrum out of one run

This is the thing a time-domain solve does that a frequency-domain one cannot, and with the
module read as a solver rather than a harness it is the main reason to choose it. The pulse
already carries every frequency in its band; a second wavelength costs one more running sum
over a monitor plane — O(nx) a step — against a whole new RCWA solve. `wavelength=` takes a
sequence and returns one `Result` per entry:

    results = wavebit.fdtd(stack=..., incidence=..., wavelength=[860., 900., 940.])

It is an identity, not an approximation: the band run and three separate single-wavelength
runs agree to 3e-3, which is the gap between two source *bandwidths* rather than between two
answers. A band buys itself a shorter pulse — a Gaussian of temporal width `w` has a 1/e
frequency half-width of `1/(πw)` — so the pulse is sized from the requested span. That is also
what makes the band edge safe: every amplitude is divided by the incident spectrum, and with a
fixed pulse width an incident spectrum 2.4e-7 of its peak returned `R = 105` on a scene whose
true `R` is 0.04. Sizing the source keeps the weakest wavelength above 0.2 of the peak, and
`tests/test_fdtd.py` pins that invariant rather than the unreachable guard behind it.

**At oblique incidence a band is a fixed-`kx` sweep, not a fixed-angle one.** The grid holds
one in-plane wavevector and `kx = k0·n·sin θ` carries `k0`, so the polar angle drifts across
the band — 28.5° at 860 nm and 31.5° at 940 for a 30° centre. `angles_deg` in the diagnostics
reports it. This is the method rather than this code: any periodic FDTD holding one transverse
wavevector does the same, and escaping it is what a split-field formulation is for (§9).
Normal incidence is unaffected, since `sin 0` is zero at every wavelength, so **a fixed-angle
band is available there and only there**.

A defect only a band at oblique incidence could expose lived here: the extraction reused
`grid.kx_inc`, normalised by `k0` at the *centre* wavelength, at every wavelength in the band.
The physical in-plane wavevector is what is held fixed, so the normalised value scales with
the wavelength — the same statement as the angle drifting. Reusing the centre's value cost
2.3e-2 at a band edge against 3.6e-3 at the centre, and is identically zero at normal
incidence. The instrument that settles it is solving the RCWA path at the **drifted** angle:
5.4e-3 after the fix.

---

## 8b. Dispersion — Drude and Lorentz

A time-domain solve cannot take a complex permittivity: one complex number describes a
material at one frequency, and the pulse carries every frequency in its band. The frequency
dependence has to be given as the thing it is, a pole model, which is then integrated
alongside the fields by an **auxiliary differential equation**:

```
d²P/dt² + γ dP/dt + ω0² P = f E
```

In `e^{-iωt}` that gives `P = f E / (ω0² − ω² − iγω)`, so

```
ε(ω) = ε_inf + Σ_p  f_p / (ω0_p² − ω² − i γ_p ω)
```

**One equation covers both models.** At `ω0 = 0` it is Drude, `ε_inf − f/(ω² + iγω)` with
`f = ω_p²`; at `ω0 ≠ 0` it is Lorentz. A metal is a Drude pole plus however many Lorentz poles
its interband structure needs. The sign works out so `Im(ε) > 0` is loss, which is wavebit's
convention and not a free choice. Central differences on both derivatives make the recursion
explicit — `P^{n+1}` reads `E^n`, never `E^{n+1}` — so nothing is solved simultaneously.

The scene is given as three pieces: `structure` holds the real `ε_inf`, `dispersion` is a
same-shape integer raster marking which cells are dispersive (0 = none), and `poles` maps each
id to its `(ω0, γ, f)` list.

Measured on a 30 nm Drude slab (gold-ish: plasma wavelength 136 nm, so `ε = −42.7 + 2.3i` and
an 11 nm skin depth) against the exact transfer matrix at that `ε`:

| cell | `R` gap | `T` gap | absorbed (exact 0.0228) |
| --- | --- | --- | --- |
| 20 nm | 3.7e-3 | 3.5e-3 | 0.0230 |
| 10 nm | 5.9e-4 | 7.6e-4 | 0.0230 |
| 5 nm | 8.5e-5 | 1.9e-4 | 0.0229 |

and on a 60 nm metal grating against the RCWA path, which is handed the same material as a
complex `ε`: worst per-order gap 2.7e-3 at 10 nm and 6.1e-4 at 5 nm, with both solvers
absorbing 1.3 %.

**This is where dispersion and a band multiply each other, and it is measured in §2.** One run
gives the whole spectrum, and each wavelength in it sees the permittivity the pole model has
*there*, while RCWA is re-solved per wavelength and handed a different `ε` each time. The
crossover drops from 2850 wavelengths on the lossless dielectric to **112** on the metal at the
same cell size, and that is the best figure measured anywhere here.

**It is still 112, and the reason is worth naming.** A metal makes RCWA converge badly, but it
makes the FDTD grid requirement worse by more: the wave inside this metal is 138 nm long and
its skin depth is 11 nm, and the grid is uniform, so the whole domain pays for them. On the
scene above the metal is 60 nm of a 1165 nm span — **95 % of the fine grid is spent outside the
material that demanded it**, even after the clearance was halved (§2). A graded or subgridded mesh is therefore a far bigger lever on
this module's usefulness than any of the formulation work in §9, and it is the one thing that
could move `N*` into the range where a caller would reach for FDTD by default.

**What is approximated.** The pole strength is weighted by the fraction of each Yee cell the
dispersive material fills. That is exact in the same limit the arithmetic permittivity average
is exact — tangential to the interface — and an approximation normal to one, where the
harmonic rule applies to a static permittivity and does not transfer to a frequency-dependent
material. Half-spaces are not dispersive.

---

## 8c. Permeability

Implemented 2026-09-01, the day `644f881` lifted the same refusal on the RCWA path's 1D and
conical operators. Until then this module refused `mu` *for parity* — there was no reference to
compare a magnetic 1D grating against. That justification is now withdrawn, and with it the
refusal.

`mu` arrives the way it arrives on the RCWA side: a raster field beside `structure` for the
layers, and inside a `Material` for the two half-spaces. So one `Stack` reaches both solvers
unchanged, which is the whole design.

The placement is the duality's, exactly as `docs/magnetic-materials.md` §11 puts it for the
RCWA operators: **`Hx` and `Hz` divide by `mu_xx` and `mu_zz` on TE, `Hy` by `mu_yy` on TM**,
each averaged harmonically along its own axis because `B` normal is continuous where `D`
normal is. An index becomes `sqrt(εμ)`, so `mu` reaches every `kz`, the order grid, the
Courant bound (speed is `1/sqrt(εμ)`) and both admittances — `kz/μ` on the s channel and
`kz/ε` on the p channel, which is the whole content of a magnetic half-space.

**Energy conservation is not a check here, and the instruments that are:**

| instrument | what it catches | measured |
| --- | --- | --- |
| An impedance-matched slab (`ε = μ`) reflects **nothing**, at any thickness, however large its index | anything that drops `mu` from an H update | `R` = 4.6e-7 against a control of 0.358 — 774,000x |
| Same index, different impedance (`ε=2, μ=3` against `ε=6, μ=1`) | a solver that reduced the pair to an index | `R` 0.150 against 0.358 |
| The transfer matrix, generalised to carry `mu` | the admittances and the normalisation | 1.5e-4 on a magnetic half-space |
| The RCWA path, on the same magnetic grating | everything at once | 5.9e-4 (TE), 3.4e-3 (TM) |

The first two are exact identities and need no oracle. `R + T = 1` is deliberately absent: it
passes for magnetic code that is wrong — on the RCWA side all six path/algorithm combinations
conserved energy on the first smoke test, before any real oracle ran.

**Two test-design traps, both of which caught this file first.** A layer at a *half* wave is
nearly invisible, so both arms of the same-index comparison collapsed onto the bare
substrate's 0.04 and measured nothing; and a thickness that straddles a cell spoils the
discrete impedance match, turning the matched slab's `R` from 4.6e-7 into 1.1e-3. Both
thicknesses are now a quarter wave *and* a whole number of cells.

Also worth stating because an earlier draft asserted the opposite: **at normal incidence TE and
TM stay degenerate in `|r|²` whether or not the material is magnetic**, since `r_p = -r_s`. A
permeability does not split them there.

---

## 8d. A conical mount at normal incidence

`psi` is accepted at `theta = 0` for any `phi`, and the result carries `given` / `s_inc` /
`p_inc` — the contract `solve_1d_conical` offers. `theta != 0` with `psi` is refused by name.

**At normal incidence a conical mount is not a different solve. It is a different output
frame.** The in-plane wavevector is the origin there, so `ky` is exactly zero whatever `phi`
is; the six-component stepper is never reached and both legs come out of two ordinary TE/TM
runs. What `phi` decides is which channel each order lands in.

And that is not one rotation, which is the whole content of this section. The zeroth order's
in-plane wavevector is the origin, so its azimuth is `phi` by convention — a wave going
straight down has a degenerate plane of incidence, but its *polarization* still needs a
reference direction. Every order with `kx_m != 0` has azimuth 0 or pi. At `phi = 90 deg` the
two are a right angle apart, so one x-polarized field reads as pure **s** in the specular
order and pure **p** in the first orders:

| | `R_s[0]` | `T_s[0]` | `T_s[+1]` | `T_p[+1]` |
|---|---|---|---|---|
| FDTD, pure-s incidence at `phi = 90 deg`, 10 nm grid | 0.2848 | 0.1656 | 0 | 0.2746 |
| RCWA on the same scene as `tetm='tm'` | 0.2823 | 0.1639 | — | 0.2769 |

The `s`/`p` split is exact — `T_s[+1]` is identically zero, not small — because it is decided
by the frame geometry alone. The *values* carry the ordinary 10 nm discretisation error.

A projection that used one frame for every order would put both in the same channel, conserve
energy exactly, and be wrong. `conventions.order_azimuth` is **called** rather than respelled;
its docstring is explicit that filling the origin with `0` collapses the s/p mixing silently
and only when `phi != 0`.

Measured against `solve_1d_conical` at `phi = 0/30/45/60/90/135 deg`, all three legs, all four
channels, every order: worst 2.7e-3 at a 10 nm grid, falling 8.2e-3 / 2.1e-3 / 5.9e-4 / 1.8e-4
across dz = 20 / 10 / 5 / 2.5 nm — an order of 1.94 / 1.86 / 1.73. The residual is the Yee
grid's; a projection defect leaves a floor instead.

### The leg reversal, and why only one azimuth sees it

`s_hat` is common to the up- and down-going legs, but `p_hat = k_hat x s_hat` reverses with
`k_hat`. So the transmitted frame is the reflected one with `sin varphi_m` negated — and
**only the specular order can carry that**, since every other order at `theta = 0` has
`sin varphi_m = 0`.

Both obvious azimuths pass either way. At `phi = 0` the term is absent; at `phi = 90 deg` it
is the entire answer. `phi = 45 deg` is what separates them, and with the reflected rule used
for both it returned `T_p[0] = 0.241` against a true 0.017 — while `R` was correct to 3e-3 and
`R + T` was 1.000.

**The weights were fitted to complex amplitudes across three azimuths and both legs, not to
efficiencies, and that is the only reason this was caught.** Twelve complex numbers fixed the
reflected rule; the transmitted rule needed its own six. An efficiency is quadratic and blind
to exactly the sign most likely to be wrong. Same family as the TM monitor offset in §5, and
as `per-order-sp-frame-half-turn` in the RCWA path.

## 8e. A conical mount away from normal

`psi` with `theta != 0` and `phi != 0` is answered by the six-component stepper, run once per
eigenpolarisation — s and p are the two eigenpolarisations of a uniform medium at fixed
`(kx, ky)`, so a leg driven in one stays in it on the way in and the grating is what mixes them.
Against `solve_1d_conical` at theta = 30 deg, phi = 40 deg, psi = 30 deg, all three legs, all
four channels, every order: 2.8e-3 in efficiency and 1.2e-2 in amplitude at a 10 nm grid.

**The p channel is not the transverse projection.** `p_hat = k_hat x s_hat` has a `z` component;
eliminating `E_z` through `E . k_hat = 0` and using `kz**2 + k_par**2 = n**2` collapses it to

    E . p_hat = -+ (n / kz_m) * (c Ex + s Ey)

and that `n / kz_m` is a per-order `1 / cos(theta_m)`, equal to 1 only for the specular order at
normal incidence — exactly the case §8d covered, which is why that path could not have found it.
A pure amplitude factor: no efficiency or energy check sees it.

**Three things blocked it and each was silent.** `solve_fdtd` never passed `phi` to `_Grid`, so
`grid.ky_inc` was `0.0` on every reachable path and the `conical` branch was dead code. The TFSF
source could not inject a conical wave — `_time_loop` built `_Incident1D` passing neither `ky`
nor `drive`, so the four injection lines read the wrong triple through the `inc.e`/`inc.h`
accessors. And `_half_space_kz` had no `ky`, so it and `_propagating_window` disagreed about
which orders propagate.

The instrument that settles the source needs no oracle: **on a vacuum scene the scattered-field
region must stay empty.** As a fraction of the incident amplitude at phi = 30 / 45 / 90 deg it
was 6.9e-2 / 9.5e-2 / 6.7e-2 against 3.7e-17 at phi = 0; it is ~3e-15 everywhere now. Thirteen
orders — and nothing else in the module could have found it, because the stepper was stable and
decaying and the public path with the refusal lifted returned `R + T = 0.9966`.

**The three leg signs were found by the mixed leg, not the pure ones.** Each leg is normalised
by its own incident amplitude, so a global sign on either leaves every `|r|**2` in that leg
exactly right. With them wrong, `s_inc` and `p_inc` matched RCWA to 2e-3 in every channel while
`given.R_s[-1]` read 0.1795 against a true 0.0197 — and `R + T` stayed 1.0005.

### What the convergence metric was measuring instead

The residual first looked sub-second-order, 1.58 / 1.37 in amplitude against classical controls
of 2.02 / 1.78 (TE) and 1.89 / 1.85 (TM) at the same angle. It was the metric, twice over.

**An evanescent order's amplitude is round-off the de-referencing multiplied up.** Order +1 has
`k_par**2 = 2.89` against `n_bot**2 = 2.25`, so it is evanescent in both half-spaces; its `t_s`
gap reads 3.1e-3 / 3.6e-4 / 6.2e-4 and gets *worse* with resolution, while `R` and `T` are
exactly zero there.

**An order near a Rayleigh point converges slowly for arithmetic reasons.** The default scene's
m = -1 sits at `kz_top = 0.286`, where the p channel multiplies by `n/kz` = 3.5 and
`kz = sqrt(n**2 - k_par**2)` has sensitivity `1/kz` on top of that. A large coefficient, not a
low order.

With R compared only where an order propagates above, T only where it propagates below, and a
scene chosen by scanning `(theta, phi)` for the largest minimum `|kz|` margin: 2.05 / 1.87.

**And the scene that makes the claim cleanest cannot see the defect.** The two monitors need two
demodulation kernels — `Ex` sits on the x half-steps and `Ey` on the nodes — and using one is a
per-order phase `exp(-i k0 kx_m dx/2)`, *first* order in dx, mostly cancelled in `|r|**2`. Its
size scales with `|kx_m|`. On the margin scene the only order propagating in air has
`kx_0 = 0.25` and the single-kernel version reads a convergence ratio of 4.24 against a true
4.15 — no separation at all. The default scene's m = -1 carries `kx = -0.903` and separates 3.62
from 2.39. **A scene chosen for margin and a scene chosen for sensitivity are not the same
scene**, and the test file carries both.

## 8f. A graded z mesh, and why it is off by default

`grading=True` builds a z mesh that keeps the fine cell in the structure and coarsens toward the
cell each half-space would have asked for on its own. `x` stays uniform and that is not
negotiable: the order extraction is an FFT over x.

The motivation was a real waste. On a Drude metal grating 112 of 178 z cells sit in air above
the structure and 57 in the substrate below, where the index is 1 and 1.5 while the cell size
was set by the metal's 6.5 — 94 % of the cells outside the thing that required them.

**An earlier revision of this document said fixing that "could change the crossover by an order
of magnitude". It does not, and the gap between that sentence and the measurement is the most
useful thing in this section.**

### What it actually buys

Cell count falls 2.2x on the dielectric benchmark and 4.5x on the metal one — but the
transition has to be paid for. A mesh step is an optical interface and it reflects: measured at
the module's own 20 cells per wavelength, a 4.5x jump taken in ONE cell reflects 5.9e-3 in
amplitude, eight times the module's worst per-order error and 540 times the PML's own. Spread
over 40 cells it is 6.0e-4. Those cells come out of the saving.

And **grading buys no time-step relief at all.** The Courant bound follows the smallest cell,
which is the structure's and does not move; the true spectral radius of the non-uniform operator
lands at 0.9993–0.9999 of `4/min(dz)**2` on six meshes including a 1:10 jump, so `min` is tight
rather than conservative.

| resolution | dz | cells saved | time saved |
|---|---|---|---|
| 0.05 | 20 nm | 1.29x | 1.11x |
| 0.10 | 10 nm | 1.62x | 1.32x |
| 0.20 | 5 nm | 2.22x | 1.83x |
| 0.40 | 2.5 nm | 3.20x | 3.15x |

At coarse grids the module is NumPy-call-bound and the cells do not buy time; at the resolution
someone would run for a converged answer the two match.

### The comparison that decides it

**Equal cell size is the wrong comparison, and it is the one that looks best.** Holding the
*answer* fixed instead — time to reach an efficiency error of 1e-3 on the metal grating:

| | uniform | 1.03 | **1.06** | 1.12 |
|---|---|---|---|---|
| time | 5.99 s | 6.60 s | **3.78 s** | 9.62 s |

A real optimum, and 1.6x faster than uniform. **On the dielectric benchmark it loses**: the
error grows 2.7x at the same cell size and buying that back costs more in cell size than the
grading saves in cell count. So it is off by default and `GRADE_RATIO` is 1.06 — it pays where
the index contrast is large and the structure occupies little of the domain.

### Two defects only a convergence study could see

`n_max` was first derived back from the cell size, which assumes the caller asked for exactly
`CELLS_PER_WAVELENGTH`. A caller who *refines* `resolution` was therefore read as having a
higher-index material and the padding was coarsened to match, pinning the coarse cells at 45 nm
however fine the structure got. The graded run **floored at 2.3e-3 and stopped converging**
(order 0.42 at the finest grid) while the uniform path went on to 1.4e-4. A single-resolution
comparison shows this as a slightly worse number, not as a floor.

And the auxiliary 1D grid was left on the scalar cell size while the 2D grid graded. The
efficiencies came back *better* than the uniform run's — 3.5e-3 against RCWA — while every
amplitude was wrong by 0.52. `R + T` never noticed. Same family as §5 and §8d: an efficiency is
quadratic and blind to exactly the phase this kind of mistake produces.

## 8g. A crossed grating

`structure` may carry a y axis and `period` a second entry; `R` and `T` come back on the
two-dimensional order grid `(orders_v, orders_u)` that `Result` has always described. A
`(n_layers, 1, Nx)` raster is the y-invariant case and takes the 2D machinery unchanged.

**Most of it was already written.** The conical mount put all twelve curl terms in the stepper,
so the third dimension turns four algebraic `i*ky` multiplications into staggered y differences
and adds no term. `conventions.order_azimuth` already takes `(ky, kx)` and already handles its
degenerate cases, so a 2D order grid needed no new frame code, and the per-order `n / kz_m`
factor follows from `E . k_hat = 0` — a statement about one order, not about the lattice.

It landed in five stages with one criterion each, and **the first four are byte-identical**: the
arrays gain a size-1 y axis, the geometry averages along y, the stepper's four terms become
differences, and the source gains a `ky_op`. A crossed grating is therefore a larger `ny` rather
than a second stepper — the alternative, a `_time_loop_3d` beside the existing one, would have
duplicated the six updates, the CPML, the TFSF and the pole coupling.

### What the third dimension broke that the second did not

**The y pass is skipped, not run, on a y-invariant raster.** An exact average of a constant is
not bit-equal to the constant, so running it would have moved every existing answer in the last
digits — a change that afterwards cannot be told from a real one.

**`E_y` and `H_y` were passed `harmonic_axis=None`, and that was right.** In a 1D-periodic
grating `E_y` is tangential to every interface there is; in a crossed one it is normal to the y
walls. The reverse edit — widening the validator to accept `'y'` and leaving the call sites —
costs an order of convergence and fails no test in the file.

**The auxiliary grid's `d/dy` must be the discrete difference, not the exact `i*ky`.** The two
grids otherwise sit on different dispersion relations. The vacuum TFSF cancellation measures it
with ten orders of resolving power: 1e-16 correct against 5.4e-7 and 2.2e-6 with the continuum
operator, at 30 and 45 degrees. Same failure `kx_op` prevents, one axis over.

**The propagating set is a disc, not a rectangle.** Calling the 1D window once per axis narrows
the x range by the *incident* `ky` while order `(m, l)` carries `ky_inc + l*wl/Py`.
`propagating_window_2d` computes the disc; checked by brute force over 4000 random scenes.

**Four demodulation kernels, not two.** Each transverse component now has an x offset and a y
offset. In 2D one kernel for both was a first-order error that every fixed-tolerance test passed
through; in 3D the y offset is first order in the *efficiencies* too, because the s/p projection
combines `Ex` and `Ey` coherently and there is no `|r|**2` cancellation left to hide behind.

### The convergence order is not settled

Against `solve_2d`, the worst per-order gap runs 9.1e-2 / 3.6e-2 over dz = 33 / 20 nm on a
crossed scene — order 1.83. On a scene with **no plan-view corner** it runs 1.3e-1 / 4.8e-2 /
3.5e-2: order 2.02 and then **0.67**. A floor at 3.5e-2 that the corner cannot explain, because
that scene has none. Candidates not yet separated: the RCWA reference's own truncation,
`n_periods` at the finest grid, something else. A 3D point costs about eighteen minutes, which
is why this is recorded rather than chased.

What is established instead, and does not depend on the reference: **the 3D path reproduces the
2D path to 4.8e-3 on a y-invariant raster.** That is the same physical scene solved two ways, so
it isolates the machinery from the geometry in a way no comparison against RCWA can.

### Cost

A 3D run is minutes to hours where the 2D one is seconds — 0.75 M cells at a 10 nm grid on a
700 nm square cell, and the Courant bound tightens by another `sqrt(3/2)`. Against `solve_2d`
this does not win: the crossed-grating RCWA solve costs 2.8 s at `truncation_order=(10,10)` and
7.0 s at (12,12), and the broadband break-even is in the hundreds of wavelengths — a band this
module can only cover to about ±10 % before the blue end is under-resolved. **The reason to have
a 3D FDTD is what RCWA cannot express at all** (§9), not what it expresses more slowly.

## 9. What is not yet answered

- **A backend other than NumPy**, which is the item that decides whether this is a first-class
  part of a differentiable package or a guest in one. Everything else in wavebit is jax/torch
  traceable; the field updates here are slice-shifted differences and would port unchanged. It
  is only the time LOOP that has nowhere to live — it needs a `scan` primitive `_core/backend.py`
  does not have, and adding one is a design decision about the whole backend layer rather than
  about this module. **fdtdx** (Mahlau et al., https://github.com/ymahlau/fdtdx, MIT;
  JOSS 10.21105/joss.08912) answers the shape of it with a `custom_vjp` carrying two gradient
  methods, reversible and checkpointed, and marks where reversibility stops: it refuses
  dispersive materials, because the ADE polarisation recurrence cannot be run backwards. A metal
  grating is Drude, so the headline problem lands on the checkpointed path. Cited as prior art
  only — **nothing here is ported from it**, and it carries no notice in `LICENSE` for that
  reason. Its own non-uniform-mesh design (one realised grid on which "uniform" is a policy
  rather than a separate code path) is the same shape as §8f's, arrived at independently and
  landed before that survey ran; the resemblance is convergence, not lineage.
- **A fixed-*angle* spectrum at oblique incidence**, rather than the fixed-`kx` sweep of §8.
  This is the gap that matters most once the module is read as a solver rather than a harness:
  a spectrum at a tilt is an ordinary thing to want, and it is exactly what one run cannot give
  today. The textbook answer is a **split-field** formulation — transform the fields so the
  frequency dependence in the periodic boundary becomes a time derivative, then absorb it by a
  change of variable (Roden, Gedney, Kesler, Maloney and Harms, *IEEE Trans. Microwave Theory
  Tech.* **46**, 420 (1998)). Its rival is **spectral FDTD** (Aminian and Rahmat-Samii, *IEEE
  Trans. Antennas Propag.* **54**, 1818 (2006)), whose building block this module already
  produces: several fixed-`kx` runs tile the `(kx, f)` plane, a fixed-angle curve is a line
  through the origin on it, and it is read off by interpolation. **The two lose on different
  axes and the choice is not obvious.** Measured on this scene, ten wavelengths at 30 degrees
  cost 75 s as ten separate runs; split-field would be *one* run (the angle is a constant, so
  the whole band is that angle's answer) at perhaps twice the per-run cost, while spectral
  needs three to five runs. So split-field is roughly twice as cheap, and spectral is the one
  that leaves the time stepper alone — a driver loop and an interpolation rather than a second
  formulation. Split-field implementations are reported to have stability trouble at large
  angles and spectral's criterion is reported to be angle-independent; neither claim has been
  checked against a primary source.

  **None of it is worth choosing yet.** RCWA solves those same ten wavelengths in 0.34 s, 44x
  ahead of the best FDTD route, so on a scene like this one no formulation wins. The decision
  only becomes real on a scene where the RCWA side is expensive — many layers, a high
  truncation order, or a metal, which is why dispersion is above this on the list.
- **A non-uniform grid that pays on every scene**, rather than the one in §8f that pays on a
  metal grating and loses on a dielectric one. What would change that is a transition cheaper
  than a geometric ramp — the literature's subgridding schemes (Okoniewski 1997, Berenger 2006)
  buy a smaller reflection for a more complicated interface, and none of them is measured here.
- **Field output.** The solve returns two monitor planes; cross-checking `calculate_field` —
  the part of the solver with the most recent bug history — is impossible without it.
- **The Yee grid's numerical anisotropy** is unmeasured. Convergence order was measured at
  normal incidence only, and phase-velocity error as a function of propagation angle is
  exactly what limits the high-`|m|` orders a grating oracle exists to check.
- **What only a time-domain solver can do, now that the geometry is general.** With a crossed
  grating (§8g) and a time record () in place, the features that are not a slower way of
  answering an RCWA question become reachable: a medium whose permittivity changes *while* the
  light is inside it (photonic time crystals, time boundaries — a one-frequency formulation
  cannot express a medium that mixes frequencies), and a nonlinear one whose response depends on
  the instantaneous field. Neither exists in any permissively-licensed FDTD surveyed. Both have
  closed forms to check against — a harmonic-balance sideband S-matrix for the first
  (Zurita-Sanchez, Halevi and Cervantes-Gonzalez, *Phys. Rev. A* **79**, 053821 (2009)), the
  temporal-boundary amplitudes (Morgenthaler 1958; Xiao, Maywar and Agrawal, *Opt. Lett.* **39**,
  574 (2014)), self-phase modulation for the second — and both have a trap measured before
  either was built: the naive `eps dE/dt` where `d(eps E)/dt` is meant is **the right equation
  for the wrong field**, so the band gap, the growth rate and energy conservation all come out
  right, and only a temporal boundary separates them, by a factor of exactly two that no
  resolution shrinks.
- **The convergence order of a crossed grating**, which §8g records as unresolved: 1.83 on a
  cornered scene, and 2.02 then 0.67 on one without a corner. Separating the reference's
  truncation from `n_periods` from anything else costs about eighteen minutes a point.
- **The corner cells** where a vertical and a horizontal interface meet are the one place the
  axis-aligned averaging rule is not exact; whether they contribute at first or second order
  has not been measured, and TM's convergence order (1.71–1.94, against TE's 1.96–2.02) is the
  place to look.

# FDTD — the time-domain solve beside the RCWA one

Written: 2026-08-31, extended through 2026-09-04.
Targets: `wavebit/_core/fdtd.py`, `wavebit/main.py`, `tests/test_fdtd.py`

A second method reading the same `Stack` and `Incidence`, so that one structure can be solved
two ways and the answers compared without a translation layer.

**Scope.** Pure TE / TM in the classical mount (`phi = 0`) at any polar angle, which mirrors
`wavebit/_core/solve_1d.py` down to the amplitude conventions; a conical mount through `psi` at
any `theta` and `phi` (§8d, §8e), which mirrors `solve_1d_conical`; and a crossed grating, the
raster periodic along both axes (§8g). Drude/Lorentz dispersion (§8b) and permeability (§8c)
are carried on all of them, the z mesh can be graded (§8f), and a run can keep a record of the
field in time (§8h). Everything outside that is refused by name; §6 lists the refusals and why
each is a refusal rather than an approximation.

---

## TL;DR

On the benchmark binary grating, at the same 5 nm grid Meep uses, the worst per-order gap
against the converged RCWA answer is **6.6e-4 (TE) and 6.9e-4 (TM)**, against Meep's own
4.9e-4 and 7.4e-4 on that grid — within 1.4x on TE, where Meep is closer, and within 1.1x on
TM, where this module is. Both codes are the same distance from the reference to within a
small factor, which is the claim; neither gap is tighter than the 1.6e-3 the Meep column
itself moves across two of its own PML settings, so read them as agreement at ~1e-3 rather
than as a ranking. (Until 2026-09-07 this said "the same number on TM", which was true of the
superseded pair 7.4e-4 / 6.9e-4 against 7.4e-4 / 6.9e-4 and is not true of these.) Refine once
more and it reaches 1.4e-4 / 1.9e-4, at a clean second order throughout. The two methods agree
in the limit, which is the only thing this module was built to establish. All three columns
of that comparison are committed — `tests/references/golden_fdtd_table.npz`, with the Meep
half taken from `golden_meep_1d.npz` — and CI reads §2's tables out of the markdown and
compares them cell by cell. Until 2026-09-06 the Meep column was quoted from a sibling
repository; until 2026-09-07 the conditions all three were taken at were the phrase "at the
shipped defaults", which is what let one column go stale. §2 records both, and
[`tests/references/_meep_1d_cases.py`](../tests/references/_meep_1d_cases.py) is now the
record a reader regenerating this table reads.

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
`R[±1]` is identically zero.

**All three columns are committed, and CI reads this table out of the markdown and compares
it to them.** The numbers are `tests/references/golden_fdtd_table.npz` — the reference, this
module's answer and Meep's, with the conditions they were taken at stored beside them; the
Meep column is copied there from `tests/references/golden_meep_1d.npz`, which is its own
oracle and carries a closed-form planar calibration.
`tests/references/test_fdtd_table_golden.py` parses both of this section's tables cell by
cell and holds every value to half a unit in its last printed place, so a digit edited here,
or a table left behind by a solver change, is a failing test rather than a stale paragraph.
The Meep column stopped being a quotation on 2026-09-06 and the other two stopped being
unread on 2026-09-07; what each of those found is recorded below.

**The conditions are a file, not the phrase "at the shipped defaults".** That phrase is what
let this section go stale: it stayed true-looking while `PAD_WAVELENGTHS` was halved
underneath it, and it is unfalsifiable by construction, so no test could have checked it.
Every determinant of all three columns is now written out **by value** in
[`tests/references/_meep_1d_cases.py`](../tests/references/_meep_1d_cases.py), and
`test_fdtd_table_golden.py::test_the_recorded_conditions_are_the_ones_the_module_ships`
asserts that the recorded values are still the shipped ones — which is the check that would
have caught the drift on the day it happened. The load-bearing ones, so that a reader can
tell whether their run is this run without opening the file:

* **this module** — lengths in **micrometres**; `resolution=200`, which is a 5 nm cell only
  in that unit (140 × 281 nodes, `dt = 3.5001785668734102e-3`, 20571 steps = 80.0 optical
  cycles); clearance **0.45 µm above the structure and 0.30 µm below**; PML **12 cells =
  0.060 µm** at each end; `courant=0.99`; subpixel smoothing **on**; the z mesh **uniform**;
  the CPML at `m = 4`, `κ_max = 2`, `α = 1.5 ω` and a 1e-8 round-trip target. The clearance
  is stated as a *length* because `pad_scale` and `PAD_WAVELENGTHS` multiply into one product
  and no result can tell them apart — `PAD_WAVELENGTHS=1.0` reproduces `pad_scale=2.0` bit
  for bit, so naming either one alone is not a record;
* **Meep** — `pymeep 1.34.0` (conda-forge `nompi_py312h46debac_101`, serial), resolution 200,
  `until_after_sources=120`, `dpml=0.6`, `pad=0.7`, monitors 0.10 µm clear of the structure,
  a Gaussian source of `fwidth = 0.2·fcen` at z = −0.49 in a 0.7 × 2.9 µm cell — **and,
  because the harness never passes them, meep's own `eps_averaging=True`,
  `ensure_periodicity=True`, `Courant=0.5`, quadratic PML profile, `R_asymptotic=1e-15` and
  source `cutoff=5`.** Those six are pinned by nothing but the version string, and four of
  them move this column by 5e-4 to 1.5e-2 — `eps_averaging` alone by 1.5e-2 here, thirty
  times the agreement below. Recording a Meep resolution and stopping, which is what this
  section did until 2026-09-07, records almost none of what fixes the answer.

**What a deviation costs**, worst per-order against the shipped configuration, one knob at a
time (the record tabulates ~30 more): subpixel smoothing off **3.0e-2**; 6 PML cells
**2.6e-2**; `resolution` left to the module's own default **4.4e-3**; grading on **9.2e-4**;
`pad_scale=2.0` **7.9e-5**. **`R + T` sees almost none of it** — turning subpixel smoothing
off moves the per-order efficiencies by 3.0e-2 while Σ moves 4.5e-6, so an energy check would
have passed every configuration in that sweep except the 6-cell PML. Amplitudes are 3–10×
more sensitive than the efficiencies printed here.

Reference is the RCWA path at `truncation_order=80`. TM's worst per-order value still moves
by 1.30e-5 out to `truncation_order=400`, so the reference itself carries that much
uncertainty; TE is converged (1.21e-6 over the same span). The smallest gap below clears the
uncertainty of the column it is read against by 18x (TM `T0`, 2.4e-4 against 1.3e-5) and the
rest by more. Both figures are frozen in the golden and
`test_fdtd_table_golden.py::test_the_reference_is_converged_enough_to_carry_the_gaps`
asserts that margin, because a gap quoted against a reference is only as good as the
reference and this is the one number in the table with no column of its own.

| | RCWA (ref) | this module, 5 nm | gap | Meep, res 200 | gap |
| --- | --- | --- | --- | --- | --- |
| TE `R0` | 0.204898 | 0.205554 | 6.6e-4 | 0.204410 | 4.9e-4 |
| TE `T0` | 0.352474 | 0.352789 | 3.1e-4 | 0.352379 | 9.5e-5 |
| TE `T±1` | 0.221314 | 0.220831 | 4.8e-4 | 0.221391 | 7.7e-5 |
| TM `R0` | 0.282393 | 0.283085 | 6.9e-4 | 0.283135 | 7.4e-4 |
| TM `T0` | 0.163805 | 0.164245 | 4.4e-4 | 0.164044 | 2.4e-4 |
| TM `T±1` | 0.276901 | 0.276274 | 6.3e-4 | 0.276526 / 0.276413 | 3.8e-4 / 4.9e-4 |
| Σ, TE / TM | 1 | 1.000006 / 0.999877 | | 0.999572 / 1.000118 | |

**Which reading of the Meep column.** A per-order efficiency is not one thing meep returns.
The harness projects the monitor's `E` onto each Rayleigh order and converts with the
analytic `kz`; two other legitimate readings of the *same* run — a per-order
`Re(E × conj(H))` built from the `H` field meep stepped, and meep's own `add_flux` — do not
agree with it exactly. Measured at res 200: **≤ 7.2e-5 per order**, and 2.06e-4 (TE) /
1.17e-4 (TM) on the total. The column above is the E-projection, and it is the one that sits
apart — the two Poynting readings agree with each other to 5e-6 (TM) and 8.0e-5 (TE).

**That number moved, and it moved down.** Until 2026-09-07 this paragraph said the readings
differ by 2.5e-4 (TE) and 3.4e-4 (TM), a figure inherited from the 2026-08-25 lab run and
never re-derived here. It reproduces on TE only as a statement about the *total*, and is 2.9x
too large on TM: **per order the ambiguity is four times smaller than the figure it was
quoted as**. It is quoted below as a bound on what "reproduced" can mean, so the smaller
number is the honest one to quote there too. `get_eigenmode_coefficients` — what "mode
decomposition" means in meep's own vocabulary — is not one of the three readings compared, so
7.2e-5 is a lower bound on the ambiguity rather than necessarily the same pairing the lab run
meant.

### What moved when the table was re-derived, and why

Until 2026-09-06 the Meep column was computed in a *sibling lab repository* on 2026-08-25 and
**quoted here, not recomputed** — `grep meep` over this repository returned nothing at all. So
the strongest published evidence for this module was not in the project, and neither of the
two columns beside the reference was checked by anything. Both moved when they were
re-derived, for two unrelated reasons, and both are corrected above rather than explained
away. (The same lab run also solved this scene with gyptis, an FEM; that column is not
reproduced here and so is not printed.)

**The Meep column is a fresh run and differs from the quoted one by up to 5.3e-4** — worst at
TM `R0`, 0.283135 here against the quoted 0.282607; TE's two first orders land within 5e-6.
That gap cannot be closed from here and is not a disagreement, but the reason is **not** the
one this section gave until 2026-09-07. It used to point at the two readings' 2.5e-4, which
has since come down to 7.2e-5 per order and cannot carry a 5.3e-4 gap. The reason is that
this section recorded the quoted run's resolution and **nothing else** — not the run length,
the pad, the PML depth, the monitor distance or the source — and the column is measurably
sensitive to two of those: changing `dpml` 0.6 → 0.9, or `pad` 0.7 → 1.4, moves it by
**1.6e-3** at this very resolution, which is 3.2x the agreement the column is being credited
with. Two FDTD runs of one scene agreeing to 5.3e-4 with that much unrecorded between them is
the claim, and `test_meep_1d_golden.py::test_the_2026_08_25_lab_column_is_reproduced` holds
it at 1e-3 for that reason rather than at anything chosen to fit.

**The Meep column has a configuration sensitivity larger than the agreement it certifies, and
the cause is not established.** In `(dpml, pad)` it is *bimodal*: `dpml` 0.30 / 0.60 / 0.65 /
0.70 / 0.80 land on one value and 0.75 / 0.90 / 1.20 on another, ~1e-6 apart within a branch
and 6.5e-3 apart between them at res 100 — and `dpml=0.9` and `pad=1.4` land on the *same*
other value to 1e-6, which is two branches rather than a continuum. At the frozen res 200,
re-measured against the committed golden itself: `dpml=0.9` gives TE `R0` **0.205985** where
`dpml=0.6` gives 0.204410 and RCWA 0.204898, so it is 1.575e-3 away, on the far side of the
reference, **with `Σ` unchanged at 0.99957 to five decimals**.
Eleven candidate mechanisms were tested on 2026-09-07 and every one refuted by at least one
row (cell length, z pixel parity, PML cell count, absorber quality, source plane,
source-to-PML distance, `ensure_periodicity`, `eps_averaging`, the incident normalisation and
two more); the sensitivity lives entirely in the scattered field, with `R + T` preserved to
1e-5. Settling it would need meep's source, which this repository does not open, so it is
recorded as unexplained in `_meep_1d_cases.py::MEEP_UNEXPLAINED` rather than guessed at. The
shipped `dpml=0.6` is the branch that agrees better with RCWA and nothing anywhere says why
those values. **Read the 4.9e-4 / 7.4e-4 gaps with that in mind**: they are what this
harness's configuration gives, and a differently configured Meep run of the same scene can
sit 1.6e-3 away.

**The headline pair moves with it.** Against the converged reference, Meep's own worst
per-order gap is **4.9e-4 (TE) / 7.4e-4 (TM)** where the quoted column gave 7.4e-4 / 6.9e-4.

**The "this module" column was a `pad_scale=2.0` run, which has not been the shipped default
since the clearance was halved.** At `pad_scale=2.0` all eight values of the old column come
back to the last printed digit, both `Σ` included, together with the old convergence table's
whole `Σ` column — which is what identifies the pad rather than a machine or a version.
(Re-measured 2026-09-07: `pad_scale=2.0` gives TE 0.205633 / 0.352724 / 0.220785, Σ 0.999928
and TM 0.283018 / 0.164256 / 0.276292, Σ 0.999859 — the old column exactly, and 7.93e-5 /
6.69e-5 from what the shipped clearance gives.) The paragraph below on halving the clearance
records the same 5 nm run moving 7.37e-4 → 6.57e-4, so the table and that paragraph were
taken on either side of one change and only the paragraph was re-derived.

**Nothing about the table moved on the 2026-09-07 pass, and that is the result.** Both tables
were re-derived from scratch under the recorded conditions and every printed digit came back
the same, the Meep golden regenerated bit-for-bit identical, and what changed instead is that
the conditions became a file and the digits became a committed artefact with a test that
reads them out of this markdown. Three numbers *outside* the tables did move: the two-readings
figure above (2.5e-4 / 3.4e-4 → ≤7.2e-5 per order), the `wall` column (a different machine
load, and the one column no test holds), and a stale 7.5e-4 in
`tests/test_fdtd.py::test_the_binary_grating_agrees_with_the_rcwa_solve`'s docstring, which
neither configuration reproduces and which is now 6.9e-4.

**Convergence**, at the same recorded conditions — the 5 nm row is *shared* with the table
above, which is what pins the two to one configuration; at HEAD on 2026-09-06 they had been
taken on either side of the `PAD_WAVELENGTHS` change and only one of them re-derived, so this
section disagreed with itself by 8e-5. Worst per-order gap against the same
`truncation_order=80` reference:

| cell | TE | order | TM | order | Σ (TE / TM) | wall |
| --- | --- | --- | --- | --- | --- | --- |
| 20 nm | 1.18e-2 | | 8.80e-3 | | 0.99934 / 0.99876 | 0.3 s |
| 10 nm | 2.75e-3 | 2.10 | 2.45e-3 | 1.84 | 0.99984 / 0.99965 | 1.2 s |
| 5 nm | 6.56e-4 | 2.07 | 6.91e-4 | 1.83 | 1.00001 / 0.99988 | 6.3 s |
| 2.5 nm | 1.36e-4 | 2.27 | 1.94e-4 | 1.84 | 1.00005 / 0.99992 | 44 s |

TM's last two orders read 1.83 and 1.84 against a `truncation_order=80` reference and 1.85 and
1.87 against a `truncation_order=400` one, which is the reference's own ~1e-5 showing up once
the run gets below 2e-4; the reference is what the table is quoted against, so it is what is
printed.

Second order throughout, which is the point of the table — the number in any one row is a
property of the grid, but the *order* is a property of the method, and it is what says the
remaining error is discretisation and not a floor.

The cost column is the honest one, and it is the main thing to weigh when choosing between
the two solvers: 5 nm is 6 s a polarisation against the RCWA solve's fraction of a second, and
2.5 nm is three quarters of a minute. It is **the one column here no test can hold**, because
it is a property of the machine rather than of the method — the error and `Σ` columns beside
it are compared against the golden on every push and the seconds are only frozen in it, so
that a reader knows which run they came from. They moved on the 2026-09-07 re-derivation (7.1
→ 6.3 s, 54 → 44 s) while every error and every `Σ` beside them stayed to the digit, which is
the clearest possible statement of what a wall time is worth.

**A band buys some of that back, but less than it sounds.** A whole spectrum comes out of one
FDTD run (§8) while RCWA re-solves per wavelength, so the two meet at
`N* = t_fdtd / t_rcwa_per_wavelength`. **Matched on accuracy rather than on settings** — the
cheapest `truncation_order` and the coarsest cell that each reach the same worst per-order
error, against an RCWA reference at `truncation_order=400`:

| scene | FDTD | its error | the cheapest RCWA that matches it | `N*` |
| --- | --- | --- | --- | --- |
| this grating, TE, 20 nm cell | 0.5 s | 1.2e-2 | truncation_order 5, 0.51 ms | 1009 |
| this grating, TE, 10 nm cell | 2.0 s | 2.9e-3 | truncation_order 10, 0.71 ms | 2850 |
| **60 nm metal grating, TM, 10 nm cell** | 2.4 s | 1.6e-3 | truncation_order 60, 21.4 ms | **112** |

**Half of that came from one constant.** The clearance between the structure and each of the
TFSF plane, the monitors and the PML was a full wavelength, chosen so a grating's evanescent
orders would decay before the monitor. They do not need to — the spatial transform separates
the orders exactly, so an evanescent order sitting on the monitor plane cannot contaminate a
propagating one. Halving it is *more* accurate as well as twice as fast (7.37e-4 / 22.0 s
against 6.57e-4 / 13.9 s on the dielectric; 7.43e-4 / 31.9 s against 7.01e-4 / 15.9 s on the
metal), because a shorter outgoing leg carries less of the analytic-versus-discrete `kz`
mismatch. Below a quarter wavelength it stops paying. `pad_scale` moves it.

Those are the numbers the decision was taken on and they are left as they were taken.
Re-derived on 2026-09-07 from the two configurations' own printed columns, the dielectric
pair is **7.35e-4 → 6.56e-4** (TE) and 6.25e-4 → 6.91e-4 (TM) — the same conclusion on TE, and
on TM the halving is 6.6e-5 *worse*, which the single worst-of-both-legs figure above hides.
The wall times are not re-derivable: they belong to that machine on that day. This paragraph
is not in the golden and no test holds it; the two tables above are, which is the difference.

**Matching on accuracy is the whole point of the table**, and an earlier draft of this document
did not: it compared a 10 nm FDTD run against `truncation_order=80`, which was a *thousand
times* more accurate than it needed to be, and reported a crossover near a hundred. The real
figure on a lossless dielectric is nearer a thousand.

`N*` is a property of the scene. It moves toward FDTD as the RCWA side gets harder — the metal
row is 19x better than the dielectric at the same cell size, because RCWA in TM on a metal
converges slowly and non-monotonically (its error at `truncation_order` 40 is *worse* than at 20) — and away
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
cell on 2026-08-31 — the *solve* was unaffected because the raster's index convention and the
transform's `[0, 1]` integration did not move with it, and that is pinned rather than reasoned
about. **The paragraph's scope is the solve, and this is worth saying because the conclusion was
read too widely once.** The record's x axis (§8h) was added four commits later and did not take
the rule: it was published from the corner until 7d7a943. A verdict about a module can be older
than a part of it.

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

Ten, each refused by name. The common thread: every one returns a *plausible* number if it
is quietly ignored, and a plausible wrong number from the cross-check solver is worse than no
cross-check at all.

Three rows have left this table since it was written, and they left by being implemented rather
than by being reconsidered: `psi` (§8d, §8e), `phi != 0` (§8e) and `(n_layers, Ny>1, Nx)` (§8g).
A refusal is a statement about today, and these three outlived their subject by four commits
inside this document's own branch.

| Refused | Why not just implement it |
| --- | --- |
| complex `theta` | Evanescent incidence has no propagating incident wave to normalise against, and the TFSF plane injects one by construction. The RCWA path answers this — see `docs/branch-cuts.md` |
| `incidence_side='bottom'` | The mirror the ETM applies has no counterpart here yet |
| complex permittivity | One complex number describes a material at **one** frequency, and a run carries every frequency in its band. Give the dispersion as poles instead (§8b) — that *is* the frequency dependence, rather than one sample of it |
| `Re(ε) ≤ 0` | A metal below its plasma frequency — the same dispersion problem, and no real wave speed, so no stable time step exists |
| parametric shape structure | Rasterising it would put a second geometry builder beside `draw.slice_structure`, which is NumPy-only, square-only and refuses a traced shape parameter. The two would disagree about one scene |
| `lattice`, axis-aligned or not | The Yee grid is Cartesian, so a skewed cell needs the metric in every update. An axis-aligned one would not, and is a gap rather than a limit: pass `period` |
| a run shorter than the source | The source is not finished emitting until 48 optical cycles. A 20-cycle run returns 0.033 where the answer is 0.543 — an efficiency-shaped number, so it is refused rather than reported with a residual nobody reads |
| `\|m\| ≥ nx/2` | Past Nyquist the x transform aliases one order onto another and returns a plausible amplitude — with a flipped sign on TM, from the half-cell sample offset |
| `backend` | §7 |

Beyond the refusals, the solve checks that what it hands back is finite, and nothing more.
Three defects this module has had did announce themselves first as `R + T` between 1.1 and
106 — an over-conducting PML, a mis-scaled CFS pole, an out-of-band DFT — and there was a
generic check that reported it. It was removed on 2026-09-03: a Yee grid conserves energy to
`O(dz^2)` rather than exactly, so on this path that check also fired on **85 of 243 ordinary
solves**, the uniform slab against its own closed form among them. A signal present on a third
of correct results is why `R + T = 2.0042` (§10) went unheard. Energy is asserted per scene in
`tests/test_fdtd.py` instead, at a tolerance chosen for the discretisation. See
[`result-guards.md`](result-guards.md).

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

Implemented 2026-09-01, the day `545dd4c` lifted the same refusal on the RCWA path's 1D and
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
as the half-turn the RCWA path's per-order frame lost, in `docs/amplitude-sp-frame.md`.

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

### The convergence order, and the scene that was measuring itself

Against `solve_2d` on a crossed scene the worst per-order gap runs 9.1e-2 / 3.6e-2 / 1.8e-2 over
dz = 33 / 20 / 12.5 nm — orders 1.83 and 1.50, still improving. On a *y-invariant* control it
runs 1.3e-1 / 4.8e-2 / 3.5e-2: order 2.02 and then **0.67**, stalled.

**The control converging worse than the thing it controls for is the tell, and the cause is the
control's own incidence.** Its `m = -1` order sits at `kz_top = 0.0916j` — five times closer to
the Rayleigh cutoff than the margin scene §8e was chosen to give (0.5101j) — and a near-cutoff
order converges slowly for the arithmetic reason §8e records: `kz = sqrt(n**2 - k_par**2)` has
sensitivity `1/kz`, and the p channel multiplies by `n/kz` on top.

Two measurements settle it. The RCWA reference is not the floor: at `truncation_order` x = 14 it
is accurate to 5.6e-4, sixty times below the 3.5e-2 the FDTD stalls at. And **the floor is not a
property of the third dimension at all** — the long-validated 2D path shows it on the same
scene, running 1.2e-1 / 4.2e-2 / 2.9e-2 / 1.6e-2 / 2.1e-2 down to dz = 3.3 nm, orders 2.08 then
0.78, 0.92, -0.35.

So the crossed grating's 1.83 / 1.50 is the honest figure for a scene near a Rayleigh point, and
what a converged number would take is a scene chosen for margin in 3D the way §8e's was in 2D —
which costs about eighteen minutes a point, and is why it is recorded rather than run here.

What is established without a reference at all: **the 3D path reproduces the 2D path to 4.8e-3
on a y-invariant raster.** The same physical scene solved two ways, which isolates the machinery
from the geometry in a way no comparison against RCWA can.

### Cost

A 3D run is minutes to hours where the 2D one is seconds — 0.75 M cells at a 10 nm grid on a
700 nm square cell, and the Courant bound tightens by another `sqrt(3/2)`. Against `solve_2d`
this does not win: the crossed-grating RCWA solve costs 2.8 s at `truncation_order=(10,10)` and
7.0 s at (12,12), and the broadband break-even is in the hundreds of wavelengths — a band this
module can only cover to about ±10 % before the blue end is under-resolved. **The reason to have
a 3D FDTD is what RCWA cannot express at all** (§9), not what it expresses more slowly.

## 8h. A time record

`record_field=N` keeps every `N`-th step of the electric field, in the diagnostics, alongside
the axes it is measured on. It is off by default and costs nothing when it is off: the snapshot
buffer is not allocated, and the keyword is the only thing that allocates it.

It exists because it is the one thing this module can do that the RCWA path cannot state. A
harmonic solve answers what the field *is*, having assumed `e^{-iωt}` and therefore assumed that
a single frequency describes the whole problem. There is no picture of light arriving, because
the assumption is that it always has been. A time-domain solve is stepping a real field anyway,
so a record of it is a read rather than a computation.

**The record carries its own abscissae, and that is the whole design.** A reader who rebuilds
them is wrong three separate ways, each silently:

| the axis | rebuilt as | why that is wrong |
| --- | --- | --- |
| `field_z` | `arange(nz) * dz` | a graded mesh (§8f) has no single `dz` |
| `field_x` | `arange(nx) * dx` | in-plane coordinates are measured from the centre of the cell |
| `field_x` | the pixel centres | `Ey` sits on the x *nodes*, half a cell from them |

The third is the one worth stating twice. The Yee grid stages `Ey` on `(x-node, y-half)` and
`Ex` on `(x-half, y-node)`, so the two components do not share an abscissa and neither shares
one with the raster. `field_e_component` says which component the record holds — `ey` for a TE
run and for either conical leg, `ex` for TM — and `field_x` / `field_y` are that component's own
axes. A conical run also carries `field_ex` with `field_ex_x` / `field_ex_y` beside it, because
a conical mount genuinely has two electric components and showing one of them is the lesser
answer.

Each of those three was a real error here, not a hypothetical: `field_x` was published from the
cell corner for as long as the feature existed, and the document above had already reasoned
about the origin move and concluded the module was unaffected — which was true when written,
four commits before this axis was added.

`wavebit.plot_fdtd_field` and `wavebit.animate_fdtd_field` read this record directly. They
use each component's measured axes and nonuniform z coordinates, mark PML and the TFSF
boundary (scattered field above, total field below), and keep the color scale fixed across
frames. The default is signed real E; magnitude is available but is not intensity. A record
that undersamples the shortest requested optical period warns rather than depicting an
aliased oscillation without comment. Matplotlib is optional and imported only on use.

The public `psi` route records both electric components of the requested coherent mixture.
Previously it dropped `record_field` while dispatching, so no snapshots survived that route.
The coupled route now combines the raw s/p histories; the scalar route rotates the incident
electric vector onto its unit-Ey and unit-Ex drives, preserving the negative tangential p
direction and the separate Yee coordinates. Field superposition and recording-on/off
frequency-domain equality are checked in `tests/test_fdtd_record.py`.
Crossed rasters always use the coupled route, including zero polar or azimuthal angles;
the scalar 1D combiner cannot retain their second order axis. Settling checks include Ex
as well as Ey, and a `psi` solve reports the larger residual of its two basis runs.

### Recording all six field components

Pass `record_components='all'` alongside `record_field=N` to record Ex, Ey, Ez,
Hx, Hy and Hz. A component name or sequence such as `('ez', 'hy')` records only
those additional outputs. Omitting the option retains the original transverse
record and memory cost. The legacy `field_e`/`field_ex` outputs remain available.

```python
result, diag = wavebit.fdtd(
    structure=np.full((1, 1, 2), 2.25), thickness=[100.], top=1., bot=1.,
    period=[200.], wavelength=900., theta=.3, phi=.4, polarization=(1., 1j),
    resolution=.02, n_periods=80, record_field=5,
    record_components='all', return_diagnostics=True,
)
hy = diag['fields']['hy']
# hy['values'].shape == (len(hy['t']), len(hy['z']), len(hy['y']), len(hy['x']))
fig, ax, mesh = wavebit.plot_fdtd_field(diag, component='hy', frame=20)
```

Each `diag['fields'][component]` contains `values` and its own `t`, `z`, `y`, `x`
vectors. Values stay on the native Yee grid, including a graded z mesh:

| Component | x | y | z | Time |
| --- | --- | --- | --- | --- |
| Ex | half | node | node | `(step+1) dt` |
| Ey | node | half | node | `(step+1) dt` |
| Ez | node | node | half | `(step+1) dt` |
| Hx | node | half | half | `(step+1/2) dt` |
| Hy | half | node | half | `(step+1/2) dt` |
| Hz | half | half | node | `(step+1/2) dt` |

A y-invariant cell has one y coordinate. Components absent in a pure TE/TM solve
are identically zero. Magnetic fields are the normalized `Z0 H` used by the solver,
not SI amperes per metre. Complex Jones recording uses the same propagated temporal
quadrature for all requested components. Plotting and animation use each component's
native coordinates and timestamp; absorber/source markers stay at their physical planes.

For an instantaneous Poynting vector, interpolate the fields onto a common spatial
position **and time** before taking a cross product. Direct products of raw E/H array
indices combine different Yee positions and half-step times. These records provide
transient fields, not the RCWA `reconstruct_field` frequency-domain API.

### Point probes and instantaneous energy flow

`sample_fdtd_fields` interpolates the six recorded components to one common point
and time vector. `fdtd_poynting` adds the instantaneous vector
`S = cross(E.real, H.real)`. Use either with `record_components='all'`:

```python
probe = wavebit.fdtd_poynting(diag, x=0., y=0., z=50.)
time, flux = probe['t'], probe['S']
# E/H/S have shape (number of times, 3), with x/y/z columns.
sz = flux[:, 2]  # positive means down into the stack
# An explicit time vector or scalar is also accepted, within the common interval.
fields = wavebit.sample_fdtd_fields(diag, z=50., t=time[10])
```

By default the probe uses recorded Ex times within the common E/H interval; the last
E sample is usually excluded because H ends half a step earlier. Linear interpolation
uses each component's actual x/y/z/t axes, including a graded mesh. It reads only the
neighbouring spatial samples, rather than resampling the whole recorded volume.
Refine both the mesh and `record_field` for quantitative accuracy: interpolation of a
sparsely recorded oscillation can attenuate its amplitude. Temporally aliased records warn.

The probe refuses extrapolation, periodic wrapping and interpolation that touches or
crosses the TFSF discontinuity. Singleton axes accept only their recorded coordinate;
this includes y=0 for the algebraic-y formulation. `region` is `'total'` below the TFSF
plane and `'scattered'` above it. Scattered flux above that plane does not include the
incident beam. Positions near material interfaces require spatial refinement.

This is **instantaneous** flow: there is no complex conjugation and no factor of 1/2.
Those belong to a harmonic cycle average. H is stored as Z0 H, so S is in squared
source-electric-field units. It is Z0 times SI flux only if E is calibrated in V/m;
it is not an absolute W/m² output. Surface integration and frequency-domain flux
monitors are separate from this point-probe API.

### Elliptical and circular incidence

The diffraction solve accepts the same polarization coordinates as RCWA:

- `psi` alone means `(a_s, a_p) = (sin(psi), cos(psi))`.
- `psi` with `chi` gives
  `a_s = sin(psi) cos(chi) + i cos(psi) sin(chi)` and
  `a_p = cos(psi) cos(chi) - i sin(psi) sin(chi)`.
  All angles are radians. With only `chi`, `psi` defaults to pi/2, as in RCWA.
- `polarization=(a_s, a_p)` accepts a finite nonzero complex Jones vector.
  Its norm is removed, but its overall phase is preserved in the complex outputs.
  `(1, 1j)` and `(1, -1j)` are the two circular states in Wavebit's s/p frame;
  the explicit pairs avoid a viewing-direction-dependent handedness label.

`tetm`, the angle chart, and a Jones pair are mutually exclusive spellings.
Both basis runs are performed for a general state. Amplitudes are combined before
squaring, retaining the interference term; averaging the two basis powers is incorrect.
The returned `s_inc` and `p_inc` are the unit basis answers, independent of the requested
state. The same Jones vector applies to each wavelength of a broadband solve; the
existing fixed-in-plane-wavevector rule still applies at oblique incidence.

```python
result = wavebit.fdtd(
    structure=np.full((1, 1, 2), 2.25), thickness=[100.],
    top=1., bot=1.44, period=[100.], wavelength=900.,
    theta=.3, phi=.4, polarization=(1., 1j),
    resolution=.03, n_periods=80,
)
```

Complex-coefficient states also support `record_field`. The source for each basis
coefficient `a` is a Gaussian envelope multiplying
`Re(a)*sin(omega_c*(t-t0)) - Im(a)*cos(omega_c*(t-t0))`.
The negative cosine gives the positive `i` quadrature under Wavebit's `exp(+i wt)`
DFT convention. Both carriers are propagated through the same Yee grid, material
response and absorber before their records are combined. This is a physical
phase-shifted pulse, rather than a complex rescaling of real-carrier snapshots.
At nonzero Bloch wavevector, **both temporal carriers are always propagated** for
recording, even for a linear polarization. Their complex combination is
`sin(omega_c*(t-t0)) + i*cos(omega_c*(t-t0))`, a positive-frequency carrier under
`exp(+i wt)`. Keeping only the real sine at fixed positive in-plane wavevector
would mix in an unintended oppositely directed temporal component and give wrong
instantaneous transverse energy flow. The diagnostics identifies this representation
as `field_temporal_representation='positive_frequency_carrier'`; its real part is
the physical instantaneous field. The complex record carries both temporal and spatial
Bloch phase, not just spatial phase.

At zero Bloch wavevector, recording adds a quadrature basis run only when its
coefficient has a nonzero imaginary part. At nonzero Bloch wavevector, recording
uses two temporal runs per polarization basis (up to four total). Diffraction-only
calls retain their original cost. The returned diffraction results use the original normalized
basis answers and are identical whether recording is on or off. The diagnostics residual
includes every propagated carrier. For a broadband pulse, the equation above specifies
the recorded waveform exactly; finite Gaussian envelopes are not strictly analytic
signals, so far spectral tails need not have an exact constant quadrature ratio.

```python
result, diag = wavebit.fdtd(
    structure=np.ones((1, 1, 2)), thickness=[100.], top=1., bot=1.,
    period=[100.], wavelength=900., polarization=(1., 1j),
    resolution=.02, n_periods=100, record_field=3, return_diagnostics=True,
)
# Both transverse components are nonzero and rotate in time.
ey, ex, time = diag['field_e'].real, diag['field_ex'].real, diag['field_t']
```

## 9. What is not yet answered

- **A backend other than NumPy**, which is the item that decides whether this is a first-class
  part of a differentiable package or a guest in one. Everything else in wavebit is jax/torch
  traceable; the field updates here are slice-shifted differences and would port unchanged. It
  is only the time LOOP that has nowhere to live — a `for` over ~10^4 steps is what `jax.jit`
  unrolls and autograd tapes, and the primitive that would fix it is a design decision about the
  whole backend layer rather than about this module.

  **fdtdx** (Mahlau et al., https://github.com/ymahlau/fdtdx, MIT; JOSS 10.21105/joss.08912) is
  the prior art, and three sentences that stood here were wrong. They were written from a survey;
  the tree is now at `third_party/fdtd/fdtdx` and was read, which is possible because it is MIT —
  the point of having it beside oracles whose source this project does not open. Corrected:

  - **There is no `scan` in it.** Not one `jax.lax.scan` in the tree. Both entry points step the
    fields with `equinox.internal.while_loop`, and its `kind` argument is what selects the AD
    policy. This paragraph used to say wavebit "needs a `scan`" and imply fdtdx had supplied the
    shape of one; the second half was invented.
  - **The `custom_vjp` is not an optimisation over a differentiable loop — it is the only thing
    that makes the loop differentiable at all.** The forward pass runs `kind="lax"`, which
    equinox documents as forward-mode differentiable and *not* reverse-differentiable.
  - **Reversibility stops at DISSIPATION, not at dispersion, and the dispersive case is not a
    physical impossibility.** Lossy conductive media run on the reversible path — the source says
    "lossy materials get gain when simulating backwards" and handles it — while the dispersive
    branch raises `NotImplementedError` worded *under active development*. The claim here that
    "the ADE polarisation recurrence cannot be run backwards" was a reason this document invented
    for an unimplemented feature. What the reversible path actually requires is that every step
    be an explicit algebraic bijection in closed form; the PML is excluded from that and handled
    by *recording* a one-cell interface plane every step instead.

  What survives, and is worth taking: **the checkpointed half**. The reversible path checkpoints
  nothing by default (`num_checkpoints_reversible` defaults to 0) and recomputes the whole
  trajectory backwards, taking a fresh `jax.vjp` of one forward step at each reverse step. For
  wavebit the concrete ask is one primitive in `_core/backend.py` —
  `scan(xp, body, carry, n_steps, *, checkpoints=None)`, `body(carry, i) -> carry` over a flat
  tuple carry — which is the existing Python `for` on NumPy, that `for` plus
  `torch.utils.checkpoint` on PyTorch, and a scan on JAX. A primitive that exists only in JAX
  would be no answer for a package whose backends are three.

  Cited as prior art only — **nothing here is ported from it**, and it carries no notice in
  `LICENSE` for that reason. Its own non-uniform-mesh design (one realised grid on which
  "uniform" is a policy rather than a separate code path) is the same shape as §8f's, arrived at
  independently and landed before that survey ran; the resemblance is convergence, not lineage.
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
- **Field output.** The solve returns two monitor planes; cross-checking `reconstruct_field` —
  the part of the solver with the most recent bug history — is impossible without it.
- **The Yee grid's numerical anisotropy** is unmeasured. Convergence order was measured at
  normal incidence only, and phase-velocity error as a function of propagation angle is
  exactly what limits the high-`|m|` orders a grating oracle exists to check.
- **What only a time-domain solver can do, now that the geometry is general.** With a crossed
  grating (§8g) and a time record (§8h) in place, the features that are not a slower way of
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
- **A crossed-grating convergence study on a scene chosen for margin**, the way §8e's was in
  2D. The figures in §8g (1.83 / 1.50) are honest for a scene whose `m = -1` order sits at
  `kz_top = 0.09i`, and that is near enough to the Rayleigh cutoff to set the constant. Scanning
  `(theta, phi)` for the largest minimum `|kz|` is the same instrument §8e used; it costs about
  eighteen minutes a point in 3D.
- **The corner cells** where a vertical and a horizontal interface meet are the one place the
  axis-aligned averaging rule is not exact; whether they contribute at first or second order
  has not been measured, and TM's convergence order (1.71–1.94, against TE's 1.96–2.02) is the
  place to look.


## Streaming fields and compiled native grids

The scene solver now accepts `field_monitors` for six-component online Fourier
planes and integrated spectral power. The separate `wavebit.fdtd_grid` provides
NumPy/JAX/Torch uniform Yee updates and checkpointed automatic differentiation.
Explicit `backend` selection on `wavebit.fdtd` connects Stack/Incidence to that
core; see the [common scene adapter](guide/time-domain.md#common-scene-adapter).
See the [time-domain guide](guide/time-domain.md) for normalization, examples,
CPU benchmark evidence and the boundaries of the two interfaces. The native-grid
engine is not a JAX port of the scene solver’s CPML/ADE implementation.


Native `fdtd_grid` now also has electric Drude/Lorentz ADE poles and homogeneous-end
z CPML. This is a limited uniform-grid implementation, with live Meep/FDTDX
comparisons documented in the [scope](fdtd-alpha.md); it does not provide the
full Stack/Incidence feature set on JAX.

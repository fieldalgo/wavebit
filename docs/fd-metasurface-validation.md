# A periodic metasurface against RCWA (stage 4, C7)

Written: 2026-10-03

Targets: `studies/fdtd_metasurface_rcwa.py`, `studies/fdtd_metasurface_corners.py`,
`studies/fdtd_metasurface_mutations.py`,
`tests/test_fd_metasurface.py`, `tests/references/fd_metasurface_rows.json`,
`wavebit/fd/_post.py`, `wavebit/fd/_assemble.py`, `wavebit/fd/monitors.py`,
`docs/guide/fd-metasurface.md`.

## TL;DR

**C7 does not pass under its registered rule (campaign of 2026-10-03, report
`studies/results/fdtd_metasurface_rcwa_report.log`).** The public path agrees with RCWA
wherever the comparison can judge it; what fails is the convergence order of one scene and one
cell RCWA holds at zero.

* **M1** (epsilon-4 cylinder, 8 orders) **passes** both polarisations: worst amplitude error
  2.9e-2, 7.4e-3, 2.1e-3 at 20, 40, 80 cells per wavelength (order 1.90 s, 1.97 p), every one
  of 72 cells consistent with RCWA, both controls failing as registered.
* **M3** (epsilon-12 box, conical incidence) **fails criterion 3**: it agrees with RCWA in all
  54 cells (criterion 4), but converges at order 1.64 (s) and 1.75 (p) in amplitude and 1.51 /
  1.62 in efficiency, under the 1.8 line -- 2.2e-2, 7.1e-3, 3.6e-3 at 40, 80, 120 for s.
* **M2 p fails criterion 4 in one cell of 27**: the cross-polarised `t_s` of the `(0, 0)` order at
  1.05, which RCWA holds at zero by the scene's mirror symmetry. FDTD's grid is not symmetric
  about the pillar, and the depolarisation it makes there (2.7e-4 at 100 cells per wavelength)
  falls slower than second order; the cell misses the band by 6%. Every amplitude and
  efficiency that RCWA makes non-zero passes, at orders 1.97 and 2.01.
* **M2 s is not judgeable**: its efficiencies converged at order 2.48, to 3.9e-4 at 100 cells,
  past what its RCWA ladder resolves (`u_F` 1.9e-4, against a third of 3.9e-4). Its amplitudes
  pass at order 1.80.

Every validity check held in every run: decay below 1e-5, 180 periods moving no cell by more
than 8e-7, JAX and Torch within 1e-14 of NumPy, `sum R + sum T` within about 1e-6 of 1, the frame
and shift instruments within 2e-11, the protocol the registered one. The prediction stated
before the data -- the cylinders, at the junction campaign's 1.73-1.74, might fail criterion 3
-- did not come true; the box did.

**The first suspect for M3, the corners' field singularity, is ruled out** at the resolutions
these runs use: a registered follow-up ran M3's cross-section as a ridge, whose only corners are
its edges, and it converged at second order from 40 to 320 cells per wavelength at epsilon 12
and 4 alike ("Is it the corners?" below). What lowers M3's order is three-dimensional.

## The question

Stage 4's use case (a) is a periodic metasurface lit by an oblique plane wave and read as
diffraction orders. C7 asks whether the public path -- `fd.plan` and `fd.run` with every
default, so the `'voxel'` discretisation and `Structure(junction='split')` where a pillar stands
flush on its substrate -- gives `wavebit.rcwa`'s answer for every propagating order's complex
amplitude, co- and cross-polarised, and for its efficiency, and whether it converges to that
answer at second order. The junction campaign ([`fd-subpixel-occupancy.md`](fd-subpixel-occupancy.md),
"The junction") compared the zeroth order of subwavelength cells at normal incidence; C7 asks
for the orders a metasurface is built to steer, at oblique and conical incidence.

The protocol, the scenes and the rule are registered in the docstring of
`studies/fdtd_metasurface_rcwa.py`, written before the campaign ran; this document records why
they are what they are and what the campaign found.

## The scenes

| Scene | Pillar | Substrate | Cell | Incidence | Orders (R + T) | Rungs |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | cylinder, epsilon 4, diameter 0.40, height 0.6 | 2.25 | 0.85 x 1.00 | theta 35, phi 0 | 2 + 6 | 20 / 40 / 80 |
| M2 | cylinder, epsilon 5.8, diameter 0.32, height 0.6 | 2.13 | 0.60 x 0.60 | theta 30, phi 0 | 1 + 2 | 25 / 50 / 100 |
| M3 | box, epsilon 12, 0.30 x 0.20, height 0.3 | 2.25 | 1.30 x 0.50 | theta 25, phi 30 | 2 + 4 | 40 / 80 / 120 |

Lengths are in vacuum wavelengths at the band centre; rungs are cells per vacuum wavelength.
Every scene is solved for both incident polarisations at three wavelengths, 0.95, 1.0 and 1.05,
from a source band of relative width 0.04. M2 splits the transmitted light between its `(0, 0)`
and `(-1, 0)` orders; in the pilot's slightly smaller pillar the `(-1, 0)` order carried 60% of
the incident s power at 0.95, 21% at 1.0 and 13% at 1.05.

**The same geometry in cells at every rung.** The pillar's axis sits a fixed fraction of a cell
past a node line, the substrate plane a fixed fraction of a cell above a node plane, and every
height, side and radius is a whole number of cells at every rung, so each face the lowering
reads sits at the same fraction of a cell at every rung (the study's `layout` checks it). The
pilot's scenes did not: M2's top sat 0.375, 0 and 0.25 of a cell above a node plane at 25, 50
and 100 cells per wavelength, and M3's narrow sides moved likewise. A face moving within the
cell from rung to rung changes the error's constant between rungs, which a convergence order
reads as convergence; the review that found it measured M2's lowest-rung efficiency error
doubling when only the substrate plane's place in the cell moved.

**Why these periods and angles.** `fd.PlaneWave` keeps the transverse wavevector fixed across
its band, so each wavelength has its own angle (`res['r'].theta_at`), and an order whose
transverse wavevector is `q` (in units of the band centre's vacuum wavenumber) sits at
`lambda q` at wavelength `lambda`: the band moves every order. The cells and angles were
searched for ones where every order that propagates anywhere in 0.95 .. 1.05 propagates
everywhere in it, leaving at `cos theta >= 0.5` in its own medium -- the grazing rule refuses
below 0.41 at these run lengths, and at the 24-cell layer with target 1e-12 a mode at 0.5
returns about 3e-6 -- and every other order sits at least 8% past its cutoff, so no Rayleigh
anomaly falls among the three wavelengths. Such layouts are rare: with a glass substrate and a
+-5% band, a square cell at normal incidence cannot carry the +-1 orders in air without
putting the +-2 orders in glass at grazing.

**The source's band is narrower than the pilot's.** The pulse's spectrum reaches further than
the three wavelengths it is read at. At a relative width of 0.06 it reached the reflected
orders' cutoffs near frequency 1.15 to 1.17 (about 2.6 standard deviations out), and M2's
reflection decayed as a power law, to 2.2e-5 of its peak after 60 periods. At 0.04 those
cutoffs sit 3.8 to 4.2 deviations out and the same run decayed to 5.9e-8, its orders unchanged
to 1e-5.

**Why rectangular cells.** The first pilot's M1 and M3 were square cells lit at phi 45 with the
pillar offset equally along x and y. Such a scene reads the same with x and y exchanged, so a
defect that transposes the two axes -- where this package's two axis-order rules meet, arrays
running v first and hand-written pairs x first -- would agree with RCWA all the same. The
pilot's transmitted `(-1, 0)` and `(0, -1)` orders were equal to round-off. Both scenes were
replaced by the rectangular cells above. M2's square cell with equal offsets and a round pillar
still reads the same transposed; only its incidence breaks the exchange. M1 and M2, lit in the
xz plane by an axially symmetric pillar, are mirror-symmetric in y about the pillar's axis: M2's
cross-polarised amplitudes vanish in RCWA (to 1e-12), and M1's `(m, +-1)` cross-polarised
amplitudes are equal and opposite. Only M1 propagates orders with `n != 0`, so only M1 tests the
y half of the lateral phase correction below; only M3, a rectangular box under conical
incidence, has no mirror at all.

## The reference

`wavebit.rcwa` solves the identical geometry with the shape centred on the cell, both incident
polarisations from one solve, at the run's own `theta_at` per wavelength, over a truncation
ladder. The cylinders take RCWA's shape path (closed-form Fourier coefficients, the
normal-vector-field factorisation). The epsilon-12 box does not, because there RCWA itself
converges too slowly to judge an FDTD error of a few 1e-3. Measured at the band centre against
the ladder's top (half-widths along x and y):

| Box path | Truncation | Worst change in an amplitude |
| --- | --- | --- |
| shape (NVF), half-widths in proportion to the periods | (32, 12) against (39, 15) | 2.3e-3 |
| raster, Li's rule, exact series, same proportion | (32, 12) against (39, 15) | 9.1e-4 |
| raster, Li | (39, 10) against (39, 15) | 1.7e-3 |
| raster, Li | (26, 15) against (39, 15) | 2.5e-4 |
| raster, Li, equal half-widths | (24, 24) against (28, 28) | 2.6e-4 |

The narrow side of the box (0.18 of a 0.50 period) wants as many orders as the long one, and
Li's rule on a raster whose pixel edges carry the box's edges -- with the exact
(`fourier_coefficients='integrated'`) series the raster then *is* the box -- converges where
the shape path's normal field, discontinuous at the corners, does not. The campaign's box
ladder is therefore the raster's, at equal half-widths (20, 26, 28, 30, 32). These pilot
numbers are for the first box, 0.30 x 0.18; the campaign's 0.30 x 0.20 box runs the same ladder.
An epsilon-6 box of the same size converged to 7.5e-5 on the old ladder; an epsilon-12
cylinder of diameter 0.25 to 3.5e-4. The high-index box is kept because it is the scene a
silicon metasurface is.

The cylinders' shape-path ladders converge slowly too, and not in step: M2's at 0.95, against
(24, 24), moves 4.3e-4, 1.5e-4, 1.3e-4 and 9.3e-5 in an amplitude at (12, 12), (15, 15),
(18, 18) and (21, 21). The normal field the rule reads is sampled on a grid (`normal_grid`,
1024 by default); at 4096 every one of those numbers is the same to the digit, so the slow
part is not that sampling. Each ladder's spread enters the rule as `u`, and a reference whose
spread is too large for the error it judges is reported as such, not used.

Two instruments hold the comparison's frame. `frame`: a box spanning the cell along one axis
has a constant normal, where the normal-vector-field rule is Li's, and the raster and shape
paths agree to 3.1e-12 (along x) and 7.2e-12 (along y) at an offset centre. `shift`: RCWA
centres its shape at the origin and fd's pillar sits at `(xc, yc)`, so order `(m, n)`'s
amplitude differs by `exp(-2 pi i (m xc/Px + n yc/Py))` and its efficiency not at all; RCWA
solved at the offset matches the centred solve times that factor to 5.6e-12 or better on all
three scenes. fd's amplitudes are multiplied by the inverse factor before any comparison.

That factor is the pilot's one surprise. Compared without it, M2's `(-1, 0)` transmission
amplitude missed RCWA by 0.30 at 25 cells per wavelength and 0.058 at 50 -- an error that falls
with the grid only because the pillar's offset, fixed in cells, shrinks with the cell. With it,
the miss is 0.065 and 0.019. Nothing in either solver is wrong; the two place the pillar
differently, and a comparison of the non-zero orders' phases has to say where.

## The rule

Registered in the study's docstring and recorded (the file's hash and a copy, with each scene's
RCWA ladder) at 2026-10-03T02:19:40+0900, before any FDTD row existed; the report compares the
file it runs from with that record. A *cell* is one compared number: an order's outgoing
amplitude (phase-corrected) or its efficiency, at one wavelength and one incident polarisation.
`A(n)` and `F(n)` are the largest amplitude and efficiency errors over a scene's cells at rung
`n`. `u_A` and `u_F` are the largest distances, over the amplitude and the efficiency cells, of
the three ladder truncations below the top from the top -- one number per kind, as the junction
campaign's `u_ref` was one per scene. The Richardson pair is the finest rung and the finest one
at most two thirds of it: 40 and 80 for M1, 50 and 100 for M2, 80 and 120 for M3.

1. **Validity.** No plan is refused and no value is missing or non-finite; every rung run and
   every 180-period run settles (both monitors' decay below 1e-5) without a growth or unsettled
   warning, and reports the fixed-k angle. A rung run that does not settle at 120 periods is
   solved once more at 180 and that run stands for it. At the middle rung the 180-period run
   moves no cell by more than a tenth of the finest rung's error of its kind, and at the lowest
   rung JAX and Torch agree with NumPy to 1e-10 over amplitudes and efficiencies.
2. **A reference that can judge.** `u_A` and `u_F` are at most a third of the finest rung's
   errors; otherwise the pair is reported *not judgeable*.
3. **Second order.** `A` and `F` fall at every refinement, at a least-squares order of at least
   1.8 over the three rungs.
4. **Consistency.** In every cell the reference lies within `3 u + 0.1 |X(n_f) - X(n_c)|` of the
   segment of Richardson limits for orders 1.4 to 2 -- the junction campaign's band.

Two controls are judged by the same rule and must fail it: FDTD's amplitudes referred to a
pillar one cell off its axis (a first-order position error) must fail criteria 3 and 4, and every
reference value moved by the finest rung's error must fail criterion 4. A pair that would pass
while a control does not fail is reported *rule inert*. Before the finer rungs ran, a check
(`project`) compared each ladder's `u` with a third of the finest-rung error projected from the
lowest rung at order 1.8, and the campaign would not start the finer rungs until every ladder
held; a ladder may only grow at its top.

The rule was rewritten twice before it ran, each time after an adversarial review. The changes
that matter for reading the results: every face of every pillar now sits at the same fraction
of a cell at every rung (the first draft's M2 top and M3 sides moved within the cell from rung
to rung); M3's rungs became 40/80/120 (at 40/60/80 the slope leaned too hard on the ends); the
source band narrowed from 0.06 to 0.04; the controls' required outcomes were written down, and
the phase control became a one-cell misplacement (dropping the correction gave M3, at 120 cells,
an error only 1.3 to 1.6 times FDTD's own); the cells RCWA holds at zero by symmetry share their
kind's `u` instead of having none. The study's docstring lists every amendment.

*Stated before the data:* the junction campaign measured this same worst-cell statistic for its
pillar scenes at orders 1.74 (epsilon 4 on 2.25, height 0.6, 20/40/80 -- M1's materials and
rungs), 1.73 (epsilon 5.8 on 2.13 -- M2's), 1.48 and 1.77 for its other cylinders, and 1.89
and 2.02 for its box and ridge. Cylinders may fail criterion 3 here for the reason C3 named,
the default voxel rule's curved wall ([`fd-mie-validation.md`](fd-mie-validation.md)).

## Results

The worst error over a pair's cells at each rung, its least-squares order over the three
rungs, the ladder's spread, and the verdict (`report` of 2026-10-03T10:52:18+0900, protocol
`f34cb37f2f16c405`, the registered one):

| Pair | `A` at the three rungs | order | `F` at the three rungs | order | `u_A`, `u_F` | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| M1 s | 2.93e-2, 7.40e-3, 2.10e-3 | 1.90 | 1.74e-2, 4.58e-3, 1.36e-3 | 1.84 | 1.9e-4, 1.9e-4 | pass |
| M1 p | 2.78e-2, 6.60e-3, 1.81e-3 | 1.97 | 1.36e-2, 3.50e-3, 9.06e-4 | 1.96 | 2.3e-4, 1.7e-4 | pass |
| M2 s | 7.41e-2, 2.02e-2, 6.08e-3 | 1.80 | 1.21e-2, 2.05e-3, 3.87e-4 | 2.48 | 2.0e-4, 1.9e-4 | not judgeable |
| M2 p | 3.48e-2, 8.58e-3, 2.27e-3 | 1.97 | 1.80e-2, 4.48e-3, 1.11e-3 | 2.01 | 1.5e-4, 8.5e-5 | FAIL (4: one cell) |
| M3 s | 2.21e-2, 7.10e-3, 3.64e-3 | 1.64 | 1.16e-2, 4.09e-3, 2.19e-3 | 1.51 | 3.4e-4, 2.3e-4 | FAIL (3) |
| M3 p | 1.29e-2, 3.72e-3, 1.91e-3 | 1.75 | 6.02e-3, 1.95e-3, 1.02e-3 | 1.62 | 3.0e-4, 1.9e-4 | FAIL (3) |

Rungs are 20/40/80 (M1), 25/50/100 (M2) and 40/80/120 (M3) cells per vacuum wavelength; `A`
is the worst amplitude error and `F` the worst efficiency error over every propagating order,
both outgoing polarisations and the three wavelengths.

**M3's order is steady, not noisy.** Its local orders are 1.64 (40 to 80) and 1.65 (80 to 120)
for s, 1.80 and 1.64 for p. The reference cannot account for it: `u_A` is a tenth of
`A(120)`, and a reference off by that much at every rung tilts a three-rung slope with these
rungs by under 0.1. Nor can the run length (8e-7) or the backends. And the error is
consistent with RCWA: in all 54 cells the reference lies inside the band around the Richardson
limits for orders 1.4 to 2. What this campaign cannot say is which part of the box's
discretisation converges at first order: the candidates are the box's creases (its vertical
and top edges, where the cross terms take the nearest surface's normal as a stand-in,
[`fd-subpixel-occupancy.md`](fd-subpixel-occupancy.md)), the junction's rim where box, glass
and air meet, and the conical incidence that only M3 has. The junction campaign's own epsilon-12
box (D', normal incidence, the zeroth order only) converged at 1.89. The corners' own singularity was tested
first and ruled out at these resolutions (below, "Is it the corners?").

**M2 p's one cell.** The cross-polarised `t_s` of the `(0, 0)` order is zero in RCWA (1e-13);
FDTD's runs give 1.68e-3, 6.0e-4 and 2.65e-4 at 1.05 and 25, 50 and 100 cells per wavelength
(the same order's cell at 0.95 goes 1.18e-3, 1.90e-4, 1.56e-4). The pillar's axis sits a
quarter cell off a node line along y, so the grid is not mirror-symmetric about it and leaks a
little of one polarisation into the other; the leak shrinks with the cell, but erratically and
slower than second order, and the band, 3 `u_A` plus a tenth of the last step, is 5.2e-4 where
the cell sits 5.5e-4 from the Richardson segment. M1, mirror-symmetric too, leaks at most
2.9e-5 at 80 cells.

**M2 s's reference.** The `project` check, run before the finer rungs, found every ladder
sufficient for an error falling at order 1.8. M2 s's efficiencies fell at 2.48 instead, to
3.9e-4 at 100 cells, and the cylinder's shape-path ladder levels off near 2e-4 (its top three
truncations against (24, 24): 1.9e-4 at most). The rule then reports the pair as not
judgeable rather than read a reference that cannot tell.

**The controls failed as registered on every pair.** Referring FDTD's amplitudes to a pillar
one cell off its axis gives amplitude orders of 0.87 to 1.12 and leaves 3 to 30 cells outside
the band; moving every reference value by the finest rung's error leaves 19 to 70 cells
outside. So criteria 3 and 4 can fail on these numbers, and a pass is not a pass by default.

**Costs.** On this machine (12 logical CPUs, JAX on the CPU): M1 at 80 cells per wavelength
took 20 minutes, M2 at 100 19 minutes, M3 at 120 84 minutes per run; the RCWA ladders took
16 minutes (M1), 18 (M2) and 2.3 hours (M3, whose top truncation took 18 minutes per
wavelength).

## Is it the corners? (2026-10-03)

The first explanation for M3 was the one the subpixel-smoothing literature gives for sharp
dielectric corners: the field there goes as `r^(nu - 1)`, and an integrated quantity's
smoothing error falls as `dx^(2 nu)` rather than `dx^2` (Farjadpour et al., Opt. Lett. 31, 2972
(2006); Oskooi et al., Opt. Lett. 34, 2778 (2009)). The owner chose to test it before anything
else. `studies/fdtd_metasurface_corners.py` registered a prediction and a rule before its
campaign (hash recorded at 2026-10-03T20:52:08+0900) and ran it.

**The prediction.** A transfer-matrix condition gives each corner's exponent; two independent
derivations (a parity and nodal-system one with closed forms, and a spectral-element angular
eigenproblem checked by a graded finite-volume solve) agree with it to 1e-9, and a third check
to 40 digits (`studies/results/fdtd_metasurface_corners_crosscheck.md`).
For an epsilon-12 right angle in air `nu` is 0.7219 (order 1.444); for the base edge where an
epsilon-12 pillar, air and the 2.25 glass meet, 0.6845 (order 1.369); at epsilon 4 the two give
1.612 and 1.591. Farjadpour et al. quote 0.702 for their epsilon-12 corner; its exact exponent
is `(2/pi) arccos(11/26) = 0.7219`, and none of the derivations could produce 0.702 from the
paper's geometry -- the measured 1.4 they report agrees with 1.444 either way.

**The test.** M3's cross-section made a ridge uniform along y -- its only corners the four
edges, its run two-dimensional and cheap enough for 320 cells per wavelength -- at epsilon 12
(R12) and epsilon 4 (R4), and an epsilon-12 film with no corner at all (F12), all under M3's
conical incidence, against 1D RCWA references converged to 1e-5. The rule asked the local order
between 160 and 320 cells per wavelength to fall within 0.25 of the prediction for both ridges,
R4's above R12's, with F12 at least 1.8 as the control.

| Scene | Predicted | `A` at 40 / 80 / 160 / 320 | Local order 160 to 320 (`A`, `F`) |
| --- | --- | --- | --- |
| R12 | 1.37 - 1.44 | 6.1e-2 / 1.4e-2 / 3.5e-3 / 8.8e-4 | 2.00, 2.01 |
| R4 | 1.59 - 1.61 | 7.6e-3 / 1.9e-3 / 4.7e-4 / 1.2e-4 | 2.00, 2.01 |
| F12 | 2 | 8.7e-2 / 2.2e-2 / 5.5e-3 / 1.4e-3 | 2.00, 2.00 |

**The explanation is contradicted.** Every run settled, every reference could judge, the
control held -- and both ridges converge at second order from 40 cells to 320, every step a
factor of four, the median cell's local order 2.00 as well. Within the resolutions any
metasurface run uses, the corners of a ridge -- the epsilon-12 right angles and the base edges
where pillar, air and glass meet -- do not hold the order down under this smoothing. Whatever
makes M3 converge at 1.5 to 1.75 is three-dimensional: the box's vertical edges, its vertices,
or the junction's rim where it turns a corner. The asymptotic `dx^(2 nu)` regime may lie beyond
320 cells per wavelength; the measurement says only that it does not reach down to where these
runs live.

Before registration a dry run of every command on a toy configuration (rungs 40 and 80, 40
periods, a small RCWA ladder) was seen: local orders between 40 and 80 of 2.10 / 2.05 (R12), 2.00
/ 1.99 (R4) and 1.99 / 1.95 (F12). The rule was written before that run and not changed after;
the study's docstring says so, and the dry run's rows and report are kept beside the campaign's.
The registration records the study file's SHA-256, and the file committed with the results is
that file byte for byte.

The follow-up changed no production source. Its gates: the full fd/docs tier 2745 passed and 8
skipped of 2753, the slow tier 81 of 81, the four freezers 143/288/140/570 arrays bit for bit,
`ruff check .` and `sphinx-build -W` clean (`studies/results/fdtd_metasurface_corners_validation.json`,
logs under `studies/results/fdtd_metasurface_corners_controls/`). A first attempt stopped on the
documentation contract -- the cross-check record cited its scripts from its own folder, and a
copy of the registered study, kept beside the logs as C7's is, cited paths from the study's
directory; the record was corrected, the copy dropped (the committed study is the registered
file), and the attempt's logs are kept.

## What the tests hold

`tests/test_fd_metasurface.py` keeps three things in the suite.

* **The frame the comparison rests on.** Moving M1's pillar by one cell along x, then along y,
  multiplies every order's amplitude by `exp(-2 pi i (m dx/Px + n dy/Py))` and leaves every
  efficiency, to 1.8e-12 and 8.8e-13 of amplitudes up to 0.44 (without the factor the gap is
  0.16 and 0.11): moving a body by a whole cell is an exact symmetry of the grid. M1 is the
  scene with `n != 0` orders, so the y half is held too.
* **The campaign's rows.** M1 at 20 cells per wavelength (p) and M2 at 25 (s), and, in the slow
  tier, M3 at 40 (s), each solved again on NumPy through `fd.run` and held to the campaign's row
  to 1e-10 in every cell and to its recorded distance from RCWA, the phase referred to the
  centred pillar as the campaign refers it.

Each test function has a production mutation that fails it on an assertion
(`studies/fdtd_metasurface_mutations.py`, logs in
`studies/results/fdtd_metasurface_controls/`): the u and the v order projections conjugated,
the half-cell stagger left in the orders, the de-embedding walked the wrong way, a standing
box's u edge mixed by its v wall's weight. Two first attempts are kept with their logs: keeping
the v stagger survived M2's test (M2's orders all have `n = 0`, so no v phase reaches it -- the
y half is M1's), and making `'staircase'` every structure's junction default raised a refusal
on the substrate rather than failing an assertion.

## Validation

No production source changed in C7 (fd source SHA-256
`50574088a3e1e766f334890bcbbd5dbd5ad1dc600c9469766f65d63c949832d9`, the tree of `273514f4`).
The full fd/docs tier collected 2753 cases and exited 0 with 2745 passed and 8 skipped
(2123.80 seconds); the separate slow tier collected and passed all 81 cases, the new M3 test
among them, exit 0 (1120.79 seconds). Reported totals match collection, with no INTERNALERROR
or worker crash. The four read-only freezers reproduced all 143/288/140/570 committed arrays
bit for bit; no golden was rewritten. `ruff check .` (0.16.2) and `sphinx-build -W` pass.
Counts and timings are in `studies/results/fdtd_metasurface_validation.json`, the runner is
`studies/fdtd_metasurface_validation.py`, and the raw logs are under
`studies/results/fdtd_metasurface_controls/`.

## What this does not show

* **Why M3 converges below second order.** Not the corners of its cross-section ("Is it the
  corners?"); its vertical edges, its vertices and the junction's rim at the box's corners
  remain, and no measurement here separates them.
* **One pillar per cell.** No supercell, so no plan with several junctions, as a
  phase-gradient metasurface has.
* **Two body kinds.** Cylinders and unturned boxes, the bodies a junction supports; no prism,
  no turned box.
* **Lossless, non-dispersive dielectrics.** No metal and no pole; their subpixel smoothing is
  C8.
* **Orders off the plane of incidence.** Only M1 propagates orders with `n != 0`, and M1 is
  mirror-symmetric in y; M2's cross polarisation is zero by symmetry and tests only the grid's
  leak. M3, the scene with no symmetry, propagates `n = 0` orders only.
* **Three wavelengths of a fixed-k band.** Each at its own angle; nothing near a Rayleigh
  anomaly, which the scenes were chosen to avoid.
* **Gradients.** The owner left the derivative comparison out of C7; the C6 records stand on
  their own checks.

## Reproducing it

```bash
python studies/fdtd_metasurface_rcwa.py
```

runs every part in order and solves only what its logs do not already hold under the current
provenance; the registration step refuses once FDTD rows exist, and the report says whether the
file it runs from is the registered protocol. The campaign of record took 8.5 hours on one
machine (02:19 to 10:52): 2.8 hours of RCWA ladders and 5.6 of FDTD, two thirds of that the M3
runs at 120 cells per wavelength. `python studies/fdtd_metasurface_mutations.py` repeats the
tests' mutation witnesses.

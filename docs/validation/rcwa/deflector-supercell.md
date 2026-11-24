# A pillar deflector solved as a supercell: the local periodic approximation against the rigorous efficiency, and what gradient ascent recovers — measurements

Written: 2026-11-21. Measured on one CPU host: four logical cores of an Intel Xeon at 2.10 GHz
in a shared cloud container, Linux, Python 3.12.3, float64, library-default threading.
**Every time and memory figure on this page is from that one host. No GPU was used.**

Targets:
- `studies/rcwa/deflector_supercell/deflector_supercell.py` — the instrument
- `studies/rcwa/deflector_supercell/records/deflector_supercell.log` — its printed output, as measured
- `examples/rcwa/metalens_phase_library.py` — the pillar family
- [`metalens-library-cost.md`](metalens-library-cost.md) — what one library point costs
- [`../../guide/common/autodiff.md`](../../guide/common/autodiff.md) — which shape derivatives exist

## TL;DR

- **The rigorous efficiency is below the sampled prediction at every ramp, and the gap
  grows as the ramp steepens.** For s polarization the sampled prediction is 0.94, 0.95, 0.94,
  0.93 and 0.89 at 12, 8, 6, 4 and 3 pillars per 2π; the supercell solve gives 0.907, 0.912,
  0.886, 0.790 and 0.638. The gap is 0.04 at 12 and 8 pillars and 0.26 at 3.
- **Holding each pillar's amplitude constant over its cell (the staircase prediction) lands
  within 0.034 of the rigorous value for s** and within 0.17 for p. Neither form is a bound:
  the sampled prediction for p exceeds 1 at 3 and 4 pillars.
- **Thirty steps of gradient ascent on the diameters add 0.04 to 0.15 for s and 0.05 to 0.17
  for p**, with both designs re-solved at M = 6 where the ascent ran at M = 4. For s:
  0.638 → 0.790, 0.790 → 0.848, 0.886 → 0.935, 0.912 → 0.953 and 0.907 → 0.953 at 3, 4, 6, 8
  and 12 pillars.
- **One value-and-gradient step on PyTorch takes 1.3 s at 3 pillars (225 orders) and 33 s at
  12 (873 orders)**, with a peak resident memory of 1.2 and 7.8 GB. The derivative agrees with
  a central difference to 1.3e-6 or better in all ten runs.
- **The ascended designs are not designs the lookup would have made.** Several of their
  pillars sit on resonances the lookup excludes, and the prediction evaluated at the ascended
  diameters is lower than at the looked-up ones in nine of the ten runs.

## The question

A pillar deflector is usually designed one pillar at a time: each pillar gets the diameter
whose *periodic-array* transmission phase matches a linear ramp, as if its neighbours were
copies of itself. That is the local periodic approximation. A deflector is itself periodic, so
its whole ramp period is one unit cell that RCWA can solve with every pillar's actual
neighbours in it. Two things were not known beforehand for this pillar family:

1. how far the rigorous efficiency into the design order is from the prediction, as the ramp
   gets steeper;
2. how much of it gradient ascent on the diameters recovers, starting from the looked-up
   design, and what one step costs.

Wavebit solves the periodic cell. Nothing here assembles a lens, propagates a field to a focus
or writes a layout.

## Scene and method

TiO2 pillars (n = 2.43, constant) on fused silica (n = 1.46), hexagonal pitch 350 nm, height
600 nm, 532 nm, normal incidence from air, transmission into the substrate. Diameters 80 to
260 nm in 10 nm steps. This is the family of `examples/rcwa/metalens_phase_library.py`; the
study computes its own library and imports nothing from the example.

**Cell.** The hexagonal lattice is written with `a1 = (303.1, -175)` nm and `a2 = (0, 350)` nm,
so one row of pillars lies along `y` and neighbouring rows are 303.1 nm apart in `x`. A
deflector with N pillars per ramp period is the cell `(N*a1, a2)` holding N circles on `a1`.
Its orders `(m, 0)` have `ky = 0`, so the deflection is in the `xz` plane; `'s'` is the
electric field along `y` and `'p'` is the field along `x` at incidence. `rcwa.check` reports no
issue for these cells.

| N | 3 | 4 | 6 | 8 | 12 |
| --- | --- | --- | --- | --- | --- |
| order (1, 0) in the substrate (deg) | 23.62 | 17.49 | 11.56 | 8.64 | 5.75 |
| the same once in air (deg) | 35.81 | 26.03 | 17.01 | 12.67 | 8.41 |

**Truncation.** The cell keeps `|m| <= N*M` and `|n| <= M`: the primitive cell's
`|m|, |n| <= M` and every order between them, so the pillars are resolved as in a primitive
cell at M. M = 4, 5 and 6 were run, which is 225 to 1885 orders.

**Library.** `t(d)`: the co-polarized zero-order transmitted amplitude ratio of the primitive
cell with one pillar of diameter `d`, at 289 orders, with its transmittance `T(d)`. Sixteen of
the nineteen entries have `T >= 0.9` and are *usable*; 180, 210 and 220 nm are on resonances
(T = 0.49, 0.79, 0.52 for s) and are not offered to the lookup.

**Design.** For the ramp `exp(i*(phi0 + 2*pi*k/N))`, pillar `k` takes the usable entry with the
largest `Re(t * conj(target))`. `phi0` is whichever of 360 offsets gives the design with the
largest predicted efficiency.

**Prediction.** The field under pillar `k` is taken to be `t(d_k)` times the incident field.
Then

```text
c_m   = (1/N) * sum_k t(d_k) * exp(-2*pi*i*m*k/N)
eta_m = |c_m|^2 * F_m,    F_m = kz_m/k0 (s),   n_sub^2 * k0/kz_m (p)
```

with `kz_m` the normal wavevector of order `m` in the substrate. `F_m` is the power an order
carries per unit squared tangential field, relative to the incident wave. This is the
*sampled* prediction: one field value per pillar. The *staircase* prediction holds `t(d_k)`
constant across pillar `k`'s cell, which multiplies `eta_m` by `sinc(m/N)^2`. The two
forms differ by 2% at 12 pillars and 32% at 3, so both are reported.

**Anchor.** For N equal pillars both predictions reduce to `|t|^2 * n_sub`, the library's
co-polarized zero-order transmittance; the script asserts that this equals the primitive
cell's `T` less the cross-polarized zero order a finite truncation leaves (3.0e-4 at most).
A cell of four equal pillars is then solved against the primitive cell at the same M:

| | s | p |
| --- | --- | --- |
| largest difference in T | 8.9e-5 | 1.8e-4 |
| largest difference in phase (rad) | 0.0107 | 0.0085 |
| largest efficiency in any other order | 4.2e-27 | 1.1e-27 |

The two cells keep the same orders, and the orders between the primitive ones stay empty to
rounding. T still differs by up to 1.8e-4 and the phase by up to 0.011 rad, and the
difference does not shrink monotonically from M = 4 to 6. The cause is the normal field of the
default factorization: it takes one shape's nearest periodic image but blends different shapes,
so four copies in the supercell get a different field between the pillars from one pillar in the
primitive cell. For the 150 nm pillar under `factorization='laurent'`, which reads no normal
field, the two cells agree to rounding, and under the default both converge to one value.
[Known limits §10](normal-field-known-limits.md) has the measurements.

**Rigorous value.** One NumPy solve of the cell with the default settings, the efficiency of
transmitted order (1, 0). Every solve asserts `R + T = 1` to 1e-9.

**Ascent.** The efficiency of order (1, 0) is maximized over the N diameters by Adam with a
2 nm step, 30 steps, starting from the looked-up design, with each diameter clipped to the
library's range of 80 to 260 nm after every step. The gradient comes from PyTorch 2.15.0
through `rcwa.Circle(radius=...)`, at M = 4 and with `Numerics(shape_nvf_grid_size=512)`
instead of the default 1024, because the step's memory grows with that grid and with the
number of pillars. The iterate with the highest value is kept and solved again on NumPy at
M = 4, 5 and 6 with the default grid.

## Prediction against the supercell

Efficiency into transmitted order (1, 0) of the looked-up design. The rigorous column is
M = 6; *spread* is the largest minus the smallest of M = 4, 5, 6.

**s polarization**

| N | sampled | staircase | rigorous | spread |
| --- | --- | --- | --- | --- |
| 3 | 0.8929 | 0.6106 | 0.6378 | 2.8e-3 |
| 4 | 0.9327 | 0.7560 | 0.7898 | 2.3e-3 |
| 6 | 0.9400 | 0.8572 | 0.8860 | 1.7e-3 |
| 8 | 0.9478 | 0.9000 | 0.9125 | 8.2e-4 |
| 12 | 0.9442 | 0.9228 | 0.9069 | 2.1e-3 |

**p polarization**

| N | sampled | staircase | rigorous | spread |
| --- | --- | --- | --- | --- |
| 3 | 1.0639 | 0.7276 | 0.5792 | 2.4e-3 |
| 4 | 1.0253 | 0.8311 | 0.6674 | 3.8e-3 |
| 6 | 0.9793 | 0.8930 | 0.8576 | 2.2e-3 |
| 8 | 0.9694 | 0.9206 | 0.8899 | 8.2e-4 |
| 12 | 0.9541 | 0.9325 | 0.8048 | 1.3e-2 |

**The sampled prediction is high everywhere, by more the steeper the ramp.** For s it exceeds
the rigorous value by 0.037, 0.035, 0.054, 0.143 and 0.255 at 12, 8, 6, 4 and 3 pillars; for p
by 0.149, 0.080, 0.122, 0.358 and 0.485. For p at 3 and 4 pillars it is above 1: the sampled
sum assigns each order a power without the orders having to add up to the incident power.

**The staircase prediction is the closer of the two.** For s the rigorous value is above it by
0.013 to 0.034 from 3 to 8 pillars and below it by 0.016 at 12. For p the rigorous value is
below it throughout, by 0.031 to 0.164.

**The gentlest ramp is not the best predicted.** At 12 pillars the s design falls short of
both predictions, and the p design gives 0.805 where 8 pillars give 0.890. The p value at 12
pillars is also the least settled on this page: it moves by 1.3e-2 from M = 4 to 6, three times
the next largest spread, and its derivative with respect to its 230 nm pillar is -0.043 per
nanometre, five times any other derivative checked. Both are what a resonance near the working
point looks like. The 230 nm entry is the neighbour of the excluded 210 and 220 nm entries; in
the primitive cell its T is 0.97.

**Where the rest goes**, for s at M = 6: the looked-up designs transmit 0.961 to 0.990 in
total and reflect the rest. At 3 pillars order (2, 0) takes 0.129 and order (-1, 0) takes
0.060; at 4 pillars order (2, 0) takes 0.064. The log has orders (0, 0), (-1, 0) and (2, 0)
for every row.

## What the ascent recovers

Efficiency into order (1, 0), solved on NumPy with the default grid. *Looked-up* and
*ascended* are at M = 6; the last two columns are the ascended design at M = 4 and M = 5. The
ascent itself ran at M = 4.

**s polarization**

| N | looked-up | ascended | gain | M = 4 | M = 5 |
| --- | --- | --- | --- | --- | --- |
| 3 | 0.6378 | 0.7901 | 0.152 | 0.7899 | 0.7901 |
| 4 | 0.7898 | 0.8484 | 0.059 | 0.8496 | 0.8471 |
| 6 | 0.8860 | 0.9353 | 0.049 | 0.9364 | 0.9353 |
| 8 | 0.9125 | 0.9528 | 0.040 | 0.9519 | 0.9524 |
| 12 | 0.9069 | 0.9533 | 0.046 | 0.9525 | 0.9532 |

**p polarization**

| N | looked-up | ascended | gain | M = 4 | M = 5 |
| --- | --- | --- | --- | --- | --- |
| 3 | 0.5792 | 0.6579 | 0.079 | 0.6610 | 0.6592 |
| 4 | 0.6674 | 0.8366 | 0.169 | 0.8365 | 0.8365 |
| 6 | 0.8576 | 0.9556 | 0.098 | 0.9561 | 0.9562 |
| 8 | 0.8899 | 0.9416 | 0.052 | 0.9421 | 0.9419 |
| 12 | 0.8048 | 0.9536 | 0.149 | 0.9532 | 0.9537 |

The gain survives the finer truncation: the ascended values move by at most 3.1e-3 between
M = 4 and 6, against gains of 0.040 and more. The ascent was stopped at 30 steps, not at
convergence. Adam's trajectory is not monotonic; the best iterate was the last one in three
runs and between steps 17 and 26 in the other seven. One diameter ended on a bound (80 nm,
p at 3 pillars).

**The ascended designs leave the lookup's rules.** The ascent is constrained only to the
diameter range, and it does not avoid resonances:

- for s at 4 pillars a pillar moves from 190 to 181.95 nm, beside the excluded 180 nm entry;
  its primitive-cell T is 0.933;
- for p at 8 pillars a pillar moves from 240 to 214.91 nm, where its primitive-cell T is
  0.172, and the deflector's efficiency is 0.942;
- for p at 12 pillars two pillars end at 176.43 and 221.64 nm, with primitive-cell T of 0.71.

The prediction evaluated at the ascended diameters, from primitive-cell solves at those
diameters, does not follow the gain:

| N | 3 | 4 | 6 | 8 | 12 |
| --- | --- | --- | --- | --- | --- |
| s, sampled, looked-up | 0.893 | 0.933 | 0.940 | 0.948 | 0.944 |
| s, sampled, ascended | 0.831 | 0.850 | 0.923 | 0.944 | 0.955 |
| s, rigorous, ascended | 0.790 | 0.848 | 0.935 | 0.953 | 0.953 |
| p, sampled, looked-up | 1.064 | 1.025 | 0.979 | 0.969 | 0.954 |
| p, sampled, ascended | 0.925 | 0.848 | 0.961 | 0.796 | 0.862 |
| p, rigorous, ascended | 0.658 | 0.837 | 0.956 | 0.942 | 0.954 |

In nine of the ten runs the prediction falls while the rigorous efficiency rises. The
per-pillar prediction does not see what the ascent gains.

## What a step costs

One step is one PyTorch solve at M = 4 with the 512 grid and its backward pass: the value and
all N derivatives. Seconds are the median over the 31 evaluations of a run; memory is the
process's peak resident size when the ascent ended, before the NumPy re-solves. The NumPy
column is one forward solve of the looked-up s design at the same M with the default grid.

Time in seconds:

| N | orders | step, s | step, p | NumPy |
| --- | --- | --- | --- | --- |
| 3 | 225 | 1.34 | 1.04 | 1.7 |
| 4 | 297 | 2.28 | 1.73 | 2.7 |
| 6 | 441 | 6.18 | 5.39 | 5.0 |
| 8 | 585 | 11.58 | 11.14 | 9.4 |
| 12 | 873 | 32.90 | 33.65 | 23.0 |

Peak resident memory when the ascent ended, in GB:

| N | 3 | 4 | 6 | 8 | 12 |
| --- | --- | --- | --- | --- | --- |
| s | 1.19 | 1.56 | 3.79 | 4.51 | 7.75 |
| p | 1.14 | 1.75 | 3.72 | 5.39 | 7.80 |

The 31 evaluations of an ascent took 0.7 minutes at 3 pillars and 17 minutes at 12.

**The derivative.** Before each ascent, the derivative with respect to one diameter was
compared with a central difference of two NumPy solves 0.002 nm apart. The relative difference
is between 3.9e-9 and 1.7e-7 in nine runs and 1.3e-6 in the tenth, p at 12 pillars, the one
with the steep derivative above.

**The coarser grid.** Solved on NumPy at M = 4, the looked-up s designs change by at most
1.2e-5 between the 512 grid and the default, and by at most 7.2e-5 between 256 and the
default. The ascent was not run with the default grid.

**What did not fit.** The largest solve on this page is 12 pillars at M = 6, 1885 orders: 195
to 234 s, with the process at 9.9 GB afterwards. The host gives about 13 GB, so M = 7 at 12
pillars (2535 orders) was not attempted, and the three truncations here are all there are.

## What this does and does not establish

- One pillar family, one wavelength, normal incidence, a lossless constant-index dielectric,
  ramps along one lattice direction. Other heights, pitches, materials with dispersion or
  loss, and oblique incidence were not run.
- Both polarizations were run, as the s and p defined above. No other polarization state.
- Neither the prediction nor the truncated solve is an exact reference. The rigorous values
  are settled to 3.8e-3 or better over M = 4 to 6 in nineteen of twenty cases and to 1.3e-2 in
  the other; no extrapolation to infinite orders was made and no second solver was run.
- "The prediction" is the lookup and the two sums defined above. A lookup with a finer table,
  a different rule for excluding entries or a different amplitude model gives other numbers.
- The ascent is thirty steps of one optimizer with one step size from one starting point. The
  gains are what those steps found, not the best this family can do, and no fabrication
  constraint beyond the diameter range was applied: the ascended diameters are not on a
  10 nm grid and several sit where a small diameter error changes the response quickly.
- The times and memory are single runs on one shared CPU host with default threading. They do
  not carry to another machine. Wavebit's JAX backend, a jitted step and any GPU were not
  measured here.

# What one point of a pillar phase library costs, and what fmmax gives for the same array — measurements

Written: 2026-11-21. Measured on one CPU host: four logical cores of an Intel Xeon at 2.10 GHz
in a shared cloud container, Linux, Python 3.12.3, float64, library-default threading.
**Every time on this page is from that one host. No GPU was used.**

Targets:
- `studies/rcwa/metalens_library_cost/metalens_library_cost.py` — the instrument
- `studies/rcwa/metalens_library_cost/records/metalens_library_cost.log` — its printed output, as measured
- `examples/rcwa/metalens_phase_library.py` — the library this point belongs to
- `examples/rcwa/metalens_phase_sensitivity.py` — the phase derivative used below for scale
- [`../../theory/rcwa/oblique-lattice.md`](../../theory/rcwa/oblique-lattice.md) — the worked case
  and its S4 comparison

## TL;DR

- **One library point at 289 retained orders takes 1.3–1.4 s with Wavebit's NumPy backend.**
  fmmax 1.7.1 on the same array and the same orders takes 1.2–1.5 s under `jax.jit` and
  1.9–4.1 s eagerly.
- **At 841 orders Wavebit is the slower of the two:** 21 s against 13–15 s for fmmax under jit,
  a factor of 1.3 to 1.6. Nothing on this page says Wavebit is faster than fmmax.
- **The value is settled at few orders.** From 121 to 841 orders Wavebit's transmittance moves by
  at most 2.0e-4 and its phase by at most 4.8e-3 rad. fmmax's `JONES_DIRECT` moves by at most
  1.1e-3 and 5.1e-3 rad. fmmax's `FFT` formulation is still moving at 841 orders.
- **At 841 orders the two solvers agree to 1.4e-4 in transmittance and differ by 0.012–0.013 rad
  in phase, and that phase difference is fmmax's raster.** At 289 orders it is 0.029, 0.013 and
  0.008 rad for rasters of 128, 256 and 512 pixels a side, while Wavebit's phase moves by
  1.3e-6 rad across its normal-field grids.

## The question

A metalens design looks up one complex number per pillar: the zero-order transmitted amplitude
of the periodic array that pillar would belong to. The table is cheap or expensive according
to two things: what one point costs, and how many orders that point needs before its phase
stops moving. This page measures both on the array of the
[worked case](../../theory/rcwa/oblique-lattice.md#a-worked-case-a-visible-metalens-phase-library),
and runs fmmax on the same array so that the numbers have something outside this repository
beside them.

## Scene and method

TiO2 pillars (n = 2.43) on fused silica (n = 1.46), hexagonal pitch 350 nm, height 600 nm,
532 nm, normal incidence from air. Two diameters away from the library's resonances, 150 and
240 nm. Both solvers keep the orders `|m|, |n| <= M` on the same 120-degree basis for
M = 5, 8, 11, 14, which is 121, 289, 529 and 841 orders; the script asserts that fmmax's
expansion is that set.

- **Wavebit 0.0.5** is given the circle itself, as an analytic shape transform, and uses its
  default NVF factorization on the NumPy 2.5.5 backend, eagerly.
- **fmmax 1.7.1** is given a raster, as fmmax takes one: 256 x 256 pixels in lattice
  coordinates, each the area average of a 4 x 4 sub-sampling. It runs `Formulation.FFT` and
  `Formulation.JONES_DIRECT` on JAX 0.12.1, eagerly and under `jax.jit`.

Each row reports the total transmittance, the phase of the co-polarized zero-order transmitted
amplitude with reference planes at the two faces of the pillar layer, and the wall time of one
solve: the median of three calls after one untimed call. For the jit rows the untimed call is
the compilation; the log prints its time beside each row (1.3 to 19 s).

The two halves ran one after the other, in separate environments. The container is shared
hardware and was not isolated beyond that. fmmax was installed with
`pip install fmmax==1.7.1` and used as shipped.

## Time for one point

Seconds, median of three, against the number of retained orders.

**150 nm**

| solver | 121 | 289 | 529 | 841 |
| --- | --- | --- | --- | --- |
| Wavebit, NumPy | 0.16 | 1.31 | 6.39 | 21.35 |
| fmmax `FFT`, eager | 0.73 | 2.04 | 6.01 | 16.83 |
| fmmax `FFT`, jit | 0.14 | 1.31 | 4.52 | 13.37 |
| fmmax `JONES_DIRECT`, eager | 2.58 | 4.06 | 8.86 | 22.17 |
| fmmax `JONES_DIRECT`, jit | 0.21 | 1.50 | 5.84 | 15.41 |

**240 nm**

| solver | 121 | 289 | 529 | 841 |
| --- | --- | --- | --- | --- |
| Wavebit, NumPy | 0.16 | 1.38 | 6.43 | 20.79 |
| fmmax `FFT`, eager | 0.72 | 1.88 | 5.58 | 16.93 |
| fmmax `FFT`, jit | 0.19 | 1.19 | 4.26 | 13.70 |
| fmmax `JONES_DIRECT`, eager | 2.67 | 4.11 | 8.05 | 20.81 |
| fmmax `JONES_DIRECT`, jit | 0.20 | 1.43 | 4.98 | 15.45 |

Against fmmax's `JONES_DIRECT` under jit, the formulation whose value is comparable, Wavebit
takes 0.8 times as long at 121 orders, 0.9 to 1.0 at 289, 1.1 to 1.3 at 529 and 1.3 to 1.4 at
841. Where Wavebit's extra time at 529 and 841 orders goes was not profiled.

The 19-point library of the example, at 289 orders, took 39 s on this host.

## Value against the number of orders

Transmittance T and phase in radians, against the number of retained orders. `JONES_DIRECT`
and `FFT` are fmmax's two formulations.

**150 nm**

| | 121 | 289 | 529 | 841 |
| --- | --- | --- | --- | --- |
| Wavebit, T | 0.966610 | 0.966428 | 0.966440 | 0.966406 |
| Wavebit, phase | 3.15073 | 3.14733 | 3.14737 | 3.14994 |
| `JONES_DIRECT`, T | 0.967389 | 0.966549 | 0.966288 | 0.966334 |
| `JONES_DIRECT`, phase | 3.15919 | 3.16046 | 3.16139 | 3.16177 |
| `FFT`, T | 0.967450 | 0.966983 | 0.966796 | 0.966690 |
| `FFT`, phase | 3.32416 | 3.26325 | 3.23548 | 3.21719 |

**240 nm**

| | 121 | 289 | 529 | 841 |
| --- | --- | --- | --- | --- |
| Wavebit, T | 0.999075 | 0.999049 | 0.999043 | 0.999042 |
| Wavebit, phase | 0.79010 | 0.79344 | 0.79468 | 0.79492 |
| `JONES_DIRECT`, T | 0.999026 | 0.998779 | 0.998864 | 0.998902 |
| `JONES_DIRECT`, phase | 0.80275 | 0.80659 | 0.80757 | 0.80787 |
| `FFT`, T | 0.990552 | 0.995022 | 0.996689 | 0.997486 |
| `FFT`, phase | 0.96815 | 0.91070 | 0.88175 | 0.86441 |

Three things can be read off these.

**Wavebit and `JONES_DIRECT` both settle within a few milliradians.** Wavebit's phase spans
3.4e-3 rad over the four truncations at 150 nm and 4.8e-3 rad at 240 nm; `JONES_DIRECT` spans
2.6e-3 and 5.1e-3 rad. Wavebit's is not monotonic at 150 nm: it rises by 2.6e-3 rad from 529
to 841 orders. A phase library with eight levels has a level spacing of 0.785 rad, so either
solver's 121-order phase is within one percent of a level of its own 841-order phase.

**`FFT` has not settled.** Its phase moves by 0.107 rad at 150 nm and 0.104 rad at 240 nm over
the same range and is still moving at 841 orders; its transmittance at 240 nm is 0.9975 there
against 0.9990 and 0.9989 from the other two. This is the direct rule on a high-contrast boundary, the
behaviour [the factorization terms](../../theory/rcwa/factorization-terms.md) describe, and it
is why a per-point time for `FFT` is not the time to an answer.

**The two vector formulations differ by a constant.** At 841 orders the transmittances agree
to 7.2e-5 at 150 nm and 1.4e-4 at 240 nm, and the phases differ by 0.0118 and 0.0130 rad. The
difference stays between 0.008 and 0.014 rad at every truncation, which points away from the
order count.

## The phase difference is the raster

With the truncation held at 289 orders and the diameter at 150 nm, each solver's input was
refined instead: Wavebit's normal-field grid, and the raster given to fmmax's `JONES_DIRECT`.

| input | T | phase (rad) | minus Wavebit's |
| --- | --- | --- | --- |
| Wavebit, grid 512 | 0.966428 | 3.1473263 | |
| Wavebit, grid 1024 (default) | 0.966428 | 3.1473253 | |
| Wavebit, grid 2048 | 0.966428 | 3.1473250 | |
| fmmax, raster 128 | 0.966516 | 3.1761636 | 0.0288 |
| fmmax, raster 256 | 0.966549 | 3.1604642 | 0.0131 |
| fmmax, raster 512 | 0.966550 | 3.1549330 | 0.0076 |

Wavebit's phase moves by 1.3e-6 rad across its grids. fmmax's moves toward it by 0.0157 rad
and then 0.0055 rad as the pixel halves. The remaining 0.0076 rad at 512 pixels is still
falling, and what it falls to was not measured.

For scale: the phase of this entry changes by 0.0559 rad per nanometre of diameter
(`metalens_phase_sensitivity.py`), so 0.013 rad is what 0.23 nm of diameter does. A 256-pixel
raster of a 350 nm cell has 1.4 nm pixels.

## What this does and does not establish

- The times are one run on one shared host with default threading, the median of three calls.
  They are not a benchmark of either package and do not carry to another machine.
- Wavebit's JAX and PyTorch backends, a jitted Wavebit solve of this circle
  (`Numerics(topology_check='host')`) and any GPU were not measured.
- fmmax was given one raster recipe. A different raster, or a different formulation among those
  fmmax offers, changes its value and was not explored beyond the table above.
- Two diameters, one wavelength, normal incidence, a lossless dielectric. Nothing here speaks
  to a metal, to an entry on a resonance, or to oblique incidence, where more than one order
  can propagate in the substrate.
- Neither solver is an exact reference for this array. The statements are about each solver's
  own convergence and about the distance between the two.
- S4 and RETICOLO were not run for this page. The S4 values quoted on the
  [oblique-lattice page](../../theory/rcwa/oblique-lattice.md) are that page's own.

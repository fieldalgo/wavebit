# Pulse settling versus spatial accuracy

Written: 2026-09-08

Targets:
- `src/wavebit/_core/grid_fdtd.py`
- `src/wavebit/_core/scene_fdtd.py`
- `studies/fdtd_pulse_convergence.py`
- `tests/test_fdtd_convergence.py`

TL;DR: a small temporal tail does not certify accurate scattering. Report the
change in calibrated complex amplitudes separately from field decay, and refine
the spatial grid independently.

## Measurement

Run `PYTHONPATH=. python studies/fdtd_pulse_convergence.py` from the checkout.
The measurements below used NumPy 2.5.1, CPU arm64, complex128 fields. They are
accuracy measurements, not a timing or GPU benchmark.

Both lossless slabs are surrounded by vacuum, with index n and thickness 0.5/n
at vacuum wavelength 1. The exact Airy transmission amplitude is -1. Normal TE
incidence, period 0.2, resolution 30, 12 PML cells, default padding and uniform
mesh are fixed; only runtime changes. Diagnostics inspect the final 100 steps.
Field decay is the maximum over both bases and both actual/reference runs;
amplitude change is the maximum absolute change of the four given-polarization
complex r/t amplitudes between the full run and the prefix before that window.

| Index | Periods | Field decay | Max amplitude change | Absolute t error against Airy |
| --- | --- | --- | --- | --- |
| 1.5 | 20 | 4.04276e-05 | 1.11172e-06 | 0.012781 |
| 1.5 | 30 | 1.43708e-05 | 3.30148e-06 | 0.0127793 |
| 1.5 | 60 | 3.95328e-06 | 9.31763e-07 | 0.0127813 |
| 4.0 | 20 | 2.29894e-04 | 3.37110e-05 | 0.178242 |
| 4.0 | 30 | 1.39036e-05 | 2.93639e-06 | 0.178248 |
| 4.0 | 60 | 3.88659e-06 | 9.28665e-07 | 0.178250 |

## What this determines

Increasing duration leaves the Airy error essentially unchanged. At index 4,
resolution 30 gives only 7.5 nominal cells per internal wavelength and 3.75 per
slab thickness; longer runtime cannot repair the spatial representation. These
measurements do not isolate interface, bulk numerical dispersion, and PML errors.
They establish that the temporal indicators alone cannot certify their absence.

Even the temporal tail need not decrease monotonically: window phase and weak
residual reflections affect the complex difference. Use multiple durations and
window lengths. The `amplitudes_settled` flag is an absolute amplitude-tolerance
check for the requested polarization, not relative accuracy of a near-zero
channel, a bound on future error, or a general convergence certificate.

The implementation recalibrates the prefix with its own two reference bases;
using the final incident calibration for both windows would hide changes in the
reference run. A singular or numerically vanishing prefix reference is rejected
instead of being inverted. Tests compare this calculation with independent
shorter scene executions for top/bottom incidence on graded meshes, and compare
JAX/Torch derivatives with material finite differences. The earlier exact
conductive-field tests verify the native DFT and decay measurements.

This study covers two planar dielectric slabs. High-Q cavities, strongly
dispersive material oscillations and deeply evanescent channels need their own
controls. Field norms omit ADE/PML auxiliary states; neither temporal flag
certifies those states have decayed. Existing spatial-convergence checks remain
necessary; see [the graded-slab Airy tests](../tests/test_fdtd_scene_graded.py).

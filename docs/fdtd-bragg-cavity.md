# Bragg cavity temporal and spatial convergence

Written: 2026-09-08

Targets:
- `studies/fdtd_bragg_cavity.py`
- `tests/test_fdtd_cavity_convergence.py`
- `wavebit/_core/scene_fdtd.py`

TL;DR: a resonant cavity can require hundreds of periods. The temporal diagnostics
reject the short runs in this case, but passing those diagnostics does not remove
spatial detuning of a narrow resonance.

## Independent reference and setup

The cavity consists of three quarter-wave n=2/1 pairs, a half-wave vacuum defect,
and the reversed mirror. At vacuum wavelength 1, the thicknesses are 0.125 for
n=2, 0.25 for n=1 mirror layers and 0.5 for the defect. Both half spaces are
vacuum. At normal incidence the product of the lossless characteristic matrices
is -I, so the interface-referenced transmission amplitude is exactly -1 and T=1.

`studies/fdtd_bragg_cavity.py` implements the two-by-two characteristic matrices
independently of the FDTD and RCWA solvers. Interpolation of the half-power
frequencies on a 40001-point frequency grid from 0.98 to 1.02 gives a linewidth
Q of about 148.38. This is the continuum reference linewidth, not a fitted FDTD
lifetime or a validation of arbitrarily high Q.

Run `PYTHONPATH=. python studies/fdtd_bragg_cavity.py`. The FDTD measurements use
NumPy 2.5.1 on arm64 CPU with complex128 fields, period 0.05, normal TE incidence,
uniform grids, default padding and 12 PML cells. No GPU or fresh external-solver
run is involved. The wavelength is fixed at the continuum resonance rather than
retuned to a mesh-dependent transmission maximum.

## Runtime control at resolution 40

| Periods | Diagnostic window (steps) | T | Absolute error in complex t | Field decay | Absolute final-window t change | Amplitudes settled |
| --- | --- | --- | --- | --- | --- | --- |
| 25 | 200 | 0.0989567 | 0.687341 | 0.185162 | 0.0441026 | False |
| 80 | 200 | 0.611545 | 0.315580 | 0.0498649 | 0.0107406 | False |
| 200 | 200 | 0.853767 | 0.385249 | 0.00349082 | 0.000753339 | False |
| 400 | 433 | 0.848353 | 0.398159 | 0.0000440490 | 0.0000199822 | True |

The source is already inactive in the short runs; the cavity is still ringing.
Increasing runtime does not guarantee monotonic approach to the *continuum*
answer at a fixed coarse mesh. Transient and spatial errors can partially cancel,
as the 80-period row illustrates. The longer window at 400 periods makes the
late-time settling check stricter; window sizes are stated explicitly.

The regression test verifies that the 25/80-period runs have valid incident
calibration and stable material/CFL inputs but fail both temporal flags, with a
large change in resonant transmission. It is a failure-detection check against
an independently known unit-transmission resonance, not a general high-Q solver
accuracy certificate.

## Spatial control after long runtime

| Resolution | Periods | Window (steps) | T | Absolute complex t error | Field decay | Absolute final-window t change |
| --- | --- | --- | --- | --- | --- | --- |
| 40 | 400 | 433 | 0.848353 | 0.398159 | 0.0000440490 | 0.0000199822 |
| 80 | 400 | 866 | 0.987863 | 0.110469 | 0.0000586005 | 0.0000259262 |

Both runs pass both temporal flags at tolerance 1e-4. Doubling spatial resolution
reduces the complex t error by about 3.6 times. The remaining error is still
substantial on the scale of a narrow resonance: T near one alone obscures a
roughly 0.11 error in complex amplitude, dominated here by phase. This two-grid
control demonstrates refinement toward the analytic answer; it does not establish
an asymptotic order or a converged phase. PML cell count is fixed, so its physical
width also changes with resolution; this experiment does not isolate PML and
bulk/interface discretization contributions.

## Scope and remaining controls

The test does not change the solver because the existing temporal indicators
correctly reject these unfinished runs. The additional validation protects that
behavior from future optimizations. Mesh, PML thickness and padding still need
independent controls for a user's cavity, especially when evanescent fields reach
the absorbing layer or dispersive material states outlive the measured fields.

Meep's public [resonant-mode documentation](https://meep.readthedocs.io/en/latest/Python_Tutorials/Resonant_Modes_and_Transmission_in_a_Waveguide_Cavity/)
discusses the long lifetime of resonances and direct excitation for lifetime
analysis; its [FAQ](https://meep.readthedocs.io/en/latest/FAQ/) identifies insufficient
runtime, overlapping source activity and inappropriate monitor placement as
possible problems. Only documentation was consulted; no Meep implementation was
opened. This study uses its own analytic reference and does not use harmonic
inversion or import an external solver implementation.

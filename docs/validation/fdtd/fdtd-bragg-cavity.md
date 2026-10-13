# Bragg cavity temporal and spatial convergence

Written: 2026-09-08

Targets:
- `studies/fdtd/bragg_cavity/fdtd_bragg_cavity.py`
- `tests/fdtd/test_fdtd_cavity_convergence.py`
- `src/wavebit/fdtd/_periodic/scene.py`

TL;DR: a resonant cavity can require hundreds of periods. The temporal diagnostics
reject the short runs in this case, but passing those diagnostics does not remove
spatial detuning of a narrow resonance.

## Independent reference and setup

The cavity consists of three quarter-wave n=2/1 pairs, a half-wave vacuum defect,
and the reversed mirror. At vacuum wavelength 1, the thicknesses are 0.125 for
n=2, 0.25 for n=1 mirror layers and 0.5 for the defect. Both half spaces are
vacuum. At normal incidence the product of the lossless characteristic matrices
is -I, so the interface-referenced transmission amplitude is exactly -1 and T=1.

`studies/fdtd/bragg_cavity/fdtd_bragg_cavity.py` implements the two-by-two characteristic matrices
independently of the FDTD and RCWA solvers. Interpolation of the half-power
frequencies on a 40001-point frequency grid from 0.98 to 1.02 gives a linewidth
Q of about 148.38. This is the continuum reference linewidth, not a fitted FDTD
lifetime or a validation of arbitrarily high Q.

Run `PYTHONPATH=. python studies/fdtd/bragg_cavity/fdtd_bragg_cavity.py`. The FDTD measurements use
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

## PML and padding controls

Run `PYTHONPATH=. python studies/fdtd/bragg_cavity/fdtd_bragg_cavity.py --controls`. All controls
use 400 periods and the same wavelength, materials, time-step rule and uniform
mesh as above. At resolution 40, PML thickness and padding are changed separately.
`pad_scale` multiplies the nominal vacuum gaps used to place the source and flux
planes, so this control moves those planes as well as the outer boundary.

| Resolution | PML cells | PML physical width | pad_scale | T | Continuum complex t error | Max complex r/t error against discrete reference |
| --- | --- | --- | --- | --- | --- | --- |
| 40 | 12 | 0.3 | 1.0 | 0.848353 | 0.398159 | 0.000152659 |
| 40 | 24 | 0.6 | 1.0 | 0.848325 | 0.398268 | 0.000157074 |
| 40 | 12 | 0.3 | 1.5 | 0.848365 | 0.398160 | 0.000159225 |
| 80 | 24 | 0.3 | 1.0 | 0.987976 | 0.110431 | 0.000228337 |

Both temporal flags pass in these runs. Doubling PML thickness changes complex
t by 0.000110714; increasing the padding scale by 50% changes it by 0.00000691036.
Those effects are much smaller than the roughly 0.4 continuum amplitude error.
The final row doubles resolution while preserving the baseline physical PML
width of 0.3. Its continuum error remains about 0.11, confirming that the earlier
refinement benefit was not caused by halving physical PML thickness. The absolute
R+T-1 residual ranges from 0.000253 to 0.000316 across these four controls.
Neither control monotonically decreases all small residual errors; these are
sensitivity measurements, not a proof of a universal boundary-error bound.

## Independent discrete-frequency reference

To separate mesh error from time truncation and boundaries, the study also
solves a one-dimensional harmonic finite-difference problem. It uses no time
loop, PML, numerical current-sheet calibration, RCWA solver or Wavebit material
sampler. At normal incidence with mu=1 and spacing h, eliminating H from the
Yee equations gives

$$
E_{j-1}+E_{j+1}+(h^2\Omega^2\epsilon_j-2)E_j=0,
\qquad \Omega=\frac{2\sin(\omega\Delta t/2)}{\Delta t}.
$$

The tangential permittivity is averaged independently over each E-node control
volume by exact layer overlap. Vacuum lead phase satisfies
$\vartheta=2\arcsin(h\Omega/2)$. Eliminating the two exterior nodes with exact
outgoing factors $e^{i\vartheta}$ produces a tridiagonal boundary problem,
including a known unit incident wave on the left. Its r/t use z=0 and the last
interface as reference planes, just like the FDTD result. The lossless reference
conserves R+T to numerical precision.

At resolution 40 the discrete reference has t approximately
-0.844761918 - 0.366679542i, very close to the FDTD values above, despite its
large difference from continuum t=-1. A new regression test checks a 400-period
FDTD run against this independent boundary solve at absolute r/t tolerance 1e-3.
That validates the finite-grid calculation at this tolerance; it does not turn
the finite-grid answer into an accurate continuum result.

A bounded search of the discrete reference transmission maximum over frequencies
0.99 to 1.01 finds:

| Resolution | Discrete peak frequency | Shift from continuum f=1 | Peak T |
| --- | --- | --- | --- |
| 40 | 0.9985061725 | -0.149383% | approximately 1 |
| 80 | 0.9996284987 | -0.0371501% | approximately 1 |

The frequency shift reduces by about four times on doubling resolution. These
are discrete boundary-problem peaks, not FDTD frequency-sweep measurements. They
identify spatial resonance detuning as the dominant effect in this example.
The two resolutions alone do not establish a general convergence order.

## Scope and remaining controls

These measurements do not motivate a solver correction: the time-domain result
matches the independent finite-grid boundary problem, while the remaining large
continuum error decreases with spatial refinement. The tests protect this
distinction from future optimizations. The PML/padding controls apply to this
planar dielectric cavity only; different geometries, stronger confinement and
dispersive material states require their own controls.

Meep's public [resonant-mode documentation](https://meep.readthedocs.io/en/latest/Python_Tutorials/Resonant_Modes_and_Transmission_in_a_Waveguide_Cavity/)
discusses the long lifetime of resonances and direct excitation for lifetime
analysis; its [FAQ](https://meep.readthedocs.io/en/latest/FAQ/) identifies insufficient
runtime, overlapping source activity and inappropriate monitor placement as
possible problems. Only documentation was consulted; no Meep implementation was
opened. This study uses its own analytic reference and does not use harmonic
inversion or import an external solver implementation.

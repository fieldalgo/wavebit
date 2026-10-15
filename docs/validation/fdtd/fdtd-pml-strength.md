# FDTD CPML strength near a Rayleigh threshold

Written: 2026-09-09

Targets:
- `src/wavebit/fdtd/_periodic/legacy.py`
- `src/wavebit/fdtd/_periodic/grid.py`
- `src/wavebit/fdtd/_periodic/scene.py`
- `tests/fdtd/test_fdtd_pml_strength.py`
- `studies/fdtd/pml_strength/fdtd_pml_strength.py`
- `studies/fdtd/rayleigh/fdtd_rayleigh.py`

TL;DR: the shared grid and both scene paths expose static CPML strength
independently of thickness. Defaults are unchanged. A stronger, resolved
boundary reduces returning near-Rayleigh waves, but boundary changes require
fresh temporal checks and do not remove the discrete cutoff shift.

## Meaning of the control

The [guide](../../guide/fdtd/time-domain.md#cpml-strength) specifies `pml_reflection`.
For thickness L and exterior index n=sqrt(epsilon*mu), the existing profile is

```text
sigma(z) = sigma_max * (depth/L)^m
sigma_max = -(m+1)*log(pml_reflection)/(2*n*L).
```

Only the previously fixed logarithm becomes configurable. In the continuous,
unshifted normal-incidence model, `exp(-2*n*integral(sigma dz))` equals the
selected value. This describes amplitude after a round trip; it is not measured
power or an accuracy tolerance. The frequency shift, kappa and polynomial
order retain their previous settings. Both uniform and nonuniform z profiles
use their own physical thickness and exterior epsilon*mu.

At fixed target, increasing L reduces sigma_max proportionally, leaving the
nominal integral unchanged. Width and strength are therefore different controls.
For a glancing propagating wave, the attenuation exponent also decreases with
the normal wavevector. Johnson's [PML notes, sections 7.1–7.2](https://arxiv.org/html/2108.05348v1#S7)
explain this angular dependence and the reflection introduced by discretization.
The [public Meep PML guide](https://meep.readthedocs.io/en/latest/Perfectly_Matched_Layer/)
also recommends gradual profiles and independent thickness convergence, with
clearance from evanescent fields. Only documentation and theory were consulted;
no external implementation was opened or copied for this change.

## Reproducible comparison

Run from the checkout with JAX installed:

```shell
python -m studies.fdtd.pml_strength.fdtd_pml_strength --case all
```

`--backend numpy` selects the same shared engine without compilation. The study
uses double precision on CPU. It retains the [directional study](fdtd-directional.md)
geometry: binary epsilon 2.25/1, thickness 0.2, period 0.8, vacuum leads,
central wavelength 0.96, fixed kx corresponding to sin(theta)=0.2, and wavelengths
0.95808 and 0.96192. Layers are aligned, Courant is 0.8, padding scale is 1,
and both diffraction planes are 0.2 from their reference interface.

The Gaussian half-width is 60 in c=1 time units. This suppresses excitation of
the earlier slowly decaying TM oscillation. Except for the explicit
`long-window` control below, each solve checks its final 200 steps; the comparison
interval is 2.30940 at resolution 40 and 1.15470 at resolution 80. Runtime is
specified by `n_periods` at the central wavelength.
Complex r/t errors are maxima over both wavelengths and orders -1 and 0 against
independent RCWA with half-order 32. Strongly evanescent order +1 is excluded
from that error metric but retained in the time-settling diagnostics. RCWA
32→64 changes the TE/TM reference by 1.72e-6/1.28e-5 in this metric.

## Measured strength and width controls

The baseline is the previously measured narrow-source case, reproduced by the
`baseline` selector. All other rows use the new public strength option. Errors
are absolute complex-amplitude differences, not relative percentages.

| Case | Resolution | PML length | Target | Periods | TE error | TM error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 40 | 0.3 | 1e-8 | 800 | 0.126447 | 0.059807 |
| Width only | 40 | 0.6 | 1e-8 | 800 | 0.131749 | 0.053833 |
| Strength only | 40 | 0.3 | 1e-40 | 800 | 0.023803 | 0.047653 |
| Combined | 40 | 0.6 | 1e-40 | 800 | 0.025048 | 0.048183 |
| Combined, longer | 40 | 0.6 | 1e-40 | 1600 | 0.022332 | 0.051702 |
| Combined, finer | 80 | 0.6 | 1e-40 | 800 | 0.015474 | 0.010338 |

The incoming columns below are maxima over the two wavelengths of backward
transmission-side amplitudes, referred to the interface. They use calibrated
unit-TE/TM illumination. They are not reflection coefficients of an isolated
PML, power fractions, or guaranteed steady-state values.

| Case | TE incoming, -1 | TM incoming, -1 | TE incoming, 0 | TE final-window change | TM final-window change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.090898 | 0.040337 | 6.42e-5 | 6.56e-6 | 6.12e-8 |
| Width only | 0.130796 | 0.050805 | 1.48e-8 | 2.20e-5 | 2.32e-7 |
| Strength only | 0.010476 | 0.003312 | 5.20e-3 | 2.78e-4 | 6.25e-5 |
| Combined | 0.009470 | 0.003621 | 1.23e-6 | 2.32e-4 | 7.36e-5 |
| Combined, longer | 0.006345 | 0.001236 | 1.23e-6 | 8.96e-5 | 1.58e-5 |
| Combined, finer | 0.007261 | 0.004740 | 2.12e-8 | 1.47e-4 | 3.76e-5 |

Width alone does not reduce the glancing TE return. Strength alone suppresses
that return but increases the ordinary order-0 return; combining strength with
enough cells largely removes that additional order-0 error. This is why the
stronger target is an optional control rather than a replacement default.

Boundary changes also alter ringdown. The strength-only and combined 800-period
runs fail the calibrated scattering-settling check at tolerance 1e-4. The
1600-period combined run passes that local-window check, but its actual TE/TM
field-decay ratios are still 0.00408/0.000888, so `state_decayed` is false. The
maximum TE/TM errors also change by about 0.00272/0.00352 between 800 and 1600
periods; a small final-window contribution alone is not a global time-error
bound. These changes are differences of the reported maxima, not a norm of the
full scattering-matrix change. All six rows have `state_decayed=False`.

The `long-window` control repeats the 1600-period solve with
`convergence_window=66511`, comparing its entire second half (768.00288 time
units) against the 800-period prefix. Final r/t errors and incoming amplitudes
are unchanged. The maximum calibrated TE/TM amplitude differences are now
**0.0067693/0.0038988**, both dominated by order -1, and `scattering_settled` is
false. The longer window exposes the time error that the 200-step check misses.
This measures a finite-duration difference; even this longer window is not a
certificate for infinite-time convergence.

Refinement to 80 reduces the combined-boundary TE/TM errors at 800 periods to
0.01547/0.01034. At the same resolution and duration, the earlier length-0.3,
target-1e-8 case gave 0.14423/0.04600. This is a substantial improvement in the
measured finite-duration result, but not yet a converged accuracy claim: the
fine combined-boundary run has actual TE/TM field-decay ratios 0.01334/0.00382
and fails the short-window scattering check. Boundary errors, time errors and
the remaining Yee cutoff shift cannot be combined into a claimed spatial order
from these rows. Strength selection does not change the reported cutoff data.

## Implementation checks

The dedicated tests check unchanged default fields, explicit-uniform versus
uniform profile agreement, and removal of a pulse from a homogeneous magnetic
medium against periodic and deliberately weak-boundary controls. They cover
nonuniform z widths, NumPy/JAX/Torch values and epsilon/mu/pole-strength
derivatives against finite differences, and both incident sides forwarding the
same static value to all four scene runs. Malformed settings are refused before
the scene time loop. These checks establish the parameter plumbing and bounded
physical behavior, not arbitrary-angle accuracy.

An additional exact-mode check covers a limitation directly: on the native
periodic grid, a plane wave independent of z has no z derivative for CPML to
stretch. Both transverse polarizations retain their exact discrete Maxwell
oscillation with targets 1e-8 and 1e-40. This globally supported parallel mode
is not an outgoing localized pulse. Increasing z-absorption strength cannot
make every possible initial field decay; nor does this test identify the
specific modes responsible for the grating's late-time tail.

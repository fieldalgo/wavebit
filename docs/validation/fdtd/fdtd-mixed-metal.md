# Low-loss mixed Drude/Lorentz metal

Written: 2026-09-08

Targets:
- `studies/fdtd/mixed_metal/fdtd_mixed_metal.py`
- `tests/references/interface/validate_grid_dispersive.py`
- `tests/fdtd/test_fdtd_mixed_metal.py`
- `tests/references/test_native_material_external.py`
- `tests/references/golden_low_loss_metal_meep.json`

TL;DR: a low-loss mixed metal can have settled measured amplitudes while material
oscillations remain. After settling, spatial refinement improves complex r/t
and damping sensitivities. A separate live Meep comparison agrees when both
solvers use the same documented dispersive-interface sampling.

## Model and measurement conditions

This is a synthetic passive metal model, not a fit to a named experimental metal.
In angular-frequency c=1 units, epsilon-infinity is 1.5 and the two
(omega0, gamma, strength) poles are (0, 0.08, 160) and (10, 0.12, 40):

```text
epsilon(omega) = 1.5 + 160/(-omega^2 - i*0.08*omega)
                    + 40/(100 - omega^2 - i*0.12*omega).
```

At wavelengths 0.9, 1.0 and 1.1, epsilon is respectively about
-1.00227+0.05036i, -1.89137+0.05983i and -2.80934+0.07471i. The band stays
away from both epsilon=0 and the Lorentz resonance.

Run `PYTHONPATH=. python studies/fdtd/mixed_metal/fdtd_mixed_metal.py --controls --gradients`
from the checkout. The scene is a thickness-0.2 uniform slab in vacuum at normal
TE incidence, period 0.05, Courant factor 0.8, uniform grids, and physical PML
width 0.3. The diagnostic window spans approximately two optical periods.
The measurements below used NumPy 2.5.1 and complex128 on arm64 CPU.

The independent continuum reference is the passive Airy formula with the
frequency-dependent epsilon above. Errors are maximum absolute complex errors
across the three wavelengths. Field decay is the maximum over both actual and
reference runs for both source bases; each material column takes the maximum
over the two actual runs for that pole.

| Resolution | Periods | Max r error | Max t error | Field decay | Drude decay | Lorentz decay | Max amplitude change | State decayed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40 | 40 | 0.00804937 | 0.00352570 | 4.15833e-4 | 1.40427e-4 | 4.39216e-3 | 5.92930e-5 | False |
| 40 | 160 | 0.00803842 | 0.00350735 | 1.43790e-7 | 2.41179e-8 | 1.66248e-6 | 1.64641e-8 | True |
| 40 | 320 | 0.00803843 | 0.00350734 | 2.33278e-9 | 1.29882e-10 | 8.44463e-11 | 3.46808e-10 | True |
| 80 | 320 | 0.00201935 | 0.000880272 | 6.03058e-12 | 8.73374e-13 | 8.34391e-11 | 8.46610e-13 | True |

All four runs pass `amplitudes_settled` at tolerance 1e-4. The 40-period run
still fails field and material decay. These flags answer different questions:
a small change in this spectral band does not certify that the state has decayed.
Extending 160 periods to 320 scarcely changes r/t, while doubling resolution
reduces their errors by about four times. The last two rows still have finite
spatial errors despite very small time tails.

## Boundary and time-step controls

Each control changes one setting at resolution 40 and 320 periods. The amplitude
shift is the maximum absolute r/t difference from the default case above.

| Change | Max r error against Airy | Max t error against Airy | Max amplitude shift |
| --- | --- | --- | --- |
| Courant 0.8 to 0.4 | 0.00782662 | 0.00321343 | 0.000614105 |
| PML width 0.3 to 0.6 | 0.00803679 | 0.00352093 | 0.000116649 |
| Padding scale 1 to 1.5 | 0.00801897 | 0.00348978 | 0.0000883239 |

Halving dt at fixed spatial resolution has a smaller effect than refining the
grid. PML and padding effects are smaller again, but they are not zero. Together
with the refinement result, these controls identify spatial discretization as
the dominant r/t error here; they do not establish a universal PML tolerance.

The regression also solves an independent discrete outgoing-boundary problem.
Substituting exp(-i omega n dt) into the centered oscillator recurrence gives

```text
omega_hat^2 = (2*sin(omega*dt/2)/dt)^2
omega_damp = sin(omega*dt)/dt
epsilon_discrete = epsilon_inf
                 + sum(strength/(omega0^2 - omega_hat^2 - i*gamma*omega_damp)).
```

This material response enters a one-dimensional Yee Helmholtz matrix. Tangential
interface fractions are integrated independently and outgoing vacuum leads are
eliminated exactly. The reference uses no time loop, PML, Wavebit sampler or
incident calibration. At resolutions 40 and 80 and 160 periods, complex r/t must
agree with it within 3e-4, while the finer grid must also approach continuum Airy.
This distinguishes errors shared with the discretization from additional
time-truncation or boundary errors.

## Material sensitivities

The backend test differentiates the real part of the sum of r_s+0.4i t_s over
the band with respect to six multiplicative factors: epsilon-infinity, Drude
gamma/strength, and Lorentz omega0/gamma/strength. At resolution 10 and 30 periods,
JAX JIT/autodiff and Torch autograd agree with NumPy central differences to the
test bound of 3e-5 relative plus 2e-8 absolute. The finite differences use factor
steps 1e-4 and 5e-5, which must also agree within that bound. Complex amplitudes
agree to 1e-10.
The objective includes phase so a power-only agreement cannot hide phase errors.
This establishes differentiation of the fixed numerical solve, not continuum
gradient accuracy on that coarse grid.

The study separately perturbs each damping by a relative factor of 1 +/- 0.001
and differentiates mean absorptance 1-|r|^2-|t|^2. Both the scene and Airy use
that same central difference; physical geometry and time step remain fixed.
The run duration is 320 periods and all perturbed runs pass amplitude settling.

| Damping factor | Airy derivative | Resolution 40 derivative | Relative error | Resolution 80 derivative | Relative error |
| --- | --- | --- | --- | --- | --- |
| Drude gamma | 0.0260056644 | 0.0262284976 | 0.857% | 0.0260622422 | 0.218% |
| Lorentz gamma | 0.00547267126 | 0.00550826754 | 0.650% | 0.00548171265 | 0.165% |

Both sensitivity errors decrease by about four times with refinement. For this
model, an accurate derivative of the coarse numerical solve still differs
measurably from the continuum derivative. Backend agreement and physical
sensitivity convergence are separate checks.

## Independent Meep execution

The external worker runs the same pole model in a separate thickness-0.5 slab,
with a length-12 vacuum-ended domain, PML width 1, source at 3, slab at 7..7.5,
and monitor at 9. Wavelengths are 0.9/1/1.1, cdt/dx=0.4, duration 160. A
period-1 sine carrier has a width-1 Gaussian envelope centered at time 5.
Complex Ex/Hy and transmitted power are divided by an empty-domain reference.

Meep 1.34.0 was executed through its public API. Its
[material documentation](https://meep.readthedocs.io/en/latest/Materials/#material-dispersion)
specifies the oscillator convention and unsmoothed dispersive contribution.
Wavebit's comparison worker therefore uses point-sampled pole strengths while
both average epsilon-infinity. The scene measurements above instead use
averaged tangential strengths; the two discretizations must not be conflated.
No Meep implementation source was inspected or copied.

| Resolution | Max complex Ex/Hy difference | Absolute power difference | Meep complex Ex error against Airy | Wavebit averaged-strength Ex error against Airy |
| --- | --- | --- | --- | --- |
| 40 | 8.04114e-8 | 2.65086e-9 | 0.0156910 | 0.000744694 |
| 80 | 1.19986e-7 | 1.71811e-9 | 0.00839886 | 0.000185961 |

The last column repeats the same native-grid worker with averaged tangential
strengths. Its error falls by about four times, whereas the point-sampled
interface error falls much more slowly. Close agreement between solvers does
not remove that shared interface error. The new external outputs, solver version, model inputs
and worker digest are frozen separately in `golden_low_loss_metal_meep.json`;
the earlier reference archive is unchanged. Tests consume the frozen data
without importing Meep. Existing frozen FDTDX update/gradient references are
also checked; no new FDTDX execution was used for this measurement.

## Limits

This validates one low-loss, uniform, normally illuminated mixed metal family.
It does not validate patterned or oblique metal interfaces, epsilon-near-zero
operation, arbitrary overlapping resonances, dispersive PML, or GPU execution.
The temporal indicators and material gradients keep their documented scope.

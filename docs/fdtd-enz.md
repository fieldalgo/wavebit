# Lossy ENZ films and calibrated scattering convergence

Written: 2026-09-08

Targets:
- `src/wavebit/_core/scene_fdtd.py`
- `studies/fdtd_enz.py`
- `tests/test_fdtd_enz.py`
- `tests/test_fdtd_scattering_convergence.py`

TL;DR: a lossy epsilon-near-zero film settles much later for TM than for TE.
The scene diagnostics now report each calibrated incident column and a bound
valid for every unit Jones vector. After temporal settling, this film still
needs time-step and grid refinement, especially for its internal normal field.

## Conditions and independent reference

```sh
PYTHONPATH=. python studies/fdtd_enz.py
PYTHONPATH=. python studies/fdtd_enz.py --sensitivity
```

Both modes use JAX 0.11.0 with x64 enabled on arm64 CPU, NumPy 2.5.1.
`--backend numpy` evaluates the same experiment without JAX. The synthetic,
isotropic Drude film has epsilon-infinity 1, gamma 0.05 and oscillator strength
(2*pi)^2 + 0.05^2, in c=1 angular-frequency units. Thus its continuum epsilon has
zero real part at wavelength 1 and a small, nonzero imaginary part. This is not
a fitted material model or the exactly lossless ENZ limit.

| Wavelength | Re(epsilon) | Im(epsilon) |
| --- | --- | --- |
| 0.98 | 0.0395976 | 0.00748979 |
| 1 | 0 | 0.00795775 |
| 1.02 | -0.0403973 | 0.00844480 |

The film is thickness 0.1 in vacuum, period 0.05, with theta=0.35 radians at
center wavelength 1 and fixed transverse k throughout the band. The supplied
illumination is TM; both calibrated s and p columns are also returned. Unless
specified otherwise: 120 optical periods, Courant fraction 0.8, physical PML
width 0.3, padding scale 1, aligned layers, Gaussian width 3. Diagnostic windows
span about two time units, rounded to an integer step count. Tolerance is 1e-4.

The reference uses the independent E/H characteristic matrices from the
[layer-fitting study](fdtd-layer-alignment.md). The internal normal electric
field is derived by propagating the transmitted tangential E/H back to the
middle of the film and applying Ez=-(kx/omega)*Hy/epsilon. The frequency-plane
comparison uses the first transverse sample, whose source phase is zero. The
reference magnitude of Ez at the center wavelength is 7.22230 for unit incident
electric field. Bottom incidence reverses the Cartesian Ez sign.

ENZ field enhancement under oblique TM illumination is discussed in the primary
experimental and transfer-matrix study by
[Passler et al., ACS Photonics (2019)](https://doi.org/10.1021/acsphotonics.9b00290).
That work concerns ultrathin polar crystal films; it motivates checking the
internal field here, but its numerical parameters and experimental data are
not reference values for this synthetic Drude film.

## The two incident columns do not settle together

At resolution 40 and duration 60, the maximum full-minus-prefix amplitude
changes are 1.81e-14 for the s column and 9.28e-4 for the p column. The s flag
passes, while the p flag, `scattering_settled` and `state_decayed` fail. At
duration 120, the p-column change falls to 2.59e-7 and all these flags pass.

The [guide](guide/time-domain.md#pulse-decay-diagnostics-in-the-common-engine)
defines the new `s_inc`, `p_inc`, `scattering_change_bound` and
`scattering_settled` diagnostics. The bound uses the Euclidean norm of each
output row's two calibrated changes. By Cauchy-Schwarz it bounds every normalized
Jones input, including constructive interference between the columns. Tests
construct an input that attains the bound and compare it with a separately
truncated solve. The existing given-polarization diagnostics keep their meaning.

## Spatial and temporal discretization

Errors below are maximum absolute complex r/t errors across the band, separately
for TE and TM. Ez error is the maximum absolute complex middle-plane error.

| Resolution | Duration | Courant fraction | TE error | TM error | Ez error |
| --- | --- | --- | --- | --- | --- |
| 40 | 60 | 0.8 | 0.00118121 | 0.00587660 | 0.0498400 |
| 40 | 120 | 0.8 | 0.00118121 | 0.00779282 | 0.0710824 |
| 40 | 240 | 0.8 | 0.00118121 | 0.00779249 | 0.0710792 |
| 80 | 120 | 0.8 | 0.000291200 | 0.00195467 | 0.0177866 |
| 160 | 120 | 0.8 | 0.0000728258 | 0.000488913 | 0.00444941 |
| 40 | 120 | 0.4 | 0.00102312 | 0.00228428 | 0.0180259 |
| 80 | 120 | 0.4 | 0.000254464 | 0.000571827 | 0.00450610 |

The early TM answer is accidentally closer to the continuum answer than the
temporally settled coarse-grid answer. A small error at one unfinished time is
not convergence evidence. Once settled, both polarizations and Ez refine by
about four times as resolution doubles. At fixed resolution 80, halving the time
step reduces Ez error by almost four times, while TE changes much less. The
NumPy accuracy regression uses Courant fraction 0.4 to meet its absolute 0.01
internal-field target at resolution 80; running longer at fraction 0.8 cannot
meet that target on the same grid.

Harmonic substitution into the existing centered ADE recurrence gives

```text
epsilon_discrete(omega) = 1 - strength / (omega_hat^2 + i*gamma*omega_damp)
omega_hat = 2*sin(omega*dt/2)/dt
omega_damp = sin(omega*dt)/dt
```

At the continuum zero crossing, the discrete real epsilon is -0.000438820 at
resolution 40, -0.000109683 at 80, and -0.0000274195 at 160, with Courant fraction
0.8. At resolution 80 and fraction 0.4 it is also -0.0000274195. This is a
time-discretization shift of the material response, separate from Yee spatial
dispersion and plane interpolation. The observed TM sensitivity is consistent
with that shift near ENZ. No pole prewarping or material-equation change was
introduced; material parameters retain their continuous-time meaning.

## Boundary controls, bottom incidence and sensitivity

At aligned resolution 80 with fraction 0.8, doubling physical PML width changes
TE/TM complex amplitudes by at most 8.19e-8/1.88e-7. Increasing padding scale to
1.5 changes them by 2.92e-8/2.21e-7. These are much smaller than the 0.001469 TM
change when halving dt. Bottom illumination gives the same scattering errors in
this symmetric film and the expected reversed Ez sign; its Ez error is 0.0177866.
All these controls pass the temporal flags. Equal top/bottom powers alone would
not test the Cartesian field sign.

The sensitivity experiment differentiates the band mean of
Re(r_p + 0.4i*t_p) with respect to a multiplicative Drude-strength factor at 1.
It uses central differences of 1e-4 and duration 240; it does not retune strength
when gamma or epsilon-infinity changes.

| Resolution | Strength-factor derivative | Continuum derivative | Absolute error |
| --- | --- | --- | --- |
| 40 | -4.63756428 | -4.69166314 | 0.0540989 |
| 80 | -4.67803796 | -4.69166314 | 0.0136252 |

The derivative error decreases by about four times under refinement. Separate
JAX/Torch tests differentiate epsilon-infinity, gamma and strength factors on a
fixed coarse mesh against NumPy central differences at two step sizes. Their
objective combines complex TM scattering, middle-plane Ez intensity, p-column
tail and the new Jones-input bound, including bottom illumination. Those checks
verify derivatives of a finite-duration calculation, not continuum accuracy at
that coarse mesh. The zero row norm uses a finite zero subgradient, tested on
both differentiating backends.

## Scope

These are CPU results for a finite-loss, planar, local-response Drude film and
one narrow wavelength band. They do not certify the exactly lossless ENZ limit,
nonlocal response, arbitrary grazing orders, or patterned metal corners. A
settled scattering operator remains a temporal comparison, not a continuum
accuracy estimate or an automatic stopping rule. Its bounds and flags cover
the retained outgoing orders and requested wavelengths only.

Validation passes 107 distinct targeted checks: 9 new scattering/ENZ checks,
60 existing convergence, DFT-sampling, material-decay and pulse-width checks,
and 38 documentation-contract checks. Ruff and a Sphinx build with warnings
treated as errors pass. This count is not the full repository test suite.

No external implementation was copied. Only public
[Meep material documentation](https://meep.readthedocs.io/en/latest/Materials/)
and primary research literature were consulted; no Meep implementation was read,
and no fresh Meep/FDTDX or GPU execution is part of this measurement.

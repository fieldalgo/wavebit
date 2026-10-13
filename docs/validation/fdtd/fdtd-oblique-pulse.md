# Oblique pulse bandwidth and metal interfaces

Written: 2026-09-08

Targets:
- `src/wavebit/fdtd/_periodic/scene.py`
- `studies/fdtd/oblique_pulse/fdtd_oblique_pulse.py`
- `tests/fdtd/test_fdtd_pulse_width.py`

TL;DR: narrowing the source spectrum removes a large temporal tail in an oblique
metal slab. It does not remove dispersive-interface or corner errors. The new
`pulse_width` control keeps these questions independently adjustable.

## Conditions and reproduction

The synthetic mixed Drude/Lorentz model is the same as the
[normal-incidence measurement](fdtd-mixed-metal.md): epsilon-infinity 1.5,
poles (omega0, gamma, strength) = (0, 0.08, 160), (10, 0.12, 40), in c=1 angular
units. At wavelength 1, epsilon is approximately -1.89137+0.05983i.
All runs use vacuum half spaces, theta=0.45 radians at the center wavelength,
Courant fraction 0.8, two current bases with actual/reference runs, and complex128 fields.
The measured backend is JAX 0.11.0 with x64 enabled on arm64 CPU, NumPy 2.5.1.

```sh
PYTHONPATH=. python studies/fdtd/oblique_pulse/fdtd_oblique_pulse.py
PYTHONPATH=. python studies/fdtd/oblique_pulse/fdtd_oblique_pulse.py --pattern
```

`--backend numpy` runs the same study without JAX. Results include the realized
grid, dt, physical upper PML width, both bases' field/material tails, and the
given TE illumination's amplitude change. Diagnostic windows are 200 updates;
their physical duration therefore changes when resolution changes. Pulse width
is an envelope half-width in time, not a frequency FWHM. The
[guide](../../guide/fdtd/time-domain.md#gaussian-source-width) defines the complete API.

## Planar oblique slab

The slab is thickness 0.2, period 0.05, wavelengths 0.9/1/1.1, and 160 optical
periods. Uniform grids keep physical PML width at 0.3. The independent reference
is oblique Fresnel/Airy, evaluated with one fixed transverse wavevector across
the band. Reported errors are maximum absolute complex r/t errors over the band
for each incident polarization, not power errors.

| Resolution | Pulse width | TE error | TM error | TE amplitude change | State decayed |
| --- | --- | --- | --- | --- | --- |
| 40 | 1 | 0.00931083 | 0.00857990 | 0.00272031 | False |
| 80 | 1 | 0.00388614 | 0.00207219 | 0.00402276 | False |
| 40 | 3 | 0.00753039 | 0.00867680 | 1.39814e-13 | True |
| 80 | 3 | 0.00189230 | 0.00217176 | 1.56178e-13 | True |

Width 1 fails both temporal flags; width 3 passes. With the narrower spectrum,
TE and TM errors decrease by approximately four times under refinement.
The small broad-pulse TM error at resolution 80 is not convergence evidence:
residual contributions can cancel other numerical errors at a particular time.

For fixed transverse k, the vacuum grazing frequency is sin(0.45)=0.43497,
while the carrier frequency is 1. The ideal Gaussian's relative amplitude there
is about 0.0428 at width 1 and 4.83e-13 at width 3. This explains why source
bandwidth can change long tails even though every requested wavelength is
propagating. The requested band still receives at least about 0.334 of the
nominal carrier amplitude at width 3. A fixed-grid, 80-period NumPy regression
also checks the temporal distinction and Fresnel/Airy refinement for both TE/TM.

Meep's public [oblique-grating documentation](https://meep.readthedocs.io/en/latest/Python_Tutorials/Mode_Decomposition/#reflectance-and-transmittance-spectra-for-planewave-at-oblique-incidence)
likewise discusses broad pulses exciting glancing waves that absorbers handle
poorly, and narrowing the spectrum to reduce them. This is a documented physical
mechanism used as context, not an external solver comparison for these runs.

## A boundary that cuts a normal-component cell

Keep width 3 and duration 160, but change the slab thickness to 0.213. Compare
the uniform grid with existing `grading=True`, whose one-layer core places both
outer interfaces exactly on z nodes.

| Resolution | Mesh | TE error | TM error | Physical upper PML width |
| --- | --- | --- | --- | --- |
| 40 | Uniform | 0.00817326 | 0.0595084 | 0.3 |
| 80 | Uniform | 0.00193001 | 0.00218669 | 0.3 |
| 40 | Graded | 0.00720025 | 0.00758452 | 0.543642 |
| 80 | Graded | 0.00248077 | 0.00232598 | 0.572690 |

All four pass the field/material and TE amplitude criteria. The coarse uniform
TM error nevertheless approaches 0.06. At this resolution the exit boundary
cuts an Ez control volume; the code harmonically averages epsilon-infinity but
arithmetically weights oscillator strengths. That is not the harmonic average
of the full frequency-dependent response. At resolution 80 the same boundary
is almost aligned, so the unusually large improvement is not evidence for an
asymptotic convergence order.

Grading reduces this particular discrepancy, but also changes dt, the padding
mesh and the physical PML thickness. It is therefore supporting evidence for
interface sensitivity, not an isolated proof that boundary placement accounts
for every difference. No general dispersive homogenization or new interface
treatment was implemented here. Meep's public
[subpixel documentation](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/#what-about-dispersive-materials)
also distinguishes instantaneous-material smoothing from a frequency-dependent
effective interface. The limitations of Wavebit's normal-component pole averaging
remain part of its support boundary.

## A patterned low-loss metal

The second study uses a thickness-0.2, period-0.8 grating: the first half of the
cell is metal and the second half vacuum. Wavelength is 1 and theta remains 0.45.
Orders -1 and 0 propagate; their complex r_s/t_s are compared with RCWA. The
first evanescent order is retained only in the reference truncation check.

| RCWA orders | TE amplitude change | TM amplitude change |
| --- | --- | --- |
| 12 to 24 | 3.69146e-5 | 0.274394 |
| 48 to 96 | 6.55609e-7 | 0.237911 |

These maxima already occur in the propagating orders.

The TE reference is stable enough for the measured errors below. The TM reference
is not: its large changes prohibit treating the FDTD-to-RCWA difference as an
accuracy error. Sharp metal corners remain an unresolved validation case. The
study reports that discrepancy explicitly rather than certifying TM agreement.

| Resolution | Periods | Width | RCWA order | TE error | TE amplitude change |
| --- | --- | --- | --- | --- | --- |
| 20 | 80 | 1 | 24 | 0.0355705 | 0.00934314 |
| 20 | 80 | 3 | 24 | 0.0352497 | 0.0000761710 |
| 20 | 160 | 3 | 24 | 0.0351916 | 0.0000273766 |
| 40 | 160 | 3 | 24 | 0.00411072 | 0.0000720057 |
| 20 | 160 | 6 | 96 | 0.0351904 | 3.03444e-9 |
| 40 | 160 | 6 | 96 | 0.00414579 | 1.01461e-8 |
| 80 | 160 | 6 | 96 | 0.00103581 | 7.17528e-9 |

The grating's -1 order has its own grazing frequency, about 0.81503, closer to
the carrier than the incident order's cutoff. Increasing width from 3 to 6
reduces the TE temporal tail further. At width 6, spatial refinement 40 to 80
reduces the TE error by about four times; making the temporal tail small does
not remove spatial error. Global `state_decayed` stays false for every row:
it checks both source bases, while `amplitudes_settled` and `amplitude_change`
describe the supplied TE illumination. Inspect the basis-specific tails before
using `s_inc` or `p_inc` as separately converged results.

## Verification and remaining scope

Tests preserve the default pulse exactly, check physical scaling and source
spectrum metadata, refuse unresolved/unfinished source settings, and verify
JAX/Torch material gradients with a nondefault width combined with bottom
incidence, grading and DFT subsampling. Pulse width itself remains static.
Existing amplitude, convergence, DFT and backend tests are also exercised.

Only public Meep documentation was read for this change; no Meep implementation
or new Meep/FDTDX execution was used. Existing stored Meep scene evidence remains
part of the regression suite. No external implementation was copied. These are
CPU results for selected planar and TE grating cases. General low-loss TM corner
accuracy, epsilon-near-zero operation and actual GPU behavior remain unvalidated.

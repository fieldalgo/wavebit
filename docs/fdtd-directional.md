# Directional FDTD extraction and persistent near-Rayleigh modes

Written: 2026-09-08

Targets:
- `src/wavebit/_core/scene_fdtd.py`
- `tests/test_fdtd_directional.py`
- `studies/fdtd_directional.py`
- `studies/fdtd_rayleigh.py`

TL;DR: separating forward and backward E/H fields prevents a returning wave
from being counted directly as outgoing diffraction. Exact discrete-mode tests
verify the separation. It does not remove the effect of boundaries on the
structure or make an unfinished DFT stationary. A nearby, slowly decaying TM
oscillation explains why merely extending the earlier near-Rayleigh run was
ineffective.

## Discrete Maxwell decomposition

The [guide](guide/time-domain.md#directional-diffraction-extraction) defines
`diffraction_method='directional'` and its incoming-amplitude diagnostics. The
default remains electric-only extraction. No native Maxwell, ADE or CPML update
was changed, and all four actual/reference runs are retained.

For a uniform, lossless exterior with real positive epsilon and mu, define

```text
Omega = 2*sin(omega*dt/2)/dt
Kx = 2*sin(kx*dx/2)/dx
Ky = 2*sin(ky*dy/2)/dy
Kz^2 = epsilon*mu*Omega^2 - Kx^2 - Ky^2.
```

The root for Kz uses Wavebit's shared outgoing/decaying branch convention. For
the +z wave, the discrete Maxwell equations give H=(K cross E)/(mu*Omega) and
K dot E=0. With transverse vector G=(Hy,-Hx), their tangential relation is

```text
G = Y E
Y = [[Kz^2+Kx^2, Kx*Ky], [Kx*Ky, Kz^2+Ky^2]] / (mu*Omega*Kz).
```

The -z wave reverses the sign of this relation. Thus
E_plus=(E+inverse(Y)G)/2 and E_minus=(E-inverse(Y)G)/2. No complex conjugation
is used: this linear decomposition also separates the locally decaying and
growing evanescent branches.

The native DFT already uses the correct, different E and H timestamps. Its
spatial interpolation multiplies Ex/Ey by cos(kx*dx/2)/cos(ky*dy/2), respectively.
Hx/Hy have the opposite transverse factors and also a longitudinal factor
cos(kz*dz/2), where kz=2*asin(dz*Kz/2)/dz. The extraction removes these factors
before applying the discrete admittance. A continuum impedance or a shared
E/H timestamp would not pass the exact-mode test.

Each measurement plane needs equal adjacent z-cell widths. Graded regions away
from the planes may still scatter; accumulated homogeneous phase through them
is an approximation. Singular discrete cutoff and Nyquist inputs are refused
before time stepping. The extra carrier DFT used for time records only needs the
incident reference order; unrequested carrier diffraction orders may be grazing.

The general idea of separating travelling amplitudes using transverse E/H is
also explained in Meep's public
[mode-decomposition theory](https://meep.readthedocs.io/en/latest/Mode_Decomposition/#theoretical-background).
The matrix above is derived directly from Wavebit's discrete Maxwell equations;
no Meep implementation was inspected or copied.

## Verification versus continuum accuracy

The exact-mode test superposes two counter-propagating discrete Maxwell waves
on a three-axis Bloch grid with unequal spacings, conical wavevector and magnetic
material. It checks native DFTs at several planes, including a wrapped boundary,
against the two prescribed complex amplitudes. NumPy, JAX and Torch agree with
the prescribed answer within 3e-13. A separate local Maxwell construction checks
both evanescent branches without treating one as a conjugate of the other.

Scene tests cover a coupled 3-by-3 diffraction window in a uniform magnetic
medium, including exact zero reflection and unit total transmission. Other tests
move the planes in a conical magnetic slab with appreciable returning waves:
complex scattering, incoming residuals and carrier-calibrated Ex/Hy records
agree within 1e-8, for top and bottom illumination. This is a position-invariance
check; it does not certify that the finite PML models an open domain accurately.

An independent Fresnel/Airy test uses a TM film of thickness 0.213, epsilon 2.25,
theta 0.25, vacuum leads and close planes. Its complex amplitude error falls
below 0.002 at resolution 80 and by more than a factor 2.5 from resolution 40.
Backend tests compare raster epsilon, mu and pole-strength derivatives with
central differences, including incoming waves, temporal bounds and cropped time
records. Explicitly truncated solves verify directional amplitude tails for DFT
strides 1 and 3.

## What the near-Rayleigh example reveals

```sh
PYTHONPATH=. python studies/fdtd_directional.py
PYTHONPATH=. python studies/fdtd_directional.py --case ringdown
PYTHONPATH=. python studies/fdtd_directional.py --case narrow
```

These measurements use JAX 0.11.0/x64, NumPy 2.5.1 and arm64 CPU. The geometry,
frequency band and RCWA-32 reference are those in the
[near-Rayleigh study](fdtd-rayleigh.md): a thickness-0.2 binary epsilon-[2.25,1]
grating of period 0.8 in vacuum, theta=asin(0.2) at center wavelength 0.96,
and wavelengths 0.95808/0.96192 at fixed transverse k. The error metric compares
complex r/t of orders -1 and 0. It excludes the strongly evanescent +1 order.
Unless stated otherwise: resolution 40, aligned layers, Courant fraction 0.8,
Gaussian width 3, physical PML width 0.3 and padding scale 1.

| Plane distance | Duration | Padding scale | PML width | TE amplitude error | TM amplitude error | Maximum p-column tail |
| --- | --- | --- | --- | --- | --- | --- |
| default | 300 | 1 | 0.3 | 0.128739 | 0.101684 | 0.0136749 |
| 0.2 | 300 | 1 | 0.3 | 0.128744 | 0.155841 | 0.0308150 |
| 0.2 | 600 | 1 | 0.3 | 0.126563 | 0.140229 | 0.0288382 |
| 0.2 | 600 | 2 | 0.3 | 0.0661107 | 0.140092 | 0.0312638 |
| 0.2 | 600 | 1 | 0.6 | 0.131533 | 0.139336 | 0.0301766 |

Every row still fails both `state_decayed` and `scattering_settled`. The 200-step
comparison window spans 2.30940 time units at resolution 40. In the close-plane
300-period row, per-order p-tail maxima are respectively 0.0308150, 0.000166368
and 0.000505984 for orders -1, 0 and +1. The dominant tail belongs to an order
included in the RCWA error metric; omitting +1 would not settle the result.

Electric-only extraction gave TE error 0.0353269 for that row. Directional
extraction makes it larger: a returning field was partly cancelling the error
in the total electric field. Directional extraction is therefore not justified
by always improving agreement with a continuum reference on an unfinished grid.
Its independent justification is the exact Maxwell decomposition above.

Incoming -1 transmission amplitudes in the 300-period close-plane row reach
0.0925609 for s input and 0.108119 for p input, in unit-incident-E amplitude.
At 600 periods the s tail is below 1e-6, but its incoming transmission amplitude
remains about 0.090927. For that settled s column, the persistent return is a
boundary effect. The p column remains unfinished: finite-DFT endpoint terms can
also appear in its incoming diagnostic. Incoming amplitudes are not power ratios.

Meep's public [PML documentation](https://meep.readthedocs.io/en/latest/Perfectly_Matched_Layer/)
explains that PML proximity to evanescent modal fields can perturb them and that
boundary width requires convergence checks. Here doubling padding changes TE
more than doubling PML width, while neither control settles TM. These measurements
do not establish a universal PML prescription or a general grazing-angle bound.

## A nearby, slowly decaying oscillation

The point-record experiment samples Ez at x=-0.2, y=0 and z=0.1125, every ten
steps, using the 300-period TM run. It fits one complex step multiplier by least
squares over progressively later portions of the record. Frequency and decay
rate follow from its phase and log magnitude; the residual tests whether one
exponential is an adequate local description.

| Start time | Fitted frequency | Amplitude decay rate | Relative one-step residual |
| --- | --- | --- | --- |
| 60 | 1.01806445 | 0.0002160 | 5.72e-5 |
| 120 | 1.01807365 | 0.0002268 | 2.16e-5 |
| 180 | 1.01807023 | 0.0002348 | 1.06e-5 |

The requested carrier frequency is 1/0.96=1.04166667. This persistent oscillation
is close enough for the previous widths 3 and 10 to excite it, but it lies outside
the narrow requested band. Its late-fit amplitude lifetime is about 4,300 time
units, much longer than the 288-time-unit run. An approximate single-mode Q is
about 14,000; changing fit windows changes the decay estimate, so this is not a
precision Q determination or a unique identification of the mode.

An independent RCWA scan over frequencies 1 to 1.035, followed by 1e-6 local
frequency sampling around its largest reflection, finds a narrow TM reflection
maximum at 1.019186 for half-order 16 and 1.019190 for 32, with reflectance above
0.998. Reproduce it with `--case rcwa-resonance`. This supports the presence of a
nearby long-lived resonance. Its location differs from the finite-grid ringdown;
spatial refinement is still needed before identifying the two as the same pole.

## Avoiding unnecessary excitation

At the fitted frequency 1.01807023, the Gaussian's ideal spectral amplitude
relative to its carrier is 0.951745 for width 3, 0.577220 for width 10, and
2.56e-9 for width 60. Width 60 still excites both requested frequencies at more
than 0.85 of the carrier amplitude. This gives a controlled way to avoid the
nearby oscillation while keeping the desired two-frequency calibration strong.
The longer pulse requires a longer run; `--case narrow` uses 800 optical periods.

At resolution 40, close-plane extraction and the original PML/padding settings,
width 60 reduces the p-column tail to 6.12e-8 and its native field-decay ratio to
1.50e-6. The s-column tail is 6.56e-6, so `scattering_settled=True` for all retained
orders and both input columns. `state_decayed` remains false because the s field
ratio is 1.41e-4, just above the 1e-4 threshold. These are finite-window tests,
not a claim that the entire field has vanished.

The temporally settled complex TE/TM errors are still 0.126447/0.0598065. Incoming
-1 transmission amplitudes remain 0.0908984/0.0403368 for s/p incidence. Narrowing
the source addresses unnecessary excitation; it does not fix the boundary or
discrete cutoff error. Those require separate controls and spatial refinement.

With the same width and duration at resolution 80 (`--case fine`), both
`state_decayed` and `scattering_settled` pass. The p-column tail falls to 5.51e-9;
the s/p field ratios are 4.19e-5/2.54e-7. Its comparison window is 1.15470 time
units because it still spans 200 steps. Nevertheless, TE/TM amplitude errors are
0.144231/0.0459980, and the returning -1 transmission amplitudes are
0.134897/0.0536012. TE error grows as the mesh is refined under this fixed boundary
policy. The narrowed-source experiment has removed the large TM transient, but
does not demonstrate uniform spatial convergence to an open-domain answer.

No new external Meep/FDTDX execution or GPU validation is included. The incoming
wave diagnostics, ringdown fit and continuum-reference errors describe different
limits and should not be substituted for each other.

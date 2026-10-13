# Source spectral zeros and remaining cutoff error

Written: 2026-09-09

Targets:
- `src/wavebit/fdtd/_periodic/scene.py`
- `src/wavebit/_common/conventions.py`
- `src/wavebit/compat/legacy.py`
- `tests/fdtd/test_fdtd_pulse_notches.py`
- `tests/references/test_meep_source_notch.py`
- `tests/references/interface/meep_source_notch.py`
- `tests/references/golden_meep_source_notch.npz`
- `studies/fdtd/pulse_notches/fdtd_pulse_notches.py`
- `studies/fdtd/rayleigh/fdtd_rayleigh.py`

TL;DR: `pulse_notches` places one or two finite-sample spectral zeros in a
scene's Gaussian source. It suppresses an independently known material mode
and preserves reference-calibrated optics and gradients. Near a diffraction
cutoff it reduces some returning components and accumulated TM time error,
but does not reliably reduce both polarizations' total complex-amplitude error.
The default source is unchanged; this is a source control, not a convergence fix.

## Construction and validation

The [guide](../../guide/fdtd/time-domain.md#suppressing-selected-source-frequencies)
defines units, required-frequency guards and sampled-spectrum diagnostics.
A notch at positive frequency fn applies a causal delayed difference with
coefficient `exp(-2*pi*i*fn*dt)`. The complete finite convolution multiplies the
Fourier spectrum by `1-exp(2*pi*i*(f-fn)*dt)`, which vanishes at fn. Each notch
reserves one final sample of the runtime for that convolution. The final source
has unit sampled peak amplitude and is shared by all four basis/reference runs.
The implementation is independent; no external filtering or solver source was
consulted or incorporated.

Tests check the finite convolution theorem, frequency sign, one/two/repeated
zeros, physical scaling, input refusal and source precision. A uniform undamped
Lorentz oscillator has an independently known discrete frequency. A notch there
reduces the final E and polarization norms by more than 1e8 compared with the
unfiltered pulse. Oblique dielectric slab amplitudes refine to independent
TE/TM Airy solutions at second order. Bottom incidence, static grading,
directional extraction, DFT subsampling, coherent time records and derivatives
of epsilon, mu and pole strength agree on NumPy, JAX and Torch.

A new Meep 1.34.0 CPU check uses only the documented
[CustomSource and simulation APIs](https://meep.readthedocs.io/en/latest/Python_User_Interface/#customsource).
The uniform material has epsilon=2, omega0=3, gamma=0 and strength=2, with
Courant=0.4 and resolution 10 (dt=0.04). The Gaussian has width 5, peak time 25,
and carrier equal to the discrete material frequency 0.503628254518. Its finite
support is 0 through 50; the delayed difference extends to 50.04. Each source
has unit sampled peak, and Ey is recorded for ten units after the source ends.

| Public-API Meep control | Measured result |
| --- | ---: |
| Unfiltered maximum late Ey magnitude | 0.222001179 |
| Filtered maximum late Ey magnitude | 3.08e-12 |
| Filtered/unfiltered late record norm | 1.30e-11 |
| Fitted unfiltered frequency | 0.503628254518 |
| Fitted unfiltered amplitude decay | consistent with zero to 1e-9 |

Native E records use the same half-step current samples and integer field times.
The complex values agree with the frozen Meep traces within absolute tolerance
1e-10 on all three CPU backends. This checks uniform ADE and source timing; it
does not validate arbitrary open-system Q or guarantee shorter grating runs.
Ordinary tests read the archived output and do not import Meep.

## Why the interface stops at two zeros

Repeated differences amplify both finite Gaussian endpoints and arithmetic
errors. A source-only control uses the near-cutoff pulse below and measures the
largest FFT amplitude farther than 0.2 from the carrier, divided by the largest
FFT amplitude overall. This is a sampled spectral diagnostic, not a bound at
all continuous frequencies.

| Resolution | One zero | Two equal zeros | Three equal zeros, unsupported control |
| --- | ---: | ---: | ---: |
| 40 | 4.93e-10 | 2.63e-6 | 0.01128 |
| 80 | 7.55e-10 | 5.75e-6 | 0.05124 |

At resolution 80 the three-zero pulse's largest sample moves to the initial
edge near t=0.00577, instead of the Gaussian center near t=300. The public
interface therefore refuses more than two zeros. Even a double zero increases
out-of-band excitation and is not intrinsically preferable to a single zero.
No unsupported control changes a solver default.

## Near-Rayleigh comparison

The geometry and reference are the [ringdown](fdtd-ringdown.md) and
[CPML-strength](fdtd-pml-strength.md) cases: binary epsilon 2.25/1, thickness 0.2,
period 0.8, wavelengths 0.95808/0.96192, fixed transverse wavevector corresponding
to sin(theta)=0.2 at the central wavelength 0.96. Both incidence columns are
computed. All rows use pulse width 60, physical PML length 0.6, target 1e-40,
layer alignment, padding scale 1 and directional extraction 0.2 from each
interface. The nominal physical cutoff is 1.041666666667. Single/double Yee
notches follow the independently computed discrete cutoff of each grid.
The RCWA reference retains order 32 and compares physical orders -1/0.

Errors below are the maximum absolute complex r/t differences across both
requested wavelengths and those two orders. They include spatial, boundary and
finite-time contributions; a smaller number is not proof that each contribution
has decreased. All listed field/state decay checks remain false.

| Resolution | Periods | Source | TE error | TM error |
| --- | ---: | --- | ---: | ---: |
| 40 | 800 | Gaussian, previous control | 0.025048 | 0.048183 |
| 40 | 800 | One Yee-cutoff zero | 0.032591 | 0.051804 |
| 40 | 800 | Double Yee-cutoff zero | 0.028908 | 0.051259 |
| 40 | 800 | One physical-cutoff zero | 0.028720 | 0.050206 |
| 40 | 1600 | Gaussian, previous control | 0.022332 | 0.051702 |
| 40 | 1600 | One Yee-cutoff zero | 0.024928 | 0.051516 |
| 40 | 1600 | Double Yee-cutoff zero | 0.024983 | 0.051266 |
| 80 | 800 | Gaussian, previous control | 0.015474 | 0.010338 |
| 80 | 800 | One Yee-cutoff zero | 0.013935 | 0.011963 |
| 80 | 800 | Double Yee-cutoff zero | 0.014372 | 0.011892 |

At resolution 40 and 800 periods, one Yee-cutoff zero reduces the maximum
transmission-side incoming order -1 amplitudes from TE/TM 0.009470/0.003621 to
0.003984/0.001382. The double zero gives 0.002295/0.001056. These are apparent
incoming components of finite field spectra, not standalone PML reflection
coefficients. Their reduction does not imply a matching reduction in total
scattering error, as the table shows.

For the 1600-period single-zero run, comparison with its entire first half
(66511 updates, about 768 time units) gives TE/TM maximum amplitude changes
0.007789/0.000913. The Gaussian's corresponding changes were 0.006769/0.003899.
The TM time sensitivity is reduced by about a factor of four; TE is not improved.
The double-zero long run gives changes 0.004653/0.001063: both are below the
Gaussian controls, but still far above the 1e-4 convergence tolerance.
A double-zero 800-period run and a single-zero resolution-80 run pass the short
200-update scattering test while still failing state decay. Those local flags
must not be treated as infinite-time certificates.
The double-zero resolution-80 run also passes that short test, yet has larger
TE error than the single-zero run and larger TM error than the Gaussian. Its
transmission-side incoming TE/TM order -1 amplitudes are 0.007336/0.002005;
the coarse-grid reduction in incoming TE does not persist under refinement.

A notch at the physical threshold here also removes the unrequested carrier.
It is valid for band-only scattering, but the common scene adapter refuses
carrier-normalized time recording with that source. This is distinct from
failing to illuminate either of the two requested frequencies.

The measured controls justify exposing an optional source-shaping tool and
retaining the Gaussian default. They do not close the near-Rayleigh accuracy
limit or justify automatic spectral correction. Long-time, mesh and independent
boundary/source-position controls remain necessary.

## Reproduce

```shell
python -m studies.fdtd_pulse_notches --source-only
python -m studies.fdtd_pulse_notches --case single40
python -m studies.fdtd_pulse_notches --case double40
python -m studies.fdtd_pulse_notches --case physical40
python -m studies.fdtd_pulse_notches --case long40
python -m studies.fdtd_pulse_notches --case double-long40
python -m studies.fdtd_pulse_notches --case single80
python -m studies.fdtd_pulse_notches --case double80
```

The default study backend is double-precision JAX on CPU. `--backend numpy`
uses the same time-domain equations. Each run reports its actual source spectrum,
notch frequencies, mesh, incoming components and comparison window. The study
also provides double-zero long/fine controls. External-generator instructions
are in `tests/references/interface/README.md` in the checkout. GPU execution is
outside these measurements.

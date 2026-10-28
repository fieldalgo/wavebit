# A focused Gaussian beam

Written: 2026-10-27

Targets: `src/wavebit/fdtd/_beam_source.py`, the `GaussianBeam` declaration in
`src/wavebit/fdtd/_sources.py`, `tests/fdtd/test_fd_gaussian_beam.py`.

`fdtd.GaussianBeam` launches a focused beam from a plane, one way. It uses the one-sided
sheet of the mode source (`fd-mode-source.md`), with a beam for its incident field. On the
focal plane, the polarized transverse E sampled at its own Yee points is
`exp(-r^2 / waist^2)` about the focus (`waist` is the 1/e amplitude radius, as for
`rcwa.GaussianBeam`). The beam is the propagating part of that plane's angular spectrum,
built from the grid's own discrete plane waves in the homogeneous medium at the sheet.

At the pulse's centre frequency the beam is therefore an exact solution of the grid, and the
sheet launches it one way only. Away from the centre the sheet also launches a little of it
backward. That part is computed exactly, per plane wave, from the actual waveform rows. The
angular-spectrum representation is standard (Goodman, *Introduction to Fourier Optics*,
ch. 3). The oracles below are independent continuum computations written in the test.

## Registered acceptance criteria (2026-10-27, before computing)

**Setup B (two-dimensional).** Vacuum, x `Open` at ±6 µm, y `InvariantAxis`, z `Open` at
±5 µm, a 24-cell `PML` at 1 µm. `GaussianBeam('b')` on the plane z = -4 µm, direction `+z`,
waist 1.5 µm, focus at the origin, amplitude 1 (square-root watts per metre). Its pulse is
centred at 1 µm with `frequency_sigma_hz = 2e13` and `cutoff = 1e-12`. Polarization `y`
(TE: Ey, Hx, Hz) and `x` (TM: Ex, Ez, Hy). Spacings 0.05 and 0.025 µm. No
`UnsettledFieldWarning` or `GrowingFieldWarning` in any run.

* **G1, waist.** On the focal plane at 1 µm, `w = 2 sqrt(<x^2>)` of the transverse
  polarized component's `|E|^2` matches the waist: `|w / 1.5 µm - 1| <= 1e-6`, for both
  polarizations and both spacings.
* **G2, Gouy phase.** TE, on the axis (a `FieldMonitor` on the plane x = 0), at 1 µm. The
  phase is taken relative to the grid's axial plane wave,
  `psi(z) = arg Ey(0, z) - beta_axis z` (with `beta_axis` from the grid's dispersion), and
  referred to z = 0. It is compared with the same quantity of the exact continuum angular
  spectrum of the same focal field, `arg integral A(kx) exp(i kz z) dkx - k z`. The largest
  difference over z in [-3, 3] µm is at most 2e-3 rad at 0.05 µm and falls to 0.025 µm with
  an observed order of at least 1.8. The paraxial `arctan(z / zR) / 2` is reported beside it.
* **G3, focal shift.** TE, waist 0.6 µm, a dielectric half-space `epsilon = 2.25` for
  z > -1 µm (a `SnappedSlab` across z reaching the z+ face). The sheet is at z = -2.5 µm and the
  focus is declared at z = +0.5 µm (where the beam would focus in vacuum). The on-axis
  intensity peak inside the dielectric (a parabola through the five largest samples) is
  compared with the peak of the exact continuum angular spectrum with the TE Fresnel
  transmission at the interface. The difference is at most 0.03 µm at 0.05 µm and at most
  0.01 µm at 0.025 µm, smaller at the finer spacing. The paraxial focus
  (`-1 + 1.5 * 1.5 = 1.25` µm) is reported beside it.
* **G4, power.** A continuous wave of amplitude `A` carries `A^2` watts (per metre here), so
  the spectral energy through a `FluxMonitor` at z = +3 µm at 1 µm is `2 A^2 |W(f0)|^2`, `W`
  the continuous Fourier transform of the unit waveform. It matches to 2e-5 (relative) for
  both polarizations and both spacings. The same holds in three dimensions: x and y `Open`
  at ±3 µm, z `Open` at ±2.5 µm, a 12-cell `PML`, waist 1 µm, polarization `x`, the sheet at
  z = -2 µm, the monitor at z = +2 µm, spacing 0.05 µm.
* **G5, back-radiation.** TE, both spacings, at 0.95, 1.0 and 1.05 µm. The backward spectral
  energy through a `FluxMonitor` at z = -4.5 µm (behind the sheet) matches the predicted
  backward launch to `1e-6` of the predicted forward launch. The forward spectral energy at
  z = +3 µm matches the predicted forward launch to 1e-5 (relative). The predicted backward
  launch at 1.0 µm is at most 1e-12 of the forward; at 0.95 and 1.05 µm it is reported.
* **G6, backends.** The TE run of G1 at 0.05 µm on NumPy, JAX and Torch: the flux at
  z = +3 µm and the on-axis field agree to 1e-12 (relative).
* **G7, derivatives.** On JAX and Torch, TE at 0.05 µm: the derivatives of the flux at
  z = +3 µm at 1 µm with respect to the beam's `amplitude`, of `Re Ey(0, z = 3 µm)` with
  respect to its `phase` and `delay_seconds`, and of the flux with respect to the
  permittivity of a traced `SnappedBox` scatterer between the sheet and the monitor, off the
  axis and clear of the sheet plane. Each matches a central difference (two steps,
  Richardson-extrapolated) to 1e-6 relative.
* **G8, refusals by code.** A direction off the plane's normal (`source.direction`); a
  polarization along the normal (`source.polarization`); a non-positive waist
  (`source.waist`); a focus that is not a point (`source.focus`); a traced waist
  (`source.traced`); a traced pulse wavelength (`mode.traced`); a medium that is not uniform
  on the plane (`source.medium`); a metal transverse face (`source.boundary`); a beam that
  the cross-section does not hold (`source.beam_size`); a monitor within a cell of the sheet
  (`monitor.source_sheet`); the engine placeholder `fd.GaussianBeam` (`NotImplementedError`
  naming the public source).

Amendment to G4, before its three-dimensional case was first run: at the sheet (2 µm from the
focus, Rayleigh range 3.14 µm) the beam's radius is 1.19 µm, so on the outermost cells of a
±3 µm cross-section it is about 2e-3 of its peak, above the 1e-3 the source admits
(`source.beam_size`). That cross-section is widened to ±3.5 µm in x and y (the beam is then
2e-4 of its peak there). Everything else in G4 is as registered.

Second amendment to G4's three-dimensional case, again before it computed anything: with a
1 µm waist the source refused it (`source.beam_size`). The angular spectrum is 5e-5 of its
peak at the light line, and in three dimensions the ripple of its cut stays at 2.8e-7 of the
peak at the edge of the largest transform grid the source builds (2^24 points), above its
1e-7. The waist is 1.5 µm, as in Setup B. The cross-section is then ±4.5 µm in x and y, so
the beam (radius 1.56 µm at the sheet) is 3e-4 of its peak on the outermost cells. The
monitor spans ±4.4 µm. The limit is unchanged.

Amendment to G3, before it was first run: with a 0.6 µm waist the focal field's angular
spectrum is still 3% of its peak at the light line, where the beam's propagating part is cut,
and that part rings out as 1/r. The source refuses such a beam (`source.beam_size`): its
ripple stays above 1e-7 of the peak on any transform grid it builds. G3's waist is 1.0 µm
instead (the spectrum is 5e-5 of its peak at the light line). Everything else in G3, the
limits included, is as registered.

Amendment to G7, before any derivative was computed: for the reason S7's amendment in
`fd-mode-source.md` states (reverse mode keeps every step; a full Setup B run does not fit
in memory here), the derivatives and central differences are of the same truncated run: a
9 µm by 3 µm physical domain (x and z `Open`, 12-cell `PML`), the sheet at z = -1 µm, the
flux and the probe at z = +1 µm, the pulse with `cutoff = 1e-3`, `Duration(steps=1000)`, and
the scatterer (permittivity 2.0) centred at (2, 0, 0) µm. The four derivatives, the steps and
the 1e-6 limit are as registered.

## Results

Measured with `tests/fdtd/test_fd_gaussian_beam.py` on NumPy (JAX and Torch where named),
double precision.

| Criterion | Limit | 0.05 µm | 0.025 µm | Verdict |
| --- | --- | --- | --- | --- |
| G1 waist, TE / TM | `|w / waist - 1|` at most 1e-6 | 1.07e-8 / 1.08e-8 | -7.2e-9 / -7.2e-9 | passed |
| G2 Gouy phase against the continuum, largest over z in [-3, 3] µm | 2e-3 rad at 0.05 | **2.20e-3** | 5.42e-4 | **failed** |
| G2 observed order | at least 1.8 | | 2.02 | passed |
| G3 focal peak in the dielectric (continuum 1.37398 µm, paraxial 1.25 µm) | 0.03 at 0.05, 0.01 at 0.025, smaller | 1.34846 (0.0255) | 1.37084 (0.0031) | passed |
| G4 power, TE / TM | 2e-5 | -1.5e-12 / -7.6e-13 | -9.0e-13 / -5.5e-13 | passed |
| G4 power, three dimensions | 2e-5 | -3.6e-8 | | passed |
| G5 backward energy against the prediction, per forward | 1e-6 | 6.2e-12 / 6.2e-14 / 5.0e-12 | 2.5e-12 / 4.1e-14 / 5.0e-12 | passed |
| G5 forward energy against the prediction | 1e-5 | 1.3e-11 / 6.3e-13 / 5.0e-12 | 7.4e-12 / 3.8e-13 / 6.1e-12 | passed |
| G5 predicted backward per forward at 1.0 µm | 1e-12 | 1.4e-13 | 8.5e-14 | passed |
| G5 predicted backward per forward at 0.95 / 1.05 µm | reported | 2.6e-7 / 2.8e-7 | 2.5e-7 / 2.8e-7 | |
| G6 JAX / Torch against NumPy, flux and axis | 1e-12 | 2.3e-16, 1.0e-15 / 1.1e-16, 1.2e-15 | | passed |
| G7 derivatives against extrapolated central differences | 1e-6 | pending | | |
| G8 refusals by code | as listed | all raised | | passed |

**G2 failed as registered.** The largest Gouy-phase difference from the continuum at
0.05 µm is 2.20e-3 rad, above the registered 2e-3. The error falls at second order
(5.42e-4 at 0.025 µm), the registered second part. The difference is the grid's dispersion:
the run's beam is exact for the grid, and the grid's plane waves have a phase velocity that
depends on their direction, which the continuum's do not. The limit is not changed. The test is kept as a strict expected
failure beside a passing test of its order. For scale, the exact beam's Gouy phase differs
from the paraxial `arctan(z / zR) / 2` by 2.1e-3 rad over the same range.

Two details of the runs, neither a change to a limit. G3's half-space crosses the x faces,
so it declares `continuation=('x-', 'x+')`, the slab door of `fd-waveguide-through-pml.md`.
The `FluxMonitor`s span x in ±5.9 µm, clear of the absorbing layers, as a flux plane must.
A `FieldMonitor` is not refused within a cell of the sheet (the refusal covers flux planes,
as for a TFSF box), so G8 tests that refusal with a `FluxMonitor`.

A design constant moved once, before any result depended on it. When first written, the
source required the beam to fall below 1e-13 of its peak at the edge of its padded transform
grid. The first run (G1's setup) found the light-line ripple at 5e-11, so the constant became
1e-7, with the padding doubled until the beam meets it (at most 2^24 points). The G3 and
three-dimensional G4 amendments above are refusals under that rule.

Added after registration, as a further test: the beam is centred on a transverse focus.
Thirteen deliberate breaks of the source were caught: each sign of the discrete Faraday law,
the cosine rows' quarter period, the 1 W scaling, the polarization mapping, the axial and
the transverse focus, the edge, medium and face rules, the traced-waist refusal, and two
errors in the back-radiation prediction. Each time the source was restored byte for byte.
As for the mode source, the H rows' half-cell group delay changes only the off-centre
back-radiation. G5 reports that part and the exact prediction follows it, so no limit bounds
the delay.

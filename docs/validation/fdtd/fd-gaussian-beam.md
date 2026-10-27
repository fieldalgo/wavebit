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

## Results

Pending.

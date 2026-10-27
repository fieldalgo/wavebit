# Gaussian beams

`fdtd.GaussianBeam` launches a focused Gaussian beam from a plane, one way.

```python
from wavebit import Units, fdtd

beam = fdtd.GaussianBeam('b', plane=fdtd.Plane(z=-4.), direction='+z', waist=1.5,
                         focus=(0., 0., 0.), polarization='y',
                         pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=2e13))
```

## What it launches

On the focal plane, the E component along `polarization` (a transverse axis) is
`exp(-r**2 / waist**2)` about `focus`, and the other transverse component is zero.
`waist` is the 1/e amplitude radius, as for `rcwa.GaussianBeam`. The beam is the propagating
part of that plane's angular spectrum. It is built from the grid's own plane waves in the
homogeneous medium that fills the source plane, so at the pulse's centre wavelength it is an
exact solution of the grid. `focus` is where the beam would focus if that medium filled
space. A beam that enters another medium focuses where refraction takes it.

`amplitude` is in square-root watts (per metre of each invariant axis): a continuous wave of
amplitude `A` carries `A**2` watts. `phase` is in radians, and `delay_seconds` delays the
pulse. In a two-dimensional problem (one `InvariantAxis`) the beam is a sheet beam, focused
in the resolved transverse axis only.

The plane is a one-sided total-field/scattered-field sheet, as for `ModeSource` (see
[Waveguide modes](waveguide-modes.md)). At the centre wavelength it launches exactly the
beam, one way. Away from it, each plane wave's profile and half-cell phase are the centre's,
and a little of the beam goes backward. The [validation record](../../validation/fdtd/fd-gaussian-beam.md)
measures that part against its exact prediction.

## What it requires

- The faces across the plane are `Open` (or `InvariantAxis`). A metal, mirror or periodic
  face is refused (`source.boundary`): the beam would become the modes of that guide or
  lattice.
- The medium on the plane is uniform (`source.medium`), lossless and non-dispersive, and the
  same along the axis for a cell each way (the window rules of `solve_modes`).
- The cross-section holds the beam: on the outermost cells of the plane it must be at most
  1e-3 of its peak (`source.beam_size`). The sheet stops at the window, one cell inside an
  `Open` face, and launches nothing beyond it.
- `waist` and `focus` are concrete numbers (`source.traced`): the beam is built on the host.
  `amplitude`, `phase` and `delay_seconds`, and structures elsewhere, may carry traced values.
  The pulse wavelength may not (`mode.traced`).
- Normal incidence only: the beam's axis is the plane's normal. A flux monitor within one
  cell of the sheet is refused (`monitor.source_sheet`).

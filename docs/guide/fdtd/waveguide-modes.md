# Waveguide modes

`fdtd.solve_modes` finds the guided modes of a cross-section **on the grid a run uses**.
It prepares the problem exactly as `fdtd.prepare` would, reads the permittivity and
permeability the time stepper multiplies by on a plane, and solves the eigenproblem of the
discrete update itself. A mode it returns is therefore a mode of the run, not of the
continuum: started in a run it stays itself to round-off, and its distance from the
continuum mode is the grid's discretization error.

```python
from wavebit import Material, Units, fdtd

problem = fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
    domain=fdtd.Domain(size=(3., 3., 0.4)),
    boundaries=fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=fdtd.Periodic()),
    structures=(fdtd.Structure('core', shape=fdtd.SnappedBox(size=(1., .5, 10.)),
                               material=Material(epsilon=4.)),)))
numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=1))
modes = fdtd.solve_modes(problem, numerics=numerics, plane=fdtd.Plane(z=0.),
                         wavelength=1.0, count=2)
modes[0].effective_index        # beta c / omega on the grid's own dispersion
modes[0].E['Ex'], modes[0].H['Hy']   # V/m and A/m on each component's Yee points
modes.axes, modes.coordinates['Ex']  # ('y', 'x') and the coordinates of those points
```

## What it takes

- `problem`, `numerics`: the same declarations a run takes. Only the scene and the grid
  matter; sources and monitors are ignored. The time step does matter: the leapfrog's
  `Omega = (2/dt) sin(omega dt/2)` replaces `omega`.
- `plane`: an `fdtd.Plane` across the propagation axis. Its position snaps to the nearest
  node plane, which must lie in the physical domain where the medium is uniform along the
  axis (one layer either side is compared). Without transverse spans the window is the
  physical cross-section; spans (`fdtd.Plane(z=0., x=(-1., 1.))`) snap to node planes and
  close the window with a wall there.
- `wavelength`: the vacuum wavelength in `Problem.units`.
- `count`: how many modes, by decreasing effective index. Fewer confined modes than asked
  is a refusal (`mode.count`) that says how many were found.
- `backend`: `'numpy'`, `'jax'` or `'torch'` for the returned arrays. The eigensolve runs
  once on the host (SciPy) whatever the backend, so the three return the same bits.

## What it returns

`WaveguideModes` holds the modes (`modes[0]` is the largest index), the plane's `axis` and
snapped `position`, the `wavelength`, the two array `axes` in storage order (`('y', 'x')`
for a z plane) and the `coordinates` of every component. Each `WaveguideMode` carries

- `effective_index`: `beta c / omega`, with `beta` from the grid's own dispersion
  `q = (2/dx) sin(beta dx/2)`; `beta_per_meter` is `beta` in rad/m;
- `E` and `H`: complex arrays in V/m and A/m on the component's own Yee points. The
  transverse components are real and the longitudinal ones imaginary. They are the phasors
  of a run with `exp(-i omega t)`, each component at its own times and its own position: E
  on the node plane, and the H components half a cell along the axis referred to that half
  plane;
- `polarization`: the share of the transverse electric energy along each transverse axis
  (`{'x': 0.98, 'y': 0.02}` for a mode polarized along x);
- `power_watts`: 1. The flux a `FluxMonitor` would measure for the mode -- the node-plane E
  against the mean of the two H half planes -- is normalized to 1 W toward `+axis` (W/m with
  one `InvariantAxis`, W/m² with two).

The modes are orthogonal under the unconjugated cross product over the window,
`sum (E_u H_v - E_v H_u) dA = 0` for two modes with different indices, exactly on the grid.
A degenerate pair (a square guide's two polarizations) is returned already orthogonalized.

## Transverse faces

| Face | Treatment |
| --- | --- |
| `PEC`, `PMC` | The kernel's own wall, where the kernel puts it (a PMC wall lies half a cell off a low face). |
| `Periodic` | Periodic, when the window spans the axis. |
| `Open` | A wall one cell inside the physical domain's edge (the window's edge node is the wall, and a flux plane over the window must not read the absorbing layer). A mode is admitted only if its transverse E on the outermost cells is at most 1e-3 of its peak; a leaky mode, one near cutoff, or a mode of the truncated box is not. |
| `InvariantAxis` | The one-dimensional (slab) limit, for `k = 0`. |
| `Bloch`, `MirrorSymmetry`, `InvariantAxis` with `k != 0` | Refused (`mode.boundary`). |

## What it refuses

Each refusal is a structured error with its code:

- `mode.material`: a conductive or dispersive medium in the window, or coupled smoothed
  (constitutive) material or `discretization='stencil'` anywhere in the scene. Cell-snapped
  bodies (`SnappedBox`, `SnappedSlab`) and unturned smoothed `Box` faces lower to a diagonal
  medium and are admitted; a `Cylinder`, `Prism`, `Sphere` or turned `Box` couples components
  and is not.
- `mode.nonuniform`: the medium changes along the axis at the plane.
- `mode.traced`: a JAX tracer or a Torch tensor that carries a gradient in a material, the
  wavelength, or a ranged geometry. The mode solve does not differentiate; solve it from
  concrete values.
- `mode.outside_domain`, `mode.plane`: a plane outside the physical domain, across an
  `InvariantAxis`, or with an empty or outside span.
- `mode.count`, `mode.convergence`: fewer confined modes than requested; an eigenpair whose
  residual exceeds 1e-8.

## Measuring modes in a run: `ModeMonitor`

```python
outputs = fdtd.Outputs(monitors=(
    fdtd.ModeMonitor('out', plane=fdtd.Plane(z=3.), wavelengths=(1.5, 1.55), modes=2),))
spectrum = fdtd.run(problem.replace(outputs=outputs), numerics=numerics)['out']
spectrum.forward_energy, spectrum.backward_energy   # (wavelengths, modes), J/Hz
spectrum.forward, spectrum.backward                  # complex amplitudes, sqrt(J/Hz)
spectrum.effective_index, spectrum.modes[0][0]       # the modes it projected on
```

At each wavelength the monitor solves the plane's modes exactly as `solve_modes` would and
projects the run's Fourier-transformed fields onto them: the node-plane E and the mean of
the two H half planes, the pairing a `FluxMonitor` uses. Because the modes are exactly
orthogonal on the grid, the projection separates each mode and each direction without
crosstalk from the other modes, including the evanescent ones. `forward_energy` and
`backward_energy` are the two-sided pulse spectral energies carried toward the plane's `+`
and `-` normal, in the units of `FluxMonitor.spectral_energy`; for a field made only of the
projected modes, `sum(forward_energy - backward_energy)` over the modes is the flux. Divide by
the same quantity from a reference run (or, with a mode source, by its injected spectrum) to
get a transmission.

The plane must cross a uniform section of the guide, with the window, faces and refusals of
`solve_modes`. A structure that reaches the plane must be concrete. Other structures may
carry traced values, so the energies differentiate through, say, a scatterer between source
and monitor (measured against central differences in the
[validation record](../../validation/fdtd/fd-waveguide-modes.md)). Across an `Open` face the
window stops one cell inside the physical edge, for the monitor and for `solve_modes` alike;
the monitor reads no absorbing cells.

A guide that runs out through an `Open` face is written with a cell-snapped core that
continues through it: `fdtd.Structure('core', shape=fdtd.SnappedBox(size=(1., .5, 10.)),
material=..., continuation=('z-', 'z+'))`, or a `SnappedSlab` with the faces of the axes it
runs along. The guided mode then enters the absorbing layer without meeting an end face; its
measured reflection is in the [validation record](../../validation/fdtd/fd-waveguide-through-pml.md).
A smoothed core is refused at an `Open` face.

## Launching a mode: `ModeSource`

```python
source = fdtd.ModeSource('src', plane=fdtd.Plane(z=-2.), direction='+z', mode=0,
                         pulse=fdtd.GaussianPulse(wavelength=1.55, frequency_sigma_hz=1.5e13))
monitor = fdtd.ModeMonitor('t', plane=fdtd.Plane(z=2.), wavelengths=(1.55,), source='src')
# ... run ...
t = result['t']
t.forward_energy[:, 0] / t.injected_energy   # the TE0 power transmission
```

`ModeSource` launches mode `mode` of `solve_modes` on `plane`, solved at the pulse's centre
wavelength, toward `direction` (`'+'` or `'-'` the plane's normal). The plane has the window,
faces and refusals of `solve_modes`. `amplitude` is in square-root watts (per metre of each
invariant axis): a continuous wave of amplitude `A` carries `A**2` watts. `phase` is in
radians, and `delay_seconds` delays the pulse.

The plane is a one-sided total-field/scattered-field sheet whose incident field is the mode.
At the pulse's centre wavelength it launches exactly that mode, exactly one way. At other
wavelengths in the band it also launches a little of the mode the other way, because the
profile and the half-cell phase it uses are the centre's. Both parts are known exactly for
the grid: a `ModeMonitor` with `source=` reports them as `injected` (the source's way) and
`leakage` (the other way), as complex amplitudes of the source's mode on the source's plane,
with `injected_energy` and `leakage_energy`. Divide a monitor's `forward_energy` by
`injected_energy` for a transmission; nothing needs a reference run.

Other modes of the guide are not launched at the centre wavelength. Away from it the profile
projects a little onto other modes of the same symmetry; a monitor that reads several modes
shows them. A monitor within one cell of the sheet is refused (`monitor.source_sheet`): the
field jumps there. A structure that reaches the source plane must be concrete. The source's
`amplitude`, `phase` and `delay_seconds`, and structures elsewhere, may carry traced values.
The pulse wavelength may not, because the mode is solved at it.

## Accuracy

The registered checks and their measured results are in the
[validation record](../../validation/fdtd/fd-waveguide-modes.md). In short: a uniform medium
gives its own index and the closed-form Yee dispersion to 1e-12; a hollow metal guide gives
the discrete closed form to about 1e-12; a symmetric slab converges to the exact TE and TM
dispersion at second order; the published dielectric-corner benchmark is reached at second
order; and a solved mode started in a run stays itself to round-off. The effective index on
a finite grid is the grid's, so converge the spacing as for any FDTD result.

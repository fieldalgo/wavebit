# Far-field radiation in the general FDTD engine

Use `fdtd.FarField` for a closed measurement box in an open scene.
The box must enclose every object and stay in a uniform isotropic background,
clear of absorbing layers and source surfaces.

```python
from wavebit import fdtd

monitor = fdtd.FarField(
    'ff', wavelengths=(1.,), box=fdtd.Region(size=(1.5,)*3), propagation='yee')
```

List the monitor in `fdtd.Outputs(monitors=...)` before calling `fdtd.run`.
`propagation` chooses how the sampled surface fields propagate to infinity:

| Value | Meaning |
| --- | --- |
| `'continuum'` (default) | Original continuum Green-function approximation. Preserves earlier results. Its phase and power can differ from discrete grid propagation at finite resolution. |
| `'yee'` | Outgoing field of the homogeneous Yee lattice, including directional dispersion, staggered reciprocal fields, and the lattice energy-flux weight. |

Continuum monitors issue a `RadiationApproximationWarning` (from
`wavebit.fdtd.diagnostics`) for the whole scene during planning. `fdtd.check` lists
the same note in `report.capabilities['warnings']` with the measured weak-sphere
example; those example errors are not a bound for your scene.

For a result `res`, `res['ff'].fields(theta, phi)` returns electric theta/phi
projections. Angles are in radians, theta measured from +z and phi from +x
toward +y. They broadcast together; each output has shape `(nf, ...angles)`.
`res['ff'].intensity(theta, phi)` gives the pulse's two-sided spectral energy per
solid angle, in `res['ff'].units` (J/Hz/sr for a three-dimensional scene). For the
net flux through the box, before any far-field map, add an `fdtd.Flux` monitor on
the same box: it reads the same faces.

With `'yee'`, angles specify outgoing energy rays. The asymptotic phase is
`k_star dot position`, where `k_star` lies on the lattice dispersion surface.
Electric polarization is transverse to the discrete wavevector, which need not
follow the ray. Use `intensity()` for power; squaring only the two angular
projections omits the possible radial component. The three complex Cartesian
amplitudes, `vector(theta, phi)` with shape `(nf, ...angles, 3)`, are not in the
current result yet; the earlier interface, `wavebit.compat.legacy.fd`, has them.

Recorded fields remain differentiable on JAX and Torch. Yee background
parameters and requested angles must be concrete host values. The background
must have positive scalar epsilon and mu. Planning refuses all frequencies
unless `omega*dt < pi` and `0 < Omega*sqrt(epsilon*mu)*dx < sqrt(2)`, where
`Omega=2*sin(omega*dt/2)/dt`. These conditions select a unique convex outgoing
dispersion surface. No numerical clamp extends that supported band.

Selecting the lattice map does not establish material-boundary accuracy.
Check time, boundary and spatial convergence for the requested observable.
See [the derivation and measured validation](../../validation/fdtd/fd-lattice-radiation.md).

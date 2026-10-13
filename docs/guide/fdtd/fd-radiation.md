# Far-field radiation in the general FDTD engine

:::{note}
This detailed guide documents the retained `wavebit.compat.legacy` interface and its measured scope. For the current problem/numerics lifecycle, start with [the common guide](../common/lifecycle.md). Advanced capabilities are preserved here with explicit compatibility calls; their limits are unchanged.
:::

Use `wavebit.fd.FarField` for a closed measurement box in an open scene.
The box must enclose every object and stay in a uniform isotropic background,
clear of absorbing layers and source surfaces.

```python
from wavebit.compat.legacy import fd

monitor = fd.FarField(
    'ff', box=fd.Region(size=(1.5,)*3), wavelengths=(1.,), propagation='yee')
```

Add the monitor to the scene before calling `fd.run`. `propagation` chooses
how the sampled surface fields propagate to infinity:

| Value | Meaning |
| --- | --- |
| `'continuum'` (default) | Original continuum Green-function approximation. Preserves earlier results. Its phase and power can differ from discrete grid propagation at finite resolution. |
| `'yee'` | Outgoing field of the homogeneous Yee lattice, including directional dispersion, staggered reciprocal fields, and the lattice energy-flux weight. |

Continuum monitors issue one `fd.RadiationApproximationWarning` for the whole
scene during planning. `fd.check` includes the same note with the measured
weak-sphere example; those example errors are not a bound for your scene.

For a result `res`, `res['ff'].fields(theta, phi)` returns electric theta/phi
projections. Angles are in degrees, theta measured from +z and phi from +x
toward +y. They broadcast together; each output has shape `(nf, ...angles)`.
`res['ff'].intensity(theta, phi)` gives power per solid angle.
`res['ff'].flux` gives the net power through the box, before any far-field map.

With `'yee'`, angles specify outgoing energy rays. `res['ff'].vector(theta,
phi)` returns the three complex Cartesian amplitudes with shape
`(nf, ...angles, 3)`. The asymptotic phase is `k_star dot position`, where
`k_star` lies on the lattice dispersion surface. Electric polarization is
transverse to the discrete wavevector, which need not follow the ray.
Use `intensity()` for power; squaring only the two angular projections omits
the possible radial component. `vector()` refuses a continuum result.

Recorded fields remain differentiable on JAX and Torch. Yee background
parameters and requested angles must be concrete host values. The background
must have positive scalar epsilon and mu. Planning refuses all frequencies
unless `omega*dt < pi` and `0 < Omega*sqrt(epsilon*mu)*dx < sqrt(2)`, where
`Omega=2*sin(omega*dt/2)/dt`. These conditions select a unique convex outgoing
dispersion surface. No numerical clamp extends that supported band.

Selecting the lattice map does not establish material-boundary accuracy.
Check time, boundary and spatial convergence for the requested observable.
See [the derivation and measured validation](../../validation/fdtd/fd-lattice-radiation.md).

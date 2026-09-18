"""General-purpose FDTD on a uniform Yee grid: the stage-0 kernel, at its interim name.

``import wavebit.fd as fd``. The final home is ``wavebit.fdtd``, which today is the legacy
periodic-only door exported from the package root; until that door is retired this engine
lives here, imports nothing from the old engines, and is imported by nothing at the
package root -- ``import wavebit`` does not load it, and loading it does not load JAX.

Conventions, fixed for every module below (each repeats the ones it uses):

* Units: ``c = 1``, ``eps0 = mu0 = 1``, lengths in the user's unit, so ``f = 1/wavelength``
  and ``dt`` is a length; H is stored as ``Z0 * H`` so the vacuum update is
  ``H -= dt curl E``, ``E += dt curl H``.
* Layout: fields are ``[component, z, y, x]`` arrays of shape ``(3, nz, ny, nx)``,
  component 0 = x, 1 = y, 2 = z. Typed triples in the public API are ``(x, y, z)``. Faces
  are ``'x-'``, ``'x+'``, ``'y-'``, ``'y+'``, ``'z-'``, ``'z+'`` and never ordered.
* Yee offsets (half cells along x, y, z): Ex (1,0,0), Ey (0,1,0), Ez (0,0,1); Hx (0,1,1),
  Hy (1,0,1), Hz (1,1,0). Cell ``i`` covers ``[x_i, x_i + dx)``. All six components share
  one array shape; walls are neighbour rules and masks, never ``N + 1`` arrays.
* ``curl E`` is a forward difference and pads E on the high faces; ``curl H`` is a
  backward one and pads H on the low faces. Nothing else is ever padded.
* Uniform grid; ``dt = courant * dx / sqrt(3)``.
* Time convention ``e^{-i w t}``: a Bloch phase ``phi`` means ``field(x + L) = exp(+i phi)
  field(x)``; a run is real unless a face is Bloch.

The public contract of stage 0::

    scene = fd.Scene(domain=fd.Domain(size=(1, 1, 1)),
                     boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                     resolution=fd.Resolution(dx=1/16), run=fd.Run(steps=400),
                     initial=fd.InitialField(E=(None, lambda x, y, z: ..., None)),
                     monitors=(fd.Record('probe', at=(0.1, 0.0, -0.05), component='Ey'),))
    report = fd.check(scene)                  # prints cells, dt, dtype, bytes, warnings
    plan = fd.plan(scene)                     # static, hashable; plan.skeleton is the hash
    res = fd.run(scene, backend='jax')        # 'numpy' | 'jax' | 'torch' | wavebit.Backend
    res['probe'].t, res['probe'].values       # the time series
    res.final.E, res.final.H                  # (3, nz, ny, nx) on the backend that ran

Out of scope in stage 0, and refused by name rather than silently: absorbers and materials
(stage 1), sources (stage 2), DFT/flux/far-field monitors and ``algebraic`` faces (stage 3),
shapes (stage 4), GPU claims (stage 6), ``Decay`` (stage 7), non-uniform grids (not in v1).
"""
from ._check import Report, check
from ._lower import Plan, lower, plan
from ._run import run
from .boundaries import PEC, PMC, Bloch, Boundaries, Periodic, Symmetry
from .monitors import Record
from .results import Final, Results, Series
from .scene import Domain, Resolution, Run, Scene
from .sources import InitialField, Term

__all__ = [
    'Domain', 'Resolution', 'Run', 'Scene',
    'Boundaries', 'Periodic', 'Bloch', 'PEC', 'PMC', 'Symmetry',
    'InitialField', 'Term', 'Record',
    'check', 'plan', 'lower', 'run',
    'Plan', 'Report', 'Results', 'Series', 'Final',
]

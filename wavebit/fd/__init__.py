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

Stage 1 adds absorbers and materials, and nothing else::

    glass = fd.Material(eps=2.25)                      # real, scalar or (ex, ey, ez); mu=
    lossy = fd.Conductor(eps=2.25, sigma=0.01)         # sigma >= 0, in the scene's units
    metal = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=..., gamma=...),
                                              fd.Lorentz(omega_0=..., gamma=..., strength=...),
                                              fd.Debye(tau=..., delta_eps=...)))
    scene = fd.Scene(..., background=fd.Material(eps=1.0),
                     boundaries=fd.Boundaries.open(fd.PML(cells=12, wavelength=0.55)),
                     structures=(fd.Structure(fd.Slab(z=(-0.2, 0.0)), glass),))
    fd.eps_at(metal.poles, 0.55, metal.eps_inf)        # the closed form, for a cross-check

A ``PML`` face is a PEC wall with a stretched slab in front of it; ``Absorber`` is the
fallback where a stretch cannot be matched (a conductive or dispersive medium crossing the
face). Slabs are **cell-snapped, not subpixel**, and ``check`` prints the snapped span beside
the one asked for -- until stage 4 a convergence study in ``dx`` is the only route to a right
answer for an interface that does not land on a cell boundary.

Stage 2 adds sources, each a term in the kernel's additive slot::

    band = fd.Band(wavelength=0.55, df=0.3)            # Gaussian-envelope sine; df in frequency
    fd.Dipole('d', center=(x, y, z), component='Ey', band=band)          # kind='magnetic': 'Hy'
    pw = fd.PlaneWave('pw', z=1.0, direction='-z', theta=20, polarization='s', band=band)
    fd.Boundaries(**fd.Bloch.from_source(pw), z=fd.PML(cells=12))       # the matching cell
    fd.TFSF('t', box=fd.Box(size=(...), center=(...)), direction='-z', polarization='x',
            band=band)                                 # axis directions only (measured)
    scene = fd.Scene(..., sources=(...,), run=fd.Run(periods=40))
    res.sources['pw'].t, res.sources['pw'].incident    # each source's reference series

A PML with no wavelength takes the sources' band centre; the mesh is checked at
``band.shortest_wavelength``. ``_kernel/sources.py`` derives every sign and half step.

Out of scope in stage 2, and refused by name rather than silently: DFT/flux/far-field
monitors and ``algebraic`` faces (stage 3), shapes other than ``Slab`` -- a ``Box`` is a TFSF
region only -- and subpixel averaging (stage 4), GPU claims (stage 6), ``GaussianBeam``,
``ModeSource`` and ``Decay`` (stage 7), non-uniform grids (not in v1).
"""
from ._check import Report, check
from ._lower import Plan, lower, plan
from ._run import run
from .boundaries import (PEC, PMC, Absorber, Absorbing, Bloch, Boundaries, Periodic, PML,
                         Symmetry)
from .diagnostics import AbsorberProximityWarning, CoarseMeshWarning
from .geometry import Box, Slab, Structure
from .materials import (Conductor, Debye, Dispersive, Drude, Lorentz, Material, MaterialBase,
                        Pole, eps_at)
from .monitors import Record
from .results import Final, Results, Series, SourceSeries
from .scene import Domain, Resolution, Run, Scene
from .sources import (TFSF, Band, Dipole, GaussianBeam, InitialField, ModeSource, PlaneWave,
                      Term)

__all__ = [
    'Domain', 'Resolution', 'Run', 'Scene',
    'Boundaries', 'Periodic', 'Bloch', 'PEC', 'PMC', 'Symmetry', 'PML', 'Absorber',
    'Absorbing',
    'Material', 'Conductor', 'Dispersive', 'Drude', 'Lorentz', 'Debye', 'eps_at',
    'MaterialBase', 'Pole', 'Slab', 'Structure',
    'CoarseMeshWarning', 'AbsorberProximityWarning',
    'InitialField', 'Term', 'Record',
    'Band', 'Dipole', 'PlaneWave', 'TFSF', 'Box', 'GaussianBeam', 'ModeSource',
    'check', 'plan', 'lower', 'run',
    'Plan', 'Report', 'Results', 'Series', 'Final', 'SourceSeries',
]

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
    fd.TFSF('t', box=fd.Region(size=(...), center=(...)), direction='-z', polarization='x',
            band=band)                                 # axis directions only (measured)
    scene = fd.Scene(..., sources=(...,), run=fd.Run(periods=40))
    res.sources['pw'].t, res.sources['pw'].incident    # each source's reference series

A PML with no wavelength takes the sources' band centre; the mesh is checked at
``band.shortest_wavelength``. ``_kernel/sources.py`` derives every sign and half step.

Stage 3 adds the frequency-domain monitors, the sources' spectral normalisation, the decay
diagnostic, ``Run(time=)`` and the cell-snapped ``Region`` structure::

    ws = (0.5, 0.55, 0.6)                                        # vacuum wavelengths, static
    fd.Flux('out', box=fd.Region(size=(0.5,)*3), wavelengths=ws)    # net outward flux of a box
    fd.Flux('t', plane=fd.Plane(z=-1.0), wavelengths=ws)         # flux along +z through a plane
    fd.Field('f', plane=fd.Plane(z=0.2), components=('Ex', 'Ey'), wavelengths=ws)
    fd.Diffraction('r', z=1.2, side='reflect', wavelengths=ws)   # orders of a periodic cell
    fd.FarField('ff', box=fd.Region(size=(0.5,)*3), wavelengths=ws) # near-to-far, open scenes
    fd.Structure(fd.Region(size=(0.2, 1.0, 0.1), center=(...)), glass)   # a lamellar tooth
    fd.Boundaries(x=fd.Bloch(k=kx), y=fd.Axis1D(k=ky, k_max=...), z=fd.PML(...))   # conical
    fd.Run(time=12.0)                                            # an end time instead of steps
    res['out'].flux, res['r'].orders.order(1, 0).R, res['r'].theta_at
    res['ff'].intensity(theta, phi), res['ff'].flux              # degrees; (nf, ...)
    res.sources['pw'].incident_amplitude(ws), res.sources['t'].intensity(ws),
    res.sources['d'].power(ws), res.diagnostics['decay'], res.diagnostics['interior_growth'],
    res.diagnostics['growth'], res.diagnostics['stable'], res.diagnostics['grazing']

Every one of those monitors lowers to the kernel's one frequency-domain observer, a running
DFT over a region with each component at its own time level (``_kernel/observe.py``);
everything built on it is post-processing in the backend's arrays (``_post.py`` derives the
2-cell flux collocation, the Yee-kz de-embedding of the orders and the E/H direction split).
A frequency-domain monitor whose region still holds field at the end warns with
``fd.UnsettledFieldWarning``; when the energy of the scene's interior still rises after every
source has ended, every monitor that sees it warns ``fd.GrowingFieldWarning`` (its subclass)
instead, naming the PML's ``stabilize=`` knob (``res.diagnostics['interior_growth']``, and
per monitor ``res.diagnostics['growth']``). ``FarField`` is the equivalence-principle
transform of a closed box's DFT surface fields (``_post.py``), for open scenes only;
``Axis1D`` is a one-cell axis whose derivative is ``i k`` exactly (a user ``Bloch`` phase on a
one-cell axis is refused, naming it); and a propagating order a PML face cannot absorb to the
measured tolerance is refused by the grazing rule (``_lower/grazing.py``, which quotes the
measurement, judges each wavelength with the CFS shift the layer has there, and judges a face
with ``PML(stabilize=)`` by what its loss returns on each order, computed exactly for this
kernel in ``_lower/layer_return.py``). A flux
surface -- a Flux or Diffraction plane, a Flux or FarField box -- keeps off every source: one
that cuts a Dipole's stamp, lies on a PlaneWave's sheet (or a box around part of it) or cuts a
TFSF surface is refused, because the 2-cell face would read part of the source
(``_lower/monitors.py``).

Stage 4's C2 adds the first shape whose edge is **not** a cell boundary::

    fd.Structure(fd.Sphere(radius=0.2, center=(0, 0, 0)), glass)   # subpixel-smoothed

``Slab`` and ``Region`` do not move and every stage-0..3 number is unchanged to the bit; a
``Sphere`` is asked, at each of the six components' **own** Yee points, what fraction of that
voxel it fills and which way its surface faces there, and its material is mixed with whatever
the cell already held by the Farjadpour/Kottke rule (``_lower/occupancy.py`` produces the
pair, ``_lower/coefficients.py::mix`` is the rule and carries its citations). The rule is two
isotropic phases meeting at one interface, and C2 refuses every neighbouring case **by name**
rather than answering it approximately (``_lower/checks.py::check_smoothed``): a smoothed
shape may not share cells with an ``fd.Conductor`` or an ``fd.Dispersive`` (``sigma`` and the
poles are not smoothed yet) or with an anisotropic ``eps``/``mu``; it may not reach a
``Periodic``, ``Bloch`` or ``Axis1D`` face, nor a face of the box (it would count material
outside a ``-`` face and lose the last half cell at a ``+`` one -- two arms, two messages);
and two smoothed shapes may not put their surfaces through one voxel. The differentiable
``range=`` edge and CSG (which is how two smoothed bodies that touch are spelled) are later
commits.

Stage 4's C4 adds the extruded bodies, smoothed the same way, and renames the cell-snapped box
``fd.Region`` so that ``fd.Box`` can be the body (one name per concept)::

    fd.Structure(fd.Box(size=(0.3, 0.2, 0.1), center=(...), angle=30), glass)  # turned 30 deg
    fd.Structure(fd.Cylinder(radius=0.1, height=0.2, axis='z'), glass)
    fd.Structure(fd.Prism(vertices=((0, 0), (0.2, 0), (0.1, 0.17)), height=0.2), glass)

``angle`` is in degrees and turns a body about its own axis only; a tilted axis is stage 5 and
refused by name. A body **uniform along** an axis over the whole domain -- extruded along it,
or an unturned ``Box`` spanning it -- may run through a ``Periodic``, ``Bloch`` or ``Axis1D``
face, since every layer holds the same section; any other body there is still refused.

Stage 4's C3 adds a second material discretisation, **opt in and per structure**::

    fd.Structure(fd.Sphere(radius=0.2), glass, discretisation='stencil')

``'voxel'`` is the default and is every scene's behaviour above, to the bit; ``'stencil'``
asks instead for the symmetric extended-stencil operator ``docs/fd-symmetric-stencil.md``
derives -- exact edge and dual-face interface constraints, solved once on the host at
``fd.plan`` time and never inside a trace, whose rows couple each edge to every edge within
two cells. ``check`` prints it. It has its own doors and not one of them falls back to the
other rule: only ``fd.Sphere`` itself, isotropic, lossless and non-magnetic, with concrete
geometry and material values at plan time **and** at lowering; the only structure in the
scene, in a vacuum background, in a real run with no initial field; no wall, absorbing
layer, source or monitor surface inside the region the operator owns (its support plus
three nodes a side); under a declared preparation memory bound (``memory_bound=`` raises
it); and an operator that passes its own consistency, energy, halo and time-step
certificates (``_lower/stencil.py``). ``fd.run(scene, plan=p)`` takes the operator ``p``
already carries rather than solving it again.

Smoothed scenes emit ``SubpixelAccuracyWarning`` during planning, also shown by ``check``,
with the benchmark behind the structure's discretisation. ``'stencil'`` passes C3 (the
epsilon-12 sphere converges at order 2.244; ``docs/fd-symmetric-stencil.md``); the default
``'voxel'`` rule passes every sphere arm and still fails the C3 cylinder
(``docs/fd-mie-validation.md``). Exact volume and matching backends do not establish a
second-order electromagnetic result for any other body.

Out of scope, and refused by name rather than
silently: shapes other than ``Slab``, ``Region``, ``Sphere``, ``Box``, ``Cylinder`` and
``Prism`` (CSG is later in stage 4, a tilted axis stage 5, a raster never), GPU claims (stage
6), ``GaussianBeam``, ``ModeSource``,
mode overlaps (``ModeMonitor``) and decay-driven stopping (``Decay``, ``Run(decay=)``) (stage 7),
non-uniform grids (not in v1).
"""
from ._check import Report, check
from ._lower import Plan, lower, plan
from ._run import run
from .boundaries import (PEC, PMC, Absorber, Absorbing, Axis1D, Bloch, Boundaries, Periodic,
                         PML, Symmetry)
from .diagnostics import (AbsorberProximityWarning, BoundedVoxelWarning, CoarseMeshWarning,
                          GrowingFieldWarning, PMLStabilityWarning, RadiationApproximationWarning, SubpixelAccuracyWarning,
                          UnsettledFieldWarning)
from .geometry import Box, Cylinder, Plane, Prism, Region, Shape, Slab, Sphere, Structure
from .materials import (Conductor, Debye, Dispersive, Drude, Lorentz, Material, MaterialBase,
                        Pole, eps_at)
from .monitors import Decay, Diffraction, FarField, Field, Flux, ModeMonitor, Record
from .results import (DiffractionResult, FarFieldResult, FieldResult, Final, FluxResult, Results,
                      Series, SourceSeries)
from .scene import Domain, Resolution, Run, Scene
from .sources import (TFSF, Band, Dipole, GaussianBeam, InitialField, ModeSource, PlaneWave,
                      Term)

__all__ = [
    'Domain', 'Resolution', 'Run', 'Scene',
    'Boundaries', 'Periodic', 'Bloch', 'Axis1D', 'PEC', 'PMC', 'Symmetry', 'PML', 'Absorber',
    'Absorbing',
    'Material', 'Conductor', 'Dispersive', 'Drude', 'Lorentz', 'Debye', 'eps_at',
    'MaterialBase', 'Pole', 'Shape', 'Slab', 'Sphere', 'Box', 'Cylinder', 'Prism', 'Structure',
    'CoarseMeshWarning', 'AbsorberProximityWarning', 'UnsettledFieldWarning',
    'GrowingFieldWarning', 'PMLStabilityWarning', 'BoundedVoxelWarning', 'SubpixelAccuracyWarning',
    'RadiationApproximationWarning',
    'InitialField', 'Term', 'Record', 'Flux', 'Field', 'Diffraction', 'FarField', 'Plane',
    'ModeMonitor', 'Decay', 'FluxResult', 'FieldResult', 'FarFieldResult', 'DiffractionResult',
    'Band', 'Dipole', 'PlaneWave', 'TFSF', 'Region', 'GaussianBeam', 'ModeSource',
    'check', 'plan', 'lower', 'run',
    'Plan', 'Report', 'Results', 'Series', 'Final', 'SourceSeries',
]

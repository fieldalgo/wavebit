"""Internal normalized general-scene engine on a uniform Yee grid.

The physical API in ``wavebit.fdtd`` lowers Problem/Numerics declarations into
this implementation, converting time, material rates and H fields at its boundary.
Numerical regression instruments may use these private declarations directly.
Importing this module does not load JAX or PyTorch.

The following conventions describe this normalized compatibility API and its kernels:

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
the one asked for: for an interface that does not land on a cell boundary, a convergence study
in ``dx`` is the route to a right answer, or, for a lossless non-dispersive layer, the layer
spelled as the smoothed ``fd.Box`` spanning the other two axes, whose faces are where they were
asked (stage 4, gate G5); a Conductor or Dispersive layer stays a cell-snapped ``fd.Slab``
outside the bounded conductive and dispersive film scopes described below. General lossy-body and
dispersive-interface smoothing remain later C8 work.

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
measured tolerance is refused by the grazing rule (``src/wavebit/fdtd/_prepare/grazing.py``, which quotes the
measurement, judges each wavelength with the CFS shift the layer has there, and judges a face
with ``PML(stabilize=)`` by what its loss returns on each order, computed exactly for this
kernel in ``src/wavebit/fdtd/_prepare/layer_return.py``). A flux
surface -- a Flux or Diffraction plane, a Flux or FarField box -- keeps off every source: one
that cuts a Dipole's stamp, lies on a PlaneWave's sheet (or a box around part of it) or cuts a
TFSF surface is refused, because the 2-cell face would read part of the source
(``src/wavebit/fdtd/_prepare/monitors.py``).

Stage 4's C2 adds the first shape whose edge is **not** a cell boundary::

    fd.Structure(fd.Sphere(radius=0.2, center=(0, 0, 0)), glass)   # subpixel-smoothed

``Slab`` and ``Region`` do not move and every stage-0..3 number is unchanged to the bit; a
``Sphere`` is asked, at each of the six components' **own** Yee points, what fraction of that
voxel it fills and which way its surface faces there, and its material is mixed with whatever
the cell already held by the Farjadpour/Kottke rule (``src/wavebit/fdtd/_prepare/occupancy.py`` produces the
pair, ``src/wavebit/fdtd/_prepare/coefficients.py::mix`` is the rule and carries its citations). The rule is two
isotropic phases meeting at one interface, and C2 refuses every neighbouring case **by name**
rather than answering it approximately (``src/wavebit/fdtd/_prepare/checks.py::check_smoothed``), in six doors
and in the order that function numbers them: a smoothed shape may not share cells with an
``fd.Conductor`` or an ``fd.Dispersive`` (the isolated film scopes below are the only
exceptions); it may not reach a ``Periodic``, ``Bloch`` or ``Axis1D`` face; it may not reach a
face of the box in a way the grid cannot carry whole (it may leave the domain, or the plan may
trim its last half cell at a ``+`` face -- two arms split by cause, two messages); it may not
share cells with an anisotropic ``eps``/``mu``; two smoothed shapes may not put their surfaces
through one voxel except for the flush junction described below; and (from C4) a smoothed
surface may not cut a voxel a cell-snapped structure's face cuts -- which is how a pillar
standing flush on a ``Slab`` substrate is refused rather than read half a cell off. General
CSG remains later work. ``Sphere(range=...)``, ``Cylinder(range=...)``, ``Box(range=...)`` and
``Prism(range=...)`` reserve fixed support boxes for sweeps and compiled-program reuse.
Radius and centre, extrusion height, a box's three sizes and a prism's vertices may
be differentiated with a concrete prebuilt plan. Extrusion axes, rotation angles
and a prism's vertex count remain static. The
range contains the whole body. Traced scenes may contain multiple ranged bodies when
their reserved supports plus interpolation halos are pairwise disjoint; every ranged body
needs isotropic lossless material, and there is no initial field. Concrete
partners without a range -- a cell-snapped Slab substrate running into a PML, a Region, an
unranged body, a conductive or dispersive layer -- may stand beside them when each,
widened by one cell, shares no cell with a traced body's halo; a flush junction among such
partners lowers as in a concrete scene. A structure's ``eps`` and ``mu`` may be traced
together with the geometry, each as one scalar of a plain ``Material``, and so may a
smoothed structure's with concrete geometry: the plan is built from concrete values and
the run reads the traced ones, held at run time to positivity, the time-step bound, the
lifted operator's stability, 8 cells per wavelength and the grazing rule (a substrate
filling a PML may be traced). A medium filling a stabilized PML layer the grazing rule
judges, the background of a scene with a PML matched to it, a PlaneWave or TFSF source or a
lattice FarField, conductivities and pole parameters stay concrete there. See
``docs/guide/fdtd/fd-differentiation.md``.
Every source (Dipole, PlaneWave, TFSF) and monitor (Record, Field, Flux, Diffraction,
FarField) is admitted; the two cell layers where a PlaneWave or TFSF normalisation or a
Diffraction plane reads its one plain medium must lie outside every traced body's reserved
support. All reserved halos must clear the domain boundaries and absorbers. Runtime guards
check every body's actual sizes, containment, polygon simplicity/orientation and
constitutive stability; they do not establish continuum convergence. See the C6 sections
of ``docs/validation/fdtd/fd-subpixel-occupancy.md``.

A Cylinder or unturned Box standing flush on a smoothed, unturned Box face uses
``Structure(junction='split')`` by default: it layers the substrate below the
pillar/background mixture. The standing body may instead request ``'staircase'``
(the medium at each point) or ``'refuse'``. Materials must be isotropic, lossless,
non-dispersive and non-magnetic; the junction rim must clear the substrate's edges.
``check`` prints the choice, its voxel counts and measured accuracy limitations.
See ``docs/validation/fdtd/fd-subpixel-occupancy.md``, The junction. A ranged standing body may be traced
under ``'split'`` -- the metasurface pillar's radius, sizes, height and transverse centre --
while its substrate stays concrete and unranged: the junction's points are recomputed from
the live geometry, the substrate's face must cover the reserved halo and junction rows to
every voxel's edge, its far face must lie more than a cell beyond them, and the runtime
guard keeps the base on the face (vary the height with the centre at ``plane + height/2``).
A concrete ranged pillar's plan belongs to its geometry; differentiation reuses the seed
plan, a concrete sweep plans each geometry.

Stage 4's C4 adds the extruded bodies, smoothed the same way, and renames the cell-snapped box
``fd.Region`` so that ``fd.Box`` can be the body (one name per concept)::

    fd.Structure(fd.Box(size=(0.3, 0.2, 0.1), center=(...), angle=30), glass)  # turned 30 deg
    fd.Structure(fd.Cylinder(radius=0.1, height=0.2, axis='z'), glass)
    fd.Structure(fd.Prism(vertices=((0, 0), (0.2, 0), (0.1, 0.17)), height=0.2), glass)

``angle`` is in degrees and turns a body about its own axis only; a tilted axis is stage 5 and
refused by name. A body **uniform along** an axis over the whole domain -- extruded along it,
or an unturned ``Box`` spanning it -- may run through a ``Periodic``, ``Bloch`` or ``Axis1D``
face, since every layer holds the same section; any other body there is still refused.

The first C8 scope admits one concrete conductive film: a single unturned, unranged
``Box`` with ``Conductor(eps=..., sigma=...)`` in a plain lossless ``Material`` host,
both isotropic and non-magnetic. The box spans exactly two full grid axes with at
least one cell of overhang on each side; its finite thickness is at least ``dx``
and its support clears the normal domain walls and every absorbing layer. No
additional structure, initial field or traced geometry/material value is admitted.
Use the default ``discretisation='voxel'`` and build a new plan when these concrete
inputs change. Tangential conductivity is averaged arithmetically; normal mixed
cells require an induced passive Debye memory even when the two real
permittivities are equal. This does not admit curved conductors, lossy junctions,
general dispersive interface smoothing or conductive shape/material differentiation.
The implementation and its physical-validation record are documented in
``docs/validation/fdtd/fd-conductive-film.md``.

A finite stack of two or more such ``Conductor`` films is also admitted when
every member has the same normal axis and their padded grid supports are
pairwise disjoint. All structures must be members of the stack; a zero-sigma
Conductor is allowed and retains its own reserved interface memory. The
complete ordered stack is part of the prepared plan identity. The plain host,
concrete geometry/material/band values, absorber clearance and no-initial-field
restrictions remain. Every member has a bulk spectral and local relaxation-rate
check. Touching films and stacks mixing declared pole families are outside this
scope. See ``docs/validation/fdtd/fd-conductive-stack.md`` for the validation record.

The next C8 scope admits the same concrete isolated film geometry with
``Dispersive(eps_inf=..., poles=(Drude(...),))`` or one ``Lorentz`` pole instead.
It uses one spatial oscillator: arithmetic response along the film and harmonic
response across it, including the normal resonance induced in a mixed Drude cell.
Only passive parameters and a lossless isotropic non-magnetic host are admitted.
The source band must be concrete; a conservative bound checks its entire positive
frequency interval, including interior Lorentz resonances, and every requested
monitor wavelength is checked separately for phase and skin resolution. These are
bulk mesh floors, not error bounds. See ``docs/validation/fdtd/fd-single-pole-film.md`` for its
validation status. Multiple poles, curved dispersive bodies,
lossy junctions and geometry/material/band differentiation remain outside this scope. C8 as a whole is
not complete, and C7 remains unpassed.

The same isolated geometry also admits exactly one passive ``Debye(tau=...,
delta_eps=...)`` pole. The normal harmonic mixture shifts its relaxation time
and strength; tangential cells retain the arithmetic response. The explicit
split memory includes a local instantaneous permittivity shift and spatial
energy weights. Its conservative time-step policy includes the maximum of
the bulk and actual mixed-cell relaxation rates, which can exceed the bulk
value. Concrete positive source bands and every requested wavelength receive
a bulk spectral-resolution check. Zero strength is allowed; geometry, material
and source-band differentiation and multiple poles remain outside this scope.
See ``docs/validation/fdtd/fd-debye-film.md`` for the bounded contract and validation record.

An isolated film may also contain exactly two passive Debye poles. Its normal
mixture is reduced to two local passive relaxations; equal times and zero
strengths collapse without dropping the reserved memory slots. The time-step
policy includes the sum of both local rates before taking the spatial maximum.
All geometry, host and differentiation restrictions of the single-Debye film
remain. See ``docs/validation/fdtd/fd-two-debye-film.md`` for its validation record.

A finite passive Debye sum with three or more declared poles uses the same
isolated film contract with one reserved memory slot per declared pole. The
normal mixture is reduced as a whole, retaining positive modes and collapsing
only exact equal times or zero increments. The existing one- and two-pole
routes retain their calculations. Mixed pole families, curved interfaces,
dispersive hosts and differentiation remain outside this extension.
See ``docs/validation/fdtd/fd-debye-sum-film.md`` for numerical admission and validation limits.

Two or more separated parallel Debye films may share one plain lossless host.
Their padded supports must be disjoint; each member retains the existing
one-, two- or finite-sum reduction and its declared memory slots. The policy
checks every member and takes the maximum member material rate. Initial
fields, traced inputs, overlapping padded supports, mixed-family stacks and automatic
stopping are refused. See docs/validation/fdtd/fd-debye-stack.md for this bounded contract
and its validation status.

Separated parallel films may also contain exactly one passive Drude or Lorentz
pole per member. Different member kinds are explicitly admitted; within-film
sums and Debye-containing stacks remain outside this extension. Existing
single-film coefficients and energy weights are retained on disjoint padded
supports. The material policy takes the maximum complete member rate and the
bulk mesh check includes every member's interior Lorentz resonances. Inactive
Lorentz frequencies retain the singleton's finite-range and conservative
stability rules. Fixed duration is required; initial fields, traced inputs and
Decay are refused. See ``docs/validation/fdtd/fd-oscillator-stack.md`` for the bounded contract
and current numerical validation status; general C8 remains incomplete.

A finite sum of two or more passive Drude/Lorentz poles, with unequal damping,
uses coupled physical polarizations in the same isolated concrete film. Each
declared pole retains its memory slot; the normal restoring force couples
their sum, with the corresponding cross term in the energy observer. A
conservative whole-band bulk bound includes every active interior resonance.
Mixed Debye/oscillator families, lossy hosts, curved interfaces and
differentiation remain outside this extension. See
``docs/validation/fdtd/fd-oscillator-sum-film.md`` for its contract and validation status.

A concrete isolated film may additionally declare a mixture of passive Debye
and Drude/Lorentz poles. Its hybrid term preserves the bilinear Debye and
centered oscillator response symbols, coupled by the normal harmonic rule.
It carries four arrays per declared pole and uses an integer-time passive
storage distinct from the single-family observers. The stability policy
uses physical high-frequency permittivity and retains a conservative Debye
penalty. No lossy host, curved interface, junction or differentiation support
is implied. See ``docs/validation/fdtd/fd-hybrid-film.md`` for the precise admitted scope and
validation record; the general C8 work remains incomplete.

Stage 4's C3 adds a second material discretisation, **opt in and per structure**::

    fd.Structure(fd.Sphere(radius=0.2), glass, discretisation='stencil')

``'voxel'`` is the default and is every scene's behaviour above, to the bit; ``'stencil'``
asks instead for the symmetric extended-stencil operator ``docs/design/fd-symmetric-stencil.md``
derives -- exact edge and dual-face interface constraints, solved once on the host at
``fd.plan`` time and never inside a trace, whose rows couple each edge to every edge within
two cells. ``check`` prints it. It has its own doors and not one of them falls back to the
other rule: only ``fd.Sphere`` itself, isotropic, lossless and non-magnetic, with concrete
geometry and material values at plan time **and** at lowering; the only structure in the
scene, in a vacuum background, in a real run with no initial field; no wall, absorbing
layer, source or monitor surface inside the region the operator owns (its support plus
three nodes a side); under a declared preparation memory bound (``memory_bound=`` raises
it); and an operator that passes its own consistency, energy, halo and time-step
certificates (``src/wavebit/fdtd/_prepare/stencil.py``). ``fd.run(scene, plan=p)`` takes the operator ``p``
already carries rather than solving it again.

Smoothed scenes emit ``SubpixelAccuracyWarning`` during planning, also shown by ``check``,
with the benchmark behind the structure's discretisation. ``'stencil'`` passes C3 (the
epsilon-12 sphere converges at order 2.244; ``docs/design/fd-symmetric-stencil.md``); the default
``'voxel'`` rule passes every sphere arm and still fails the C3 cylinder
(``docs/validation/fdtd/fd-mie-validation.md``). Exact volume and matching backends do not establish a
second-order electromagnetic result for any other body.

``Run(decay=Decay(threshold=..., window=...))`` optionally stops concrete CPU forward
NumPy/JAX/Torch runs at complete state-amplitude windows, with an explicit steps/periods/time
hard cap. InitialField, Dipole, PlaneWave and axis-aligned TFSF excitation are supported; AD and
enclosing JAX traces are refused. Incident columns and layered drivers are monitored. Every evolving field, CPML and material array has its own peak;
this heuristic is not a spectral-error or future-echo guarantee. Results retain only the
actual prefix and report cap/reason in ``diagnostics['stopping']``. Both values are required.

Out of scope, and refused by name rather than
silently: shapes other than ``Slab``, ``Region``, ``Sphere``, ``Box``, ``Cylinder`` and
``Prism`` (CSG is later in stage 4, a tilted axis stage 5, a raster never), GPU claims (stage
6), ``GaussianBeam``, ``ModeSource``,
mode overlaps (``ModeMonitor``) (stage 7),
non-uniform grids (not in v1).
"""
from wavebit.fdtd._prepare.report import Report
from wavebit.fdtd._prepare.report import check
from wavebit.fdtd._prepare import Plan
from wavebit.fdtd._prepare import lower
import wavebit.fdtd._prepare.plan as plan
from wavebit.fdtd._execution import run
from wavebit.fdtd._legacy_boundaries import PEC
from wavebit.fdtd._legacy_boundaries import PMC
from wavebit.fdtd._legacy_boundaries import Absorber
from wavebit.fdtd._legacy_boundaries import Absorbing
from wavebit.fdtd._legacy_boundaries import Axis1D
from wavebit.fdtd._legacy_boundaries import Bloch
from wavebit.fdtd._legacy_boundaries import Boundaries
from wavebit.fdtd._legacy_boundaries import Periodic
from wavebit.fdtd._legacy_boundaries import PML
from wavebit.fdtd._legacy_boundaries import Symmetry
from wavebit.fdtd.diagnostics import AbsorberProximityWarning
from wavebit.fdtd.diagnostics import BoundedVoxelWarning
from wavebit.fdtd.diagnostics import CoarseMeshWarning
from wavebit.fdtd.diagnostics import GeometryClippedWarning
from wavebit.fdtd.diagnostics import GrowingFieldWarning
from wavebit.fdtd.diagnostics import PMLStabilityWarning
from wavebit.fdtd.diagnostics import RadiationApproximationWarning
from wavebit.fdtd.diagnostics import SubpixelAccuracyWarning
from wavebit.fdtd.diagnostics import UnsettledFieldWarning
from wavebit.fdtd._legacy_geometry import Box
from wavebit.fdtd._legacy_geometry import Cylinder
from wavebit.fdtd._legacy_geometry import Plane
from wavebit.fdtd._legacy_geometry import Prism
from wavebit.fdtd._legacy_geometry import Region
from wavebit.fdtd._legacy_geometry import Shape
from wavebit.fdtd._legacy_geometry import Slab
from wavebit.fdtd._legacy_geometry import Sphere
from wavebit.fdtd._legacy_geometry import Structure
from wavebit.materials._time_domain import Conductor
from wavebit.materials._time_domain import Debye
from wavebit.materials._time_domain import Dispersive
from wavebit.materials._time_domain import Drude
from wavebit.materials._time_domain import Lorentz
from wavebit.materials._time_domain import Material
from wavebit.materials._time_domain import MaterialBase
from wavebit.materials._time_domain import Pole
from wavebit.materials._time_domain import eps_at
from wavebit.fdtd._legacy_monitors import Decay
from wavebit.fdtd._legacy_monitors import Diffraction
from wavebit.fdtd._legacy_monitors import FarField
from wavebit.fdtd._legacy_monitors import Field
from wavebit.fdtd._legacy_monitors import Flux
from wavebit.fdtd._legacy_monitors import ModeMonitor
from wavebit.fdtd._legacy_monitors import Record
from wavebit.fdtd._legacy_results import DiffractionResult
from wavebit.fdtd._legacy_results import FarFieldResult
from wavebit.fdtd._legacy_results import FieldResult
from wavebit.fdtd._legacy_results import Final
from wavebit.fdtd._legacy_results import FluxResult
from wavebit.fdtd._legacy_results import Results
from wavebit.fdtd._legacy_results import Series
from wavebit.fdtd._legacy_results import SourceSeries
from wavebit.fdtd._scene import Domain
from wavebit.fdtd._scene import Resolution
from wavebit.fdtd._scene import Run
from wavebit.fdtd._scene import Scene
from wavebit.fdtd._legacy_sources import TFSF
from wavebit.fdtd._legacy_sources import Band
from wavebit.fdtd._legacy_sources import Dipole
from wavebit.fdtd._legacy_sources import GaussianBeam
from wavebit.fdtd._legacy_sources import InitialField
from wavebit.fdtd._legacy_sources import ModeSource
from wavebit.fdtd._legacy_sources import PlaneWave
from wavebit.fdtd._legacy_sources import Term

__all__ = [
    'Domain', 'Resolution', 'Run', 'Scene',
    'Boundaries', 'Periodic', 'Bloch', 'Axis1D', 'PEC', 'PMC', 'Symmetry', 'PML', 'Absorber',
    'Absorbing',
    'Material', 'Conductor', 'Dispersive', 'Drude', 'Lorentz', 'Debye', 'eps_at',
    'MaterialBase', 'Pole', 'Shape', 'Slab', 'Sphere', 'Box', 'Cylinder', 'Prism', 'Structure',
    'CoarseMeshWarning', 'AbsorberProximityWarning', 'UnsettledFieldWarning',
    'GrowingFieldWarning', 'PMLStabilityWarning', 'BoundedVoxelWarning', 'SubpixelAccuracyWarning',
    'RadiationApproximationWarning', 'GeometryClippedWarning',
    'InitialField', 'Term', 'Record', 'Flux', 'Field', 'Diffraction', 'FarField', 'Plane',
    'ModeMonitor', 'Decay', 'FluxResult', 'FieldResult', 'FarFieldResult', 'DiffractionResult',
    'Band', 'Dipole', 'PlaneWave', 'TFSF', 'Region', 'GaussianBeam', 'ModeSource',
    'check', 'plan', 'lower', 'run',
    'Plan', 'Report', 'Results', 'Series', 'Final', 'SourceSeries',
]

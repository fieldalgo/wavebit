# How the FDTD solver models a problem

`wavebit.fdtd` solves Maxwell's equations in the time domain on a uniform Yee grid. This page
explains what happens to a declared problem on the way to a result: how the domain becomes
cells, how each structure is drawn into those cells, what the boundaries, sources and monitors
are on the grid, which accuracy rules are enforced before a run starts, and what the solver does
not do yet. It is the background for reading a `check` report and for deciding whether a result
can be trusted. For a first run, start with [the runnable pulse example](../../getting-started/fdtd.md);
for the declaration API, see [domains, sources and observations](general-fdtd.md) and
[the reference](../../reference/fdtd/index.md).

## At a glance

| Piece | What it is on the grid |
| --- | --- |
| `UniformGrid(spacing=dx)` | cubic cells of one size `dx` on all three axes |
| `Domain(size, center)` | the physical region, rounded **up** to whole cells about its centre |
| `Slab`, `Region` | **cell-snapped** structures: whole cells, faces moved to the nearest cell boundary |
| `Box`, `Sphere`, `Cylinder`, `Prism` | **smoothed** bodies: each boundary cell gets a mix weighted by how much of it the body fills |
| `Open()` face | a PML added **outside** the physical region, matched to the medium that reaches the face |
| `Periodic()`, `Bloch(...)` | the cell repeats; `Bloch` adds a phase and makes the run complex |
| `PEC()`, `PMC()`, `Symmetry(parity)`, `Axis1D(k, k_max)` | conducting walls, a mirror plane, and an exactly differentiated invariant axis |
| `Numerics(courant=0.5)` | time step `dt = courant · dx / (c·√3)` |
| `Record` | the time series of one component at its nearest Yee point |
| `Field`, `Flux`, `Diffraction`, `FarField` | running Fourier transforms of the fields, accumulated during the run |

Runs use double precision with NumPy, JAX or PyTorch on CPU. A restricted Torch
CUDA route admits fixed-duration voxel scenes with Periodic/Open boundaries,
PML, nondispersive Material/ConductiveMaterial, cell-snapped Slabs, PlaneWave,
fixed-center electric/magnetic Dipole (`bspline` or `tent`), or initial-field
excitation, and Record/Field/Flux observations. Dipole center and stamp are
static; `current_moment` uses A·m/V·m rather than plane-wave V/m amplitude. Pass
`backend='torch', device='cuda:0'`; see the [CUDA scope and validation](../../validation/fdtd/general-cuda.md).
JAX and PyTorch runs are differentiable; see [Differentiating a run](#differentiating-a-run).

For CUDA, initial E/H patterns remain static host declarations (NumPy arrays,
CPU tensors without gradients, or host callables). Direct CUDA or differentiable
tensor patterns are not admitted; the separate `InitialField.amplitude` remains
a live differentiable input.

## Units and conventions

| Quantity | Unit |
| --- | --- |
| lengths, positions, vacuum wavelengths | the problem's `Units(length="m" \| "mm" \| "um" \| "nm")` |
| times, pulse delays, `Duration(seconds=...)` | seconds |
| `GaussianPulse(frequency_sigma_hz=...)` | Hz |
| angles: `PlaneWave(theta, phi)`, `Box`/`Prism(angle)` | radians |
| electric field (`Record`, final fields), plane-wave and TFSF amplitudes | V/m |
| magnetic field | A/m |
| `Field` phasors | V·s/m and A·s/m (the Fourier transform of the field time series) |
| `Flux` | J/Hz, a two-sided spectral energy density (J/(Hz·m) with one `Axis1D` face) |
| `ConductiveMaterial(conductivity=...)` | S/m |
| `Dipole(current_moment=...)` | A·m (electric), V·m (magnetic equivalent) |

The time convention is `exp(-iωt)`: a wave travelling toward `+x` varies as `exp(+ikx)`, and a
`Bloch` phase `φ` means the field at `x + L` is `exp(+iφ)` times the field at `x`. Spectral
monitors transform with `∫ f(t) exp(+i2πft) dt` over the finite run, with no window. Internally
the engine works in normalized units (`c = 1`, H scaled by the vacuum impedance); every public
input and output is converted at the boundary of the module.

## The grid

**One spacing on every axis.** `UniformGrid(spacing=dx)` makes cubic cells. Cell `i` along x
covers `[x_lo + i·dx, x_lo + (i+1)·dx)`, and the same on y and z. There is no non-uniform grid
and no per-axis spacing.

**The domain is rounded up to whole cells about its centre.** A domain of size `s` gets
`ceil(s / dx)` cells (up to round-off), and its low corner is `center - n·dx/2`. `check` reports
the requested size, the size actually used (`resolved["physical_domain"]`) and the cell counts
(`resolved["grid_shape"]`).

**The six field components sit at staggered points (the Yee cell).** Within a cell, offsets in
half cells along `(x, y, z)` are

| Component | Offset | Component | Offset |
| --- | --- | --- | --- |
| Ex | (1, 0, 0) | Hx | (0, 1, 1) |
| Ey | (0, 1, 0) | Hy | (1, 0, 1) |
| Ez | (0, 0, 1) | Hz | (1, 1, 0) |

so each electric component sits half a cell along its own axis and each magnetic component half
a cell along the other two. This matters whenever a position is read back: a `Record` samples its
component at that component's nearest point and reports both the requested and the sampled
coordinates, and `Field` returns each component on its own points. Nothing is interpolated.

**Absorbing layers are added outside.** Each `Open()` face gets `PML(cells=12)` cells (the
default) beyond the physical domain, so adding PML cells never moves or shrinks the region you
declared (`resolved["padding_cells"]`, `resolved["allocated_domain"]`). Explicit
`Numerics(absorber=fdtd.Absorber(cells=...))` instead chooses graded conductivity
padding for supported media that PML cannot match; see [its attenuation floor and
scope](stopping-and-absorbers.md).

## Time stepping

E and H are updated alternately (leapfrog): E at whole steps, H half a step earlier. The step is
`dt = courant · dx / (c·√3)` with `Numerics(courant=0.5)` by default; `courant` must lie in
`(0, 1)`, the three-dimensional stability limit in vacuum. Materials can tighten that limit: a
medium with `ε·μ < 1`, or a dispersive pole, lowers the largest stable step, and a scene whose
`dt` would exceed it is refused with the bound in the message (lower `courant` then).
`check` reports `resolved["dt_seconds"]` and `resolved["steps"]`.

`Duration` takes exactly one of `steps`, `seconds`, or `periods` with `source="name"` (that
many periods of the named source's centre wavelength). By default a run lasts that duration.
An explicit `Numerics(decay=fdtd.Decay(threshold=..., window_seconds=...))` makes it a hard cap
for concrete forward CPU execution and may stop at an earlier completed decay window.
See [automatic stopping](stopping-and-absorbers.md) for supported sources, actual-prefix
results and why this policy refuses differentiation. End-of-run decay warnings remain
separate from convergence guarantees (see [Warnings](#warnings)).

## Structures: cell-snapped and smoothed

A scene is a background material plus an ordered tuple of named structures,
`Structure(name, shape, material)`. **Structures are painted in order: where two overlap, the
later one wins.** Each shape is drawn into the grid in one of two ways, and the difference is the
most important accuracy decision in a scene.

### Cell-snapped shapes: `Slab` and `Region`

- `Slab(z=(lo, hi))` is a layer: one axis bounded, the other two filling the domain.
- `Region(size=(sx, sy, sz), center=(cx, cy, cz))` is an axis-aligned block.

Both fill **whole cells**. Each face moves to the nearest cell boundary, `i = round((x - x_lo) / dx)`,
and every cell between the moved faces holds the material entirely, for all six components. So a
face is moved by up to half a cell, which is a first-order error in `dx`, and any two spans that
round to the same cells give identical results. At `dx = 0.025` µm, 0.317 µm and 0.325 µm films
run bit for bit alike:

```
cell boundaries (dx = 0.025):  0.275    0.300    0.325    0.350
                                 |--------|--------|--------|
requested interface 0.317 ----------------------^  runs at 0.325
```

`check` reports every moved face as a warning issue with code `geometry.cell_snapped`, the
requested span in `expected` and the cell boundaries it runs at in `actual`; `report.ok` stays
true. Choose `dx` so that layer thicknesses are whole numbers of cells, and check convergence by
refining `dx`.

Even on a cell boundary a snapped face is not exact for every component. The tangential electric
field on the face's node plane takes the material over its whole surrounding cell, so for that
field a `Slab` sits half a cell low as a whole ([G5 in smoothed occupancy](../../validation/fdtd/fd-subpixel-occupancy.md)).
Moving a layer as a whole changes the phases of `r` and `t` measured at the declared planes, not
the efficiencies; when phases must match another solver, a smoothed `Box` layer puts its faces
where they were declared.

Cell-snapped shapes support these scopes without a smoothing declaration:

- **Continue into an absorbing face.** A `Slab` across the axis of an `Open()` face (a substrate,
  a superstrate) continues through the PML, and that PML is matched to the slab's material.
- **Hold any supported material,** including `ConductiveMaterial` and `DispersiveMaterial`.
- **Run the whole period.** A `Region` that spans a `Periodic()` axis is clipped to the cell,
  which is how a lamellar grating tooth is written. A `Region` that crosses a periodic seam
  without spanning the period is refused, because its image on the other side would be missing.

Their geometry is static: a size, centre or span is not differentiable (the material value is).

### Smoothed bodies: `Box`, `Sphere`, `Cylinder`, `Prism`

- `Box(size, center, angle=0, axis="z")` — a rectangular block, turned by `angle` radians
  counter-clockwise about `axis`.
- `Sphere(radius, center)`.
- `Cylinder(radius, height, axis="z", center)`.
- `Prism(vertices, height, axis="z", center, angle=0)` — a polygon extruded along `axis`.
  `vertices` are counter-clockwise `(u, v)` offsets from `center` in the plane across the axis:
  `(x, y)` for `"z"`, `(y, z)` for `"x"`, `(z, x)` for `"y"`. Non-convex outlines are allowed;
  a clockwise or self-crossing outline is refused.

A smoothed body does not snap. At each of the six components' own Yee points the engine computes
the fraction of that component's cell the body fills and the orientation of the surface there,
and mixes the body's material with whatever the cell already held using the anisotropic
effective-medium rule of Farjadpour et al. (*Opt. Lett.* **31**, 2972, 2006) and Kottke,
Farjadpour and Johnson (*Phys. Rev. E* **77**, 036611, 2008). A face between cell boundaries
therefore stays where it was declared: the 0.317 and 0.325 films above give different answers
when written as a `Box`.

The rule has limits, each enforced by `check`:

| Rule | Why |
| --- | --- |
| The smallest extent the body bounds must be at least 4 cells; 4 to 8 cells warns | across fewer cells, where the mixed cells fall decides the answer more than the body does |
| Material: `Material` with a scalar permittivity and permeability; no per-axis values | the mixing rule is applied per component and assumes an isotropic body |
| No `ConductiveMaterial` or `DispersiveMaterial`, except a flat film spanning the cell | lossy and dispersive mixing is validated only for those film scopes |
| Must not touch an `Open()` face unless an unturned, transversely uniform `Box` explicitly declares `Structure.continuation` | finite-body continuation is not inferred; see [smoothed substrates](substrate-gradients.md) |
| Must not reach a `Periodic()` face, unless it runs the whole period reaching **more than** one cell past both faces | otherwise part of it would be lost at the seam |
| Two smoothed surfaces may not pass through the same cell | the mixing rule describes two media and one surface per cell; separate them by a cell or write one body |
| A smoothed surface may not cut a cell that a cell-snapped face also cuts | the cell would hold three media (see below) |

A smoothed body may sit **inside** a cell-snapped structure, or on top of another smoothed body:
a `Cylinder` or unturned `Box` standing flush on the face of an unturned smoothed `Box` is handled
by `Numerics(junction="split")`, the default (`"staircase"` and `"refuse"` are the alternatives).

The global `Numerics.junction` policy is read only by the standing body of each
supported flush pair, using the engine's role selection after substrate continuation.
The substrate and unrelated structures retain their ordinary rule, regardless of listing
order. `"staircase"` selects the owning medium at junction points; `"refuse"` rejects the
actual shared interface. Invalid values and non-default policies with no supported flush
pair are errors at `numerics.junction`. The default `"split"` needs no junction, and
prepared traced junction geometry remains supported only with `"split"`. Prepared reuse
retains the recorded policy: a traced standing body explicitly rejects `"staircase"`
at `numerics.junction`, while unrelated traced bodies leave a static staircase pair
unchanged. `"refuse"` rejects the pair during preparation; it cannot prepare a split
fallback.

Invalid selector values are reported with the public `Numerics(junction=...)` or
`Numerics(discretisation=...)` spelling and the supported choices. `check` returns
an `invalid_input` issue and `prepare` raises `ValidationError` with the same issue:
`path` identifies the selector and `actual` retains the supplied value.

`check(...).resolved["numerical_policies"]` and `prepare(...).report.resolved` record
both requested selectors and actual per-structure application. The `discretisation`
rows report `stencil` only for a prepared engine operator and `voxel` for an ordinary
built-in smoothed body. `rule=None` with `whole_cell_fill` means Slab/Region retain
whole-cell fills; `custom_shape_protocol` means the custom Shape owns its fill semantics.
The built-in public `Slab` and `Region` classes are recognised exactly; user subclasses
retain the custom protocol classification in check, prepare, validate, and run provenance.
The `junction` rows report a rule only for a recorded standing body, distinguish
`junction_face` from `no_resolved_junction`, and list each pair by stable structure names
and axis. A default split request with no pair therefore reports no application.
Invalid/unsupported requests return issues rather than successful application rows.
`Prepared.validate(problem)` refreshes this report for an accepted concrete reuse;
changed structure targets or invalid ranged contacts are refused, with no stale resolved
application rows. The preparation's policy records remain an immutable snapshot.

`Result.provenance["numerical_policies"]` carries the same resolved policy schema,
computed from that run's scene and executed plan, including when a runtime guard
invalidates the fields. It describes the applied operator, not numeric admission.
Reports expose `capabilities["runtime_validation"]`: `"concrete_inputs"` when the
lowered inputs can be checked concretely, or `"deferred"` when traced inputs cannot
be read on the host. An otherwise successful deferred `Prepared.validate` report
certifies structure and policies; use the run's `validity.checks["runtime_stability"]`
for its live numeric verdict. A concrete refused value returns issues and no
resolved policy rows. Validation and execution do not mutate earlier reports or
result policy records.

**Pillars on a substrate.** Because of the last rule, a smoothed pillar cannot stand flush on a
`Slab` substrate. The choices are:

1. a fixed, transversely uniform smoothed `Box` substrate with
   `Structure(..., continuation=('z-',))`, with the pillar flush on its interior surface;
2. a finite smoothed `Box` substrate that does not reach the `Open()` face, with the pillar flush
   on it (the junction rule above);
3. a cell-snapped `Region` pillar on a `Slab` substrate (rectangular, faces snapped);
4. a smoothed pillar raised at least half a cell above the `Slab` face.

The first choice continues a half-space substrate through the absorber. See
[the physical example](substrate-gradients.md) for its geometric restrictions and the fixed
`PML.reference_media` needed when differentiating the substrate material.

**Accuracy of smoothing.** The default rule, `Numerics(discretisation="voxel")`, converged at
order 2.02 on a controlled glass-sphere study, but has not passed the curved-interface accuracy
gate in general; every run with a smoothed body emits `SubpixelAccuracyWarning` saying so. The
alternative `discretisation="stencil"` passed that gate for a sphere (order 2.24 for an
`ε = 12` sphere) but applies to `Sphere` only, takes plain numbers (no derivatives), and needs a
`PlaneWave` or `TFSF` source. The measurements are in
[the Mie validation](../../validation/fdtd/fd-mie-validation.md) and
[smoothed occupancy](../../validation/fdtd/fd-subpixel-occupancy.md).

The scene-wide `Numerics(discretisation="stencil")` selector is forwarded only to
smoothed bodies. `Slab` and `Region` retain their whole-cell fills. `Box`, `Cylinder`,
`Prism`, other unsupported shapes, and scenes without an applicable `Sphere` are
refused at `numerics.discretisation`. Only the exact built-in `Sphere`, `Slab`, and
`Region` types pass this selector gate; custom shapes and subclasses may change the
fill protocol and are refused. This does not extend the stencil's scope:
it still requires one lone sphere in vacuum and concrete, non-differentiated values;
adding a slab or region does not make a multi-structure stencil scene supported.

### Stencil preparation memory

`Numerics(discretisation="stencil", memory_bound=2 * 1024**3, ...)` sets the
host preparation prediction limit in bytes (2 GiB in this example). Omit
`memory_bound`, or use `None`, to retain the engine's default of 6 GiB. A positive
finite real number is required; booleans, zero, negative and non-finite values
are refused at `numerics.memory_bound`. A value without `discretisation="stencil"`
and an applicable `Sphere` is also refused, because no preparation would read it.

The guard compares the existing prediction with the limit **before** the sparse
assembly and solve. It is an estimate, not a cap on total process memory or a
guarantee against exhausting memory. Coarsen `UniformGrid(spacing=...)`, shrink
the sphere, or explicitly adjust the bound to the available resources. Raising
it does not extend the stencil's physical scope: the current operator requires
one sphere in vacuum. For multiple structures, use
`Numerics(discretisation="voxel")` or reduce the scene to the supported single
sphere. Numerical policy belongs to `Numerics`, not `Structure`.

## Materials

| Model | Use |
| --- | --- |
| `Material(epsilon, mu=None)` | real, positive, non-dispersive; a scalar or one value per axis `(x, y, z)` |
| `ConductiveMaterial(epsilon, conductivity, mu=None)` | a constant conductivity in S/m |
| `DispersiveMaterial(epsilon_inf, poles, mu=None)` | passive `Drude`, `Lorentz` and `Debye` poles with SI rates |

Refused, each with a message: a complex permittivity constant (it is right at one frequency only
— use a pole model), a full 3×3 tensor, a tabulated or evaluated material. The background
material fills every cell no structure claims and continues into the PML on every `Open()` face
it reaches.

## Boundaries

Every face is declared; there is no default. `Boundaries(x=..., y=..., z=...)` takes one face
type per axis or a `(low, high)` pair.

| Face | Meaning |
| --- | --- |
| `Open()` | outgoing: a PML outside the physical domain, matched to the uniform medium that reaches the face. Only uniform media may reach it — the background, or a `Slab` across that axis; a finite body touching it, or a `Slab` crossing an orthogonal `Open()` face, is refused. |
| `Periodic()` | both faces wrap with no phase; normal incidence in a periodic cell. |
| `Bloch(phase=φ)` or `Bloch(k=k)` | both faces wrap with a phase; the run becomes complex (twice the memory and work). `Bloch.from_source(source, units=...)` writes the faces an oblique `PlaneWave` needs. |
| `PEC()` / `PMC()` | tangential E / tangential H vanishes on the wall. A structure reaching past one is clipped, with `GeometryClippedWarning`. |
| `Symmetry(parity=±1)` | a mirror plane; `+1` makes tangential E even across it (a PMC-like plane), `-1` odd. |
| `Axis1D(k=..., k_max=...)` | a one-cell invariant axis whose derivative is exactly `ik`: the uniform axis of a 2-D conical grating, without the one-cell difference's artificial dispersion. |

`Numerics(absorber=PML(...))` sets every `Open()` face's layer: `cells=12` (fewer is refused
unless `allow_thin=True`), `reflection=1e-8`, `wavelength` (the sources' shared centre wavelength
when omitted) and `stabilize`, an extra loss that stops a bound mode next to the face from growing
(see `GrowingFieldWarning` below).

### Acting on a grazing refusal

`check`, `prepare` and `run` name the worst refused incidence or diffraction order,
its face, wavelength and return criterion, and list the other refused rows. The
public control is `Numerics.absorber`: its numerical settings apply to **every
Open face**, while each face keeps its physical or fixed reference medium.

When the diagnostic prints an `updated = numerics.replace(...)` candidate, use
the existing `numerics` value. Its nested `absorber.replace(...)` preserves
`wavelength`, `reference_media` and `allow_thin`. A duration change includes the
actual source name in `fdtd.Duration(periods=..., source=...)`. Stronger reflection
targets, deeper layers and already sufficient durations are retained. Increasing
the absorber depth adds padding **outside** the physical region automatically;
keep `problem.scene.domain` unchanged. More cells and periods cost memory and time.

Apply one candidate, then run `fdtd.check(problem, numerics=updated)` again. A
candidate for one face is not a guarantee for all faces, orders, media and
wavelengths. If refused faces have different resolved reference media, the
message asks for a common absorber change and a full recheck without offering a
numeric combination. Reducing `stabilize` can let a bound mode grow again; inspect
decay as well. Below the measured grazing floor, changing absorber controls or
duration is not an accepted remedy: change the period, angle or wavelengths.
These messages do not relax the return tolerance or the measured support limits.

## Sources

`Problem.sources` accepts `PlaneWave`, `Dipole` and `TFSF` declarations with
unique, nonempty string names and a `GaussianPulse`. `check` and
`Prepared.validate` report malformed declarations at `sources[i]`, invalid or
duplicate names at `sources[i].name`, and invalid pulses at
`sources.<name>.pulse`. `prepare`, `run` and `Prepared.run` raise the same
structured error before reading a duration's source or a reuse signature.
The pulse's numerical leaves keep their existing concrete and runtime checks.

`TFSF.box` must be a cell-snapped `fdtd.Region`, not a smoothed `fdtd.Box` or a
plane. Its type is checked at `sources.<name>.box` before reading the box bounds;
a valid Region still has to fit inside the physical domain.

Every source carries a `GaussianPulse(wavelength, frequency_sigma_hz, cutoff=1e-3)`: a sine at
the centre frequency `f0 = c / wavelength` under a Gaussian envelope whose **amplitude spectrum**
has standard deviation `frequency_sigma_hz`. The pulse starts at `cutoff` of its peak, and the
spectrum falls to `cutoff` at `f0 ± k·σ` with `exp(-k²/2) = cutoff`; the short-wavelength edge of
that band is the wavelength the resolution rule reads (next section). `amplitude`, `phase` and
`delay` (seconds) shift and scale the waveform.

`Plane` names its normal with one scalar, for example `fdtd.Plane(z=.3)`.
`Field` and `Flux` also accept finite transverse spans, such as
`fdtd.Plane(z=.3, x=(-.05, .05))`. `PlaneWave` and `Diffraction` require the
whole transverse cell: omit both transverse spans. Any explicit span, even
one matching the cell boundaries, is unsupported and produces a structured
`plane_span` issue at `sources.<name>.plane` or `outputs.<name>.plane`.
`check` reports the issue; `prepare` and `run` raise `UnsupportedFeatureError`.

| Source | What it does |
| --- | --- |
| `PlaneWave(name, plane, pulse, direction="-z", theta=0, phi=0, polarization="s")` | a one-way sheet at the node plane nearest `plane`, launching toward `direction`. `theta` is the polar angle from the propagation axis and `phi` the azimuth from the first transverse axis (x for a z wave); `"s"` has E perpendicular to the plane of incidence, `"p"` in it. The transverse faces must be `Periodic()` (normal incidence) or the `Bloch` faces from `Bloch.from_source`; the transverse wavevector is fixed at the pulse's centre wavelength, so an oblique broadband pulse is not constant-angle illumination. |
| `Dipole(name, center, pulse, component="Ey", kind="electric", current_moment=1)` | a point **current** (A·m), or with `kind="magnetic"` its magnetic equivalent (V·m), spread over a smooth stamp (`"bspline"`, the default, or `"tent"`, each with a half-width of two cells) that keeps the moment and its centroid exact wherever the centre sits. It is not a dipole moment in C·m. |
| `TFSF(name, box, pulse, direction="-z", polarization="x")` | a plane wave inside a `Region` box only (total field inside, scattered field outside), for a finite scatterer. Axis directions only; the two faces along its axis must be `Open()`. |
| `InitialField(E, H, amplitude=1)` | the field the run starts from: E at `t = 0` and H at `t = -dt/2`, given as arrays or as callables evaluated at each component's own Yee points. |

`PlaneWave.plane` must be an `fdtd.Plane`, for example `fdtd.Plane(z=.5)`;
a scalar `.5` does not specify an axis. `check` reports invalid types at
`sources.<name>.plane`, and `prepare` and `run` raise `ValidationError` at the
same path before reading any plane coordinates. A valid Plane must still lie
inside the physical domain.

## Outputs

Monitors are declared by name in `Outputs(monitors=...)` and read with `result[name]`; each
frequency-domain monitor accumulates its transforms during the run at the wavelengths it lists.

For finite monitor apertures, `Field` returns each component's Yee points inside the
requested transverse intervals, including endpoints; its array axes are frequency,
z, y, x, and its coordinate arrays describe those samples. `Flux` instead snaps each
transverse endpoint to the nearest node and integrates the surface between those
nodes, using midpoint samples and half weights at end nodes. Its area can therefore
differ from the requested geometric area for a nonaligned aperture. No interpolation
or fractional edge-cell integration is performed. A finite Flux span must cover at
least one cell and end at or before the last stored node (the high domain face is
not stored); a Field span must contain a point of every requested component.
Both refuse requested transverse intervals outside the physical domain, before
clipping or snapping, with `monitor_extent` at `outputs.<name>.plane.<axis>`.
Reversed or zero-width intervals are refused when constructing `Plane`.

| Monitor | Returns |
| --- | --- |
| `Record(name, at, component)` | the time series of one component at its nearest Yee point, in V/m or A/m, with the sample time axis in seconds |
| `Field(name, wavelengths, plane=... or box=..., components=...)` | Fourier phasors of the components on their own points (V·s/m, A·s/m) |
| `Flux(name, wavelengths, plane=... or box=...)` | the net Poynting energy through a plane (along its `+` normal) or out of a closed box, per unit frequency: a pulse energy density in J/Hz, not watts. A closed box must enclose or exclude each structure whole. |
| `Diffraction(name, plane, wavelengths, source, side, orders=None)` | the orders leaving a periodic cell on one side (`"reflect"` or `"transmit"`), as complex s/p amplitudes in each order's own frame and efficiencies, normalized by the named `PlaneWave`'s own measured incident field. Requires exactly one `PlaneWave`. `reference_plane` and `incident_plane` move the phase references; `raw=True` keeps the unnormalized spectrum. Wavelengths outside the pulse's usable band are returned as NaN, with `validity.checks["source_reference"]` false there. |
| `FarField(name, wavelengths, box, propagation="continuum")` | the radiation of an open scene: a closed box around everything, transformed to infinity. `intensity(theta, phi)` takes radians. Periodic cells are refused; their far field is the diffraction orders. `propagation="yee"` uses the lattice's own dispersion. |
| `Outputs(final_fields=True)` | the E and H arrays at the end of the run, with each component's coordinates |

(fdtd-model-accuracy-rules)=
## Accuracy rules enforced before a run

`check`, `prepare` and `run` share these doors; a refusal names the setting to change.

- **8 cells per wavelength in every material.** At the shortest wavelength of the source band (and
  at every `Diffraction` wavelength), each medium must be sampled at 8 cells per wavelength
  *inside that medium*, `λ / (n·dx) ≥ 8`. Below 8 the scene is refused; between 8 and 12 a
  `CoarseMeshWarning` says so. A structure of index `n` needs `n` times finer cells than vacuum.
- **4 cells across a smoothed body**, as in the table above.
- **The time step** stays inside the stability bound of the scene's materials.
- **The PML** is at least 12 cells unless `allow_thin=True`.
- **Placement.** Sources and monitors must lie inside the physical domain; a closed `Flux` box and
  a `FarField` box keep clear of structures as their rows above describe.

Passing these rules is an admission check, not a convergence result. The answer still has to be
refined in `dx` and in duration until the quantity you need stops changing.

## Warnings

Warnings describe the answer, not the spelling: the run produces numbers that are probably worse
than expected. `check` lists the planning ones in `report.capabilities["warnings"]`, and `run`
emits them as Python warnings from `wavebit.fdtd.diagnostics`.

| Warning | Meaning |
| --- | --- |
| `CoarseMeshWarning` | 8 to 12 cells per wavelength in a medium, or a smoothed body 4 to 8 cells across |
| `SubpixelAccuracyWarning` | the scene has a smoothed body; the default rule's accuracy gate is incomplete |
| `AbsorberProximityWarning` | a structure lies within half a wavelength (in its own medium) of an absorbing face, whose PML is matched to the medium that reaches the face rather than to the structure's near field. A `Slab` that continues into the absorber, such as a substrate, does not warn: its PML is matched to it. |
| `GeometryClippedWarning` | a structure reaches past a `PEC` or `PMC` wall and loses that part |
| `UnsettledFieldWarning` | a monitored region still held field at the end; its spectrum includes the truncation — run longer |
| `GrowingFieldWarning` | the field still grows after every source ended; a longer run makes it worse. Usually a bound mode next to a PML: set `PML(stabilize=...)` to the value the message quotes |
| `PMLStabilityWarning` | a stabilized PML asks for less loss than was measured to stop such growth |
| `RadiationApproximationWarning` | `FarField(propagation="continuum")` need not match the discrete flux exactly |

## The lifecycle and the report

`check(problem, numerics=...)` validates and estimates without advancing any field. Its
`CheckReport` holds `issues` (errors, and warnings such as `geometry.cell_snapped`; `report.ok` is
false only for errors), `resolved` (grid shape, spacing, domains, padding, `dt_seconds`, steps),
`capabilities` (backend, warnings) and `estimates` (memory of the main field and material arrays).

`prepare(problem, numerics=...)` fixes the computational structure; `prepared.run(problem)`
runs a complete problem against it. Material values, source amplitudes, phases, delays and pulse
parameters, the initial-field amplitude and the geometry of bodies with a reserved range are
**live** and may change between runs (within the plan they fixed: a pulse wavelength that changes
a `Duration(periods=...)` step count is refused). Everything else — units, domain, boundaries,
shapes, monitor layout, the material model types — is fixed; changing it raises an
`IncompatibleProblemError`. `run` is `prepare` followed by one `run`.

A `Result` carries `validity.checks`: `admitted`, `runtime_stability` (false when a runtime guard
rejected a traced value, see below) and, per `Diffraction` monitor, `source_reference`.
`result.provenance` records the backend, the units, `dt_seconds`, the actual step count and every
domain.

`admitted` describes the static declaration; it does not override a failed
`runtime_stability` check. In a differentiated run, `diagnostics['grazing']` marks a
traced physical medium as `skipped`: its index cannot be surveyed on the host.
Acceptance rows computed from the preparation's concrete material seed are not
reported as acceptance of the live material. The runtime guard supplies the live
validity verdict instead. A fixed PML `reference_media` entry fixes the numerical
stretch, not that physical material's grazing admission.

An off-design `Diffraction` wavelength can require an exit cosine at or above
one, even when that order is evanescent in the prepared material. A requirement
above one admits no propagating order. If a live material makes the order
propagate, the runtime guard refuses it just as `check` does for concrete values.
At or just below one, admission uses the same rounded cosine as the concrete
check, without clipping the requirement or relaxing the grazing tolerance.
For an explicit very small `Axis1D(k=...)`, an interval lower endpoint can round
to zero. Zero is an open-boundary sentinel only; the eager arithmetic for every
positive material index and all admission thresholds remain unchanged.

Each `prepared.run(problem)` starts a fresh field state and returns its own checks.
A guarded invalid value returns NaN with failed runtime validity; a host-only
unsupported trace raises explicitly. Neither outcome makes a later valid run
invalid or changes an earlier result. Reuse the complete valid problem to retry.

(fdtd-model-differentiating)=
## Differentiating a run

With `backend="jax"` or `backend="torch"`, a prepared run is differentiable in its live inputs:
material values, source amplitude, phase, delay and pulse, and the initial-field amplitude. JAX
needs `jax.config.update("jax_enable_x64", True)` before arrays are built.

For fixed-duration Torch CUDA runs in the admitted scope, set
`Numerics(checkpoint_steps=11)` to recompute blocks during reverse mode.
`None` keeps the ordinary scan. An explicit setting must be a positive static
integer; CPU execution and `Decay` are refused. Source and observation histories
remain retained, so the setting is not a memory cap. A block longer than the
run leaves an ordinary tail. The chosen schedule appears in the check report's
`resolved['checkpoint_steps']`, the prepared contract's
`resolved_checkpoint_steps`, and the result's `provenance['checkpoint_steps']`.
Prepare again to change the schedule. See the [CUDA checkpoint validation](../../validation/fdtd/general-cuda.md)
for derivative coverage and measured costs: checkpointing increased or reduced
memory depending on the differentiated inputs in the conductor fixture.
Default v1 Job keys are preserved
because `None` is omitted; explicit schedules round-trip with the Job.

Under `torch.no_grad()` or `torch.inference_mode()`, execution bypasses the
backward checkpoint wrappers and block copies. Policy metadata remains intact;
later prepared reuse with gradient recording enabled uses the requested schedule.

Float64 Drude susceptibility rescales large damping to preserve the tested frequency
values and tangents in both the public SI model and the normalized FDTD model.
Public calls carrying float32 retain their previous arithmetic and dtype; this
adds no extreme-value support for float32 or mixed precision. It does not change
Lorentz response or ADE coefficients, or certify every extreme finite input and
objective. See [the differentiation guide](fd-differentiation.md) for a normalized
example, SI/unit-conversion scope and intermediate-underflow limitations.

Geometry is differentiable only for smoothed bodies given a reserved support,
`Numerics(geometry_ranges={"name": ((x0, x1), (y0, y1), (z0, z1))})`: the whole body must stay
inside that box for every value it takes, the box must stay inside the domain and clear of
absorbing layers, its material must be an isotropic lossless `Material`, the reserved boxes of
differentiated bodies must not overlap, and the scene may not use an initial field. A cell-snapped shape's geometry is never differentiable.

Values a traced computation produces cannot be checked before the run, so the engine checks them
during it: a time step past the stability bound, a non-positive permittivity, a medium sampled
below 8 cells per wavelength, or a ranged body leaving its box turns the run's fields into NaN and
sets `validity.checks["runtime_stability"]` to false, instead of returning a plausible number.
A derivative of the discrete solver is not a statement about convergence; compare it with a
finite difference and refine the forward solve as well. The [differentiation guide](fd-differentiation.md)
covers the measurements, using the compatibility spelling of the same engine.

## Example: a coating on a substrate, against the closed form

Every interface here lies on a cell boundary, so nothing snaps; the substrate continues into the
bottom PML, which is matched to it.

```python
import numpy as np
from wavebit import Material, Units, fdtd

dx = .02                                       # um; 0.1 um coating = 5 cells
wavelengths = (.7, .8, .9)
problem = fdtd.Problem(
    units=Units(length="um"),
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 1.6)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        structures=(
            fdtd.Structure("substrate", fdtd.Slab(z=(-.8, 0.)), Material(epsilon=2.25)),
            fdtd.Structure("coating", fdtd.Slab(z=(0., .1)), Material(epsilon=6.25)),
        )),
    sources=(fdtd.PlaneWave("light", plane=fdtd.Plane(z=.5),
                            pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13)),),
    outputs=fdtd.Outputs(monitors=(
        fdtd.Diffraction("r", plane=fdtd.Plane(z=.3), wavelengths=wavelengths,
                         source="light", side="reflect"),
        fdtd.Diffraction("t", plane=fdtd.Plane(z=-.3), wavelengths=wavelengths,
                         source="light", side="transmit"),
    )),
)
numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=dx),
                         duration=fdtd.Duration(periods=60, source="light"))

report = fdtd.check(problem, numerics=numerics)
assert report.ok and not report.issues         # nothing snapped
print(report.resolved["grid_shape"], report.resolved["steps"])

result = fdtd.run(problem, numerics=numerics)
R = result["r"].efficiency[:, 0, 0]
T = result["t"].efficiency[:, 0, 0]


def airy(wavelength, n1=2.5, n2=1.5, d=.1):  # normal incidence, air / coating / substrate
    r01, r12 = (1 - n1) / (1 + n1), (n1 - n2) / (n1 + n2)
    t01, t12 = 2 / (1 + n1), 2 * n1 / (n1 + n2)
    phase = np.exp(2j * np.pi * n1 * d / wavelength)
    denominator = 1 + r01 * r12 * phase**2
    return (abs((r01 + r12 * phase**2) / denominator)**2,
            n2 * abs(t01 * t12 * phase / denominator)**2)


for w, r, t in zip(wavelengths, R, T):
    print(w, r, t, airy(w))
```

Measured on 2026-10-15: R + T is 1 to 5e-5 at each wavelength, and R at 0.8 µm is 0.3504 against
the closed form's 0.3420. Halving `dx` to 0.01 µm gives 0.3440: the error falls from 8.4e-3 to
2.1e-3, a factor of 4.1, second order as expected when no face snaps. The phase is another
matter: with `reference_plane=.1, incident_plane=.1` on the reflection monitor, `arg r` at the
declared top face differs from the closed form by 0.163 rad at 0.8 µm, and by 0.080 rad at
`dx = 0.01` -- about `k0·dx`, first order, because the snapped layers act half a cell low for the
tangential field (see [Cell-snapped shapes](#cell-snapped-shapes-slab-and-region)). Written with a
coating of 0.105 µm instead, the same scene would warn `geometry.cell_snapped` and run as 0.1 µm.

## General-scene limits

- CUDA is limited to the [documented Torch scope](../../validation/fdtd/general-cuda.md);
  other GPU backends and single precision are refused. The grid is uniform and cubic.
- Per-cell sampled grids use `fdtd.native`; this general-scene formulation has no raster body.
  Constructive solid geometry (union, intersection, difference) and tilted extrusion axes are unsupported.
- No Gaussian beams, mode sources or mode monitors.
- No smoothed lossy or dispersive bodies outside flat films.

Custom static geometry implements `fdtd.Shape`; prepared reuse checks its declared instance state.
Its cell support must stay clear of Open padding. See the [migration guide](../../migration/index.md)
for the protocol and [stopping/absorbers](stopping-and-absorbers.md) for numerical policies.

## Further reading

- [Domains, sources and observations](general-fdtd.md) — the declaration API with a short example.
- [Smoothed occupancy](../../validation/fdtd/fd-subpixel-occupancy.md) and [the Mie validation](../../validation/fdtd/fd-mie-validation.md)
  — how smoothing is measured, including the half-cell placement of a snapped slab's face.
- [A periodic metasurface against RCWA](../../validation/fdtd/fd-metasurface-validation.md) — what agrees with
  RCWA and at what convergence order.
- [Radiation consistent with the Yee lattice](../../validation/fdtd/fd-lattice-radiation.md) — the far-field transform.
- The film studies (for example [one Drude or Lorentz film](../../validation/fdtd/fd-single-pole-film.md)) — the bounded
  scopes for lossy and dispersive layers.
- [Learning FDTD](../../learn/fdtd/index.md) — the order in which to establish a trustworthy result.

The default `memory_bound=None` is omitted from `wavebit.job/v1` serialization, so existing job keys remain unchanged. Explicit bounds are serialized and round-trip with the job.

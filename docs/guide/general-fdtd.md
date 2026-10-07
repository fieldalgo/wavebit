# General FDTD: `wavebit.fd`

`import wavebit.fd as fd` selects the general electromagnetic time-domain engine:
a uniform Yee grid with explicit sources, boundaries, materials and monitors.
It supports finite domains and periodic cells through its own `fd.Scene` API.
The API and its validated scope are still developing.

Wavebit currently has two distinct FDTD entry points:

| Entry point | Input and result | Manual |
| --- | --- | --- |
| `wavebit.fdtd(...)` | The periodic `Stack`/`Incidence` interface shared with RCWA; returns its scattering `Result`. The default route is legacy NumPy; an explicit backend selects the common NumPy/JAX/PyTorch route. | [Periodic-stack time-domain solver](time-domain.md) |
| `fd.run(scene, ...)`, after `import wavebit.fd as fd` | An `fd.Scene` with a spatial domain and named monitors; returns `fd.Results`, indexed by monitor name. | This page |

These scene and material classes are separate APIs. For example, `fd.Material`
uses a real time-domain material model, while the root `wavebit.Material` can
hold a single-frequency complex permittivity. Do not transfer keyword or angle
conventions between the two entry points.

## Declare, inspect and run

| Declaration | Contract |
| --- | --- |
| `Domain(size=..., center=...)` | Triples are `(x, y, z)`. The grid rounds the requested size up to whole cells while keeping the centre fixed. |
| `Boundaries(x=..., y=..., z=...)` | All faces must be stated. One face type applies to both ends of an axis; a `(low, high)` pair sets them separately. |
| `Resolution(dx=...)` | One uniform cell size for all axes; alternatively give `cells_per_wavelength=` together with `wavelength=`. |
| `Run(...)` | Exactly one of `steps=`, `time=` or `periods=`. Periods require a shared concrete source-band centre. `courant` defaults to 0.5. |
| `background=`, `structures=`, `sources=`, `monitors=` | Vacuum and empty tuples are the defaults. A structure pairs a shape and a material; source and monitor names identify their results. |

This small PEC cavity records an electric pulse. It deliberately has reflecting
walls, so the field can persist after the source ends; this is a time-series
example, not an open-space scattering or convergence calculation.

```python
import wavebit.fd as fd

scene = fd.Scene(
    domain=fd.Domain(size=(1.0, 1.0, 1.0)),
    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
    resolution=fd.Resolution(dx=0.05),
    run=fd.Run(time=8.0),
    background=fd.Material(eps=1.0),
    sources=(fd.Dipole(
        'pulse', center=(0.0, 0.0, 0.0), component='Ey',
        band=fd.Band(wavelength=1.0, df=0.15),
    ),),
    monitors=(fd.Record('probe', at=(0.2, 0.0, 0.0), component='Ey'),),
)

plan = fd.plan(scene)
report = fd.check(scene)  # Prints a report; does not advance the fields.
result = fd.run(scene, backend='numpy', plan=plan)
probe = result['probe']

print(probe.t.shape, probe.values.shape)
print(result.final.E.shape, result.final.H.shape)
print(result.report.bytes_per_state)
```

The point-probe class is `Record`. It reads the nearest Yee point of the named
component, not an interpolated field at the exact requested position.
`probe.t` contains the actual sample times: electric components are recorded at
integer steps and magnetic components half a step earlier. `probe.values` lives
on the backend that ran; the time coordinate is a NumPy array. The final `E` and
`H` arrays retain the same half-step time offset.

`fd.plan(scene)` constructs the static grid and checks its declared constraints.
`fd.check(scene)` independently plans and lowers the scene on NumPy to print and
return a `Report`; it is not a time integration. Pass a concrete plan to
`fd.run(..., plan=plan)` to reuse it. Build a new plan when changing a concrete
scene; differentiated runs follow the narrower plan-reuse rules in
[Differentiating the general engine](fd-differentiation.md). The completed result
also carries `result.plan`, `result.report`, `result.sources` and
`result.diagnostics`.

## Optional early stopping

For a concrete CPU transient with an initial field or Dipole sources, add an
explicit amplitude threshold and time window to the existing capped run:

```python
scene = scene.replace(run=scene.run.replace(
    decay=fd.Decay(threshold=1e-5, window=1.0),
))
result = fd.run(scene, backend='numpy')
print(result.diagnostics['stopping']['reason'])  # 'decay' or 'cap'
```

These values illustrate the spelling; choose them for the transient being
studied. Both arguments are required. `window` uses the same time units as
`Run(time=...)`; `threshold` is a finite amplitude fraction strictly between
zero and one. Keep exactly one existing `steps`, `time` or `periods` limit as
the hard cap. A final partial window cannot satisfy the stop condition.

Each full field array and each evolving material/absorber array must remain
below the requested fraction of its own peak throughout a complete window.
A window is eligible only after every Dipole's falling Gaussian envelope is
below both its Band cutoff and this threshold. The waveform remains unchanged.
Static residues or persistent oscillations can reach the cap; `reason='cap'`
does not claim convergence. A small amplitude is not a spectral error bound.

`result.plan.steps`, `result.report.steps`, histories and DFT endpoint
corrections describe the actual executed prefix. The stopping diagnostics also
retain `cap_steps`, the criterion and per-group amplitudes. The returned short
plan cannot be reused for the original capped scene; keep its original
`fd.plan(scene)` instead. Histories initially allocate for the cap, and state
reductions add work every step, so early termination need not be faster.

PlaneWave/TFSF sources, automatic differentiation, enclosing JAX transforms
and non-CPU execution are refused for this option. Ordinary fixed-duration
runs retain their existing contracts. See the [stopping measurement and
limitations](../fd-decay-stopping.md).

## Units, coordinates and backends

| Quantity | General-engine convention |
| --- | --- |
| Length | Choose one unit for the domain, geometry, `dx` and vacuum wavelengths. |
| Time and frequency | `c = eps0 = mu0 = 1`: `time` and `dt` are lengths, `f = 1/wavelength`, and angular frequency is `2*pi*f`. A normalized time of one corresponds to your length unit divided by the physical speed of light. |
| Material | `eps` and `mu` are relative values. Conductivity `sigma` has units of inverse length; pole frequencies and damping rates use angular frequency. A complex constant is refused: use `Conductor` or `Dispersive`. |
| Angles | `PlaneWave(theta=..., phi=...)` and shape rotation angles use degrees. A Bloch phase and a source waveform phase use radians. |
| Arrays | Fields have shape `(3, nz, ny, nx)`, component order x/y/z. Public coordinate triples remain `(x, y, z)`. Magnetic fields are stored as `Z0 * H`. |
| Step size | `dt = courant * dx / sqrt(3)`. Material and geometric stability checks may refuse that choice; this vacuum formula alone does not certify a material scene. |

`fd.run` accepts `backend='numpy'`, `'jax'` or `'torch'` (also `wavebit.Backend`).
NumPy is the default. The three CPU backends are tested; request `device='cpu'`
when selecting JAX or Torch to keep that placement explicit. GPU correctness and
performance are not established by these CPU records. The plan uses `float64`,
or `complex128` for Bloch or `Axis1D` faces. JAX and Torch support the documented
[derivative scope](fd-differentiation.md); choosing a differentiable backend does
not make every geometry or material parameter admissible.

The report's `bytes_per_state` is the logical payload of the main-grid E/H
arrays, CPML histories and material memories, including reserved inactive pole
slots and a hybrid film's auxiliary arrays. It excludes source histories and
auxiliary source columns, observers and energy diagnostics, runtime temporaries,
checkpoints and allocator/Python overhead. It is not peak process or device
memory. Coefficients and frequency-domain monitor accumulators have separate
report entries; summing those entries still does not include every runtime cost.

## Supported combinations and their limits

| Use | Current contract and further reading |
| --- | --- |
| Sources and readings | `Dipole`, `PlaneWave` and axis-directed `TFSF`; point `Record`, spectral `Field`/`Flux`, periodic `Diffraction`, and open-domain `FarField`. See [metasurfaces](fd-metasurface.md) and [radiation](fd-radiation.md) for normalization and boundary restrictions. |
| Boundaries | `PEC`, `PMC`, `Periodic`, `Bloch`, `Axis1D`, `Symmetry`, `PML` and `Absorber`, subject to material matching, source clearance and geometry checks. Absorbing layers occupy cells inside the domain. |
| Cell-snapped material regions | `Slab` and `Region`; real scalar/diagonal `Material`, conductivity through `Conductor`, and passive Drude/Lorentz/Debye models through `Dispersive`, subject to the planning checks. `fd.eps_at` evaluates a pole model at a vacuum wavelength. |
| Smoothed dielectric shapes | `Sphere`, `Box`, `Cylinder` and `Prism` have bounded isotropic lossless material, separation and boundary contracts; only specific flush junctions are admitted. See [smoothed occupancy](../fd-subpixel-occupancy.md). `discretisation='stencil'` has a separate [isolated-sphere contract](../fd-symmetric-stencil.md). |
| Smoothed lossy or dispersive films | One concrete, unturned, unranged `Box` in a plain isotropic non-magnetic host, spanning exactly two axes with the required overhang and clearing absorbers along its finite thickness. Conductors, finite Debye sums, finite Drude/Lorentz sums and mixed families each have their own restrictions: [conductive](../fd-conductive-film.md), [Debye](../fd-debye-sum-film.md), [oscillator](../fd-oscillator-sum-film.md), [mixed-family](../fd-hybrid-film.md). |
| Separated conductive films | Two or more concrete, unturned, unranged voxel `Box` structures, all with `Conductor` materials (including `sigma=0`), in one plain positive isotropic lossless non-magnetic `Material` host. All films share a normal axis and their padded grid supports are pairwise disjoint. Initial fields, traced geometry/material/band values, touching films and stacks mixing declared pole families are refused. See [conductive stacks](../fd-conductive-stack.md). |

Declare a conductive stack with the ordinary `Scene(structures=(...))` tuple;
there is no separate stack constructor. Each film must span the same two
transverse axes with at least one cell of overhang on each side. Its normal
thickness must be at least `dx`, and its padded support must clear the normal
domain walls and all absorbing layers.

Separation is checked on the actual padded, half-open grid supports, not on
a fixed physical gap. A refusal identifies the adjacent members and their
support intervals; enlarge the host gap or refine `dx`. A zero-conductivity
member remains part of the declared stack and its memory allocation. Changing
any member, the declaration order or the host requires a new plan.

The film support does not extend to arbitrary curved lossy bodies, dispersive
hosts, lossy junctions, or their geometry/material/band derivatives. General CSG,
raster geometry, tilted extrusion axes, non-uniform grids, `GaussianBeam`,
`ModeSource` and `ModeMonitor` are not implemented for
this engine. Exported placeholder names can still refuse construction.

A successful plan or stable run is not an accuracy certificate. Refine space,
duration and absorbing boundaries independently for the quantity you need, and
read the warnings. The [C7 metasurface comparison](../fd-metasurface-validation.md)
remains unpassed, including the unresolved cause of M3's slow convergence.
The bounded film work does not complete general C8 smoothing; its validation
records retain mixed and noise-unresolved cases. A rule derived for a bulk mesh
scale is not a convergence theorem for every interface.

# Automatic stopping and graded absorbing padding

`fdtd.Duration` remains the hard run cap. To stop a concrete forward run earlier,
set `Numerics(stopping=fdtd.DecayStop(threshold=.02, window_seconds=...))`. The window is
an SI duration even when geometry lengths use nanometres or micrometres.

Complete, non-overlapping windows check field amplitudes and every supported
material/CPML memory group against that group's own peak. A field must have been
excited, all groups must pass, and sources must have passed their trailing-envelope
gate before an eligible window begins. An unexcited scene reaches the cap. Delayed
sources, nonfinite states and residual material memory cannot silently disappear
from the criterion.

The supported scope is concrete CPU NumPy, JAX and PyTorch with `InitialField`,
`CurrentDipole`, `PlaneWave` or axis-aligned `TFSF` excitation. PlaneWave/TFSF incident
columns, and the extra driver of a layered TFSF column, contribute their own E/H,
CPML and supported material-memory groups. Accumulating source reference histories
are excluded. Unknown nested memory schemas are refused. Differentiation, batching
and external JAX transforms remain unsupported; use a fixed duration for AD.
Decay is an amplitude heuristic: it does not bound spectral error, predict a later
reflection, or establish spatial convergence.

Every group's peak is independent, even when the whole group contains only tiny
round-off leakage. Such CPML or material residuals can keep a run at its cap, and
round-off differences can change whether different backends stop early. No memory
group is discarded just because its absolute amplitude is small. Inspect `reason`,
`reached` and the per-group ratios; reaching the cap is not a successful decay.

The returned result describes the executed prefix. `provenance['actual_steps']`,
`TimeSeriesMonitor` timestamps, final E/H timestamps (`final_fields.metadata['E_time_seconds']`
and `['H_time_seconds']`) and Fourier windows use the actual stop,
while `prepared.report.resolved` describes the requested cap. The
`diagnostics['stopping']` report includes reason, cap and actual step counts,
`window_seconds`, `eligible_time_seconds`, and separate dimensionless state ratios.
Its raw peak/window amplitudes and cap skeleton retain explicitly labelled internal
engine units; they are not interchangeable physical energies.

For an `Open()` face, `Numerics(absorber=fdtd.ConductiveAbsorber(cells=12))` chooses the graded
matched-conductivity fallback. Padding is allocated **outside** the physical Domain,
just as for PML. Uniform boundary media continue through it; finite-body continuation
still requires an admitted explicit declaration. This option allows the existing
conductive/dispersive boundary scopes that PML refuses.

PML remains the default. The graded absorber's fixed cubic profile has a
normal-incidence round-trip amplitude near `exp(-3n)`, where `n` is the refractive index
of the medium filling the layer: `exp(-3)`, about 0.05, in vacuum and about 0.011 at
`n = 1.5`. Adding cells does not remove that attenuation floor. It is not equivalent to PML, and it does not accept
PML reference-media, frequency-shift or reflection-target controls. Study reflection
and resolution for the actual scene; see the retained
[absorber evidence](../../validation/fdtd/fd-constitutive-absorber.md) and
[stopping validation](../../validation/fdtd/fd-decay-stopping.md).

That 0.05 is an amplitude, and it is coherent with the scene's own reflections: a
reflectance or transmittance moves at first order, by up to about `2 × 0.05 × |r|` per
face, not by the 2.5e-3 of its power. On a glass slab (`eps = 2.25`) between two such
faces at normal incidence, the transmitted flux moves by up to 6 % against the PML run,
which stays within the grid's dispersion of the closed form. `fdtd.check()` warns
(`numerics.absorber_reflection`) when a `PlaneWave` or a `TFSF` propagates into these faces.

A `PlaneWave` computes its incident wave in a 1-D column of the background. Behind a
`ConductiveAbsorber` face that column ends in a PML of its own, so the source does not
re-inject the absorber's echo; the main grid keeps the absorber, and its return is
measured as scattered field. (Before this, the echo was sent back down as a second
incident wave: up to ±10 % in the transmitted flux of an empty cell.) A PML cannot be
matched to a conductive or dispersive background, so a `PlaneWave` in such a background
behind a `ConductiveAbsorber` is refused at `check()` with `fdtd.unsupported`; give the
lossy medium as a layer clear of the sheet instead.

A `TFSF` box takes its incident field from the same kind of column (and, over a layered
background, from a uniform driver column), and behind a `ConductiveAbsorber` it now ends
in a PML of its own as well. A layer that reaches the face, such as a substrate, continues
through that PML, which is matched to it. An empty box never showed the echo, because it
cancels whatever its column holds. A scatterer did: the measured incident spectrum moved
by up to ±3 % and a glass cube's scattered flux by up to +17 %, and over a glass substrate
the reflected amplitude moved by up to ±17 %. A conductive or dispersive medium at such a
face of a `TFSF` is refused at `check()` with `fdtd.unsupported`.

A smoothed structure (`Sphere`, `Cylinder`, `Box`, `Prism`) whose support reaches into a
`ConductiveAbsorber` layer couples to it, and that coupling is admitted only with plain
lossless isotropic materials (one positive real `epsilon`, no `mu`), electric
`CurrentDipole` sources, Open faces on one axis with `PEC`, `PMC`, `Periodic` or `Bloch`
elsewhere, `Numerics(discretization='voxel')` without `geometry_ranges`, a fixed duration
(no `DecayStop`), no `InitialField`, plain numbers rather than traced or differentiated
ones, and the CPU. Anything else is refused at `check()`, and at `prepare`, `run` and
`Prepared.validate` alike, with `numerics.absorber` at `numerics.absorber` and a message that
names which of these it is. A `ConductiveAbsorber(cells=...)` that is not an integer of at
least two is refused with the same code at `numerics.absorber.cells`. Moving the smoothed structures at least three grid cells clear of every
Open face keeps their support out of the layer and lifts all of these; two cells were not
always enough. Differentiating such a structure's material then goes through
`Prepared.run` of a problem prepared from plain numbers, as everywhere else.

A `PML` without `wavelength=` tunes its complex-frequency shift to the band centre its
sources share, so `prepare` fixes that wavelength with the rest of the absorber. Reusing the
preparation with another `GaussianPulse.wavelength`, concrete or traced (`jax.grad`,
`jax.jit`, a Torch tensor that requires a gradient), is refused as `prepared.pml_wavelength`
at `sources.<name>.pulse.wavelength`. Give the absorber its own tuning wavelength,
`fdtd.PML(cells=..., wavelength=...)`, to vary or differentiate the pulse wavelength under one
preparation: the layer then stays at that wavelength for every run, as it would in a fresh
`fdtd.run` with the same numerics.

For current PlaneWave/TFSF coverage and the limits of the tested time windows, see
[source-column stopping validation](../../validation/fdtd/fd-source-decay.md).

Both policies serialize through current `Job` files. The public regression checks
stopping against an independent uniform-conductor recurrence, exact executed-prefix
fields and time series on all three CPU backends, nm/µm invariance, and graded-padding E/SI-H
against an independently declared normalized scene.

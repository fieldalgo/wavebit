# Automatic stopping and graded absorbing padding

`fdtd.Duration` remains the hard run cap. To stop a concrete forward run earlier,
set `Numerics(decay=fdtd.Decay(threshold=.02, window_seconds=...))`. The window is
an SI duration even when geometry lengths use nanometres or micrometres.

Complete, non-overlapping windows check field amplitudes and every supported
material/CPML memory group against that group's own peak. A field must have been
excited, all groups must pass, and sources must have passed their trailing-envelope
gate before an eligible window begins. An unexcited scene reaches the cap. Delayed
sources, nonfinite states and residual material memory cannot silently disappear
from the criterion.

The existing engine's scope is concrete CPU NumPy, JAX and PyTorch with `InitialField`
or `Dipole` excitation. PlaneWave/TFSF source columns, differentiation, batching and
external JAX transforms are refused with this policy. Use a fixed duration for AD.
Decay is an amplitude heuristic: it does not bound spectral error, predict a later
reflection, or establish spatial convergence.

The returned result describes the executed prefix. `provenance['actual_steps']`,
record timestamps, final E/H timestamps and Fourier windows use the actual stop,
while `prepared.report.resolved` describes the requested cap. The
`diagnostics['stopping']` report includes reason, cap and actual step counts,
`window_seconds`, `eligible_time_seconds`, and separate dimensionless state ratios.
Its raw peak/window amplitudes and cap skeleton retain explicitly labelled internal
engine units; they are not interchangeable physical energies.

For an `Open()` face, `Numerics(absorber=fdtd.Absorber(cells=12))` chooses the graded
matched-conductivity fallback. Padding is allocated **outside** the physical Domain,
just as for PML. Uniform boundary media continue through it; finite-body continuation
still requires an admitted explicit declaration. This option allows the existing
conductive/dispersive boundary scopes that PML refuses.

PML remains the default. The graded absorber's fixed cubic profile has a
normal-incidence round-trip amplitude near `exp(-3)`, about 0.05; adding cells does
not remove that attenuation floor. It is not equivalent to PML, and it does not accept
PML reference-media, frequency-shift or reflection-target controls. Study reflection
and resolution for the actual scene; see the retained
[absorber evidence](../../validation/fdtd/fd-constitutive-absorber.md) and
[stopping validation](../../validation/fdtd/fd-decay-stopping.md).

Both policies serialize through current `Job` files. The public regression checks
stopping against an independent uniform-conductor recurrence, exact executed-prefix
fields/records on all three CPU backends, nm/µm invariance, and graded-padding E/SI-H
against an independently declared normalized scene.

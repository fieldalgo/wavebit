
# A solve in the time domain

`wavebit.fdtd()` is the second solver. It reads the same `Stack` and `Incidence` the RCWA path
reads and hands back the same `Result`, so one scene is solved two ways with nothing translated
between the answers.

This path is a **research alpha**. Read the [supported scope and validation evidence](../fdtd-alpha.md)
before using a new combination of geometry, incidence and material.

It returns the answer directly, in the same `Result` container as RCWA. With no
`backend` argument, it preserves the legacy NumPy solver, including time records
and graded meshes. Explicit `backend=0/1/2` (or `numpy/jax/torch`) selects the
shared Yee engine and supports material automatic differentiation.
See [the common scene adapter](#common-scene-adapter) for its supported inputs.

## The same scene, both ways

```python
import numpy as np
import wavebit
from wavebit import Stack, Layer, Incidence

eps_hi = wavebit.eps_from_index(3.5)                       # 12.25
scene = Stack(layers=[Layer(thickness=300.,
                            structure=np.array([[1., 1., eps_hi, eps_hi]]))],
              top=1., bot=wavebit.eps_from_index(1.5), period=(700.,))
light = Incidence(wavelength=900., theta=0., phi=0., tetm='te')

harmonic = wavebit.rcwa(backend=0, stack=scene, incidence=light,
                        truncation_order=dict(x=40, y=0)).solve()
time_domain = wavebit.fdtd(stack=scene, incidence=light, resolution=0.1)   # 10 nm cells

for m in (-1, 0, 1):
    a, b = harmonic.order(m, 0).T, time_domain.order(m, 0).T
    print(f'T[{m:+d}]   rcwa {a:.6f}   fdtd {b:.6f}   gap {abs(a - b):.1e}')
print(f'R + T    rcwa {harmonic.R.sum() + harmonic.T.sum():.6f}   '
      f'fdtd {time_domain.R.sum() + time_domain.T.sum():.6f}')
print(f'T[+1] - T[-1] = {abs(time_domain.order(1, 0).T - time_domain.order(-1, 0).T):.1e}')
```

```
T[-1]   rcwa 0.221306   fdtd 0.219304   gap 2.0e-03
T[+0]   rcwa 0.352483   fdtd 0.353580   gap 1.1e-03
T[+1]   rcwa 0.221306   fdtd 0.219304   gap 2.0e-03
R + T    rcwa 1.000000   fdtd 0.999840
T[+1] - T[-1] = 2.5e-16
```

Nothing was converted. The `Stack` is the one [`scene.md`](scene.md) describes, the `Incidence`
is [`light.md`](light.md)'s, and the `Result` is the one [`quickstart.md`](quickstart.md) reads
— `R`, `T`, `order(m, n)` and the complex amplitudes, in the same conventions. The gap is the
grid's: it is second order, so a 5 nm cell brings it to ~7e-4 and 2.5 nm to ~1.9e-4.

**`R + T` is not exactly 1, and that is the method rather than a defect.** A Yee grid conserves
energy to `O(dz^2)`, the same order it computes everything else to, so a lossless scene comes
back a little short; the RCWA row above is 1 to a few times 1e-15 for contrast. There is
deliberately no generic guard on it here — one existed and fired on 85 of 243 ordinary
time-domain solves, which is why it was removed ([`../result-guards.md`](../result-guards.md)).

**`T[+1] == T[-1]` is a symmetry check for this scene at normal incidence.** A symmetric
discretization can preserve that equality even on a coarse grid, so it does not bound the
error in either efficiency. Use grid and runtime refinement to assess accuracy.

## Which of the two to reach for

RCWA answers a single wavelength between 112 and 2850 times faster at matched accuracy, so the
question is only ever how many wavelengths you want. Matched on accuracy rather than on settings
— the coarsest cell and the cheapest `truncation_order` that reach the same worst per-order
error — the break-even is:

| scene | one FDTD run | the cheapest RCWA that matches it | wavelengths to break even |
| --- | --- | --- | --- |
| lossless dielectric grating, 20 nm cell | 0.5 s | `truncation_order` 5, 0.51 ms | ~1000 |
| the same, 10 nm cell | 2.0 s | `truncation_order` 10, 0.71 ms | ~2850 |
| 60 nm Drude metal grating, 10 nm cell | 2.4 s | `truncation_order` 60, 21.4 ms | ~112 |

The figure moves toward the time domain as the RCWA side gets harder, and away as the grid is
refined. So reach for it when **a band is wanted from one run and the frequency-domain solve is
expensive** — a metal, many layers, a high truncation order — and for two things RCWA cannot
state at all: a record of the field arriving, and a material given as a pole model rather than
as one complex number per wavelength.

Reach for `rcwa()` for everything else, and for everything the time domain refuses: near fields
through `reconstruct_field`, gradients, JAX and PyTorch, a skewed `lattice`, parametric shapes,
and evanescent (complex-angle) incidence. Elliptical diffraction is available using
`psi` with `chi`, or `polarization=(a_s, a_p)`; for example `(1, 1j)` is a circular
state in the s/p frame. The pair is normalized while preserving its overall phase.
Time recording also supports complex coefficients through additional quadrature-source
runs; see [polarization details](../fdtd.md#elliptical-and-circular-incidence).

## The keywords a time-domain run adds

The scene keywords are unchanged — `structure`, `thickness`, `top`, `bot`, `period`, `mu`,
`wavelength`, `theta`, `phi`, `tetm`, `psi`. **Angles are radians**, as everywhere in wavebit.
What follows is what a time-domain run adds on top; every default here was measured, and
[`../fdtd.md`](../fdtd.md) is the measurement.

| Keyword | Default | What it is for |
| --- | --- | --- |
| `resolution` | 20 cells per wavelength **in the densest material in the scene** | cells per unit length, so `resolution=0.1` is a 10 nm cell when lengths are nm. The knob that costs: doubling it halves the nominal cell size and can cost roughly 16x the work in full 3D (8x the cells and 2x the steps). The convergence rate depends on the geometry; sharp corners need a measured refinement study. A pole model raises the index it is sized against, so a Drude metal gets a fine grid without being asked for |
| `n_periods` | `80` | optical cycles to run. The pulse has to enter, scatter and leave before the running DFT means anything: at 40 the transmission monitor still holds 61% of its peak and the answer is wrong by 0.09; at 80 the residual is 5e-6 and the answer no longer moves. A run under 48 cycles is refused outright — the source has not finished emitting — and one merely too short to settle warns `UnsettledFieldWarning` and reports `residual`. An oblique or near-grazing scene often needs twice a normal-incidence one |
| `n_pml` | `12` | absorber depth in **cells**, not in length. A correctly graded CPML is dz-independent, which is what makes cells the right unit: at 12 the layer reflects 1.1e-5, and 20 buys 1.5e-6 |
| `courant` | `0.99` | fraction of the stability bound. The bound carries `n_min`, so a material faster than vacuum (`eps < 1`) is stable rather than divergent. Lower it only if a run diverges |
| `subpixel` | `True` | average the material over each Yee component's own cell — harmonically along the axis that component points along, arithmetically along the others. It is what buys second order on a boundary that cuts cells rather than first: 2.7e-4 against a point-sampled 3.4e-3 at a 5 nm grid |
| `grading` | off | a geometric ratio, e.g. `1.06`, that keeps the fine cell in the structure and coarsens toward each half-space. `x` stays uniform — the order extraction is an FFT over it. Off by default is a measurement rather than caution: at equal accuracy it is 1.6x faster on a metal grating and *loses* on a dielectric one |
| `dispersion`, `poles` | off | the frequency dependence, as the pair it has to be: a same-shape integer raster marking which cells are dispersive (`0` = none), and `{id: [(omega0, gamma, strength), ...]}` for each other id. `omega0 = 0` is Drude, anything else Lorentz |
| `record_field` | off | keep every N-th step of the electric field in the diagnostics, with the axes it is measured on. Costs nothing when off — the keyword is the only thing that allocates the buffer |
| `pad_scale` | `1.0` | multiplier on the clearance between the structure and each of the TFSF plane, the monitors and the PML; `1.0` is half a wavelength in that half-space |
| `truncation_order` | every propagating order | how many diffraction orders to report, computed for the **shortest** wavelength in the band |
| `return_diagnostics` | `False` | return `(result, diagnostics)` instead of `result` |

`wavelength` takes a **sequence** as well as a number, and that is the one keyword whose meaning
grows here: a band comes out of one run, one `Result` per entry.

## Dispersion, and a band out of one run

A time-domain solve cannot take a complex permittivity — one complex number describes a material
at one frequency, and the pulse carries a whole band. The frequency dependence is given as the
thing it is, a pole model, and each wavelength in the band then sees the permittivity that model
has *there*. That is the combination the break-even table is measuring: one run, against one
RCWA solve per wavelength each handed a different `eps`.

```python
import numpy as np
import wavebit
from wavebit import Stack, Layer, Incidence

# Gold-ish Drude, one pole: `(omega0, gamma, strength)` per material id. `omega0 = 0` is
# Drude and `strength` is the plasma frequency squared; frequencies are the reciprocal of
# `wavelength`'s unit, since c = 1 here.
gold = [(0.0, 2 * np.pi / 17000., (2 * np.pi / 136.) ** 2)]
band = [820., 900., 980.]

scene = Stack(layers=[Layer(thickness=60., structure=np.ones((1, 2)))],
              top=1., bot=wavebit.eps_from_index(1.5), period=(700.,))
light = Incidence(wavelength=band, theta=0., phi=0., tetm='te')

spectrum = wavebit.fdtd(stack=scene, incidence=light, resolution=0.1, n_periods=100,
                        dispersion=np.array([[1, 0]]), poles={1: gold})   # one run

for wl, r in zip(band, spectrum):
    eps = wavebit.permittivity_at(gold, wl)     # the closed form the stepping discretises
    ref = wavebit.rcwa(backend=0, wavelength=wl, period=[700.], theta=0., phi=0., tetm='te',
                       truncation_order=dict(x=60, y=0), top=1.,
                       bot=wavebit.eps_from_index(1.5),
                       structure=np.array([[[eps, 1.]]]), thickness=[60.]).solve()
    print(f'{wl:.0f} nm  eps {eps.real:8.3f}{eps.imag:+.3f}j   '
          f'T0 fdtd {r.order(0, 0).T:.4f} rcwa {ref.order(0, 0).T:.4f}   '
          f'absorbed fdtd {1 - r.R.sum() - r.T.sum():.4f} rcwa '
          f'{1 - ref.R.sum() - ref.T.sum():.4f}')
```

```
820 nm  eps  -35.269+1.749j   T0 fdtd 0.4036 rcwa 0.4049   absorbed fdtd 0.0112 rcwa 0.0111
900 nm  eps  -42.671+2.312j   T0 fdtd 0.4039 rcwa 0.4060   absorbed fdtd 0.0135 rcwa 0.0131
980 nm  eps  -50.753+2.983j   T0 fdtd 0.4144 rcwa 0.4172   absorbed fdtd 0.0162 rcwa 0.0154
```

The band is an identity rather than an approximation: three separate single-wavelength runs
agree with it to 3e-3, which is the gap between two source *bandwidths* rather than between two
answers. The pulse is sized from the requested span, which is what keeps the band edge safe —
every amplitude is divided by the incident spectrum, and with a fixed pulse width an incident
spectrum 2.4e-7 of its peak once returned `R = 105` on a scene whose true `R` is 0.04.

**At oblique incidence a band is a fixed-`kx` sweep, not a fixed-angle one.** The grid holds one
in-plane wavevector while `kx = k0 n sin(theta)` carries `k0`, so the polar angle drifts across
the band — 28.5 degrees at 860 nm and 31.5 at 940 for a 30-degree centre. `angles_deg` in the
diagnostics reports what each wavelength was actually solved at. Normal incidence is unaffected,
`sin 0` being zero at every wavelength, so a fixed-angle band is available there and only there.

## Watching the field arrive

The one thing this solver states that a harmonic one cannot. A harmonic solve answers what the
field *is*, having assumed `e^{-iwt}` and therefore assumed a single frequency describes the
whole problem; there is no picture of light arriving, because the assumption is that it always
has been. `record_field=N` keeps every N-th step of the electric field, **with its own
abscissae** — a reader who rebuilds them is wrong three ways: a graded mesh has no single `dz`,
in-plane coordinates are measured from the centre of the cell, and `Ey` sits on the x nodes
rather than at the pixel centres.

```python
import numpy as np
import wavebit
from wavebit import Stack, Layer, Incidence

eps_hi = wavebit.eps_from_index(3.5)
scene = Stack(layers=[Layer(thickness=300.,
                            structure=np.array([[1., 1., eps_hi, eps_hi]]))],
              top=1., bot=wavebit.eps_from_index(1.5), period=(700.,))
light = Incidence(wavelength=900., theta=0., phi=0., tetm='te')

result, diagnostics = wavebit.fdtd(stack=scene, incidence=light, resolution=0.1,
                                   record_field=100, return_diagnostics=True)

print('residual', f"{diagnostics['residual']:.1e}")
print('component', diagnostics['field_e_component'])
print('field_e  ', diagnostics['field_e'].shape, '(snapshots, nz, ny, nx)')
print('field_x  ', f"{diagnostics['field_x'][0]:.1f} .. {diagnostics['field_x'][-1]:.1f}")
print('field_z  ', f"{diagnostics['field_z'][0]:.1f} .. {diagnostics['field_z'][-1]:.1f}")
```

```
residual 4.6e-06
component ey
field_e   (103, 154, 1, 70) (snapshots, nz, ny, nx)
field_x   -350.0 .. 340.0
field_z   -810.0 .. 720.0
```

`field_e` is `(snapshots, nz, ny, nx)` on every run, so a crossed grating is a larger `ny`
rather than a differently shaped array. `field_e_component` says which component it holds —
`ey` for a TE run and for either conical leg, `ex` for TM — and that matters because the two do
not sit in the same place: the Yee grid stages `Ey` on the x nodes and `Ex` on the half steps.
`field_x` and `field_y` are therefore the axes of *that* component, and a conical run carries
`field_ex` with its own pair beside it.

Rebuilding an axis instead of reading it is wrong three ways, each silently: `field_z` is not
`arange(nz) * dz` once the mesh is graded, `field_x` does not start at zero because every
in-plane coordinate in this library is measured from the centre of the cell, and the pixel
centres `wavebit.meshgrid_xy` returns are half a cell from where `Ey` lives. The measurements
behind all three are in [`../fdtd.md`](../fdtd.md) §8h.

The same record can be plotted or animated with optional Matplotlib
(`pip install 'wavebit[plot]'`):

```python
# Use record_field=10 in the solve above to resolve the optical oscillation.
fig, ax, mesh = wavebit.plot_fdtd_field(diagnostics, frame=len(diagnostics['field_t']) // 2)
animation = wavebit.animate_fdtd_field(diagnostics, interval=50)
# Keep animation alive, then use plt.show() or display(HTML(animation.to_jshtml())).
# animation.save('field.gif', writer='pillow')
```

Both functions show an x-z slice at the nearest recorded `y` coordinate (default 0); the
title reports the row actually selected. `component='ex'` selects Ex when recorded, using
its own staggered coordinates. `part='real'` shows signed E by default; `'imag'` and `'abs'`
are also available. Absolute E is field amplitude, not intensity. The color scale is fixed
across the whole selected time history. Graded z coordinates are drawn as a nonuniform mesh,
PML cells are shaded, and the TFSF line marks where the record changes from scattered field
above to total field below. Layer planes are marked, without implying an in-plane geometry.

Time is in the solver's `c=1` units, the same units used for lengths. Playback `interval`
is milliseconds per snapshot and does not change simulated time. An undersampled record
warns: reduce `record_field` in the solve, since slowing the animation cannot recover missing
optical cycles. `record_field` must be a positive integer.

With `psi`, the record contains the requested coherent combination of the two source
polarisations, including both Ey and Ex. The amplitudes retain the raw source scale; they
are not normalized monitor coefficients. The settling diagnostic reports the larger
residual of the two basis runs. Recording is still off by default.

## Where to read further

[`../fdtd.md`](../fdtd.md) is the decision document: what each default was measured against,
the full list of refusals and what each one protects, and the two questions a time-domain
solver can state that a harmonic one cannot. [`../result-guards.md`](../result-guards.md) says
what a solve checks about its own answer, and why there is no generic energy warning.


To record every electric and magnetic component, add `record_components='all'`.
Read `diag['fields']['hz']['values']` and its `t/z/y/x` axes, or pass `component='hz'`
to the plotting/animation helpers. H is normalized as `Z0 H`; its timestamps differ
from E by half a time step. See [the component layout](../fdtd.md#recording-all-six-field-components)
before combining staggered fields.


`wavebit.fdtd_poynting(diag, z=50., x=0., y=0.)` returns a point's instantaneous
energy flow in `S[:, 0:3]`, after interpolating all six fields to common Yee positions
and times. `sample_fdtd_fields` returns the interpolated E/H directly. Both require
`record_components='all'`; see [point probes](../fdtd.md#point-probes-and-instantaneous-energy-flow)
for units, sampling limits and the distinction between total and scattered flux.


## Streaming frequency planes

Pass `field_monitors={'near': z_near, 'transmitted': z_out}` together with
`return_diagnostics=True` to `wavebit.fdtd`. No `record_field` is needed. Each entry
of `diag['monitors']` contains six complex `fields` arrays `(wavelength, y, x)`,
`x`/`y` coordinates, `flux_density` `(wavelength, y, x, 3)` and `power_z`
`(wavelength,)`. Spatial interpolation collocates native Yee components, including
Bloch phases at lateral edges; Fourier integration uses each component's native
time. Complex Jones coefficients combine fields before computing power.

The normalization is the soft-source pulse spectrum, **not incident power**.
With H stored as Z0 H, flux is `0.5*real(E cross conj(H))`; `power_z` integrates
its z component over the periodic cell, or per unit y depth for a y-invariant
scene. Run the matching empty scene to normalize transmission. On a scattered
reflection plane, negate its power and divide by the empty incident power.
A total-field plane contains incident/reflected interference; it is not a
separated reflection measurement. `region` identifies the side of the TFSF
boundary. Planes whose interpolation touches that boundary are refused.
These local interpolated fluxes converge with mesh refinement and need not equal
modal diffraction powers exactly at a finite resolution.

See `examples/fdtd_frequency_planes.py` for a three-wavelength slab check.
Storage scales with six components times wavelengths times plane area; it does
not grow with the number of time steps. These are streaming Fourier measurements,
not a whole-volume harmonic field reconstruction.

## Shared NumPy, JAX and Torch native grids

`wavebit.fdtd_grid` is a Cartesian Yee interface sharing one Array API calculation
core across NumPy, JAX and Torch. `backend=0/1/2` has the same meaning as RCWA;
`'numpy'/'jax'/'torch'` remain accepted aliases.
It accepts native `(z,y,x)` permittivity, or `(3,z,y,x)` diagonal coefficients,
and `(3,z,y,x)` initial E/H and electric-current source arrays. Specify
`spacing=(dz,dy,dx)`, `dt` and `steps` explicitly, in c=1 units. E starts at zero
time and H at `-dt/2`. Scalar or broadcastable positive permeability and nonnegative
electric conductivity are supported. Coefficients belong on each component's
native Yee location; the function does not rasterize or average interfaces.

```python
import jax
import jax.numpy as jnp
import wavebit

source = jnp.zeros((3, 32, 2, 2)).at[1, 8].set(1.)
pulse = jnp.sin(jnp.arange(80) * .3)

def objective(epsilon):
    out = wavebit.fdtd_grid(
        epsilon, spacing=(1., 1., 1.), dt=.2, steps=80,
        source=source, pulse=pulse, wavelengths=[4.], monitor_z=[20],
        backend='jax', checkpoint_steps=16)
    return out['power_z'][0, 0]

value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.ones((32, 2, 2)))
```

Native-grid spectra are **unnormalized time integrals**, unlike the scene
solver's source-normalized planes. Electric/magnetic spectra have shape
`(wavelength, plane, component, y, x)`, flux density has shape
`(wavelength, plane, y, x, 3)`, and power has shape `(wavelength, plane)`.
Use a matching reference simulation for source or incident-power normalization.

All axes are periodic by default. `absorber_cells=N` adds graded electric/magnetic damping
at both z ends; it is not CPML and needs a width/reflection convergence check.
`pml_cells` additionally provides convolutional PML at two homogeneous z ends;
see the material/boundary section below. This engine does not yet accept
curved or skew meshes. Static nonuniform z widths and scene grading are now supported.
The common scene adapter below converts Stack/Incidence inputs. The legacy scene solver
retains those capabilities on NumPy; selecting a JAX backend on that interface
is still unsupported.

JAX uses compiled scans and recomputes blocks during reverse differentiation.
Torch uses eager autograd with non-reentrant block checkpointing; Torch compile
has not been validated.
`checkpoint_steps` trades stored block boundaries against recomputation and
within-block storage. Time history is optional via `record_field=N`; reverse-mode
memory and requested snapshot storage still grow with simulation length. Shapes, spacing, step count and monitor configuration
are static. Materials, source, pulse and initial fields remain differentiable.
Choose a fixed dt valid throughout optimization: concrete invalid materials or
CFL violations raise; traced coefficients expose `stable` in the result and must
be constrained by the caller. An optimizer must preserve positive epsilon/mu and
nonnegative conductivity. Enable JAX x64 before array creation for precision studies.

`device='cpu'` or `device='gpu'` selects a device; Torch also accepts `'cuda'`
and device objects. Torch inherits a tensor epsilon's device when device is
omitted, otherwise uses CPU. Unavailable hardware raises.
CPU parity, discrete eigenmode propagation, material/pulse derivatives, and block
checkpoint equivalence are tested. External CPU checks additionally compare slab
complex fields/power with Meep and native 3D updates/material gradients with
FDTDX; frozen external outputs support regression tests. See the
[validation scope](../fdtd-alpha.md). Actual GPU execution has not been validated.
`examples/fdtd_grid_gradient.py` differentiates slab transmission;
`studies/fdtd_grid_benchmark.py` reports compilation separately from warm execution.
On one CPU run (32×8×8 cells, 256 steps, complex128), NumPy took 0.0437 s,
JAX compilation plus first execution 0.2857 s, and median warm execution 0.00603 s
(7.25×). This is a small native-grid benchmark, not a scene-solver speed claim.


## Native dispersive materials and CPML

`fdtd_grid(..., poles=[(omega0, gamma, strength), ...])` advances electric
polarization with a centered ADE recurrence. Each parameter broadcasts to
`(3,z,y,x)` at the native electric component locations. The material law is
`epsilon(omega) = epsilon_infinity + sum(strength / (omega0**2 - omega**2 - 1j*gamma*omega))`.
The `epsilon` argument is epsilon-infinity; `omega0=0` gives Drude. Frequencies
and damping are **angular** in c=1 time units. Parameters must be finite,
nonnegative, and real; epsilon-infinity and mu must be positive. Pole frequencies
and strengths tighten the conservative CFL bound. Traced invalid parameters
return `stable=False` and must be excluded by the optimization parameterization.

Both initial polarization levels are zero. Final `polarization` and
`previous_polarization` are tuples of component arrays at E time and one step
before it. Thus each active pole retains two volume arrays in addition to E/H.
This API does not yet take a restart state or magnetic/off-diagonal dispersion.
There is no interface rasterization: supply native coefficient weights explicitly.
For a tangential planar interface, half-cell strength weights converge against
the analytic slab solution. They are not a general tensor interface averaging rule.

`pml_cells=N, pml_wavelength=lambda_center` adds convolutional PML at both z
ends, using Wavebit's existing coefficient formulation evaluated at native E/H
positions. The end layers **and one adjacent interior cell** must match the static
`pml_background=((epsilon_top, mu_top), (epsilon_bottom, mu_bottom))` values
(default: vacuum)
(epsilon=mu=1), zero electric conductivity and zero oscillator strength. x/y
remain periodic. Cannot combine this with `absorber_cells`; the latter remains
a simpler graded damping option. Keep physical sources and measurement planes
in the interior. Four CPML memory arrays are retained, each with the spatial
volume shape. The CPML coefficients depend on static geometry and center
wavelength; material/pole gradients propagate through the interior updates.

Validation includes displacement conservation, NumPy/JAX agreement, finite
differences for all three pole parameters, checkpoint agreement, and normal/
oblique pulse removal. External CPU runs compare complex slab fields and power
against Meep for dielectric, Lorentz, Drude and two-pole media, and compare 3D
mixed-pole updates/strength gradients against FDTDX. Meep's documented point
sampling of dispersion is matched explicitly for the direct comparison; a
separate weighted run checks convergence to the Airy solution. See the external
reference record in `tests/references/README.md` in the checkout.

PML thickness and run time are convergence parameters. Broad pulses with a
near-zero-frequency component can leave slowly decaying CPML tails; simply
increasing width does not demonstrate temporal convergence. The reference pulse
is symmetric with zero DC, and tests also extend runtime and double PML width.
This does not establish arbitrary-band or grazing-angle accuracy.
`examples/fdtd_dispersive_gradient.py` differentiates Lorentz oscillator strength
with CPML enabled. Actual GPU execution remains unverified.


## Native Bloch phases and incidence gradients

`bloch_phase=(phase_z, phase_y, phase_x)` specifies real phase advances in radians
across each simulation period: `F(r+L_axis)=exp(1j*phase_axis)*F(r)`. It follows
the `(z,y,x)` order of `spacing`, not physical vector component order. Zero
phases recover periodic boundaries. Both updates and monitor interpolation apply
the phases at wrapped edges. z CPML requires `phase_z=0`; transverse phases
remain available with CPML and dispersive materials.

Phases are JAX-differentiable. For a plane source, vary its spatial phase with
the boundary phase as well: `phase_x=kx*nx*dx`, `source_y(x)=exp(1j*kx*x)`,
where `kx=2*pi*sin(theta)/wavelength` in vacuum. A complex temporal carrier
`envelope*exp(-1j*omega*t)` selects the intended propagation frequency; a real
sine at fixed nonzero Bloch wavevector contains both temporal frequencies.
The engine advances the supplied complex fields without automatically adding a
quadrature solve or normalizing incident power. For transmission ratios, use a
matching empty reference with the same source and Bloch phases.

Tests check an exact discrete Maxwell mode with arbitrary phases on all three
axes, monitor interpolation at a wrapped plane, phase derivatives against finite
differences, and FDTDX complex updates/material gradients. The derivative of a
fixed-field boundary problem is not automatically an incidence-angle derivative;
source/initial-field dependencies must be included in the differentiated function.
`examples/fdtd_bloch_gradient.py` varies the source and boundary together with
vacuum-end CPML and reports a spectral plane-power derivative.


## Backend architecture

The native solver now uses the same `array_api_compat` namespaces and
`_core.backend.as_array` conversion as RCWA. Maxwell curls, CPML, ADE, Bloch
wrapping and Fourier-plane accumulation have one implementation in
`_core/grid_fdtd.py`. Array creation/device selection and time-loop execution
live in `backends/time_domain.py`: a NumPy loop, JAX scan, or Torch checkpointed
loop. The physics code does not import JAX or Torch and has no backend-specific
update branches. Optional frameworks load only when their backend is selected.

`examples/fdtd_backends.py` evaluates one CPML/Lorentz objective with all three
backends and compares both automatic derivatives with a NumPy finite difference.
Torch tests also preserve live input graphs, compare checkpoint block sizes, and
exercise pole, material, source, initial-field and Bloch-phase derivatives. Existing
FDTDX non-dispersive, two-pole and complex-Bloch goldens are checked through Torch
as well as JAX; they remain frozen references, not a fresh external execution
on each test run. Use explicit double-precision input tensors for precision checks.

This unifies native-grid **calculation backends**. The existing `wavebit.fdtd`
Stack/Incidence adapter selects this engine when `backend` is explicitly supplied.
Integrating its rasterization, TFSF injection and modal Result extraction with
this shared core remains separate work; `fdtd(stack=..., backend=2)` is not yet
supported. Existing native `fdtd_grid` calls remain compatible.


## Common scene adapter

```python
import torch
import wavebit
from wavebit import Stack, Incidence

value = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
scene = Stack.from_structure(value.reshape(1, 1, 1), [.3],
                             top=1., bot=1., period=.4)
result = wavebit.fdtd(scene, Incidence(wavelength=1., tetm='te'),
                     backend=2, resolution=20, n_periods=30,
                     truncation_order=0)
derivative, = torch.autograd.grad(result.T.sum(), value)
```

The same adapter runs on NumPy (`0`), JAX (`1`) and Torch (`2`). JAX results are
registered pytrees, so a function returning the complete `Result` can be jitted.
Raster epsilon, raster mu, electric Drude/Lorentz parameters and Jones amplitudes
retain their backend graphs. Cell overlap weights are static; harmonic averaging
is applied normal to each field component and arithmetic averaging tangentially.
Pole masks use arithmetic fill fractions, including the normal component; this
is an approximation to a dispersive interface, not a general tensor homogenizer.

Supported scene inputs include rectangular one-dimensional and crossed rasters,
positive real magnetic half spaces, normal/oblique/conical incidence, Jones
polarization, electric pole labels, and scalar or vector wavelengths. Output
contains `given`, `s_inc`, `p_inc`, diffraction order metadata, complex r/t and
power R/T. Declared scalar TE/TM uses RCWA's fixed x-z frame; coupled Jones output
uses each outgoing order's own s/p frame. Reflection is referenced to the top
interface and transmission to the bottom interface.

Two electric-current basis sheets and two uniform incident-medium reference runs
calibrate incident polarization and subtract the incident field at the reflection
plane. All four runs share the same update core. This is reference normalization,
not a TFSF source. Extraction corrects Yee plane interpolation and uses discrete
homogeneous propagation to move the phase reference to the interfaces. At oblique
incidence the transverse wavevector stays fixed across the wavelength band;
`return_diagnostics=True` includes the corresponding angles.

Geometry, half-space coefficients, wavelength and incidence angles are static.
An explicit `resolution` is required so material optimization cannot resize the
grid. Defaults here are `courant=.8`, `n_periods=40`, `n_pml=12`; choose a smaller
Courant fraction when the minimum material values or pole frequencies demand it.
The time-step stability bound is checked by the native core. Concrete invalid
inputs raise; invalid traced JAX runs return NaN amplitudes and efficiencies, with
`stable=False` in optional diagnostics. Refine resolution,
PML width, padding and duration for each new structure, especially resonances.
Exact grazing orders and orders beyond the grid's sampling limit are refused.
Far-plane extrapolation of strongly evanescent amplitudes is ill-conditioned.

The explicit-backend adapter currently does not accept skewed lattices or geometry
derivatives; unsupported explicit-backend keywords raise an error. There is no implicit backend fallback. Actual GPU execution remains
unverified on this CPU-only machine.

Tests compare complex slab amplitudes with Airy under refinement, magnetic
conical Jones amplitudes and asymmetric diffraction with RCWA, a wavelength-band
transmission with stored Meep public-API data, and JAX/Torch material derivatives
with finite differences. Crossed dispersive scenes additionally check Torch/
NumPy agreement and a pole-strength derivative. These scene checks supplement
the native-core Meep/FDTDX field and derivative references.


### Shared-backend time records and user monitors

```python
result, diag = wavebit.fdtd(
    scene, Incidence(wavelength=1., tetm='te'), backend=2,
    resolution=20, n_periods=30, truncation_order=0,
    record_field=5, record_components='all',
    field_monitors={'inside': .15, 'after': .6},
    return_diagnostics=True,
)
trajectory = diag['fields']['ey']      # values(time,z,y,x), t/z/y/x axes
plane = diag['monitors']['after']     # six fields(wavelength,y,x), flux, power
```

Both options now use the common NumPy/JAX/Torch stepper. Records sample after
steps `0,N,2N,...`; E has time `(step+1)*dt` and H `(step+.5)*dt`. Each component
retains its native Yee coordinates. An invariant-y raster returns a singleton
y axis. The low-level `fdtd_grid` accepts the same recording options and retains
its full native grid. All six components are recorded by default when recording
is enabled; select a component or sequence to reduce storage. JAX scans across
record intervals; Torch checkpoints inside them. Unsampled forward fields are
not added to the record. Backpropagation and snapshot memory still need sizing.

User monitor coordinates are relative to the top interface (`z=0`). Monitors
interpolate the core's collocated node planes linearly in z, and must stay outside
PML and away from the current sheet. Fields are calibrated independently at each
requested wavelength to unit incident electric amplitude at the top interface.
`power_z` integrates the cycle-averaged normalized `E x H* / 2`; invariant-y
outputs are per unit y length. Spatial interpolation causes finite-grid errors;
refine the grid when comparing flux with modal diffraction efficiencies.

Time records use **constant centre-frequency calibration**, multiplied by the
pulse's carrier-frequency spectrum. They remain a causal complex carrier with
real physical field `Re(values)`. This fixes the incident Jones state at the
carrier wavelength. Away from that wavelength a fixed current sheet's emitted
polarization need not equal the independently normalized frequency-domain
monitor state. Applying a wavelength-dependent calibration directly to time
samples would not preserve this waveform. At the carrier, an offline record DFT
divided by the source spectrum matches the streaming normalized monitor.

The current sheet radiates in both directions. Records and user monitors contain
**total fields on both sides**, unlike the legacy TFSF scattered-field region.
Diffraction reflection still subtracts the separate uniform reference run.
Diagnostics name the normalization, temporal representation, source coordinate,
layer planes and PML boundaries. Existing NumPy plotting, animation and point
probe helpers accept these scene records and label the current sheet correctly.
These visualization/probe helpers remain NumPy postprocessing; convert/detach
JAX/Torch diagnostics explicitly before using them. Native record and monitor
arrays themselves preserve material and polarization gradients. JIT functions
can return selected array outputs; the descriptive diagnostic strings are not
JAX array outputs.

The recording tests propagate an independent exact discrete Yee eigenmode,
compare streaming and offline DFTs, check every recording-interval tail, compare
three backends, differentiate histories and monitor powers, preserve circular
quadrature, and exercise plot/probe compatibility. Enabling recording leaves the
diffraction result unchanged.


### Bottom illumination in the common engine

Set `Incidence(..., incidence_side='bottom')` or pass `incidence_side='bottom'`
with an explicit backend. The structure, top/bottom materials and layer ordering
remain in the original scene. Angles are measured relative to the incidence
medium's inward normal, matching RCWA. R is reflected power into that medium;
T leaves through the opposite side. Complex r is referenced to the bottom
interface and t to the top interface. `given`, `s_inc`, `p_inc` and order labels
keep their incidence-relative conventions.

The adapter reverses layer-associated epsilon, mu and dispersion labels, swaps
half spaces, and uses the same top-incident stepper. No backend duplicates the
field update. Recorded coordinates and monitor locations are mapped back with
`z_original = total_thickness - z_mirror`. E is a polar vector, so Ez changes
sign; H is axial, so Hx and Hy change sign. Returned z axes remain increasing.
The source/PML/layer markers use original coordinates. Time is not reversed.
User-plane `power_z` remains a laboratory-coordinate signed flux: transmission
from below normally has **negative** power_z, while diffraction T stays positive.

This route supports the same raster, positive magnetic medium, electric pole,
Jones state, record and monitor options as top incidence, including JAX/Torch
material gradients. It is independent of the legacy NumPy path, which still
refuses bottom incidence unless an explicit backend selects the common engine.
Grid refinement is needed for reciprocity comparisons: reflecting a stack whose
interfaces do not align to grid nodes changes its sampled interfaces. For an
asymmetric lossy two-layer test the matched-wavevector transmission difference
falls from 0.0083 at resolution 40 to 0.0018 at resolution 80.

Verification includes complex magnetic multilayer amplitudes against RCWA,
crossed dispersive layer reversal, physical field and flux signs, input
immutability, typed-scene gradients and a live Meep 1.34.0 reference. The Meep
reference places the source physically below the original stack; it does not
mirror the simulation. Three-wavelength r/t agree within 6e-4 in complex amplitude
at resolution 80. Increasing Meep's duration from 30 to 40 changes the reference
by less than 7e-15. Only documented Meep APIs were used; its implementation was
not read. See `studies/validate_scene_bottom_meep.py` and the frozen reference
in `tests/references/golden_scene_bottom_meep.json`.


### Static nonuniform z meshes

The shared engine now accepts `z_spacing`, a static positive vector with one
cell width per z index. It overrides `spacing[0]`; x and y stay uniform.
The last width is the closing periodic cell, so the complete z period is the
sum of all widths. Native E nodes start at zero and follow cumulative widths;
z-half components lie at each cell midpoint. Returned `z_nodes`, `z_spacing`
and the per-component record axes describe the actual geometry.

The magnetic update uses adjacent E-node distances; the electric update uses
the distances between adjacent H midpoints. The CFL bound uses the smallest
cell width. Plane interpolation weights depend on physical distances. CPML
and the optional matched conductivity absorber use each end's physical width,
so equal counts need not imply equal absorber thicknesses. Materials, poles,
fields and sources retain NumPy/JAX/Torch behavior and gradients. The mesh itself
is static and is not a differentiable design variable.

For `Stack` scenes, pass `grading=True` (ratio 1.06) or a ratio between 1 and 2:

```python
import numpy as np
import wavebit
from wavebit import Stack, Incidence

scene = Stack.from_structure(np.full((1, 1, 1), 2.25), [.3],
                             top=1., bot=1., period=.1)
result, diag = wavebit.fdtd(
    scene, Incidence(wavelength=1., tetm='te'), backend=1,
    resolution=40, grading=1.06, n_periods=30,
    truncation_order=0, return_diagnostics=True,
)
print(diag['shape'], diag['z_spacing'].min(), diag['z_spacing'].max())
```

The structure remains uniform in z, with both outer interfaces exactly on nodes.
Half-space padding grows gradually to at most twice the core width, capped by
20 cells per shortest in-medium wavelength. If the requested mesh is already
coarser than that cap, grading does not further coarsen it. This geometry policy
depends on static half spaces and resolution, not optimized raster values.
Existing `graded_cells` supplies the geometric ramps. E-node material averaging
uses the actual asymmetric dual-cell bounds; half-node values use primal cells.

Source and monitor indices are selected on the realized mesh. If the padding
cannot fit source, monitors and PML, increase `pad_scale`. CPML still counts
`n_pml` cells per end, with physical thickness measured from the mesh. Phase
extraction integrates local discrete homogeneous dispersion between each monitor
and its reference interface. Grading introduces mesh reflections; this phase
correction does not remove them. Refine resolution, grading, padding and duration
for new structures. The slab example at resolution40 uses 112 z cells instead
of 200; this is a cell-count saving, not a universal speedup guarantee.

Records, user planes, bottom incidence and material differentiation all work with
scene grading. Native-grid tests compare against an independent generalized
Yee eigenproblem, check physically weighted interpolation, uniform-grid parity,
CPML/ADE sensitivities and absorption. Scene tests check Airy refinement,
magnetic conical amplitudes against RCWA, existing independent Meep bottom-
illumination data, and JAX/Torch derivatives through graded records and monitors.
The Meep reference uses a uniform grid, so the comparison establishes optical
agreement under refinement rather than identical discretizations. No Meep
implementation was read.

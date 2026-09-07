
# A solve in the time domain

`wavebit.fdtd()` is the second solver. It reads the same `Stack` and `Incidence` the RCWA path
reads and hands back the same `Result`, so one scene is solved two ways with nothing translated
between the answers.

This path is a **research alpha**. Read the [supported scope and validation evidence](../fdtd-alpha.md)
before using a new combination of geometry, incidence and material.

Two things about the door differ from `rcwa()` and are worth having before the first line of
code. It returns **the answer, not a solver** — there is no eigendecomposition to cache, no
frozen geometry and no gradient, so there is no per-instance state an object would hold. And it
runs on **NumPy only**: the time loop is a Python `for` over ~10^4 steps, which needs a `scan`
primitive the backend layer does not have, so `backend=` is refused by name rather than
half-supported.

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

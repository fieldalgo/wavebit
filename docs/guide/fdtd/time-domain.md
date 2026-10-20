# A solve in the time domain

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


:::{note}
Four examples use the current `wavebit.fdtd` module: the first two, the material derivative under
[the common scene adapter](#common-scene-adapter) and [source placement](#source-position-independent-of-padding).
The others still call the earlier periodic factory `wavebit.compat.legacy.fdtd`, or its native grid `fdtd_grid`,
because raster layers, whole-field time records, plotting, mode fitting, graded or layer-fitted meshes, angle,
shape, thickness and half-space derivatives, Jones polarization and that route's source, extraction and
convergence controls are not in the current module yet; each such example says so in its first line.
:::

`wavebit.fdtd` is the second solver. A periodic stack lit by a plane wave is an `fdtd.Problem`
like any other: `Periodic()` x and y faces (or the `Bloch` faces `Bloch.from_source` writes for
an oblique wave), `Open()` z faces with a PML added outside them, the layers as structures, one
`PlaneWave`, and a `DiffractionMonitor` on each side. Each monitor returns the per-order
complex s/p amplitudes and efficiencies RCWA returns, normalized by the plane wave's own
measured incident field. [How the solver models a problem](model.md) describes each piece.

The earlier periodic factory, `wavebit.fdtd(stack=..., incidence=...)`, read the same `Stack`
and `Incidence` as the earlier RCWA factory and returned the same `Result` container. It is
retained as `wavebit.compat.legacy.fdtd(...)`. With no `backend` argument it runs the legacy
NumPy solver, including time records and graded meshes; an explicit `backend="numpy"`, `"jax"`
or `"torch"` selects the shared Yee engine and supports material automatic differentiation.
See [the common scene adapter](#common-scene-adapter) for its supported inputs, and
[route parity](../../validation/fdtd/fdtd-route-parity.md) for how the two compare on the same scenes.

This page covers periodic stacks. For `from wavebit.compat.legacy import fd` and its
explicit `fd.Scene` domain, sources, boundaries and named monitors, see the
[general FDTD guide](../../migration/general-fdtd-v1.md). The scene types and conventions are
separate.

Both are a **research alpha**. Read the [supported scope and validation evidence](fdtd-alpha.md)
before using a new combination of geometry, incidence and material.

## Shared periodic conventions

In the current module both solvers take lengths in the problem's `Units` and angles in
radians, with +z upward. In FDTD a layer is a cell-snapped `SnappedSlab(z=(lo, hi))` or a smoothed
`Box`; a half-space is a `SnappedSlab` that reaches an `Open()` face and continues into that face's
PML, and the scene's `background` fills every cell no structure claims. A `PlaneWave` with
`direction='-z'` (the default) lights the stack from above and `'+z'` from below; its
`polarization` is `'s'` or `'p'`. A `DiffractionMonitor`'s `amplitude_s` and `amplitude_p`
use each outgoing order's own s/p frame with the cell centre as lateral phase origin, its
`efficiency` is a power fraction, and `reference_plane` and `incident_plane` name the planes
its phases refer to (the monitor's own plane by default). Fields are E in V/m and H in A/m.

The earlier factories differ: `wavebit.compat.legacy.rcwa(...)` returns a solver whose
`.solve()` returns a result, and `wavebit.compat.legacy.fdtd(...)` returns the result directly.
The general `fd` scene API and the native-grid `fdtd_grid` API have their own input contracts.
On the two earlier periodic routes:

- Layers are **bottom-to-top**. The bottom interface is `z=0`, the top is
  `z=sum(thickness)`, and +z points upward. `top` and `bot` always name the
  physical upper and lower half-spaces. Both periodic engines support
  `incidence_side='top'` and `'bottom'` through the public adapter.
- Angles are radians. Both engines require real incidence; the backend-omitted
  engine accepts signed scalar `theta`, while explicit backends require
  `0 <= theta < pi/2`. On the explicit route, use azimuth `phi` to select the
  transverse direction. Cartesian `kx`/`ky` incidence is refused.
- `result.incident` is the requested polarization state. TE/TM declarations
  accept `'te'`, `'tm'` or `wavebit.compat.legacy.Tetm` members, with case-insensitive strings.
  Complex `r_s`, `r_p`, `t_s`, `t_p` use each outgoing order's own s/p frame,
  including pure TE/TM, and the cell centre as their lateral phase origin.
  Reflection is referenced to the illuminated face and transmission to the
  opposite face. R/T are power fractions; `power_entering` and `absorption`
  use the real positive incident half-space supported by these engines.
- Cartesian order amplitudes `r_xyz`/`t_xyz` and RCWA reconstruction state are
  not supplied by periodic FDTD. Use its recorded fields or frequency monitors.

Backend names are `'numpy'`, `'jax'` and `'torch'`, or the corresponding
`wavebit.Backend` members; integer selectors are refused. Omitting `backend`
selects the legacy NumPy engine. This differs from explicitly selecting `'numpy'`.

The result's scattering amplitudes share a normalization across the two engines;
raw diagnostic fields do not:

| Diagnostic | Backend omitted | Explicit backend |
| --- | --- | --- |
| Frequency monitor fields | Divided by the soft-source pulse spectrum; use a matching empty reference for incident normalization | Calibrated at each wavelength to unit incident E at the illuminated face |
| Time records | Propagated source waveform, including the requested Jones state | Causal carrier with constant centre-frequency calibration |
| Field region | TFSF total/scattered regions, identified by `region` | Total fields on both sides of the current sheet |

H is stored as `Z0 H` on both periodic routes, and `power_z` is signed along
public +z: transmitted power from above is negative, from below positive.
Diffraction T remains a nonnegative outgoing power fraction. Read each field's
native `t/z/y/x` axes and the public `source_z`, `layer_planes`, `z_nodes` and
`pml_bounds` metadata. A legacy diagnostic `grid` is an internal numerical grid,
not the public coordinate map. Invariant-y fields are sampled at `y=0`; their
phase does not change when an unused second period is changed.

For migration details and the exact integration scope, see the
[solver integration checkpoint](../../development/solver-integration.md).

## The same scene, both ways

```python
import numpy as np
from wavebit import Material, Units, fdtd, rcwa

units = Units(length='nm')
air, glass, silicon = Material(epsilon=1.), Material(epsilon=1.5**2), Material(epsilon=3.5**2)

# A 700 nm cell centred on x = 0, lit from above at 900 nm; a 300 nm ridge fills x = 0..350 nm.
harmonic = rcwa.run(rcwa.Problem(
    units=units,
    stack=rcwa.Stack(period=(700., 700.), top=air, bottom=glass, layers=(rcwa.Layer(
        thickness=300., background=air, regions=(rcwa.Region(
            geometry=rcwa.Rectangle(size=(350., 700.), center=(175., 0.)), material=silicon),)),)),
    excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization='s'),))),
    numerics=rcwa.Numerics(orders=(40, 0)))

# The same cell, one cell deep in y; the substrate continues into the bottom PML.
problem = fdtd.Problem(
    units=units,
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(700., 10., 1200.), center=(0., 0., 150.)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        structures=(
            fdtd.Structure('substrate', shape=fdtd.SnappedSlab(z=(-450., 0.)),
                           material=glass),
            fdtd.Structure('ridge',
                           shape=fdtd.SnappedBox(size=(350., 700., 300.), center=(175., 0., 150.)),
                           material=silicon))),
    sources=(fdtd.PlaneWave('light', plane=fdtd.Plane(z=650.), polarization='s',
                            pulse=fdtd.GaussianPulse(wavelength=900., frequency_sigma_hz=2e13)),),
    outputs=fdtd.Outputs(monitors=(
        fdtd.DiffractionMonitor('r', plane=fdtd.Plane(z=500.), wavelengths=(900.,),
                                source='light', wave='reflected', orders=(1, 0),
                                reference_plane=300., incident_plane=300.),
        fdtd.DiffractionMonitor('t', plane=fdtd.Plane(z=-300.), wavelengths=(900.,),
                                source='light', wave='transmitted', orders=(1, 0),
                                reference_plane=0., incident_plane=300.))))
time_domain = fdtd.run(problem, numerics=fdtd.Numerics(
    grid=fdtd.UniformGrid(spacing=10.),                                  # 10 nm cells
    duration=fdtd.Duration(periods=80, source='light')))

reflected, transmitted = time_domain['r'], time_domain['t']
for m in (-1, 0, 1):
    a, b = harmonic.transmission.order(m).efficiency[0], transmitted.order(m).efficiency[0]
    print(f'T[{m:+d}]   rcwa {a:.6f}   fdtd {b:.6f}   gap {abs(a - b):.1e}')
print(f'R + T    rcwa {harmonic.R.sum() + harmonic.T.sum():.6f}   '
      f'fdtd {reflected.efficiency.sum() + transmitted.efficiency.sum():.6f}')
print(f'T[+1] - T[-1] = '
      f'{abs(transmitted.order(1).efficiency[0] - transmitted.order(-1).efficiency[0]):.1e}')
```

```
T[-1]   rcwa 0.221306   fdtd 0.218569   gap 2.7e-03
T[+0]   rcwa 0.352483   fdtd 0.350980   gap 1.5e-03
T[+1]   rcwa 0.221306   fdtd 0.218569   gap 2.7e-03
R + T    rcwa 1.000000   fdtd 1.000014
T[+1] - T[-1] = 2.8e-17
```

The two problems share their `Material` and `Units` objects, not a scene: RCWA reads a
[`Stack`](../rcwa/scene.md) of layers, FDTD a domain of structures. Every face here lies on a
cell boundary, so `fdtd.check` reports no snapped face. Each `DiffractionMonitor` returns what
RCWA's `result.transmission` and `result.reflection` return — `order(m).efficiency` and the
complex `amplitude_s` and `amplitude_p` with a leading wavelength axis — with the phases moved
to the faces named by `reference_plane` and `incident_plane`, as RCWA's are. The FDTD numbers
were regenerated with the current module on 2026-10-15; the earlier factory printed a 2.0e-3
gap on the same 10 nm grid. The gap is the grid's: it is second order, so a 5 nm cell brings
the largest to 7.0e-4 (the earlier route measured ~7e-4 at 5 nm and ~1.9e-4 at 2.5 nm).

**`R + T` is not exactly 1, and that is the method rather than a defect.** A Yee grid conserves
energy to `O(dz^2)`, the same order it computes everything else to, so a lossless scene comes
back a little off 1; the RCWA row above is 1 to a few times 1e-15 for contrast. There is
deliberately no generic guard on it here — one existed and fired on 85 of 243 ordinary
time-domain solves, which is why it was removed ([`result-guards.md`](../../reference/common/result-guards.md)).

**`T[+1] == T[-1]` is a symmetry check for this scene at normal incidence.** A symmetric
discretization can preserve that equality even on a coarse grid, so it does not bound the
error in either efficiency. Use grid and runtime refinement to assess accuracy.

## Which of the two to reach for

RCWA answers a single wavelength between 112 and 2850 times faster at matched accuracy, so the
question is only ever how many wavelengths you want. Matched on accuracy rather than on settings
— the coarsest cell and the cheapest truncation (`rcwa.Numerics(orders=...)`; `truncation_order`
on the earlier interface this table was measured with) that reach the same worst per-order
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

Reach for `rcwa` for harmonic fields (`rcwa.Outputs(fields=True)` and `result.field(...)`), a
skewed `lattice`, elliptical regions (the current FDTD module has `Cylinder` and `Prism` bodies
but no ellipse; the earlier explicit route's [area recipes](#rotated-rectangle-area-integration)
take one circle or ellipse per layer) and evanescent (complex-angle) incidence. The current
FDTD module differentiates material values, source parameters and the geometry of ranged
smoothed bodies on JAX and PyTorch ({ref}`differentiating a run <fdtd-model-differentiating>`);
the earlier explicit-backend route adds the angle, shape and thickness derivatives described
below. The current `PlaneWave` takes `'s'` or `'p'`. Elliptical diffraction is available on
the earlier factory using `psi` with `chi`, or `polarization=(a_s, a_p)`; for example `(1, 1j)`
is a circular state in the s/p frame. The pair is normalized while preserving its overall phase.
Time recording also supports complex coefficients through additional quadrature-source
runs; see [polarization details](fdtd.md#elliptical-and-circular-incidence).

## The keywords a time-domain run adds

On the earlier factory the scene keywords are those of the earlier RCWA factory — `structure`,
`thickness`, `top`, `bot`, `period`, `mu`, `wavelength`, `theta`, `phi`, `tetm`, `psi`.
**Angles are radians**, as everywhere in wavebit.
The table below gives defaults for the **backend-omitted NumPy route**. Explicit-backend
defaults are listed under [the common scene adapter](#common-scene-adapter). These defaults were measured, and
[`fdtd.md`](fdtd.md) is the measurement.

| Keyword | Default | What it is for |
| --- | --- | --- |
| `resolution` | 20 cells per wavelength **in the densest material in the scene** | cells per unit length, so `resolution=0.1` is a 10 nm cell when lengths are nm. The knob that costs: doubling it halves the nominal cell size and can cost roughly 16x the work in full 3D (8x the cells and 2x the steps). The convergence rate depends on the geometry; sharp corners need a measured refinement study. A pole model raises the index it is sized against, so a Drude metal gets a fine grid without being asked for |
| `n_periods` | `80` | optical cycles to run. The pulse has to enter, scatter and leave before the running DFT means anything: at 40 the transmission monitor still holds 61% of its peak and the answer is wrong by 0.09; at 80 the residual is 5e-6 and the answer no longer moves. A run under 48 cycles is refused outright — the source has not finished emitting — and one merely too short to settle warns `UnsettledFieldWarning` and reports `residual`. An oblique or near-grazing scene often needs twice a normal-incidence one |
| `n_pml` | `12` | absorber depth in **cells**, not in length. A correctly graded CPML is dz-independent, which is what makes cells the right unit: at 12 the layer reflects 1.1e-5, and 20 buys 1.5e-6. Fewer than 2 is refused. Thinner layers above that are accepted but not safe: on a plain slab, 2 cells returned T = 5.03 where RCWA gives 0.885, with no warning. To change it, double it and check that the answer holds |
| `pml_reflection` | `1e-8` | the absorber's strength, as the nominal round-trip amplitude its grading aims at; `n_pml` is its depth. Applied to both absorbers and to the incident wave's own. See [CPML strength](#cpml-strength) |
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

The current module has no defaults of this kind: the grid and the duration are always
declared, and `check` admits or refuses the declared scene by the
{ref}`accuracy rules <fdtd-model-accuracy-rules>`. Where each keyword went, and
what differs:

| Earlier keyword | Current module | Default and rule there |
| --- | --- | --- |
| `resolution` | `fdtd.UniformGrid(spacing=dx)`, one cubic spacing on every axis | none; `check` refuses fewer than 8 cells per wavelength in any material and warns below 12 |
| `n_periods` | `fdtd.Duration(periods=N, source=name)`, or `seconds=` or `steps=` | none; no minimum, and `UnsettledFieldWarning` when a monitored region still holds field at the end |
| `n_pml` | `fdtd.PML(cells=12)` in `Numerics(absorber=...)` | 12; fewer is refused unless `allow_thin=True` |
| `pml_reflection` | `fdtd.PML(reflection=1e-8)` | the same default value |
| `courant` | `fdtd.Numerics(courant=0.5)` | 0.5 of the three-dimensional vacuum bound `dx / (c√3)` |
| `subpixel` | the shape decides: `SnappedSlab` and `SnappedBox` are cell-snapped, `Box`, `Sphere`, `Cylinder` and `Prism` smoothed | — |
| `grading` | not available; the grid is uniform | — |
| `dispersion`, `poles` | `DispersiveMaterial(epsilon_inf, poles=(Drude(...), Lorentz(...), Debye(...)))`, rates in SI | — |
| `record_field` | `fdtd.TimeSeriesMonitor(name, at=..., component=...)`: one component at one Yee point | — |
| `pad_scale` | the `Domain` size; each PML is added outside it | `AbsorberProximityWarning` closer than half a wavelength in the structure's own medium |
| `truncation_order` | `fdtd.DiffractionMonitor(orders=...)` | every propagating order |
| `return_diagnostics` | `result.validity`, `result.diagnostics` and `result.provenance`, always present | — |
| `wavelength` (a band) | each frequency-domain monitor's `wavelengths` | — |

The current module also refuses a diffraction order that leaves through a PML at a grazing
angle the default absorber cannot hold, and names the deeper `PML` and longer `Duration` that
admit it; the next example meets that rule.

## Dispersion, and a band out of one run

A time-domain solve cannot take a complex permittivity — one complex number describes a material
at one frequency, and the pulse carries a whole band. The frequency dependence is given as the
thing it is, a pole model, and each wavelength in the band then sees the permittivity that model
has *there*. That is the combination the break-even table is measuring: one run, against one
RCWA solve per wavelength each handed a different `eps`.

```python
import numpy as np
from wavebit import DispersiveMaterial, Drude, Material, Units, fdtd, rcwa
from wavebit.units import C0

units = Units(length='nm')
# Gold-ish Drude, one pole, with SI rates: omega_p = 2 pi c / 136 nm, gamma = 2 pi c / 17 um.
gold = DispersiveMaterial(epsilon_inf=1., poles=(
    Drude(omega_p=2 * np.pi * C0 / 136e-9, gamma=2 * np.pi * C0 / 17e-6),))
air, glass = Material(epsilon=1.), Material(epsilon=1.5**2)
band = (820., 900., 980.)

# A 60 nm gold strip over x = -350..0 nm of a 700 nm cell, on glass, lit from above.
problem = fdtd.Problem(
    units=units,
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(700., 10., 1050.), center=(0., 0., 75.)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        structures=(
            fdtd.Structure('substrate', shape=fdtd.SnappedSlab(z=(-450., 0.)),
                           material=glass),
            fdtd.Structure('strip',
                           shape=fdtd.SnappedBox(size=(350., 700., 60.), center=(-175., 0., 30.)),
                           material=gold))),
    sources=(fdtd.PlaneWave('light', plane=fdtd.Plane(z=500.),
                            pulse=fdtd.GaussianPulse(wavelength=900., frequency_sigma_hz=2e13)),),
    outputs=fdtd.Outputs(monitors=(
        fdtd.DiffractionMonitor('r', plane=fdtd.Plane(z=350.), wavelengths=band,
                                source='light', wave='reflected'),
        fdtd.DiffractionMonitor('t', plane=fdtd.Plane(z=-300.), wavelengths=band,
                                source='light', wave='transmitted'))))
# The -1 and +1 orders leave the glass at cos(theta) = 0.36 at 980 nm, a grazing exit the
# default PML refuses; the refusal asks for this deeper absorber and at least 160 periods.
spectrum = fdtd.run(problem, numerics=fdtd.Numerics(                      # one run
    grid=fdtd.UniformGrid(spacing=10.), duration=fdtd.Duration(periods=160, source='light'),
    absorber=fdtd.PML(cells=24, reflection=6e-10)))

reference = rcwa.run(rcwa.Problem(                                        # one solve per wavelength
    units=units,
    stack=rcwa.Stack(period=(700., 700.), top=air, bottom=glass, layers=(rcwa.Layer(
        thickness=60., background=air, regions=(rcwa.Region(
            geometry=rcwa.Rectangle(size=(350., 700.), center=(-175., 0.)), material=gold),)),)),
    excitation=rcwa.Excitation(wavelength=np.array(band),
                               sources=(rcwa.PlaneWave(polarization='s'),))),
    numerics=rcwa.Numerics(orders=(60, 0)))

eps = gold.at(np.array(band), units=units).epsilon   # the closed form the stepping discretises
R = spectrum['r'].efficiency.sum(axis=(1, 2))
T = spectrum['t'].efficiency.sum(axis=(1, 2))
for i, wl in enumerate(band):
    print(f'{wl:.0f} nm  eps {eps[i].real:8.3f}{eps[i].imag:+.3f}j   '
          f"T0 fdtd {spectrum['t'].order(0).efficiency[i]:.4f} "
          f'rcwa {reference.transmission.order(0).efficiency[i]:.4f}   '
          f'absorbed fdtd {1 - R[i] - T[i]:.4f} '
          f'rcwa {1 - reference.R[i].sum() - reference.T[i].sum():.4f}')
```

```
820 nm  eps  -35.269+1.749j   T0 fdtd 0.4073 rcwa 0.4049   absorbed fdtd 0.0107 rcwa 0.0111
900 nm  eps  -42.671+2.312j   T0 fdtd 0.4090 rcwa 0.4060   absorbed fdtd 0.0126 rcwa 0.0131
980 nm  eps  -50.753+2.983j   T0 fdtd 0.4206 rcwa 0.4172   absorbed fdtd 0.0149 rcwa 0.0154
```

The FDTD columns were regenerated with the current module on 2026-10-15 (the RCWA columns are
unchanged to 1e-12); the earlier factory, at 100 periods and its own 12-cell absorber, printed
T0 0.4036, 0.4039, 0.4144 and absorbed 0.0112, 0.0135, 0.0162. The cell-snapped strip also draws
an `AbsorberProximityWarning`: that rule sizes its half-wavelength clearance with the real part
of the metal's index, which is small, so it asks for micrometres; the 450 nm and 540 nm gaps
here are at least the half wavelength the earlier route padded with in the glass (300 nm) and
the air (450 nm).

The band is an identity rather than an approximation: each wavelength is its own Fourier
transform of the one run, and a run asking for 900 nm alone, with the same pulse, returns the
same efficiencies to 1e-14. The earlier factory sized its pulse from the requested span, and
three separate single-wavelength runs there agreed with its band to 3e-3, the gap between two
source *bandwidths* rather than between two answers. Here the pulse is declared: its
`frequency_sigma_hz` sets the band, every amplitude is divided by the measured incident
spectrum, and a wavelength where the pulse has fallen below its `cutoff` comes back NaN with
`result.validity.checks['source_reference']` false. That is the guard against what the earlier
route once measured with a fixed pulse width: an incident spectrum 2.4e-7 of its peak returned
`R = 105` on a scene whose true `R` is 0.04.

**At oblique incidence a band holds the physical transverse `(kx, ky)` fixed.**
The polar angle therefore varies with wavelength: about 28.5 degrees at 860 nm
and 31.5 at 940 nm for a 30-degree centre at 900 nm. In the current module the
`Bloch` faces from `Bloch.from_source` fix it at the pulse's centre wavelength, and
each `DiffractionMonitor` result reports the polar angles in radians in
`metadata['theta_radians']`. On the earlier factory `diagnostics['angles']`
reports the polar angles in radians and `azimuth` reports phi. The backend-omitted
route also retains `angles_deg` as the degree-valued compatibility view of `angles`.
Normal incidence is unaffected,
`sin 0` being zero at every wavelength, so a fixed-angle band is available there and only there.

## Watching the field arrive

The one thing this solver states that a harmonic one cannot. A harmonic solve answers what the
field *is*, having assumed `e^{-iwt}` and therefore assumed a single frequency describes the
whole problem; there is no picture of light arriving, because the assumption is that it always
has been. `record_field=N` keeps every N-th step of the electric field, **with its own
abscissae** — a reader who rebuilds them is wrong three ways: a graded mesh has no single `dz`,
in-plane coordinates are measured from the centre of the cell, and `Ey` sits on the x nodes
rather than at the pixel centres.

The current module keeps the time series of one component at its nearest Yee point,
`fdtd.TimeSeriesMonitor(name, at=..., component=...)` (times in seconds, fields in V/m or A/m), and the final E
and H, `fdtd.Outputs(final_fields=True)`, each with the coordinates it was sampled at. It does
not record the whole domain over time yet, so this example and the plotting below keep the
earlier factory.

```python
# Earlier interface: whole-domain time records are not in the current API yet.
import numpy as np
from wavebit.compat import legacy as wavebit
from wavebit.compat.legacy import Stack, Layer, Incidence

eps_hi = wavebit.eps_from_refractive_index(3.5)
scene = Stack(layers=[Layer(thickness=300.,
                            structure=np.array([[1., 1., eps_hi, eps_hi]]))],
              top=1., bot=wavebit.eps_from_refractive_index(1.5), period=(700.,))
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
field_z   -420.0 .. 1110.0
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
centres `wavebit.compat.legacy.meshgrid_xy` returns are half a cell from where `Ey` lives. The measurements
behind all three are in [`fdtd.md`](fdtd.md) §8h.

The same record can be plotted or animated with optional Matplotlib
(`pip install 'wavebit[plot]'`):

```python
# Earlier interface: plot_fdtd_field and animate_fdtd_field are not in the current API yet.
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
on the incident side to total field toward the stack. For top incidence this is
scattered field above and total field below; bottom incidence reverses those regions.
Layer planes are marked, without implying an in-plane geometry.

Time is in the solver's `c=1` units, the same units used for lengths. Playback `interval`
is milliseconds per snapshot and does not change simulated time. An undersampled record
warns: reduce `record_field` in the solve, since slowing the animation cannot recover missing
optical cycles. `record_field` must be a positive integer.

With `psi`, the record contains the requested coherent combination of the two source
polarisations, including both Ey and Ex. The amplitudes retain the raw source scale; they
are not normalized monitor coefficients. The settling diagnostic reports the larger
residual of the two basis runs. Recording is still off by default.

## Where to read further

[`fdtd.md`](fdtd.md) is the decision document: what each default was measured against,
the full list of refusals and what each one protects, and the two questions a time-domain
solver can state that a harmonic one cannot. [`result-guards.md`](../../reference/common/result-guards.md) says
what a solve checks about its own answer, and why there is no generic energy warning.
[How the solver models a problem](model.md) covers the current module: its grid, structures,
boundaries, sources, monitors and the accuracy rules a run must pass.


To record every electric and magnetic component, add `record_components='all'`.
Read `diag['fields']['hz']['values']` and its `t/z/y/x` axes, or pass `component='hz'`
to the plotting/animation helpers. H is normalized as `Z0 H`; its timestamps differ
from E by half a time step. See [the component layout](fdtd.md#recording-all-six-field-components)
before combining staggered fields.


`wavebit.compat.legacy.fdtd_poynting(diag, z=50., x=0., y=0.)` returns a point's instantaneous
energy flow in `S[:, 0:3]`, after interpolating all six fields to common Yee positions
and times. `sample_fdtd_fields` returns the interpolated E/H directly. Both require
`record_components='all'`; see [point probes](fdtd.md#point-probes-and-instantaneous-energy-flow)
for units, sampling limits and the distinction between total and scattered flux.


## Streaming frequency planes

In the current module, an `fdtd.FieldMonitor(name, wavelengths=..., plane=fdtd.Plane(z=...))`
returns the raw Fourier phasors of the six components on their own Yee points (V·s/m and
A·s/m), and an `fdtd.FluxMonitor` the net spectral energy through a plane (J/Hz); neither is divided by
the source spectrum. The rest of this section describes the earlier factory.

Pass `field_monitors={'near': z_near, 'transmitted': z_out}` together with
`return_diagnostics=True` to `wavebit.compat.legacy.fdtd`. No `record_field` is needed. Each entry
of `diag['monitors']` contains six complex `fields` arrays `(wavelength, y, x)`,
`x`/`y` coordinates, `flux_density` `(wavelength, y, x, 3)` and `power_z`
`(wavelength,)`. Spatial interpolation collocates native Yee components, including
Bloch phases at lateral edges; Fourier integration uses each component's native
time. Complex Jones coefficients combine fields before computing power.

This section describes **backend-omitted** monitor normalization. It is the
soft-source pulse spectrum, **not incident power**. Explicit-backend calibration
is described under [shared-backend records and monitors](#shared-backend-time-records-and-user-monitors).
With H stored as Z0 H, flux is `0.5*real(E cross conj(H))`; `power_z` integrates
its z component over the periodic cell, or per unit y depth for a y-invariant
scene. Run the matching empty scene to normalize transmission. On a scattered
reflection plane, negate its power and divide by the empty incident power.
A total-field plane contains incident/reflected interference; it is not a
separated reflection measurement. `region` identifies the side of the TFSF
boundary. Planes whose interpolation touches that boundary are refused.
These local interpolated fluxes converge with mesh refinement and need not equal
modal diffraction powers exactly at a finite resolution.

See `examples/fdtd/fdtd_frequency_planes.py` for a three-wavelength slab check.
Storage scales with six components times wavelengths times plane area; it does
not grow with the number of time steps. These are streaming Fourier measurements,
not a whole-volume harmonic field reconstruction.

## Shared NumPy, JAX and Torch native grids

`wavebit.compat.legacy.fdtd_grid` is a Cartesian Yee interface sharing one Array API calculation
core across NumPy, JAX and Torch. The `backend` selector accepts `'numpy'`, `'jax'`,
`'torch'` or a `wavebit.Backend` member, as RCWA does; integers are refused.
Its native grid coordinates do not pass through the periodic Stack adapter.
It accepts native `(z,y,x)` permittivity, or `(3,z,y,x)` diagonal coefficients,
and `(3,z,y,x)` initial E/H and electric-current source arrays. Specify
`spacing=(dz,dy,dx)`, `dt` and `steps` explicitly, in c=1 units. E starts at zero
time and H at `-dt/2`. Scalar or broadcastable positive permeability and nonnegative
electric conductivity are supported. Coefficients belong on each component's
native Yee location; the function does not rasterize or average interfaces.

```python
# Earlier interface: a native grid of per-cell coefficients is not in the current API yet.
import jax
import jax.numpy as jnp
from wavebit.compat import legacy as wavebit

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
Torch uses eager autograd with non-reentrant block checkpointing; under
`torch.compile` it runs eagerly, outside the compiled graph (see
[`torch.compile`](../../development/torch-compile.md)).
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
[validation scope](fdtd-alpha.md). Those historical comparisons used CPU.
Actual Torch CUDA evidence for the current public native API is recorded
separately in [native CUDA validation](../../validation/fdtd/native-cuda.md).
`examples/fdtd/fdtd_grid_gradient.py` differentiates slab transmission;
`studies/fdtd/grid_benchmark/fdtd_grid_benchmark.py` reports compilation separately from warm execution;
its measurements, and why the speed-up belongs to the host, are in
[the design notes](fdtd.md#streaming-fields-and-compiled-native-grids).


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
positions. The end layers **and one adjacent interior cell** must match the
`pml_background=((epsilon_top, mu_top), (epsilon_bottom, mu_bottom))` values
(default: vacuum)
(epsilon=mu=1), zero electric conductivity and zero oscillator strength. x/y
remain periodic. Cannot combine this with `absorber_cells`; the latter remains
a simpler graded damping option. Keep physical sources and measurement planes
in the interior. Four CPML memory arrays are retained, each with the spatial
volume shape. Geometry and center wavelength are static; the positive real
`pml_background` pairs can retain gradients. The CPML stretch follows their
refractive indices, so exterior-material derivatives include the absorbing
updates. End coefficients must still match the supplied backgrounds exactly.

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
`examples/fdtd/fdtd_dispersive_gradient.py` differentiates Lorentz oscillator strength
with CPML enabled. The historical external comparisons above remain CPU
evidence; the current API has separate [bounded CUDA field and derivative checks](../../validation/fdtd/native-cuda.md).

### CPML strength

In the current module the strength is `fdtd.PML(reflection=1e-8)` and the depth
`fdtd.PML(cells=12)`, both set once for every `Open()` face through `Numerics(absorber=...)`.
On the earlier interface, `fdtd_grid`, `wavebit.compat.legacy.fdtd(..., backend=...)` and
`wavebit.compat.legacy.fdtd` with `backend` omitted all accept `pml_reflection=1e-8`. It must
be a finite real static scalar strictly between zero and one; booleans, complex values and arrays
with dimensions are refused. The same setting applies at both ends, including
bottom incidence, and to all four scene calibration/structure runs; with
`backend` omitted it also grades the absorbers of the one-dimensional grid that
carries the incident wave. It is recorded as `pml_reflection` in the native
result or scene diagnostics. It has no effect when native `pml_cells=0`.

This is a **nominal normal-incidence round-trip amplitude target**, defined for
the continuous profile before its frequency shift. It is neither a power target
nor a bound on measured reflection. Smaller values increase the polynomial
conductivity without changing its order, kappa, frequency shift, cell widths or
PML depth. `pml_reflection=1e-40` gives five times the conductivity of `1e-8`;
it does not imply forty-digit accuracy. The default profile is unchanged.

Use `n_pml` (scene) or `pml_cells` (native grid) to vary thickness independently
of strength. Stronger absorption can increase reflection on an under-resolved
grid. Compare strength, physical thickness and resolution, and recheck temporal
decay after changing the boundary. In a directional scene, returning amplitudes
and calibrated scattering tails help distinguish boundary and time errors.
The [near-Rayleigh strength study](../../validation/fdtd/fdtd-pml-strength.md) records a concrete
comparison and its remaining limits.


## Native Bloch phases and incidence gradients

In the current module a `Bloch(phase=...)` or `Bloch(k=...)` face pair uses the same
convention and makes the run complex, and `Bloch.from_source` writes the pair an oblique
`PlaneWave` needs. The rest of this section describes the earlier native grid, `fdtd_grid`.

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
`examples/fdtd/fdtd_bloch_gradient.py` varies the source and boundary together with
vacuum-end CPML and reports a spectral plane-power derivative.


## Backend architecture

The native solver now uses the same `array_api_compat` namespaces and
`wavebit._common.backend.as_array` conversion as RCWA. Maxwell curls, CPML, ADE, Bloch
wrapping and Fourier-plane accumulation have one implementation in
`src/wavebit/fdtd/_periodic/grid.py`. Array creation/device selection and time-loop execution
live in `src/wavebit/fdtd/_periodic/runtime.py`: a NumPy loop, JAX scan, or Torch checkpointed
loop. The physics code does not import JAX or Torch and has no backend-specific
update branches. Optional frameworks load only when their backend is selected.

`examples/fdtd/fdtd_backends.py` evaluates one CPML/Lorentz objective with all three
backends and compares both automatic derivatives with a NumPy finite difference.
Torch tests also preserve live input graphs, compare checkpoint block sizes, and
exercise pole, material, source, initial-field and Bloch-phase derivatives. Existing
FDTDX non-dispersive, two-pole and complex-Bloch goldens are checked through Torch
as well as JAX; they remain frozen references, not a fresh external execution
on each test run. Use explicit double-precision input tensors for precision checks.

This unifies native-grid **calculation backends**. The `wavebit.compat.legacy.fdtd`
Stack/Incidence adapter selects this engine when `backend` is explicitly supplied.
Its static rasterization, current-sheet reference calibration and modal Result
extraction are described below. Existing native `fdtd_grid` calls remain compatible.


## Common scene adapter

The current module differentiates live material values on the JAX and PyTorch backends, here a
slab's permittivity through the whole run and the `DiffractionMonitor` normalization:

```python
import torch
from wavebit import Material, Units, fdtd

value = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
problem = fdtd.Problem(
    units=Units(length='um'),
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.4, .05, 2.), center=(0., 0., .15)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        structures=(fdtd.Structure('slab', shape=fdtd.SnappedSlab(z=(0., .3)),
                                   material=Material(epsilon=value)),)),
    sources=(fdtd.PlaneWave('light', plane=fdtd.Plane(z=.9),
                            pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=2e13)),),
    outputs=fdtd.Outputs(monitors=(
        fdtd.DiffractionMonitor('t', plane=fdtd.Plane(z=-.5), wavelengths=(1.,), source='light',
                                wave='transmitted', orders=(0, 0)),)))
result = fdtd.run(problem, backend='torch', numerics=fdtd.Numerics(
    grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(periods=30, source='light')))
derivative, = torch.autograd.grad(result['t'].efficiency.sum(), value)
```

Here `T` is 0.9849 and its derivative 0.0470 per unit permittivity, against 0.9837 and 0.0435
from the closed form (current module, 2026-10-15; the earlier adapter gave 0.9866 and 0.0405 on
its own grid). On JAX, prepare once with concrete values and differentiate
`prepared.run(problem)` with the traced value placed in the problem; the two backends agree to
1e-15 here. A material that reaches an `Open()` face is the exception: its PML is built once
from concrete values, so a half-space is not differentiable in the current module (see
[half-space material derivatives](#half-space-material-derivatives)).
{ref}`Differentiating a run <fdtd-model-differentiating>` lists what is live.

The rest of this section describes the earlier adapter, `wavebit.compat.legacy.fdtd(...)` with
an explicit `backend`, which most examples below still use.

That adapter runs on NumPy, JAX and Torch. JAX results are
registered pytrees, so a function returning the complete `Result` can be jitted.
Incidence `theta`/`phi`, raster epsilon, raster mu, electric Drude/Lorentz parameters and Jones amplitudes
retain their backend graphs. Raster cell boundaries are static; harmonic averaging
is applied normal to each field component and arithmetic averaging tangentially.
Within the transverse plane, normal harmonic averaging is performed before
tangential arithmetic averaging, for both epsilon and mu. This ordering makes
exchanging x and y rotate the same discretized material. It remains a diagonal,
separable approximation at raster corners, not full tensor homogenization.
Pole masks use arithmetic fill fractions, including the normal component; this
is an approximation to a dispersive interface, not a general tensor homogenizer.

Supported scene inputs include rectangular one-dimensional and crossed rasters,
rectangle/square/circle/ellipse recipes using the shared RCWA conventions (rotation and curved shapes through
opt-in area integration),
positive real magnetic half spaces, normal/oblique/conical incidence, Jones
polarization, electric pole labels, and scalar or vector wavelengths. Output
contains `incident`, `s_inc`, `p_inc`, diffraction order metadata, complex r/t and
power R/T. Both declared TE/TM and coupled Jones output use each outgoing
order's own s/p frame, with cell-centred amplitudes. Reflection is referenced to
the illuminated interface and transmission to the opposite interface.
`r_xyz`/`t_xyz` and RCWA field reconstruction are unavailable on this route.

Two electric-current basis sheets and two uniform incident-medium reference runs
calibrate incident polarization and subtract the incident field at the reflection
plane. All four runs share the same update core. This is reference normalization,
not a TFSF source. Extraction corrects Yee plane interpolation and uses discrete
homogeneous propagation to move the phase reference to the interfaces. At oblique
incidence the transverse wavevector stays fixed across the wavelength band;
`return_diagnostics=True` includes the corresponding angles.

The mesh, periods and wavelength are static. Rectangle centres and sizes can
vary on that mesh; see [rectangle derivatives](#transverse-rectangle-shape-derivatives). Internal layer
boundaries can vary at fixed total depth with `mesh_thickness`; opt-in
`variable_depth=True` also varies their total. See [layer derivatives](#internal-layer-thickness-derivatives)
and [total-depth derivatives](#changing-the-total-stack-depth).
Half-space epsilon/mu can retain gradients
with explicit `mesh_indices` and `truncation_order`, as described below.
An explicit `resolution` is required so material optimization cannot resize the
grid. Defaults here are `courant=.8`, `n_periods=40`, `n_pml=12`; choose a smaller
Courant fraction when the minimum material values or pole frequencies demand it.
The time-step stability bound is checked by the native core. Concrete invalid
inputs raise; invalid traced JAX runs return NaN amplitudes and efficiencies, with
`stable=False` in optional diagnostics. Refine resolution,
PML width, padding and duration for each new structure, especially resonances.
Exact grazing orders and orders beyond the grid's sampling limit are refused.
Far-plane extrapolation of strongly evanescent amplitudes is ill-conditioned.

The explicit-backend adapter currently does not accept skewed lattices or
arbitrary curved-shape derivatives; unsupported
explicit-backend keywords raise an error. There is no implicit backend fallback.
Those historical scene measurements used CPU. The current periodic API has
[two CUDA slab checks](../../validation/fdtd/native-cuda.md);
they do not qualify the wider scene family described here. The current general
API has a [separate bounded CUDA route](../../validation/fdtd/general-cuda.md).

Tests compare complex slab amplitudes with Airy under refinement, magnetic
conical Jones amplitudes and asymmetric diffraction with RCWA, a wavelength-band
transmission with stored Meep public-API data, and JAX/Torch material derivatives
with finite differences. Crossed dispersive scenes additionally check Torch/
NumPy agreement and a pole-strength derivative. These scene checks supplement
the native-core Meep/FDTDX field and derivative references.


### Transverse rectangle shape derivatives

The current module differentiates the geometry of a smoothed body given a reserved
`Numerics(geometry_ranges=...)` box, but not a `Box` angle, not a smoothed body standing flush
on a half-space `SnappedSlab` or near a `Periodic()` face, and not two smoothed surfaces in one cell;
it has no ellipse. Each example in this and the next four sections meets one of those limits,
so they keep the earlier explicit-backend route.

Explicit backends accept the same layer dictionaries and `Material` objects
as RCWA for rectangles and squares. The default `shape_integration='separable'`
accepts axis-aligned shapes; opt-in area integration supports rotation. A centre is measured from
the centre of the cell; sizes are absolute lengths in the units of `period`.
Later shapes paint over earlier shapes, including periodic images. Coordinates,
sizes and scalar positive real epsilon/mu can be NumPy, JAX or Torch values.

```python
# Earlier interface: a differentiable rectangle on a half-space substrate is not in the current API yet.
import jax
import jax.numpy as jnp
from wavebit.compat import legacy as wavebit

jax.config.update('jax_enable_x64', True)

def reflection(width):
    scene = wavebit.Stack.from_structure([
        dict(background=1., shapes=[
            dict(type='rectangle', lx=width, ly=.05, cx=.083, material=2.25)])
    ], [.2], top=1., bot=1.21, period=[1.3, .05])
    result = wavebit.fdtd(scene, wavebit.Incidence(wavelength=1.), backend='jax',
                          resolution=20, n_pml=20, n_periods=65, pulse_width=3.,
                          truncation_order=(1, 0), shape_smoothing=[.05, 0.],
                          align_layers=True, diffraction_method='directional')
    return result.s_inc.R.sum()

value, slope = jax.jit(jax.value_and_grad(reflection))(jnp.array(.43))
```

The recipe requires two finite positive rectangular periods and
`subpixel=True`. For separable rectangles, each size must be positive and no
larger than its period.
The separable `type` values are `rectangle` and `square`; `l` and omitted
centres retain their shared RCWA meaning. The default separable method requires
a static `angle=0`. Circles and [ellipses](#elliptical-shape-derivatives) require the area method. Use each shape's `Material(epsilon,
mu=...)` for magnetic media. Separate raster `mu`, dispersion labels and poles
are not accepted on this recipe path. Raster inputs retain their existing
dispersion support. Combining live layer thicknesses also requires the usual
`mesh_thickness` and the selected fixed-total or variable-depth constraints.

Shape count, type, paint order, periods and the Yee mesh remain static.
A separable recipe whose shapes all have static `ly=period[1]` uses the one-row grating
shortcut. A live `ly`, including a tensor currently equal to the full period,
retains full y sampling. The existing array-phi rules also keep full y sampling.
Unsupported inputs raise; invalid traced geometry produces NaN scattering and
`stable=False` in diagnostics. No unsupported shape or angle is silently dropped.

By default, `shape_smoothing=0` integrates sharp periodic material regions over
each Yee cell. Derivatives are piecewise smooth and can jump as a face crosses
a cell face. An optional static nonnegative scalar or `(x,y)` pair averages
common translations of the entire painted layer before cell integration.
Widths use physical length units, must not exceed their respective periods,
and can be zero independently. Positive widths give C1 cell fractions at cell
crossings within a fixed paint topology. They also change the sampled material
profile: reduce the widths with grid refinement and independently check the
intended sharp geometry.

A contact at which one shape starts covering another can have a physical
one-sided derivative, even with positive averaging. This is different from
unrelated projected edges sharing a coordinate; those retain correct derivatives.
Derivatives at a size bound, topology changes, and general second derivatives
are not promised. Normal harmonic and tangential arithmetic averaging retain
the existing separable corner approximation, rather than a general effective
tensor for a corner.

Diagnostics report `geometry_kind`, `shape_integration`, `shape_smoothing`, and live `shape_edges`
in cell-centred x/y coordinates for the separable method. These edges describe the material partition;
the returned Yee `shape`, `spacing` and sampling coordinates remain fixed.
See the [measured refinement and independent references](../../validation/fdtd/fdtd-rectangle-gradients.md).

### Rotated rectangle area integration

Set `shape_integration='area'` on an explicit backend to differentiate a
rectangle's `angle` in radians along with its centre, size and scalar material.
The same shared RCWA recipe is used:

```python
# Earlier interface: a rotation-angle derivative is not in the current API yet.
import jax
import jax.numpy as jnp
from wavebit.compat import legacy as wavebit

jax.config.update('jax_enable_x64', True)

def reflection(angle):
    stack = wavebit.Stack.from_structure([
        dict(background=1., shapes=[
            dict(lx=.3, ly=.18, angle=angle, material=2.25)])],
        [.2], top=1., bot=1.21, period=[.6, .6])
    result = wavebit.fdtd(stack, wavebit.Incidence(wavelength=1.),
                          backend='jax', resolution=16, n_pml=16, n_periods=60,
                          pulse_width=2.5, truncation_order=0, align_layers=True,
                          shape_integration='area', diffraction_method='directional')
    return result.s_inc.R.sum()

value, slope = jax.jit(jax.value_and_grad(reflection))(jnp.asarray(.37))
```

The area route accepts multiple rectangles/squares with later-shape painting,
or one circle/disk/ellipse per layer. Empty shape lists describe uniform layers. Periods must be static and positive,
with two entries. Sizes must be positive. For rectangles, the projected extents
`abs(cos(angle))*lx + abs(sin(angle))*ly` and
`abs(sin(angle))*lx + abs(cos(angle))*ly` must not exceed their respective
periods. Centres wrap periodically. `subpixel=True` is required. Use the shape's `Material` for epsilon/mu; recipe dispersion
labels and separate raster mu remain unsupported. Rotation and geometry may
be live on JAX/Torch, while period, mesh and shape count stay static.

The full y-period mesh is always retained, including when a live angle is
zero or a rectangle currently fills the period. Geometry integrates exactly
against each Yee cell, and transverse material coefficients use arithmetic
area averages. This differs from the separable method even at zero angle.
It is not a full rotated effective tensor and does not promise second-order
interface accuracy. Layer sampling retains its existing z averaging rules;
live layer thicknesses require `mesh_thickness` and the selected depth contract.

With the default `shape_smoothing=0`, coincident parallel cell/shape edges
can have different one-sided derivatives. To average transverse translations,
add `shape_smoothing=.05` or a static `(x, y)` pair to the solve above. Each
width must be finite, nonnegative and no larger than its period; a zero
component keeps that axis sharp. Positive widths use exact bilinear polygon
moments, without a sampling grid or quadrature-order setting. This smooths
cell-crossing derivatives on the averaged axes; general second derivatives
are not supplied. The widths stay static during differentiation.

Averaging changes the material distribution: it represents the rectangle
convolved with a uniform translation box. Hold the physical widths fixed
when refining this graded target, or explicitly study the additional bias
when shrinking them toward a sharp target. It preserves period-integrated
material area. It does not introduce a rotated effective tensor. See the
[averaging measurements](../../validation/fdtd/fdtd-rotation-smoothing.md) for independent
geometry and matching graded-profile references.

Check grid refinement and an independent physical reference when using the
result as a design gradient. The [rotation measurement](../../validation/fdtd/fdtd-rotation-gradients.md)
records the independent area, complete graph and optical checks.



### Multiple rotated rectangles

Use the same `shapes` list with `shape_integration='area'` to put multiple
rectangles or squares in one layer. Each rectangle may have its own live
position, two sizes, angle and positive scalar `Material(epsilon, mu=...)`.
**Later shapes overwrite earlier shapes**, including overlaps through a
periodic cell edge. A completely hidden shape has no material or geometry
sensitivity until its boundary becomes exposed.

```python
# Earlier interface: overlapping painted rectangles are not in the current API yet.
import jax
import jax.numpy as jnp
from wavebit.compat import legacy as wavebit

jax.config.update('jax_enable_x64', True)

def reflection(angle):
    stack = wavebit.Stack.from_structure([
        dict(background=1., shapes=[
            dict(lx=.17, ly=.1, cx=-.035, cy=-.012, angle=angle, material=1.8),
            dict(lx=.15, ly=.12, cx=.025, cy=.014, angle=-.27, material=2.25)])
    ], [.2], top=1., bot=1.21, period=[.4, .3])
    result = wavebit.fdtd(stack, wavebit.Incidence(wavelength=1.), backend='jax',
                          resolution=15, n_pml=15, n_periods=60, pulse_width=2.5,
                          truncation_order=0, shape_integration='area',
                          shape_smoothing=.035, diffraction_method='directional')
    return jnp.real(result.s_inc.r_s[0, 0])

value, slope = jax.jit(jax.value_and_grad(reflection))(jnp.asarray(.31))
```

The shape count, shape types and paint order remain static. Positions and
angles may move across overlap configurations inside one JAX trace; a host
clipping plan is not required. This retains **piecewise first derivatives**.
Actual shape contacts and coincident boundaries can have distinct one-sided
shape derivatives even with positive `shape_smoothing`. Smoothing translates
the already-painted material pattern as a whole; it does not soften the
relative contact between two shapes.

Each rectangle's projected extents must fit the periods. Every layer that
contains a circle/disk/ellipse must still contain only that shape; mixing curved
shapes with other area shapes is refused. Layer-thickness, incidence and scalar
material constraints are unchanged.

Boundary intersections select the material on each side. Exact straight-line
cell integrals recover the painted material without an auxiliary raster or
pairwise area subtraction. Translation averaging leaves cubic integrands on
split segments, evaluated exactly in arithmetic by two Gaussian nodes. Roundoff
still limits floating-point accuracy. Work and storage grow with the number of
boundary events; the absence of a hard shape-count limit is not a performance
promise for large layouts.

[Painted-shape measurements](../../validation/fdtd/fdtd-painted-shapes.md) separate geometric and
complete-graph checks from graded optical refinement. Transverse material
coefficients remain arithmetic averages; this extension does not introduce a
full interface-effective tensor.

### Circular shape derivatives

Use `type='circle'` (alias `'disk'`) with `shape_integration='area'` and an
explicit backend. The shared RCWA convention uses `lx` or `l` as the **diameter**,
not the radius; `ly` is refused. The default centre is `(cx, cy)=(0, 0)` relative
to the period centre. Centres wrap periodically. A finite `angle` has no effect
on a circle, and its derivative is zero.

```python
# Earlier interface: a differentiable disk on a half-space substrate is not in the current API yet.
import jax
from wavebit.compat import legacy as wavebit

jax.config.update('jax_enable_x64', True)

def transmission(diameter):
    stack = wavebit.Stack.from_structure([
        dict(background=1., shapes=[dict(type='circle', lx=diameter,
                                         cx=.071, cy=-.047, material=2.25)])
    ], [.2], top=1., bot=1.21, period=[.6, .6])
    result = wavebit.fdtd(stack, wavebit.Incidence(wavelength=1.), backend='jax',
                          resolution=20, n_pml=20, n_periods=60, pulse_width=2.5,
                          truncation_order=0, shape_integration='area',
                          shape_smoothing=.05, diffraction_method='directional')
    return result.s_inc.t_s[0, 0]

t = jax.jit(transmission)(.3)
dt_ddiameter = jax.jit(jax.jacfwd(transmission))(.3)
```

The diameter must be positive and no larger than **either** rectangular period.
The one-shape-per-layer restriction includes circles, so a circle and rectangle
can occupy different layers but cannot be painted together in one area layer.
The two periods, shape types/count, smoothing widths and mesh stay static.
`subpixel=True` is required. Centres, diameter and positive scalar shape
`Material` epsilon/mu retain JAX/Torch gradients, including bottom incidence
and live layer thicknesses under their existing mesh contract.

At `shape_smoothing=0`, disk/cell areas use analytic circular-segment integrals.
The boundary is not approximated by a polygon or an auxiliary raster. Tangent
contacts have finite first derivatives; second derivatives can be singular.
Positive smoothing has the same physical box-translation meaning as for
rectangles. Disk integration uses exact flat strips and a fixed 16-point
Gaussian rule on angular arc strips, whose integrands are low-degree
trigonometric polynomials. There is no user quadrature-order setting. This
avoids cancellation between global moments when a kernel piece is very narrow.
Widths of zero, a full period and a cell width are supported.

Transverse material coefficients still use arithmetic area averages. These
are not full effective tensors, so exact cell geometry does not establish
sharp-interface optical gradient accuracy. Check resolution and a matching
physical reference. [Disk measurements](../../validation/fdtd/fdtd-circle-gradients.md) separate
geometry, complete-graph derivatives, profile bias and independent optics.

### Elliptical shape derivatives

The explicit common route accepts `type='ellipse'` with
`shape_integration='area'`. `lx` and `ly` are the **full lengths of the two
principal axes**, before rotation, in the same units as the periods. `l` is the
shared fallback for both lengths. `angle` rotates the ellipse counterclockwise
in radians; it may be live even when its current value is zero. Centres keep
the common period-centred convention and wrap periodically.

```python
# Earlier interface: an elliptical body is not in the current API yet.
import jax
import jax.numpy as jnp
from wavebit.compat import legacy as wavebit

jax.config.update('jax_enable_x64', True)

def transmission(parameters):
    angle, width, height = parameters
    stack = wavebit.Stack.from_structure([
        dict(background=1., shapes=[dict(type='ellipse', angle=angle,
             lx=width, ly=height, cx=.071, cy=-.047, material=2.25)])
    ], [.2], top=1., bot=1.21, period=[.6, .6])
    result = wavebit.fdtd(stack, wavebit.Incidence(wavelength=1.), backend='jax',
                          resolution=20, n_pml=20, n_periods=60, pulse_width=2.5,
                          truncation_order=0, shape_integration='area',
                          shape_smoothing=.05, diffraction_method='directional')
    return result.s_inc.t_s[0, 0]

parameters = jnp.array([.37, .34, .2])
t = jax.jit(transmission)(parameters)
dt_dshape = jax.jit(jax.jacfwd(transmission))(parameters)
```

Each axis length must be positive. The projected full widths
`sqrt((lx*cos(angle))**2 + (ly*sin(angle))**2)` and
`sqrt((lx*sin(angle))**2 + (ly*cos(angle))**2)` must fit in the x and y periods.
A principal axis may therefore exceed its unrotated period when the rotated
ellipse fits. Shape count/type, the two rectangular periods, the mesh and the
smoothing widths remain static. Ellipse layers require one shape and `subpixel=True`. Centres, both lengths, angle and positive scalar shape
`Material` epsilon/mu retain JAX/Torch gradients. Layer-thickness and bottom
incidence contracts also apply.

Both sharp and averaged ellipse cells use angular strip integration with a
fixed 16-point Gaussian rule. Cartesian cell faces intersect the analytic
ellipse; its boundary is not polygonized or sampled on an auxiliary raster.
Projected tangencies within a few floating-point rounding units use their
limiting endpoint to avoid an ill-conditioned inverse-sine derivative. General
second derivatives are not provided.

`shape_smoothing` averages translations along the **laboratory x/y axes**.
Rotating the ellipse does not rotate this averaging box. Zero, narrow, full-cell
and full-period widths keep the same meaning as for circles. The transverse
material coefficients are arithmetic averages, not full curved-interface
effective tensors. Check an independent reference and resolution convergence
for the material profile actually optimized. [Ellipse measurements](../../validation/fdtd/fdtd-ellipse-gradients.md)
separate integration accuracy, complete graphs and graded optical refinement.

### Incidence-angle derivatives

In the current module a `PlaneWave`'s `theta` and `phi` are static: they fix the grid and the
time loop, and a traced value is refused. These examples keep the earlier route.

With an explicit `backend`, `theta` and `phi` may be real scalar NumPy, JAX or
Torch arrays. Supply an explicit `truncation_order` whenever either is an array,
including a constant zero-dimensional array. This keeps the output shape fixed
while an angle changes. Python or NumPy scalar numbers continue to support
automatic order selection.

```python
# Earlier interface: an incidence-angle derivative is not in the current API yet.
import torch
from wavebit.compat import legacy as wavebit

theta = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
result = wavebit.fdtd(
    structure=[[[2.25]]], thickness=[.2], top=1., bot=1., period=.2,
    wavelength=1., theta=theta, tetm='tm', backend='torch',
    resolution=20, n_periods=40, truncation_order=0)
dT_dtheta, = torch.autograd.grad(result.incident.T.sum(), theta)
```

The derivative is per radian, at fixed geometry, cell sizes, time step, number
of updates, source location, PML and diffraction window. `Incidence(theta=theta, phi=phi)`
preserves the same graph. The path includes source-sheet phase, Bloch boundaries,
spatial Fourier extraction, interpolation, propagation to the interfaces,
outgoing s/p frames, power factors and both incident reference runs. It also
includes normalized frequency monitors, cropped time records and prefix
convergence diagnostics. Material and electric-pole parameters can be
differentiated together with both angles; independent transverse resolution,
static z grading, layer alignment, source notches and bottom incidence remain
available.

For azimuth derivatives, use `polarization=(a_s, a_p)` or `psi`/`chi`. Array
`phi` is incompatible with `tetm`, including an array whose value is zero:
the scalar TE/TM convention fixes the x-z plane and cannot follow a changing
azimuth. Pure s and p are `polarization=(1., 0.)` and `(0., 1.)` respectively.

```python
# Earlier interface: angle derivatives and Jones polarization are not in the current API yet.
import torch
from wavebit.compat import legacy as wavebit

angles = torch.tensor([.3, .4], dtype=torch.float64, requires_grad=True)
result = wavebit.fdtd(
    structure=[[[1.3, 2.2], [1.8, 1.1]]], thickness=[.2],
    top=1., bot=1., period=[.3, .4], wavelength=1.,
    theta=angles[0], phi=angles[1], polarization=(1., .6j),
    backend='torch', resolution=20, n_periods=40, truncation_order=0)
dT_dangles, = torch.autograd.grad(result.incident.T.sum(), angles)
```

An array azimuth always uses the full represented y period, including at zero:
for declared `Ly`, `Ny = max(2, ceil(Ly Ry))`. A one-row raster with a plain
scalar `phi=0` keeps the existing two-y-cell shortcut. Consequently, switching
that scalar to an array can change the grid and time step once; subsequent
changes to the array value keep them fixed. Use arrays consistently for
finite-difference comparisons. NumPy vector indexing produces scalar numbers;
wrap such a phi in `np.asarray(phi)` when comparing with an array-angle solve.
A single x period still has its auxiliary
two-cell y period `2/Ry`. The existing y-invariant monitor/record slice and
power-per-unit-y convention are unchanged.

For multiple wavelengths there is one transverse Bloch wavevector, set using
the midpoint `lambda_c` of the requested wavelength range. In the static
incident half space,

`theta(lambda) = asin((lambda/lambda_c) * sin(theta))`.

Thus an angle derivative changes the whole broadband experiment along this
curve. It is not a separate fixed-angle derivative at each wavelength.
Diagnostics return backend arrays in `angles` for the wavelength-dependent
polar angles and scalar `azimuth` for phi. Changing phi rotates the shared
transverse vector; its magnitude and the polar angles stay fixed. The
fixed-wavevector convention is also described in the
[Meep public FAQ](https://meep.readthedocs.io/en/latest/FAQ/).

Theta must satisfy finite real `0 <= theta < pi/2`; phi accepts any finite
real value in radians, including negative values. The incident wave must
propagate throughout the band. Exact physical grazing, interpolation Nyquist
zeros and the directional method's discrete singularities remain unsupported.
Concrete invalid angles raise before stepping; abstract invalid angles under
JAX JIT produce `stable=False` and NaN scattering, normalized monitors and time
records. Boolean propagation masks are classifications, not differentiable
observables. At normal incidence the theta derivative follows a fixed phi ray,
and the phi derivative rotates the specified s/p polarization frame. The
transverse frame has no unique arbitrary direction at its origin. At
thresholds or long-lived resonances, establish spatial and
temporal convergence before interpreting a sensitivity. See the
[theta](../../validation/fdtd/fdtd-scene-angles.md) and [azimuth](../../validation/fdtd/fdtd-scene-azimuth.md) validation
notes for the checked cases and limits.


### Independent transverse resolution

The explicit-backend adapter accepts `transverse_resolution=R` or
`transverse_resolution=(Rx, Ry)`, in cells per unit length. These positive finite
static values set x/y sampling while `resolution` continues to set the z mesh,
layer alignment and PML cell thickness. `None` uses `resolution` for both axes.
The pair is in **x, y order**; native `spacing` and grid `shape` remain in z, y, x
order. Material and pole gradients retain their usual meaning; mesh sizes and
these resolution controls are not differentiable.

```python
# Earlier interface: per-axis cell sizes are not in the current API yet (one cubic spacing).
from wavebit.compat import legacy as wavebit

result, diagnostic = wavebit.fdtd(
    structure=[[[2.25, 1.]]], thickness=[.2], top=1., bot=1., period=.8,
    wavelength=1., tetm='te', backend='numpy', resolution=40,
    transverse_resolution=(80, 40), truncation_order=1,
    return_diagnostics=True)
print(diagnostic['transverse_resolution'])  # (80.0, 40.0), requested x/y targets
print(diagnostic['spacing'])                # actual dz, dy, dx
```

Each represented transverse period is divided into at least two cells, rounding
the target count upward: `Nx = max(2, ceil(Lx Rx))`, and similarly for a fully
represented y axis. A one-row raster at plain scalar `phi=0` retains two y cells. If a y
period is supplied, their width is `Ly/2`; changing `Ry` does not subdivide that
collapsed axis. Array phi uses full-period sampling even at zero. With a single
x period, the auxiliary y period is `2/Ry`, so
its two cells have width `1/Ry`, including conical illumination of a y-invariant
structure. Diagnostics expose the actual spacing rather than assuming targets
were attained identically on every axis.

Changing transverse sampling preserves z nodes, layer faces, the selected
source sheet, PML faces and diffraction planes. It recomputes the CFL time step
from **all three actual spacings**, and may therefore increase the update count
even when the number of z cells is unchanged. Source zeros specified as numerical
frequencies are not automatically retuned to a different discrete cutoff.
Nyquist and grazing checks use the new mesh and still reject unresolved retained
orders. All four current-basis/reference runs use the same mesh.

Independent refinement is useful when one direction limits accuracy, but it
does not guarantee less work or monotonically smaller error. The
[transverse refinement study](../../validation/fdtd/fdtd-transverse-resolution.md) separates
harmonic lattice accuracy from finite-time and finite-boundary differences.


### Source position independent of padding

In the current module the incident sheet sits where the problem puts it.
`PlaneWave(plane=fdtd.Plane(z=...))` places it on the node plane nearest that coordinate, each
`DiffractionMonitor(plane=...)` places a measurement plane, and the `Domain` sets the padding, so none
of the three moves another. A film 0.213 µm thick, with the sheet 1 µm above its top face:

```python
from wavebit import Material, Units, fdtd

thickness, distance = .213, 1.                     # um; the sheet 1 um above the top face
problem = fdtd.Problem(
    units=Units(length='um'),
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 2.), center=(0., 0., .5)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        # A smoothed Box keeps both faces where they are declared; a SnappedSlab would snap them.
        structures=(fdtd.Structure('film', shape=fdtd.Box(size=(.2, .2, thickness),
                                                          center=(0., 0., thickness / 2)),
                                   material=Material(epsilon=2.25)),)),
    sources=(fdtd.PlaneWave('light', plane=fdtd.Plane(z=thickness + distance),
                            pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=2.25e13)),),
    outputs=fdtd.Outputs(monitors=(
        fdtd.DiffractionMonitor('r', plane=fdtd.Plane(z=thickness + .2), wavelengths=(1.,),
                                source='light', wave='reflected',
                                reference_plane=thickness, incident_plane=thickness),
        fdtd.DiffractionMonitor('t', plane=fdtd.Plane(z=-.2), wavelengths=(1.,),
                                source='light', wave='transmitted',
                                reference_plane=0., incident_plane=thickness))))
result = fdtd.run(problem, numerics=fdtd.Numerics(
    grid=fdtd.UniformGrid(spacing=.025), duration=fdtd.Duration(periods=50, source='light')))
print(f"{problem.sources[0].plane.position:.4f}")     # requested sheet position
print(f"{result.source_references['light']['plane_E']:.4f}")  # the node plane E sits on
```

```
1.2130
1.2250
```

The values come from the current module (2026-10-15). `frequency_sigma_hz=2.25e13` is the
earlier `pulse_width=3.` (see [Gaussian source width](#gaussian-source-width)), and the film's
`T` is 0.8759 against 0.8771 on the earlier route and 0.8752 from the closed form. The Box
keeps the 0.213 µm faces that the earlier route's `align_layers=True` put on nodes; the Box
draws the `SubpixelAccuracyWarning` every smoothed body draws.

The earlier explicit-backend route spells the same placement as distances from the stack's
faces. `source_distance` optionally places the incident electric-current sheet a
positive static distance from the incident interface, in length units. `None`
keeps the original source position. The nearest z node is selected without
changing the mesh, PML, layer boundaries or extraction planes. With top
incidence its target is `z=total_thickness+source_distance`; with bottom incidence
the target is `z=-source_distance`. With `variable_depth=True`, use the nominal
mesh depth for this placement; the source stays fixed as the physical face moves.
`source_distance=1., diffraction_distance=.2` with `align_layers=True` placed the earlier
route's sheet on the node at 1.213 for the film above.

The selected node must lie in the incident half space, at least two cells past
the PML's inner face, and at least three cell intervals before the reflection
plane. Invalid, nonfinite, nonpositive and unplaceable distances raise before
time stepping. A user frequency monitor still cannot interpolate across the
current sheet. On a graded mesh, distance is physical rather than a cell count;
the reported node may differ from the request by the local snapping distance.

This makes two separate experiments possible: vary `pad_scale` while keeping
both `source_distance` and `diffraction_distance` fixed, or move the source at
fixed padding. The former changes boundary clearance; the latter changes the
illumination path. Inspect realized coordinates because changing a graded mesh
can change the nearest nodes. Each wavelength still uses its own uniform-medium
reference calibration. Source placement changes finite-time transients and
boundary returns, so equal calibrated amplitudes require convergence rather
than an assumption of exact invariance on every grid.

The option supports the shared NumPy/JAX/Torch scene engine, bottom incidence,
grading, source notches, recordings and material gradients. The distance itself
is static and has no shape derivative; the legacy scene path has no corresponding
keyword. See the [source-clearance measurements](../../validation/fdtd/fdtd-source-clearance.md)
for a near-cutoff control where moving the source and moving the boundary need
to be distinguished.


### Diffraction planes and cutoff diagnostics

In the current module each measurement plane is the `DiffractionMonitor(plane=...)` the problem
declares, and `reference_plane` and `incident_plane` move its phase references (as in the
example above); each result carries the Yee `k_normal` (the wavevector component along the port's normal
axis) and `k_parallel` of every order. It has no
cutoff classification yet, so the example below keeps the earlier route.

`diffraction_distance` controls the extraction-plane distances from the reflection
and transmission reference interfaces. It accepts a positive static scalar for
both legs, or a `(reflection, transmission)` pair in length units. The default
`None` preserves the original placement. Each target is snapped to the nearest
electric z node; the setting leaves the mesh, source and PML placement unchanged.
The reflection plane must be between the current sheet and stack; the transmission
plane must precede the PML. Neither plane may touch a mixed interface cell.
An unplaceable request raises instead of silently expanding the simulation.

```python
# Earlier interface: the cutoff classification diagnostics are not in the current API yet.
import numpy as np
from wavebit.compat import legacy as wavebit

result, diag = wavebit.fdtd(
    structure=np.full((1, 1, 1), 2.25), thickness=[.213],
    top=1., bot=1., period=.1, wavelength=1., tetm='tm', theta=.25,
    backend='numpy', resolution=80, n_pml=24, align_layers=True,
    diffraction_distance=(.2, .3), return_diagnostics=True,
)
print(diag['diffraction']['reflection_monitor_z'])
print(diag['diffraction']['cutoff_resolved'])
```

`diag['diffraction']` reports the realized `reflection_monitor_z` and
`transmission_monitor_z` in original scene coordinates, including for bottom
incidence. The pair is always incidence-relative. Complex r/t remain referenced
to the interfaces, not to these measurement planes. Moving a plane can change
finite-grid reference calibration and hence calibrated fields as well as r/t;
compare positions under spatial, PML and runtime refinement.

The diagnostic also contains `(order_y, order_x)` arrays `order_x`, `order_y`,
`kx`, `ky`; the latter two are fixed transverse wavevectors in radians per length
throughout the band. Its `reflection` and `transmission` mappings describe each
lossless half space with the following arrays. Their shapes are fixed; numeric
values follow live angles/materials, while the Boolean masks classify the
current parameters:

| Key | Shape | Meaning |
| --- | --- | --- |
| `q` | `(wavelength, order_y, order_x)` | Outgoing normalized longitudinal wavevector: positive real for propagation, positive imaginary for decay. It is not signed Cartesian kz for bottom incidence. |
| `propagating` | same | Continuum open-channel mask; closed channels have exactly zero outgoing R/T power. |
| `yee_cutoff_parameter` | same | Discrete lower-cutoff parameter defined below. |
| `cutoff_resolved` | same | Whether the continuum and discrete lower-cutoff signs agree. |
| `p_projection_gain` | same | `n/abs(q)`, the tangential-E to p-amplitude projection gain; large values amplify extraction error. |
| `cutoff_wavelength` | `(order_y, order_x)` | `2*pi*n/hypot(kx,ky)` for fixed transverse k, or infinity at zero transverse k. |

Here `n=sqrt(epsilon*mu)`, `k0=2*pi/wavelength`, and the Yee cutoff parameter is

```text
(2/k0)^2 * ((n*sin(k0*dt/2)/dt)^2
            - (sin(kx*dx/2)/dx)^2 - (sin(ky*dy/2)/dy)^2).
```

Its positive sign is only the lower edge of the longitudinal passband. It does
not rule out longitudinal Nyquist limits, mesh reflections or PML errors.
The outer scalar `diag['diffraction']['cutoff_resolved']` is true only when all
retained orders at all requested wavelengths agree in both half spaces. It is
independent of temporal convergence flags and is not an accuracy certificate.
Exact grazing, including roundoff-sized distance from it, is refused before time
stepping. Orders are not silently discarded. See the
[near-Rayleigh measurements](../../validation/fdtd/fdtd-rayleigh.md) for a false-open coarse-grid
channel and an unresolved TM response despite correct cutoff classification.


### Directional diffraction extraction

The default `diffraction_method='electric'` preserves the existing extraction.
With an explicit backend, `diffraction_method='directional'` uses both E and H
to separate propagation toward and away from the stack at each measurement plane:

```python
# Earlier interface: directional E/H extraction is not in the current API yet.
import numpy as np
from wavebit.compat import legacy as wavebit

result, diag = wavebit.fdtd(
    structure=np.full((1, 1, 1), 2.25), thickness=[.2],
    top=1., bot=1., period=.1, wavelength=1., tetm='tm',
    backend='numpy', resolution=40, n_pml=12,
    diffraction_method='directional', diffraction_distance=.2,
    return_diagnostics=True, convergence_window=100,
)
print(diag['diffraction']['incoming']['p_inc']['t_p'])
```

This option uses the **discrete Yee admittance**, including electric/magnetic
spatial interpolation and their separate DFT timestamps. It supports conical
incidence, magnetic half spaces, coupled order windows and evanescent branches.
Each plane must have equal z-cell widths immediately above and below it; a
plane inside a grading transition is refused. Grading elsewhere and aligned
layer interiors remain available, but their propagation errors still need
refinement. Discrete cutoff and longitudinal/temporal Nyquist singularities
are refused before time stepping. The physical grazing refusal also applies.

Reflection uses the backward actual field minus the backward uniform-medium
reference. Transmission uses the forward actual field. Incident normalization
uses the **forward** reference field. The four native solves, material gradients
and interface reference planes remain the same. Time records use the forward
reference calibration at the carrier. Calibrated convergence diagnostics use
the same directional extraction on the full and prefix DFTs, including when
`dft_subsample` exceeds one.

`diag['diffraction']['method']` identifies the method. For directional extraction,
`diag['diffraction']['incoming']` contains `incident`, `s_inc` and `p_inc` mappings,
each with complex `r_s`, `r_p`, `t_s`, `t_p` arrays of shape
`(wavelength, order_y, order_x)`:

| Keys | Meaning |
| --- | --- |
| `r_s`, `r_p` | Forward actual field minus forward reference at the reflection plane, referred to the incident interface. This excludes the prescribed reference illumination. |
| `t_s`, `t_p` | Backward actual field at the transmission plane, referred to the outgoing interface. |

The prefixes name the measurement **leg**, while these components travel in the
opposite direction from the corresponding outgoing r/t. The p sign follows each
wave's own propagation direction. Legs remain incidence-relative for bottom
incidence. Values are unit-incident-E amplitudes, not reflected-power fractions;
evanescent entries should not be interpreted as powers.

Direction separation prevents an incoming wave from being added directly to an
outgoing coefficient. It cannot undo how returning waves alter the illumination
and structure response. A finite DFT also has endpoint terms while fields are
still oscillating; its apparent incoming components may include that error.
Compare amplitude tails, runtime, grid, padding and PML independently. See the
[directional measurements](../../validation/fdtd/fdtd-directional.md) for exact-mode verification
and a near-Rayleigh example that still fails temporal convergence.


### Gaussian source width

In the current module every source carries a declared `GaussianPulse(wavelength,
frequency_sigma_hz)`, the standard deviation of its amplitude spectrum in Hz. The earlier
`pulse_width=w` below corresponds to `frequency_sigma_hz = c / (sqrt(2) * pi * w)` with `w` a
length: 2.25e13 Hz for `w = 3` µm. The usable band ends where that spectrum falls to the
pulse's `cutoff` (1e-3 by default), and a `DiffractionMonitor` wavelength outside it comes back NaN.
The rest of this section describes the earlier route's `pulse_width`.

The explicit-backend scene adapter accepts `pulse_width`, a positive static scalar
in c=1 time units (the same numerical units as length). It is the Gaussian
envelope's 1/e **amplitude half-width**, not a frequency bandwidth or a FWHM.
Omitting it, or passing `None`, uses the central vacuum wavelength
`centre = (min(wavelengths)+max(wavelengths))/2`, preserving the original pulse.

At sample time `t=n*dt`, both actual and reference runs use

```text
pulse(t) = exp(-((t-5*pulse_width)/pulse_width)^2) * exp(-2*pi*i*t/centre).
```

A larger width narrows the frequency band. This can reduce slowly decaying
glancing waves at oblique incidence, but also weakens the source at wavelengths
away from its center. Use a longer total run when increasing the width:

```python
# Earlier interface: the pulse and convergence-window diagnostics are not in the current API yet.
import numpy as np
from wavebit.compat import legacy as wavebit

result, diag = wavebit.fdtd(
    structure=np.full((1, 1, 1), 2.25), thickness=[.2],
    top=1., bot=1., period=.1, wavelength=[.9, 1., 1.1],
    theta=.45, tetm='te', backend='numpy', resolution=40,
    pulse_width=3., n_periods=80, truncation_order=0,
    return_diagnostics=True, convergence_window=200,
)
print(diag['pulse']['relative_spectrum'])
print(diag['convergence']['state_decayed'])
```

`diag['pulse']` provides `width`, `peak_time=5*width`, `frequency=1/centre`, and
`relative_spectrum` in requested-wavelength order. The latter is the ideal
untruncated Gaussian's amplitude spectrum relative to its carrier:
`exp(-(pi*width*(1/wavelength-1/centre))^2)`. It describes the source envelope,
not measured incident flux or a calibration error bound.

Widths shorter than one time step, and widths that leave a requested frequency
below 100 machine epsilons of the nominal peak, are refused. That numerical-tail
guard follows the backend source dtype; a passing value does not certify accuracy
at a weakly excited frequency. The guard also applies when the width is omitted.
Reduce `pulse_width` or narrow the requested
wavelength interval if it fails. The minimum runtime includes ten pulse widths
plus the scene round-trip allowance. `n_periods` still measures total time from
zero in central-wavelength optical cycles, not time after the source peak.

This setting works with bottom incidence, grading, recording, DFT subsampling
and material autodiff. The width itself is static. The legacy scene path has no
`pulse_width` keyword; the native grid API accepts an explicit pulse array.
See the [oblique-pulse measurement](../../validation/fdtd/fdtd-oblique-pulse.md) for the separate
effects of source tails, dispersive interfaces and patterned metal corners.


### Suppressing selected source frequencies

The explicit-backend scene adapter also accepts `pulse_notches`: a positive
static frequency or a vector of one or two frequencies in inverse c=1 time units,
**not wavelengths or angular frequencies**. `None` preserves the original
Gaussian. Each frequency must be below the time grid's Nyquist frequency.
For example, to weaken an unwanted oscillation at frequency 0.8 while measuring
wavelengths 0.9, 1.0 and 1.1:

```python
# Earlier interface: source spectral zeros are not in the current API yet.
import numpy as np
from wavebit.compat import legacy as wavebit

result, diag = wavebit.fdtd(
    structure=np.full((1, 1, 1), 2.25), thickness=[.2],
    top=1., bot=1., period=.1, wavelength=[.9, 1., 1.1],
    backend='numpy', resolution=40, truncation_order=0,
    pulse_width=2., pulse_notches=[.8], n_periods=45,
    return_diagnostics=True,
)
print(diag['pulse']['relative_spectrum'])
```

At each notch frequency `fn`, shaping applies the causal two-sample filter

```text
p_new[n] = p[n] - exp(-2*pi*i*fn*dt)*p[n-1],
```

with zero extension before the first sample. The complete finite convolution
has a spectral zero at `fn`, subject to floating-point roundoff. Each notch
reserves one final sample before filtering; the runtime allowance includes
those samples. The shaped source is normalized to unit sampled peak amplitude.
Two equal frequencies create a double zero. Higher multiplicities are refused:
repeated differences amplify the Gaussian's finite start/stop edges and
floating-point errors. Even two zeros can increase out-of-band excitation;
compare runtime and source width rather than assuming more zeros help.
All four actual/reference runs
use the same shaped pulse, and each requested wavelength retains its own
incident-field calibration. No field data are filtered after the solve.

For a shaped source, `diag['pulse']['spectrum_kind']` is `'sampled_l1'`:
`relative_spectrum` is now the **actual finite-sample** Fourier magnitude at
each requested frequency divided by `sum(abs(pulse))`. This bounds it between
zero and one; it is not a ratio to the carrier. `carrier_fraction` uses the
same normalization at the carrier, `notches` reports the applied frequencies,
and `sampled_peak_time` locates the actual shaped pulse maximum. The existing
`width`, `peak_time` and `frequency` still describe the underlying Gaussian.
Without notches, `spectrum_kind='ideal_gaussian'` and the original analytic
`relative_spectrum` above is preserved.

A notch at a requested frequency is refused. When recording time fields, the
carrier must also remain excited because it normalizes those records; without
time records an unrequested carrier may be suppressed. Required sampled
spectra at or below 100 epsilons of the source's L1 bound are refused, using
the backend's source precision. This is only a numerical floor. Weak remaining
excitation can still amplify finite-time, boundary and roundoff errors.

The control supports bottom incidence, grading, directional extraction,
recording and material/pole gradients. Notch frequencies and source width are
static, and the legacy scene path has no `pulse_notches` option. Use the native
grid's explicit `pulse` array for arbitrary waveforms or source gradients.
A spectral zero changes excitation, not the material or open-boundary model;
it does not guarantee shorter convergence. See the
[cutoff-source measurements](../../validation/fdtd/fdtd-pulse-notches.md) before using it near a
diffraction threshold.


### Shared-backend time records and user monitors

```python
# Earlier interface: whole-domain records and calibrated user planes are not in the current API yet.
import torch
from wavebit.compat import legacy as wavebit
from wavebit.compat.legacy import Stack, Incidence

value = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
scene = Stack.from_structure(value.reshape(1, 1, 1), [.3], top=1., bot=1., period=.4)
result, diag = wavebit.fdtd(
    scene, Incidence(wavelength=1., tetm='te'), backend='torch',
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

User monitor coordinates have `z=0` at the bottom interface and +z upward. Monitors
interpolate the core's collocated node planes linearly in z, and must stay outside
PML and away from the current sheet. Fields are calibrated independently at each
requested wavelength to unit incident electric amplitude at the illuminated interface.
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


### Half-space material derivatives

The current module refuses this derivative: a half-space reaches an `Open()` face, and the
PML there is built once from the material's concrete values, so a traced permittivity or
permeability in it raises `ValueError` before any time step. These examples keep the earlier
route.

With an explicit backend, `top` and `bot` accept positive real scalar
permittivities, or `Material(epsilon, mu)` with positive real scalar
coefficients. Array coefficients retain their graphs when both
`mesh_indices=(n_top, n_bottom)` and `truncation_order` are explicit.
`mesh_indices` contains **refractive indices**, while the materials continue
to take **permittivity and permeability**. Keep that nominal pair constant
over the optimization or finite-difference sweep.
NumPy vector indexing can return a host scalar rather than an array; still
provide the same explicit mesh indices and order window for every perturbed
evaluation. Recomputing the nominal pair from each perturbation changes the
experiment being compared.

```python
# Earlier interface: half-space material derivatives are not in the current API yet.
import jax
import jax.numpy as jnp
from wavebit.compat import legacy as wavebit

jax.config.update('jax_enable_x64', True)

def transmission(p):
    result = wavebit.fdtd(
        structure=jnp.full((1, 1, 1), 2.25), thickness=[.2],
        top=wavebit.Material(p[0], p[1]), bot=wavebit.Material(p[2], p[3]),
        period=.2, wavelength=1., backend='jax', resolution=20,
        mesh_indices=(1.1, 1.4), truncation_order=0,
        align_layers=True, n_periods=40,
    )
    return result.s_inc.T.sum()

value, gradient = jax.jit(jax.value_and_grad(transmission))(
    jnp.array([1.2, 1.1, 1.6, 1.15]))
```

The nominal indices select padding length and any graded z-cell sizes.
Actual epsilon/mu determine rasterized boundary cells, Maxwell updates,
CPML stretch, Bloch phase, directional projection, interface phase reference
and incident-field/power normalization. The mesh, source and monitor locations,
time step and update count stay fixed. The same contract applies to NumPy
finite differences and Torch autograd. Plain host coefficients retain the
previous automatic mesh selection when `mesh_indices` is omitted.

At fixed theta/phi, changing the **incident** medium changes the transverse
wavevector so that the central-wavelength incidence angle stays fixed. The
whole band shares that wavevector. This is not a derivative at fixed kx/ky.
Angles, raster materials and electric poles can vary jointly with the four
half-space coefficients, and with constant-total layer thicknesses using
`mesh_thickness`. Periods, wavelength and mesh geometry remain static;
lossy/dispersive half spaces are not supported.

The pair always names original **top, bottom**, including bottom illumination.
Diagnostics return static `mesh_indices` and live `background`, the latter a
`(2,2)` array of `(epsilon, mu)` in original top/bottom order. Increasing an
actual index can require a longer pulse run; it never extends the run silently.
Concrete invalid materials, inadequate duration, CFL violations and singular
channels raise. Under JAX JIT they invalidate `stable` and mask scattering and
normalized fields with NaNs. Optimizers must stay inside the valid domain.

Nominal indices do not guarantee physical mesh or PML accuracy across a wide
material range. Check resolution, padding, physical PML thickness and duration
at relevant design points. The [verification note](../../validation/fdtd/fdtd-halfspace-gradients.md)
separates finite-grid derivative correctness from continuum and external-solver
agreement.

### Internal layer thickness derivatives

In the current module a `SnappedSlab`'s span is static, so a traced `z` is refused, and a ranged,
differentiable `Box` layer cannot stand flush on a half-space `SnappedSlab` or another cell-snapped
face. The examples in this and the next section keep the earlier route.

Pass static `mesh_thickness` to choose the z mesh while backend `thickness`
values move the internal interfaces. The actual lengths must stay positive,
and their sum must equal the nominal total depth. This keeps both exterior
reference planes, sources, monitors, PML, record coordinates and time step
fixed. The spelling is shared with RCWA and typed `Stack` objects:

```python
# Earlier interface: layer-thickness derivatives are not in the current API yet.
import jax
import jax.numpy as jnp
from wavebit.compat.legacy import Stack, Incidence, fdtd

jax.config.update('jax_enable_x64', True)

def reflection(d):
    scene = Stack.from_structure(
        jnp.array([[[2.25]], [[1.5]]]), [.4-d, d],
        top=1., bot=2.25, period=.05,
    )
    light = Incidence(wavelength=1., polarization=(1., 0.))
    result = fdtd(
        scene, light, backend='jax', resolution=40,
        mesh_thickness=[.2, .2], align_layers=True, truncation_order=0,
        n_pml=40, n_periods=60, pulse_width=3.,
        diffraction_method='directional',
    )
    return result.s_inc.R.sum()

value, derivative = jax.jit(jax.value_and_grad(reflection))(.173)
```

Here the first, bottom layer is a substrate spacer. Its changing length keeps
the bottom output face at z=0 and the top illuminated face at z=0.4. For more layers, normalize positive design
fractions to a fixed total. Gradients are defined along constant-total paths;
they do not represent independent changes to the total stack depth.
The shared [`wavebit.layer_thicknesses`](../../reference/common/index.md)
function performs this normalization while enforcing specified minimum
lengths. The [multilayer optimization example](../../../examples/fdtd/fdtd_multilayer_optimization.py)
combines three independent thickness coordinates with three permittivities;
its [validation report](../../validation/fdtd/fdtd-multilayer-optimization.md) includes a dense
independent spectrum and refined gradients.

`subpixel=True` is required. Cell-overlap fractions retain the thickness graph
through epsilon/mu averaging and electric-pole masks. Values are continuous,
but derivatives may have kinks when a boundary crosses a cell face. Do not
use a subgradient at that crossing as a unique physical shape derivative.
Set the optional `layer_smoothing` described below for continuous first derivatives.
Refine the grid and sample several interface positions when assessing gradient
accuracy. The [validation note](../../validation/fdtd/fdtd-layer-gradients.md) separates AD/finite
differences, Airy convergence and independent public-API Meep perturbations.
The [optimization example](../../../examples/fdtd/fdtd_layer_optimization.py) includes
backtracking and a final check at twice the resolution.

JAX/Torch thickness arrays require explicit `mesh_thickness`; no eager tensor
is detached to select the mesh. NumPy finite differences use the same option.
`align_layers=True` fits nominal faces, while actual interfaces move through
the fixed cells. The option also works with grading, bottom illumination,
cropped records and field monitors. Diagnostics return original-order static
`mesh_thickness` and live `layer_planes`. Invalid concrete values raise; JIT
invalidity masks scattering and normalized fields and sets `stable=False`.

### Changing the total stack depth

Set `variable_depth=True` with an explicit backend, `mesh_thickness` and
`subpixel=True` to optimize layer lengths independently. All physical lengths
must remain positive. Their sum may differ from the static nominal sum.
The bottom face stays at z=0 and the top face is at z=sum(thickness).
The illuminated face is the entrance and the opposite face is the exit; complex
amplitudes are referenced to those physical faces, using the same convention as RCWA. Both the material fractions and reference-plane propagation phases
retain JAX/Torch derivatives.

```python
# Earlier interface: a moving outer face is not in the current API yet.
import jax
import numpy as np
from wavebit.compat.legacy import fdtd

jax.config.update('jax_enable_x64', True)

def transmission(d):
    result = fdtd(
        structure=np.array([[[2.25]]]), thickness=[d],
        mesh_thickness=[.2], variable_depth=True, layer_smoothing=.025,
        top=1., bot=1.21, period=.05, wavelength=1.,
        backend='jax', resolution=40, n_pml=40, n_periods=60,
        pulse_width=3., truncation_order=0, diffraction_method='directional',
    )
    return result.s_inc.t_s.sum()

amplitude, derivative = jax.jit(
    lambda d: (transmission(d), jax.jacfwd(transmission)(d))
)(.173)
```

`variable_depth` is a static boolean, defaulting to False. Nominal mesh lengths
still determine nodes, source sheets, extraction planes, PML, time step and
record samples. These laboratory coordinates remain fixed for both top and
bottom incidence. `source_distance` and `diffraction_distance` select distances
from nominal faces; they do not move the source or detectors with an optimized
boundary. Static user monitors and record bounds retain their original coordinates.
The returned fields remain normalized to unit incident E at the physical
incident face, so a moving face can change their normalization phase.

Every physical interface and its averaging support must stay clear of the
fixed diffraction stencils. Each exterior stencil has a conservative clearance
of one adjacent cell width. Increase nominal padding or adjust nominal lengths
when the intended design range would cross it. Eager invalid inputs raise;
traced invalid inputs give `stable=False` and NaN scattering/normalized fields.
Diagnostics expose `variable_depth`, static `mesh_thickness`, live `layer_planes`
and live `reflection_z`/`transmission_z` in the original coordinates.

Positive `layer_smoothing` also averages the moving exterior face, while the
fixed physical face at z=0 remains sharp. First and last layer lengths must be
at least half that width, including a single-layer stack. The physical graded
transition extends half a width past the exit reference plane. The reference
is the plane-wave extrapolation to its centre, not a claim that the material
there equals the outgoing half space. Zero smoothing retains sharp cell overlaps.

On fitted or graded meshes, propagation to a face inside a cell uses that
cell's discrete phase per unit length. At a node the right cell defines the
one-sided phase derivative. Different adjacent cell widths can therefore
produce a phase-slope kink even when material fractions are smoothed. Assess
physical accuracy with mesh refinement; see the
[moving-depth measurements](../../validation/fdtd/fdtd-variable-depth.md).

### Continuous first derivatives at internal interfaces

Set `layer_smoothing=h` with an explicit backend and `mesh_thickness` to average
each internal interface position uniformly over `[-h/2, h/2]` before material
sampling. `h` is a static, finite, nonnegative real scalar in the same physical
length unit as `thickness`. The default `0.` preserves exact cell overlaps.
Positive widths require `subpixel=True`; backend tensors, even eager scalar
tensors, cannot select this width. For a uniform mesh, `h=1/resolution` gives
one cell of averaging. On a fitted or graded mesh, use the actual cell widths
when choosing a physical width.

By default both outer stack faces remain fixed and sharp. Every internal interface must
stay at least `h/2` from the outer faces; internal averaging regions may overlap
one another. For `[d, D-d]`, this means `h/2 <= d <= D-h/2`, in addition to
positive lengths. Invalid concrete scenes raise; invalid JAX-traced scenes
use the same `stable=False` and NaN masking as other geometry failures.
Clearance checks allow the same 32 machine epsilons times D as the total-depth
check, so touching a face stays valid after floating-point reflection.
A one-layer stack has no internal interface, so the option has no physical
effect there unless `variable_depth=True` averages its moving exit as described
above. Diagnostics include the static `layer_smoothing` value.

The cell fractions have continuous first derivatives, including where a
moving boundary crosses a cell face. This is a C1 guarantee for internal
interface sampling, not for Boolean validity decisions, every possible loss
function, or higher derivatives. Epsilon/mu and electric-pole fills retain
their existing averaging rules and JAX/Torch graphs.

A positive width changes the material profile. Keep it fixed within one
optimization; shrink it along with the mesh and check the final sharp-interface
design separately when that is the intended physical structure. The
[smoothing measurements](../../validation/fdtd/fdtd-layer-smoothing.md) distinguish this bias from
finite-grid error. Run the coating example with `--smoothing-cells 1` for
backtracking optimization followed by both smoothed and sharp finer-grid checks.

### Bottom illumination in the common engine

In the current module, light from below is a `PlaneWave(direction='+z')` placed below the
stack, with the `wave='reflected'` monitor below it and the `wave='transmitted'` monitor above the
stack; the scene itself does not change.

On the earlier factory, set `Incidence(..., incidence_side='bottom')` or pass `incidence_side='bottom'`.
Both of its periodic routes support this; this section details the explicit engine. The structure, top/bottom materials and layer ordering
remain in the original scene. Angles are measured relative to the incidence
medium's inward normal, matching RCWA. R is reflected power into that medium;
T leaves through the opposite side. Complex r is referenced to the bottom
interface and t to the top interface. `incident`, `s_inc`, `p_inc` and order labels
keep their incidence-relative conventions.

The public adapter binds the lower and upper half-spaces to the appropriate
internal engine ports. All layer arrays remain bottom-to-top and all public
coordinates retain +z upward, independently of illumination. E is a polar vector
and H an axial vector when an internal solve uses reflected coordinates; the
adapter transforms both before returning fields. User-plane `power_z` is signed
laboratory flux: transmission from below normally has **positive** `power_z`,
while transmission from above is negative. Diffraction T is positive in either case.

This route supports the same raster, positive magnetic medium, electric pole,
Jones state, record and monitor options as top incidence, including JAX/Torch
material gradients. The backend-omitted NumPy route also supports bottom
incidence through the public adapter, with its own source normalization and
without autodiff. Its private stepping kernel remains one-way.
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
not read. See `tests/references/interface/validate_scene_bottom_meep.py` and the frozen reference
in `tests/references/golden_scene_bottom_meep.json`.


### Static nonuniform z meshes

The current module's grid is uniform and cubic; this section and the next describe the
earlier engine.

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
# Earlier interface: a graded z mesh is not in the current API yet (its grid is uniform).
import numpy as np
from wavebit.compat import legacy as wavebit
from wavebit.compat.legacy import Stack, Incidence

scene = Stack.from_structure(np.full((1, 1, 1), 2.25), [.3],
                             top=1., bot=1., period=.1)
result, diag = wavebit.fdtd(
    scene, Incidence(wavelength=1., tetm='te'), backend='jax',
    resolution=40, grading=1.06, n_periods=30,
    truncation_order=0, return_diagnostics=True,
)
print(diag['shape'], diag['z_spacing'].min(), diag['z_spacing'].max())
```

With the default `align_layers=False`, the structure remains uniform in z, with
both outer interfaces exactly on nodes. Internal layer interfaces may cut cells.
Half-space padding grows gradually to at most twice the core width, capped by
20 cells per shortest in-medium wavelength. If the requested mesh is already
coarser than that cap, grading does not further coarsen it. This geometry policy
depends on resolution and static nominal `mesh_indices` (actual host half-space
indices when omitted), not optimized raster or half-space values.
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

### Aligning every layer boundary

The explicit-backend scene API accepts `align_layers=True` independently of
`grading`. For a layer of thickness `d`, it uses `ceil(d * resolution)` equal
z cells with width `d / ceil(d * resolution)`. Every layer retains its physical
thickness; every boundary lies on a z node. Cell widths inside the structure are
at most `1/resolution`, and x/y sampling is unchanged. The default is `False`.

```python
# Earlier interface: layer-fitted z cells are not in the current API yet.
import numpy as np
from wavebit.compat import legacy as wavebit

result, diag = wavebit.fdtd(
    structure=np.array([[[2.25]], [[1.5]]]), thickness=[.113, .213],
    top=1., bot=1., period=.1, wavelength=1., tetm='tm', theta=.45,
    backend='numpy', resolution=40, align_layers=True,
    n_periods=50, pulse_width=3., truncation_order=0,
    return_diagnostics=True,
)
print(diag['align_layers'], diag['layer_planes'])
```

Without grading, padding keeps width `1/resolution`; a jump in cell width can
occur at a layer face. With grading, each half-space ramp starts from its adjacent
layer's cell width, with the same coarsening policy described above. Grading does
not smooth jumps between different layers. Material overlap weights, time step,
CPML, phase extraction and field coordinates all use the realized mesh.
`pml_bounds` includes the final closing cell at the outer boundary, so each
reported absorber span contains all `n_pml` cells.

Thin layers can force a much smaller global time step, including when a layer
fits in just one cell. Inspect `diag['dt']`, `diag['steps']`, `diag['shape']`,
`diag['z_nodes']` and `diag['z_spacing']` before scaling up. A fitted layer is not
necessarily well resolved. Mesh selection stays static. For thickness
differentiation, supply `mesh_thickness` as described above; alignment then fits
nominal faces. This is not adaptive refinement. The option requires a boolean
and explicit backend selection; the legacy backend-omitted path does not accept it.

Records, physical record bounds, frequency planes, bottom incidence and
NumPy/JAX/Torch material and pole gradients retain their usual semantics.
Bottom coordinates remain in the original stack frame. Alignment avoids mixing
different layers inside a normal Ez cell, but does not change the dispersive
averaging model or align lateral raster edges. The
[multilayer measurements](../../validation/fdtd/fdtd-layer-alignment.md) check complex TE/TM
refinement and distinguish this planar improvement from unresolved metal corners.

### Pulse-decay diagnostics in the common engine

`fdtd_grid(..., convergence_window=N, convergence_tol=1e-4)` inspects the last
`N` steps without changing the time loop or stopping early. `N` must be a positive
integer smaller than `steps`; the relative amplitude tolerance must lie strictly
between zero and one. The default `convergence_window=None` disables the extra
reductions and spectral accumulators. The explicit-backend scene API accepts the
same options with `return_diagnostics=True`.

The native result's `convergence` mapping contains:

| Field | Meaning |
| --- | --- |
| `field_decay` | Maximum field norm in the final window divided by the maximum over the entire run, including initial fields |
| `signal_present` | A finite, nonzero peak field norm was observed; a zero simulation is not evidence of convergence |
| `source_inactive` | The final-window pulse maximum is at most `convergence_tol` times its full-run maximum, or the current pattern is zero |
| `field_decayed` | Valid input/CFL, a nonzero signal, an inactive source, and `field_decay <= convergence_tol` |
| `spectral_tail` | For each wavelength and monitor plane, the joint E/H norm of the final-window DFT contribution divided by the full-run DFT norm |
| `spectral_signal_present` | The denominator of that spectral ratio is finite and nonzero |
| `window_steps`, `tolerance` | The requested settings |

Undefined ratios are NaN. The field norm is the square root of the volume-weighted
sum of `abs(E)**2 + abs(Z0 H)**2`, using each component's native control volumes,
including absorbing layers. E and H retain their half-step time separation. This
is a field-decay indicator, **not stored electromagnetic energy**, particularly
for dispersive media; ADE and PML auxiliary states are not included. Persistent
lossless fields and active continuous-wave sources should not pass this pulse
criterion. Choose a window long enough to cover the relevant oscillations and
propagation, and repeat with a longer run: a quiet window cannot exclude delayed
re-emission or a later source pulse.

The scene diagnostics contain `convergence['bases']`, with one mapping per source
basis and separate `actual` and `reference` native diagnostics. Their spectra are
uncalibrated basis spectra, in the native run's wavelength/monitor order (including
the extra carrier frequency when recording time fields). The scene-level
`field_decayed` requires all four runs to pass. This does not certify r/t accuracy,
spatial resolution, PML convergence, or the absence of long-lived material
oscillations. `stable` continues to mean material/CFL validity, not convergence.

The diagnostic arrays retain backend gradients; the maxima and boolean decisions
are not smooth everywhere. Fixed step counts preserve the existing differentiation
contract. The distinction between pulse decay and detector convergence was informed
by [FDTDX's stopping-condition design](https://github.com/ymahlau/fdtdx/blob/main/src/fdtdx/fdtd/stop_conditions.py).
This implementation does not copy its source code.

The native convergence mapping also returns `electric_spectrum_before_window`
and `magnetic_spectrum_before_window`, with the same shape and units as the full
spectra. These are prefix integrals obtained by subtracting the final-window
integral; extremely weak prefixes can lose relative precision.

For the scene API, `convergence['amplitude_change']` maps `r_s`, `r_p`, `t_s`,
`t_p` to **signed complex full-minus-prefix differences** for the requested
polarization, shaped `(wavelength, order_y, order_x)`, including a length-one
wavelength axis for scalar input. Both windows use their own incident-reference
calibration and the same interface reference planes. `comparison_time` is the
prefix end time. `calibration_valid` is a per-wavelength boolean: singular,
nonfinite, or numerically vanishing prefix references yield NaN changes.

`amplitudes_settled` requires finite valid calibration, inactive sources in all
four runs, valid material/CFL inputs, and every requested amplitude change at
most `convergence_tol` in absolute value. It is separate from `field_decayed`.
It does not certify relative accuracy of weak diffraction orders or future
settling. For measured examples of temporal settling with remaining spatial
error, see [pulse settling versus spatial accuracy](../../validation/fdtd/fdtd-pulse-convergence.md).

The same convergence mapping now includes `s_inc` and `p_inc`, matching the
two unit-electric-field incidence bases returned in `result.s_inc` and
`result.p_inc`. Each contains `amplitude_change`, `calibration_valid` and
`amplitudes_settled` with the same units, shapes, prefix calibration and validity
rules as the supplied-polarization quantities above. These are calibrated s/p
columns, not the raw current bases stored in `convergence['bases']`. They remain
independent of the supplied Jones vector. Existing top-level `amplitude_change`
and `amplitudes_settled` retain their supplied-polarization meaning.

```python
# Earlier interface: convergence-window diagnostics are not in the current API yet.
c = diagnostic['convergence']
print(c['s_inc']['amplitudes_settled'], c['p_inc']['amplitudes_settled'])
print(c['scattering_settled'])
```

`scattering_change_bound` maps each of `r_s`, `r_p`, `t_s`, `t_p` to the largest
possible absolute full-minus-prefix change for **any unit Jones input**, separately
at each wavelength and outgoing order. For an output row with changes `ds` and
`dp` in the two incident columns, its bound is `sqrt(abs(ds)**2 + abs(dp)**2)`.
The norm accounts for constructive interference between the columns; checking
each column against a tolerance alone does not provide this bound. A single Jones
vector need not attain all output rows' bounds at once.

`scattering_settled` requires all these bounds to be at most `convergence_tol`,
both valid calibrated columns, inactive sources and valid material/CFL inputs.
It does not require `state_decayed` and is not an error bound against the
continuum solution. An invalid prefix gives NaN bounds and a false flag.
The arrays retain material gradients; at an exactly zero row norm, the reported
bound uses a zero subgradient. Flags are boolean decisions, not smooth objectives.

All of these fields are enabled by the existing `convergence_window`; they reuse
the four scene runs and prefix spectra, without additional time stepping.
There is still no automatic termination. The [lossy ENZ measurements](../../validation/fdtd/fdtd-enz.md)
show a case where the s column settles well before the p column, and where both
eventually pass despite a significant time-discretization error.

### Subsampling frequency monitors

`dft_subsample=K` is available on `fdtd_grid` and the explicit-backend scene API.
The default is `1`. It must be a positive integer no larger than the number of
steps. With `K > 1`, all requested wavelengths must stay strictly below the
sampling Nyquist frequency, including the extra carrier used for time records.
This check does not establish that the entire transient signal is band-limited.

The engine evaluates DFT planes after updates `0,K,2K,...`, with the existing E/H
half-step timestamps. A sample represents `min(K, steps-n)` updates; the final
partial block is weighted accordingly. This is left-sample quadrature of the
**demodulated** field. No anti-alias filter is applied. Compare with `K=1` before
using it on broadband pulses, rapidly changing fields, or resonant materials.
There is no automatic stride selection.

```python
# Earlier interface: DFT subsampling is not in the current API yet.
result, diagnostic = wavebit.fdtd(
    stack, incidence, backend='numpy', resolution=40,
    dft_subsample=2, return_diagnostics=True)
print(diagnostic['dft_samples'])
```

Every Maxwell/ADE/CPML update is still executed. Unsampled steps skip monitor
interpolation and Fourier accumulation through a Python branch on NumPy/Torch
and `lax.cond` on JAX. Accumulator storage is unchanged. Native final fields and
raw time records are unchanged; calibrated scene records can inherit the small
change in reference normalization. The result/scene diagnostics report
`dft_subsample` and `dft_samples` (zero for native runs without spectra).

With `convergence_window`, a sample block crossing the prefix boundary splits
its quadrature weight between prefix and tail. Thus the prefix spectrum matches
a shorter run with the same stride and starting sample. The field-decay norm is
still evaluated at every update. An apparently settled subsampled spectrum does
not certify freedom from aliasing; see the [sampling measurement](../../validation/fdtd/fdtd-dft-sampling.md).

### Recording a region of the native grid

`fdtd_grid(..., record_field=N, record_region=(z, y, x))` stores only the selected
region at each requested snapshot. Each entry is an integer index or a slice
with positive step; negative indices and slice bounds follow Python indexing.
Integer selections preserve length-one axes. Empty regions, reversed slices,
out-of-range integer indices and noninteger entries are refused. The region is
static and requires `record_field`; it is available on the native grid API.

```python
# Earlier interface: the native grid and its record regions are not in the current API yet.
out = wavebit.fdtd_grid(
    epsilon, spacing=(dz, dy, dx), dt=dt, steps=steps,
    record_field=10, record_components=('ey', 'hx'),
    record_region=(slice(4, 20, 2), 0, slice(None)))
plane_history = out['fields']['ey']  # values: (time, 8, 1, nx)
```

Selection occurs before stacking snapshots in NumPy, JAX and Torch. It reduces
retained field-history arrays; it does not reduce the Maxwell grid, checkpoint
state, final E/H arrays or frequency-monitor accumulators. Raw values, timestamps
and coordinate vectors match the same region selected from full-domain records.
Yee staggering is preserved: the same cell index can have different physical
coordinates for different field components. With nonuniform z spacing, use the
returned component coordinates instead of reconstructing them from nominal dz.

For example, selecting `(3,1,2)` cells from a `(12,4,6)` grid stores 48 times fewer
field-history values at the same components and snapshot count. This is an array
storage ratio, not a peak-process-memory or speed claim. Spatial subsampling
selects nodes without filtering or interpolation. Region coordinates are not
differentiable; recorded field values retain material/source gradients.

### Physical record bounds for Stack scenes

The explicit-backend `fdtd` scene API accepts `record_bounds`, a nonempty mapping
from any of `x`, `y`, `z` to `(lower, upper)` physical coordinates. It requires
`record_field` and `return_diagnostics=True`. Bounds must be finite real numbers
with `lower <= upper`; unspecified axes keep their entire extent.

```python
# Earlier interface: whole-field records within physical bounds are not in the current API yet.
result, diagnostic = wavebit.fdtd(
    stack, incidence, backend='jax', resolution=40,
    record_field=10, record_components=('ey', 'hx'),
    record_bounds={'z': (-0.2, 0.5), 'x': (-0.1, 0.1)},
    return_diagnostics=True)
```

Use the same coordinates as returned field records: x/y are centered on the
periodic cell, z=0 is the bottom interface and positive z proceeds upward.
These coordinates stay fixed for bottom incidence. A y-invariant scene exposes
only y=0. Bounds select native sample coordinates inclusively, with an eight-epsilon
floating-point tolerance scaled by coordinate magnitude. Equal bounds select an
existing coordinate plane; they do not request interpolation. Bounds extending
outside the simulation select their intersection with its samples.

Each requested component must have at least one sample on every axis, otherwise
the call raises an error suggesting wider bounds or finer resolution. Because
components are staggered, their returned coordinate arrays and shapes can differ.
Use each component's own coordinates when combining or interpolating fields.
The `field_e`/`field_z` convenience aliases still follow the primary recorded
component.

The engine stores the smallest common native box covering the selected component
samples, then trims individual staggered boundaries. Spatial selection happens before
snapshot accumulation; bounds covering the full domain still record that domain. The full simulation, frequency monitors and material
gradients retain their original behavior. Bounds are static geometry, not
optimization parameters. For direct grid-index selection on `fdtd_grid`, use
`record_region` instead.

### Fitting frequencies and decay rates from a time trace

`wavebit.compat.legacy.fit_fdtd_modes(t, signal, modes=...)` analyzes a single point/component
after the source becomes negligible. It has no current equivalent yet. It is NumPy
post-processing of any uniformly sampled trace, so the series of a current `fdtd.TimeSeriesMonitor`
(`result[name].times_seconds` in seconds, `.values`) also works; frequencies and decay rates then come
out in Hz and 1/s. Select a trace from the recorded fields:

```python
# Earlier interface: fit_fdtd_modes and whole-field records are not in the current API yet.
field = diagnostics['fields']['ey']
signal = np.asarray(field['values'])[:, 0, 0, 0]
fit = wavebit.fit_fdtd_modes(
    np.asarray(field['t']), signal, modes=2, start=60., stop=120.,
)
print(fit['frequency'], fit['decay_rate'], fit['relative_residual'])
```

The example assumes that the recorded interval contains 60 through 120 and the
source is negligible there. Choose the indices using that component's own
staggered coordinates. The analysis uses NumPy CPU arrays; move JAX/Torch data
to the host explicitly and detach Torch records when they carry gradients. It
does not alter the simulation or provide gradients through the fit.

The model sums `amplitude * exp((-decay_rate-2j*pi*frequency)*(t-time_origin))`.
Amplitudes refer to the first selected sample. `frequency` is in cycles per c=1
time unit, and positive `decay_rate` describes amplitude decay, not energy decay.
`quality_factor` is `pi*abs(frequency)/decay_rate`; nonpositive decay produces
NaN Q and is retained for inspection. Very small fitted decay may be roundoff,
so a large Q requires independent checks. Real cosines need both conjugate
frequencies: use two modes per damped cosine. Complex records need not contain
conjugate pairs. The default requests one exponential.

The selected samples must be uniformly spaced. Frequencies use the signed
Nyquist branch; the function does not filter or repair aliased records. At least
`2*modes+1` samples are required. Optional `pencil` chooses the Hankel matrix's
column count, between `modes` and `samples-modes`. Its default is
`min(samples//2, max(128, 2*modes))`, with storage proportional to
`samples*pencil`. `rcond=1e-10` is a relative singular-value floor. Unresolved
mode counts and amplitude bases are refused. This floor is not a physical noise
estimate or mode-frequency uncertainty.

Results include frequency-sorted modes, normalized `singular_values`,
`basis_condition`, the selected `t` and reconstructed `fitted` trace.
`relative_residual` is the fit error over this interval only. Compare different
start/stop times, mode counts and pencil widths before identifying a resonance.
An algebraic tail can be approximated by several exponentials with a small
residual but changing frequencies and decay rates. This neither certifies Q nor
justifies automatically extrapolating a DFT to infinite time. The
[ringdown measurements](../../validation/fdtd/fdtd-ringdown.md) show this limitation near a diffraction
threshold and validate the fitter against exact modes and external Meep records.

### Checking residual material oscillations

When `convergence_window` is enabled, the native convergence mapping also reports
activity for every ADE pole, in input pole order. At each E-component control
volume it evaluates

$$
A_j=\sum_{\text{cells, components}} \Delta V\,
\frac{\omega_{0,j}^2 |P_j|^2+|(P_j-P_{j,\mathrm{previous}})/\Delta t|^2}{s_j},
$$

where $s_j$ is pole strength and cells with zero strength contribute zero.
This positive oscillator activity combines polarization and its discrete
velocity. They are evaluated at different time levels, so it is not a conserved
discrete energy or a complete electromagnetic/PML energy measurement. For Drude
poles, omega0=0: a static polarization offset has no activity once its velocity
vanishes.

| Field | Meaning |
| --- | --- |
| `material_activity_peak` | Maximum activity for each pole over the full run |
| `material_activity_tail` | Maximum activity for each pole over the last requested window |
| `material_signal_present` | Each pole's activity peak is finite and positive |
| `material_decay` | Square root of tail/peak for each excited pole; NaN for unexcited poles |
| `material_decayed` | Valid material/CFL inputs and finite activity; every excited pole has `material_decay <= convergence_tol` |
| `state_decayed` | Both `field_decayed` and `material_decayed` |

Absent poles yield empty per-pole arrays and `material_decayed=True`. Unexcited
poles do not block the material flag. Each excited pole uses its own peak, so even
a weakly excited slow oscillator can keep this conservative flag false. The
combined flag still requires a nonzero observed field and an inactive source.
It does not inspect CPML auxiliary memory, establish spatial accuracy, or
predict future behavior.

The scene diagnostics expose `material_decayed` and `state_decayed` aggregated
over both actual and reference runs for both source bases; per-pole values remain
in `convergence['bases']`. Existing `field_decayed` and `amplitudes_settled` retain
their separate meanings. All diagnostics remain opt-in, fixed-step and backend
native. Activity reductions add work proportional to the number of poles and
grid cells, retaining only two scalars per pole across the time loop. Ratios and
maxima retain gradients where differentiable.

See the [low-loss mixed-metal measurement](../../validation/fdtd/fdtd-mixed-metal.md) for a case
where calibrated amplitudes settle before material motion decays, together with
separate checks of grid error and material-parameter sensitivities.

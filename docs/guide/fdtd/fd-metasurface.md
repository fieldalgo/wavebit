# Periodic metasurfaces with the general FDTD engine

:::{note}
The FDTD example on this page retains the normalized declarations used for its historical measurements, through `wavebit.compat.legacy.fd`; the RCWA comparison uses the current `wavebit.rcwa`. The current FDTD API now supports the fixed Box substrate through explicit `Structure.continuation`; see [smoothed substrates](substrate-gradients.md) for its physical units and declaration constraints. This page's measurements have not been rerun through that API.
:::

`fd` solves one cell of a periodic array under a plane wave at any angle and reads the
light that leaves it as diffraction orders: each order's complex amplitude in its own s/p frame
and its efficiency, in the same `Result` layout the earlier `wavebit.compat.legacy.rcwa`
returns. This page builds such a cell, says how to choose its period and angle, how to read the
orders, and how to compare them with RCWA. The measurements behind it are in
[`fd-metasurface-validation.md`](../../validation/fdtd/fd-metasurface-validation.md).

## A grating cell

A TiO2-like pillar on glass, 0.6 wavelengths apart, lit at 30 degrees: the light that goes
through it leaves in two orders, `(0, 0)` and `(-1, 0)`, and the share of the second falls
across the band (60% of the incident s power at 0.95, 21% at 1.0, 13% at 1.05).

```python
# Earlier normalized declarations retained with the measurements on this page.
from wavebit.compat.legacy import fd

N = 25                                    # cells per vacuum wavelength
DX = 1.0 / N
GLASS = fd.Material(eps=2.13)
H = 0.625                                 # pillar height; the substrate's face is z = 0

pw = fd.PlaneWave('pw', z=H + 0.6, direction='-z', theta=30.0, phi=0.0,
                  polarization='s', band=fd.Band(wavelength=1.0, df=0.06))
substrate = fd.Structure(fd.Box(size=(10.0, 10.0, 3.0), center=(0.0, 0.0, -1.5)), GLASS)
pillar = fd.Structure(fd.Cylinder(radius=0.15, height=H, axis='z', center=(0.0, 0.0, H / 2)),
                      fd.Material(eps=5.8))
wavelengths = (0.95, 1.0, 1.05)
Z_LO, Z_HI = -0.8 - 24 * DX, H + 1.4 + 24 * DX   # 0.3 past each monitor, then the absorber
scene = fd.Scene(
    domain=fd.Domain(size=(0.6, 0.6, Z_HI - Z_LO), center=(0.0, 0.0, (Z_HI + Z_LO) / 2)),
    boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=(
        fd.PML(cells=24, reflection=1e-12, background=GLASS),
        fd.PML(cells=24, reflection=1e-12))),
    resolution=fd.Resolution(dx=DX), run=fd.Run(periods=120),
    structures=(substrate, pillar), sources=(pw,),
    monitors=(fd.Diffraction('r', z=H + 1.1, side='reflect', wavelengths=wavelengths,
                             reference_plane=H, incident_plane=H),
              fd.Diffraction('t', z=-0.5, side='transmit', wavelengths=wavelengths,
                             reference_plane=0.0, incident_plane=H)))

res = fd.run(scene, 'numpy')
t = res['t'].orders
print(res['r'].theta_at)                  # each wavelength's own angle, degrees
print(t.order(-1, 0).T, t.order(0, 0).T)  # efficiencies per wavelength
```

What each part is doing:

* **One cell, Bloch faces from the source.** The domain is one period along x and y, and
  `fd.Bloch.from_source(pw)` gives those faces the plane wave's own phase.
* **The substrate runs through the bottom absorber**, and that absorber is told what it
  absorbs: `fd.PML(..., background=GLASS)`. The substrate is a smoothed `fd.Box`, wider than the
  cell, so it fills every layer below `z = 0` exactly.
* **The pillar stands flush on the substrate.** Its base is the substrate's face; the cells
  that hold both are treated by `Structure(junction='split')`, the default (see
  [`fd-differentiation.md`](fd-differentiation.md) and `fd.check(scene)`).
* **Both monitors sit in one uniform medium each** -- air above the source, glass below the
  pillar -- and refer their amplitudes where RCWA does: `r` to the pillar's top, `t` to the
  substrate's face, both against the incident wave at the top (`incident_plane=H`).
* **The absorbers are 24 cells at a 1e-12 target** rather than the default, because the
  oblique orders leave at `cos theta` near 0.55 and an absorber returns roughly its target to
  the power `0.919 cos theta`.

## Choosing the period and the angle

A plane wave in `fd` keeps its transverse wavevector fixed across its band: the angle
you give is exact at the band's centre wavelength, and every other wavelength gets its own
(`res['r'].theta_at`). An order `(m, n)` then has the transverse wavevector

```
q = (sin(theta) cos(phi) + m / Px,  sin(theta) sin(phi) + n / Py)
```

in units of the centre wavelength's vacuum wavenumber, and at wavelength `lambda` it
propagates in a medium of index `n_med` when `lambda |q| < n_med`. Two things go wrong near
`lambda |q| = n_med`, and the planner refuses or warns about both rather than returning a
plausible number:

* an order leaving close to the surface (small `cos theta`) is not absorbed well -- below
  `cos theta = 0.41` at the default absorber the plan refuses it;
* an order crossing its cutoff inside the band (a Rayleigh anomaly) makes the run slow to
  settle and the answer sharp in wavelength.

A period and an angle where every order is clearly in or clearly out over the whole band are
worth a minute's search before a long run:

```python
import math

def layout(Px, Py, theta, phi, n_sub, wavelengths=(0.95, 1.0, 1.05), reach=4):
    s = math.sin(math.radians(theta))
    q0 = (s * math.cos(math.radians(phi)), s * math.sin(math.radians(phi)))
    for side, n_med in (('r', 1.0), ('t', n_sub)):
        for m in range(-reach, reach + 1):
            for n in range(-reach, reach + 1):
                u = math.hypot(q0[0] + m / Px, q0[1] + n / Py)
                ratios = [lam * u / n_med for lam in wavelengths]
                if min(ratios) < 1:
                    print(side, (m, n), 'propagates' if max(ratios) < 1 else 'CROSSES CUTOFF',
                          [round(r, 3) for r in ratios])

layout(0.6, 0.6, 30.0, 0.0, math.sqrt(2.13))
```

A ratio above about 0.87 (`cos theta` below 0.5) or within a few per cent above 1 is where the
trouble is.

## Reading the orders

`res['r'].orders` and `res['t'].orders` are `wavebit.observations.diffraction.Result`s:

* `.order(m, n)` gives one order's numbers, each an array over the monitor's wavelengths:
  efficiencies `R` / `T` and their s/p parts, amplitudes `r_s`, `r_p` (reflected) or `t_s`,
  `t_p` (transmitted) in that order's own s/p frame, with RCWA's p sign convention.
* `.R`, `.T` and the amplitudes are also whole arrays of shape `(wavelengths, n_v, n_u)` --
  the `v` (y) order index first -- labelled by `.orders_v` and `.orders_u`. Use `.order(m, n)`
  rather than indexing by position.
* By default a monitor keeps every order that propagates at any of its wavelengths in the
  medium it sits in; `orders=` widens or narrows that window.
* `polarizations='both'` on a monitor runs the scene a second time with the other polarisation
  and fills `.s_inc` and `.p_inc` beside `.incident`.

A current `fdtd.Diffraction` monitor returns the same selection as a `DiffractionOrders`:
`.order(m, n)` gives `efficiency`, `amplitude_s` and `amplitude_p` over its wavelengths, the
whole arrays have the same `(wavelengths, n_v, n_u)` shape, and each wavelength's angle is
`metadata['theta_radians']`. It has no `polarizations='both'` yet.

## Comparing with RCWA

Solve the same cell with `wavebit.rcwa` at each wavelength's own angle:

```python
import math
import numpy as np
from wavebit import Material, Units, rcwa

theta_at = np.asarray(res['r'].theta_at)
for i, lam in enumerate(wavelengths):
    problem = rcwa.Problem(
        units=Units(length='um'),
        stack=rcwa.Stack(
            period=(0.6, 0.6), top=Material(epsilon=1.0), bottom=Material(epsilon=2.13),
            layers=(rcwa.Layer(thickness=H, background=Material(epsilon=1.0), regions=(
                rcwa.Region(geometry=rcwa.Circle(radius=0.15), material=Material(epsilon=5.8)),)),)),
        excitation=rcwa.Excitation(wavelength=lam, sources=(rcwa.PlaneWave(
            side='top', polarization='s', theta=math.radians(theta_at[i]), phi=0.0),)))
    ref = rcwa.run(problem, numerics=rcwa.Numerics(orders=(15, 15)))
    print(lam, t.order(-1, 0).t_s[i], complex(ref.transmission.order(-1, 0).amplitude_s[0]))
```

`polarization='s'` is s incidence and `'p'` is p; a complex `(a_s, a_p)` pair is any mixture
of the two.

**Where the pillar is matters for the phases.** Both engines quote an order's amplitude at the
cell's centre, and RCWA places its shape from there, but `fd` places the pillar
wherever its `center=` says. If the pillar sits at `(xc, yc)` from the centre and the RCWA
shape at `(0, 0)`, order `(m, n)`'s amplitude differs by `exp(-2 pi i (m xc / Px + n yc / Py))`. The
efficiencies and the zeroth order do not care. Either give the RCWA circle the same
`center=(xc, yc)`, or multiply the FDTD amplitudes by `exp(+2 pi i (m xc / Px + n yc / Py))`
before comparing.

**RCWA has an error too.** Raise its `Numerics(orders=...)` until the numbers you compare stop
moving by much less than the difference you are judging. A high-index rectangle converges
slowly on RCWA's shape path; on a raster (`rcwa.Layer(epsilon=...)`) whose pixel edges are the
rectangle's edges, with `Numerics(fourier_coefficients='integrated', factorization='li')`, it
converges faster, and its narrow side needs as many orders as its long one.

## How fine a grid

The C7 campaign measured three cells against RCWA at three resolutions each. The numbers are
the worst error over every propagating order, both outgoing polarisations, both incident
polarisations and three wavelengths -- the worst cell, not a typical one:

| Pillar | Cells per wavelength | Worst amplitude error | Worst efficiency error |
| --- | --- | --- | --- |
| epsilon-4 cylinder on glass | 20 / 40 / 80 | 2.9e-2 / 7.4e-3 / 2.1e-3 | 1.7e-2 / 4.6e-3 / 1.4e-3 |
| epsilon-5.8 cylinder on glass | 25 / 50 / 100 | 7.4e-2 / 2.0e-2 / 6.1e-3 | 1.8e-2 / 4.5e-3 / 1.1e-3 |
| epsilon-12 box on glass, conical | 40 / 80 / 120 | 2.2e-2 / 7.1e-3 / 3.6e-3 | 1.2e-2 / 4.1e-3 / 2.2e-3 |

"Cells per wavelength" counts the vacuum wavelength; the planner refuses fewer than 8 cells per
wavelength *inside* the densest material (so 25 is the floor for epsilon 5.8 at this band, 40
for epsilon 12) and warns below 12. The two cylinders' largest errors fall at second order -- each
doubling divides them by about four; the epsilon-12 box's falls at order 1.5 to 1.75, so a
doubling divides it by about three ([`fd-metasurface-validation.md`](../../validation/fdtd/fd-metasurface-validation.md)
says what is and is not known about why). The slower rate does not need the box's contact with
the glass. The same box lifted 0.1 off its substrate converges at the same rate, with errors
1.3 times (amplitudes) and 1.6 to 1.7 times (efficiencies) larger. Whether the contact adds an
error of its own when the box stands on the glass is not known, because lifting the box also
leaves its base edges in air. Leaving a gap under the pillar will not buy back the
order. Run as 2D ridges, the box's xz cross-section converges at second order through 320
cells per wavelength. The yz cross-section has slower individual amplitudes through 640:
at wavelength 0.95 its s-to-s transmitted amplitude has a reference-based order about 1.52
over the final doubling, and a reference-free self-order about 1.59 over 160/320/640.
The registered time and absorber checks resolve that low order. Moving the ridge's strong
sampled reflection peak outside the measured band by reducing its height also leaves low
orders. Changing height changes the corner fields too, so these controls do not identify the
mechanism or show that it causes the 3D box's coarser-grid behaviour. This extension adds no
new 3D convergence ladder; the existing 40-cell M3 regression row is replayed by the slow tests.

A later 3D comparison doubles the box's height and uses a common 40/60/80
ladder after a separate run-length calibration. With time and absorber controls,
the original height has 22 low, 23 second and nine mixed self-order cells; the
tall box has 11 low, 20 second and 23 mixed. The changes go in both directions,
so height does not restore uniform second order. These are finite-grid
reference-free classes: the capped RCWA ladder cannot resolve any tall cell's
final direct-error order under the registered spread rule. Fewer low cells do
not establish a more accurate design or identify the slow-convergence cause.
C7 remains unpassed; the full 108-cell comparison is in the validation document.

Three practical consequences:

* **For a percent on every order, 40 cells per wavelength is about right for these indices**,
  and a coarse run is a fair first look: at the floor the worst order is off by a few percent,
  most by much less.
* **Check the wavelength and observable you care about at several resolutions.** A maximum
  over many cells can hide smaller components that converge more slowly. If the error follows
  a single power `C n^(-p)` and the number moves by `d` from `n` to `2n` cells per wavelength,
  the finer error is `d/(2^p - 1)`: `d/3` at second order and `d` at first order. Use at least
  three resolutions to estimate the order and check its stability before using the second-order
  Richardson estimate `X(2n) + (X(2n) - X(n))/3`. The ridge extension contains cancellations
  for which a single-power model does not describe the sampled rungs.
* **Check run length separately from mesh refinement.** Doubling M3's height to 0.60
  leaves a substantial transient: at 40 cells, both 120 and 180 periods fail the
  1e-5 monitor-decay line. At 240 periods the decay check passes, but the largest
  `|R + T - 1|` across the band is still about 0.5%. That warning line is not an
  error bound on a spectral observable. Compare longer runs and thicker absorbers
  at matched times, and require their changes to be small relative to the grid
  changes being interpreted. The time calibration is recorded in the validation
  document above; it is not itself a spatial-convergence result.

For a fixed duration, a run's cost grows as the fourth power of the resolution
(three dimensions and the time step). A longer-lived geometry can also require more periods.
On a 12-thread CPU with JAX the 80-cell epsilon-4 cell took 20 minutes and the 120-cell
epsilon-12 cell 84; at 20 to 40 cells a run takes seconds to a few minutes.

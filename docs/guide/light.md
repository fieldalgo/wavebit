# Saying what the light is

Specify the incident direction with either angles or transverse wavevector components.
Polarization, illuminated diffraction orders and incidence side are separate settings.
See [Keyword reference](../reference/keywords.md#the-light) for defaults.

## Two ways to specify direction

| Input | Meaning | Held fixed during a wavelength sweep |
| --- | --- | --- |
| `theta`, `phi` | phase polar angle and azimuth, in radians | angles; k changes with wavelength and incident material dispersion |
| `kx`, `ky` | real Cartesian transverse angular wavenumbers | physical kx and ky; the phase angle may change |

`kx` and `ky` have inverse length units: nm**-1 when `wavelength`, `period` and
`thickness` use nm. They are **not divided by k0** and are not reciprocal-lattice
order numbers. An omitted component is zero. Giving either component together
with a non-`None` `theta` or `phi` raises an error, even if an angle is zero.
With neither form supplied, the default is normal incidence with azimuth zero.

For example, these specify the same wave in a lossless medium with epsilon=2.25
and mu=1 (n=1.5):

```python
import numpy as np
import wavebit

scene = dict(
    top=2.25, bot=1., wavelength=500., period=[700., 700.],
    structure=np.full((1, 2, 2), 4.), thickness=[100.],
    truncation_order=0, polarization=(1., 0.),
)
by_angle = wavebit.rcwa(**scene, theta=np.pi/6, phi=0.)
k0 = 2*np.pi/scene['wavelength']
by_k = wavebit.rcwa(**scene, kx=0.75*k0, ky=0.)
a, b = by_angle.solve(), by_k.solve()
np.testing.assert_allclose(a.T, b.T, atol=1e-12)
```

Both forms still require the actual `top`/`bot` materials: epsilon and mu determine
the normal wavevector, the electric/magnetic field relation and incident power.
Direct k also works in absorbing half-spaces. It specifies the wave inside that
half-space, without assuming an additional medium above it.

The components use the fixed physical x/y axes for either incidence side and for
oblique lattices. The solver adds reciprocal-lattice offsets for each diffraction
order. `order_kxy()` reports those **per-order components divided by k0**; multiply
by `2*pi/wavelength` to recover inverse length units.

Direct components must be finite real scalars. This sets uniform single-wave
intensity along the surface; a zero-power evanescent incident wave cannot define
normalized diffraction efficiencies and is refused. A complex component is refused
too, in absorbing media as well: there a real `theta` is the input, and the phase
angle keeps `kx` real. A complex transverse wavevector is reachable only through a
complex `theta`/`phi`, whose `R` and `T` are formal (see
[Phase incidence](../conventions.md#phase-incidence)). With `kx=ky=0`, the s/p basis
uses azimuth zero (s along y, p in xz); choose `polarization` to orient the field.
For pure 1D `tetm` input, omit `ky` (it is zero); use `polarization` for conical incidence.

`wavebit.Incidence(kx=..., ky=...)`, `JobSpec` serialization, `kx`/`ky` sweeps and
JAX/PyTorch differentiation use the same inputs. To switch an existing solver
from angles to direct k, clear `theta` and `phi` with `None` before assigning k.
To switch back, clear both k components first. Changing k requires another solve
before reconstructing fields.

## Phase direction and absorbing incidence

`theta` is the angle of the incident wave's **phase direction** from the inward
normal; `phi` is its azimuth in the fixed x/y axes. Real angles specify uniform
incident intensity along the surface. They refer to the selected `top` or `bot`
half-space itself. No additional air layer is assumed.

In an absorbing incident medium, Wavebit computes real transverse wavevectors
from that phase direction and the full complex permittivity and permeability.
The normal wavevector remains complex and describes absorption during propagation.
For a lossless positive-index medium, this reduces to the familiar
`kx = k0*n*sin(theta)*cos(phi)` and `ky = k0*n*sin(theta)*sin(phi)`.

```python
import numpy as np
import wavebit

solver = wavebit.rcwa(
    top=4. + 3j, bot=2.25,
    wavelength=1000., period=[1200., 1200.],
    theta=np.deg2rad(30.), phi=0.,
    structure=np.full((1, 2, 2), 2.25), thickness=[100.],
    truncation_order=0,
)
kx_over_k0, ky_over_k0 = solver.order_kxy()
kx = (2 * np.pi / solver.wavelength) * kx_over_k0
ky = (2 * np.pi / solver.wavelength) * ky_over_k0
# With lengths in nm: kx ≈ [0.00677139] nm^-1, ky = [0.] nm^-1.
result = solver.solve()
```

Here 30 degrees is the phase angle **inside the absorbing incident half-space**.
In absorbing media the phase direction and energy-flow direction need not agree.
The [derivation and complex-angle convention](../conventions.md#phase-incidence)
state the precise meaning of the input. Absorbing phase incidence requires
`abs(theta) < pi/2`.

Power is referenced to the incident wave at its interface with the finite stack.
`result.power_entering` includes incident/reflected interference;
`result.absorption = result.power_entering - result.T.sum()` is the absorption
in all finite layers. It excludes absorption in either half-space.

## The polarization comes in three kinds

| | write | what it is |
| --- | --- | --- |
| **declaration** | `tetm='te'` / `'tm'` | this problem has no polarization and no azimuth degree of freedom. 1D structures only, and it is what buys the pure TE/TM formulation -- 2 to 4 times faster than the conical one, which gives the same amplitudes, fields and efficiencies as `polarization=(1, 0)` / `(0, 1)` at `phi=0`, to 1e-13 (since 2026-09-27; before, the amplitudes differed by a sign on the orders across `kx = 0` from the incident one) |
| **coordinates** | `psi`, `chi` | the state's two angles: `psi` the ellipse's tilt, `chi` its ellipticity. `chi=0` is linear (and is exactly the `psi` spelling), `chi=+/-pi/4` circular. This is the form to sweep and to optimize |
| **amplitudes** | `polarization=(a_s, a_p)` | the Jones pair itself, in the same (s, p) basis the results come back in. The only form that carries an overall phase; its magnitude is normalized away, so `(3, 4j)` and `(0.6, 0.8j)` are one incidence |

```python
wavebit.rcwa(tetm='te', ...)                    # fast 1D path; psi reads back as None
wavebit.rcwa(psi=0.4, ...)                      # linear, at an angle
wavebit.rcwa(psi=0.4, chi=np.pi / 4, ...)       # circular
wavebit.rcwa(polarization=(1, -1j), ...)        # the pair, straight from another code
```

`tetm`, `psi`/`chi` and `polarization` are three *kinds* of thing, not three spellings of one.
Naming `tetm` with either of the others raises, and so does `psi` with `polarization`. Which to
reach for is usually decided by what you are doing: `(psi, chi)` is two real numbers with
nothing redundant in them, so a gradient over the polarization state is an ordinary 2D problem,
where the pair spends two of its four real numbers on directions that cannot change any power.

Every order named in `excitation` has to propagate in the incidence medium. One past the light
line would drive the structure while contributing nothing to the incident flux every efficiency
is divided by, so `R + T` would exceed one on a passive stack; asking for one is refused by
name. That also caps how many beams can be combined, at roughly `2 * period / wavelength`.

An `excitation` weight multiplies *its own order's* `s_hat` or `p_hat`, on every path. On a
1D solve at `phi = 0` that is `sign(kx) * y_hat` for TE, so a lit order with `kx < 0` arrives
polarized along `-y`; see [`../conventions.md`](../conventions.md).

See [`../conventions.md`](../conventions.md) for the normative statement,
[`../multimode-incidence.md`](../multimode-incidence.md) for `excitation`, and tutorials
[`06`](https://github.com/fieldalgo/wavebit/blob/develop/tutorials/06-polarization-pair.ipynb) and
[`07`](https://github.com/fieldalgo/wavebit/blob/develop/tutorials/07-multimode-incidence.ipynb).

## Which side the light arrives from

`incidence_side='bottom'` illuminates through the substrate. **The scene labels never move**:
`top` is still the superstrate, `bot` still the substrate, and `structure[0]` is still the layer
adjacent to `bot`. What changes is that results are incidence-relative and `theta` / `phi` are
measured in the incidence medium -- so a cross-solver comparison needs the angle converted into
that medium before the numbers line up.

`incidence_side='top'` propagates toward `-z`; `'bottom'` propagates toward `+z`.
`theta` is the angle from the inward normal of the selected face. Layer order is always
bottom-to-top, and `phi` uses the same fixed x/y axes on either side.

## Beams from both sides or with independent directions

Use `excitation={"top": wavebit.PlaneWave(...), "bottom": wavebit.PlaneWave(...)}`.
Each PlaneWave can specify its own direction, polarization and complex amplitude.
The calculation combines complex boundary fields before evaluating power and
returns `output_top` and `output_bottom`. See
[coherent illumination](../coherent-illumination.md) for examples, the calculation
flowchart, phase references and the required averaging region for different Bloch sectors.

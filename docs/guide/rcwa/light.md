# Declaring RCWA illumination

`rcwa.Excitation(wavelength=..., sources=(...))` separates vacuum wavelengths from the sources that own side, direction, polarization and amplitude. All source angles are radians. A source uses either `theta/phi` or `kx/ky` in inverse problem-length units, never both.

```python
from wavebit import rcwa
excitation = rcwa.Excitation(
    wavelength=(.85, .9, .95),
    sources=(rcwa.PlaneWave(side="top", polarization="s", amplitude=1.,
                            theta=.2, phi=0.),))
```

Plane-wave amplitude is in V/m. Polarization is `"s"`, `"p"`, or a complex `(s,p)` Jones pair normalized to unit electric magnitude; its relative phase is retained. The outgoing amplitudes are linear in that normalized pair: `amplitude_s`, `amplitude_p` and `cartesian` for `(a_s, a_p)` are `a_s/n` times those for `"s"` plus `a_p/n` times those for `"p"`, with `n = sqrt(|a_s|**2 + |a_p|**2)`, and the source `amplitude` does not change them. Two runs therefore give the amplitudes of every polarization at one direction and wavelength. Combine amplitudes, never `R`, `T` or `power`: those are squared magnitudes, and adding them drops the cross term between the two states. `OrderExcitation` names coherent incident diffraction orders. `GaussianBeam` describes an angular spectrum. `ElectricDipole(position=..., moment=...)` specifies a physical electric dipole moment in C·m. These source families have different support and normalization requirements; `check` validates the requested combination.

The [reference](../../reference/rcwa/index.md) lists their constructors. The following earlier-interface derivations remain useful for polarization, phase and complex transverse wavevectors, with compatibility calls identified explicitly.

## Earlier-interface details

:::{note}
The sections below document `wavebit.compat.legacy` and preserve their earlier coordinate, output and validation contracts. Use the current declarations above for new code.
:::

Specify the incident direction with either angles or transverse wavevector components.
Polarization, illuminated diffraction orders and incidence side are separate settings.
See [Keyword reference](../../reference/keywords.md#the-light) for defaults.

## Two ways to specify direction

| Input | Meaning | Held fixed during a wavelength sweep |
| --- | --- | --- |
| `theta`, `phi` | the direction the incident light travels, polar angle and azimuth, in radians | angles; k changes with wavelength and incident material dispersion |
| `kx`, `ky` | real Cartesian transverse angular wavenumbers | physical kx and ky; the angle may change |

`kx` and `ky` have inverse length units: nm**-1 when `wavelength`, `period` and
`thickness` use nm. They are **not divided by k0** and are not reciprocal-lattice
order numbers. An omitted component is zero. Giving either component together
with a non-`None` `theta` or `phi` raises an error, even if an angle is zero.
With neither form supplied, the default is normal incidence with azimuth zero.

For example, these specify the same wave in a lossless medium with epsilon=2.25
and mu=1 (n=1.5):

```python
import numpy as np
from wavebit.compat import legacy as wavebit

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
too, in absorbing media as well: there a real `theta` is the input, and a real
angle keeps `kx` real. A complex transverse wavevector is reachable only through a
complex `theta`/`phi`, whose `R` and `T` are formal (see
[Real-angle incidence](../../reference/common/conventions.md#real-angle-incidence)). Direct components carry no
angle, so they are taken as given in any medium, a negative-index one included. With
`kx=ky=0`, the s/p basis uses azimuth zero (s along y, p in xz); choose `polarization` to
orient the field.
For pure 1D `tetm` input, omit `ky` (it is zero); use `polarization` for conical incidence.

`wavebit.Incidence(kx=..., ky=...)`, `JobSpec` serialization, `kx`/`ky` sweeps and
JAX/PyTorch differentiation use the same inputs. To switch an existing solver
from angles to direct k, clear `theta` and `phi` with `None` before assigning k.
To switch back, clear both k components first. Changing k requires another solve
before reconstructing fields.

## What `theta` and `phi` mean

`theta` and `phi` are **the direction the incident light comes in**: the polar angle from the
inward normal of the lit half-space and the azimuth from `+x` toward `+y`, in the fixed x/y
axes. They refer to the selected `top` or `bot` half-space itself; no additional air layer is
assumed. Real angles specify uniform incident intensity along the surface.

What that fixes is the in-plane wavevector:

```text
k_par / k0 = s * N * sin(theta) * (cos(phi), sin(phi))
```

`N` is the lit medium's phase index -- `|n|` if it is lossless -- and `s` the sign of its index
`n = sqrt(eps)*sqrt(mu)`. Three cases cover every medium:

* **A lossless positive-index medium** (`s = +1`, `N = n`): the familiar
  `kx = k0*n*sin(theta)*cos(phi)`, `ky = k0*n*sin(theta)*sin(phi)`. Phase and power travel
  together, and `theta` is both directions.
* **A negative-index medium** (`s = -1`: a double negative, `Re(eps) < 0` and `Re(mu) < 0`, or
  another negative-phase-velocity medium such as a passive magnetic metal). The phase travels
  against the power, so the same light has its wavevector the other way:
  `kx = -k0*|n|*sin(theta)*cos(phi)`. Lossless, that is still `kx = k0*n*sin(theta)*cos(phi)`
  with the signed `n < 0` -- Snell's law with a negative index. Light arriving from
  `n = -sqrt(2)` at 20 degrees enters glass at -18.81 degrees, on the same side of the normal
  it came from: negative refraction.
* **An absorbing medium** of either sign: `N` comes from the full complex `eps*mu`, the normal
  wavevector is complex and the incident wave weakens on its way to the structure. `theta` and
  `phi` are then the angles of `s * Re(k)`, and the Poynting vector departs from them by an
  amount set by the loss and the polarization -- by 1.7 degrees for p light at 45.8 degrees in
  `eps = 2+0.5i`, and to the other side of the normal for p light in a metal (the table is in
  [negative-index media](../../theory/rcwa/negative-index-media.md) §7). Absorbing real-angle incidence
  requires `abs(theta) < pi/2`.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

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

Here 30 degrees is the angle **inside the absorbing incident half-space**. Lit from a
negative-index medium the same script runs unchanged; only the sign of `kx` differs:

```python
nim = wavebit.Material(-3 + 0.1j, mu=-0.5 + 0.05j)      # n_top = -1.2254+0.0816j
solver = wavebit.rcwa(
    top=nim, bot=2.25,
    wavelength=1000., period=[1200., 1200.],
    theta=np.deg2rad(30.), phi=0.,
    structure=np.full((1, 2, 2), 2.25), thickness=[100.],
    truncation_order=0,
)
solver.order_kxy()[0]           # negative: the light travels along +x, its phase along -x
```

At exactly `theta = 0` the s/p frame takes the direction `k_par` approaches as `theta -> 0+`:
`phi`, or `phi + pi` from a negative-index side, so the incident field is continuous through
normal incidence on either. The [derivation and complex-angle
convention](../../reference/common/conventions.md#real-angle-incidence) state the precise meaning of the input, and
[negative-index media](../../theory/rcwa/negative-index-media.md) everything about double negatives.

Power is referenced to the incident wave at its interface with the finite stack.
`result.power_entering` includes incident/reflected interference;
`result.absorption = result.power_entering - result.T.sum()` is the absorption
in all finite layers. It excludes absorption in either half-space.

## The polarization comes in three kinds

| | write | what it is |
| --- | --- | --- |
| **declaration** | `tetm='te'` / `'tm'` | this problem has no polarization and no azimuth degree of freedom. One-row raster structures only (a shape structure is refused, even one invariant along y; use `polarization=(1, 0)` / `(0, 1)` at `phi=0` there), and it is what buys the pure TE/TM formulation -- 2 to 4 times faster than the conical one, which gives the same amplitudes, fields and efficiencies as `polarization=(1, 0)` / `(0, 1)` at `phi=0`, to 1e-13 (since 2026-09-27; before, the amplitudes differed by a sign on the orders across `kx = 0` from the incident one) |
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

**A polarization sweep needs one solve.** Every solve at `answers='amplitude'` or above also
returns `result.s_inc` and `result.p_inc`, the responses to pure s and pure p from the same
factorization. For any pair `(a_s, a_p)`, the amplitudes `r_s`, `r_p`, `t_s`, `t_p`, `r_xyz` and
`t_xyz` equal `a_s/n` times the `s_inc` array plus `a_p/n` times the `p_inc` one, with
`n = sqrt(|a_s|**2 + |a_p|**2)` and the pair's overall phase kept. This holds on both
formulations, under ETM and SMM, from either side and under `excitation=`. Combine the
amplitudes, not `R` and `T`: an efficiency is a squared magnitude, so adding the two variants'
efficiencies drops their cross term. `s_inc` and `p_inc` are `None` under `tetm=` and at
`answers='efficiency'`.

Every order named in `excitation` has to propagate in the incidence medium. One past the light
line would drive the structure while contributing nothing to the incident flux every efficiency
is divided by, so `R + T` would exceed one on a passive stack; asking for one is refused by
name. That also caps how many beams can be combined, at roughly `2 * period / wavelength`.

An `excitation` weight multiplies *its own order's* `s_hat` or `p_hat`, on every path. On a
1D solve at `phi = 0` that is `sign(kx) * y_hat` for TE, so a lit order with `kx < 0` arrives
polarized along `-y`; see [`../conventions.md`](../../reference/common/conventions.md). A weight is its beam's
amplitude at `x = y = 0` on the lit face, the cell centre: the same weights
`OrderExcitation(orders=...)` takes, and the same light.

See [`../conventions.md`](../../reference/common/conventions.md) for the normative statement,
[`../multimode-incidence.md`](multimode-incidence.md) for `excitation`, and tutorials
[`06`](https://github.com/fieldalgo/wavebit/blob/develop/tutorials/rcwa/06-polarization-pair.ipynb) and
[`07`](https://github.com/fieldalgo/wavebit/blob/develop/tutorials/rcwa/07-multimode-incidence.ipynb).

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
[coherent illumination](coherent-illumination.md) for examples, the calculation
flowchart, phase references and the required averaging region for different Bloch sectors.

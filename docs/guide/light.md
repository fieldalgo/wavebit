# Saying what the light is

Three keywords describe the incident polarization and they are three *kinds* of thing rather
than three spellings of one. A fourth says which diffraction orders the light arrives on, and a
fifth which side of the stack it comes from. The table of all five, with defaults, is
[Keyword reference](../reference/keywords.md#the-light).

## The polarization comes in three kinds

| | write | what it is |
| --- | --- | --- |
| **declaration** | `tetm='te'` / `'tm'` | this problem has no polarization and no azimuth degree of freedom. 1D structures only, and it is what buys the pure TE/TM formulation -- 2 to 4 times faster than the conical one, which agrees with it to 1e-13 |
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

See [`../conventions.md`](../conventions.md) for the normative statement,
[`../multimode-incidence.md`](../multimode-incidence.md) for `excitation`, and tutorials
[`06`](https://github.com/fieldalgo/wavebit/blob/develop/tutorials/06-polarization-pair.ipynb) and
[`07`](https://github.com/fieldalgo/wavebit/blob/develop/tutorials/07-multimode-incidence.ipynb).

## Which side the light arrives from

`incidence_side='bottom'` illuminates through the substrate. **The scene labels never move**:
`top` is still the superstrate, `bot` still the substrate, and `structure[0]` is still the layer
adjacent to `top`. What changes is that results are incidence-relative and `theta` / `phi` are
measured in the incidence medium -- so a cross-solver comparison needs the angle converted into
that medium before the numbers line up.

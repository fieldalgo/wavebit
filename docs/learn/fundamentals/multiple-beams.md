(multiple-incident-waves)=
# 09. Multiple incident waves and internal sources

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

[Chapter 8](light.md) specified the direction and polarization of a single incident wave. This chapter calculates the fields and output power produced by several waves together. We first examine the common calculation flow for external illumination and internal sources, then explain the assumptions behind superposition and the input formats. Examples cover illumination from one side, illumination from both sides, and order-resolved inputs, followed by Gaussian beams and electric dipoles inside layers.


(coherent-illumination-flow)=
## How multiple sources are calculated

An input that produces the light being calculated is called a **source**. You can combine externally incident plane waves and Gaussian beams with electric dipoles that emit inside a layer. The following flow applies to calculations with **one common wavelength and linear materials**, with a specified magnitude and phase for each source.

External waves enter as **incident amplitudes at the top and bottom boundaries**; internal dipoles enter as **current terms at their source positions**. Both follow the same principle: add the calculated complex electric fields E and magnetic fields H separately, then calculate power from the combined fields.

```{figure} ../../_static/learning/09-source-calculation.svg
:figclass: learning-figure
:alt: External plane waves and Gaussian beams enter as boundary incident amplitudes; internal dipoles enter as current terms. RCWA is calculated for components sharing a Fourier array, and the complex fields are summed. Power is then normalized by incident power for external-only illumination; with dipoles, directional output power is read instead.

Figure 9-1. Common calculation flow for external illumination and internal sources. One-sided and two-sided illumination and dipoles are handled together, while their input locations and ways of reading output power remain distinct.
```

**An internal dipole supplies power directly, so reflection and transmission are not defined solely relative to external incident power.** When a result includes a dipole, `output_top`, `output_bottom`, and `absorption` are `None`; read directional power from `result.power`. Units and examples are explained in the {ref}`dipole section <electric-dipole-source>`.

(source-calculation-groups)=
### Which components can be calculated together?

If the difference between two plane waves' transverse wavevectors is an integer combination of reciprocal lattice vectors, they can share one diffraction-order array and one RCWA calculation. We call components that can be processed together a **calculation group**. For a 1D grating, their `kx` values differ by an integer multiple of `2π/Px`. [Chapter 6](lattice.md) explains the relation between reciprocal lattice vectors and diffraction orders.

**Waves incident from the same side are still calculated separately if their transverse wavevectors do not meet this condition.** In this chapter's {ref}`same-side example <same-side-beams>`, normal incidence and 20° incidence belong to different groups. The {ref}`example specifying orders 0 and +1 <incident-order-inputs>` forms one group. Waves entering from the top and bottom are grouped by the same transverse-wavevector condition.

A Gaussian beam consists of plane-wave components with different directions. A periodically repeated dipole array uses `bloch_k` to set the phase between neighboring cells; an isolated dipole is approximated by summing results for several Bloch wavevectors. These components are also calculated with external waves when they share a Fourier array.

Even when groups are calculated separately, **their complex fields are summed using common coordinates and phase references, so interference between sources is included**. We next establish why fields can be added, then examine each source's input format.

(linear-source-superposition)=
### Conditions for adding waves

Wavebit treats **linear materials**. At a given wavelength, permittivity and permeability do not depend on light intensity, and the relation between electric field and material response is linear. For example, electric field **E** and electric displacement **D** obey the following relation.

```{math}
\mathbf D=\varepsilon_0\boldsymbol{\varepsilon}_r\mathbf E.
```

Here ε₀ is the vacuum permittivity and **ε**<sub>r</sub> is relative permittivity. With fixed material values, the response to a sum of two electric fields equals the sum of their individual responses. **Adding the electric fields and, separately, the magnetic fields calculated for each source in the same structure gives the fields for simultaneous excitation.** Lossy materials with complex permittivity and materials with anisotropic tensors can satisfy this linearity condition. Nonlinear materials whose permittivity depends on light intensity are outside this calculation's scope.

Sources combined in one `excitation` are assumed to oscillate at the same frequency, with relative phases that remain constant in time. For example, assigning phases of 0° and +90° calculates a state that maintains that 90° difference. Sources with such a fixed phase relation are called **coherent sources**.

Set `wavelength` to the **vacuum wavelength** corresponding to this common frequency.

Fields can be added this way, but **power includes interference between the fields**. For coherent sources, therefore, calculate power after adding the fields. The later {ref}`power discussion <coherent-and-incoherent>` distinguishes how to combine powers from mutually incoherent incident waves.

(source-types)=
## Source types and input formats

Use `PlaneWave` or `GaussianBeam` for external illumination and `ElectricDipole` for an electric dipole emitting inside a layer. These source objects belong to the common `Source` type and enter `excitation` individually or as a list.

| Source | Physical quantities specified | Where it enters the calculation |
| --- | --- | --- |
| `PlaneWave` | Incident direction, polarization, complex amplitude | Top or bottom exterior boundary |
| `GaussianBeam` | Central direction, polarization, beam width, focus, complex amplitude | Decomposed into plane waves and applied at exterior boundaries |
| `ElectricDipole` | Position and complex electric dipole moment | Current term inside the containing layer |
| `OrderExcitation` | Reference direction and complex amplitude of each order | Multiple incident plane waves specified by order |

Set each plane wave's incident side with its `side` attribute. In a list, the default is `"top"`. Simultaneous illumination from the top and bottom can be written as follows.

```python
from wavebit.compat import legacy as wavebit

sources = [
    wavebit.PlaneWave(side="top", kx=0.002, ky=0., amplitude=1.),
    wavebit.PlaneWave(side="bottom", kx=0.002, ky=0., amplitude=0.5j),
]
# Pass this to rcwa(..., excitation=sources).
```

You can also group inputs by exterior boundary with `excitation={"top": [...], "bottom": [...]}`. Internal dipoles belong to neither `top` nor `bottom`, so place them in a source list. Positions use the same xyz frame as the other fields: z=0 is the bottom of the first layer and +z points upward.

(id3)=
### Choosing a plane-wave input format

Start with plane waves. Use `PlaneWave` to set each wave's angle and polarization independently; use complex amplitudes indexed by order to choose directions from the grating's incident orders.

| Desired input | Format | Example |
| --- | --- | --- |
| Independent angles and polarizations for waves from one side | `excitation={"top": [wavebit.PlaneWave(...), ...]}` | {ref}`Multiple waves from one side <same-side-beams>` |
| Simultaneous top and bottom illumination | `excitation={"top": ..., "bottom": ...}` | {ref}`Two-sided illumination <two-sided-illumination>` |
| Selected incident orders around one reference direction | `excitation={0: 1., 1: 0.5j}` | {ref}`Order-resolved inputs <incident-order-inputs>` |
| Order-resolved input included in a source list | `wavebit.OrderExcitation(orders={0: 1., 1: 0.5j}, ...)` | {ref}`Grouping incident orders <beams-detail-id31>` |

Set the incident side with `PlaneWave.side` or the `"top"`/`"bottom"` keys of the exterior-boundary dictionary. For a dictionary containing only order amplitudes, such as `excitation={0: ..., 1: ...}`, select one side with `incidence_side`. These input formats differ in phase reference and result object; the examples below explain both.

(excitation-amplitude-phase)=
## Amplitude, phase, and light intensity

**A complex amplitude represents both the magnitude and phase of the electric-field amplitude.** Supply it as `amplitude` for a `PlaneWave`, or as each order's value for order-resolved input. In code, write `magnitude * np.exp(1j * phase)`.

- The leading **magnitude** sets the electric-field amplitude magnitude.
- `np.exp(1j * phase)` has magnitude 1 and **sets only the phase**. Phase is in radians; convert degrees with `np.deg2rad()`.
- Python's `1j` is the imaginary unit. At phase +90°, `np.exp(1j * np.deg2rad(90.))` is mathematically equal to `1j`.

For example, magnitude 0.5 and phase +45° give `0.5 * np.exp(1j * np.deg2rad(45.))`. Changing only the phase angle preserves the incident wave's amplitude magnitude while changing its phase relative to other waves.

**Light intensity** is time-averaged power per unit area, measured in W/m². Electric-field amplitude describes the oscillation magnitude of the electric field and is measured in V/m. For plane waves in the same medium with the same direction and polarization, intensity is proportional to the squared amplitude magnitude. Doubling amplitude therefore quadruples intensity; halving amplitude reduces intensity to 1/4. When comparing power crossing a boundary for different directions or media, also account for incidence angle and medium properties.

For external illumination alone, amplitudes can be specified relative to 1 and output-power ratios read directly. For absolute power or superposition with a dipole, give electric-field amplitudes physical magnitudes in V/m.

(same-side-beams)=
## Several incident waves from one side

To illuminate from above with several waves, supply **a list of incident waves for one side**, such as `excitation={"top": [PlaneWave(...), PlaneWave(...)]}`. Set each `PlaneWave`'s direction, polarization, magnitude, and phase independently. All waves illuminate the same structure and share the `wavelength` passed to `rcwa()`.

Specifying only `"top"` gives illumination from above only. For several waves incident solely from below, use the key `"bottom"`. In either case, `top` and `bot` still specify both exterior media.

(beams-detail-id30)=
### Setting each incident wave's parameters

| Setting | Meaning |
| --- | --- |
| `theta`, `phi` | This wave's incidence and azimuthal angles, in radians, measured in its incident medium. |
| `kx`, `ky` | Transverse wavevectors specified directly instead of angles. With lengths in nm, their unit is nm⁻¹. Do not combine them with `theta`, `phi` in the same `PlaneWave`. |
| `polarization=(a_s, a_p)` | Relative s/p electric-field components and their phase difference. `(1., 0.)` is s and `(0., 1.)` is p polarization. The components are normalized so their squared magnitudes sum to 1. |
| `psi`, `chi` | Polarization-ellipse angles can replace `polarization`, using the convention in {ref}`Chapter 8's polarization discussion <incident-polarization>`. |
| `amplitude` | Magnitude and phase multiplying the normalized polarization coefficients. In `A * np.exp(1j * np.deg2rad(delta))`, `A` is magnitude and `delta` is phase angle (°). The default is `1.`. |

`polarization` sets the relation between a wave's s/p components; `amplitude` sets the magnitude of the entire wave and its phase relative to other waves.

A wave with no direction specified uses common `kx`, `ky` values from `rcwa()` if present; otherwise it is normally incident. Polarization also follows common settings, defaulting to s if none are provided. The following example specifies each wave explicitly.

(beams-detail-id29)=
### Example: combining normal and 20° incidence

Two s-polarized waves enter an absorbing slab from air. The first is normally incident; the second tilts 20° toward +x. The second wave has half the first's electric-field amplitude magnitude and a relative phase of +90° at the reference point.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

beams = [
    wavebit.PlaneWave(
        theta=0., phi=0.,
        polarization=(1., 0.),          # s polarization
        amplitude=1.,                  # Amplitude magnitude 1, phase 0°
    ),
    wavebit.PlaneWave(
        theta=np.deg2rad(20.), phi=0.,
        polarization=(1., 0.),          # s polarization
        amplitude=0.5 * np.exp(1j * np.deg2rad(90.)),
    ),
]

same_side = wavebit.rcwa(
    wavelength=600.,                    # Length unit: nm
    period=(900., 850.),
    top=1., bot=2.25,
    structure=[dict(background=4.+0.6j, shapes=[])],
    thickness=[100.],
    truncation_order=0,                 # Uniform, unpatterned slab
    excitation={"top": beams},
)
same_side_result = same_side.solve(power_region="infinite")

print("Reflectance:", same_side_result.output_top)
print("Transmittance:", same_side_result.output_bottom)
print("Absorption:", same_side_result.absorption)
```

This example gives reflection approximately `0.154612`, transmission `0.624306`, and absorption `0.221082`; their sum is 1.

| Input | Incident direction | Polarization | Amplitude magnitude | Phase at reference point |
| --- | --- | --- | --- | --- |
| `beams[0]` | Vertically downward | s | 1 | 0° |
| `beams[1]` | Downward, tilted 20° toward +x | s | 0.5 | +90° |

The numbers in `beams[0]`, `beams[1]` **identify incident waves in a list**. They do not mean diffraction orders 0 and 1. Add more `PlaneWave` objects to include three or more waves.

Because illumination comes only from above, `output_top` is total reflection and `output_bottom` is total transmission. Both are normalized by the power of the two incident waves together, excluding output waves. With illumination only from below, `output_bottom` is reflection and `output_top` is transmission.

Both waves from above specify phase at **the cell center of the top boundary**, `(x, y) = (0, 0)`. The example's +90° is therefore the second wave's phase relative to the first at that point. Their directions differ, so their relative phase differs elsewhere on the surface. They remain coherent: the phase difference at any fixed position stays constant in time.

Amplitude magnitude 0.5 does not mean incident-power ratio 0.5. For electric-field amplitudes in the same lossless isotropic medium, as here, the power entering normally through the boundary is proportional to `amplitude magnitude² × cos(theta)`. The second wave's incident power is `0.5² × cos(20°) ≈ 0.235` times the first's.

The example explicitly sets the power-averaging region with `power_region` because the waves have different directions. The {ref}`calculation flow <coherent-illumination-flow>` explains which waves are processed together. With `power_region="infinite"`, interference between distinct transverse wavevectors vanishes under spatial averaging, although it remains in position-resolved fields. The {ref}`power-reading discussion <illumination-power-region>` explains the different averaging regions.

(two-sided-illumination)=
## Simultaneous illumination from above and below

For illumination from both sides, put `"top"` and `"bottom"` in `excitation`, each with a `wavebit.PlaneWave(...)`. As in the same-side example, specify direction, polarization, magnitude, and phase separately for each wave.

`top` and `bot` are the material values of the upper and lower half-spaces. Since `excitation` keys specify the illumination side, do not also set `incidence_side` in this format. Layers are entered from bottom to top.

(beams-detail-id19)=
### Example: a 90° phase difference between the two boundaries

The example below places an absorbing slab between air above and glass below. Both incident waves have the same `kx`, `ky`; the lower wave receives phase +90°.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

two_sided = wavebit.rcwa(
    wavelength=600.,                    # Length unit: nm
    period=(900., 850.),
    top=1., bot=2.25,                   # Top and bottom relative permittivities
    structure=[dict(background=4.+0.6j, shapes=[])],
    thickness=[100.],
    truncation_order=0,                 # Uniform, unpatterned slab example
    kx=0.002, ky=0.,                    # Common transverse wavevector, in nm⁻¹
    answers="field",
    excitation={
        "top": wavebit.PlaneWave(polarization=(1., 0.)),
        "bottom": wavebit.PlaneWave(
            polarization=(1., 0.),
            amplitude=1.0 * np.exp(1j * np.deg2rad(90.)),
        ),
    },
)
two_sided_result = two_sided.solve()

print("Power fraction leaving upward:", two_sided_result.output_top)
print("Power fraction leaving downward:", two_sided_result.output_bottom)
print("Power fraction absorbed in the slab:", two_sided_result.absorption)
```

`amplitude` is a complex number multiplying the normalized s/p coefficients. Here `1.0` is magnitude and `90.` is phase angle. **The top incident wave specifies amplitude and phase at the top boundary's cell center, and the bottom incident wave at the bottom boundary's cell center.** The example's 90° is the relative phase between those two reference points.

Different top and bottom media give different incidence angles even at the same `kx`, `ky`. Here the upper angle is about 11.01° and the lower angle about 7.31°. Although both amplitude magnitudes are 1, their incident powers through the boundaries differ because the media and angles differ.

Changing only the lower incident wave's phase gives the following results.

| Lower incident-wave phase | Upper output fraction | Lower output fraction | Absorbed fraction |
| --- | --- | --- | --- |
| 0° | 0.481181 | 0.242506 | 0.276313 |
| 90° | 0.739880 | 0.059363 | 0.200757 |
| 180° | 0.393406 | 0.411418 | 0.195176 |

The structure and each wave's incident power are unchanged, but interference changes the outputs and fields inside the slab. All ratios in this table divide by **the sum of the incident-wave-only powers entering from both sides**.

(beams-detail-id20)=
### Specifying the upper and lower angles separately

Set `theta`, `phi` directly in each `PlaneWave`. Angles are measured in that wave's incident medium. Alternatively, specify `kx`, `ky` for each wave. Do not mix the two formats within one wave.

```python
two_sided_result = two_sided.solve(
    excitation={
        "top": wavebit.PlaneWave(theta=np.deg2rad(20.), phi=0.),
        "bottom": wavebit.PlaneWave(
            theta=np.deg2rad(35.), phi=0.,
            amplitude=0.7 * np.exp(1j * np.deg2rad(90.)),
        ),
    },
    power_region="infinite",
)
print(two_sided_result.group_count)  # 2 in this example
```

With independently chosen angles, the waves may belong to different calculation groups, so specify `power_region`. The grouping rule and power averaging are explained in the {ref}`calculation flow <coherent-illumination-flow>`.

(incident-order-inputs)=
## Specifying several incident waves by order

`excitation={0: 1., 1: 0.5j}` selects waves incident from one side by diffraction order. The reference direction and lattice period determine each order's direction; all waves share one polarization setting.

(beams-detail-id13)=
### How order labels determine incident direction

First define a reference incident wave with `theta`, `phi` or `kx`, `ky`. This is the **zeroth-order incident wave**. For a 1D grating of x-period P<sub>x</sub>, integers m distinguish waves whose x wavevectors differ by `2π/Px`.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>k<sub>x,m</sub> = k<sub>x,0</sub> + m · 2π/P<sub>x</sub></p>
</div>
```

**m=0 is the reference direction; m=+1 has an x wavevector one interval larger.** The `0`, `1` in `excitation` are these order labels. This chooses waves on one periodic lattice's wavevector spacing, rather than assigning an arbitrary angle to each numbered wave. [Chapter 6](lattice.md) explains the connection between wavevector spacing and lattice.

(beams-detail-id15)=
### Example: combining a normal and an oblique incident wave

The example below illuminates a grating of period 1500 nm from air at wavelength 600 nm. Because `theta=0`, order 0 is normally incident. Order +1 has `kx/k0 = 600/1500 = 0.4`, giving angle `arcsin(0.4)`, approximately 23.58°. Both waves travel downward; order +1 also has a +x component.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

amplitude_1 = 0.5                 # Electric-field amplitude magnitude of order +1
phase_1 = np.deg2rad(90.)          # Phase of order +1 relative to order 0

order_solver = wavebit.rcwa(
    backend='numpy',
    top=1., bot=2.25,
    incidence_side="top",
    wavelength=600., period=[1500., 1500.],  # Length unit: nm
    theta=0., phi=0.,                      # Direction of the zeroth-order incident wave
    structure=np.array([[[1., 1., 4., 4.]]]),
    thickness=[200.],
    truncation_order=dict(x=4, y=0),
    polarization=(1., 0.),                 # Both incident waves are s polarized
    excitation={
        0: 1.,                            # Magnitude 1, phase 0°
        1: amplitude_1 * np.exp(1j * phase_1),  # Magnitude 0.5, phase +90°
    },
)
order_result = order_solver.solve()
print("Reflectance for both incident waves together:", order_result.R.sum())
print("Transmittance for both incident waves together:", order_result.T.sum())
```

The two incident waves in the preceding code have the following magnitudes and phases.

| Order label | Incident direction in this example | Amplitude magnitude | Phase relative to order 0 |
| --- | --- | --- | --- |
| `0` | Normal incidence | 1 | 0° |
| `1` | Tilted approximately 23.58° toward +x | 0.5 | +90° |

The +1-order value `amplitude_1 * np.exp(1j * phase_1)` equals `0.5j` here: **one complex number representing magnitude 0.5 and phase +90°**. Since `j` has magnitude 1, `0.5j` has magnitude 0.5. Amplitude magnitude is not a power fraction. Incident power crossing the surface depends on each wave's direction and medium as well as its squared amplitude magnitude. `R.sum()` and `T.sum()` are normalized by **the total incident power of both waves**.

For this order-amplitude dictionary, the phase reference is the beginning of the Fourier cell at the incident boundary: `(x, y)=(-Px/2, -Py/2)` for a rectangular cell, and `−(a₁+a₂)/2` for an oblique cell. The `PlaneWave` format instead references amplitudes to the cell center. Because the waves have different `kx`, moving along the surface changes their relative phase and may change the combined incident intensity. Each single incident wave in [Chapter 8](light.md) has constant intensity along the surface. The intensity of multiple waves includes their interference.

To keep magnitude 0.5 but set **relative phase +45°**, use the following input. Recalculate with the incident directions and materials already specified in `order_solver`.

```python
order_solver.excitation = {
    0: 1.,
    1: 0.5 * np.exp(1j * np.deg2rad(45.)),  # Magnitude 0.5, phase +45°
}
result_45 = order_solver.solve()
```

The +1-order complex amplitude is then approximately `0.353553 + 0.353553j`. Although both real and imaginary parts change, its magnitude remains 0.5 and its phase relative to order 0 is +45°.

(beams-detail-id17)=
### Scope of a single dictionary of order amplitudes

- **Order-amplitude inputs such as `excitation={0: 1., 1: 0.5j}` share wavelength, incident side, and polarization settings.** `excitation` sets each wave's order and amplitude; `polarization` or `psi`, `chi` applies to every wave. Each wave's s/p directions are defined relative to its own propagation direction.
- **Specified incident orders must lie inside the retained order window.** For example, order +1 requires an x range of at least 1 in `truncation_order`. [Chapter 10](fourier-orders.md) discusses selecting orders for accuracy.
- **Each order must carry power toward the structure in the incident medium.** The allowed orders depend on wavelength, period, and reference direction. An evanescent nonpropagating order or an order with zero incident power is rejected.

(excitation-order-pairs)=
(d-m-n)=
### 2D incident orders: the order of `(m, n)`

**For a rectangular lattice aligned with x and y, `(m, n)` means `(x order, y order)`.** The first integer `m` is the x order and the second `n` is the y order. They label orders; they are not the wavevector values `kx`, `ky` themselves.

| `excitation` key | x order `m` | y order `n` | Addition `(Δkx, Δky)` to the reference transverse wavevector |
| --- | --- | --- | --- |
| `(0, 0)` | 0 | 0 | `(0, 0)` — reference incident wave |
| `(1, 0)` | +1 | 0 | `(2π/Px, 0)` |
| `(0, 1)` | 0 | +1 | `(0, 2π/Py)` |
| `(-1, 1)` | −1 | +1 | `(−2π/Px, 2π/Py)` |

Here `Px`, `Py` are the x and y periods. For example, `(1, 0)` adds `2π/Px` to the reference `kx` and leaves `ky` unchanged. `(0, 0)` follows the reference direction specified by `theta`, `phi` or `kx`, `ky`, so both transverse components vanish only at normal incidence.

The following specifies the reference wave together with x order +1. Magnitudes and phases are entered as in 1D.

```python
excitation = {
    (0, 0): 1.,  # Reference incident wave: magnitude 1, phase 0°
    (1, 0): 0.5 * np.exp(1j * np.deg2rad(90.)),  # x order +1, y order 0
}
```

**For an oblique lattice, `m`, `n` multiply the first and second reciprocal lattice vectors.** If the reciprocal lattice vectors from [Chapter 6](lattice.md) are **b**<sub>1</sub>, **b**<sub>2</sub>, each incident order's transverse wavevector is as follows.

```{raw} html
<div class="power-equations" style="text-align:center">
<p><b>k</b><sub>∥,(m,n)</sub> = <b>k</b><sub>∥,(0,0)</sub> + m <b>b</b><sub>1</sub> + n <b>b</b><sub>2</sub></p>
</div>
```

Thus `(1, 0)` adds **b**<sub>1</sub> to the reference transverse wavevector; `(0, 1)` adds **b**<sub>2</sub>. In an oblique lattice, reciprocal vectors need not align with x and y, so changing `m` alone may change both `kx` and `ky`. For a rectangular lattice, **b**<sub>1</sub> = `(2π/Px, 0)` and **b**<sub>2</sub> = `(0, 2π/Py)`, giving the x/y order interpretation in the table above.

The [detailed multiple-incidence guide](../../guide/rcwa/multimode-incidence.md) covers inputs and supported calculation ranges; {ref}`Chapter 13 <half-space-power>` explains power normalization.

(beams-detail-id31)=
### Grouping incident orders

`OrderExcitation` groups **several incident plane waves separated by reciprocal lattice vectors** around one reference direction. Specify each order's complex amplitude in `orders`; all orders share s/p polarization coefficients. A single `PlaneWave` specifies one plane wave.

```python
beam_order_result = same_side.solve(
    truncation_order=dict(x=1, y=0),    # Include the incident order +1
    excitation={
        "top": wavebit.OrderExcitation(
            theta=0., phi=0.,
            polarization=(1., 0.),
            amplitude=1.,
            orders={
                0: 1.,
                1: 0.5 * np.exp(1j * np.deg2rad(90.)),
            },
        ),
    },
    power_region="cell",
)
```

The `0`, `1` in `orders` are **x incident orders**. This example has `ky=0`, and each `kx` adds `m × 2π/Px` to the reference value. In air at wavelength 600 nm and period 900 nm, order 0 is normally incident and order +1 tilts approximately 41.81° toward +x. This differs from the 20° wave in the {ref}`same-side example <same-side-beams>`. In 2D, use `(m, n)`; the two integers multiply the first and second reciprocal lattice vectors respectively.

Each order's complex coefficient is `OrderExcitation.amplitude × orders[order]`, further multiplied by the shared s/p polarization coefficients. For different polarizations per order, use separate `PlaneWave` objects. Input orders must be inside `truncation_order` and able to carry power toward the structure in the incident medium.

Within a calculation group, the array is expanded to include the diffraction orders surrounding each input. Widely separated incident orders in one group therefore increase array size and computation.

(illumination-output-power)=
## Calculating output power from combined fields

Read `PlaneWave` and `OrderExcitation` results as `output_top` or `output_bottom` according to the outgoing side. With illumination only from above, these are reflection and transmission; with illumination only from below, they are transmission and reflection.

The upward outgoing electric field combines **reflection of the upper incident wave and transmission of the lower incident wave**. The lower output similarly combines contributions from both inputs. For two-sided illumination, therefore, results are named `output_top`, `output_bottom` by outgoing side rather than one `R`, `T` pair.

```{math}
\mathbf E=\sum_j\mathbf E_j,\qquad
\mathbf H=\sum_j\mathbf H_j,\qquad
\langle\mathbf S\rangle=\frac12\operatorname{Re}(\mathbf E\times\mathbf H^*).
```

Here j labels each calculation's contribution and is unrelated to Python's imaginary unit `1j`. **S** is the Poynting vector, or power-flow density; ⟨ ⟩ denotes time averaging and the asterisk complex conjugation. **H** in the equation is the physical magnetic field. If calculating with the Z₀H returned by Wavebit, divide the result by vacuum impedance Z₀.

Adding electric fields and magnetic fields separately, then calculating power from the two total fields, includes interference terms. Integrating the normal component of power-flow density over a boundary gives the power through it. **Only exterior-boundary fields are needed for output power.** Fields throughout every layer need not be reconstructed first.

(coherent-and-incoherent)=
### Power from mutually incoherent incident waves

The examples above describe coherent illumination with specified relative phases. **Even when combining separately calculated results, coherent illumination requires adding complex electric and magnetic fields first.** Adding reflection and transmission efficiencies alone cannot capture interference between the waves.

**For mutually incoherent waves, relative phase varies irregularly, and cross-wave interference vanishes when averaged over the observation time.** Calculate each wave separately, then add their reflected powers and their transmitted powers. Multiple reflections and interference experienced by each wave inside the structure are already included in its RCWA calculation. Only interference between mutually incoherent incident waves disappears under time averaging.

Distinguish power from efficiency here.

- **Reflected/transmitted power:** add the powers produced by each incident wave.
- **Reflection/transmission efficiency:** divide the summed output power by total incident power. This is equivalent to weighting each efficiency by its incident-power fraction.

Let the waves' incident powers be P<sub>1</sub>, P<sub>2</sub>, their separately calculated total reflections R<sub>1</sub>, R<sub>2</sub>, and total transmissions T<sub>1</sub>, T<sub>2</sub>. Then the following applies. For one incident wave calculated with ordinary `rcwa()` input, read total reflection and transmission as `result.R.sum()`, `result.T.sum()`.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>R = (P<sub>1</sub> R<sub>1</sub> + P<sub>2</sub> R<sub>2</sub>) / (P<sub>1</sub> + P<sub>2</sub>)</p>
<p>T = (P<sub>1</sub> T<sub>1</sub> + P<sub>2</sub> T<sub>2</sub>) / (P<sub>1</sub> + P<sub>2</sub>)</p>
</div>
```

The following example takes two mutually incoherent waves with separate reflection efficiencies of 20% and 60% and computes their weighted average.

| Incident wave | Incident power | Separately calculated reflection | Reflected power |
| --- | --- | --- | --- |
| 1 | 3 W | 20% | 3 W × 0.2 = 0.6 W |
| 2 | 1 W | 60% | 1 W × 0.6 = 0.6 W |

Total incident power is 4 W and reflected power is 1.2 W, giving combined reflection 1.2 / 4 = 30%. With equal incident powers, this reduces to the simple mean of the two reflection efficiencies. Transmission is calculated in the same way.

Specifying complex amplitudes in `excitation` calculates a fixed phase difference. To model the incoherent case above, calculate each wave's RCWA result separately and add powers afterward.

(illumination-power-region)=
(beams-detail-id22)=
### Over what area is power averaged?

Combining waves from different calculation groups may produce an intensity pattern that does not repeat identically in every structure cell. For two or more groups, therefore, specify the power-averaging region with `power_region`.

| Setting | Averaging region |
| --- | --- |
| `power_region="cell"` | One lattice cell about the cell center. This is the default for one calculation group. |
| `power_region="infinite"` | Infinite-area average over the surface. Interference between distinct transverse wavevectors vanishes under this spatial average. |
| `power_region=((-400., 400.), (-300., 300.))` | Rectangle from −400 to 400 in x and −300 to 300 in y, in the structure's length unit. |

Interference vanishes under the infinite-area average because of **spatial averaging**. The waves have not become incoherent; position-resolved fields still interfere. A finite rectangle includes the interference over that region in its power.

When both exterior media are lossless and intensity repeats cell by cell, or when using an infinite-area average, `absorption = 1 - output_top - output_bottom`. In lossy exterior media, interference between incident and outgoing waves can also contribute to net power, so read `result.absorption`. {ref}`Chapter 13 <half-space-power>` explains this power reference.

A finite observation region can also exchange power through its lateral faces. A rectangular average or a multiple-group single-cell average therefore does not call the top/bottom power difference total absorption; `absorption` is `None`. You can still read `result.power.net_into_stack`: total +z flux at the lower boundary minus total +z flux at the upper boundary. Directional powers over a finite region may be negative because of interference, so their ratios are not always efficiencies between 0 and 1.

To retain the fields from the {ref}`same-side two-wave example <same-side-beams>` while changing only the observation region, read them as follows. The denominator is also the incident-wave-only mean power flow through that rectangle. Changing the observation region after solving requires `answers="amplitude"` (the default) or `"field"`. With `"efficiency"`, only the originally requested region's power is kept in `result.power`.

```python
window_power = same_side_result.boundary_power(
    ((-450., 450.), (-425., 425.)),     # x range, y range (nm)
)
print(window_power.output_top, window_power.output_bottom)
```

(source-expansions)=
## Extending to Gaussian beams and internal sources

The preceding examples specify incident plane waves directly. `GaussianBeam` constructs a beam from plane waves using its width and focus; `ElectricDipole` emits through an internal current source. They enter at different locations but follow the same principle of summing complex electric and magnetic fields before calculating power.

(gaussian-beam-source)=
### Adding a Gaussian beam with finite width

A Gaussian beam combines plane waves in several directions to produce an incident field strongest at the center and weaker laterally. `GaussianBeam` assigns Gaussian weights to plane waves around its central direction and makes each wave's electric field perpendicular to that wave's propagation direction. After calculating the components, it adds their complex E and H fields.

| Setting | Meaning |
| --- | --- |
| `side`, direction, polarization | Same conventions as `PlaneWave`; direction specifies the beam's central direction. |
| `waist` | Radius where the scalar Gaussian envelope's amplitude at the focus reaches 1/e of its central value, in the structure's length unit. |
| `focus=(x,y,z)` | Focus position in global structure coordinates. |
| `amplitude` | Magnitude and common phase of the complex electric-field vector at the focus in the absence of the structure. `1.` means magnitude 1. |
| `sampling=(Nx,Ny)` | Number of samples along the two wavevector directions perpendicular to the central direction. Default `(9,9)`. |
| `spectral_extent` | Includes each transverse wavevector direction through `±spectral_extent/waist`. Default `4.`. |

Plane-wave weights are proportional to `exp(-waist**2 * (ku**2 + kv**2) / 4)`, where `ku`, `kv` are perpendicular to the central beam direction. The source uses only components that propagate in the incident medium and point toward the structure. Lossless isotropic incident media with positive permittivity and permeability are supported. Very narrow or strongly tilted beams may differ from a scalar Gaussian shape because of component selection and electric-field transversality.

```python
from wavebit.compat import legacy as wavebit

gaussian_solver = wavebit.rcwa(
    wavelength=600., period=(900., 850.),
    top=1., bot=2.25,
    structure=[dict(background=4., shapes=[])],
    thickness=[100.], truncation_order=0,
    excitation=[wavebit.GaussianBeam(
        side="top", theta=0., phi=0., polarization=(1., 0.),
        waist=1500., focus=(0., 0., 100.), sampling=(9, 9),
    )],
)
gaussian_result = gaussian_solver.solve(
    power_region=((-3000., 3000.), (-3000., 3000.)),
)
print(gaussian_result.output_top, gaussian_result.output_bottom)
```

Specify a finite rectangular `power_region` for `GaussianBeam`. The reported ratio divides output power collected within that rectangle by incident power in the same rectangle. To determine reflection or transmission of the whole beam, including light outside the observation region, enlarge the region and check convergence. Also check `sampling` and `spectral_extent`. Finite wavevector spacing approximates a spatially repeated beam, so increase the sample count to reduce neighboring-copy effects.

**Design reference: FMMAX**

Wavebit's Gaussian-beam calculation was designed with reference to FMMAX's [Brillouin-zone integration discussion](https://github.com/facebookresearch/fmmax/blob/main/README.md#brillouin-zone-integration) and [FMMAX Gaussian-beam example](https://github.com/facebookresearch/fmmax/blob/main/examples/crystal.py). The central idea is **representing a finite-width beam by calculating several wavevector components and then summing their complex fields**.

(electric-dipole-source)=
### Placing an electric dipole inside a layer

Use `ElectricDipole` for an electric dipole emitting from one point inside a layer. The user prescribes its moment's magnitude, direction, and phase; it does not represent a particle whose moment is determined automatically by material response. With time dependence `exp(-iωt)`, its current is `J = -iω p δ(r-r₀)`, where `p` is `moment`.

| Setting | Meaning |
| --- | --- |
| `position=(x,y,z)` | Position in global structure coordinates; it must lie inside a layer. |
| `moment=(px,py,pz)` | Complex xyz electric dipole moment, in C·m. Each component includes magnitude and phase. |
| `repetition="periodic"` | Dipole array repeated in every lattice cell. |
| `bloch_k=(kx,ky)` | Sets phase between neighboring cells of a repeated array. A lattice translation R multiplies the moment by `exp(i k∥·R)`. Omission means equal phase. |
| `repetition="isolated"` | Approximates an isolated dipole by integration over several Bloch wavevectors. |
| `sampling=(Nx,Ny)` | Bloch-wavevector integration grid for an isolated source. Default `(5,5)`; check convergence. |
| `rcwa(length_unit=...)` | Metres per structure-length unit: `1e-9` for nm, `1e-6` for μm. |

Specify the source-containing layer with `shapes=[]` or a raster containing the same material in every spatial cell. Other layers may be patterned. Uniform anisotropic and magnetic layers are also allowed. The solver splits the layer at source height and inserts a current term; users need not split it themselves. Sources on layer interfaces and field sampling at exactly source height are rejected: interface sources and the singular point-source field need separate definitions.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

dipole_solver = wavebit.rcwa(
    wavelength=600., period=(900., 850.),
    top=1., bot=2.25,
    structure=[dict(background=1., shapes=[])],
    thickness=[300.], truncation_order=dict(x=1, y=1),
    length_unit=1e-9, answers="field",
    excitation=[wavebit.ElectricDipole(
        position=(0., 0., 150.),
        moment=(0., 1e-29, 0.),
        repetition="periodic", bloch_k=(0., 0.),
    )],
)
dipole_result = dipole_solver.solve(power_region="cell")
Z0 = 376.730313668
print("Mean upward power density (W/m²):", dipole_result.power.outgoing_top / Z0)
print("Mean downward power density (W/m²):", dipole_result.power.outgoing_bottom / Z0)
dipole_field = dipole_result.reconstruct_field(
    sample_x=np.linspace(-400., 400., 41), sample_y=[0.],
    sample_z=[50., 100., 200., 250.],
)
```

This example is an array of y-oriented dipoles oscillating in phase in every cell. To approximate a single isolated source, set `repetition="isolated"` and omit `bloch_k`. Specify a finite observation rectangle for `power_region`, then increase the integration grid and Fourier orders to check convergence. A finite integration grid corresponds to sources repeated in a larger periodic cell, so assess the isolated-source approximation over a region smaller than that repetition spacing.

For dipole input, E fields are in V/m and the last three components are Z₀H. `result.power.outgoing_top` and `outgoing_bottom` are mean power densities before division by Z₀. To obtain whole-cell output power in W, divide by Z₀ and multiply by cell area in m². For a rectangular observation region, use that rectangle's area instead.

With an internal source, total output cannot be normalized solely by external incident power, so `output_top`, `output_bottom`, and `absorption` are `None`. Directional powers and `net_into_stack` remain available in `result.power`. Boundary net power then includes both absorption and source-supplied power and must not be interpreted as absorptance. This API does not return supplied source power, self-field normalization, or Purcell factor.

Putting `PlaneWave` and `ElectricDipole` in one source list adds their complex fields. Specify the plane wave's `amplitude` in V/m as well so the units agree.

(beam-input-support)=
(planewave)=
## Supported scope and field access

Passing `PlaneWave`, `OrderExcitation`, `GaussianBeam`, or `ElectricDipole` source objects through `excitation` uses **the first-order formulation (`first_order`, formerly 4ff) and SMM**. Ordinary NumPy, JAX, and PyTorch execution is supported; `jit`, `vmap`, automatic differentiation, and `JobSpec` serialization are outside this interface's supported scope. Under `torch.compile` it runs eagerly inside the compiled caller. The upper and lower half-spaces must be isotropic. Gaussian-beam incident media and dipole-containing layers must meet the conditions described in their respective sections.

This source API does not separately define solution selection for amplifying media. If a retained order reaches exact grazing incidence and makes the matrix singular, the wavelength is not changed automatically; check convergence at nearby wavelengths or incidence conditions. The [detailed API and calculation definitions](../../guide/rcwa/coherent-illumination.md) describe inputs and result objects further.

(id20)=
### Reading fields on boundaries and inside layers

This chapter specified sources and read output power. To separate incident and outgoing boundary fields or calculate fields inside layers, continue to {ref}`Chapter 14's PlaneWave field example <illumination-fields>`. That chapter also explains complex-field components, coordinate inputs, and magnetic-field units. {ref}`Chapter 13 <half-space-power>` defines power flow and diffraction efficiency.

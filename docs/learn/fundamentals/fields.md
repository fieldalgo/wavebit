# 14. Electric and magnetic fields at specified positions

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

Reflectance and transmittance describe fractions of the power carried by light. To see the electric and magnetic field distributions inside a layer, read the **fields** at the desired positions after solving. Reading a field means obtaining its electric and magnetic components at that position.

For ordinary incidence, multiple plane waves, Gaussian beams, and dipole sources alike, fields are read from the **result object** returned by `solve()`. This chapter covers preparation, sampling coordinates, returned arrays, and boundary fields, then explains complex amplitudes and magnetic-field units.

**Current example versus earlier conventions:** the runnable file uses `Outputs(fields=True)` and `result.field(x=..., y=..., z=...)`. Its arrays include a wavelength axis and H is already A/m. The remaining explanations of `answers`, `reconstruct_field`, Z₀H arrays and conversion below describe the earlier interface. Do not apply that conversion twice to current results; see the [current field guide](../../guide/rcwa/fields.md).

(field-preparation)=
## Reading fields from a solved result

Use `answers="field"` when you need fields inside the layers.

```python
result = solver.solve(answers="field")
cell = result.reconstruct_field(sample_x=41, sample_y=21, sample_z=20)
```

`result` retains the wavevectors, thicknesses, and per-layer state used for this solve. Changing only the sampling positions does not require another solve. You can repeatedly read fields at different positions from the same result. Even if you change the settings of `solver` or solve again, the earlier `result` continues to return fields for its original conditions.

The information available after a solve depends on `answers` as follows. These rules apply regardless of how the incident waves and sources are specified.

| `answers` setting | Power results | Boundary fields | Interior fields |
| --- | --- | --- | --- |
| `"efficiency"` | Retained | — | — |
| `"amplitude"` — default | Retained | Retained | — |
| `"field"` | Retained | Retained | Retained |

Read boundary electric and magnetic fields with `result.boundary_fields()`; the solve must use `answers="amplitude"` or `"field"`. Reading fields inside layers with `result.reconstruct_field()` requires `answers="field"`.

With `answers="efficiency"`, the result does not retain the complex fields used to calculate power. If you subsequently need fields, call `solve()` again with the required `answers` setting.

For a result with specified sources, recalculating power over a different observation region with `result.boundary_power(region)` also requires at least `"amplitude"`. The power already calculated for the original region is available as `result.power` with every setting.

(ordinary-field-sampling)=
## Fields inside layers

### Runnable example

This example illuminates a uniform 120 nm layer with TE-polarized light. After calling the current `rcwa.run()`, it reads the fields at 25 heights from the bottom to the top of the layer. The length unit in this example is nm.

Example file: {download}`03_fields.py <../../../examples/rcwa/learning/learn/03_fields.py>`

```{literalinclude} ../../../examples/rcwa/learning/learn/03_fields.py
:language: python
:start-at: import numpy as np
:end-at: Ey = np.asarray(field.values[0, :, 0, 0, 1])
```

`field.values` has shape `(1, 25, 1, 1, 6)`, with wavelength first. `field.values[0, :, 0, 0, 1]` contains Ey at the 25 heights. The current API returns E in V/m and H in A/m; do not divide its H by Z₀ again. Because the layer is uniform and the incidence is normal, this field does not vary with x or y.

```{figure} ../../figures/learn/learn/12-field-data.svg
:alt: Real part, imaginary part, and squared magnitude of Ey versus height inside a uniform 120 nm layer
:figclass: learning-figure

Figure 14-1. The electric-field y component from the example, showing the real part, imaginary part, and squared magnitude of its complex amplitude.
```

(field-sample-coordinates)=
### Specifying sampling positions

For `sample_x`, `sample_y`, and `sample_z`, supply either an integer **number of points** or a one-dimensional array of **coordinates**. In `reconstruct_field()`, each omitted argument defaults to `20`.

| Input | Meaning | Example |
| --- | --- | --- |
| `sample_x` | Number of points along the cell x direction, or an array of x coordinates | `41`, `[-100., 0., 100.]` |
| `sample_y` | Number of points along the cell y direction, or an array of y coordinates | `21`, `[0.]` |
| `sample_z` | Number of points per layer, or an array of z coordinates for the entire stack | `20`, `[0., 60., 120.]` |

To sample just three heights at the cell center in the example above, write:

```python
three_heights = result.reconstruct_field(
    sample_x=[0.], sample_y=[0.], sample_z=[0., 60., 120.],
)
print(three_heights.shape)  # (3, 1, 1, 6)
```

`sample_z=20` places **20 points in each layer**. Two layers therefore produce 40 samples along z. In contrast, `sample_z=[20.]` reads one sample at a height of 20 nm.

- The x and y origins are at the cell center. For a rectangular cell, a point count produces uniform samples including both endpoints in each direction. A count of 1 selects the negative endpoint, so use `[0.]` for a single point at the center.
- The z origin is at the bottom of the lowest layer, and z increases upward. Coordinate arrays must lie in `0 ≤ z ≤ sum(thickness)`.
- Coordinates use the same length unit as `wavelength`, `period`, and `thickness`. If these values are given in nm, as in the example, the coordinates are also in nm.

When x and y are coordinate arrays, every x–y combination is sampled at every z height. The order of the supplied coordinates is preserved. Even a pure 1D field with no y dependence is returned at all requested y positions.

:::{dropdown} Coordinates at layer boundaries and in oblique lattices

**Layer boundaries:** A z coordinate exactly on an internal interface reads the value in the upper layer. At `z=0` and `z=sum(thickness)`, it reads the interior value of the bottommost and topmost layer, respectively.

When `sample_z` is a point count, each layer is sampled separately. A count of at least 2 includes both ends of each layer, so an internal interface appears once from the lower layer and once from the upper layer. The normal components `Ez` and `Hz` can differ across a material interface. `sample_z=1` reads the lower face of each layer.

**Oblique lattices:** Integer `sample_x` and `sample_y` place points along the two lattice vectors; the horizontal array axes correspond to `(v, u)`. If both inputs are coordinate arrays, fields are read at every combination of Cartesian x and y coordinates. For an oblique lattice, do not mix a point count on one axis with a coordinate array on the other.

**Dipoles:** The field is singular at the height of a point dipole. Choose positions above and below it, excluding that height. If count-based sampling includes the source height, choose different sampling positions.
:::

(fields-detail-id26)=
### Selecting components from the returned array

Interior fields from every result have shape `(nz, ny, nx, 6)`. The first three axes select the position; the last axis selects the component.

```{figure} ../../figures/learn/learn/13-field-layout.svg
:alt: The common field array has z, y, x, and component axes; its last axis contains Ex Ey Ez Z0Hx Z0Hy Z0Hz
:figclass: learning-figure

Figure 14-2. Axes and six components of the field array read from a result object. The magnetic field is normalized by multiplication by Z₀.
```

| Index | `0` | `1` | `2` | `3` | `4` | `5` |
| --- | --- | --- | --- | --- | --- | --- |
| Component | E<sub>x</sub> | E<sub>y</sub> | E<sub>z</sub> | Z<sub>0</sub>H<sub>x</sub> | Z<sub>0</sub>H<sub>y</sub> | Z<sub>0</sub>H<sub>z</sub> |

```python
E = cell[..., :3]           # Three electric-field components
H_normalized = cell[..., 3:] # Three normalized magnetic-field components
Ey = cell[..., 1]           # Electric-field y component at every position
```

The six components occupy the same positions for TE/TM. In pure TE, `Ex`, `Ez`, and `Hy` are zero; in pure TM, `Ey`, `Hx`, and `Hz` are zero.

For a general coupled solve that also computes pure s and p responses, read them individually with `incidence="s"` or `"p"`. The default, `"incident"`, returns the response for the specified incidence.

```python
# For a result that includes both s and p responses
s_field = result.reconstruct_field(incidence="s")
p_field = result.reconstruct_field(incidence="p")
```

This option does not separate contributions from different incident waves. For pure TE/TM and results combining multiple sources, use only `incidence="incident"`.

:::{dropdown} Fields while differentiating with respect to thickness

Whenever the thickness changes, call `solve()` with that thickness and read the fields from the returned result.

```python
def field_for_thickness(d):
    result = solver.solve(thickness=d, answers="field")
    return result.reconstruct_field(
        sample_x=[0.], sample_y=[0.], sample_z=20,
    )
```

The common field function for ordinary-incidence results supports automatic differentiation with respect to thickness in JAX and PyTorch. `sample_z=20` places points at relative positions within each layer, so their physical heights change with the thickness.

Coordinates such as `sample_z=[50., 100.]` instead give the change at fixed heights. Differentiation is valid over intervals in which the samples remain within the same layers. When an interface crosses a sample, the local material and normal field components may change, so a smooth derivative cannot be expected there. With JAX `jit`, keep the input range such that coordinates remain inside the stack.

Source solves involving `PlaneWave`, dipoles, and similar sources use concrete numerical values to classify sources and split layers. The differentiation discussion above applies to ordinary-incidence solves.
:::

(illumination-fields)=
(fields-detail-id23)=
## Incoming, outgoing, and total fields at boundaries

Read a boundary with `result.boundary_fields(...)`. The same function serves single incidence and results combining multiple sources. The example below sends s-polarized waves from both above and below, with a relative phase of 90° for the bottom incident wave.

```python
field_solver = wavebit.rcwa(
    wavelength=600., period=(900., 850.),  # Length unit: nm
    top=1., bot=2.25,
    structure=[dict(background=4.+0.6j, shapes=[])],
    thickness=[100.], truncation_order=0,
    kx=0.002, ky=0.,                       # Unit: nm⁻¹
    excitation={
        "top": wavebit.PlaneWave(polarization=(1., 0.)),
        "bottom": wavebit.PlaneWave(
            polarization=(1., 0.),
            amplitude=np.exp(1j * np.deg2rad(90.)),
        ),
    },
)
field_result = field_solver.solve(answers="field")
x = np.linspace(-300., 300., 61)
y = np.array([0.])
```

(fields-detail-id24)=
### Selecting the wave to read

The upper boundary is at `z=100 nm` and the lower boundary at `z=0`. The following code reads fields along the `y=0` line on the upper boundary.

```python
incoming = field_result.boundary_fields("top", x, y, part="incoming")
outgoing = field_result.boundary_fields("top", x, y, part="outgoing")
boundary = field_result.boundary_fields("top", x, y, part="total")
print(boundary.shape)  # (1, 61, 6)
# At the same positions, boundary = incoming + outgoing
```

| Input | Meaning |
| --- | --- |
| `side` | `"top"` or `"bottom"`, selecting the boundary to read |
| `sample_x`, `sample_y` | Point counts or coordinate arrays; each defaults to `[0.]`, the cell center |
| `part="incoming"` | Waves entering toward the layers from the selected side |
| `part="outgoing"` | Waves leaving the layers toward the selected side |
| `part="total"` — default | The total field, summing incoming and outgoing waves |

The returned shape is `(ny, nx, 6)`, with the same component order as interior fields. Contributions from multiple sources are added as complex amplitudes. For two-sided incidence, the upper `outgoing` field includes both reflection of the top incident wave and transmission of the bottom incident wave.

(fields-detail-id25)=
### Outside and inside a boundary

`boundary_fields()` returns the value on the **half-space side** of the boundary. Read the value inside the layer with `reconstruct_field()`.

```python
inside = field_result.reconstruct_field(x, y, sample_z=[0., 50., 100.])
print(inside.shape)  # (3, 1, 61, 6)
middle = inside[1]   # Field along the line at z=50 nm
```

| Call | Side sampled |
| --- | --- |
| `field_result.boundary_fields("top", x, y)` | The `top` half-space side of the upper boundary |
| `field_result.reconstruct_field(x, y, sample_z=[100.])` | Inside the 100 nm layer at the upper boundary |

At an interface with no free surface charge or current, the tangential components `Ex`, `Ey`, `Hx`, and `Hy` are continuous. The normal components `Ez` and `Hz` can differ with the material. The interior-field function returns the total field from all sources, so it takes no `part` argument.

Fields at specified positions are neither area-averaged over `power_region` nor divided by incident power. They are the complex electric and magnetic fields at those positions, including the amplitudes and phases of the sources.

(fields-detail-id2)=
(complex-field-time)=
## Amplitude, phase, and instantaneous electric field

Extract the electric-field y component at the middle height from `inside` above:

```python
Ey = inside[1, 0, :, 1]              # Ey along z=50 nm, y=0
Ey_amplitude = np.abs(Ey)           # Amplitude magnitude
Ey_phase_deg = np.angle(Ey, deg=True) # Phase in degrees
```

If the returned value at a position is {math}`E_y=Ae^{i\varphi}`, then {math}`A=|E_y|` is the amplitude magnitude and {math}`\varphi=\arg(E_y)` is the phase. The complex number {math}`E_y` expressing both is called a **complex amplitude**, or phasor.

The actual electric-field component oscillating in time, {math}`e_y(t)`, is given below. Here {math}`\omega` is the angular frequency of the light and {math}`t` is time; the spatial position is held fixed.

```{math}
e_y(t)
=\operatorname{Re}\!\left[E_y e^{-i\omega t}\right]
=A\cos(\varphi-\omega t).
```

Thus, `Ey.real` is the instantaneous electric field at `t=0`, whereas `np.abs(Ey)` is the oscillation amplitude. Giving the time phase `omega_t = ωt` in radians lets you calculate another instant.

```python
omega_t = np.pi / 2
Ey_instant = np.real(Ey * np.exp(-1j * omega_t))
```

For example, {math}`E_y=e^{i\pi/3}` has amplitude magnitude 1 and phase 60°. Its instantaneous field is {math}`\cos(\pi/3-\omega t)`: it equals 0.5 at `t=0` and oscillates between −1 and 1 as time advances.

```{raw} html
<section class="exercise" aria-labelledby="phasor-title"><h3 id="phasor-title">Try it · From one complex number to a real oscillation</h3><p class="concept">Illustrative single component E=exp(iπ/3), magnitude 1. This conceptual example is separate from the thin-film field calculated above.</p><div class="controls"><label for="phasor-time">Time phase ωt <output id="phasor-time-out" for="phasor-time">0°</output><input id="phasor-time" type="range" min="0" max="360" value="0"></label></div><svg viewBox="0 0 400 260" role="img" aria-label="Phasor rotating in the complex plane with time and its projection onto the real axis"><circle cx="155" cy="125" r="94" fill="none" stroke="#cad7df"/><line x1="30" y1="125" x2="293" y2="125" stroke="#abbec5"/><line x1="155" y1="15" x2="155" y2="240" stroke="#abbec5"/><text x="299" y="132">Real</text><text x="163" y="21">Imaginary</text><line id="phasor-vector" x1="155" y1="125" x2="202" y2="44" stroke="#285ab0" stroke-width="3"/><line id="phasor-drop" x1="202" y1="44" x2="202" y2="125" stroke="#b56626" stroke-dasharray="4 4"/><circle id="phasor-point" cx="202" cy="125" r="6" fill="#b56626"/><text x="42" y="258">Orange point: real part, the instantaneous value</text></svg><div class="readout" id="phasor-readout" aria-live="polite">At ωt=0, the real part is 0.5 and |E|²=1.</div></section>
```

The sum of the squared magnitudes of the three electric-field components, {math}`|\mathbf E|^2`, is obtained as follows:

```python
E_squared = np.sum(np.abs(inside[..., :3])**2, axis=-1)
print(E_squared.shape)  # (3, 1, 61)
```

Use this quantity to compare electric-field strength at different positions. The direction and magnitude of power flow instead follow from the Poynting vector, which uses the magnetic field as well.

(magnetic-field-units)=
## Magnetic-field normalization and physical units

When solving Maxwell’s equations, Wavebit uses the **normalized magnetic field** {math}`\widetilde{\mathbf H}=Z_0\mathbf H` in place of {math}`\mathbf H`. Here {math}`Z_0\approx376.73\,\Omega` is the vacuum impedance. The field array contains this normalized value, so divide by {math}`Z_0` to obtain the magnetic field itself.

```{math}
\mathbf H=\frac{\widetilde{\mathbf H}}{Z_0}.
```

In the six-component array above, the last three entries are the normalized magnetic field.

```python
E = inside[..., :3]
H_normalized = inside[..., 3:]
Z0 = 376.730313668
H = H_normalized / Z0
```

:::{dropdown} Amplitude convention for pure TM input

A pure 1D solve with `tetm="tm"` specifies amplitude relative to the normalized magnetic field of the incident wave. In contrast, `polarization=(a_s, a_p)` and `PlaneWave` use the electric field. The common field function preserves these input conventions; it neither divides the returned fields by incident power nor automatically rescales them to equal amplitudes.
:::

If an input electric-field amplitude of `1` represents `1 V/m`, `E` has units of V/m and the converted `H` has units of A/m. For example, a returned magnetic-field component of `1` becomes approximately `0.00265 A/m` after conversion.

If the input amplitude is only relative, you must also choose a physical scale. For example, to make a calculated electric field of `1` correspond to `10 V/m`, multiply both `E` and `H` above by 10. Dividing by {math}`Z_0` is a constant scaling, so it does not alter the magnetic field’s spatial pattern or phase.

### Calculating power density

The time-averaged Poynting vector follows from the complex electric and magnetic amplitudes. The asterisk {math}`*` denotes complex conjugation.

```{math}
\langle\mathbf S\rangle
=\frac12\operatorname{Re}(\mathbf E\times\mathbf H^*)
=\frac{1}{2Z_0}\operatorname{Re}
  (\mathbf E\times\widetilde{\mathbf H}^{\,*}).
```

Using the converted `H` above:

```python
S = 0.5 * np.real(np.cross(E, np.conj(H)))
Sz = S[..., 2]
```

If `E` is in V/m and `H` in A/m, `S` is in W/m². `Sz > 0` means upward power flow along +z, and `Sz < 0` means downward flow along −z. [Chapter 13](power-and-efficiency.md) explains how these quantities determine reflectance, transmittance, and net entering power.

For power ratios such as reflectance and transmittance, the factor `1/Z0` cancels between numerator and denominator. Therefore, do not apply another magnetic-field unit conversion to returned `R`, `T`, `output_top`, or `output_bottom`.

:::{dropdown} Converting result.power from a PlaneWave result to watts

`result.power.incoming_top`, `incoming_bottom`, `outgoing_top`, and `outgoing_bottom` are the normal power flows of the respective incoming and outgoing waves, averaged over `power_region`. Each wave’s incoming or outgoing direction is taken as positive. Values are returned in computational units with the common factor `1/Z0` omitted.

Let a selected value of `result.power` be {math}`P_{\mathrm{calc}}`. If a calculated electric-field amplitude of `1` corresponds to a physical {math}`E_{\mathrm{scale}}\,\mathrm{V/m}` and the physical observation area is {math}`A_{\mathrm{obs}}\,\mathrm{m}^2`, the total power is:

```{math}
P\;[\mathrm W]
=P_{\mathrm{calc}}\,\frac{E_{\mathrm{scale}}^2}{Z_0}\,A_{\mathrm{obs}}.
```

For a rectangular observation region, for example, convert its physical area to m². An average over an infinite region alone does not determine a finite total power; you must also specify an area to report total power.
:::

:::{dropdown} Magnetic-field conventions in other RCWA tools

| Tool | Meaning of the magnetic field and supporting reference |
| --- | --- |
| <span style="white-space:nowrap">Meent</span> | The values named `Hx`, `Hy`, and `Hz` in its field code also represent a normalized magnetic field. The scale follows from the equations in its [field reconstruction code](https://github.com/kc-ml2/meent/blob/main/meent/on_numpy/emsolver/field_distribution.py) and [matrix assembly code](https://github.com/kc-ml2/meent/blob/main/meent/on_numpy/emsolver/transfer_method.py). |
| <span style="white-space:nowrap">S4</span> | The returned `H` corresponds to Z₀H<sub>SI</sub>. This unit conversion is stated in [§3.1 of the official paper](https://web.stanford.edu/group/fan/publication/Liu_ComputerPhysicsCommunications_183_2233_2012.pdf). |
| <span style="white-space:nowrap">FMMAX</span> | Its [official conventions](https://facebookresearch.github.io/fmmax/#fmm-conventions) set the vacuum ε₀, μ₀, and c to 1. The returned magnetic field in this unit system is interpreted using the same normalization. |
| <span style="white-space:nowrap">RETICOLO</span> | [Page 4 of the official manual](https://arxiv.org/pdf/2101.00901#page=4) uses Maxwell’s equations with ε₀, μ₀, and c set to 1. The `H` components from `res3()` follow this unit system. |

A returned value named `H` therefore does not automatically mean A/m. When comparing complex fields between tools, align the incident-amplitude reference, coordinate axes, and phase conventions as well as magnetic-field normalization.
:::

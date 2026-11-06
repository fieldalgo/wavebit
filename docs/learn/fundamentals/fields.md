# 14. Electric and magnetic fields at specified positions

:::{note}
Figure 14-2 still draws the earlier field layout: no wavelength axis, and Z₀H in the last three
slots. The current `.values` array is `(n_wavelengths, nz, ny, nx, 6)` and its last three
components are H in A/m. The text and code use the current layout.
:::

Reflectance and transmittance describe fractions of the power carried by light. To see the electric and magnetic field distributions inside a layer, read the **fields** at the desired positions after solving. Reading a field means obtaining its electric and magnetic components at that position.

For a single plane wave, several plane waves, Gaussian beams, and dipole sources alike, fields are read from the **result object** returned by `rcwa.run()`. This chapter covers preparation, sampling coordinates, returned arrays, and boundary fields, then explains complex amplitudes and magnetic-field units. The [field guide](../../guide/rcwa/fields.md) states the same rules in reference form.

(field-preparation)=
## Reading fields from a solved result

Ask for fields before running, with `rcwa.Outputs(fields=True)` in the problem.

```python
problem = problem.replace(outputs=rcwa.Outputs(fields=True))
result = rcwa.run(problem, numerics=numerics)
cell = result.field(x=41, y=21, z=20)
```

`result` retains the wavevectors, thicknesses, and per-layer state used for this run. Changing only the sampling positions does not require another run. You can repeatedly read fields at different positions from the same result. Running another problem, or the same preparation again, does not change the fields read from this `result`; it keeps returning the fields of its own conditions.

What a result retains depends on `Outputs` as follows. These rules apply regardless of how the incident waves and sources are specified.

| `Outputs` | Power results | Boundary fields | Interior fields |
| --- | --- | --- | --- |
| `Outputs()` — default | Retained | — | — |
| `Outputs(fields=True)` | Retained | Retained | Retained |

Read boundary electric and magnetic fields with `result.boundary_fields(...)` and fields inside layers with `result.field(...)`. Both need `Outputs(fields=True)`; without it they raise `UnsupportedFeatureError` with code `outputs.unavailable`. A run without fields uses the boundary fields to compute power and then releases them. If you later need fields, run again with `Outputs(fields=True)`.

For a result with sources other than a single plane wave, `result.outputs['top']` and `result.outputs['bottom']` hold the power over the problem's `Outputs(power_region=...)`, with or without fields. The current result has no method that recomputes power over another region from its retained fields; run the problem again with that region.

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

For `x`, `y`, and `z`, supply either a positive integer **number of points** or a nonempty one-dimensional array of **coordinates**. `field()` requires all three and `boundary_fields()` requires `x` and `y`; there are no defaults.

| Input | Meaning | Example |
| --- | --- | --- |
| `x` | Number of points along the cell x direction, or an array of x coordinates | `41`, `[-100., 0., 100.]` |
| `y` | Number of points along the cell y direction, or an array of y coordinates | `21`, `[0.]` |
| `z` | Number of points per layer, or an array of z coordinates for the entire stack | `20`, `[0., 60., 120.]` |

To sample just three heights at the cell center in the example above, write:

```python
three_heights = result.field(x=[0.], y=[0.], z=[0., 60., 120.])
print(three_heights.values.shape)  # (1, 3, 1, 1, 6)
```

`z=20` places **20 points in each layer**. Two layers therefore produce 40 samples along z. In contrast, `z=[20.]` reads one sample at a height of 20 nm.

- The x and y origins are at the cell center. For a rectangular cell, a point count produces uniform samples including both endpoints in each direction. A count of 1 selects the negative endpoint, so use `[0.]` for a single point at the center.
- The z origin is at the bottom of the lowest layer, and z increases upward. Coordinate arrays must lie in `0 ≤ z ≤ sum(thickness)`.
- Coordinates use the same length unit as `wavelength`, `period`, and `thickness`. If these values are given in nm, as in the example, the coordinates are also in nm.

When x and y are coordinate arrays, every x–y combination is sampled at every z height. The order of the supplied coordinates is preserved. Even a pure 1D field with no y dependence is returned at all requested y positions.

:::{dropdown} Coordinates at layer boundaries and in oblique lattices

**Layer boundaries:** A z coordinate exactly on an internal interface reads the value in the upper layer. At `z=0` and `z=sum(thickness)`, it reads the interior value of the bottommost and topmost layer, respectively.

When `z` is a point count, each layer is sampled separately. A count of at least 2 includes both ends of each layer, so an internal interface appears once from the lower layer and once from the upper layer. The normal components `Ez` and `Hz` can differ across a material interface. `z=1` reads the lower face of each layer.

**Oblique lattices:** Integer `x` and `y` place points along the two lattice vectors; the horizontal array axes correspond to `(v, u)`. If both inputs are coordinate arrays, fields are read at every combination of Cartesian x and y coordinates. For an oblique lattice, do not mix a point count on one axis with a coordinate array on the other.

**Dipoles:** The field is singular at the height of a point dipole. Choose positions above and below it, excluding that height. If count-based sampling includes the source height, choose different sampling positions.
:::

(fields-detail-id26)=
### Selecting components from the returned array

`result.field(...)` returns a field-sample object. Its `.values` array has shape `(n_wavelengths, nz, ny, nx, 6)` for every result: the first axis selects the wavelength, even for a single wavelength, the next three select the position, and the last axis selects the component. `.components` names the six components and `.units` gives the unit of each component and of `'coordinates'`.

```{figure} ../../figures/learn/learn/13-field-layout.svg
:alt: The common field array has z, y, x, and component axes; its last axis contains Ex Ey Ez Z0Hx Z0Hy Z0Hz
:figclass: learning-figure

Figure 14-2. Position axes and six components of the field array read from a result object. The figure still draws the earlier layout: the current `.values` array has a leading wavelength axis, and its last three components are H in A/m, not Z₀H.
```

| Index | `0` | `1` | `2` | `3` | `4` | `5` |
| --- | --- | --- | --- | --- | --- | --- |
| Component | E<sub>x</sub> | E<sub>y</sub> | E<sub>z</sub> | H<sub>x</sub> | H<sub>y</sub> | H<sub>z</sub> |
| Unit | V/m | V/m | V/m | A/m | A/m | A/m |

```python
E = cell.values[..., :3]    # Three electric-field components, V/m
H = cell.values[..., 3:]    # Three magnetic-field components, A/m
Ey = cell.values[..., 1]    # Electric-field y component at every wavelength and position
print(cell.components)      # ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
```

The six components occupy the same positions for every polarization. For s (TE) light in the xz plane of incidence (`phi=0`) on a structure invariant along y, `Ex`, `Ez`, and `Hy` are zero; for p (TM) light, `Ey`, `Hx`, and `Hz` are zero.

`field()` returns the response to the declared excitation. To obtain the pure s and pure p responses, run the problem twice, once with `polarization="s"` and once with `"p"`, and read each result's fields.

:::{admonition} Earlier interface
:class: earlier-interface

The earlier interface's coupled solves also retained pure s and p responses, read with `result.reconstruct_field(incidence="s")` or `incidence="p"`; its default, `"incident"`, returned the declared incidence. The current `field()` and `boundary_fields()` take no `incidence` argument.
:::

:::{dropdown} Fields while differentiating with respect to thickness

Prepare once outside the derivative, then run the live problem with each thickness and read the fields from the returned result.

```python
prepared = rcwa.prepare(problem, numerics=numerics, backend="jax")

def field_for_thickness(d):
    layer = problem.stack.layers[0].replace(thickness=d)
    live = problem.replace(stack=problem.stack.replace(layers=(layer,)))
    return prepared.run(live).field(x=[0.], y=[0.], z=20).values
```

For single-plane-wave results, `prepared.run` and `field` support automatic differentiation with respect to thickness in JAX and PyTorch. `z=20` places points at relative positions within each layer, so their physical heights change with the thickness.

Coordinates such as `z=[50., 100.]` instead give the change at fixed heights. Differentiation is valid over intervals in which the samples remain within the same layers. When an interface crosses a sample, the local material and normal field components may change, so a smooth derivative cannot be expected there. With JAX `jit`, keep the input range such that coordinates remain inside the stack.

Runs with other sources, such as several `PlaneWave`s, a `GaussianBeam` or a dipole, use concrete numerical values to group sources and split layers and refuse a JAX-traced input. The differentiation discussion above applies to single-plane-wave runs.
:::

(illumination-fields)=
(fields-detail-id23)=
## Incoming, outgoing, and total fields at boundaries

Read a boundary with `result.boundary_fields(...)`. The same method serves a single plane wave and results combining several sources. The example below sends s-polarized waves from both above and below, with a relative phase of 90° for the bottom incident wave. A run with sources on both sides reports port powers instead of diffraction efficiencies, so it needs `Outputs(diffraction=False)`.

```python
import numpy as np
from wavebit import Material, Units, rcwa

field_problem = rcwa.Problem(
    units=Units(length="nm"),
    stack=rcwa.Stack(
        period=(900., 850.),
        top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(thickness=100., background=Material(epsilon=4. + 0.6j)),)),
    excitation=rcwa.Excitation(wavelength=600., sources=(
        rcwa.PlaneWave(side="top", polarization="s", kx=0.002, ky=0.),       # Unit: nm⁻¹
        rcwa.PlaneWave(side="bottom", polarization="s", kx=0.002, ky=0.,
                       amplitude=np.exp(1j * np.deg2rad(90.))),
    )),
    outputs=rcwa.Outputs(diffraction=False, fields=True),
)
field_result = rcwa.run(field_problem, numerics=rcwa.Numerics(orders=(0, 0)))
x = np.linspace(-300., 300., 61)
y = np.array([0.])
```

(fields-detail-id24)=
### Selecting the wave to read

The upper boundary is at `z=100 nm` and the lower boundary at `z=0`. The following code reads fields along the `y=0` line on the upper boundary.

```python
incoming = field_result.boundary_fields(side="top", x=x, y=y, part="incoming")
outgoing = field_result.boundary_fields(side="top", x=x, y=y, part="outgoing")
boundary = field_result.boundary_fields(side="top", x=x, y=y, part="total")
print(boundary.values.shape)  # (1, 1, 61, 6)
# At the same positions, boundary = incoming + outgoing
```

| Input | Meaning |
| --- | --- |
| `side` | `"top"` or `"bottom"`, selecting the boundary to read |
| `x`, `y` | Point counts or coordinate arrays; required. `[0.]` is the cell center |
| `part="incoming"` | Waves entering toward the layers from the selected side |
| `part="outgoing"` | Waves leaving the layers toward the selected side |
| `part="total"` — default | The total field, summing incoming and outgoing waves |

The `.values` shape is `(n_wavelengths, ny, nx, 6)`, with the same component order and units as interior fields. Contributions from multiple sources are added as complex amplitudes. For two-sided incidence, the upper `outgoing` field includes both reflection of the top incident wave and transmission of the bottom incident wave.

(fields-detail-id25)=
### Outside and inside a boundary

`boundary_fields()` returns the value on the **half-space side** of the boundary. Read the value inside the layer with `field()`.

```python
inside = field_result.field(x=x, y=y, z=[0., 50., 100.])
print(inside.values.shape)  # (1, 3, 1, 61, 6)
middle = inside.values[0, 1]  # Field along the line at z=50 nm
```

| Call | Side sampled |
| --- | --- |
| `field_result.boundary_fields(side="top", x=x, y=y)` | The `top` half-space side of the upper boundary |
| `field_result.field(x=x, y=y, z=[100.])` | Inside the 100 nm layer at the upper boundary |

At an interface with no free surface charge or current, the tangential components `Ex`, `Ey`, `Hx`, and `Hy` are continuous. The normal components `Ez` and `Hz` can differ with the material. The interior-field function returns the total field from all sources, so it takes no `part` argument.

Fields at specified positions are neither area-averaged over `power_region` nor divided by incident power. They are the complex electric and magnetic fields at those positions, including the amplitudes and phases of the sources.

(fields-detail-id2)=
(complex-field-time)=
## Amplitude, phase, and instantaneous electric field

Extract the electric-field y component at the middle height from `inside` above:

```python
Ey = inside.values[0, 1, 0, :, 1]    # Ey along z=50 nm, y=0
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
E_squared = np.sum(np.abs(inside.values[..., :3])**2, axis=-1)
print(E_squared.shape)  # (1, 3, 1, 61)
```

Use this quantity to compare electric-field strength at different positions. The direction and magnitude of power flow instead follow from the Poynting vector, which uses the magnetic field as well.

(magnetic-field-units)=
## Magnetic-field normalization and physical units

When solving Maxwell’s equations, Wavebit uses the **normalized magnetic field** {math}`\widetilde{\mathbf H}=Z_0\mathbf H` in place of {math}`\mathbf H`. Here {math}`Z_0\approx376.73\,\Omega` is the vacuum impedance. Before returning, the result divides by {math}`Z_0`, so the field array already contains the magnetic field itself.

```{math}
\mathbf H=\frac{\widetilde{\mathbf H}}{Z_0}.
```

In the six-component array above, the last three entries are H in A/m. Do not divide them by {math}`Z_0` again.

```python
E = inside.values[..., :3]   # V/m
H = inside.values[..., 3:]   # A/m
print(inside.units["Hx"])    # A/m
```

::::{dropdown} Amplitude reference of every source

Every source amplitude is an electric amplitude: a `PlaneWave` `amplitude` is in V/m for `polarization="s"`, `"p"` or `(a_s, a_p)`, and a dipole moment is in C·m. The field methods preserve that scale; they neither divide the returned fields by incident power nor rescale them to equal amplitudes.

:::{admonition} Earlier interface
:class: earlier-interface

Until 2026-09-27 the earlier interface's pure 1D solve with `tetm="tm"` was lit by a unit normalized magnetic field, so its field was `mu_inc/n_inc` of the conical p field for the same light. That interface also returned Z₀H in the last three slots; divide those by Z₀ before comparing them with a current result.
:::
::::

A `PlaneWave` amplitude of `1` represents `1 V/m`, so `E` is in V/m and `H` in A/m. For example, a normalized magnetic field Z₀H of `1` corresponds to about `0.00265 A/m`.

The fields are linear in the source amplitude. To make the incident electric field `10 V/m`, set `amplitude=10.` on the source, or multiply both `E` and `H` from a unit-amplitude run by 10. Dividing by {math}`Z_0` is a constant scaling, so it does not alter the magnetic field’s spatial pattern or phase.

### Calculating power density

The time-averaged Poynting vector follows from the complex electric and magnetic amplitudes. The asterisk {math}`*` denotes complex conjugation.

```{math}
\langle\mathbf S\rangle
=\frac12\operatorname{Re}(\mathbf E\times\mathbf H^*)
=\frac{1}{2Z_0}\operatorname{Re}
  (\mathbf E\times\widetilde{\mathbf H}^{\,*}).
```

Using `E` and `H` from the field array:

```python
S = 0.5 * np.real(np.cross(E, np.conj(H)))
Sz = S[..., 2]
```

If `E` is in V/m and `H` in A/m, `S` is in W/m². `Sz > 0` means upward power flow along +z, and `Sz < 0` means downward flow along −z. [Chapter 13](power-and-efficiency.md) explains how these quantities determine reflectance, transmittance, and net entering power.

For power ratios such as reflectance and transmittance, the factor `1/Z0` cancels between numerator and denominator. Do not apply another magnetic-field unit conversion to the returned `R`, `T`, `result.outputs['top']`, or `result.outputs['bottom']`.

::::{dropdown} Port powers of a source result are in watts

For a result with sources other than a single plane wave, `result.outputs['top']` and `result.outputs['bottom']` each hold `incoming` and `outgoing`: the normal power flows of the incoming and outgoing waves at that port, with each wave’s own direction taken as positive. They are physical powers for the declared source amplitudes in V/m. Their `units` is `'W'` over the cell or the `Outputs(power_region=...)` rectangle, named in `power_region`, and `'W/m^2'` for the `'infinite'` region, a mean per unit area. In the two-sided example above, `field_result.outputs['top'].incoming` is about `9.97e-16` W: the top wave’s 1 V/m in air, `cos(theta)/(2 Z0)` W/m², over the 900 × 850 nm cell.

Powers scale with the square of the source amplitude. An average over an infinite region alone does not determine a finite total power; to report a total, run with a finite `power_region`.

:::{admonition} Earlier interface
:class: earlier-interface

The earlier interface's `result.power.incoming_top`, `incoming_bottom`, `outgoing_top`, and `outgoing_bottom` were averages over `power_region` in computational units with the common factor `1/Z0` omitted. For a selected value {math}`P_{\mathrm{calc}}`, a calculated electric-field amplitude of `1` corresponding to {math}`E_{\mathrm{scale}}\,\mathrm{V/m}`, and a physical observation area {math}`A_{\mathrm{obs}}\,\mathrm{m}^2`, the total power was

```{math}
P\;[\mathrm W]
=P_{\mathrm{calc}}\,\frac{E_{\mathrm{scale}}^2}{Z_0}\,A_{\mathrm{obs}}.
```

Do not apply this conversion to a current result.
:::
::::

:::{dropdown} Magnetic-field conventions in other RCWA tools

| Tool | Meaning of the magnetic field and supporting reference |
| --- | --- |
| <span style="white-space:nowrap">Meent</span> | The values named `Hx`, `Hy`, and `Hz` in its field code also represent a normalized magnetic field. The scale follows from the equations in its [field reconstruction code](https://github.com/kc-ml2/meent/blob/main/meent/on_numpy/emsolver/field_distribution.py) and [matrix assembly code](https://github.com/kc-ml2/meent/blob/main/meent/on_numpy/emsolver/transfer_method.py). |
| <span style="white-space:nowrap">S4</span> | The returned `H` corresponds to Z₀H<sub>SI</sub>. This unit conversion is stated in [§3.1 of the official paper](https://web.stanford.edu/group/fan/publication/Liu_ComputerPhysicsCommunications_183_2233_2012.pdf). |
| <span style="white-space:nowrap">FMMAX</span> | Its [official conventions](https://facebookresearch.github.io/fmmax/#fmm-conventions) set the vacuum ε₀, μ₀, and c to 1. The returned magnetic field in this unit system is interpreted using the same normalization. |
| <span style="white-space:nowrap">RETICOLO</span> | [Page 4 of the official manual](https://arxiv.org/pdf/2101.00901#page=4) uses Maxwell’s equations with ε₀, μ₀, and c set to 1. The `H` components from `res3()` follow this unit system. |

Wavebit's returned `H` is in A/m, but a value named `H` in another tool does not automatically mean A/m. When comparing complex fields between tools, align the incident-amplitude reference, coordinate axes, and phase conventions as well as magnetic-field normalization.
:::

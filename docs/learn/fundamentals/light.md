# 08. Choosing incident light: direction and polarization

To specify one incident wave, choose its **wavelength, incident side, propagation direction and polarization**. In code, the wavelength belongs to `rcwa.Excitation(wavelength=...)` and the rest to one `rcwa.PlaneWave(side=..., theta=..., phi=..., polarization=...)` in its `sources`. This chapter explains illumination from above or below and wavevectors in lossy media.

[Chapter 9](multiple-beams.md) continues with independent amplitudes and phases for multiple incident waves and simultaneous illumination from both sides. [Chapter 13](power-and-efficiency.md) explains output power and efficiency, and [Chapter 14](fields.md) explains electric- and magnetic-field queries.

| What to specify | Input and explanation |
| --- | --- |
| Wavelength | `Excitation(wavelength=...)`: use the same unit as all lengths; all incident waves in one calculation share it. |
| Propagation direction | {ref}`theta·phi or kx·ky <incident-direction-inputs>` |
| Polarization | {ref}`polarization='s', 'p' or an (s, p) pair <incident-polarization>` |
| One-sided illumination | {ref}`Choose top or bottom with PlaneWave(side=...) <incident-from-bottom>` |

Wavevectors and attenuation in lossy exterior media are discussed separately in the {ref}`last section of this chapter <absorbing-incidence>`.

(incident-direction-inputs)=
## Two ways to specify the incident propagation direction

`theta` is the angle between the **direction of phase propagation** in the incident medium and the inward surface normal (`−z` for top incidence, `+z` for bottom incidence); `phi` is the azimuth. Phase propagation describes how points of equal phase, such as wave crests, move. In a negative-index incident medium, where phase runs against the energy flow, `theta` instead gives the direction in which light enters the structure (since 2026-10-05; [negative-index media](../../theory/rcwa/negative-index-media.md), Section 7). Angles are in radians. Polarization, the direction in which the electric field oscillates, is specified separately in the next section.

This direction input specifies one incident wave with uniform intensity along the surface. In an absorbing incident medium, phase propagation and power flow can differ in direction; see the {ref}`lossy-medium section <absorbing-incidence>` for the corresponding wavevectors.

The diagram below uses a lossless incident medium. s polarization is perpendicular to the plane of incidence, while p lies within it. At normal incidence, propagation direction alone does not uniquely define that plane, so the selected azimuth and basis convention also matter.

```{figure} ../../figures/learn/learn/08-light-azimuth.svg
:name: light-figure-08-light-azimuth
:alt: Three-dimensional view of top incidence with theta 40 degrees and phi 30 degrees. theta lies between the vertical and the light path; phi lies between the surface +x axis and the xy component of the wavevector. s is normal to the incidence plane; p lies in that plane and is perpendicular to k.
:figclass: learning-figure

Figure 8-1. Read the incidence angle θ, azimuth φ and s·p directions in the same three-dimensional view.
```

**The gray surface is the sample’s upper xy surface, and the translucent blue plane is the imaginary plane of incidence.** This plane contains both propagation direction k and the surface normal. For φ=30°, it is the xz plane rotated 30° about z. Light follows the blue arrow down toward the sample.

Along the same light path, **orange s points out of the plane and green p lies within it**. Both are electric-field basis directions perpendicular to k. `(1, 0)` means an electric field oscillating along s, and `(0, 1)` one oscillating along p; these are not light-propagation directions. Projection onto a flat screen does not preserve the apparent right angles between all three directions.

**φ is measured from the +x surface axis to the xy component of incident wavevector k.** The dashed blue arrow is that component, and the orange arc marks φ. Viewed from +z, counterclockwise rotation from +x toward +y is positive. θ controls the tilt of the light, while φ chooses its direction along the surface.

The shape cross-section and boundary normals beneath the diagram are explained in [Chapter 11 on NVF](nvf.md).


Specify propagation direction either with angles or with wavevector components parallel to the surface.

| Input | Quantity specified directly | When wavelength changes |
| --- | --- | --- |
| `theta`, `phi` | Phase-propagation angle and azimuth in the incident medium, in radians | Keep the angles and recompute wavevectors |
| `kx`, `ky` | Physical incident wavevector components in x and y | Keep the input wavevectors; the phase angle can change |

**Use one of these two forms.** `kx` and `ky` have inverse-length units. If wavelength is in nm, wavevectors are in nm⁻¹; they are not divided by `k0`. If only one component is supplied, the other is zero. Omitting all direction inputs selects normal incidence.

For example, an incident medium with `ε=2.25`, `μ=1` has index 1.5. A wave at phase angle 30° and azimuth 0° in that medium has `kx=0.75 k0`, `ky=0`, so these inputs specify the same incident wave.

```python
import numpy as np
from wavebit import Material, Units, rcwa

def problem(source):
    return rcwa.Problem(
        units=Units(length="nm"),                  # Length unit: nm
        stack=rcwa.Stack(
            period=(700., 700.),
            top=Material(epsilon=2.25), bottom=Material(epsilon=1.),
            layers=(rcwa.Layer(thickness=100., background=Material(epsilon=4.)),)),
        excitation=rcwa.Excitation(wavelength=500., sources=(source,)))

numerics = rcwa.Numerics(orders=(0, 0))
by_angle = rcwa.run(problem(rcwa.PlaneWave(
    side="top",                                     # Downward incidence from the upper medium (default)
    polarization="s", theta=np.pi/6, phi=0.)), numerics=numerics)

k0 = 2*np.pi/500.                                   # Vacuum wavevector magnitude, nm⁻¹
by_k = rcwa.run(problem(rcwa.PlaneWave(
    side="top", polarization="s", kx=0.75*k0, ky=0.)), numerics=numerics)

np.testing.assert_allclose(by_angle.T, by_k.T, atol=1e-12)
```

**The incident medium’s ε and μ remain necessary** when entering `kx` and `ky` directly. At the same transverse wavevector, changing material changes the normal wavevector, the electric-to-magnetic-field relation and incident power. Direct wavevector input also works for a lossy incident medium; no extra air layer is assumed.

These inputs are **real wavevectors in physical xyz coordinates**. They are not the oblique lattice’s `u, v` order indices. The solver adds reciprocal-lattice wavevectors for the diffraction orders.

`PlaneWave(side=...)` chooses which side illuminates the structure. In the example, `side="top"` means **downward (−z)** incidence from the upper medium `Stack.top`; this is also the default. `side="bottom"` means **upward (+z)** incidence from the lower medium `Stack.bottom`. Directly specified `kx` and `ky` retain their signs for bottom incidence.

Direct inputs must be finite real numbers. Evanescent incidence with zero incident power cannot normalize diffraction efficiency and is rejected. Complex values are also rejected, with an error pointing to the alternative input. See {ref}`when complex transverse wavevectors are needed <complex-transverse-wavevector>` for the reason and method. `kx=ky=0` means **normal incidence**, perpendicular to the surface. Since its propagation direction does not uniquely determine an incidence plane, the azimuth-zero convention sets **s along y and p along x**. Specify the electric-field direction through `polarization`.

A 1D grating solved for TE or TM alone uses `ky=0`. See the {ref}`polarization section <d-te-tm>` for conditions and inputs.

Supplying `theta`·`phi` together with `kx`·`ky` raises an error (code `source.direction_ambiguous`). To change a source’s input form, clear the opposite pair in the same replacement: `source.replace(theta=None, phi=None, kx=..., ky=...)`. {ref}`Chapter 18 <jobs-and-sweeps>` explains sweeps over wavelength, angle and wavevector.

The calculated order wavevectors are in `result.reflection.k_parallel` and `result.transmission.k_parallel`, on a trailing `(kx, ky)` axis. They are **in the same inverse-length units as the direct inputs**, not divided by `k0`.

(incident-polarization)=
## Specifying polarization

**Polarization describes how the electric-field direction and magnitude change with time at one position.** A tip moving back and forth along a line is linearly polarized; a circular trajectory is circular polarization; an elliptical trajectory is elliptical polarization. Keep propagation direction as specified by `theta`·`phi` or `kx`·`ky`, and use polarization inputs to choose the electric-field oscillation.

Wavebit represents the electric field using **s and p components**. s is perpendicular to the incidence plane; p lies in that plane and is perpendicular to propagation. The trajectories and angle descriptions below assume a **lossless isotropic incident medium**. The diagrams lay out s and p as horizontal and vertical axes.

```{figure} ../../figures/learn/learn/08-polarization-states.svg
:name: light-figure-08-polarization-states
:alt: Electric-field tip trajectories at one position, with s horizontal and p vertical. In-phase (1, 1) forms a line; equal amplitudes with 90-degree phase difference, (1, 1j), form a circle; amplitude ratio 2:1 with 90-degree phase difference, (2, 1j), forms an ellipse. Orange arrows show the initial field.
:figclass: learning-figure

Figure 8-2. Green traces the electric-field tip over one period; orange shows the field at one instant. This is not the path of light through space. All three amplitude pairs are normalized and drawn at the same scale.
```

(s-p)=
### Entering s and p amplitudes and phases directly

In `polarization=(a_s, a_p)`, **the first value is the s electric-field amplitude and the second is the p electric-field amplitude**. Together, these two complex values form a **Jones vector**. Each magnitude sets an oscillation amplitude; their phase difference sets the relative timing of the component oscillations.

| Desired polarization | Input | Relation between components |
| --- | --- | --- |
| Linear s | `polarization="s"` or `(1., 0.)` | Only s oscillates |
| Linear p | `polarization="p"` or `(0., 1.)` | Only p oscillates |
| Linear at 45° between s and p | `polarization=(1., 1.)` | Equal magnitudes and phases |
| Linear along the other diagonal | `polarization=(1., -1.)` | Equal magnitudes, 180° phase difference |
| Circular | `polarization=(1., 1j)` | Equal magnitudes, 90° phase difference |
| Circular with opposite rotation | `polarization=(1., -1j)` | Equal magnitudes, −90° phase difference |
| Elliptical, longer along s | `polarization=(2., 1j)` | s amplitude twice p, 90° phase difference |

Python’s **`1j` is the imaginary unit i**. With `a_s=1`, `a_p=1j`, the s component at one position oscillates proportionally to `cos(ωt)` and p to `sin(ωt)`. Here ω is angular frequency and t is time. When s is maximal, p is zero; a quarter-period later s is zero and p maximal. Combining them gives a circular electric-field-tip trajectory. Wavebit uses the `exp(−iωt)` time convention.

**Using both components does not necessarily produce an ellipse.** At phase difference 0° or 180°, the field moves back and forth on a line. Circular polarization requires both equal magnitudes and a ±90° phase difference. Left- and right-circular names can be confusing because they depend on viewing direction; here the input pair and time-direction arrows specify the rotation.

The input pair is normalized by `sqrt(abs(a_s)**2 + abs(a_p)**2)`. Thus `(1, 1j)` and `(10, 10j)` specify the same incident state; you do not need to divide `(1, 1j)` by `sqrt(2)` yourself. `(0, 0)` cannot be used because it supplies no electric field. **This normalization sets the coefficient magnitudes; it does not specify 1 W of incident power.** Power also depends on medium and incidence direction.

This runnable example illuminates a grating with circularly polarized light from air. Lengths are in nm, and the numbers in the layer's `epsilon` raster are relative permittivities: one row along y, four pixels along x.

```python
import numpy as np
from wavebit import Material, Units, rcwa

def grating(polarization):
    return rcwa.Problem(
        units=Units(length="nm"),
        stack=rcwa.Stack(
            period=(700., 700.),
            top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
            layers=(rcwa.Layer(thickness=300.,
                               epsilon=np.array([[1., 1., 12.25, 12.25]])),)),   # Raster [y, x]
        excitation=rcwa.Excitation(wavelength=900., sources=(
            rcwa.PlaneWave(side="top", theta=np.deg2rad(30.), phi=0.,
                           polarization=polarization),)))

numerics = rcwa.Numerics(orders=(4, 0))      # Orders −4…4 along x, only 0 along y
result = rcwa.run(grating((1., 1j)), numerics=numerics)
print("Reflectance:", result.R.sum())
print("Transmittance:", result.T.sum())
```

Change only the pair passed to `grating` to another one in the table to compare responses at fixed structure and incidence direction. Even s-polarized incidence can produce reflected or transmitted p components if the structure couples the polarizations. Input polarization specifies the incident wave’s state.

(psi-chi)=
### Angle inputs: psi and chi

Two angles, ψ and χ, describe the polarization orientation and ellipse shape. **Both are real angles in radians.** The current `PlaneWave` takes the polarization only as `"s"`, `"p"` or an `(a_s, a_p)` pair, so convert the angles to a pair with the formulas below before passing them.

- **ψ (`psi` in the code below)** is the major-axis angle of the polarization ellipse measured from p toward s. For linear polarization it is the angle of the line along which the field oscillates.
- **χ (`chi` in the code below)** sets the ellipse’s flattening: `abs(tan(χ))` is the minor-to-major radius ratio. Its range is −45° to +45°; changing its sign reverses the field’s rotation.

```{figure} ../../figures/learn/learn/08-polarization-angles.svg
:name: light-figure-08-polarization-angles
:alt: Polarization ellipse with p vertical and s horizontal. psi is measured from p to the major axis. chi determines the radius ratio: 0 degrees gives a line, absolute value 45 degrees gives a circle, and intermediate values give ellipses.
:figclass: learning-figure

Figure 8-3. ψ sets the major-axis direction, while χ sets the radius ratio and rotation direction. For a circle, the two radii are equal; changing ψ preserves the trajectory shape and rotation direction.
```

| Polarization | Angles (ψ, χ) | Corresponding amplitude pair |
| --- | --- | --- |
| Linear p | `0, 0` | `(0, 1)` |
| Linear s | `π/2, 0` | `(1, 0)` |
| Linear at 45° | `π/4, 0` | `(1, 1)` |
| Circular | `0, −π/4` | Same polarization as `(1, 1j)` |
| Circular with opposite rotation | `0, π/4` | Same polarization as `(1, -1j)` |

χ = 0 gives linear polarization. Omitting `polarization` selects linear s. The two representations in each circular-polarization row need not have the same common phase. Adding the same phase to both components preserves the polarization ellipse, reflectance and transmittance but can change the phase of complex field amplitudes.

Using the same `grating` above, enter linear polarization tilted 30° from p toward s by converting the two angles to a pair.

```python
psi, chi = np.deg2rad(30.), 0.
pair = (np.sin(psi)*np.cos(chi) + 1j*np.cos(psi)*np.sin(chi),    # a_s
        np.cos(psi)*np.cos(chi) - 1j*np.sin(psi)*np.sin(chi))    # a_p
result = rcwa.run(grating(pair), numerics=numerics)
```

With χ = 0 this pair is `(0.5, np.sqrt(3)/2)`, and `grating((0.5, np.sqrt(3)/2))` gives the same result.

:::{admonition} Earlier interface
:class: earlier-interface

The earlier interface (retired with `wavebit.compat` on 2026-10-15) also took the two angles directly, as `psi=` and `chi=` keywords of its constructor. Omitting `chi` set it to 0, supplying `psi`·`chi` together with `polarization` raised an error, and the conversion was the one in the dropdown below. The current `PlaneWave` has no angle form yet.
:::

````{dropdown} Formulas: obtaining two amplitudes from psi and chi
The two electric-field coefficients are as follows.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>a<sub>s</sub> = sin ψ cos χ + i cos ψ sin χ</p>
<p>a<sub>p</sub> = cos ψ cos χ − i sin ψ sin χ</p>
</div>
```

When χ=0, `a_s=sin(ψ)` and `a_p=cos(ψ)`. This is why ψ=0 gives p and ψ=π/2 gives s. Their squared magnitudes already sum to 1, so the normalization `PlaneWave` applies to a pair leaves them unchanged.
````

```{raw} html
<section class="exercise polarization-exercise" id="polarization-explorer" aria-labelledby="pol-title">
<h3 id="pol-title">Try it: choose polarization with psi and chi</h3>
<p class="concept">A lossless isotropic incident medium. View the electric-field tip at one position in the s·p plane.</p>
<div class="controls">
<label for="pol-psi">Major-axis angle ψ <output id="pol-psi-out" for="pol-psi">30°</output>
<input id="pol-psi" type="range" min="0" max="180" step="1" value="30">
<span class="control-hint">Measured from p toward s.</span></label>
<label for="pol-chi">Ellipse shape χ <output id="pol-chi-out" for="pol-chi">20°</output>
<input id="pol-chi" type="range" min="-45" max="45" step="1" value="20">
<span class="control-hint">0°: line · ±45°: circle · sign: rotation direction</span></label>
</div>
<svg viewBox="0 0 400 320" role="img" aria-label="Electric-field-tip trajectory specified by psi and chi">
<defs>
<marker id="pol-motion-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#117c77"/></marker>
<marker id="pol-field-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#b56626"/></marker>
</defs>
<line x1="45" y1="150" x2="355" y2="150" stroke="#abbec5"/>
<line x1="200" y1="15" x2="200" y2="280" stroke="#abbec5"/>
<text x="358" y="155">s</text><text x="210" y="28">p</text>
<polyline id="pol-curve" points="" fill="none" stroke="#117c77" stroke-width="3"/>
<polyline id="pol-motion" points="" fill="none" stroke="#117c77" stroke-width="3" marker-end="url(#pol-motion-arrow)"/>
<line id="pol-vector" x1="200" y1="150" x2="200" y2="150" stroke="#b56626" stroke-width="3" marker-end="url(#pol-field-arrow)"/>
<circle cx="200" cy="150" r="3" fill="#354b60"/>
<text x="30" y="303">Green: one period · Orange: electric field at t = 0</text>
</svg>
<div class="readout" id="pol-readout" aria-live="polite" aria-atomic="true">
<strong id="pol-state">Elliptical polarization</strong>
<p id="pol-shape-note">Minor radius / major radius = 0.364</p>
<p>Angles<code id="pol-angle-code">ψ=30°, χ=20°</code></p>
<p>Amplitude-pair input (rounded)<code id="pol-pair-code">polarization=(0.4698+0.2962j, 0.8138-0.1710j)</code></p>
</div>
<p class="explain">Pass the amplitude pair as rcwa.PlaneWave(polarization=...). Sliders use degrees (°); the angle line converts them to radians with np.deg2rad().</p>
</section>
```

First set χ=0° and vary ψ to rotate a linear polarization. Moving χ toward ±45° changes the line through an ellipse into a circle. Reversing the sign of χ reverses the small green time-direction arrow. For circular polarization, changing ψ preserves the circle but changes the initial electric-field direction shown in orange.

(d-te-tm)=
### Computing only TE or TM for a 1D structure

The two polarizations do not mix **when the structure is invariant in y, incidence lies in the xz plane, and the material does not couple TE and TM**. TE has its electric field along y; TM has its magnetic field along y and electric field in xz. Under these conditions, TE corresponds to s and TM to p.

The `grating` above is an isotropic grating invariant in y with `phi=0`, so s and p incidence give the TE and TM responses.

```python
te_result = rcwa.run(grating("s"), numerics=numerics)   # TE
tm_result = rcwa.run(grating("p"), numerics=numerics)   # TM
```

These run the general (conical) formulation, which also handles 2D patterns and nonzero `ky`. On a one-row raster grating at `phi=0` its amplitudes and efficiencies match a dedicated TE/TM solve to 5e-15 (measured 2026-10-15); it does not have that solve's speed.

:::{admonition} Earlier interface
:class: earlier-interface

The earlier interface (retired with `wavebit.compat` on 2026-10-15) had a dedicated declaration for this case: `tetm="te"` or `tetm="tm"` solved TE or TM alone, 2 to 4 times faster than the conical formulation. It required `ky` to be zero (omitted, or explicitly `ky=0.`); nonzero `ky` raised an error. Because it declared a problem with no polarization degree of freedom, it could not be combined with `psi`, `chi` or `polarization`. The current API has no `tetm` declaration yet.

```python
# Earlier interface only; runs on a historical checkout.
te_result = wavebit.rcwa(**scene, tetm="te").solve()
tm_result = wavebit.rcwa(**scene, tetm="tm").solve()
```
:::

(incident-from-bottom)=
## Choosing incidence from above or below

For one-sided illumination, `PlaneWave(side=...)` selects the incoming side. Its default is `"top"`.

| `side` | Incident medium and direction | Reflectance `R` | Transmittance `T` |
| --- | --- | --- | --- |
| `"top"` | Downward (−z) from upper medium `Stack.top` | Light returning upward | Light leaving downward |
| `"bottom"` | Upward (+z) from lower medium `Stack.bottom` | Light returning downward | Light leaving upward |

`side="bottom"` launches light upward (+z) from the lower half-space `Stack.bottom`. At the bottom interface `z=0`, it first meets the lowest layer, `layers[0]`. List `layers` from bottom layer to top layer, each `rcwa.Layer` with its own `thickness`; `top` remains the upper medium and `bottom` the lower medium.

Reflectance `R` is the power fraction returning into the lower medium `bottom`; transmittance `T` is the fraction leaving into the upper medium `top`. Both are normalized by the power incident from below.

(incidence-side-angles)=
(thetaphipsichi)=
### Angles and polarization follow the same rules on both sides

**`theta`, `phi` and the polarization angles ψ and χ follow the same conventions for top and bottom incidence.** Select the side with `PlaneWave(side=...)`, then read the angles as follows. This discussion assumes a lossless isotropic incident medium.

| Quantity | Meaning |
| --- | --- |
| `theta` | Angle between propagation and the inward surface normal: −z for top incidence, +z for bottom incidence. Zero degrees is normal incidence on either side. |
| `phi` | Azimuth of the propagation vector projected into xy, measured from +x toward +y. Both sides use the same fixed x·y axes. |
| ψ | Major-axis angle of the polarization ellipse measured from that incident wave’s p direction toward its s direction. |
| χ | Angle setting the ellipse shape and relative phase of s/p components. The same conversion to an `(a_s, a_p)` pair is used on both sides. |

For example, at `theta=30°`, `phi=0`, **both waves arrive from the −x side and travel toward increasing x**. Relative to the point of incidence on the surface:

| Incident side | Where the light comes from | Direction toward the surface |
| --- | --- | --- |
| `bottom` | Below, on the −x side | Upward along +x·+z |
| `top` | Above, on the −x side | Downward along +x·−z |

Both propagation vectors have positive x components; their z components have opposite signs. The top-incidence line extending upward toward −x shows where the light comes from, and its arrow points along the line toward the surface.

Polarization is specified in each incident wave’s own s·p directions. For linear polarization, χ=0: ψ=0 gives p, ψ=90° gives s, and ψ=45° gives equal, in-phase s and p components, the pair `(1, 1)`. `abs(χ)=45°` gives circular polarization. Use these rules directly for either incident side.

(light-detail-id11)=
### Example: linearly polarized light incident upward from glass

Here light enters upward from glass, `bottom=Material(epsilon=2.25)`, toward two layers. With `theta=30°`, `phi=0`, it approaches the surface from below on the −x side; its direction in glass makes 30° with the +z normal. It is linearly polarized with ψ=45°, χ=0°, which is the pair `(1., 1.)`. The code supplies angles in radians.

```python
import numpy as np
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(
    units=Units(length="nm"),                          # Length unit: nm
    stack=rcwa.Stack(
        period=(1500., 1500.),
        top=Material(epsilon=1.), bottom=Material(epsilon=2.25),   # Air above, glass below
        layers=(
            rcwa.Layer(thickness=100., background=Material(epsilon=4.)),   # Lowest layer: z=0–100 nm
            rcwa.Layer(thickness=200., background=Material(epsilon=9.)),   # Layer above: z=100–300 nm
        )),
    excitation=rcwa.Excitation(wavelength=600., sources=(
        rcwa.PlaneWave(side="bottom",                  # Upward incidence from glass
                       theta=np.deg2rad(30.), phi=0.,  # kx > 0, kz > 0
                       polarization=(1., 1.)),)))      # Equal, in-phase s and p
result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(0, 0)))
print("Reflectance returning into glass:", result.R.sum())
print("Transmittance leaving into air:", result.T.sum())
```

Wavevectors for angle-defined incidence are calculated from the incident medium’s ε and μ and the wavelength. Here those are the lower glass values. If the two incident media differ, the same angle and wavelength can give different wavevector magnitudes.

(absorbing-incidence)=
## Wavevectors in lossy incident and transmitted media

Of the half-spaces above and below the structure, the one supplying light is the **incident medium** and the one receiving transmitted light is the **transmitted medium**. Either can be a lossy, absorbing material.

This section considers isotropic exterior media and real surface-parallel wavevectors `kx`, `ky`. Wavevector calculations use **both real and imaginary parts of each medium’s permittivity ε and permeability μ**. In a lossy medium, the surface-normal wavevector `kz` is complex; its imaginary part describes attenuation normal to the surface.

| Incidence setting | Incident medium | Transmitted medium |
| --- | --- | --- |
| `PlaneWave(side="top")` | Upper `top` | Lower `bottom` |
| `PlaneWave(side="bottom")` | Lower `bottom` | Upper `top` |

**First determine the incident `kx`, `ky`, then calculate `kz` separately in each medium.** Angle inputs determine the incident transverse wavevector from ε and μ of the incident medium and `theta`, `phi`. Direct inputs use the supplied `kx`, `ky`. The transmitted direction follows from this transverse wavevector and ε and μ of the transmitted medium.

(light-detail-id4)=
### When the incident medium is lossy

The incidence condition here gives **the same incident intensity at every position along the surface**. To retain this condition in an absorbing incident medium, `kx` and `ky` are real. The surface-normal component `kz` can have an imaginary part, so **the full vector k is complex**.

`Re` and `Im` denote real and imaginary parts. Collecting them separately for all three components gives:

```{raw} html
<div class="power-equations" style="text-align:center">
<p><b>k</b> = Re(<b>k</b>) + i Im(<b>k</b>)</p>
<p>Re(<b>k</b>) = (k<sub>x</sub>, k<sub>y</sub>, Re(k<sub>z</sub>))</p>
<p>Im(<b>k</b>) = (0, 0, Im(k<sub>z</sub>))</p>
</div>
```

- **Re(k) sets the direction of phase propagation.** A surface joining equal-phase points is a wavefront, and Re(k) is normal to it. Nonzero `kx` or `ky` makes phase propagation oblique to the surface.
- **Im(k) sets the direction and rate of spatial amplitude attenuation.** Here its x and y components are zero, so moving sideways at fixed height does not change amplitude. Its z component causes amplitude and intensity to decrease toward the structure.

Thus **phase can propagate obliquely while intensity varies only with height**. The two panels below show intensity and phase of that same incident wave separately.

```{figure} ../../figures/learn/learn/08-absorbing-wave.svg
:name: light-figure-08-absorbing-wave
:alt: Two panels separate intensity and phase of one incident wave. Left: shading shows equal intensity at equal height and weaker intensity downward. Right: tilted wavefronts and their normal Re(k), the phase-propagation direction. theta is measured between Re(k) and the downward normal.
:figclass: learning-figure

Figure 8-4. A and B on the left have equal intensity; the lower C and D also match each other but are weaker than A and B. On the right, thin lines are wavefronts and the thick arrow is Re(k), the phase-propagation direction.
```

**Equal intensity does not mean equal instantaneous electric-field values.** A and B at the same height can have different oscillation timing if their phases differ. Intensity here means time-averaged power flow calculated from electric and magnetic fields.

The diagram’s `theta` is **the angle between Re(k) and the inward normal**. For top incidence, it is measured from −z. Power-flow direction is computed from electric and magnetic fields and can differ from Re(k) in a lossy medium.

**Wavebit accepts only real direct `kx`, `ky` inputs.** A nonzero imaginary part raises an error advising real `theta` for absorbing incidence. The description and diagram above concern this surface-uniform incident intensity. Cases requiring complex wavevectors are explained {ref}`below <complex-transverse-wavevector>`.

:::{dropdown} Formulas: separating phase and attenuation
For position **r = (x, y, z)**, time t and angular frequency ω, the plane wave’s space-time factor separates as follows.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>exp[i(<b>k</b> · <b>r</b> − ωt)]</p>
<p>= exp[−Im(<b>k</b>) · <b>r</b>] × exp[i(Re(<b>k</b>) · <b>r</b> − ωt)]</p>
</div>
```

The first factor sets amplitude; the second sets oscillation phase. For real `kx` and `ky`, their position dependence is:

```{raw} html
<div class="power-equations" style="text-align:center">
<p>Amplitude factor = exp[−Im(k<sub>z</sub>) z]</p>
<p>Phase = k<sub>x</sub>x + k<sub>y</sub>y + Re(k<sub>z</sub>) z − ωt</p>
</div>
```

**Amplitude depends only on z, but phase can depend on x, y and z.** This permits uniform intensity at fixed height alongside oblique phase propagation.

In the diagram, **the incident wave weakens as it travels downward**. Let `d > 0` be distance traveled vertically downward and `κ > 0` the normal amplitude-attenuation coefficient, with inverse-length units. Then:

```{raw} html
<div class="power-equations" style="text-align:center">
<p>Amplitude after displacement = amplitude before displacement × exp(−κd)</p>
</div>
```

Since `exp(−κd)` is less than 1, amplitude decreases. Greater downward depth d gives stronger attenuation.

Signs enter when expressing this in Wavebit’s z coordinates. **Since +z points upward, a downward displacement d gives `Δz = −d`.** Attenuation is downward too, so `Im(kz) = −κ`. Comparing the amplitude factor at the two positions gives the same result.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>Amplitude after displacement / amplitude before displacement</p>
<p>= exp[−Im(k<sub>z</sub>) Δz] = exp[−(−κ)(−d)] = exp(−κd)</p>
</div>
```

This describes **the amplitude variation of the incident wave alone**. With reflection present, add incident and reflected electric fields to obtain the total field.
:::

Reflection and transmission efficiencies are normalized by **the power of the incident wave alone at the incident-side interface**. See {ref}`Chapter 13 <half-space-power>` for reading results with this power normalized to 1.

For a lossless isotropic incident medium with positive index n, transverse wavevectors follow from `theta` and `phi` as below. Here n is the incident index and k₀ = 2π/λ is the vacuum wavevector magnitude.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>k<sub>x</sub> = k<sub>0</sub> n sin(θ) cos(φ)</p>
<p>k<sub>y</sub> = k<sub>0</sub> n sin(θ) sin(φ)</p>
</div>
```

For a lossy incident medium, wavevectors are chosen to **preserve the phase-propagation direction specified by `theta`, `phi` while keeping incident intensity uniform at fixed height**. Thus `kx`, `ky` are real while `kz` may be complex. All three must also satisfy the following relation set by relative permittivity ε and permeability μ.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>k<sub>x</sub>² + k<sub>y</sub>² + k<sub>z</sub>² = k<sub>0</sub>² ε<sub>r</sub> μ<sub>r</sub></p>
</div>
```

For both lossless and absorbing media, **reflection and transmission efficiencies require `theta` strictly between −90° and 90°**. In radians, `abs(theta) < π/2`. Zero degrees means normal phase propagation; approaching either endpoint makes propagation increasingly parallel to the surface.

The reason for excluding exactly ±90° differs between the two cases.

- **Lossless medium:** the wavevector remains finite but `kz=0`, so light travels parallel to the surface. Incident power through the surface is zero, making reflection and transmission efficiencies normalized by it undefined.
- **Absorbing medium:** surface-parallel phase propagation at θ=±90° requires `Re(kz)=0`. Under the absorbing incidence conditions considered here, however, the product `Re(kz) Im(kz)` must be nonzero. These conditions cannot both hold, as the imaginary part of the wavevector relation below shows.

:::{dropdown} Calculation: real kx and ky from complex material values
Start from the medium’s wavevector relation above. Real `kx`, `ky` have real squares, while the imaginary part of `kz²` is `2 Re(kz) Im(kz)`. Equating imaginary parts gives:

```{raw} html
<div class="power-equations" style="text-align:center">
<p>k<sub>x</sub>² + k<sub>y</sub>² + k<sub>z</sub>² = k<sub>0</sub>² ε<sub>r</sub> μ<sub>r</sub></p>
<p>2 Re(k<sub>z</sub>) Im(k<sub>z</sub>) = k<sub>0</sub>² Im(ε<sub>r</sub> μ<sub>r</sub>)</p>
</div>
```

**If ε μ has an imaginary part, the right side is nonzero.** At θ=±90°, phase propagation has no normal component, so `Re(kz)=0`. For finite `Im(kz)`, the left side is then zero and cannot satisfy the equation. This prevents exactly ±90° incidence for an absorbing wave with surface-uniform intensity.

To calculate wavevectors from material values and angles, first **split the product of relative permittivity and permeability into real part a and imaginary part b**.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>ε<sub>r</sub> μ<sub>r</sub> = a + ib</p>
<p>a = Re(ε<sub>r</sub> μ<sub>r</sub>), &nbsp; b = Im(ε<sub>r</sub> μ<sub>r</sub>)</p>
</div>
```

For example, `ε=4+3j`, `μ=1` gives `a=4`, `b=3`. At vacuum wavelength λ, `k0=2π/λ`.

Next define N as **the magnitude of the real phase-propagation vector divided by the vacuum wavevector magnitude**. Use real `kx` and `ky` directly and only the real part of complex `kz`.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>N = √[k<sub>x</sub>² + k<sub>y</sub>² + (Re(k<sub>z</sub>))²] / k<sub>0</sub></p>
</div>
```

**N is a real, dimensionless number.** For N=2, for example, the phase repeats over distance λ/2 along the phase-propagation direction. The attenuating component `Im(kz)` is excluded from this vector magnitude.

Calculate N from material values and phase angle θ, then determine `kx`, `ky` from incidence direction θ·φ.

```text
N² = (a + sqrt(a² + b² / cos²(theta))) / 2

kx = k0 s N sin(theta) cos(phi)
ky = k0 s N sin(theta) sin(phi)
```

`s` is the sign of the real part of the incident medium’s signed index `n = √ε·√μ`. Ordinary and nonmagnetic media have `s = +1`, recovering the previous expression. Only negative-index (negative-phase-velocity) media have `s = −1`. Since phase runs against energy flow there, the sign is reversed so θ consistently describes **the direction in which light (energy) enters the structure** (since 2026-10-05; [negative-index media](../../theory/rcwa/negative-index-media.md), Section 7).

The formula enforces `kx² + ky² + kz² = k0² εr μr` while keeping `kx`, `ky` real. In a lossy medium, `N` also depends on angle and is distinct from complex refractive index `n`. The `kz` direction is selected toward −z for top incidence and +z for bottom incidence.

For a lossless positive-index medium, `b=0` and `a=n²>0`, so `N=n`. At normal incidence both transverse components vanish. See [coordinate and sign conventions](../../reference/common/conventions.md#real-angle-incidence) for precise definitions and the separate meaning of complex-angle input.
:::

(absorbing-incidence-example)=
#### Example: a 30° phase angle inside a lossy incident medium

In this example, `theta=30°` is **the phase angle within the medium `top=Material(epsilon=4+3j)`**. All lengths use nm. The finite layer and `bottom` use the same lossless material; we inspect the incident transverse wavevector.

```python
import numpy as np
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(
    units=Units(length="nm"),
    stack=rcwa.Stack(
        period=(1200., 1200.),
        top=Material(epsilon=4. + 3j), bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(thickness=100., background=Material(epsilon=2.25)),)),
    excitation=rcwa.Excitation(wavelength=1000., sources=(
        rcwa.PlaneWave(side="top", theta=np.deg2rad(30.), phi=0.,
                       polarization="s"),)))                       # s incidence
result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(0, 0)))   # Uniform layer: order 0 suffices

kx, ky = result.reflection.k_parallel[0, 0, 0]   # wavelength, order_v, order_u; nm⁻¹
print(kx, ky)
```

`k_parallel` holds each order's physical transverse wavevector, not divided by `k0`. Only order 0 is used here, so indexing the first wavelength and order leaves the `(kx, ky)` pair: `kx ≈ 0.00677139 nm⁻¹`, `ky = 0 nm⁻¹`. **The 30° angle is defined inside the selected lossy incident medium.** See {ref}`Chapter 13 <half-space-power>` for reading the resulting powers.

This also shows **why the lossless-medium formula cannot be used unchanged**. The complex index of the incident medium `ε=4+3j` is `n = 2.1213 + 0.7071i`. Substituting it into the lossless formula `kx = k0 n sin(θ)` gives complex `kx/k0 = 1.0607 + 0.3536i`, violating the required real `kx`. Instead, interpret 30° as the angle of Re(k) and use:

```{raw} html
<div class="power-equations" style="text-align:center">
<p>tan(θ) = k<sub>x</sub> / |Re(k<sub>z</sub>)|</p>
</div>
```

Finding `kx` this way requires `Re(kz)`, but evaluating `kz` from `kz² = k0² εr μr − kx²` requires `kx` again. **The two conditions are coupled and must be solved together.** Varying `kx` and checking both gives:

| `kx/k0` | Magnitude of `Re(kz)` from the wavevector relation / `k0` | Phase-propagation angle |
| ---: | ---: | ---: |
| 1.0 | 1.9030 | 27.72° |
| 1.1 | 1.8556 | 30.66° |
| 1.077700 | 1.866632 | 30.00° |

The formula `N² = (a + sqrt(a² + b² / cos²(theta))) / 2` solves this trial-and-error process in closed form. With `a=4`, `b=3`, θ=30°, it gives `N = 2.155400`, `kx/k0 = N sin(30°) = 1.077700`, and `|Re(kz)|/k0 = N cos(30°) = 1.866632`, matching the last table row. Using only the real part of n instead, `kx/k0 = Re(n) sin(30°) = 1.0607`, yields a phase angle of 29.50°, slightly different from 30°.

(complex-transverse-wavevector)=
#### When complex transverse wavevectors are needed: complex theta

Direct `kx`, `ky` inputs reject complex values. They represent **illumination uniform along the surface**, whereas a complex transverse wavevector makes intensity vary along that surface. This remains true for an absorbing incident medium, whose loss enters through `Im(kz)`. For absorbing incidence, **specify real `theta`**, as above, and Wavebit calculates real `kx`, `ky`.

The rejection message states these points directly.

- Direct `kx`, `ky` must be real because they specify uniform illumination.
- Use real `theta`, `phi` for absorbing incidence.
- Complex transverse wavevectors can be specified only through complex `theta`, `phi`; the message also provides the inverse conversion.

To deliberately use complex transverse wavevectors, supply complex `theta`, `phi`. These angles are **formal parameters**, not physical phase-propagation directions. The full complex incident index n is used in:

```{raw} html
<div class="power-equations" style="text-align:center">
<p>k<sub>x</sub> / k<sub>0</sub> = n sin(θ) cos(φ), &nbsp; k<sub>y</sub> / k<sub>0</sub> = n sin(θ) sin(φ)</p>
</div>
```

To invert a desired `kx` when `ky = 0`, set `theta = arcsin(kx / (k0 n))`, `phi = 0`.

If this produces an imaginary transverse component, the interpretation of results changes.

- **Intensity varies along the surface.** For `top=Material(epsilon=4+3j)` at vacuum wavelength 1,000 nm, `kx/k0 = n sin(30°) = 1.0607 + 0.3536i` reduces intensity to about 1/207 over one 1200 nm period.
- **There is no incident power to divide by.** The uniform-intensity power normalization does not hold, so `R`, `T`, `power_entering`, `absorption` and the port powers are `NaN`, and `result.reflection.metadata['power_defined']` is false. The complex amplitudes `amplitude_s`, `amplitude_p` and `amplitude_xyz` are returned. {ref}`Chapter 13 <half-space-power>` explains the normalization.
- **Each run emits one `ComplexIncidenceWarning`.** Because this has its own warning category, an intentional calculation can silence just this warning with `warnings.filterwarnings('ignore', category=wavebit.rcwa.diagnostics.ComplexIncidenceWarning)`.
- **Other solvers may give different answers.** Choosing between the two `kz` roots for each exterior-medium order becomes a convention, so Wavebit, fmmax and RETICOLO can return different reflection coefficients for the same input. {ref}`Appendix A <complex-kx-roots>` explains these rules in lossless media. {ref}`Appendix B <complex-kx-lossy-gain>` explains why real and complex `theta` yield different `kx` in an absorbing incident medium, and what changes with absorbing or gain half-spaces.

:::{admonition} Earlier interface
:class: earlier-interface

The earlier interface (retired with `wavebit.compat` on 2026-10-15) returned formal `R` and `T` ratios here instead of `NaN`. They could sum to more than 1 even for a lossless structure: in the Chapter 13 example, changing only `top` to lossless `4` and setting `theta = 30° + 0.2i` gave `T = 1.379`. Its `power_entering` and `absorption` were `NaN`.
:::

For a complex angle of the form `theta = π/2 + iα`, as used for evanescent incidence, `sin(θ) = cosh(α)` is real. If the incident medium is lossless, the normalized wavevector `kx/k0 = n cosh(α)` also remains real, so no `ComplexIncidenceWarning` is emitted. The incident wave itself is then evanescent: it carries no power toward the surface, so there is nothing to divide the reflected and transmitted power by. `R`, `T`, `incident_power`, `power_entering` and `absorption` are therefore `NaN`, with no warning, and `result.reflection.metadata['efficiency_defined']` is false. `power_defined` stays true: the port powers in W are returned, because light that tunnels through to the far side carries real power. The amplitudes such as `result.reflection.amplitude_s` are returned as usual. For an absorbing incident medium, n is complex and so is `kx`, triggering the warning.

(absorbing-transmission)=
### When the transmitted medium is lossy

**The same diffraction order retains its `kx`, `ky` across the interface into the transmitted medium.** Matching electric and magnetic fields at the interface requires matching phase variation along the surface. The zeroth order has the incident transverse wavevector; other orders add the reciprocal-lattice wavevectors from [Chapter 6](lattice.md). Each order therefore has the same transverse wavevector on the incident and transmitted sides.

Each transmitted order satisfies the following relation. The subscript t denotes the transmitted medium; ε and μ are relative permittivity and permeability. `kx`, `ky` are that order’s physical wavevector components.

```{raw} html
<div class="power-equations">
<p>k<sub>z,t</sub>² = k<sub>0</sub>² ε<sub>r,t</sub> μ<sub>r,t</sub> − k<sub>x</sub>² − k<sub>y</sub>²</p>
<p>k<sub>0</sub> = 2π / λ</p>
</div>
```

There are two signs for `kz`. In an absorbing transmitted medium, choose **the solution decaying away from the structure**. Wavebit’s +z points upward, so transmission into lower medium `bottom` decays toward decreasing z, while transmission into upper medium `top` decays toward increasing z.

Thus real `kx`, `ky` remain valid with a lossy transmitted medium. Loss appears as attenuation from the imaginary part of complex `kz`. **`theta`, `phi` specify incidence; they are not separate transmitted-angle inputs.** Read the transmitted phase direction from Re(k), recognizing that it can differ from power-flow direction.

(light-detail-id7)=
#### Example: transmission from air into a lossy medium

Light enters from air, `top=Material(epsilon=1.)`, at phase angle 30°, with lossy `bottom=Material(epsilon=4+3j)` below. Both half-spaces have μ=1. The finite layer is also air, matching the incident medium. Wavelength and all lengths use nm.

```python
import numpy as np
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(
    units=Units(length="nm"),
    stack=rcwa.Stack(
        period=(1200., 1200.),
        top=Material(epsilon=1.), bottom=Material(epsilon=4. + 3j),
        layers=(rcwa.Layer(thickness=100., background=Material(epsilon=1.)),)),   # Finite air layer
    excitation=rcwa.Excitation(wavelength=1000., sources=(
        rcwa.PlaneWave(side="top", theta=np.deg2rad(30.), phi=0.,
                       polarization="s"),)))                       # s incidence
result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(0, 0)))   # Uniform layer: order 0 suffices

k0 = 2 * np.pi / 1000.
kx, ky = result.transmission.k_parallel[0, 0, 0]  # nm⁻¹, the same as the incident wave's

# Here μ=1. Choose kz for a wave outgoing and decaying downward.
kz_transmitted = -np.sqrt(k0**2 * (4. + 3j) - kx**2 - ky**2)
print(kx, ky, kz_transmitted)
print(result.R.sum(), result.T.sum())
```

| Calculated quantity | Result |
| --- | --- |
| `kx` of the incident and zeroth transmitted waves | Approximately 0.00314159 nm⁻¹ |
| `ky` of the incident and zeroth transmitted waves | 0 nm⁻¹ |
| Transmitted `kz` | Approximately −0.01299294 − 0.00455768i nm⁻¹ |
| Angle between transmitted phase direction and downward normal (−z) | Approximately 13.59° |
| Interface reflectance `R.sum()` | Approximately 0.215748 |
| Interface transmittance `T.sum()` | Approximately 0.784252 |

The negative real part of `kz` indicates downward phase propagation. Its imaginary part is also negative, but z decreases downward, so the magnitude of `exp(i kz z)` decreases. **A negative imaginary part does not mean amplification in this example.**

`T` is the power fraction crossing the finite structure’s lower interface, `z=0`, into the lossy medium. The transmitted wave continues to attenuate afterward. At normal depth `d` beyond the interface, this single wave’s normal power-flow magnitude is `exp(−2 |Im(kz)| d)` times its interface value. For `d=100 nm` here, the factor is approximately 0.4019.

The incident medium and finite layer are both lossless, so `R+T=1` in this example. Subsequent absorption inside the semi-infinite transmitted medium is not part of the finite layers’ `result.absorption`, which is `result.outputs['power_entering']` minus the total `T`. {ref}`Chapter 13 <half-space-power>` explains reference surfaces and absorption scope. If both incident and transmitted media are lossy, the sequence is unchanged: determine transverse wavevectors from incidence, then calculate normal wavevectors using each medium’s ε and μ.

For uniform multilayers, transverse-wavevector continuity and solution selection in absorbing exterior media are also discussed in [Byrnes’s multilayer tutorial, Sections 2 and 5](https://arxiv.org/html/1603.02720v5#S2). Its z direction differs from this example, so interpret wavevector signs using the respective coordinate conventions.

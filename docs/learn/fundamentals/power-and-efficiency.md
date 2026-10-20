(power-and-efficiency)=
# 13. Power flow and diffraction efficiency

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

Reflection and transmission efficiencies are **power ratios**. Squaring a complex electric-field amplitude coefficient gives a ratio of squared electric-field magnitudes; converting it to a power ratio also requires the magnetic field and propagation direction. At a boundary where incident and reflected waves overlap, calculate net entering power from their combined total fields.

This chapter first defines power and its reference planes, then derives reflection and transmission efficiencies and checks them with normal incidence at an interface between two media. It then includes incident–reflected interference to obtain net entering power and absorption of the entire layer stack. We examine **calculating each wave's power separately** and **combining the waves to calculate net boundary power** in that order.

We consider **illumination from one side, isotropic upper and lower half-spaces, and real transverse wavevectors `kx`, `ky`**. [Chapter 9](multiple-beams.md) covers illumination from both sides and superposition of waves with different transverse wavevectors.

## Time-averaged power from electric and magnetic fields

### Power density S and power P

Light's electric and magnetic fields oscillate in time. At one position, denote their values at time {math}`t` by {math}`\mathbf e(t)` and {math}`\mathbf h(t)`. Their cross product is the **Poynting vector** {math}`\mathbf S(t)`, describing instantaneous power flow per unit area and its direction.

```{math}
\mathbf S(t)=\mathbf e(t)\times\mathbf h(t).
```

For fields oscillating at one frequency, it is convenient to use **complex amplitudes carrying oscillation magnitude and phase**. Denoting the complex electric and magnetic amplitudes by {math}`\mathbf E`, {math}`\mathbf H`, the physical time-dependent fields are written as follows. Here {math}`\omega` is the angular frequency of the light.

```{math}
\mathbf e(t)=\operatorname{Re}[\mathbf E e^{-i\omega t}],
\qquad
\mathbf h(t)=\operatorname{Re}[\mathbf H e^{-i\omega t}].
```

Multiplying uppercase {math}`\mathbf E`, {math}`\mathbf H` by the time dependence {math}`e^{-i\omega t}` and taking the real part gives the physical lowercase fields {math}`\mathbf e(t)`, {math}`\mathbf h(t)`. **Complex amplitudes represent phase even in lossless media.**

Instantaneous power flow {math}`\mathbf S(t)` also varies in time. Reflection and transmission use its average over one period, calculated from complex amplitudes as follows.

```{math}
\langle\mathbf S\rangle
=\frac12\operatorname{Re}[\mathbf E\times\mathbf H^*].
```

The asterisk {math}`{}^*` denotes complex conjugation, and brackets {math}`\langle\;\rangle` denote time averaging. The magnetic field {math}`\mathbf H` in this equation is the physical field in A/m. Wavebit's returned field array contains {math}`Z_0\mathbf H`, so divide it by vacuum impedance {math}`Z_0` before direct substitution. {ref}`Chapter 14 <magnetic-field-units>` explains array units and conversion.

Power through a surface is the integral of the power density's **component normal to that surface**. With unit normal {math}`\hat{\mathbf n}`:

```{math}
P=\int_{\text{cell cross-section}}
\langle\mathbf S\rangle\cdot\hat{\mathbf n}\,dA.
```

| Quantity | Unit | Meaning |
| --- | --- | --- |
| {math}`\langle\mathbf S\rangle` | W/m² | Time-averaged power flow per unit area and its direction |
| {math}`P` | W | Power through a unit-cell cross section |
| {math}`R`, {math}`T` | Dimensionless | Reflected/transmitted power divided by incident power |

These units apply when fields have physical amplitudes. For efficiencies alone, use the same amplitude reference in numerator and denominator. When comparing waves with uniform power density along the surface over the same area, area cancels, so power ratios equal normal power-density ratios. If density varies with position, integrate over the surface first. For an infinite periodic structure, use **power in one unit cell** rather than the entire infinite area.

### How the E–H phase difference affects power

Consider one plane wave with electric field along x and magnetic field along y. Writing their components at one position as {math}`e(t)`, {math}`h(t)`, instantaneous +z power density is {math}`S_z(t)=e(t)h(t)`.

Choose the time origin so that the electric field at this position oscillates as follows. {math}`E_0` is the positive real electric-field amplitude magnitude.

```{math}
e(t)=E_0\cos\omega t,
\qquad E=E_0.
```

The magnetic field oscillates at the same frequency but may differ in phase. Here {math}`E`, {math}`H` are the **complex amplitudes** of the electric x and magnetic y components. Each is a complex scalar representing one component, not the full vector; write their ratio as follows.

```{math}
Y=\frac{H}{E}=|Y|e^{i\delta},
\qquad H=YE=|Y|E_0e^{i\delta}.
```

{math}`Y` is the **wave admittance** relating the chosen electric and magnetic components. {math}`|Y|=|H|/|E|` is their amplitude-magnitude ratio, and {math}`\delta=\arg H-\arg E` their phase difference. This ratio depends on medium, propagation direction, and polarization and follows from Maxwell's equations. With the time convention {math}`e^{-i\omega t}`, the physical magnetic field is:

```{math}
\begin{aligned}
h(t)
&=\operatorname{Re}[H e^{-i\omega t}]\\
&=\operatorname{Re}[|Y|E_0e^{i(\delta-\omega t)}]\\
&=|Y|E_0\cos(\delta-\omega t)\\
&=|Y|E_0\cos(\omega t-\delta).
\end{aligned}
```

```{figure} ../../figures/learn/learn/12-time-average.svg
:alt: Time signals and their products are compared for E–H phase differences of 0 and 45 degrees. The normalized product averages to 0.5 and approximately 0.354 respectively.
:figclass: learning-figure

Figure 13-1. Time averaging with different electric–magnetic phase differences. Each signal is divided by its own amplitude; the lower plots show the products of the normalized signals. These are explanatory plots, not results for a particular structure.
```

#### Multiplying the signals and averaging over one period

**First multiply the two fields.**

```{math}
e(t)h(t)=|Y|E_0^2\cos\omega t\cos(\omega t-\delta).
```

Use the trigonometric product-to-sum identity

```{math}
\cos\alpha\cos\beta
=\frac12\left[\cos(\alpha-\beta)+\cos(\alpha+\beta)\right]
```

below. Substituting {math}`\alpha=\omega t`, {math}`\beta=\omega t-\delta` gives:

```{math}
\begin{aligned}
e(t)h(t)
&=\frac{|Y|E_0^2}{2}
\left[\cos\bigl(\omega t-(\omega t-\delta)\bigr)
+\cos\bigl(\omega t+(\omega t-\delta)\bigr)\right]\\[4pt]
&=\frac{|Y|E_0^2}{2}
\left[\cos\delta+\cos(2\omega t-\delta)\right].
\end{aligned}
```

The first term in brackets, {math}`\cos\delta`, is constant in time. The second, {math}`\cos(2\omega t-\delta)`, oscillates at {math}`2\omega`, twice the fields' angular frequency {math}`\omega`.

**Now average over one period.** If the electric and magnetic period is {math}`\tau=2\pi/\omega`, the time average is the integral over that period divided by its duration.

```{math}
\begin{aligned}
\langle e h\rangle
&=\frac1{\tau}\int_0^{\tau}e(t)h(t)\,dt\\[4pt]
&=\frac{|Y|E_0^2}{2}
\left[
\frac1{\tau}\int_0^{\tau}\cos\delta\,dt
+\frac1{\tau}\int_0^{\tau}\cos(2\omega t-\delta)\,dt
\right].
\end{aligned}
```

The average of the constant first term equals its value.

```{math}
\frac1{\tau}\int_0^{\tau}\cos\delta\,dt
=\frac1{\tau}[t\cos\delta]_0^{\tau}
=\frac{\tau\cos\delta}{\tau}
=\cos\delta.
```

The second term averages to zero as follows.

```{math}
\begin{aligned}
\frac1{\tau}\int_0^{\tau}\cos(2\omega t-\delta)\,dt
&=\frac1{\tau}
\left[\frac{\sin(2\omega t-\delta)}{2\omega}\right]_0^{\tau}\\[4pt]
&=\frac{\sin(2\omega\tau-\delta)-\sin(-\delta)}{2\omega\tau}\\[4pt]
&=\frac{\sin(4\pi-\delta)-\sin(-\delta)}{4\pi}\\[4pt]
&=0.
\end{aligned}
```

The last line uses {math}`\sin(4\pi-\delta)=\sin(-\delta)`. A term oscillating at {math}`2\omega` completes exactly two cycles during {math}`\tau`, so positive and negative contributions cancel. It remains in instantaneous power; its **contribution to the period average is zero**.

The remaining average power density is therefore:

```{math}
\langle S_z\rangle=\langle e h\rangle
=\frac{|Y|E_0^2}{2}[\cos\delta+0]
=\frac12|Y|E_0^2\cos\delta.
```

This can also be expressed through the real part of {math}`Y`. Expanding {math}`Y` with Euler's formula:

```{math}
Y=|Y|e^{i\delta}
=|Y|\cos\delta+i|Y|\sin\delta,
\qquad
\operatorname{Re}Y=|Y|\cos\delta.
```

Therefore:

```{math}
\boxed{\langle S_z\rangle
=\frac12|Y|E_0^2\cos\delta
=\frac12E_0^2\operatorname{Re}Y.}
```

Writing magnetic amplitude magnitude as {math}`H_0=|Y|E_0` gives {math}`\langle S_z\rangle=\tfrac12 E_0H_0\cos\delta`. Thus **average power depends on the phase difference as well as the magnitudes of the electric and magnetic fields**.

The complex-amplitude formula introduced earlier gives the same result. Substitute {math}`E=E_0`, {math}`H=YE_0`:

```{math}
\frac12\operatorname{Re}[EH^*]
=\frac12\operatorname{Re}[E_0(YE_0)^*]
=\frac12 E_0^2\operatorname{Re}[Y^*]
=\frac12 E_0^2\operatorname{Re}Y.
```

## Power-reference boundaries and incident normalization

Incident power {math}`P_{\mathrm{in}}` is **power calculated from the incident wave alone at the interface between the incident half-space and the first layer it encounters**. For incidence from below, this is the boundary with the first layer reached from below. Reflected power {math}`P_r` is read at the same plane; transmitted power {math}`P_t` is read at the opposite boundary between the final layer and the exit half-space. Subscripts in, r, and t denote incident wave (or medium), reflected wave, and transmitted wave; i is used only for the imaginary unit.

Each wave's power is positive in that wave's direction of travel. Total reflection and transmission are:

```{math}
R_{\mathrm{total}}=\frac{P_r}{P_{\mathrm{in}}},
\qquad
T_{\mathrm{total}}=\frac{P_t}{P_{\mathrm{in}}}.
```

Dividing every power by {math}`P_{\mathrm{in}}` makes the reference incident power 1. This does not set the physical light source to 1 W. Defining efficiency requires {math}`P_{\mathrm{in}}>0`; this normalization cannot be used for an input consisting only of evanescent waves that carry no normal power.

Both incident and reflected waves are present at the incident boundary. To calculate **net power entering the layers** through it, add their **complex electric amplitudes together and their complex magnetic amplitudes together**.

```{math}
\mathbf E=\mathbf E_{\mathrm{in}}+\mathbf E_r,
\qquad
\mathbf H=\mathbf H_{\mathrm{in}}+\mathbf H_r.
```

Use the combined {math}`\mathbf E`, {math}`\mathbf H` to calculate {math}`\tfrac12\operatorname{Re}(\mathbf E\times\mathbf H^*)`, then integrate the component normal into the layers over the boundary's unit cell. Figure 13-2's {math}`P_{\mathrm{enter}}` is this net power divided by {math}`P_{\mathrm{in}}`. Combining fields includes interference. The detailed expansion and lossless/lossy distinction follow in {ref}`Combining incident and reflected waves for net entering power <incident-reflected-power>`.

```{figure} ../../figures/learn/learn/12-reference-planes.svg
:alt: For top illumination, purple line ② is the incident boundary between top and the first encountered layer; ③ is the exit boundary between the final layer and bot. Subtracting transmitted power at the exit boundary from net entering power at the incident boundary gives absorption in the entire stack.
:figclass: learning-figure

Figure 13-2. Power-reference boundaries and the absorption region for top illumination. The plotted {math}`P_{\mathrm{enter}}`, {math}`T`, {math}`A` are divided by incident-wave-only power {math}`P_{\mathrm{in}}`.
```

In absorbing media, amplitude decreases during propagation, so **the plane at which power is read matters**. Incident power means the power on reaching the reference plane, excluding earlier absorption in the incident medium. Transmission likewise means power just crossing the exit boundary, excluding subsequent absorption in the half-space.

Net entering power is positive into the layer stack. Wavebit's +z points upward, so inward is −z for top incidence and +z for bottom incidence.

(amplitude-to-efficiency)=
## Reflection and transmission from complex amplitude coefficients

First calculate incident, reflected, and transmitted powers **separately**. Here one plane wave is incident, and every outgoing electric-field component is divided by the same reference incident amplitude. {ref}`Chapter 12 <diffraction-amplitude-coefficients>` explains result arrays and reading coefficient magnitude and phase.

(refractive-index-and-angle)=
### Why refractive index and angle appear in lossless media

The following power-density equation, transmission derivation, normal-incidence example, and order-efficiency formulas assume **lossless, isotropic, nonmagnetic incident and exit half-spaces with positive real refractive indices**. Their angles and power formulas apply to propagating waves in those media. Lossy media are treated separately in {ref}`Loss in the incident or exit medium <lossy-amplitude-to-efficiency>`.

For propagation direction {math}`\hat{\mathbf k}` in a medium of refractive index {math}`n`:

```{math}
\mathbf H=\frac{n}{Z_0}\hat{\mathbf k}\times\mathbf E,
\qquad
\langle\mathbf S\rangle
=\frac{n}{2Z_0}|\mathbf E|^2\hat{\mathbf k}.
```

{math}`Z_0\approx376.73\,\Omega` is vacuum impedance, and {math}`|\mathbf E|^2=\mathbf E\cdot\mathbf E^*`. Equal electric-field magnitude can correspond to different magnetic-field magnitude at different refractive indices.

The first equation's {math}`\times` is a cross product. Taking propagation direction {math}`\hat{\mathbf k}` crossed with electric field {math}`\mathbf E`, in that order, gives magnetic-field direction. Physical power flow is {math}`\mathbf e(t)\times\mathbf h(t)`, with **electric field first**. Both follow the same cyclic right-hand rule; reversing a cross product reverses its direction.

If {math}`\theta` is the angle between propagation and **the surface normal toward which the wave passes**, the time-averaged normal power density is:

```{math}
I_\perp=\frac{n\cos\theta}{2Z_0}|\mathbf E|^2.
```

Refractive index {math}`n` comes from the electric–magnetic relation; {math}`\cos\theta` comes from selecting the normal component of power flow.

```{figure} ../../figures/learn/learn/13-wave-directions.svg
:alt: Left: an upward linearly polarized plane wave, with instantaneous electric field +x and magnetic field +y into the screen. Right: an xz section of top, the layer stack, and bot. A wave incident upward from bot first meets the bottommost layer; their interface is the xy incident plane. The angle between green incident power flow in bot and its purple normal component is theta; the normal component is cos theta times the flow magnitude.
:figclass: learning-figure
:name: wave-directions-and-normal-power

Figure 13-3. Electric and magnetic directions and power flow for bottom incidence. The right-hand xz section marks the incident interface between bot and the bottommost layer.
```

The left side illustrates Wavebit's **bottom incidence**. ⊗ indicates a vector into the screen, at one instant of linear polarization. Electric field +x and magnetic field +y produce +z power flow. The cross-product order is unchanged for top incidence: with electric field still +x, magnetic field points −y and power flows −z.

The **bottommost layer** at right is the lowest finite-thickness layer in the stack; the half-space `bot` lies below it. For bottom incidence, light travels from `bot` to this layer first, so **their interface is the incident boundary**. It spans x and y but appears as a horizontal line in the side-on xz section. The inward normal is +z.

The green arrow is **incident-wave-only time-averaged power flow** inside `bot`; the purple arrow is its **component normal to the boundary**. θ is the angle between them. This figure illustrates reading incident power and omits the reflected wave and fields inside the layers.

(transmission-amplitude-to-power)=
### Transmission from an electric-field amplitude ratio

We have obtained **the power density of one plane wave crossing a boundary**. Apply that expression separately to the incident wave and the transmitted wave emerging from the opposite side of the structure. Comparing their powers gives **transmission efficiency**, the fraction of incident power transmitted.

First consider one incident and one transmitted wave, each with one polarization component. Distinguish their quantities as follows. Subscript in means incident wave, t means transmitted wave, and out means exit medium.

| Quantity | Incident wave | Transmitted wave |
| --- | --- | --- |
| Refractive index of the containing medium | {math}`n_{\mathrm{in}}` | {math}`n_{\mathrm{out}}` |
| Angle between propagation and the normal toward which the wave passes | {math}`\theta_{\mathrm{in}}` | {math}`\theta_t` |
| Complex electric amplitude | {math}`E_{\mathrm{in}}` | {math}`E_t` |
| Time-averaged normal power density | {math}`I_{\perp,\mathrm{in}}` | {math}`I_{\perp,t}` |

The **amplitude transmission coefficient** {math}`t` is the transmitted complex electric amplitude divided by the incident complex electric amplitude.

```{math}
t=\frac{E_t}{E_{\mathrm{in}}},
\qquad
|t|^2=\frac{|E_t|^2}{|E_{\mathrm{in}}|^2}.
```

RCWA obtains this coefficient by solving electric and magnetic fields that satisfy boundary conditions. Applying the power-density expression then gives **transmission as a power ratio** {math}`T`.

Transmission is transmitted power divided by incident power. A plane wave's intensity is uniform along a boundary, so comparing power through equal unit-cell areas at the incident and exit boundaries cancels area. The ratio can therefore be written using normal power densities.

```{math}
T=\frac{P_t}{P_{\mathrm{in}}}
 =\frac{I_{\perp,t}}{I_{\perp,\mathrm{in}}}.
```

Calculate incident power from the incident wave alone. Substituting **each wave's refractive index, angle, and electric amplitude** into {math}`I_\perp=n\cos\theta\,|E|^2/(2Z_0)` gives:

```{math}
\begin{aligned}
I_{\perp,\mathrm{in}}
 &=\frac{n_{\mathrm{in}}\cos\theta_{\mathrm{in}}}{2Z_0}
   |E_{\mathrm{in}}|^2,\\[5pt]
I_{\perp,t}
 &=\frac{n_{\mathrm{out}}\cos\theta_t}{2Z_0}|E_t|^2.
\end{aligned}
```

Substitute these expressions into the numerator and denominator. The shared factor {math}`1/(2Z_0)` cancels:

```{math}
\begin{aligned}
T
&=\frac{\dfrac{n_{\mathrm{out}}\cos\theta_t}{2Z_0}|E_t|^2}
        {\dfrac{n_{\mathrm{in}}\cos\theta_{\mathrm{in}}}{2Z_0}|E_{\mathrm{in}}|^2}\\[4pt]
&=\frac{n_{\mathrm{out}}\cos\theta_t}
        {n_{\mathrm{in}}\cos\theta_{\mathrm{in}}}
  \frac{|E_t|^2}{|E_{\mathrm{in}}|^2}.
\end{aligned}
```

Replace the squared-amplitude ratio by the previously defined {math}`|t|^2`:

```{math}
\boxed{T=\frac{n_{\mathrm{out}}\cos\theta_t}
              {n_{\mathrm{in}}\cos\theta_{\mathrm{in}}}|t|^2.}
```

Here {math}`|t|^2` is **the ratio of squared electric-field magnitudes**. For equal electric magnitude, refractive index changes magnetic magnitude and propagation angle changes normal power. Obtaining **a power ratio** therefore requires the refractive-index and cosine ratio as well. If the two indices and the two angles are equal, that factor is 1 and {math}`T=|t|^2`.

(normal-incidence-interface-example)=
### Example: normal incidence from refractive index 1 to 1.5

Suppose media of refractive indices 1 and 1.5 meet directly at a flat interface, with no intervening layer. Light enters normally from the index-1 medium into the index-1.5 medium. Both are lossless, isotropic, and nonmagnetic.

#### Obtaining amplitude transmission from boundary conditions

The Fresnel formulas give complex reflected and transmitted electric amplitudes at a flat interface. **First solve boundary conditions for the amplitude coefficient, then convert it to a power ratio.**

Let incident, reflected, and transmitted complex electric amplitudes at the boundary be {math}`E_{\mathrm{in}}`, {math}`E_r`, {math}`E_t`. Read all three as **the same tangential electric-field component in one fixed direction**. Define amplitude transmission by:

```{math}
t=\frac{E_t}{E_{\mathrm{in}}}.
```

This is the **definition** of {math}`t`. We now determine its value from boundary conditions. Figure 13-4 shows the three waves, the two boundary conditions, and their solution together.

```{figure} ../../figures/learn/learn/13-fresnel-interface.svg
:alt: Left: incident and exit media meeting at an interface. Incident and transmitted waves travel downward; the reflected wave travels upward, with electric and magnetic amplitudes labeled. Right: tangential continuity gives E_in + E_r = E_t and n_in(E_in − E_r) = n_out E_t. Reversed reflected propagation gives H_r = −(n_in / Z_0) E_r. Solving both conditions yields t = E_t / E_in = 2 n_in / (n_in + n_out).
:figclass: learning-figure

Figure 13-4. Fresnel boundary conditions at normal incidence. Incident plus reflected tangential electric and magnetic fields must separately equal the transmitted values. Reversed reflected propagation introduces a minus sign in its electric-to-magnetic relation.
```

**Electric boundary condition:** at normal incidence, the electric field is parallel to the interface. Total tangential electric field must agree on both sides, so incident plus reflected field equals transmitted field.

```{math}
E_{\mathrm{in}}+E_r=E_t.
```

**Magnetic boundary condition:** use {math}`\mathbf H=(n/Z_0)\hat{\mathbf k}\times\mathbf E`. Read the same magnetic component for all three waves, taking the incident magnetic direction as positive. Reversed reflected propagation gives a minus sign in the electric-to-magnetic conversion.

```{math}
H_{\mathrm{in}}=\frac{n_{\mathrm{in}}}{Z_0}E_{\mathrm{in}},
\qquad
H_r=-\frac{n_{\mathrm{in}}}{Z_0}E_r,
\qquad
H_t=\frac{n_{\mathrm{out}}}{Z_0}E_t.
```

With no separate surface current, total tangential magnetic field must also agree on both sides. Therefore:

```{math}
\begin{aligned}
H_{\mathrm{in}}+H_r&=H_t,\\[4pt]
\frac{n_{\mathrm{in}}}{Z_0}E_{\mathrm{in}}
-\frac{n_{\mathrm{in}}}{Z_0}E_r
&=\frac{n_{\mathrm{out}}}{Z_0}E_t,\\[4pt]
n_{\mathrm{in}}(E_{\mathrm{in}}-E_r)&=n_{\mathrm{out}}E_t.
\end{aligned}
```

The last line multiplies both sides by {math}`Z_0`. Substitute {math}`E_r=E_t-E_{\mathrm{in}}` from electric continuity to eliminate reflected amplitude:

```{math}
\begin{aligned}
n_{\mathrm{in}}\bigl[E_{\mathrm{in}}-(E_t-E_{\mathrm{in}})\bigr]
 &=n_{\mathrm{out}}E_t,\\[4pt]
2n_{\mathrm{in}}E_{\mathrm{in}}-n_{\mathrm{in}}E_t
 &=n_{\mathrm{out}}E_t,\\[4pt]
2n_{\mathrm{in}}E_{\mathrm{in}}
 &=(n_{\mathrm{in}}+n_{\mathrm{out}})E_t.
\end{aligned}
```

Divide by {math}`(n_{\mathrm{in}}+n_{\mathrm{out}})E_{\mathrm{in}}` to obtain normal-incidence amplitude transmission.

```{math}
\boxed{t=\frac{E_t}{E_{\mathrm{in}}}
       =\frac{2n_{\mathrm{in}}}{n_{\mathrm{in}}+n_{\mathrm{out}}}.}
```

The first equality defines the amplitude coefficient; the second is the solution of electric and magnetic boundary conditions.

#### Substituting refractive indices and converting to transmission

Insert {math}`n_{\mathrm{in}}=1`, {math}`n_{\mathrm{out}}=1.5` into the Fresnel formula.

```{math}
t=\frac{2n_{\mathrm{in}}}{n_{\mathrm{in}}+n_{\mathrm{out}}}
 =\frac{2\times1}{1+1.5}
 =0.8.
```

The result is {math}`t=0.8`: the transmitted electric amplitude magnitude is 80% of the incident magnitude.

**Next convert it to transmission as a power ratio.** Substitute {math}`t=0.8` into the preceding transmission formula. At normal incidence both angles are 0° and both cosines are 1.

```{math}
T=\frac{n_{\mathrm{out}}\cos\theta_t}
        {n_{\mathrm{in}}\cos\theta_{\mathrm{in}}}|t|^2
 =\frac{1.5\times1}{1\times1}|0.8|^2
 =0.96.
```

Thus the electric amplitude is 80% of the incident amplitude, while transmitted power is 96% of incident power. Different refractive indices prevent reading the squared-field ratio {math}`|t|^2=0.64` directly as transmission. With no absorption here, the remaining 4% is reflected.

### The s/p components of each diffraction order

If order {math}`(m,n)` contains both s and p, add the powers of these orthogonal polarization components.

```{math}
\begin{aligned}
T_{mn}
&=\frac{n_{\mathrm{out}}\cos\theta_{t,mn}}
       {n_{\mathrm{in}}\cos\theta_{\mathrm{in}}}
\left(|t_s^{(m,n)}|^2+|t_p^{(m,n)}|^2\right),\\[5pt]
R_{mn}
&=\frac{\cos\theta_{r,mn}}{\cos\theta_{\mathrm{in}}}
\left(|r_s^{(m,n)}|^2+|r_p^{(m,n)}|^2\right).
\end{aligned}
```

Reflected and incident waves occupy the same medium, so refractive index cancels. A reflected order may travel in a different direction, however, leaving the cosine ratio. {math}`\theta_{r,mn}`, {math}`\theta_{t,mn}` are measured between each outgoing wave and **the normal pointing out of the structure**.

Summing order efficiencies gives {math}`R_{\mathrm{total}}`, {math}`T_{\mathrm{total}}`. Within one calculation group with real transverse wavevectors, cross terms between different outgoing orders cancel in the unit-cell integral. Read zero-normal-power orders, such as evanescent waves in a lossless exit medium, as explained in [Chapter 12](results.md).

(lossy-amplitude-to-efficiency)=
### Loss in the incident or exit medium

Power in a lossy medium still follows {math}`\tfrac12\operatorname{Re}(\mathbf E\times\mathbf H^*)`. Obtain each wave's electric and magnetic fields using complex materials, **calculate incident and outgoing powers separately, then divide**. Substituting complex values into the lossless index/cosine ratio and taking the real part afterward gives a different result. Take the real part separately for each power; its form depends on polarization. For s below, the factor is {math}`\operatorname{Re}Y`. For p referenced to electric amplitude, it is {math}`\operatorname{Re}(\kappa/\varepsilon_r)` multiplied by {math}`|n/\mu_r|^2`, where {math}`\kappa` is normal wavevector divided by vacuum wavevector. The final section, “s/p boundary amplitudes used in code,” explains their origins.

#### Defining tangential components and admittance

Consider **s polarization at a flat, nondiffracting interface** between linear isotropic media. Incidence is from the upper into the lower medium in the xz plane. For all three waves, read electric y and tangential magnetic x components in the same coordinate directions.

Write the boundary components as {math}`E_j=E_{y,j}`, {math}`H_j=H_{x,j}`, with {math}`j=\mathrm{in},r,t` denoting incident, reflected, and transmitted waves. **These are complex amplitudes of individual components, not full vectors.** For s polarization, {math}`\mathbf E_j=E_j\hat{\mathbf y}`, and {math}`E_j` is a complex scalar.

Define amplitude reflection {math}`r` and transmission {math}`t` by:

```{math}
r=\frac{E_r}{E_{\mathrm{in}}},
\qquad
t=\frac{E_t}{E_{\mathrm{in}}}.
```

Thus {math}`E_r=rE_{\mathrm{in}}`, {math}`E_t=tE_{\mathrm{in}}`. Calculate each wave's power next and divide by incident power.

Each wave's **wave admittance** is the ratio of tangential magnetic to tangential electric field.

```{math}
Y_{\mathrm{in}}=\frac{H_{\mathrm{in}}}{E_{\mathrm{in}}},\qquad
Y_r=\frac{H_r}{E_r},\qquad
Y_t=\frac{H_t}{E_t}.
```

At a flat interface, incident and reflected waves share transverse wavevector and have opposite normal wavevectors. Applying these reflection conditions to Maxwell's equations gives:

```{math}
Y_r=-Y_{\mathrm{in}},
\qquad
H_{\mathrm{in}}=Y_{\mathrm{in}}E_{\mathrm{in}},\quad H_r=-Y_{\mathrm{in}}E_r,\quad H_t=Y_tE_t.
```

When calculating reflected power below, substitute **the relation read in the same coordinate components**, {math}`Y_r=-Y_{\mathrm{in}}`, to express it using {math}`Y_{\mathrm{in}}`. The sign change is derived below.

::::{dropdown} Derivation: why reflected admittance changes sign

**1. Frequency and transverse wavevector are conserved at a flat interface.**

At a time-independent interface uniform in x and y, incident and reflected waves have the same frequency and transverse wavevector, because boundary conditions hold at every position and time. In the chosen xz incidence plane, {math}`k_y=0`, and:

```{math}
\omega_r=\omega_{\mathrm{in}}=\omega,
\qquad
k_{x,r}=k_{x,\mathrm{in}}=k_x.
```

Both waves lie in the upper medium and therefore share permittivity {math}`\varepsilon` and permeability {math}`\mu`. Here these are absolute values, {math}`\varepsilon=\varepsilon_0\varepsilon_r`, {math}`\mu=\mu_0\mu_r`, rather than relative values.

**2. Obtain the normal wavevector from dispersion.**

The plane-wave dispersion relation in an isotropic medium is:

```{math}
k_x^2+k_z^2=\omega^2\varepsilon\mu,
\qquad
k_z^2=\omega^2\varepsilon\mu-k_x^2.
```

All right-hand values are identical for the two waves, so {math}`k_{z,r}^2=k_{z,\mathrm{in}}^2`. Of the two roots sharing that square, choose the solution entering the boundary as incident and the one leaving it as reflected:

```{math}
k_{z,r}=-k_{z,\mathrm{in}}.
```

Thus **only the normal component changes sign; tangential components remain unchanged**. At oblique incidence, reflection does not retrace the incident path. In a lossy medium {math}`k_z` may be complex; its two roots still have opposite signs. The dispersion expression {math}`k_z^2` is a complex square, distinct from {math}`|k_z|^2`.

**3. Obtain the magnetic/electric ratio from Maxwell's equations.**

With time convention {math}`e^{-i\omega t}`, Faraday's law is:

```{math}
\nabla\times\mathbf E=i\omega\mu\mathbf H.
```

The left-hand {math}`\nabla\times` is the **curl**, calculated by differentiating field components with respect to position. Substitute a plane-wave electric field and evaluate it step by step.

**Spatial dependence of a plane wave.** In a uniform medium, its complex electric field can be written:

```{math}
\mathbf E(x,y,z)
=\mathbf E_0e^{i(k_xx+k_yy+k_zz)}.
```

{math}`\mathbf E_0` is a position-independent complex amplitude vector, and {math}`\mathbf k=(k_x,k_y,k_z)` is the wavevector. The exponential represents spatial phase variation and, for complex wavevectors, amplitude variation. The oscillatory time dependence {math}`e^{-i\omega t}` has been factored out of this complex-field expression.

**Differentiating with respect to each coordinate.** First differentiate in x. {math}`\mathbf E_0` is constant and the exponent's x derivative is {math}`ik_x`, so:

```{math}
\begin{aligned}
\frac{\partial\mathbf E}{\partial x}
&=\mathbf E_0\frac{\partial}{\partial x}
  e^{i(k_xx+k_yy+k_zz)}\\[5pt]
&=ik_x\mathbf E_0e^{i(k_xx+k_yy+k_zz)}\\[5pt]
&=ik_x\mathbf E.
\end{aligned}
```

Differentiation in y and z likewise multiplies by the corresponding wavevector component.

```{math}
\frac{\partial\mathbf E}{\partial y}=ik_y\mathbf E,
\qquad
\frac{\partial\mathbf E}{\partial z}=ik_z\mathbf E.
```

Differentiating a vector means differentiating each component. For example, {math}`\partial E_z/\partial y=ik_yE_z` and {math}`\partial E_y/\partial z=ik_zE_y`. Complex wavevectors in lossy media use the same exponential differentiation rule.

**Evaluating the curl.** In components:

```{math}
\nabla\times\mathbf E=
\begin{pmatrix}
\dfrac{\partial E_z}{\partial y}-\dfrac{\partial E_y}{\partial z}\\[8pt]
\dfrac{\partial E_x}{\partial z}-\dfrac{\partial E_z}{\partial x}\\[8pt]
\dfrac{\partial E_y}{\partial x}-\dfrac{\partial E_x}{\partial y}
\end{pmatrix}.
```

Insert the plane-wave derivatives above and factor out the common {math}`i`.

```{math}
\begin{aligned}
\nabla\times\mathbf E
&=\begin{pmatrix}
ik_yE_z-ik_zE_y\\[3pt]
ik_zE_x-ik_xE_z\\[3pt]
ik_xE_y-ik_yE_x
\end{pmatrix}\\[8pt]
&=i\begin{pmatrix}
k_yE_z-k_zE_y\\[3pt]
k_zE_x-k_xE_z\\[3pt]
k_xE_y-k_yE_x
\end{pmatrix}\\[8pt]
&=i\mathbf k\times\mathbf E.
\end{aligned}
```

The last three bracketed components are those of {math}`\mathbf k\times\mathbf E`. Thus the plane-wave curl equals **i times the cross product of wavevector with electric field**.

**Substituting into Faraday's law.** Put {math}`\nabla\times\mathbf E=i\mathbf k\times\mathbf E` into the original equation's left side.

```{math}
i\mathbf k\times\mathbf E=i\omega\mu\mathbf H.
```

Divide by {math}`i` and solve for the magnetic field:

```{math}
\mathbf k\times\mathbf E=\omega\mu\mathbf H,
\qquad
\mathbf H=\frac{\mathbf k\times\mathbf E}{\omega\mu}.
```

**Tangential components for s polarization.** With {math}`\mathbf E=(0,E_y,0)` and {math}`\mathbf k=(k_x,0,k_z)`, the cross product gives:

```{math}
\mathbf k\times\mathbf E=(-k_zE_y,\,0,\,k_xE_y),
\qquad
H_x=-\frac{k_z}{\omega\mu}E_y.
```

The admittance of the chosen tangential components is therefore:

```{math}
Y=\frac{H_x}{E_y}=-\frac{k_z}{\omega\mu}.
```

Apply this to both incident and reflected waves and insert {math}`k_{z,r}=-k_{z,\mathrm{in}}`:

```{math}
Y_{\mathrm{in}}=-\frac{k_{z,\mathrm{in}}}{\omega\mu},
\qquad
Y_r=-\frac{k_{z,r}}{\omega\mu}
   =\frac{k_{z,\mathrm{in}}}{\omega\mu}
   =-Y_{\mathrm{in}}.
```

This relation applies to an incident/reflected pair in the same medium sharing one transverse wavevector. Different grating orders have different transverse wavevectors, so calculate each order's normal wavevector and admittance separately.

::::

#### Power density of each wave

First calculate the incident wave alone. For the chosen components, {math}`(\mathbf E\times\mathbf H^*)_z=-E_yH_x^*`. Normal power density in **the incident −z direction** is therefore {math}`\tfrac12\operatorname{Re}(E_{\mathrm{in}}H_{\mathrm{in}}^*)`. Substituting {math}`H_{\mathrm{in}}=Y_{\mathrm{in}}E_{\mathrm{in}}` gives:

```{math}
\begin{aligned}
I_{\mathrm{in}}
&=\frac12\operatorname{Re}(E_{\mathrm{in}}H_{\mathrm{in}}^*)\\[4pt]
&=\frac12\operatorname{Re}\!\left[E_{\mathrm{in}}(Y_{\mathrm{in}}E_{\mathrm{in}})^*\right]\\[4pt]
&=\frac12\operatorname{Re}\!\left[Y_{\mathrm{in}}^*E_{\mathrm{in}}E_{\mathrm{in}}^*\right]\\[4pt]
&=\frac12\operatorname{Re}(Y_{\mathrm{in}}^*)|E_{\mathrm{in}}|^2\\[4pt]
&=\frac12\operatorname{Re}(Y_{\mathrm{in}})|E_{\mathrm{in}}|^2.
\end{aligned}
```

{math}`E_{\mathrm{in}}`, {math}`Y_{\mathrm{in}}`, and {math}`E_{\mathrm{in}}^*` are complex scalars, so their multiplication order can change; no vector cross product has been reversed. Also, {math}`E_{\mathrm{in}}E_{\mathrm{in}}^*=|E_{\mathrm{in}}|^2` is real and can leave the real-part operation. Conjugation preserves real part, so {math}`\operatorname{Re}(Y_{\mathrm{in}}^*)=\operatorname{Re}(Y_{\mathrm{in}})`.

Read reflected {math}`I_r` as positive in **+z, back into the incident medium**. Reversing the normal relative to incidence gives a minus sign in its power expression. Substitute {math}`H_r=Y_rE_r` and then {math}`Y_r=-Y_{\mathrm{in}}`:

```{math}
\begin{aligned}
I_r
&=-\frac12\operatorname{Re}(E_rH_r^*)\\[4pt]
&=-\frac12\operatorname{Re}(Y_r)|E_r|^2\\[4pt]
&=\frac12\operatorname{Re}(Y_{\mathrm{in}})|E_r|^2.
\end{aligned}
```

Both **the power-reading normal** and **admittance** change sign, so no minus remains. Read transmitted power relative to the −z normal, as for incidence, using exit admittance {math}`Y_t`. Taking each wave positive in its own crossing direction gives:

```{math}
\begin{aligned}
I_{\mathrm{in}}&=\frac12\operatorname{Re}(Y_{\mathrm{in}})|E_{\mathrm{in}}|^2,\\[3pt]
I_r&=\frac12\operatorname{Re}(Y_{\mathrm{in}})|E_r|^2,\\[3pt]
I_t&=\frac12\operatorname{Re}(Y_t)|E_t|^2.
\end{aligned}
```

#### Dividing by incident power for reflection and transmission

Comparing power through equal boundary areas cancels area, so reflection and transmission follow from these power-density ratios. Using the definitions of {math}`r`, {math}`t`:

```{math}
\begin{aligned}
R&=\frac{I_r}{I_{\mathrm{in}}}
  =\frac{|E_r|^2}{|E_{\mathrm{in}}|^2}
  =|r|^2,\\[6pt]
T&=\frac{I_t}{I_{\mathrm{in}}}
  =\frac{\operatorname{Re}(Y_t)}{\operatorname{Re}(Y_{\mathrm{in}})}
   \frac{|E_t|^2}{|E_{\mathrm{in}}|^2}\\[4pt]
 &=\frac{\operatorname{Re}(Y_t)}{\operatorname{Re}(Y_{\mathrm{in}})}|t|^2.
\end{aligned}
```

The denominator's incident power is assumed positive. Generally {math}`\operatorname{Re}(Y_t/Y_{\mathrm{in}})` differs from {math}`\operatorname{Re}(Y_t)/\operatorname{Re}(Y_{\mathrm{in}})`, so **distinguish taking the real part after forming a ratio from calculating each power before dividing**. For other orders and polarizations, use the appropriate electric–magnetic relation.

This reflection ratio describes the reflected wave alone. Next we examine why net power at a boundary containing both incident and reflected waves need not equal {math}`1-R`.

(incident-reflected-power)=
## Combining incident and reflected waves for net entering power

The preceding section calculated each wave's power separately. Now **combine incident and reflected waves first**, then calculate net power through their shared boundary. Their admittance sign relation is unchanged, but power is calculated from total fields instead of individual waves.

### Electric and magnetic fields of the two waves

Consider normal incidence in an isotropic medium. To make signs explicit, choose **bottom incidence**. The incident wave travels +z, the reflected wave −z, and +z points into the layer stack.

At one reference-plane position, let {math}`A_{\mathrm{in}}`, {math}`A_r` be the complex amplitudes of the incident and reflected electric x components. Total electric x and magnetic y components there are:

```{math}
E_x=A_{\mathrm{in}}+A_r,
\qquad
H_y=Y(A_{\mathrm{in}}-A_r).
```

Here {math}`Y=1/Z` is **admittance referenced to the incident wave**, with {math}`Z` its impedance. This example writes {math}`Y` for the preceding section's {math}`Y_{\mathrm{in}}`. With electric x and magnetic y chosen, incident magnetic field is {math}`Y A_{\mathrm{in}}` and reflected magnetic field {math}`-Y A_r`. Applying {math}`Y_r=-Y` and adding gives {math}`H_y=Y(A_{\mathrm{in}}-A_r)`. **These field-superposition expressions apply to both lossless and lossy media.**

### Interference terms from multiplying total fields

Time-averaged net power density comes from the product of total fields.

```{math}
\begin{aligned}
\langle S_z\rangle
&=\frac12\operatorname{Re}(E_x H_y^*)\\[3pt]
&=\frac12\operatorname{Re}\!\left[
Y^*(A_{\mathrm{in}}+A_r)(A_{\mathrm{in}}^*-A_r^*)\right]\\[3pt]
&=\frac12\operatorname{Re}\!\left[
Y^*\left(|A_{\mathrm{in}}|^2-|A_r|^2+A_rA_{\mathrm{in}}^*-A_{\mathrm{in}}A_r^*\right)\right].
\end{aligned}
```

The first two bracketed terms are the individual-wave contributions. The last two are **cross terms**, multiplying one wave's electric field by the other's magnetic field. Use {math}`A_rA_{\mathrm{in}}^*-A_{\mathrm{in}}A_r^*=2i\operatorname{Im}(A_{\mathrm{in}}^*A_r)`:

```{math}
\langle S_z\rangle
=\frac{\operatorname{Re}Y}{2}
\left(|A_{\mathrm{in}}|^2-|A_r|^2\right)
+\operatorname{Im}Y\,\operatorname{Im}(A_{\mathrm{in}}^*A_r).
```

Name the incident and reflected power densities {math}`I_{\mathrm{in}}`, {math}`I_r`, and the cross contribution {math}`I_{\mathrm{cross}}`:

```{math}
\begin{aligned}
I_{\mathrm{in}}&=\frac{\operatorname{Re}Y}{2}|A_{\mathrm{in}}|^2,
& I_r&=\frac{\operatorname{Re}Y}{2}|A_r|^2,\\[3pt]
I_{\mathrm{cross}}&=\operatorname{Im}Y\,\operatorname{Im}(A_{\mathrm{in}}^*A_r),
& \langle S_z\rangle&=I_{\mathrm{in}}-I_r+I_{\mathrm{cross}}.
\end{aligned}
```

{math}`I_r` is positive in the reflected wave's outward direction, so subtract it from inward net power. {math}`I_{\mathrm{cross}}` is the contribution in the total-field product when both waves are present. **It is not an additional light source or a third wave.**

### What cancels in a lossless medium

For a propagating wave in an ordinary lossless medium, {math}`Y` is real, so {math}`\operatorname{Im}Y=0`. Therefore:

```{math}
I_{\mathrm{cross}}=0,
\qquad
\langle S_z\rangle=I_{\mathrm{in}}-I_r.
```

The squared total electric amplitude still contains a cross term.

```{math}
|E_x|^2=|A_{\mathrm{in}}|^2+|A_r|^2+2\operatorname{Re}(A_{\mathrm{in}}^*A_r).
```

It produces an interference pattern of stronger and weaker electric field with position. **What cancels in a lossless medium is the cross contribution to time-averaged power normal to the surface.** Electric and magnetic interference itself does not disappear.

### What remains in a lossy medium, and its sign

Admittance {math}`Y` can be complex in a lossy medium, allowing power interference to remain. With both amplitudes nonzero, define relative phase {math}`\Delta\varphi=\arg A_r-\arg A_{\mathrm{in}}`:

```{math}
I_{\mathrm{cross}}
=\operatorname{Im}Y\,|A_{\mathrm{in}}|\,|A_r|\sin\Delta\varphi.
```

The cross term's magnitude and sign depend on **relative wave phase** as well as {math}`\operatorname{Im}Y`. It vanishes if {math}`\operatorname{Im}Y=0`, if {math}`\sin\Delta\varphi=0`, or if either amplitude is zero.

| Interference term | Net entering power density at the reference plane |
| --- | --- |
| {math}`I_{\mathrm{cross}}>0` | Greater than {math}`I_{\mathrm{in}}-I_r` |
| {math}`I_{\mathrm{cross}}<0` | Less than {math}`I_{\mathrm{in}}-I_r` |
| {math}`I_{\mathrm{cross}}=0` | Equal to {math}`I_{\mathrm{in}}-I_r` |

**A positive interference term does not imply gain, and a negative one does not imply absorption.** Whether a medium supplies or absorbs field power is separate from the cross term's sign. The absorbing incident medium in this chapter's {ref}`example <boundary-power-example>` has a positive cross term. The extra arriving power is power that the incident medium absorbs less of near the reference plane; {ref}`Where does power above 1 come from? <interference-absorption-origin>` examines this by height.

### Net entering power normalized by incident power

Let {math}`P_{\mathrm{enter}}` denote net power entering the layers through the boundary **relative to incident-wave-only power**. Read numerator and denominator at the same incident boundary.

```{math}
P_{\mathrm{enter}}
=\frac{\text{net inflow from combined incident and reflected waves}}
       {\text{power of the incident wave alone}}.
```

{math}`P_{\mathrm{enter}}` is a **dimensionless ratio** of powers. This example has normal bottom incidence, so inward power density is {math}`\langle S_z\rangle` along +z. Incident-only density is {math}`I_{\mathrm{in}}`; both have units W/m². Because density is uniform along the plane, dividing powers through equal areas cancels area.

Insert {math}`\langle S_z\rangle=I_{\mathrm{in}}-I_r+I_{\mathrm{cross}}` and use {math}`R=I_r/I_{\mathrm{in}}`:

```{math}
\begin{aligned}
P_{\mathrm{enter}}
&=\frac{\langle S_z\rangle}{I_{\mathrm{in}}}\\[4pt]
&=\frac{I_{\mathrm{in}}-I_r+I_{\mathrm{cross}}}{I_{\mathrm{in}}}\\[4pt]
&=1-\frac{I_r}{I_{\mathrm{in}}}+\frac{I_{\mathrm{cross}}}{I_{\mathrm{in}}}\\[4pt]
&=1-R+\frac{I_{\mathrm{cross}}}{I_{\mathrm{in}}}.
\end{aligned}
```

This ratio requires {math}`I_{\mathrm{in}}>0`. The right-hand **1 represents incident-wave-only reference power**, and {math}`R` is the reflected outward-power ratio. The last term expresses interference's contribution to net entering power relative to the same incident reference.

For propagating incidence in a lossless incident medium, {math}`I_{\mathrm{cross}}=0`, giving {math}`P_{\mathrm{enter}}=1-R`. In a lossy medium the term can remain, so **reflection alone cannot determine net entering power**. Calculate it from the combined electric and magnetic fields.

Incident and reflected waves of the same order have the same transverse wavevector. Their transverse phase factors cancel in the cross term, so averaging along the boundary does not remove it.

In Wavebit, `result.power_entering` gives net entering power ratio including all orders and polarizations. The next section combines it with transmission to obtain whole-stack absorption.

::::{dropdown} Net entering power from the reflection amplitude coefficient

For the normal-incidence example, let {math}`r` compare **the same electric x component** of incident and reflected waves:

```{math}
r=\frac{A_r}{A_{\mathrm{in}}},
\qquad A_r=rA_{\mathrm{in}},
\qquad R=|r|^2.
```

The amplitude product in the interference term is:

```{math}
A_{\mathrm{in}}^*A_r=A_{\mathrm{in}}^*(rA_{\mathrm{in}})=r|A_{\mathrm{in}}|^2.
```

Since {math}`|A_{\mathrm{in}}|^2` is real, {math}`\operatorname{Im}(A_{\mathrm{in}}^*A_r)=|A_{\mathrm{in}}|^2\operatorname{Im}r`. The interference term and incident-only density are therefore:

```{math}
\begin{aligned}
I_{\mathrm{cross}}
&=\operatorname{Im}Y\,\operatorname{Im}(A_{\mathrm{in}}^*A_r)\\[4pt]
&=\operatorname{Im}Y\,|A_{\mathrm{in}}|^2\operatorname{Im}r,\\[8pt]
I_{\mathrm{in}}&=\frac12\operatorname{Re}Y\,|A_{\mathrm{in}}|^2.
\end{aligned}
```

Dividing cancels their common factor {math}`|A_{\mathrm{in}}|^2`.

```{math}
\begin{aligned}
\frac{I_{\mathrm{cross}}}{I_{\mathrm{in}}}
&=\frac{\operatorname{Im}Y\,|A_{\mathrm{in}}|^2\operatorname{Im}r}
        {\tfrac12\operatorname{Re}Y\,|A_{\mathrm{in}}|^2}\\[5pt]
&=2\frac{\operatorname{Im}Y}{\operatorname{Re}Y}\operatorname{Im}r.
\end{aligned}
```

Substitute this into net entering power ratio:

```{math}
\begin{aligned}
P_{\mathrm{enter}}
&=1-R+\frac{I_{\mathrm{cross}}}{I_{\mathrm{in}}}\\[4pt]
&=1-|r|^2
+2\frac{\operatorname{Im}Y}{\operatorname{Re}Y}\operatorname{Im}r.
\end{aligned}
```

The last term depends on reflection phase as well as magnitude: writing {math}`r=|r|e^{i\varphi}` gives {math}`\operatorname{Im}r=|r|\sin\varphi`.

This formula also applies to oblique top-incidence s polarization, as in this chapter's {ref}`example <boundary-power-example>`. Read {math}`r` as the ratio of tangential electric y components and {math}`Y` as {math}`Y_{\mathrm{in}}` from the {ref}`lossy-medium s-polarization section <lossy-amplitude-to-efficiency>`. There too, tangential magnetic fields combine as {math}`Y_{\mathrm{in}}(E_{\mathrm{in}}-E_r)`, giving the same expansion. The example's interference contribution 0.158416 is this formula's value, independently checked by its script.

Here {math}`r` compares electric components along **one fixed x direction**. Wavebit's `r_p` instead uses a p basis aligned with each wave's propagation, so match basis and sign before substituting it. Read net entering power directly from `result.power_entering`.

::::

(half-space-power)=
## Whole-stack absorption and reading results

In steady state, subtract **transmitted power leaving the opposite boundary** from **net power entering through the incident boundary** to obtain absorption in the entire stack. Calling that absorption divided by incident-only power {math}`A`:

```{math}
A=P_{\mathrm{enter}}-T_{\mathrm{total}}.
```

Here {math}`P_{\mathrm{enter}}` is **net entering power divided by incident power**, not power in W. Read Wavebit results as follows.

| Result | Meaning |
| --- | --- |
| `result.R` | Per-order power fractions of reflected waves exiting toward the incident side |
| `result.T` | Per-order power fractions of transmitted waves exiting the opposite boundary |
| `result.power_entering` | Net entering power ratio calculated from combined incident and reflected fields |
| `result.absorption` | Absorbed-power ratio for all specified finite-thickness layers |

```python
R_total = result.R.sum()
T_total = result.T.sum()
P_enter = result.power_entering
A = result.absorption             # P_enter - T_total
```

For propagating incidence in a lossless incident medium, {math}`P_{\mathrm{enter}}=1-R_{\mathrm{total}}`, so:

```{math}
A=1-R_{\mathrm{total}}-T_{\mathrm{total}}.
```

A lossless incident medium alone does not imply {math}`A=0`: absorbing layers can give {math}`A>0`. If all layers are also lossless, {math}`A=0`; then {math}`P_{\mathrm{enter}}=T_{\mathrm{total}}` even with a lossy incident medium.

Absorption covers **all finite-thickness layers supplied in `structure`**, regardless of layer count. It excludes absorption inside the `top` and `bot` half-spaces themselves. Even with an absorbing exit medium, {math}`T_{\mathrm{total}}` means power crossing the exit boundary.

Net entering power and absorption are also available with `answers="efficiency"`; boundary amplitudes suffice, with no internal-field reconstruction. If either transverse wavevector (`kx`, `ky`) has a nonzero imaginary part, summing order powers may differ from integrating over a cell. In that case `power_entering` and `absorption` are `NaN`. They are `NaN` as well, together with `R` and `T`, when the incident wave is evanescent ({ref}`Chapter 8 <complex-transverse-wavevector>`): it carries no incident power to divide by. {ref}`Chapter 8 <absorbing-incidence>` explains the real transverse-wavevector conditions used here.

(boundary-power-example)=
## Checking with numerical examples

### Transmission above 1 in an absorbing incident medium

Compare power flow with and without absorption in incident medium `top`. Both calculations use top-incident s polarization at vacuum wavelength 1,000 nm. The angle between phase propagation in the incident medium and the downward normal is 30°.

Between `top` and `bot`, place one uniform layer of thickness 100 nm. Both it and `bot` are lossless with relative permittivity {math}`\varepsilon_r=2.25`; every medium and layer has {math}`\mu_r=1`. The layer and `bot` are identical, so their interface causes no reflection.

Run this setup twice, changing only `top`: first lossless {math}`\varepsilon_r=4`, then absorbing {math}`\varepsilon_r=4+3i`. The finite layer has no absorption in either case, isolating how incident-medium loss changes reflection, transmission, and net entering power.

Both angles are 30°, but different `top` materials give different transverse `kx`. This example chooses the xz incidence plane with `ky=0`. Calculate each wavevector using {ref}`Chapter 8's angle-input method <absorbing-incidence>`.

```{figure} ../../figures/learn/learn/12-power-balance.svg
:alt: Normalized by incident power, lossless incidence gives net entering power 1 minus R. In the lossy example, interference 0.158416 makes net entering power and transmission both 1.013189, while layer absorption is zero.
:figclass: learning-figure

Figure 13-5. Net entering and transmitted power for different incident media. The finite stack and exit medium are lossless in both cases. Every value is divided by incident-wave-only power at the incident reference plane.
```

| Quantity | Lossless incidence: {math}`\varepsilon_r=4` | Lossy incidence: {math}`\varepsilon_r=4+3i` |
| --- | ---: | ---: |
| `R_total` | 0.046414 | 0.145227 |
| `1 - R_total` | 0.953586 | 0.854773 |
| Normalized interference contribution | 0 | 0.158416 |
| `power_entering` | 0.953586 | 1.013189 |
| `T_total` | 0.953586 | 1.013189 |
| `absorption` | 0 | 0 |
| `1 - R_total - T_total` | 0 | −0.158416 |

Values are rounded to six decimal places. Computed zero absorption may retain floating-point error.

Although {math}`T_{\mathrm{total}}>1` for the absorbing incident medium, actual net entering and transmitted power ratios both equal 1.013189, so layer absorption is zero. This is possible because incident-only power is reference 1 while interference also contributes to net boundary inflow. Do not interpret negative {math}`1-R_{\mathrm{total}}-T_{\mathrm{total}}` as layer absorption here.

(interference-absorption-origin)=
#### Where does power above 1 come from?

If incident-only power is 1 but 1.013189 enters the layers, where does the difference originate? In **absorption by the incident medium `top` itself**. Reflection changes how much `top` absorbs near the reference plane; reduced absorption allows more power to reach that plane.

**First, 1 is not the source's entire supplied power.** It is incident-wave-only power upon reaching the reference plane, excluding prior absorption in `top`. This strongly absorbing example has incident-only power 3.112882 about 112.45 nm above the plane. That component alone decreases by 2.112882 on its way down. **This decrease belongs to the incident component alone; it is not the actual absorption by `top` in that interval.** Actual absorption also includes reflected and interference contributions, separated below.

**Absorption follows squared total electric field.** Time-averaged absorbed power per unit volume in a lossy medium is:

```{math}
p_{\mathrm{abs}}
=\frac{\omega\varepsilon_0\operatorname{Im}\varepsilon_r}{2}|\mathbf E|^2.
```

All media here have {math}`\mu_r=1`, so there is no magnetic absorption. If incident and reflected electric y components at height {math}`z` are {math}`E_{\mathrm{in}}(z)`, {math}`E_r(z)`, squared total field contains the cross term already seen in the lossless discussion of {ref}`net entering power <incident-reflected-power>`.

```{math}
|E_{\mathrm{in}}+E_r|^2=|E_{\mathrm{in}}|^2+|E_r|^2+2\operatorname{Re}(E_{\mathrm{in}}^*E_r).
```

This term creates **bright fringes** with stronger electric field and **dark fringes** with weaker field. In a lossless medium, {math}`\operatorname{Im}\varepsilon_r=0`, so the pattern does not alter absorption. An absorbing medium absorbs more at bright fringes and less at dark fringes.

**Resolve the power balance by height.** Consider `top` between the reference plane ({math}`z=0`) and a plane at height {math}`z`. In steady state, net power entering the upper plane minus net power leaving through the reference equals absorption between them. Split both net power and absorption into incident-only, reflected-only, and interference contributions; each part balances separately. The interference part is:

```{math}
I_{\mathrm{cross}}(0)
=I_{\mathrm{cross}}(z)
-\int_0^{z}\frac{\omega\varepsilon_0\operatorname{Im}\varepsilon_r}{2}\,
2\operatorname{Re}\!\left[E_{\mathrm{in}}^*(z')E_r(z')\right]dz'.
```

{math}`I_{\mathrm{cross}}(z)` is the cross contribution at height {math}`z`, calculated as before. Incident amplitude grows upward while reflected amplitude decreases, so their product is independent of height; interference oscillates and changes sign with height.

```{figure} ../../figures/learn/learn/13-interference-absorption.svg
:alt: Two panels show heights 0–200 nm above the reference in the absorbing-incidence example. Left: absorption in top, dashed for separately calculated waves and solid for the combined field. The solid curve is larger from 0 to about 45.49 nm and smaller from about 45.49 to 179.42 nm. Right: interference is zero around 112.45 nm, +0.328 around 45.49 nm, and +0.158 at the reference. Incident-only power at about 112.45 nm is 3.11.
:figclass: learning-figure

Figure 13-6. Origin of the interference contribution in an absorbing incident medium. Horizontal values are divided by incident-only power at the reference; the left panel shows absorption per 100 nm of height. Orange and teal bands mark heights with increased and decreased absorption due to interference.
```

::::{dropdown} Calculating absorption as a function of height

The left curves in Figure 13-6 evaluate electric field in `top` at each height and insert it into the absorption density {math}`p_{\mathrm{abs}}` above.

**1. Write the fields in `top` as functions of height.** The layer and `bot` are identical and no diffraction occurs, so `top` contains only incident and reflected plane waves. With incident tangential electric amplitude 1 at the reference, their y components at height {math}`z` are:

```{math}
E_{\mathrm{in}}(z)=e^{-iqk_0z},
\qquad
E_r(z)=r\,e^{iqk_0z}.
```

{math}`k_0=2\pi/\lambda` is vacuum wavevector. {math}`q` is the incident normal wavevector read in the propagation direction −z and divided by {math}`k_0` ({math}`k_{z,\mathrm{in}}=-qk_0`). Of the roots of {math}`q^2=\varepsilon_r\mu_r-(k_x/k_0)^2`, choose downward propagation with attenuation. As in {ref}`Chapter 8's example <absorbing-incidence-example>`, solve this relation together with phase angle 30° to obtain {math}`k_x` ({math}`k_x/k_0=1.077700`). Recorded values are {math}`q=1.866632+0.803587i`, {math}`r=0.333729+0.183990i`. Since {math}`\operatorname{Im}q>0`, incident amplitude increases upward and reflected amplitude decreases.

**2. Substitute into absorption density.** The solid curve uses combined {math}`|E_{\mathrm{in}}+E_r|^2` for {math}`|\mathbf E|^2` in {math}`p_{\mathrm{abs}}`; the dashed curve uses separate {math}`|E_{\mathrm{in}}|^2+|E_r|^2`. Their difference is the cross term {math}`2\operatorname{Re}(E_{\mathrm{in}}^*E_r)`.

**3. Normalize by incident-only power.** Because the structure is laterally uniform, per unit area {math}`p_{\mathrm{abs}}` gives absorption per 1 nm of height. Divide by incident-only power at the reference, {math}`I_{\mathrm{in}}(0)=\tfrac12\operatorname{Re}(q)/Z_0`, and use {math}`\omega\varepsilon_0Z_0=k_0`:

```{math}
\frac{p_{\mathrm{abs}}(z)}{I_{\mathrm{in}}(0)}
=\frac{k_0\operatorname{Im}\varepsilon_r}{\operatorname{Re}q}\,|E(z)|^2
=2k_0\operatorname{Im}q\,|E(z)|^2.
```

The second equality follows because transverse wavevector is real: taking imaginary parts of {math}`q^2=\varepsilon_r-(k_x/k_0)^2` gives {math}`\operatorname{Im}\varepsilon_r=2\operatorname{Re}q\operatorname{Im}q`. Here {math}`2k_0\operatorname{Im}q=0.010098` /nm, so the horizontal quantity, **absorption per 100 nm, is {math}`1.009817\,|E(z)|^2`**.

Check this using the incident wave alone: {math}`I_{\mathrm{in}}(z)/I_{\mathrm{in}}(0)=e^{2k_0\operatorname{Im}q\,z}` differentiates to {math}`2k_0\operatorname{Im}q\,|E_{\mathrm{in}}(z)|^2`. Its power loss per height equals absorption density calculated from its own field.

**4. Values at selected heights.** All are absorption per 100 nm, normalized by incident-only power at the reference.

| Height {math}`z` | Actual: combined field | Waves calculated separately | Interference effect |
| ---: | ---: | ---: | --- |
| 0 (reference plane) | 1.830479 | 1.156470 | More absorption |
| About 45.49 nm | 1.691179 | 1.691179 | Boundary between bright and dark fringes |
| About 112.45 nm | 2.420896 | 3.190552 | Less absorption |

**5. Absorption over an interval is an integral.** Integrating the solid curve from the reference to about 112.45 nm gives actual `top` absorption 2.053040. It equals the difference in net power at those two heights; the figure script checks this by numerical integration.

A closed form was possible because `top` is uniform and contains only two plane waves. Inside a patterned layer, permittivity varies with x and y. Reconstruct fields as in [Chapter 14](fields.md), then integrate isotropic absorption density {math}`\tfrac12\omega\varepsilon_0\operatorname{Im}\varepsilon_r(x,y)\,|\mathbf E|^2` over the unit-cell area. This gives absorption at that height; integrating once more in height gives interval absorption. For lossy permeability, add {math}`\tfrac12\omega\mu_0\operatorname{Im}\mu_r\,|\mathbf H|^2`. If only whole-stack absorption is needed, read `absorption`, the net-power difference across the two boundaries, without this volume integration.

::::

In the right panel, interference is zero near 112.45 nm. Substituting that height shows that reference-plane interference equals **the power absorbed less between those planes than when calculating the waves separately**. Reduced absorption in the dark fringe from roughly 45.49–112.45 nm builds the contribution to +0.328; added absorption in the bright fringe at 0–45.49 nm reduces it to +0.158416 on reaching the reference.

Splitting absorption in `top` over the same 0–112.45 nm interval gives:

| Incident component | Reflected component | Interference | Actual total absorption |
| ---: | ---: | ---: | ---: |
| 2.112882 | 0.098574 | −0.158416 | 2.053040 |

The incident component is the incident-only decrease discussed earlier; the reflected component is power lost by the reflected wave while traveling upward through the interval. Actual total absorption 2.053040 equals the difference in net power at its endpoints. At the reference plane:

| Incident-only power | Reflected power returning | Reduced absorption in `top` due to interference | Net power crossing the reference |
| ---: | ---: | ---: | ---: |
| 1 | −0.145227 | +0.158416 | 1.013189 |

Because the layer and `bot` are lossless, the arriving 1.013189 becomes {math}`T_{\mathrm{total}}`. **The interference contribution, including the excess 0.013189 above 1, is not newly created energy: it is power that reaches the layers instead of being absorbed by `top` in separate-wave calculations.** Its sign depends on reflection phase. Another phase can increase absorption in `top`, making net inflow smaller than {math}`1-R`.

Changing `top` loss below changes its electric interference pattern, absorption, and cross contribution together. In this structure with identical layer and `bot`, transmission is propagating in a lossless exit medium, so {math}`\operatorname{Im}r>0` and interference is nonnegative. If the layer differs from `bot`, its thickness changes reflection phase and interference can be negative.

<section class="exercise power-flux-exercise" aria-labelledby="power-flux-title">
<h3 id="power-flux-title">Try it · How does loss in top change interference?</h3>
<p class="concept">Concept explorer · Calculates the same single interface as this section using closed forms. Im ε<sub>r</sub> = 3 reproduces the Wavebit table above; 0 gives lossless incidence. Fields in the left panel are instantaneous values oscillating in time.</p>
<div class="controls">
<label for="pf-loss">Im ε<sub>r</sub> of top (Re ε<sub>r</sub> = 4) <output id="pf-loss-out" for="pf-loss">3.0</output><input id="pf-loss" type="range" min="0" max="6" step="0.1" value="3"></label>
<button id="pf-play" class="pf-play" type="button">Pause</button>
</div>
<div class="svg-scroll"><svg id="pf-fig" viewBox="0 0 680 400" role="img" aria-label="Three panels share height z above the reference plane. Left: instantaneous incident, reflected, and combined electric fields in top. Center: absorption by height with and without interference. Right: interference contribution at height z."></svg></div>
<p class="pf-legend" id="pf-legend"><span class="pf-swatch pf-more"></span>Bright fringe: more absorption from interference<span class="pf-swatch pf-less"></span>Dark fringe: less absorption from interference</p>
<div class="readout" id="pf-readout" aria-live="polite"><p>Reference: incident-only 1 − reflection 0.145227 + interference 0.158416 = net entering 1.013189 = T</p></div>
<noscript><p>Enable JavaScript to change loss in top and watch the electric fields move.</p></noscript>
</section>

### Absorption inside a layer even without reflection

Now assign the same {math}`\varepsilon_r=2.16+0.9i`, {math}`\mu_r=1` to `top`, the finite layer, and `bot`. With no material discontinuity, reflection is zero. At vacuum wavelength 500 nm and normal incidence, compare power through a 75 nm interval:

| {math}`R_{\mathrm{total}}` | {math}`P_{\mathrm{enter}}` | {math}`T_{\mathrm{total}}` | Absorption {math}`A` |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 0.568084 | 0.431916 |

This material has complex refractive index {math}`\widetilde n=1.5+0.3i`. Power attenuation through thickness {math}`d` agrees with:

```{math}
T_{\mathrm{total}}
=\exp\!\left[-\frac{4\pi\operatorname{Im}(\widetilde n)d}{\lambda}\right].
```

There is no reflection, but about 43.2% of incident power is absorbed within the specified thickness. `absorption` measures this absorption between the two reference boundaries.

### Running the example

Executable example: {download}`10_boundary_power.py <../../../examples/rcwa/learning/learn/10_boundary_power.py>`

```sh
python 10_boundary_power.py
```

The example prints Wavebit results and compares them with Fresnel single-interface power, net power from period-averaging the product of physical time signals, and exponential attenuation in a uniform lossy medium. Its calculated values supply this section's tables and Figure 13-5. Figure 13-6 reconstructs the two waves in `top` from the recorded reflection coefficient and incident normal wavevector; the figure script checks height-resolved power balances again by numerical integration.

## Conditions for comparing calculations

Match not only materials but actual incident transverse wavevectors, the s/p amplitude bases, reference planes, and normalization denominator. In lossy media especially, numerically equal angles may describe different waves under different angle definitions. Also check whether the compared quantity is **reflected-wave-only power or net power from combined incident and reflected fields**.

```{dropdown} The s/p boundary amplitudes used in the code
In `src/wavebit/observations/power.py`, `wave_power` computes the power of a wave traveling in one direction, while `net_power` computes the net power of the incident and reflected waves combined at the same boundary.

Internally, the solver uses the tangential electric field as the primary amplitude in the s channel and the normalized tangential magnetic field `Z₀H` in the p channel. Let `κ` be the normal wavevector used by that channel divided by the vacuum wavevector:

| Internal channel | Primary amplitude | Factor used to convert to power |
| --- | --- | --- |
| s (TE) | Tangential electric field E | `κ/μr` |
| p (TM) | Tangential magnetic field Z₀H | `κ/εr` |

Both channels omit the common power factor `1/(2Z₀)`, which cancels between the numerator and denominator of an efficiency. If the input electric-field coefficient for p polarization is `aₚ`, the internal magnetic-field coefficient is `(n/μr)aₚ`, so the incident power includes `|n/μr|²`. Here `n²=εr μr`.

The public results `r_s`, `r_p`, `t_s`, and `t_p` are electric-field amplitude coefficients. They must be distinguished from the magnetic-field amplitude used internally in the p channel.
```

[Byrnes, *Multilayer optical calculations*, §4.3–4.4 and Appendix B](https://arxiv.org/html/1603.02720v5#A2) also explains why net boundary inflow differs from {math}`1-R` for absorbing incidence. See the [half-space power validation record](../../validation/rcwa/half-space-power-audit.md) for Wavebit's numerical validation scope.

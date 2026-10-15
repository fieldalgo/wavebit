(complex-kx-roots)=
# Appendix A. Choosing kz in a lossless medium: two roots, branch points, and branch cuts

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

In a half-space—the semi-infinite medium above or below the structure—the normal wavevector {math}`k_z` is determined by a square root, so there are always two candidates. We must select the one describing a wave traveling away from the structure. In a lossless medium with real transverse wavevector {math}`k_x`, that choice follows automatically. When complex `theta` makes {math}`k_x` complex, however, **a separate rule is needed**. Considering only lossless media, this appendix explains from the beginning why such a rule is needed, how the rules differ, and which rules Wavebit and other solvers use. {ref}`Appendix B <complex-kx-lossy-gain>` then examines the changes introduced by absorption and gain.

This material is relevant when using complex `theta` ({ref}`Chapter 8 <complex-transverse-wavevector>`) or comparing results with other solvers. For calculations using only real angles, it is sufficient to read through {ref}`Why is the root determined automatically for real kx? <ckr-real-kx>`.

This appendix describes Wavebit's half-space root selection (`boundary_kz`) as of October 2, 2026. On that date, Wavebit changed from **path D**, introduced below, to the **Maystre rule**, {math}`\operatorname{Re}k_z+\operatorname{Im}k_z>0`. For passive media with real {math}`k_x`, the two rules are bitwise identical, so no real-angle result changed. This appendix retains path D as “Wavebit before October 2, 2026,” examining where and why the two rules differ. Section 8 of [Branch handling](../../theory/rcwa/branch-cuts.md) gives the reasons for the change and the measurements.

The figures were drawn with `tools/docs/learning/draw_complex_kx_roots.py`. Values computed with Wavebit or RETICOLO are recorded in `docs/validation/learning-records/complex-kx-roots.json`, together with the Wavebit commit used. Wavebit values obtained with path D were measured on a commit preceding the rule change.

(ckr-premises)=
## Assumptions of this appendix

The following four assumptions hold throughout. Breaking any of them changes the conclusions.

1. **The time convention is {math}`e^{-i\omega t}`**, as in {ref}`Chapter 13 <power-and-efficiency>`. Every sign-based directional statement below, such as decay for {math}`\operatorname{Im}k_z>0`, uses this convention. Under the engineering convention {math}`e^{+j\omega t}`, all imaginary-part signs are reversed.
2. **The medium is a lossless dielectric.** Its {math}`\varepsilon` is positive and real ({math}`\operatorname{Im}\varepsilon=0`, {math}`\varepsilon>0`), so it neither absorbs nor amplifies light. Air and ideal glass are examples. Its refractive index {math}`n=\sqrt\varepsilon` is also positive and real, placing the branch points {math}`k_x=\pm n`, introduced below, on the real axis. {ref}`Appendix B <complex-kx-lossy-gain>` relaxes this assumption for absorbing and gain media.
3. **The medium is nonmagnetic ({math}`\mu=1`) and isotropic.** Thus {math}`n=\sqrt\varepsilon` and {math}`k_z^2=\varepsilon-k_x^2`. Magnetic media are not covered.
4. **Root selection is needed only in the upper and lower semi-infinite half-spaces.** A finite-thickness layer includes both {math}`+k_z` and {math}`-k_z`, so the answer does not depend on which is called “outgoing” ([Time-convention audit](../../validation/rcwa/time-convention-audit.md)). Every {math}`k_z` discussed here is a half-space {math}`k_z`.

**Directional reference.** This appendix writes the half-space field as {math}`e^{ik_zd}`, where {math}`d` is distance away from the structure. Thus {math}`\operatorname{Re}k_z>0` means propagation away from the structure, and {math}`\operatorname{Im}k_z>0` means decay with distance. With the upward-positive global {math}`z` of Chapter 8, the lower half-space has {math}`d=-z`, so the sign of {math}`k_z` appears reversed.

## Part 1. Concepts: from two roots to branch cuts

(ckr-two-roots)=
### Why are there two kz values?

Consider a plane wave in a homogeneous medium. All wavevectors are divided by {math}`k_0=2\pi/\lambda`; correspondingly, distance {math}`d` is multiplied by {math}`k_0`. In a medium of refractive index {math}`n`, the plane-wave wavevector has a prescribed magnitude {math}`n`, giving the component relation called the dispersion relation:

```{math}
k_x^2+k_z^2=\varepsilon .
```

Here {math}`\varepsilon=n^2`. If {math}`k_y` is present, replace {math}`k_x^2` by {math}`k_x^2+k_y^2`. The component {math}`k_x` runs along the interface. Phase must match along that interface in every layer, so the incident illumination and lattice period determine {math}`k_x`, which stays unchanged when the medium changes. Only the remaining component, {math}`k_z`, must be found anew in each medium. Solving the dispersion relation gives:

```{math}
k_z=\pm\sqrt{\varepsilon-k_x^2}.
```

A square root has **two** answers. These roots represent two different waves sharing the same {math}`k_x`. Writing the field as {math}`e^{ik_zd}` reveals two properties:

- For {math}`\operatorname{Re}k_z>0`, phase advances in the positive {math}`d` direction. Including the time factor {math}`e^{-i\omega t}`, constant-phase surfaces move toward increasing {math}`d` as time advances. The wave travels **away from the structure**: outgoing.
- For {math}`\operatorname{Im}k_z>0`, {math}`|e^{ik_zd}|=e^{-\operatorname{Im}k_z\,d}`. The field **decays** with distance.

```{figure} ../../figures/learn/learn/a01-two-roots.svg
:alt: Left: the kz plane. For kx=0.5, the roots +0.866 and −0.866 lie on the real axis; for kx=1.5, +1.118i and −1.118i lie on the imaginary axis. Each pair is joined by a dotted line through the origin, showing central symmetry. +0.866 and +1.118i are filled points; the negative roots are open points. Four panels on the right show fields versus distance k0·d. For +0.866, a faded curve at a slightly later time shifts outward, away from the structure; −0.866 travels toward it. +1.118i decays exponentially with distance, whereas −1.118i grows exponentially.
:figclass: learning-figure

Figure A-1. Two roots share the same {math}`k_x`. In air ({math}`\varepsilon=1`), {math}`k_x=0.5` gives {math}`k_z=\pm0.866`, while {math}`k_x=1.5` gives {math}`k_z=\pm1.118i`. The small plots show how each root's field changes with distance from the structure. This is a conceptual illustration.
```

Figure A-1 shows two cases in air ({math}`\varepsilon=1`):

- For {math}`k_x=0.5`, {math}`k_z=\pm0.866` is real. The {math}`+0.866` wave travels away from the structure; {math}`-0.866` travels toward it.
- For {math}`k_x=1.5`, {math}`k_z=\pm1.118i` is purely imaginary. The {math}`+1.118i` wave decays with distance; {math}`-1.118i` grows.

The roots are always symmetric about the origin, {math}`k_z` and {math}`-k_z`. The task is to **select one**. Wavebit's `boundary_kz` answers precisely this question: “What is the half-space {math}`k_z` of the wave traveling away from the structure?”

(ckr-real-kx)=
### Why is the root determined automatically for real kx?

Physics supplies two criteria:

- **Outgoing:** {math}`\operatorname{Re}k_z>0`. There is no source in the exterior half-space, so energy may flow only away from the structure: the radiation condition.
- **Decaying:** {math}`\operatorname{Im}k_z>0`. A field must not grow without bound in a region containing no source.

Dividing the {math}`k_z` plane into four quadrants makes both criteria visible. Throughout this appendix, quadrant one is teal, quadrant two blue, quadrant three gray, and quadrant four orange. Figure A-8 is the one exception in which colors have a different meaning, explained there.

| Quadrant | Outgoing | Decaying |
| --- | --- | --- |
| First | Yes | Yes |
| Second | No (incoming) | Yes |
| Third | No (incoming) | No (growing) |
| Fourth | Yes | No (growing) |

For real {math}`k_x`, one root is always in the first quadrant, including its boundary. This follows in three steps.

**① A complex number is an arrow; a square root halves its angle.** A complex number is an arrow extending from the origin, with a length and an angle measured counterclockwise from the real axis. Squaring squares the length and doubles the angle. Taking a square root therefore **halves** the angle. A number at 180°, a negative real number, has a square root at 90°, a purely imaginary number.

**② Real {math}`k_x` places {math}`k_z^2` on the real axis.** We have {math}`k_z^2=\varepsilon-k_x^2`. Under {ref}`assumption 2 <ckr-premises>`, {math}`\varepsilon` is real; real {math}`k_x` also gives real {math}`k_x^2`. Subtracting one real number from another leaves real {math}`k_z^2`, with no imaginary part:

```{math}
\operatorname{Im}k_z^2=\operatorname{Im}(\varepsilon-k_x^2)=\operatorname{Im}\varepsilon=0 .
```

The zero imaginary part is not guaranteed by a numerical rule. It comes from **assumption 2**, losslessness: {math}`\varepsilon` has no imaginary part.

The sign splits the real axis into two cases. Since {math}`n^2=\varepsilon`, {math}`k_z^2=n^2-k_x^2`:

- A **propagating order**, {math}`|k_x|<n`, has {math}`k_z^2>0`: the positive real axis, at 0°. In air, {math}`k_x=0.5` gives {math}`k_z^2=1-0.25=0.75`.
- An **evanescent order**, {math}`|k_x|>n`, has {math}`k_z^2<0`: the negative real axis, at 180°. In air, {math}`k_x=1.5` gives {math}`k_z^2=1-2.25=-1.25`.

**③ Halving the angle places the root on the first-quadrant boundary.**

- Half of 0° is 0°, the positive real axis; half of 180° is 90°, the positive imaginary axis. Both form the boundary of **quadrant one**.
- The opposite root differs only in sign, {math}`-k_z`, and lies directly across the origin, at 180° or 270°.
- In quadrant one, including its boundary, both real and imaginary parts are nonnegative, so neither the outgoing nor the decaying criterion is violated. The opposite root violates one criterion.

We therefore choose the first-quadrant root. The criteria never point to different roots, so **the root follows automatically without a separate rule**.

```{figure} ../../figures/learn/learn/a02-half-angle.svg
:alt: Left: the kz² plane for air, with the real axis emphasized in thick teal. Its right half is labeled propagating (kz²>0, 0 degrees) and its left half evanescent (kz²<0, 180 degrees). Points mark kz²=0.75 at kx=0.5 (circle) and −1.25 at kx=1.5 (square), with a 180-degree arc around the origin. The middle arrow reads square root, divide angle by 2. Right: the quadrant-colored kz plane, with the positive real and imaginary axes—the first-quadrant boundary—in thick teal and a 90-degree arc. Selected roots 0.866 (0 degrees) and 1.118i (90 degrees) are filled points on that boundary; opposite roots −0.866 and −1.118i are open points across the origin.
:figclass: learning-figure

Figure A-2. A square root halves the angle. In a lossless medium with real {math}`k_x`, {math}`k_z^2` lies on the real axis at 0° or 180°. Its square roots are a pair: one on the first-quadrant boundary, at 0° or 90°, and the opposite root across the origin. This is a conceptual illustration.
```

The two examples in Figure A-2 are:

| Case | {math}`k_z^2` (angle) | Selected root (angle) | Opposite root |
| --- | --- | --- | --- |
| Air, {math}`k_x=0.5` | 0.75 (0°) | +0.866 (0°) | −0.866 (180°) |
| Air, {math}`k_x=1.5` | −1.25 (180°) | +1.118i (90°) | −1.118i (270°) |

```{figure} ../../figures/learn/learn/a03-real-kx.svg
:alt: The kz plane is colored by quadrant, with a legend and brief explanation on the right. A dark curve traces the selected root as kx increases from 0 to 2 in air: for kx below 1 it follows the real axis from 1 to 0, then rises along the imaginary axis to 1.73i when kx exceeds 1. It therefore follows the first-quadrant boundary. The rejected root −kz follows a faint dotted path opposite it, along the negative real and negative imaginary axes.
:figclass: learning-figure

Figure A-3. For real {math}`k_x` in a lossless medium, the selected root follows the real and imaginary axes, the first-quadrant boundary. Computed and drawn with `tools/docs/learning/draw_complex_kx_roots.py`.
```

**A criterion ties on a boundary.** Since the roots lie on the first-quadrant boundary in a lossless medium, one criterion cannot distinguish them.

- For propagation at 0°, the roots are on the real axis and their imaginary parts are exactly zero. Decay alone ties; the outgoing criterion chooses {math}`+0.866`.
- For evanescence, whose squared-root argument is 180°, the situation reverses. The roots are on the imaginary axis, the selected one at 90°, and their real parts are exactly zero. Outgoing alone ties; decay chooses {math}`+1.118i`.

In neither case do the two criteria select different roots. When one ties, the other decides. The evanescent tie returns in {ref}`When continuation ties at real kx <ckr-tie>`, and the propagating tie in {ref}`the decay-only rule <ckr-branch-cut>`. Propagating and evanescent orders are distinguished as in [Chapter 12](results.md).

(ckr-complex-kx)=
### What changes for complex kx?

**Where does complex {math}`k_x` come from?** In Wavebit it comes only through complex `theta` ({ref}`Chapter 8 <complex-transverse-wavevector>`). Real `theta` is the angle of light from the normal, giving {math}`k_x=n\sin\theta\cos\phi`, where {math}`\phi` is azimuth. Complex `theta` has no such directional meaning: no direction in space is tilted by `0.2 + 0.1i` rad. But {math}`\sin` is defined for complex arguments, so inserting complex `theta` into the same expression produces {math}`k_x`. **Discarding the original meaning (direction) while retaining the form of the expression is called a formal use.** Here `theta` is a calculation parameter that generates the desired complex {math}`k_x`, not a direction. For example, in air, {math}`\theta=0.2+0.1i` gives complex {math}`k_x`. Supplying complex {math}`k_x` directly to Wavebit, however, is refused with guidance to use complex `theta`.

**What happens to {math}`k_x` for complex `theta`?** Under {ref}`assumption 2 <ckr-premises>`, the incident medium's {math}`n` is real too, so any imaginary part of {math}`k_x` comes only from that of `theta`. Working step by step:

1. With the incidence plane in {math}`x`–{math}`z`, {math}`\phi=0` and {math}`\cos\phi=1`, so {math}`k_x=n\sin\theta`.
2. Write {math}`\theta=a+ib`, separating real part {math}`a` and imaginary part {math}`b`. Complex sine and cosine are defined through exponentials:

   ```{math}
   \sin z=\frac{e^{iz}-e^{-iz}}{2i},\qquad \cos z=\frac{e^{iz}+e^{-iz}}{2}.
   ```

   Substituting {math}`z=ib` gives the hyperbolic functions {math}`\cosh b=(e^{b}+e^{-b})/2` and {math}`\sinh b=(e^{b}-e^{-b})/2`: {math}`\cos(ib)=\cosh b`, {math}`\sin(ib)=i\sinh b`. The sine addition formula also holds for complex numbers, giving:

   ```{math}
   \sin(a+ib)=\sin a\cos(ib)+\cos a\sin(ib)=\sin a\cosh b+i\cos a\sinh b .
   ```

   For small {math}`b`, {math}`\cosh b\approx1` and {math}`\sinh b\approx b`.
3. Multiplying by {math}`n` separates the real and imaginary parts of {math}`k_x`:

   ```{math}
   \operatorname{Re}k_x=n\sin a\cosh b,\qquad \operatorname{Im}k_x=n\cos a\sinh b .
   ```

Two consequences follow:

- **A small imaginary part moves {math}`k_x` only slightly.** For small {math}`b`, {math}`\operatorname{Re}k_x\approx n\sin a`, essentially the real-`theta` value, while {math}`\operatorname{Im}k_x\approx n\cos a\cdot b` is small and proportional to {math}`b`. For {math}`n=2`, `theta = 0.4` gives {math}`k_x/k_0=0.778837`, and `theta = 0.4 + 1e-12j` gives {math}`0.778837+1.8\times10^{-12}i`, because {math}`2\cos0.4\approx1.8`. Complex `theta` with a small imaginary part is therefore a slight perturbation of real `theta`.
- **When the real part is π/2, {math}`k_x` is real regardless of the imaginary part.** Since {math}`\cos(\pi/2)=0`, {math}`\operatorname{Im}k_x=0`; since {math}`\sin(\pi/2)=1`, {math}`\operatorname{Re}k_x=n\cosh b`. For {math}`\theta=\pi/2+i\alpha`, {math}`\cosh\alpha>1` makes {math}`k_x=n\cosh\alpha` a real number larger than {math}`n`. Then even in the incident medium, {math}`k_z^2=n^2-k_x^2<0`: the incident wave itself is evanescent. This is **evanescent incidence**, a case where complex `theta` does not make {math}`k_x` complex.

Complex {math}`k_x` means that the incident intensity itself changes exponentially along the surface: the magnitude of {math}`e^{ik_xx}` is {math}`e^{-\operatorname{Im}k_x\,x}`. This is formal illumination, not laboratory illumination. `R` and `T` can be calculated, but are only formal ratios, not percentages of incident power. `power_entering` and `absorption` are `NaN`, and every solve emits `ComplexIncidenceWarning`.

**What changes?** An imaginary part of {math}`k_x` introduces an imaginary part into {math}`k_z^2=\varepsilon-k_x^2`. Squaring {math}`k_x` gives {math}`\operatorname{Im}k_x^2=2\operatorname{Re}k_x\operatorname{Im}k_x`; because {math}`\varepsilon` is real under {ref}`assumption 2 <ckr-premises>`:

```{math}
\operatorname{Im}k_z^2=-\operatorname{Im}k_x^2=-2\operatorname{Re}k_x\operatorname{Im}k_x .
```

If {math}`\operatorname{Re}k_x\operatorname{Im}k_x>0`, this is negative, moving {math}`k_z^2` into the lower half-plane, between 180° and 360°. Halving the angle puts one root between 90° and 180°, in quadrant two; its opposite lies between 270° and 360°, in quadrant four. The two roots occupy **quadrants two and four**. Conversely, if {math}`\operatorname{Re}k_x\operatorname{Im}k_x<0`, {math}`k_z^2` is in the upper half-plane and the roots are in quadrants one and three. The first-quadrant root then satisfies both criteria and is selected automatically, as for real {math}`k_x`.

```{figure} ../../figures/learn/learn/a04-complex-pairs.svg
:alt: Two kz planes above an x–z plane. (a) For kx=1.5−0.3i, the roots ±(0.393+1.146i) occupy quadrants one and three, and one root satisfies both criteria. (b) For kx=1.5+0.3i, ±(0.393−1.146i) occupy quadrants four and two: the outgoing root grows, and the decaying root is incoming. Both panels mark the kx=1.5 pair ±1.118i with gray crosses. (c) Intensity in the x–z plane for the outgoing root kz=0.393−1.146i of (b). The horizontal axis is surface coordinate k0·x from −5 to 3; the vertical axis is distance k0·z from 0 to 2.6. Background color is log10 intensity, pale for weak fields and dark orange for strong ones. A gray strip at the bottom marks the structure; surface intensity increases toward the left, proportional to exp(−0.6x). Five black rays rise obliquely at slope 0.26, parallel to color-band boundaries, with constant intensity along each ray. At x=1.5, an upward-arrow vertical dotted line explains that intensity increases upward because it encounters rays originating farther left. From blue points at z=0.5 and z=1.5 on this line, blue dotted paths follow their rays to surface points x=−0.41 and x=−4.23, respectively; each pair of endpoints has the same color.
:figclass: learning-figure

Figure A-4. For complex {math}`k_x`, the roots may occupy quadrants one and three, or two and four. In the latter case the criteria disagree. Panel (c) shows in real space why the outgoing root of (b) grows with distance: intensity remains constant along each ray, but increasing distance encounters rays originating farther left, at a stronger point on the surface. Computed and drawn with `tools/docs/learning/draw_complex_kx_roots.py`.
```

Panels (a) and (b) of Figure A-4 use {math}`k_x=1.5-0.3i` and {math}`k_x=1.5+0.3i` in air:

- {math}`k_x=1.5-0.3i`: {math}`\operatorname{Im}k_z^2=+0.9`. The roots {math}`\pm(0.393+1.146i)` lie in quadrants one and three. The first-quadrant {math}`0.393+1.146i` is outgoing and decaying; the third-quadrant {math}`-0.393-1.146i` violates both. The first-quadrant root is selected automatically.
- {math}`k_x=1.5+0.3i`: {math}`\operatorname{Im}k_z^2=-0.9`. The roots {math}`\pm(0.393-1.146i)` lie in quadrants two and four. The outgoing root, {math}`0.393-1.146i`, grows; the decaying root, {math}`-0.393+1.146i`, is incoming. **One criterion must be abandoned.**

**A growing root is not necessarily wrong.** Consider the field {math}`e^{i(k_xx+k_zz)}` using the outgoing root {math}`k_z=0.393-1.146i` of Figure A-4(b). Here {math}`z` is distance away from the structure, like {math}`d` above, not Chapter 8's global {math}`z`. Splitting wavevector {math}`\mathbf k=(k_x,\ k_z)` into real and imaginary parts gives:

```{math}
\operatorname{Re}\mathbf k=(1.5,\ 0.393),\qquad \operatorname{Im}\mathbf k=(0.3,\ -1.146).
```

1. **Intensity varies only along {math}`\operatorname{Im}\mathbf k`.** Intensity is amplitude squared, {math}`|E|^2`, proportional to power density:

   ```{math}
   \left|e^{i(k_xx+k_zz)}\right|^2=e^{-2(\operatorname{Im}k_x\,x+\operatorname{Im}k_z\,z)} .
   ```

   Only the imaginary wavevector appears in the exponent. Intensity decreases along {math}`\operatorname{Im}\mathbf k` and is constant in perpendicular directions. Here it decreases as {math}`e^{-0.6x}` along surface direction {math}`+x`, but grows away from the structure because {math}`\operatorname{Im}k_z` is negative.
2. **The wave travels along {math}`\operatorname{Re}\mathbf k`.** Its phase-propagation direction is {math}`\operatorname{Re}\mathbf k=(1.5,\ 0.393)`: almost along {math}`x`, gradually moving away from the structure.
3. **The two directions are perpendicular.** Perpendicularity is tested by a zero dot product:

   ```{math}
   \operatorname{Re}\mathbf k\cdot\operatorname{Im}\mathbf k
   =\operatorname{Re}k_x\operatorname{Im}k_x+\operatorname{Re}k_z\operatorname{Im}k_z
   =1.5\times0.3+0.393\times(-1.146)\approx0.45-0.45=0 .
   ```

   With unrounded values it is exactly zero. This is not accidental. Taking imaginary parts of the dispersion relation {math}`k_x^2+k_z^2=\varepsilon`, using {math}`\operatorname{Im}(a^2)=2\operatorname{Re}a\operatorname{Im}a`, gives:

   ```{math}
   2\,(\operatorname{Re}k_x\operatorname{Im}k_x+\operatorname{Re}k_z\operatorname{Im}k_z)=\operatorname{Im}\varepsilon .
   ```

   The parentheses are precisely that dot product, and in a lossless medium {math}`\operatorname{Im}\varepsilon=0`. Thus {math}`\operatorname{Re}\mathbf k` and {math}`\operatorname{Im}\mathbf k` are always perpendicular in a lossless medium. Combined with step 1, this says intensity is constant in the propagation direction: following a ray gives constant intensity, while intensity differs between rays.
4. **This explains growth with distance.** Measure intensity while moving straight away from the structure at fixed {math}`x_0`. Each unit increase in {math}`z` corresponds to an increase of {math}`1.5/0.393=3.82` in {math}`x` along a ray, so the ray encountered at height {math}`z` originated at {math}`x_0-3.82z` on the surface. Surface intensity is already stronger farther left, by step 1's {math}`e^{-0.6x}`. Increasing distance therefore encounters rays originating at stronger, farther-left points. The growth factor agrees too: each unit increase in height shifts the origin left by 3.82, increasing intensity by {math}`e^{0.6\times3.82}=e^{2.29}`. This equals {math}`e^{2\times1.146}=e^{2.29}`, calculated directly from step 1.

Growth along {math}`z` is therefore not amplification of light by anything. It is the oblique-ray projection of intensity decreasing along surface direction {math}`x`, as Figure A-4(c) shows. Since the incident illumination is itself a formal wave varying along the surface, growth of diffraction orders along {math}`z` is not surprising ([Time-convention audit](../../validation/rcwa/time-convention-audit.md), section 2).

Decay alone is consequently not a valid criterion here. But choosing outgoing alone—always selecting the **principal root**, {math}`\operatorname{Re}k_z>0`—is not sufficient either.

**What criterion should be used?** This appendix uses **continuity**: a small departure from a physical situation, real {math}`k_x`, should produce only a small change in the answer. The justification comes from the meaning of complex {math}`k_x`. Fourier decomposition of a real source's field along the surface contains only real {math}`k_x` components, and each component's root is fixed as in the previous section. Complex {math}`k_x` is not such a component; it is a formal extension of the answer fixed on the real axis. The extension must therefore approach the real-{math}`k_x` answer as it approaches that axis ([Time-convention audit](../../validation/rcwa/time-convention-audit.md), section 3). In a lossless incident medium, a small imaginary part of `theta` perturbs {math}`k_x` only slightly, as shown above, so the same criterion can be stated in terms of `theta`. Where the principal root violates it is examined in {ref}`When continuation ties at real kx <ckr-tie>` and {ref}`Two sweeps <ckr-sweeps>`.

(ckr-branch-point)=
### What is a branch point, and why is it a problem?

**What are we choosing?** For each {math}`k_x`, the two candidates are {math}`+k_z` and {math}`-k_z`, but only one enters the half-space calculation. We must decide which represents the half-space wave every time {math}`k_x` is specified. “Selecting a root” always means this here. The continuity criterion adds a condition: if {math}`k_x` changes slightly, the selected {math}`k_z` must also change only slightly. A simple procedure achieves this: move {math}`k_x` in small steps, each time choosing whichever candidate is closer to the previously selected value. We call this “following continuously.”

**Where the roots meet.** In air ({math}`\varepsilon=1`), increasing real {math}`k_x` from zero gives:

| {math}`k_x` | Candidates {math}`\pm\sqrt{1-k_x^2}` |
| --- | --- |
| 0 | ±1 |
| 0.9 | ±0.436 |
| 0.99 | ±0.141 |
| 1 | Both 0 |
| 1.01 | ±0.142i |
| 1.5 | ±1.118i |

The candidates differ only in sign, always facing one another across the origin. As {math}`k_x` approaches 1, {math}`k_z^2=1-k_x^2` approaches zero, so both candidates approach the origin and one another. At {math}`k_x=1`, both are zero: **the same number**. Beyond 1, they separate upward and downward along the imaginary axis. “The roots meet at a point” means that the two candidates coincide at zero in this way.

In general, {math}`k_z^2=n^2-k_x^2`, so this happens at {math}`k_x=\pm n`. Such a {math}`k_x` is a **branch point**. Since {math}`n` is positive and real under {ref}`assumption 2 <ckr-premises>`, these points lie on the real axis. At {math}`k_z=0`, the wave runs parallel to the interface rather than away from it. In a grating this is a Rayleigh point, where an order appears or disappears ([Chapter 19](convergence.md)).

**Why is this a problem?** There are two reasons.

1. **Nearby roots are difficult to distinguish.** Far from a branch point, the candidates are well separated and the one closer to the previous value is clear. Near it, they crowd together, and at the branch point they are identical. {ref}`Below <ckr-tie>` we encounter a case where passing exactly through a branch point along the real axis makes “the closer root” undecidable.
2. **One circuit exchanges the roots.** Move {math}`k_x` once around a circle enclosing the branch point in the complex plane, following continuously. On returning to the starting {math}`k_x`, the selected candidate is **the other one**, not the initial choice. Each {math}`k_x` specifies one incidence condition, so this is a computational experiment that changes incidence gradually around a closed loop. Because it passes through complex {math}`k_x`, it is a numerical experiment, not a laboratory one.

```{figure} ../../figures/learn/learn/a05-loop.svg
:alt: Left: the kx plane. Red loop A encloses branch point kx=1, with center 1 and radius 0.35, starting at 0.65. Blue loop B has center 0.3 and radius 0.2 and does not enclose it. Right: the continued-root trajectories in the kz plane. Loop A starts at +0.760 and follows an upper semicircle to −0.760. Loop B traces a small closed curve near 0.995 and returns to its starting point.
:figclass: learning-figure

Figure A-5. One circuit around a branch point changes the root to its opposite. A loop that does not enclose a branch point returns to the original root. Computed and drawn with `tools/docs/learning/draw_complex_kx_roots.py`.
```

Figure A-5 demonstrates the second property in air, with branch point {math}`k_x=1`:

- Loop A, red, encloses the branch point. Starting at {math}`k_x=0.65` with candidate {math}`+0.760`, one full circuit returns to the same {math}`k_x=0.65` with {math}`-0.760`.
- Loop B, blue, does not enclose it. It starts with {math}`+0.995` and returns to {math}`+0.995`.

Dividing loop A into four 90° steps gives the table below. Merely selecting the candidate closer to the previous value at each step makes the selected root travel around the origin to its opposite.

| Angle around the loop | {math}`k_x` | Candidates | Continued root | Continued-root angle |
| --- | --- | --- | --- | ---: |
| 0° (start) | 0.650 | ±0.760 | +0.760 | 0° |
| 90° | 1.000 − 0.350i | ±(0.645 + 0.542i) | +0.645 + 0.542i | 40° |
| 180° | 1.350 | ±0.907i | +0.907i | 90° |
| 270° | 1.000 + 0.350i | ±(0.645 − 0.542i) | **−0.645 + 0.542i** | 140° |
| 360° (back at the start) | 0.650 | ±0.760 | **−0.760** | 180° |

The reason is again angle halving:

1. {math}`k_z^2=(n-k_x)(n+k_x)`. Near branch point {math}`n`, the factor {math}`(n+k_x)\approx2n` changes little, so {math}`k_z^2` is proportional to {math}`(k_x-n)`.
2. One 360° circuit of {math}`k_x` around exactly one branch point makes {math}`k_z^2` circle zero once. If a loop encloses both branch points, each factor, {math}`(n-k_x)` and {math}`(n+k_x)`, makes one circuit, so {math}`k_z^2` circles zero twice.
3. The square root halves the angle. One circuit of {math}`k_z^2` moves {math}`k_z` through only 180°, ending at {math}`-k_z`. Two circuits of {math}`k_z^2` are needed to return {math}`k_z` to its start. Thus a loop enclosing one branch point must be traversed twice, while one enclosing both returns after a single circuit.

An analogy is a **two-level spiral parking garage**, with the branch point as its central column. The two roots are the first and second levels. One circuit around the column reaches level two; another circuit returns to level one. Unlike a normal garage, there is no third level. Mathematically, these levels are the two sheets of a Riemann surface.

**Consequently**, the continuously followed answer at the same {math}`k_x` depends on which side of a branch point the path passed. Continuous following alone cannot assign a unique answer to every {math}`k_x`; somewhere we must prescribe a jump to the opposite candidate. Rules differ in where they place that jump, as discussed under {ref}`Branch cuts <ckr-branch-cut>`. The interactive figure below lets you try a circuit yourself.

```{raw} html
<section class="exercise ckr-exercise" aria-labelledby="ckr-title">
<h3 id="ckr-title">Try it · What happens after one circuit around a branch point?</h3>
<p class="concept">Concept explorer · In air (ε = 1), follow whichever candidate is closer to the previous value. This is not an actual RCWA calculation.</p>
<div class="controls">
<label for="ckr-angle">Angle traveled around the loop <output id="ckr-angle-out" for="ckr-angle">0°</output><input id="ckr-angle" type="range" min="0" max="720" step="1" value="0" list="ckr-turns"><datalist id="ckr-turns"><option value="0"></option><option value="360"></option><option value="720"></option></datalist></label>
<fieldset class="ckr-loop"><legend>Loop</legend>
<label><input type="radio" name="ckr-loop" id="ckr-loop-a" value="a" checked>Loop A · Encloses the branch point (center 1, radius 0.35)</label>
<label><input type="radio" name="ckr-loop" id="ckr-loop-b" value="b">Loop B · Does not enclose it (center 0.3, radius 0.2)</label>
</fieldset>
<button id="ckr-play" class="ckr-play" type="button">Pause</button>
</div>
<div class="svg-scroll"><svg id="ckr-fig" viewBox="0 0 680 340" role="img" aria-label="Two panels. The left kx plane shows the dotted loop, branch point kx = 1, starting point (small square), path traveled so far, and current kx. The right kz plane shows the current candidates ±√(1 − kx²) as open circles, the continued root as a filled point and its trajectory, and the starting root as a small square."></svg></div>
<div class="readout" id="ckr-readout" aria-live="polite" aria-atomic="true">
<p id="ckr-kx">Loop A · 0° · kx = 0.650</p>
<p id="ckr-cands">Candidates ±√(1 − kx²): +0.760, −0.760</p>
<p id="ckr-roots">Continued root +0.760 · Starting root +0.760</p>
<p id="ckr-status" class="ckr-status">At the starting point. After one circuit (360°), compare with the initial root.</p>
</div>
<noscript><p>Enable JavaScript to follow the loop and see which candidate is selected.</p></noscript>
</section>
```

(ckr-continuation)=
### How does continuation determine a root?

As the previous section showed, a continuously followed answer may depend on the path taken. Fixing one starting point and one path in advance therefore assigns one answer to every {math}`k_x`. This is a **continuation** rule.

1. Start where the answer is **known**.
2. Move {math}`k_x` **gradually** to its destination.
3. At each step, choose **the root closer to the preceding value**, avoiding an abrupt jump to its opposite.

The root held at the destination is the answer. The rules differ in their **starting points**:

- **Path D:** start at normal incidence, {math}`k_x=0`, and follow a straight line in the {math}`k_x` plane to the destination. Normal incidence has the unambiguous answer {math}`k_z=n`. This was **Wavebit's rule before October 2, 2026**.
- **Path E:** start at the real {math}`k_x` sharing the destination's real part—the **nearest physical situation**—and change only the imaginary part to its destination value.

D and E are names assigned in this appendix. [Time-convention audit](../../validation/rcwa/time-convention-audit.md) and [Branch handling](../../theory/rcwa/branch-cuts.md) also use alphabetic path names, but the same letter has different meanings there. Their “path D” continues from the nearest real {math}`k_x` and corresponds to this appendix's path E.

```{figure} ../../figures/learn/learn/a06-paths-d-e.svg
:alt: Left: the kx plane. Path D is a straight red line from the origin, normal incidence, to destination 1.5+0.3i, passing above branch point kx=1. Path E is a purple vertical line from real kx=1.5 to the destination. Right: continued roots in the kz plane. Path D starts at 1, descends into quadrant four, and reaches 0.393−1.146i, outgoing and growing. Path E starts at +1.118i and reaches −0.393+1.146i in quadrant two, incoming and decaying. The endpoints are opposite roots.
:figclass: learning-figure

Figure A-6. Two paths with different starting points reach opposite roots at the same destination. The destination's imaginary part, 0.3, is deliberately large for visibility. Computed and drawn with `tools/docs/learning/draw_complex_kx_roots.py`.
```

Figure A-6 uses destination {math}`k_x=1.5+0.3i`. The same effect occurs for imaginary parts of 0.05 or {math}`10^{-9}`.

- Path D, red, passes **above** the branch point and reaches {math}`0.393-1.146i`: outgoing and growing.
- Path E, purple, does not pass the branch point and reaches {math}`-0.393+1.146i`: incoming and decaying.

The answers are **opposite roots** because of the previous section's loop. Path E starts from {math}`+1.118i`, the answer at real {math}`k_x=1.5`. That is also the value reached from the origin by passing **below** the branch point to {math}`k_x=1.5`; see Figure A-7 in the {ref}`next section <ckr-tie>`. Along the lower route, {math}`\operatorname{Im}k_x<0` and {math}`\operatorname{Re}k_x\operatorname{Im}k_x<0`, so {math}`k_z^2` stays in the upper half-plane and the root remains in quadrant one. The root rotates there from 0°, the {math}`+1` at {math}`k_x=0`, to 90°, reaching {math}`+1.118i`. Path D instead passes **above** the branch point. Joining the two routes makes a loop enclosing the branch point once, which exchanges the roots; hence the opposite endpoints.

**Path D need not be walked step by step.** In a lossless dielectric, it is simply the principal root, {math}`\operatorname{Re}k_z>0`, which can be computed directly. That is what Wavebit did before October 2, 2026. The explanation has two steps:

1. **The principal root is discontinuous only on the negative real axis of {math}`k_z^2`.** It measures the argument of {math}`k_z^2` between −180° and 180° and assigns the root half that angle, between −90° and 90°. Thus the root always lies in the right half-plane. Crossing the negative real axis downward changes the argument from 180° to −180°, making the principal-root argument jump from 90° to −90°, to the opposite root. Elsewhere it is continuous.
2. **Path D does not cross that axis.** In the {math}`k_x` plane it is a straight route from zero to the destination. Points on it are {math}`s\,k_x`, where {math}`s` increases from zero to one. Mapping into the {math}`k_z^2` plane and writing {math}`t=s^2` gives:

   ```{math}
   k_z^2(t)=\varepsilon-t\,k_x^2,\qquad t:0\to1 .
   ```

   This is a straight segment from {math}`\varepsilon` to {math}`\varepsilon-k_x^2`. Its imaginary part is {math}`-t\operatorname{Im}k_x^2`. If {math}`\operatorname{Im}k_x^2\ne0`, it vanishes only at the start, {math}`t=0`. The segment therefore touches the real axis only at its starting point, the positive real {math}`\varepsilon`. It never crosses the negative real axis, so the principal root continues without a jump.

The remaining case and the pre-October-2, 2026 implementation are as follows:

- If {math}`\operatorname{Im}k_x^2=0`, the segment lies on the real axis. If {math}`k_x` is real and greater than {math}`n`, it passes through the branch point, {math}`k_z^2=0`, and continuation ties. The {ref}`next section <ckr-tie>` treats this tie; old Wavebit used a separate **tie-break rule** there.
- Thus, in a lossless dielectric, **Wavebit before October 2, 2026 = principal root + tie-break rule**: choose {math}`\operatorname{Im}k_z>0` when {math}`|\operatorname{Re}k_z|\le10^{-12}`.
- That implementation also included a **continuity correction**, `_off_principal_branch`, flipping the principal root to {math}`-k_z` when the segment crossed the negative real axis. It never activated in a lossless dielectric for the reason above. It was removed when the rule changed to Maystre.

The drawing script uses the **closed form of path D**—principal root plus continuity correction—instead of walking it. It separately checks that this expression agrees with step-by-step continuation, and that it equals the principal root, with the correction never activating, at all 834603 grid and random points across three lossless dielectrics, {math}`\varepsilon=1,\ 2.25,\ 4`, before drawing. On the 288 complex-{math}`k_x` cases in the random lossless sample used {ref}`below <ckr-branch-cut>`, old Wavebit's `boundary_kz` matched the principal root in all 288; current `boundary_kz` matches Maystre in all 288.

(ckr-tie)=
### Why cannot path D determine the answer at real kx?

This occurs when two conditions coincide:

- The medium is **lossless**, {ref}`assumption 2 <ckr-premises>`, placing the branch point on the real axis at {math}`k_x=n`.
- The destination is **real with {math}`k_x>n`**: an evanescent order at a real incident angle.

Path D is straight from zero to the destination. A real destination keeps it on the real axis, making it pass **exactly through** {math}`k_x=n`. It does not merely graze the branch point; it crosses its center.

```{figure} ../../figures/learn/learn/a07-through-branch-point.svg
:alt: (a) Three paths D start at the origin in the kx plane. The orange path to 1.5+0.3i passes above branch point kx=1; the teal path to 1.5−0.3i passes below; the black path to real 1.5 crosses it. (b) Their root endpoints in the kz plane: orange reaches 0.393−1.146i and teal 0.393+1.146i. Black follows the real axis from 1 to 0, then splits into two dotted branches toward +1.118i and −1.118i, with question marks. (c) A close-up near the branch point: kz=0.01414 at kx=0.9999, and both candidates at kx=1.0001, +0.01414i and −0.01414i, are 0.0200 away.
:figclass: learning-figure

Figure A-7. When path D passes exactly through a branch point, “choose the closer root” cannot decide. The figure shows three paths differing only in the destination's imaginary part, plus a close-up beside the branch point. Computed and drawn with `tools/docs/learning/draw_complex_kx_roots.py`.
```

Figure A-7(a) shows the three paths; (b) shows their root endpoints:

- Positive imaginary part, {math}`+`, orange: passes **above** the branch point and reaches the growing root {math}`0.393-1.146i`.
- Negative imaginary part, {math}`-`, teal: passes **below** and reaches the decaying root {math}`0.393+1.146i`.
- Zero imaginary part, black: passes **through** the branch point. Which way should it go?

Figure A-7(c) enlarges the immediate neighborhood. At the branch point, both candidates become zero.

- One step before, at {math}`k_x=0.9999`, {math}`k_z=0.01414` is real.
- One step after, at {math}`k_x=1.0001`, the candidates are {math}`+0.01414i` and {math}`-0.01414i`.
- Both are **exactly 0.0200** from the preceding value.

The closer-root rule cannot decide. No step-size choice fixes this: before the branch point the value is real; afterward the candidates are positive and negative purely imaginary numbers, marked {math}`+` and {math}`-`. Their distances from a real value are always equal. Walking cannot resolve the tie. This is precisely the {ref}`boundary tie described earlier <ckr-real-kx>`.

A minor implementation detail then decides. Trying it directly shows that changing one inequality character reverses the answer:

```python
import cmath

root = 1.0                                   # kz = 1 at kx = 0
for i in range(1, 3001):
    k = i * 1.5 / 3000                       # Follow the real axis to 1.5; branch point at i = 2000
    v = cmath.sqrt(1 - k * k)
    root = v if abs(v - root) <= abs(-v - root) else -v   # Try replacing <= with <
print(root)                                  # 1.118j; changing to < gives -1.118j
```

Wavebit before October 2, 2026 therefore resolved this tie, where {math}`\operatorname{Re}k_z` is zero, using an explicit rule rather than path D: when the real-part magnitude is at most `1e-12`, choose {math}`\operatorname{Im}k_z>0`, the **decaying root**. That is the physically correct answer and agrees with path E. Direct queries to old and current `boundary_kz` give the following values in air:

| {math}`k_x` | {math}`k_z` before October 2, 2026 | What decided it? | Current {math}`k_z` (Maystre) |
| --- | --- | --- | --- |
| {math}`1.5-10^{-9}i` | +1.118034i (decaying) | Path D below the branch point | +1.118034i (decaying) |
| {math}`1.5` | +1.118034i (decaying) | Separate tie-break rule | +1.118034i (decaying) |
| {math}`1.5+10^{-9}i` | −1.118034i (growing) | Path D above the branch point | +1.118034i (decaying) |

**Old Wavebit answered like path E on the real axis, then switched to path D immediately above it.** The answers therefore differed just above and below the axis. Path D jumps to the opposite root when crossing the evanescent part of the real axis vertically. A line across which the selected root jumps is a **branch cut**, discussed {ref}`below <ckr-branch-cut>`. Path D's cut thus lies on the evanescent part of the real axis. Since path D equals the principal root for a lossless dielectric, this is also the principal-root cut.

Current Wavebit chooses the decaying root in all three rows. Evanescent orders on the real axis are selected without a tie, eliminating the tie-break rule. Maystre's branch cut meets the real axis only at a branch point, as discussed {ref}`below <ckr-half-plane-rules>`.

**Propagating orders**, {math}`|k_x|<n`, do not have this problem: the path ends before reaching a branch point.

(ckr-half-plane-rules)=
### What do direct selection rules look like?

Instead of continuation, a rule can directly select a root according to **which half of the {math}`k_z` plane contains it**. Draw a boundary through the origin and choose one side: exactly one of the origin-symmetric roots lies there. Three common rules are:

- **Principal root:** choose the right half-plane, {math}`\operatorname{Re}k_z>0`. In a lossless dielectric this equals path D, Wavebit's pre-October-2, 2026 rule ({ref}`Continuation <ckr-continuation>`).
- **Maystre:** choose the side above-right of the −45° line, {math}`\operatorname{Re}k_z+\operatorname{Im}k_z>0`. Wavebit since October 2, 2026, RETICOLO, and S4 use this rule.
- **{math}`\operatorname{Im}k_z>0`:** choose the upper half-plane, retaining only a decaying root. fmmax uses this rule.

Figure A-8 applies them to the roots at {math}`k_x=1.5+0.3i`. In this figure alone, shaded colors indicate the selected half-plane rather than quadrants. Colors identify rules: red for path D/principal root, blue for Maystre, gray for fmmax, and purple for the earlier path E. The same line colors identify rules in Figures A-9 and A-10.

```{figure} ../../figures/learn/learn/a08-half-planes.svg
:alt: Three kz planes. Left: the principal-root rule, equivalent to path D in a lossless medium and to Wavebit before October 2, 2026, shades the right half-plane Re kz>0, with the imaginary axis as boundary, selecting 0.393−1.146i, outgoing and growing. Middle: Maystre, used by Wavebit, RETICOLO, and S4, shades Re kz+Im kz>0, with the −45-degree line as boundary, selecting −0.393+1.146i, incoming and decaying. Right: fmmax shades the upper half-plane Im kz>0, bounded by the real axis, selecting −0.393+1.146i. Selected roots are filled points; rejected roots are open points.
:figclass: learning-figure

Figure A-8. Each rule draws a boundary through the origin in the {math}`k_z` plane and chooses one side. Roots on the boundary tie; crossing it exchanges the selected root with its opposite. Computed and drawn with `tools/docs/learning/draw_complex_kx_roots.py`.
```

| Rule | Selected half-plane | Boundary | Selected root at {math}`k_x=1.5+0.3i` |
| --- | --- | --- | --- |
| Principal root (= path D, Wavebit before October 2, 2026) | {math}`\operatorname{Re}k_z>0` | Imaginary axis | 0.393 − 1.146i (outgoing, growing) |
| Maystre (Wavebit, RETICOLO, S4) | {math}`\operatorname{Re}k_z+\operatorname{Im}k_z>0` | −45° line | −0.393 + 1.146i (incoming, decaying) |
| {math}`\operatorname{Im}k_z>0` (fmmax) | {math}`\operatorname{Im}k_z>0` | Real axis | −0.393 + 1.146i (incoming, decaying) |

Details for each solver are collected under {ref}`What each code does <ckr-codes>`.

::::{dropdown} Equations: mapping each rule's discontinuity into the kx plane
A rule jumps when the root lies on its boundary line. Rewriting those conditions in terms of {math}`k_x^2` gives the following ({math}`u\ge0`).

```{math}
\begin{aligned}
\text{Principal root:}\quad & k_z\ \text{on the imaginary axis} && \Rightarrow\ k_z^2\le0 && \Rightarrow\ k_x^2=\varepsilon+u,\\
\text{Maystre:}\quad & k_z\ \text{on the }-45^\circ\text{ line} && \Rightarrow\ k_z^2\ \text{negative imaginary} && \Rightarrow\ k_x^2=\varepsilon+iu,\\
\operatorname{Im}k_z>0:\quad & k_z\ \text{on the real axis} && \Rightarrow\ k_z^2\ge0 && \Rightarrow\ k_x^2=\varepsilon-u .
\end{aligned}
```

For air ({math}`\varepsilon=1`), these become:

- For the principal root, {math}`k_x^2=1+u`: the **real axis with {math}`|k_x|\ge1`**. This is the same conclusion as in the preceding section.
- Maystre gives half of a hyperbola with {math}`\operatorname{Re}k_x^2=1` and {math}`\operatorname{Im}k_x^2\ge0`. It curves up and to the right from the branch point +1, and down and to the left from −1.
- For {math}`\operatorname{Im}k_z>0`, {math}`k_x^2=1-u`: the **real axis with {math}`|k_x|\le1`, together with the entire imaginary axis**.

Path E is not a rule defined by drawing a boundary line in the {math}`k_z` plane, so it cannot be written this way. Its lines extend straight upward from +n and straight downward from −n. Across {math}`\operatorname{Re}k_x=n`, the starting point changes from the propagating side ({math}`<n`) to the evanescent side ({math}`>n`), and raising the imaginary part above zero then produces opposite roots. On the negative-imaginary side, the two starting points continue to the same root, so there is no line there.
::::

(ckr-branch-cut)=
### Why branch cuts cannot be eliminated

```{figure} ../../figures/learn/learn/a09-cut-maps.svg
:alt: Four maps of the kx plane in air, one for each rule. Background colors show the quadrant of the selected root: teal for quadrant I, blue for II, orange for IV, and gray for III. Thick lines are branch cuts. Path D, used by Wavebit before October 2, 2026, has cuts on the outer real axis, kx>1 and kx<−1. Path E extends straight up from +1 and straight down from −1. Maystre, used by Wavebit, RETICOLO, and S4, follows half-hyperbolas curving up and right from +1 and down and left from −1. fmmax has cuts along the real axis between the branch points, −1<kx<1, and the entire imaginary axis.
:figclass: learning-figure

Figure A-9. The roots selected by four rules in the {math}`k_x` plane (colors) and their branch cuts (thick lines), for air. At a color boundary without a thick line, the root merely passes smoothly through an axis. `tools/docs/learning/draw_complex_kx_roots.py` locates the actual jumps on a grid and checks them against the equations before drawing the figure.
```

Figure A-9 maps four rules in air: the three from the preceding section plus path E. The **color** identifies the quadrant of the root selected at that {math}`k_x`, using the same quadrant colors as before. A **thick line** marks where crossing the line makes the selected root jump to its opposite, from {math}`k_z` to {math}`-k_z`: a branch cut. The line color identifies the rule.

**Why a line must exist.** As we saw earlier, going once around a loop enclosing just one branch point changes the root. Yet a rule must assign one answer to each {math}`k_x`. It must therefore jump to the opposite root somewhere on that loop. Connecting these jump locations produces a line starting at the branch point: a branch cut. **A branch cut cannot be eliminated; only its location can be chosen.**

::::{dropdown} Derivation: why the cuts must extend to infinity
A loop enclosing both branch points flips the root twice and returns it to its starting value. Mathematically, a short line connecting the two branch points would therefore suffice. An example is {math}`k_z=i\,k_x\sqrt{1-\varepsilon/k_x^2}`. However, this rule gives {math}`+1.118i` at {math}`k_x=1.5` and {math}`-1.118i` at {math}`k_x=-1.5`. Since {math}`k_x` and {math}`-k_x` are mirror images, their answers must agree. A physical rule must therefore depend only on {math}`k_x^2`.

Set {math}`w=k_x^2`. Then {math}`k_z=\pm\sqrt{\varepsilon-w}`, with just one branch point in the {math}`w` plane, at {math}`w=\varepsilon`. A loop enclosing {math}`\varepsilon` once flips the root, however large the loop is. If the cut ended at a finite point, a larger loop enclosing that endpoint too would flip the root without crossing the cut at all. The cut must therefore extend from {math}`\varepsilon` to infinity. Mapping back into the {math}`k_x` plane gives one line extending to infinity from each branch point, as in path D, path E, and Maystre.

The fmmax cut is likewise a single half-line extending leftward from {math}`\varepsilon` to infinity in the {math}`w` plane, {math}`w=\varepsilon-u`. This half-line passes through {math}`w=0`. In the {math}`k_x` plane, its portion with {math}`0\le w\le\varepsilon` becomes the real-axis segment connecting the branch points, while its portion with {math}`w<0` becomes the entire imaginary axis extending upward and downward from {math}`k_x=0`.
::::

| Rule | Branch cut in air | Relation to physical cases (real {math}`k_x`) |
| --- | --- | --- |
| Path D (Wavebit before October 2, 2026) | Real axis extending outward from the branch points ({math}`k_x>1`, {math}`k_x<-1`) | **Every evanescent order lies on the cut** |
| Path E | Straight upward from +1, straight downward from −1 | Touches only at the branch point |
| Maystre (Wavebit, RETICOLO, S4) | Half-hyperbolas: up and right from +1, down and left from −1 | Touches only at the branch point |
| {math}`\operatorname{Im}k_z>0` (fmmax) | Real axis between the branch points ({math}`-1<k_x<1`) plus the imaginary axis | **Every propagating order lies on the cut** |

**A color change does not necessarily mean a jump.** At a color boundary without a thick line, the root simply **passes smoothly through an axis** of the {math}`k_z` plane. For example, {math}`k_x=0.5+0.05i` is a complex {math}`k_x` on the propagating side. Path D, path E, and Maystre all select {math}`0.8679-0.0288i`. Although this root grows slightly, it connects smoothly to the answer {math}`0.866` at real {math}`k_x=0.5`. Only the {math}`\operatorname{Im}k_z>0` rule selects {math}`-0.8679+0.0288i`, an incoming wave. Real {math}`k_x=0.5` lies on that rule's cut, so adding even a small imaginary part crosses it.

The map colors also show what each rule gives up.

- In air, a lossless dielectric, path D never selects {math}`\operatorname{Re}k_z<0`: only teal and orange appear. This is because path D is the principal root in a lossless dielectric.
- The {math}`\operatorname{Im}k_z>0` rule never selects {math}`\operatorname{Im}k_z<0`: only teal and blue appear.
- Path E and Maystre vary with location.

**The numerical comparison.** For 1496 randomly sampled pairs of lossless media, with real {math}`\varepsilon` between 1 and 6, and real {math}`k_x` between 0 and 3, all four rules select **the same root in all 1496 cases**. This agrees with the earlier conclusion that real {math}`k_x` fixes the root unambiguously. Adding only {math}`\pm10^{-8}i`, however, makes some rules flip the root. The counts are:

| Rule | Number of flips out of 1496 | Orders that flip |
| --- | ---: | --- |
| Path D (Wavebit before October 2, 2026) | **597** | Every evanescent order |
| Path E | 0 | |
| Maystre (Wavebit) | 0 | |
| {math}`\operatorname{Im}k_z>0` (fmmax) | **899** | Every propagating order |

The 1496 cases contain 597 evanescent orders and 899 propagating orders. Path D flips every evanescent order, without exception, while fmmax flips every propagating order.

Path E and Maystre are **nearly the same rule** near the real axis. Take 288 of the same lossless samples, add imaginary parts between −0.5 and 0.5 to make {math}`k_x` complex, and count how often each rule selects the same root as step-by-step continuation along paths D and E:

| Rule | Agrees with path D | Agrees with path E |
| --- | ---: | ---: |
| Path D (Wavebit before October 2, 2026) | 288/288 | 238/288 |
| Maystre (Wavebit) | 240/288 | **286/288** |
| {math}`\operatorname{Im}k_z>0` (fmmax) | 130/288 | 180/288 |

(ckr-sweeps)=
### What the location of a branch cut changes in practice

Two sweeps make the difference visible.

```{figure} ../../figures/learn/learn/a10-sweeps.svg
:alt: Panel (a) shows the kx plane near the branch point kx=1: path D's cut on the real axis for kx>1, path E's cut straight upward from 1, Maystre's cut curving upward from 1, and two sweeps. Sweep A is a horizontal arrow moving the real part from 0.8 to 1.2 at imaginary part 0.05. Sweep B is a vertical arrow moving the imaginary part from −0.05 to +0.05 at real part 1.5. Panel (b) plots Im kz for sweep A. Path D decreases smoothly from −0.07 to −0.67; path E and Maystre jump from −0.22 to +0.22 near Re kx=1. Panel (c) plots Im kz for sweep B. Only path D jumps, from +1.118 to −1.118 at zero imaginary part; path E and Maystre remain smooth at +1.118.
:figclass: learning-figure

Figure A-10. When sweeping the angle with a complex angle (sweep A), path E and Maystre jump once near the Rayleigh point. When slightly perturbing a physical case (sweep B), only path D jumps. Calculated and plotted with `tools/docs/learning/draw_complex_kx_roots.py`.
```

**Sweep A:** keep the imaginary part at 0.05 and move the real part from 0.8 to 1.2. This represents an angular sweep at a fixed imaginary part of the complex angle; the path crosses **above** the branch point.

- Path D does not jump. Its branch cut lies on the real axis and does not intersect this path.
- Path E jumps between {math}`\operatorname{Re}k_x=1.0000` and {math}`1.0005`, from {math}`0.226-0.221i\to-0.225+0.222i`.
- Maystre jumps between {math}`1.0010` and {math}`1.0015`. The exact location is {math}`\sqrt{1+0.05^2}=1.00125` on the hyperbola. As Figure A-10(a) shows, near the branch point its cut almost coincides with path E's.

**Sweep B:** keep the real part at 1.5 and move the imaginary part from −0.05 to +0.05. This slightly perturbs the physical case {math}`k_x=1.5`.

- Only path D jumps at zero imaginary part, {math}`+1.118i\to-1.118i`.
- Path E and Maystre remain smooth.

The implications are:

- Moving the cut off the real axis, as path E and Maystre do, leaves one jump when an angular sweep **passes the Rayleigh point**. Even for real {math}`k_x`, the Rayleigh point is already where the result has a kink, a Wood anomaly. Thus this does not introduce a new singularity; it puts the jump immediately beside the existing one.
- If the cut lies on the real axis, as in path D, the entire evanescent interval becomes a place where “even a slight perturbation causes a jump.” Grating calculations always include higher evanescent orders, so in almost every grating calculation some order lies on this cut. We check this in {ref}`a real grating <ckr-grating>`.
- This is why Wavebit changed from path D to Maystre on October 2, 2026. The sweep-A jump remains immediately beside the Rayleigh point, but the sweep-B jump disappears.

## Part 2. Wavebit and other solvers

(ckr-wavebit-rule)=
### Wavebit's rule and its caveats

The discussion above translates into the following Wavebit behavior, effective October 2, 2026.

- **Rule:** half-space {math}`k_z` follows **Maystre's rule**, {math}`\operatorname{Re}k_z+\operatorname{Im}k_z>0`. For real {math}`k_x` in a lossless dielectric, it agrees with the principal root. Before October 2, 2026, Wavebit used path D.
- **Real {math}`k_x`:** all four rules select the same root, so root selection does not cause disagreement with other solvers.
- **Complex `theta` and evanescent orders in lossless half-spaces:** Maystre's cut touches the real axis only at the Rayleigh point. Adding a small imaginary part to `theta` therefore does not flip evanescent roots. In {ref}`the grating example <ckr-grating>`, path D, used by Wavebit before October 2, 2026, jumps even for an imaginary part of 1e-9; the newer Wavebit agrees with RETICOLO. One jump remains immediately beside the Rayleigh point ({ref}`the two sweeps <ckr-sweeps>`). Complex `theta` remains a formal calculation, as explained in {ref}`Chapter 8 <complex-transverse-wavevector>`.
- **`UnresolvedBranchWarning`:** this warns when floating-point precision, especially complex64, cannot resolve the sign of the quantity used to select a root ([Branch handling](../../theory/rcwa/branch-cuts.md), Sections 4 and 8). Under path D, this issue was tied to its branch cut: the small imaginary part left in {math}`k_x` by complex64 could cross the cut lying on the real axis. With the newer rule, real {math}`k_x` no longer triggers this warning through half-space root selection; it can still arise from labeling layer modes.

More background appears in [Branch handling](../../theory/rcwa/branch-cuts.md) and the [time-convention audit](../../validation/rcwa/time-convention-audit.md).

(ckr-codes)=
### What each code does

The rules used by the codes discussed here are collected below. None of these codes uses path E.

| Code | Root rule | Complex {math}`k_x` input | Branch cut (lossless case) |
| --- | --- | --- | --- |
| Wavebit | Maystre since October 2, 2026. Previously path D, equivalent to the principal root plus a tie-break in a lossless dielectric | Direct input is refused with guidance; available only through complex `theta` | Half-hyperbolas; previously the real-axis evanescent interval |
| RETICOLO V10 | Maystre for plane waves, `retsqrt(x, -1)` | Direct `beta0 = n sin(theta)` input | Half-hyperbolas |
| S4 | Execution results agree with Maystre | Not accepted | Agrees with Maystre |
| fmmax 1.7.1 | {math}`\operatorname{Im}k_z>0`, `_select_eigenvalues_sign` | Uses complex `in_plane_wavevector` unchanged | Real-axis propagating interval plus the imaginary axis |

**RETICOLO.** Half-space plane waves use `retsqrt(x, -1)`. The third argument, `tet`, sets the angle of the square-root cut in the {math}`k_z^2` plane. Its default π/2 puts the cut on the negative imaginary axis of {math}`k_z^2`. The source comment calls this the “coupure de Maystre,” and the result agrees with Maystre's rule. Setting `tet` to π gives the principal root. Omitting the second argument (`k = 0`) gives the decay-first “coupure de Petit”: {math}`\operatorname{Im}k_z>0`, with {math}`\operatorname{Re}k_z\ge0` when the imaginary part is zero. This agrees with fmmax's rule. Two comparison traps matter:

- By default, `res2` reports only propagating orders, with `tolh` and `tolb` set to 1e-6. Complex `beta0` then produces an empty list. To obtain every order, pass `parm` as the **third argument** to `res2` and increase the tolerances. Putting these settings in the `parm` passed to `res1` has no effect.
- When configured to report every order, it reports {math}`|\text{amplitude}|^2` as an efficiency even for evanescent orders at exactly real `theta`. The sum of efficiencies therefore jumps at that single point ({ref}`the grating example <ckr-grating>`).

**S4.** Complex {math}`k_x` is not accepted. Instead, S4 constructs {math}`k_x` from the angle and the real part of the frequency. Giving it a complex frequency keeps {math}`k_x` real while making only {math}`k_0=\omega/c` complex, so {math}`k_x/k_0` becomes complex. Results obtained this way agreed with Maystre. This was checked by execution, without reading the root-selection implementation.

**fmmax.** Even for real {math}`k_x`, `s21[0, 0]` has the opposite sign to `Ey_ref/Ey_inc`, so reflection amplitudes are read from reconstructed fields.

(ckr-grating)=
### How much difference does it make in a grating?

We now solve an actual grating all the way through.

- Air is above and a substrate with {math}`\varepsilon=2.25` is below, with one grating layer between them.
- Half of each period in the grating layer has {math}`\varepsilon=9`, and the other half has {math}`\varepsilon=1`.
- Thickness 0.15, period 0.7, wavelength 0.9, all in the same length unit; TE polarization; orders ±5.
- Illumination uses {math}`\theta=0.2+i\operatorname{Im}\theta`.

Both half-spaces, air and {math}`\varepsilon=2.25`, are lossless, so the example satisfies this appendix's assumptions. In air, {math}`k_x` is 0.199 for order 0, 1.484 for order +1, and −1.087 for order −1. **Both ±1 orders are evanescent**, so at real `theta` both sit on path D's branch cut.

| {math}`\operatorname{Im}\theta` | Path D r0 | Path D R0 | Wavebit/RETICOLO r0 | Wavebit/RETICOLO R0 |
| ---: | --- | ---: | --- | ---: |
| −1e-3 | 0.091499 − 0.257051i | 0.074447 | 0.151070 + 0.092571i | 0.031392 |
| −1e-9 | 0.092037 − 0.256094i | 0.074055 | 0.149508 + 0.094586i | 0.031299 |
| 0 | 0.149508 + 0.094586i | 0.031299 | 0.149508 + 0.094586i | 0.031299 |
| +1e-9 | −0.006161 − 0.062574i | 0.003953 | 0.149508 + 0.094586i | 0.031299 |
| +1e-6 | −0.006163 − 0.062574i | 0.003954 | 0.149507 + 0.094588i | 0.031299 |
| +1e-3 | −0.007288 − 0.063145i | 0.004040 | 0.147873 + 0.096525i | 0.031184 |

Here r0 is the zeroth-order reflection amplitude and R0 the zeroth-order reflection efficiency. For complex `theta`, these are formal ratios. The path-D columns were calculated with Wavebit before October 2, 2026. The newer Wavebit, using Maystre, agrees with RETICOLO within {math}`4\times10^{-15}` in all six rows, so they share columns.

- At real `theta`, all three amplitudes agree: 0.149508 + 0.094586i.
- Path D jumps for an imaginary part of **only 1e-9**. A positive imaginary part makes positive evanescent orders, such as order +1 in air, cross the cut; a negative imaginary part does so for negative evanescent orders, such as order −1 in air. The sign of {math}`\operatorname{Re}k_x\operatorname{Im}k_x` differs between orders. Since different orders flip for the two signs, the result jumps **on both sides**, and the zeroth-order efficiency changes from 0.031 to 0.004 or 0.074.
- The newer Wavebit and RETICOLO are smooth on both sides. Even an imaginary part of 1e-3 changes the result only slightly.
- Wavebit's tests compare the same grating with RETICOLO at {math}`\operatorname{Im}\theta=\pm0.1` as well, with agreement down to {math}`2\times10^{-15}`.

If RETICOLO is configured to report every order, the summed reflection efficiency jumps to 0.182 only in the row with {math}`\operatorname{Im}\theta=0`; the other rows are around 0.031. It counts the evanescent orders' {math}`|\text{amplitude}|^2` as efficiency too. This is unrelated to root selection, so the table compares only order 0.

(ckr-beyond-lossless)=
## Relaxing the lossless assumption

This appendix's conclusions depend on {ref}`assumption 2 <ckr-premises>`, losslessness. {ref}`Appendix B <complex-kx-lossy-gain>` relaxes that assumption alone, keeping the other three, and examines what changes.

- **If the incident medium absorbs**, real `theta`, using the phase-angle convention, and complex `theta`, using the formal {math}`n\sin\theta`, produce different {math}`k_x`. Complex `theta` is no longer a slight perturbation of real `theta` ({ref}`absorbing incident media <ckr-absorbing-incidence>`).
- **If the half-space whose root is being selected absorbs**, its branch points move off the real axis, and all four rules agree near real {math}`k_x`. This is also where Wavebit's pre-October-2, 2026 continuity correction operated ({ref}`absorbing half-spaces <ckr-lossy>`).
- **If the half-space amplifies light, or has gain**, the two criteria can point to different roots even at real {math}`k_x`. A rule is then needed, and causality determines the answer ({ref}`gain media <ckr-gain>`).
- Code differences that emerge only with absorption or gain are collected in {ref}`Appendix B's code comparison <ckr-b-codes>`.

## One-page summary

- Assumptions: the {math}`e^{-i\omega t}` convention; a lossless dielectric with {math}`\varepsilon>0`; nonmagnetic, isotropic media; half-space {math}`k_z`.
- As a square root, {math}`k_z` has two candidates. For real {math}`k_x`, the roots lie on the boundary of quadrant I, either the real or the imaginary axis. When one criterion ties, the other decides. The two criteria never point to opposite roots, so the choice is unambiguous. This happens because taking a square root halves the angle.
- At complex {math}`k_x`, if {math}`\operatorname{Re}k_x\operatorname{Im}k_x>0`, then {math}`\operatorname{Im}k_z^2<0` and the criteria disagree. A growing root is not automatically wrong. The test is whether it connects to the answer at real {math}`k_x`: **continuity**.
- The two candidates coincide at zero at the branch points, {math}`k_x=\pm n`. Following them continuously around a loop enclosing one branch point exchanges them. Every rule therefore has **a branch cut starting at a branch point**; rules differ in where they put that cut.
- Path D, used by Wavebit before October 2, 2026, is simply the principal root in a lossless dielectric, and its continuity correction never activates. Its cut is **the evanescent interval on the real axis**. For a real-axis evanescent order, path D passes directly through a branch point, making continuation ambiguous. The tie-break supplies the decaying root there; moving just above the axis makes path D supply the opposite root. Thus an imaginary part of 1e-9 can flip evanescent orders.
- fmmax's cut is **the propagating interval on the real axis**, together with the imaginary axis. Maystre, used by Wavebit, RETICOLO, and S4, and path E move their cuts off the real axis: upward from +n and downward from −n. These two rules agree in 286/288 lossless samples near the real axis.
- Wavebit has used Maystre since October 2, 2026. For real {math}`k_x`, it agrees bit for bit with path D; it does not jump under a small perturbation off the real axis; and it agrees with RETICOLO for the grating example.

## Terms

| Term | Meaning in this appendix | Other names |
| --- | --- | --- |
| Outgoing / incoming | {math}`\operatorname{Re}k_z>0` / {math}`<0`: a wave traveling away from / toward the structure | Radiation condition |
| Decaying / growing | {math}`\operatorname{Im}k_z>0` / {math}`<0`: a wave weakening / strengthening with distance | Decaying / growing |
| Propagating / evanescent | An order whose real {math}`k_x` has magnitude smaller / larger than {math}`n` | Propagating / evanescent order |
| Lossless dielectric | A medium with positive real {math}`\varepsilon`, which neither absorbs nor amplifies light | Lossless dielectric |
| Formal | Borrowing the algebraic expression without its original physical interpretation. Complex `theta` is a computational parameter for constructing {math}`k_x`, not a direction, and the resulting `R`, `T` are not ratios to incident power | Formal |
| Principal root | The root with {math}`\operatorname{Re}k_z>0`; at a zero-real-part tie, the tie-break selects {math}`\operatorname{Im}k_z>0` | Principal root |
| Tie-break | The rule selecting {math}`\operatorname{Im}k_z>0` when {math}`\lvert\operatorname{Re}k_z\rvert\le10^{-12}`, used for half-spaces by Wavebit before October 2, 2026 | Tie-break |
| Branch point | A {math}`k_x`, here {math}`=\pm n`, where both candidates {math}`\pm k_z` become zero and coincide | Branch point |
| Branch cut | A locus where the rule jumps to the opposite root, starting at a branch point | Branch cut |
| Rayleigh point | A {math}`k_x` where an order just appears or disappears | Rayleigh anomaly; a point where the order lies on the light line |
| Path D / path E | Continuation from normal incidence / from the nearest real {math}`k_x` | Names used in this appendix |
| Continuity correction | Flipping the principal root to implement path D in closed form; never active in a lossless dielectric; removed from Wavebit on October 2, 2026 | `_off_principal_branch` |
| Maystre's rule | {math}`\operatorname{Re}k_z+\operatorname{Im}k_z>0`, used by Wavebit since October 2, 2026 | Coupure de Maystre |
| {math}`\operatorname{Im}k_z>0` rule | Always selecting the decaying root, as in fmmax | Coupure de Petit |

## Statements that are easy to misread

- **“Path D grazes the branch point.”** It goes directly through it. Doing so makes continuation ambiguous ({ref}`path D <ckr-tie>`).
- **“If scored against path D, Wavebit before October 2, 2026 gets every answer right.”** Correct. It implemented path D as its reference, so 288/288 is expected. The real question was whether path D was an appropriate reference ({ref}`branch cuts <ckr-branch-cut>`, {ref}`sweeps <ckr-sweeps>`). Wavebit changed to Maystre.
- **“Even in a lossless medium, Wavebit before October 2, 2026 flipped roots through its continuity correction.”** No. The correction never activated in a lossless dielectric. The evanescent-root flip immediately above the real axis came from the principal root's own cut, not from the correction ({ref}`continuation <ckr-continuation>`, {ref}`path D <ckr-tie>`).
- **“Changing to Maystre changed real-angle results too.”** No. In passive media at real {math}`k_x`, Maystre and path D agree bit for bit. Only complex `theta` and gain half-spaces changed ({ref}`Wavebit's rule <ckr-wavebit-rule>`).
- **“Selecting a growing root is wrong.”** At {math}`k_x=0.5+0.05i`, path D, path E, and Maystre all select a slightly growing root. It is the root connected to the real answer ({ref}`complex kx <ckr-complex-kx>`, {ref}`branch cuts <ckr-branch-cut>`).
- **“fmmax discards the imaginary part of complex `kx`.”** No. fmmax 1.7.1 uses complex `in_plane_wavevector` unchanged and selects roots with {math}`\operatorname{Im}k_z>0`.

## References and verification scope

Solvers used in the comparisons:

- [J. P. Hugonin, P. Lalanne, *RETICOLO software for grating analysis*, Institut d'Optique, Orsay, France (2005)](https://arxiv.org/abs/2101.00901).
- [V. Liu, S. Fan, “S4: A free electromagnetic solver for layered periodic structures,” Comput. Phys. Commun. 183, 2233–2244 (2012)](https://doi.org/10.1016/j.cpc.2012.04.026).
- [M. F. Schubert, A. M. Hammond, *Fourier modal method for inverse design of metasurface-enhanced micro-LEDs* (2023)](https://arxiv.org/abs/2308.08573).

RETICOLO values were recorded by running RETICOLO V10 in Octave. S4 was compared through execution results only. Wavebit's path-D values—the path-D columns of the grating table and the “before October 2, 2026” column of the {ref}`tie-break table <ckr-tie>`—were measured before the rule change and remain in the records under `path_d`, together with the commit used. Before writing any figure, `tools/docs/learning/draw_complex_kx_roots.py` checks its own numbers: closed-form paths D and E against step-by-step continuation, the branch-cut equations against actual jumps on a grid, and agreement between path D and the principal root in a lossless dielectric.

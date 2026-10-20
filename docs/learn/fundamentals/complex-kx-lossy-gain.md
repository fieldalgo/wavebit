(complex-kx-lossy-gain)=
# Appendix B. Extending to loss and gain: absorbing incidence, displaced branch points, and gain media

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

{ref}`Appendix A <complex-kx-roots>` explained which of the two half-space {math}`k_z` roots to choose, considering only a lossless medium. This appendix relaxes {ref}`assumption 2 <ckr-premises>` (losslessness) and examines what changes when the incident medium absorbs light, when the half-space whose root is selected absorbs light, and when that half-space amplifies light (gain). The two roots, branch points, branch cuts, paths D and E, and the four rules (path D, path E, Maystre, and {math}`\operatorname{Im}k_z>0`) retain the meanings defined in Appendix A.

This material is useful when specifying an angle in an absorbing incident medium ({ref}`the example in Chapter 8 <absorbing-incidence-example>`), using complex `theta` with an absorbing half-space, assigning gain to a half-space, or comparing solvers in absorbing media.

The behavior described here is that of Wavebit as of October 2, 2026. On that date, Wavebit changed its half-space rule from path D to the Maystre rule ({math}`\operatorname{Re}k_z+\operatorname{Im}k_z>0`; {ref}`Appendix A <complex-kx-roots>`). For real {math}`k_x` in passive media, the two rules are bitwise identical. The differences in this appendix occur for complex {math}`k_x` and on the total-internal-reflection side of gain half-spaces. Path D is retained as the rule used by “Wavebit before October 2, 2026.” Section 8 of [Branch handling](../../theory/rcwa/branch-cuts.md) gives the reasons for the change and the measurements.

The figures were drawn with `tools/docs/learning/draw_complex_kx_roots.py`; values computed with Wavebit are recorded in `docs/validation/learning-records/complex-kx-roots.json`, together with the Wavebit commit used. Wavebit values obtained with path D were measured on a commit preceding the rule change.

(ckr-b-relaxed)=
## Which assumption of Appendix A is relaxed?

Appendix A made four assumptions: (1) the time convention {math}`e^{-i\omega t}`, (2) a lossless dielectric, (3) nonmagnetic, isotropic media, and (4) root selection only in half-spaces. This appendix **relaxes only assumption 2**. The other three remain unchanged. Each section therefore explains where and how assumption 2 is relaxed and what consequently changes from Appendix A.

- {ref}`Absorbing incident medium <ckr-absorbing-incidence>`: the incident medium absorbs. The conversion from `theta` to {math}`k_x` separates into two rules. Appendix A's interpretation of a complex `theta` as a slight perturbation of a real `theta` no longer holds.
- {ref}`Absorbing half-space <ckr-lossy>`: the half-space whose root is selected absorbs. For real {math}`k_x`, the root moves into the interior of the first quadrant, eliminating ties, and the branch point leaves the real axis. The branch cut that lay on the real axis in Appendix A moves off it.
- {ref}`Gain medium <ckr-gain>`: the half-space amplifies light. Even the broader assumption of passivity—no amplification; see {ref}`below <ckr-b-passive>`—is broken. The two criteria point to different roots even for real {math}`k_x`, so a rule is needed.
- {ref}`Code differences <ckr-b-codes>`: differences between solvers that become visible only in the presence of loss or gain.

Assumption 2 actually contains two conditions: {math}`\operatorname{Im}\varepsilon=0` (losslessness) and {math}`\varepsilon>0` (a dielectric). This appendix relaxes the first and considers media with {math}`\operatorname{Im}\varepsilon\ne0`. Its equations and explanations then hold regardless of the sign of {math}`\operatorname{Re}\varepsilon`, so they also apply to metals with {math}`\operatorname{Re}\varepsilon<0`. However, except for the gold substrate in the audit case, all numerical examples and random samples have {math}`\operatorname{Re}\varepsilon>0`. Neither appendix treats the remaining case, {math}`\operatorname{Im}\varepsilon=0` with {math}`\varepsilon<0`: an idealized metal with its loss removed. There, {math}`n=\sqrt\varepsilon` is purely imaginary, so the branch points lie on the imaginary rather than the real axis, and Appendix A's diagrams do not apply unchanged.

(ckr-b-passive)=
### Passive and active: lossless, absorbing, and gain media

A **passive** medium does not generate energy: it may absorb light, but cannot amplify it. The absorbed power density {math}`\tfrac12\omega\varepsilon_0\operatorname{Im}\varepsilon\,|\mathbf E|^2` must be nonnegative, giving {math}`\operatorname{Im}\varepsilon\ge0` ({ref}`Chapter 13 <interference-absorption-origin>`). Thus, passive includes every medium that **reduces or preserves** the light's energy.

| Medium | {math}`\operatorname{Im}\varepsilon` ({math}`e^{-i\omega t}` convention) | Light energy | Classification |
| --- | --- | --- | --- |
| Lossless (air, ideal glass) | {math}`=0` | Unchanged | Passive |
| Absorbing (metals, absorbers) | {math}`>0` | Decreases | Passive |
| Gain (laser medium) | {math}`<0` | Increases | Active |

The lossless case in Appendix A sits at equality ({math}`\operatorname{Im}\varepsilon=0`), on the boundary between absorption and gain. Strictly, passivity is a property across all frequencies, and no material other than vacuum is completely free of absorption at every frequency. Dispersion requires absorption somewhere through the Kramers–Kronig relations, so losslessness is an idealization. RCWA solves one frequency at a time, however, and treats a medium as passive when {math}`\operatorname{Im}\varepsilon\ge0` at that frequency.

Most dielectrics, metals, and absorbers considered with RCWA are passive. Absorbing incident media and absorbing half-spaces extend the discussion within passivity; **only gain media also relax the passive-medium assumption**.

**Magnetic media remain excluded ({ref}`assumption 3 of Appendix A <ckr-premises>`).** With magnetism, {math}`k_z^2=\varepsilon\mu-k_x^2`, and even a passive medium can have {math}`\operatorname{Im}(\varepsilon\mu)<0`. Negative-index media provide an example: for {math}`\varepsilon=\mu=-1+0.1i`, {math}`\varepsilon\mu=0.99-0.2i`. The Maystre rule selects a growing root in such a half-space. Since 2026-10-05, Wavebit has therefore used its mirror ({math}`\operatorname{Im}k_z-\operatorname{Re}k_z>0`) for double-negative half-spaces (Re ε<0, Re μ<0); see [Negative-index media](../../theory/rcwa/negative-index-media.md). This appendix does not cover that case.

(ckr-b-roles)=
### The two roles of a medium

**The question depends on which medium absorbs.** A medium plays two roles, and losslessness or absorption must be considered separately for each.

| Medium | Question | If lossless (Appendix A) | If absorbing (this appendix) |
| --- | --- | --- | --- |
| Incident medium | How is `theta` converted to {math}`k_x`? | Real and complex `theta` connect continuously. A complex `theta` with a small imaginary part is a slight perturbation of real `theta`. | Real `theta` (the phase-angle convention: the phase travels at angle `theta`) and complex `theta` (formal {math}`n\sin\theta`) give different {math}`k_x`. Complex `theta` is a separate formal calculation ({ref}`below <ckr-absorbing-incidence>`). |
| Half-space whose root is selected (upper: reflected orders; lower: transmitted orders) | Which of the two roots is selected? | The branch point is on the real axis, so the rules disagree near real {math}`k_x`. | The branch point is off the real axis, so all four rules agree near real {math}`k_x` ({ref}`below <ckr-lossy>`). |

The upper medium is both the incident medium and the half-space for reflected orders, so both questions apply. The lower medium is involved only in selecting roots for transmitted orders. Gain media fall outside this table and are considered {ref}`separately <ckr-gain>`.

(ckr-absorbing-incidence)=
## Absorbing incident medium: two rules for converting an angle to kx

**Assumption being relaxed.** We relax assumption 2 of Appendix A in the **incident medium**: the upper medium has {math}`\operatorname{Im}\varepsilon>0`. What happens when the same medium selects roots as the half-space for reflected orders is addressed in the {ref}`next section <ckr-lossy>`.

Appendix A stated that complex {math}`k_x` enters Wavebit only through complex `theta`. In a lossless incident medium, a complex `theta` with a small imaginary part is a slight perturbation of real `theta`, so the continuity criterion can equally be expressed in terms of `theta`. Absorption in the incident medium breaks that equivalence. Wavebit uses two rules to convert an angle to {math}`k_x`, and in an absorbing medium they give different {math}`k_x`.

- **Real `theta`: the phase-angle convention.** A plane wave in an absorbing medium has a complex wavevector, so the direction of phase propagation ({math}`\operatorname{Re}\mathbf k`) differs from the direction of attenuation ({math}`\operatorname{Im}\mathbf k`). “The angle” is therefore not uniquely defined by itself. The phase-angle convention finds a real {math}`k_x` whose phase direction makes exactly `theta` with the normal. It solves {math}`\tan\theta=k_x/\operatorname{Re}k_z` together with the dispersion relation ({ref}`Chapter 8 <absorbing-incidence-example>`). Real {math}`k_x` means illumination of uniform intensity along the surface. In an absorbing incident medium, the magnitude of `theta` must be less than π/2. As `theta` approaches 90°, this {math}`k_x` grows without bound (Figure B-1 below), so at grazing incidence there is no finite {math}`k_x` with uniform surface intensity. Wavebit therefore refuses that input ({ref}`Chapter 8 <absorbing-incidence>`).
- **Complex `theta`: formal {math}`n\sin\theta`.** The {ref}`formal expression <ckr-complex-kx>` from Appendix A, {math}`k_x=n\sin\theta\cos\phi`, is used unchanged, multiplying by the incident medium's complex refractive index {math}`n` without taking its real part. Here the angle is a calculation parameter, not a physical direction.

For {math}`\varepsilon=4+3i` ({math}`n=2.121+0.707i`), as in the example in Chapter 8, Wavebit gives:

| `theta` | Interpretation | {math}`k_x/k_0` |
| --- | --- | --- |
| `0.4` | Phase-angle convention | 0.833318 |
| `0.4 + 1e-12j` | Formal {math}`n\sin\theta` | 0.826081 + 0.275360i |
| `0.4 - 1e-12j` | Formal {math}`n\sin\theta` | 0.826081 + 0.275360i |

The first row can be checked directly: at this {math}`k_x`, {math}`k_z=1.971+0.761i`, and the phase direction is {math}`\arctan(0.8333/1.971)=22.9^\circ`, or 0.4 rad.

Yet adding an imaginary part of only {math}`10^{-12}` to `theta` makes the imaginary part of {math}`k_x` jump from 0 to 0.275. Complex `theta` invokes the formal expression, and because `theta` is almost real, {math}`\sin\theta\approx\sin0.4=0.3894`:

```{math}
k_x\approx n\times0.3894=(2.1213+0.7071i)\times0.3894\approx0.826+0.275i .
```

Thus, 0.275 comes from the imaginary part of {math}`n`, not that of `theta` ({math}`0.7071\times0.3894\approx0.275`). The value is the same whether the imaginary part of `theta` is positive or negative: the discontinuity occurs only at the point where `theta` is exactly real. In an absorbing medium, complex `theta` is therefore a different input, not a slight perturbation of real `theta` ([Coordinate and sign conventions](../../reference/common/conventions.md#real-angle-incidence)).

```{figure} ../../figures/learn/learn/b01-theta-to-kx.svg
:alt: Two kx planes. (a) For an absorbing incident medium ε=4+3i, the solid teal curve shows kx from the phase-angle convention as real θ increases from zero. It lies on the real axis, is labeled as unbounded near θ=90 degrees, and has ticks at θ=0.2, 0.4, …, 1.2. The dashed orange curve applies formal n·sinθ to real θ: a segment from the origin to the branch point n=2.121+0.707i (θ=90 degrees), with small orange points at θ=0.2, 0.4, …, 1.2 and labels at θ=0.8 and 1.2. An arrow joins the teal real-θ=0.4 point, 0.833318, to the orange θ=0.4±10⁻¹²i point, 0.826081+0.275360i, noting the jump caused by an imaginary part of only 10⁻¹². (b) For lossless ε=4 (n=2), the solid teal and dashed orange curves overlap on the real axis from zero to branch point 2, with labels 0.778837 at θ=0.4 and 0.778837+1.8×10⁻¹²i at θ=0.4+10⁻¹²i.
:figclass: learning-figure

Figure B-1. Two rules for converting angle {math}`\theta` to {math}`k_x`. In an absorbing incident medium, the real-{math}`\theta` phase-angle convention (on the real axis) and formal {math}`n\sin\theta` (on the segment from the origin to {math}`n`) separate; in a lossless medium they coincide. Computed and drawn with `tools/docs/learning/draw_complex_kx_roots.py`.
```

Figure B-1 plots this in the {math}`k_x` plane.

- **(a) Absorbing incident medium, {math}`\varepsilon=4+3i`.** The solid teal curve is the phase-angle {math}`k_x` as real `theta` increases from zero. It lies on the real axis and grows without bound as `theta` approaches 90°. The dashed orange curve applies formal {math}`n\sin\theta` to real `theta`: the segment from the origin to branch point {math}`n`, reaching {math}`n` at 90°. Giving `theta` a tiny imaginary part moves {math}`k_x` from the teal curve to a point on the orange curve. Thus, adding only {math}`10^{-12}` to the imaginary part of `theta = 0.4` produces the jump between the teal and orange points.
- **(b) Lossless {math}`\varepsilon=4` ({math}`n=2`).** Because {math}`n` is real, the phase-angle convention is simply {math}`n\sin\theta`. Both curves overlap on the real axis: `theta = 0.4` gives {math}`k_x/k_0=0.778837`, while `theta = 0.4 + 1e-12j` gives {math}`0.778837+1.8\times10^{-12}i`, a change of only order {math}`10^{-12}`.

Consequently, Appendix A's continuity criterion still holds with respect to {math}`k_x` in an absorbing incident medium, but cannot be restated in terms of `theta`. A small imaginary part of `theta` is a small perturbation of {math}`k_x` only when the incident medium is lossless, for example air. To slightly perturb a real {math}`k_x` in an absorbing incident medium, work backward from the desired complex {math}`k_x` to {math}`\theta=\arcsin(k_x/n)` and supply that complex `theta` ({ref}`Chapter 8 <complex-transverse-wavevector>`).

Before drawing, the script checks three things: that the phase direction computed from the phase-angle {math}`k_x` equals `theta`; that the formal values with {math}`\pm10^{-12}i` added to `theta` differ from {math}`n\sin\theta` for real `theta` only at order {math}`10^{-12}`; and that the two rules agree in a lossless medium.

**Why are there two rules?** Previously, {math}`n\sin\theta` was also used for real `theta`, so even a real angle produced complex {math}`k_x` in an absorbing medium. That caused the case found by the [Time-convention audit](../../validation/rcwa/time-convention-audit.md): in a lossy dielectric above gold, changing the angle by {math}`6.7\times10^{-5}` rad made the reflection amplitude jump by 2.96. The {ref}`next section <ckr-lossy>` identifies the branch cut involved. Now, real `theta` is interpreted only as a physical direction, and formal calculations use complex `theta`.

**Evanescent incidence.** Here the incident wave itself is evanescent in the incident medium. As Appendix A showed, {math}`\theta=\pi/2+i\alpha` has real {math}`\sin\theta=\cosh\alpha`. In a lossless incident medium, {math}`k_x=n\cosh\alpha` remains real; the incident wave then carries no power toward the surface, so `R`, `T`, `power_entering` and `absorption` are `NaN` while the amplitudes are kept. In an absorbing incident medium, complex {math}`n` makes {math}`k_x` complex too, producing `ComplexIncidenceWarning`.

(ckr-lossy)=
## Absorbing half-space: the branch point leaves the real axis

**Assumption being relaxed.** We relax assumption 2 of Appendix A in the **half-space whose root is selected**. That half-space has {math}`\operatorname{Im}\varepsilon>0`; it absorbs and is still passive. Two things change: the real-{math}`k_x` root moves into the interior of the first quadrant, and the branch point moves off the real axis.

One sign convention needs care. Like {ref}`Appendix A <ckr-premises>`, this appendix writes the half-space field as {math}`e^{ik_zd}`, where {math}`d` is distance away from the structure. Chapter 8 used the upward-positive global {math}`z`, so a transmitted wave attenuating into the lower absorbing medium had negative imaginary {math}`k_z` ({ref}`Chapter 8 <absorbing-transmission>`). The same wave has {math}`\operatorname{Im}k_z>0` in this appendix.

**For real {math}`k_x`, the root is still determined automatically.** In {ref}`step ② of Appendix A <ckr-real-kx>`, real {math}`k_x` gave {math}`\operatorname{Im}k_z^2=\operatorname{Im}\varepsilon`, which was zero in a lossless medium. With absorption, this value is positive, placing {math}`k_z^2` in the **interior** of the upper half-plane, between 0° and 180°. The square root halves the angle, so the selected root lies in the **interior** of the first quadrant: it is both outgoing and decaying. The opposite root lies inside the third quadrant and violates both criteria. Unlike the lossless case, neither lies on a boundary, so there is no tie.

```{figure} ../../figures/learn/learn/b02-real-kx-absorbing.svg
:alt: Two panels. (a) In the kz² plane, the upper half-plane with positive imaginary part is pale teal. For absorbing ε=2.25+0.5i, increasing real kx from 0 to 2.5 moves kz² left along Im(kz²)=0.5, from 2.25+0.5i (kx=0), through 0.50i (kx=1.5), to −4.00+0.50i (kx=2.5). The dark line has a leftward arrow. Lossless ε=2.25 with the same real part follows the real axis as a gray dotted line. (b) In the quadrant-colored kz plane, the selected root follows a dark curve inside the first quadrant from 1.509+0.166i (kx=0), through 0.500+0.500i (kx=1.5), to 0.125+2.004i (kx=2.5). The rejected root −kz is a faint dotted curve inside the third quadrant. The lossless ε=2.25 root follows the real and positive imaginary axes, the boundary of the first quadrant, as a gray dotted curve.
:figclass: learning-figure

Figure B-2. Real {math}`k_x` in an absorbing medium. {math}`k_z^2` follows the horizontal line {math}`\operatorname{Im}k_z^2=\operatorname{Im}\varepsilon` inside the upper half-plane, and the selected root stays inside the first quadrant. Gray dotted curves show, for comparison, a lossless medium with the same real permittivity: they follow the real axis and first-quadrant boundary as in Appendix A. Computed and drawn with `tools/docs/learning/draw_complex_kx_roots.py`.
```

The {math}`k_x=0` point of Figure B-2 is:

| Case | {math}`k_z^2` (angle) | Selected root (angle) | Opposite root |
| --- | --- | --- | --- |
| Absorbing {math}`\varepsilon=2.25+0.5i`, {math}`k_x=0` | 2.25+0.5i (12.5°) | 1.509+0.166i (6.3°) | −1.509−0.166i |

Of the 3000 random samples drawn together with Appendix A's 1496 lossless samples, 1504 are absorbing: {math}`\operatorname{Re}\varepsilon` ranges from 1 to 6, {math}`\operatorname{Im}\varepsilon` from 0.01 to 3, and real {math}`k_x` from 0 to 3. The four rules select **the same root in all 1504 cases**.

**For complex {math}`k_x`,** the imaginary part of {math}`k_z^2` is Appendix A's expression plus {math}`\operatorname{Im}\varepsilon`:

```{math}
\operatorname{Im}k_z^2=\operatorname{Im}\varepsilon-2\operatorname{Re}k_x\operatorname{Im}k_x .
```

The criteria disagree only where this is negative. In a lossless medium, {math}`\operatorname{Re}k_x\operatorname{Im}k_x>0` sufficed. With absorption, {math}`2\operatorname{Re}k_x\operatorname{Im}k_x` must exceed {math}`\operatorname{Im}\varepsilon`.

Appendix A's explanation of a {ref}`“growing root” <ckr-complex-kx>` changes slightly too. As shown there, the dispersion relation gives {math}`\operatorname{Re}\mathbf k\cdot\operatorname{Im}\mathbf k=\tfrac12\operatorname{Im}\varepsilon`, the dot product of the propagation and intensity-variation directions. In a lossless medium it is zero and the directions are perpendicular. With absorption it is positive, so they are not perpendicular: intensity is no longer constant along a ray, but decreases by the amount absorbed by the medium.

```{figure} ../../figures/learn/learn/b03-air-vs-lossy.svg
:alt: Two kx planes. (a) For air ε=1, branch point n=1 lies on the real axis. The region where the criteria disagree (Im kz²<0, pale orange) is the first and third quadrants of the kx plane. Path D's branch cut is the real-axis segment kx>1; fmmax's cut is the real axis left of 1 (−1<kx<1) together with the imaginary axis; path E rises vertically from 1; and Maystre follows a curve upward from 1. (b) For absorbing ε=4+3i, branch point n=2.121+0.707i lies above the real axis. The pale-orange region is beyond the hyperbola Re kx·Im kx=1.5, and the real axis is outside it. Path D's cut extends the line from the origin through the branch point beyond that point. fmmax's cut follows the boundary hyperbola upward-left from the branch point; the unflipped principal-root cut follows it downward-right.
:figclass: learning-figure

Figure B-3. In an absorbing medium, the branch point is off the real axis, so all four rules choose the same root near real {math}`k_x`. Computed and drawn with `tools/docs/learning/draw_complex_kx_roots.py`.
```

Figure B-3(a) is the same kind of map for air, overlaying Appendix A's {ref}`four branch cuts <ckr-branch-cut>`. Figure B-3(b) takes the previous section's incident medium, {math}`\varepsilon=4+3i`, and now treats it as the half-space whose root is selected. Its branch point, {math}`n=\sqrt\varepsilon=2.121+0.707i`, is **off the real axis**.

- **Pale orange** marks {math}`\operatorname{Im}k_z^2<0`, where the criteria disagree. The expression above makes this equivalent to {math}`\operatorname{Re}k_x\operatorname{Im}k_x>\operatorname{Im}\varepsilon/2`: above the hyperbola {math}`\operatorname{Re}k_x\operatorname{Im}k_x=\operatorname{Im}\varepsilon/2=1.5`, on the side farther from the origin. **The entire real axis is outside this region.**
- Outside the orange region, all four rules select **the same root**. The drawing script checks this across the entire grid.
- The branch point is one point on that boundary hyperbola ({math}`k_z^2=0`). On the boundary, {math}`k_z^2` is real. Upward-left from the branch point, {math}`k_z^2>0`, so the {math}`\operatorname{Im}k_z>0` rule jumps there; downward-right, along the gray dotted curve, {math}`k_z^2<0`, so the unflipped principal root jumps there. These are exactly the {ref}`equations in Appendix A <ckr-half-plane-rules>`. The path-D, path-E, and Maystre cuts extend from the branch point into the orange region.
- Path D's cut is **the ray obtained by extending the line from the origin through the branch point beyond that point** (red). Those destinations make the straight path D from the origin pass exactly through the branch point. In a lossless medium, this ray lies on the real axis: it is the {ref}`tie <ckr-tie>` encountered in Appendix A.

**Absorption determines the displacement.** Starting at real {math}`k_x=a` ({math}`a>0`), the hyperbola shows that its imaginary part must reach {math}`\operatorname{Im}\varepsilon/(2a)` before it enters the orange region. Below that imaginary part, all four rules agree. Weak absorption makes this gap small. For {math}`\varepsilon=2.25+10^{-6}i`, path D jumps at {math}`k_x=1.6\pm10^{-3}i` just as in the lossless case ({math}`0.0029+0.5568i\to0.0029-0.5568i`).

**Numerical counts.** Repeating Appendix A's two tables with the absorbing members of the same random sample gives:

- Adding {math}`\pm10^{-8}i` to real {math}`k_x` flips the root **zero** times for all four rules among 1504 absorbing cases. Among the 1496 lossless cases, path D flipped 597 times and fmmax 899 times. Here the samples have {math}`\operatorname{Im}\varepsilon\ge0.01` and the perturbation is {math}`10^{-8}`, so it cannot cross the gap above.
- For 312 of the same absorbing samples, adding an imaginary part between −0.5 and 0.5 produces complex {math}`k_x`. Counting matches between each rule and each continuation gives:

| Rule | Matches path D | Matches path E |
| --- | ---: | ---: |
| Path D (Wavebit before October 2, 2026) | 312/312 | 304/312 |
| Maystre (Wavebit) | 304/312 | **312/312** |
| {math}`\operatorname{Im}k_z>0` (fmmax) | 289/312 | 297/312 |

Agreement is much more frequent than among the 288 lossless cases: path D matched E in 238, Maystre matched E in 286, and fmmax matched D in 130. Absorption reduces differences between the rules because all four agree outside the orange region.

**This is where Wavebit's pre-October-2, 2026 continuity correction operated.**

- Using the principal root alone places its branch cut on the gray dotted curve in Figure B-3(b), which descends toward the real axis.
- The audit case in the {ref}`previous section <ckr-absorbing-incidence>`—a reflection-amplitude jump of 2.96 over {math}`6.7\times10^{-5}` rad—crossed this principal-root cut in the gold substrate, the lower half-space selecting transmitted-order roots. At the time, a real angle in an absorbing incident medium made {math}`k_x=n_{\mathrm{top}}\sin\theta` complex, where {math}`n_{\mathrm{top}}` is the incident refractive index.
- The correction moved the cut to path D's ray. Thus, in passive media, pre-October-2, 2026 Wavebit used principal root + continuity correction = path D. Complex {math}`k_x` in gain media is discussed {ref}`below <ckr-gain-complex-kx>`.
- Once real `theta` was changed to produce real {math}`k_x`, a real angle no longer generated complex {math}`k_x`, and the correction operated only for complex `theta`.
- With the correction active, path D sometimes selected a second-quadrant root. For a lossless dielectric, path D (= principal root) never selects {math}`\operatorname{Re}k_z<0`; absorbing media differ.
- Measured with the old `boundary_kz`, all 312 absorbing cases matched path D, while 309 matched the principal root. The remaining three are where the correction activated. All 288 lossless cases matched the principal root ({ref}`Appendix A <ckr-continuation>`).
- Current Wavebit uses the single Maystre rule without that correction. On the same 312 cases, current `boundary_kz` matches Maystre in all 312, path D in 304, and the principal root in 301. Outside the orange region it chooses the same root as path D.

::::{dropdown} Equations: the closed form of the continuity correction
In the {math}`k_z^2` plane, path D is the straight segment from {math}`\varepsilon` to {math}`\varepsilon-k_x^2`:

```{math}
k_z^2(t)=\varepsilon-t\,k_x^2,\qquad t:0\to1 .
```

The principal root—{math}`\operatorname{Re}k_z>0`, with {math}`\operatorname{Im}k_z>0` as the tie-break when the real part is zero—is discontinuous only on the negative real axis of {math}`k_z^2`. It therefore suffices to test whether the segment crosses that axis. Its imaginary part, {math}`\operatorname{Im}\varepsilon-t\operatorname{Im}k_x^2`, is linear in {math}`t` and crosses the real axis at most once, at {math}`t^*=\operatorname{Im}\varepsilon/\operatorname{Im}k_x^2`. The principal root is flipped to {math}`-k_z` if all three conditions hold:

```{math}
\operatorname{Im}k_x^2>0,\qquad t^*<1,\qquad \operatorname{Re}\varepsilon-t^*\operatorname{Re}k_x^2<0 .
```

This flip was Wavebit's continuity correction (`_off_principal_branch`) before October 2, 2026. The expression assumes {math}`\operatorname{Im}\varepsilon\ge0` (passivity). It examines only {math}`\operatorname{Im}k_x^2>0` and does not separately require {math}`t^*\ge0`. Gain media break that assumption ({ref}`Gain half-spaces with complex kx <ckr-gain-complex-kx>`).

In a lossless medium, {math}`t^*=0`, so the third condition becomes {math}`\operatorname{Re}\varepsilon<0`, which a dielectric ({math}`\varepsilon>0`) cannot satisfy. This explains Appendix A's statement that the correction never activates in a lossless dielectric. Conversely, for lossless negative {math}`\varepsilon`—a case neither appendix treats—it activates where {math}`\operatorname{Im}k_x^2>0`. A closed form for path E can be derived the same way. In a separately drawn set of 600 random samples, five had at least one path pass within 0.01 of a branch point and were excluded. In all remaining 595 cases (287 lossless, 308 absorbing), both closed forms agreed with step-by-step continuation.
::::

(ckr-gain)=
## Gain media: also relaxing passivity

(ckr-gain-premise)=
### The assumption breaks

Only this section **relaxes passivity**. A **gain medium**, such as a laser medium, amplifies light and has {math}`\operatorname{Im}\varepsilon<0`. That reverses {ref}`step ② for real kx in Appendix A <ckr-real-kx>`. Even when {math}`k_x` is real:

```{math}
\operatorname{Im}k_z^2=\operatorname{Im}\varepsilon<0 .
```

Now {math}`k_z^2` is in the lower half-plane, with its two roots in the second and fourth quadrants. The two criteria select different roots even without complex {math}`k_x`. **A rule is required even for real {math}`k_x`.**

### What determines the answer: causality

Causality requires that an effect cannot precede its cause. Computationally, it means applying Appendix A's {ref}`continuation <ckr-continuation>` in frequency {math}`\omega` rather than {math}`k_x`, starting where the imaginary part of {math}`\omega` is very large.

- With time factor {math}`e^{-i\omega t}`, a large positive imaginary part of {math}`\omega` describes a signal growing rapidly in time. Light propagates at finite speed, so the field a distance {math}`d` from the structure originates from a weaker cause at an earlier time, delayed by the travel time. The field must therefore decrease with distance.
- Moreover, at very large {math}`\omega`, any medium approaches vacuum, {math}`\varepsilon\to1`, and the two roots are {math}`k_z\approx\pm\omega/c`. Here only, these values are not divided by {math}`k_0`. Because {math}`\operatorname{Im}\omega>0`, the decaying root is {math}`+\omega/c`, which is the correct starting root.

The causal root is obtained by following that root continuously to the physical {math}`\omega`, staying inside the upper half of the complex-frequency plane (Skaar 2006; Nistad and Skaar 2008). The path samples {math}`\varepsilon(\omega)` at many frequencies, so {math}`\varepsilon` at one frequency does not determine the answer. Media with the same {math}`\varepsilon` at the same frequency can yield opposite roots if their dispersion differs.

### Ordinary weak gain: continuing from zero gain

For common weak gain, such as population-inverted Lorentz gain, and angles away from the critical angle, the causal root equals the root obtained by “holding {math}`k_x` fixed and continuously turning on gain from zero.” The answer at zero gain is known, so this uses the same continuation idea. Write {math}`k_z^2=X-ig`, where {math}`X=\operatorname{Re}\varepsilon-k_x^2` is the zero-gain {math}`k_z^2` and {math}`g=-\operatorname{Im}\varepsilon>0` is the gain. Turning on gain traces a vertical downward path from {math}`X` in the {math}`k_z^2` plane.

- **Propagation ({math}`X>0`)**: the path descends from the positive real axis, continuously retaining the principal root. It lies in the fourth quadrant: an **outgoing, growing wave**. This is natural for propagation through an amplifying medium (Skaar 2006).
- **Total internal reflection ({math}`X<0`, evanescent)**: the starting root is {math}`+i\sqrt{|X|}` (decaying). As the path moves just below the negative real axis, the root enters the second quadrant. It **decays with distance but carries energy toward the interface**. The gain medium amplifies the evanescent field near the interface and supplies energy to it, making reflectance exceed one. This is amplified total internal reflection, or evanescent gain (Grepstad and Skaar 2011).

### Example: from glass into a gain medium

TE light is incident from glass ({math}`n=1.5`) onto gain medium {math}`\varepsilon=1-0.05i`. The critical angle is 41.81°.

| Incident angle | Root continued from zero gain | R for that root | Wavebit (Maystre) | Path D (Wavebit before October 2, 2026; principal root) | fmmax ({math}`\operatorname{Im}k_z>0`) |
| --- | --- | ---: | --- | --- | --- |
| 0° | 1.0003 − 0.0250i (4) | 0.0400 | Same | Same | Opposite root, R = 24.98 |
| 30° (propagating) | 0.6625 − 0.0377i (4) | 0.1056 | Same | Same | Opposite root, R = 9.467 |
| 41.81° | 0.1581 − 0.1581i (4) | 0.5723 | Same | Same | Opposite root, R = 1.747 |
| 41.82° | −0.1575 + 0.1587i (2) | 1.7434 | Same | **Opposite root, R = 0.5736** | Same |
| 45° (total internal reflection) | −0.0694 + 0.3603i (2) | **1.2646** | Same | **Opposite root, R = 0.7907** | Same |

Numbers in parentheses indicate quadrants. The TE reflection coefficient is {math}`r=(k_{z1}-k_{z2})/(k_{z1}+k_{z2})`, with {math}`k_{z1}` in glass and {math}`k_{z2}` in the gain medium. Replacing {math}`k_{z2}` by {math}`-k_{z2}` makes {math}`r` become {math}`1/r`, so the two reflectances are reciprocals. The 41.81° and 41.82° rows are only roots continued from zero gain; they cannot confidently be called causal roots. Immediately beside the critical angle, dispersion shifts the angle where the causal answer jumps (caution 1 below). Figure B-4 plots the 30° and 45° rows in the {math}`k_z` plane.

```{figure} ../../figures/learn/learn/b04-gain.svg
:alt: Two kz planes. (a) At 30-degree propagating incidence, the roots are ±(0.6625−0.0377i). The root continued from zero gain, causal for ordinary weak gain, is 0.6625−0.0377i (outgoing and growing, R=0.1056), selected by both Wavebit (Maystre) and path D. fmmax chooses −0.6625+0.0377i (R=9.4669). (b) At 45-degree total internal reflection, Wavebit (Maystre) and fmmax select the causal root −0.0694+0.3603i (decaying but carrying energy toward the interface, R=1.2646). Path D, used by Wavebit before October 2, 2026, selects 0.0694−0.3603i (R=0.7907).
:figclass: learning-figure

Figure B-4. The two roots and causal choice for a gain half-space: TE incidence from glass onto {math}`\varepsilon=1-0.05i`. For ordinary weak gain, the causal root is outgoing and growing for propagation; for total internal reflection it decays but carries energy toward the interface. Computed and drawn with `tools/docs/learning/draw_complex_kx_roots.py`.
```

### Comparing the rules: real kx and a gain half-space

- **Path D (Wavebit before October 2, 2026), hence the principal root:** correct for propagation, wrong for total internal reflection because it selects the growing evanescent wave. With real {math}`k_x`, the path-D segment from {math}`\varepsilon` to {math}`\varepsilon-k_x^2` is horizontal at {math}`\operatorname{Im}=-0.05` in the {math}`k_z^2` plane. It does not cross the negative real axis, so the continuity correction never activates and the principal root is retained. For total internal reflection, {math}`k_z^2` lies just below that axis, near −180°; halving its angle gives approximately −90°, the growing root.
- **fmmax:** correct for total internal reflection, wrong for propagation because it selects an incoming wave. Even at normal incidence it gives R = 24.98.
- **Maystre (Wavebit since October 2, 2026):** **exactly** matches the root continued from zero gain, regardless of gain strength. For propagation, the argument of {math}`k_z^2` is between −90° and 0°, giving a principal-root argument between −45° and 0°. Since {math}`\operatorname{Re}+\operatorname{Im}>0`, the fourth-quadrant root is retained. For total internal reflection, the argument of {math}`k_z^2` is between −180° and −90°, giving a principal-root argument between −90° and −45°, so the rule flips to the second-quadrant root. Thus, for ordinary weak gain, it agrees with the causal answer away from the critical angle.

No rule using only single-frequency values can work for every gain medium. Media can be constructed that reverse the answer even far from the critical angle, although this requires unusual dispersion: see the engineered medium in Grepstad and Skaar (2011).

**Path E** starts from the answer at real {math}`k_x`. But in a gain medium, the two criteria already disagree for real {math}`k_x`, so even its starting root is not determined automatically. None of the compared solvers uses path E.

(ckr-gain-complex-kx)=
### Gain half-spaces with complex kx

The table and comparison above use real {math}`k_x`. Complex {math}`k_x` introduced one further issue. Wavebit's pre-October-2, 2026 continuity correction—the closed form in {ref}`Absorbing half-space <ckr-lossy>`—assumed {math}`\operatorname{Im}\varepsilon\ge0`, so for gain half-spaces it could differ from step-by-step continuation along path D.

- If {math}`\operatorname{Im}k_x^2>0`, then {math}`t^*=\operatorname{Im}\varepsilon/\operatorname{Im}k_x^2` is negative, outside the segment, yet the third condition can still trigger a flip.
- If {math}`\operatorname{Im}k_x^2<0`, the closed form ignores a crossing of the negative real axis even when one occurs.

Among 399 random gain-media cases—{math}`\operatorname{Re}\varepsilon` from 1 to 6, {math}`\operatorname{Im}\varepsilon` from −3 to −0.01—and complex {math}`k_x` with real part 0 to 3 and imaginary part −0.5 to 0.5, the closed form disagreed with walked path D in 14 cases: six of the first kind, eight of the second. For example, with {math}`\varepsilon=1-0.05i`:

| {math}`k_x` | Closed form (Wavebit's continuity correction before October 2, 2026) | Walked path D |
| --- | --- | --- |
| 0.1 + 5i | −5.0991 + 0.1030i | 5.0991 − 0.1030i |
| 3 − 0.05i | 0.0442 + 2.8283i | −0.0442 − 2.8283i |

Whether a gain half-space's causal root follows path D is itself outside this appendix's scope. The point here is only that “closed form = path D” fails for complex {math}`k_x` in gain media. Current Wavebit no longer has this closed form. The Maystre rule remains the same single rule for gain media; its root is obtained by giving the half-space large absorption and continuing downward to its actual value ([Branch handling](../../theory/rcwa/branch-cuts.md), section 8).

::::{dropdown} Equations: the thick-gain-layer limit is not a valid criterion
Replacing the gain half-space by a layer of thickness {math}`h`—glass | gain layer | air, wavelength 1 µm—removes the need to select a root within the layer because both {math}`\pm k_z` are included. The Airy formula agrees with Wavebit's layer calculation to nine digits: at 45° and {math}`h=2` µm, both give 1.264637712. It is also known that the root sign does not change the answer for a finite layer (Lakhtakia et al. 2007). Root selection is a problem only for semi-infinite half-spaces ({ref}`assumption 4 of Appendix A <ckr-premises>`).

Nevertheless, the limit as {math}`h` grows without bound cannot be used as the half-space answer. As {math}`h` increases, the Airy factor {math}`e^{2ik_zh}` tends to zero when written with the {math}`\operatorname{Im}k_z>0` root and to infinity with the opposite root. Either representation tends to the single-interface reflection coefficient computed with {math}`\operatorname{Im}k_z>0`. The formula therefore makes the thick-layer limit select the decaying, fmmax-side root at every angle.

- 45°: R = 1.264638 at {math}`h=2` µm and 1.264630 from {math}`h=5` µm onward. This equals the causal value, but is agreement between rules, not independent evidence.
- 30°: beyond approximately 9.9 µm, round-trip gain exceeds one and the layer enters the oscillation regime. Thereafter R approaches the fmmax-side value: 3.05 at 10 µm, 9.488 at 20 µm, and 9.467 at 100 µm. This is not the causal value (Skaar 2006).

The causal answer is supported by continuation in complex frequency. While preparing this appendix, it was independently checked for two dispersion models along four different paths in the complex-frequency plane. For the model with gain confined to a narrow band and no {math}`k_z` branch point in the upper complex-frequency half-plane, all four paths gave the same root. A single-Lorentz model had a branch point in the upper half-plane, and only the path passing on its other side gave the opposite root. The reason is the instability discussed in caution 1 below. This check is not included in `tools/docs/learning/draw_complex_kx_roots.py`.
::::

### Three cautions

1. **Near the critical angle, the plane-wave answer loses its meaning.** In the table, the root continued from zero gain jumps from quadrant four to quadrant two between 41.81° and 41.82°. At the critical angle, {math}`X=0`, so turning on gain starts at a branch point ({math}`k_z^2=0`): the {ref}`“stepping on a branch point” situation <ckr-tie>` in Appendix A. The causal answer also jumps near the critical angle, but the jump is not exactly at that angle and shifts slightly with dispersion. Immediately beside it, one therefore cannot identify any rule as correct. A more fundamental physical problem is present. At fixed {math}`k_x`, there is a frequency where the transmitted wave runs parallel to the interface at grazing incidence ({math}`k_z=0`). If the medium provides gain at that frequency, the wave travels along an infinitely wide interface and amplifies without bound. Mathematically, a {math}`k_z` branch point appears in the upper complex-frequency half-plane, making continuation there path-dependent. A single Lorentz gain line has gain at all frequencies and always has this instability. If gain is restricted to a narrow band, it occurs only in a narrow angular interval near the critical angle; the interval depends on the gain band. Thus a plane-wave solution for a semi-infinite gain half-space is meaningful only as a quasi-steady state over a limited time, and trustworthy only away from the critical angle (Nistad and Skaar 2008; Grepstad and Skaar 2011).

   **Viewed as a beam.** The picture becomes clearer by computing the steady state of a Gaussian beam of width {math}`W` instead of an infinite plane wave. Near grazing, roughly within {math}`|\operatorname{Re}k_z^2|<|\operatorname{Im}k_z^2|`, the field at the beam center grows without bound as beam width increases. For 42° incidence from glass onto {math}`\varepsilon=1-0.05i`, widths of 2.5, 5, and 7.5 gain lengths, {math}`L_g=2\sqrt{\operatorname{Re}\varepsilon}/(k_0|\operatorname{Im}\varepsilon|)`, give {math}`|E_r/E_\text{inc}|^2` of 2.13, 181, and 1.1e8. Outside that region, a beam a few gain lengths wide converges to the Maystre value independently of dispersion. The dispersion-dependent flip angle at fixed {math}`k_x` does not appear for the beam. Sharp-edged beams, however, generate edge-growing waves at any angle, so this conclusion applies to smooth beams. Wavebit emits `GainCutoffWarning` for each order satisfying {math}`|\operatorname{Re}k_z^2|<2|\operatorname{Im}k_z^2|`. Values and supporting evidence appear under “Gain near grazing” in [Branch handling](../../theory/rcwa/branch-cuts.md).
2. **Strong gain can eliminate a steady-state solution.** If {math}`\varepsilon(\omega)` has poles or odd-order zeros in the upper complex-frequency half-plane, the medium itself is unstable and oscillates spontaneously: absolute instability. A solver examining {math}`\varepsilon` at only one frequency cannot detect it.
3. **The literature contains opposing views.** Whether amplified total internal reflection truly occurs at a semi-infinite half-space is disputed. Siegman (2010) and T. S. Mansuripur and M. Mansuripur (2014) argue against it. Choosing the root whose energy travels away from the interface gives a growing root in total internal reflection and R < 1, the value Wavebit produced before October 2, 2026. Willis et al. (2008, FDTD), Grepstad and Skaar (2011), and M. Mansuripur and Jakobsen (2021, causal wavepacket analysis) support it. This appendix follows the causality-based calculation of Grepstad and Skaar, and current Wavebit (Maystre) gives that side's value.

**Practical conclusion: put gain in finite layers and use passive upper and lower half-spaces.** Then this root-selection problem does not arise. The {ref}`next section <ckr-b-codes>` describes how current Wavebit handles gain half-spaces.

(ckr-b-codes)=
## Code differences visible only with absorption or gain

The {ref}`code comparison in Appendix A <ckr-codes>` collected root rules, complex-{math}`k_x` inputs, and lossless branch cuts. This section collects differences specific to absorbing incident media and gain half-spaces. The gain-half-space column concerns real {math}`k_x`.

| Code | Incident angle to {math}`k_x` in an absorbing medium | Gain half-space |
| --- | --- | --- |
| Wavebit | Real `theta`: phase-angle convention (real {math}`k_x`); complex `theta`: formal {math}`n\sin\theta` | Maystre since October 2, 2026: correct for ordinary weak gain. Previously always the principal root: correct for propagation, wrong for total internal reflection. |
| RETICOLO V10 | User supplies `beta0` directly | Its rule gives the answer for ordinary weak gain. |
| S4 | Real {math}`k_x` from {math}`\operatorname{Re}(n)\sin\theta` | Not measured. |
| fmmax 1.7.1 | Angle helper uses {math}`\sqrt{\operatorname{Re}\varepsilon}\sin\theta` | Wrong for propagation, correct for total internal reflection. |

**The same “30°” gives different {math}`k_x` in different codes.** For 30° in an absorbing medium with {math}`\varepsilon=4+3i`, the values are below. Wavebit's value was derived in the {ref}`Chapter 8 example <absorbing-incidence-example>`.

| Code | {math}`k_x/k_0` | Method |
| --- | --- | --- |
| Wavebit | 1.077700 | Phase-angle convention: solve for real {math}`k_x` together with the dispersion relation. |
| S4 | 1.060660 | {math}`\operatorname{Re}(n)\sin\theta` |
| fmmax helper | 1.000000 | {math}`\sqrt{\operatorname{Re}\varepsilon}\sin\theta`, equivalent to 27.72° under Wavebit's convention. |
| RETICOLO | User-supplied `beta0` | Supplying formal {math}`n\sin\theta` gives 1.060660 + 0.353553i. |

Thus, comparisons between solvers with an absorbing incident medium must match {math}`k_x`, not the angle.

**Wavebit.**

- Real `theta` in an absorbing incident medium gives real {math}`k_x` through the phase-angle convention; complex `theta` uses formal {math}`n\sin\theta` ({ref}`Absorbing incident medium <ckr-absorbing-incidence>`).
- Only the experimental conforming NVF constructions, `ConformingNVF` and its relatives in [Chapter 20](code-map.md), reject gain media outright. The ordinary solver handles both gain layers and gain half-spaces. Its sole half-space refusal is a gain-bearing double-negative medium ({math}`\operatorname{Re}\varepsilon<0`, {math}`\operatorname{Re}\mu<0`) that supports propagation ({math}`\operatorname{Re}(\varepsilon\mu)>0`), because single-frequency values cannot determine the root. The same medium is supported as a layer. Since October 2, 2026, half-space roots use Maystre, agreeing with the causal answer for ordinary weak gain away from the critical angle. Immediately beside that angle, no rule provides a definitive answer; see caution 1 in {ref}`Gain media <ckr-gain>`.
- No warning is issued merely because R + T exceeds incident power: that can be correct for gain media. Before October 3, 2026, the generic `UnphysicalResultWarning` appeared for gain layers and gain half-spaces exhibiting amplified total internal reflection (R > 1). It was removed because it labeled valid answers as failures ([Result checks](../../reference/common/result-guards.md)). The warning remains in `docs/validation/learning-records/complex-kx-roots.json` because those values were measured earlier, on `6bb364b`.
- The warning specific to gain half-spaces is `GainCutoffWarning`. It is emitted for each half-space containing an order near grazing, {math}`|\operatorname{Re}k_z^2|<2|\operatorname{Im}k_z^2|`; `solver.gain_cutoff_band()` identifies the orders separately for the upper and lower half-spaces. In the example above, this band spans approximately 39.2° to 44.4°, with no warning at 30° or 45°.
- Direct Wavebit calculations give 0.105632 at 30° and 1.264630 at 45°. An equally lossy substrate ({math}`1+0.05i`) gives 0.105632 at 30° and 0.790745 at 45°. Gain and loss give the same R on the propagating side, but differ on the total-internal-reflection side.
- Before October 2, 2026, real {math}`k_x` always selected the principal root, so a gain substrate's R equaled that of an equally lossy substrate to rounding error, including 0.790745 at 45°. Conjugating {math}`\varepsilon` conjugates the principal {math}`k_z`; with a lossless upper medium, it also conjugates {math}`r`. Thus old Wavebit treated a gain half-space like an equally lossy half-space, without a warning. For gain half-spaces with complex {math}`k_x`, the continuity correction's closed form could differ even from path D ({ref}`above <ckr-gain-complex-kx>`).
- It is safer to put gain in finite layers.

**RETICOLO.**

- Even with an absorbing incident medium, `res2` returns empty lists by default. This is the same `tolh`/`tolb` trap described in Appendix A's {ref}`code comparison <ckr-codes>`, with the same remedy.
- Constructing `beta0` from an angle as {math}`n\sin\theta` silently makes it complex when the incident medium absorbs. This is not the phase-angle {math}`k_x` Wavebit uses for real `theta`; it is Figure B-1's orange curve, formal {math}`n\sin\theta`.
- For gain half-spaces, its Maystre rule matches the root continued from zero gain.

**S4.** It constructs real {math}`k_x` from {math}`\operatorname{Re}n` and {math}`\operatorname{Re}\omega`. For absorbing incidence, this differs from the phase-angle convention, as the table shows. Gain half-spaces were not measured.

**fmmax.** Its angle helper, `plane_wave_in_plane_wavevector`, uses {math}`\sqrt{\operatorname{Re}\varepsilon}\sin\theta`. In absorbing media, supplying {math}`k_x` directly is safer for cross-validation. Its {math}`\operatorname{Im}k_z>0` rule for gain half-spaces chooses an incoming wave for propagation and the root continued from zero gain for total internal reflection.

(ckr-b-summary)=
## One-page summary

- Relaxed assumption: Appendix A's assumption 2, losslessness. Absorbing incident media and absorbing half-spaces stay within passivity; only gain media also relax passivity.
- In an **absorbing incident medium**, real `theta` (phase-angle convention, real {math}`k_x`) and complex `theta` (formal {math}`n\sin\theta`) give different {math}`k_x`. An imaginary part of only {math}`10^{-12}` in `theta` makes the imaginary part of {math}`k_x` jump from zero to {math}`\operatorname{Im}(n)\sin\theta`. Thus complex `theta` is not a perturbation of real `theta` there. In a lossless incident medium the rules coincide.
- In an **absorbing half-space**, the branch point is off the real axis. All four rules agree while {math}`\operatorname{Im}k_x` is smaller than {math}`\operatorname{Im}\varepsilon/(2\operatorname{Re}k_x)`. For real {math}`k_x`, the root lies inside quadrant one and has no tie. Extremely weak absorption still permits lossless-like jumps.
- Wavebit's **continuity correction** before October 2, 2026 never activated in lossless dielectrics, but did activate in absorbing half-spaces, moving the principal-root branch cut to path D's ray. Its closed form assumed passive media, so for complex {math}`k_x` in gain half-spaces it could differ even from path D. Current Wavebit uses a single Maystre rule without the correction.
- In **gain media**, the two criteria disagree even for real {math}`k_x`. Causality determines the answer, which cannot in general be inferred from {math}`\varepsilon` at one frequency. For ordinary weak gain away from the critical angle, propagation uses quadrant four (amplification), and total internal reflection uses quadrant two (amplified total internal reflection, R > 1). Maystre agrees, and Wavebit has used it since October 2, 2026. Earlier Wavebit gave R < 1 on the total-internal-reflection side. Near the critical angle or with strong gain, the plane-wave answer itself loses meaning. Put gain in finite layers.
- For absorbing incidence, **the same 30° produces different {math}`k_x` across codes**. Match {math}`k_x` in cross-validation.

## Terminology

| Term | Meaning in this appendix | Other name |
| --- | --- | --- |
| Passive medium | A medium that generates no energy; {math}`\operatorname{Im}\varepsilon\ge0`. | passive medium |
| Lossless / absorbing / gain | {math}`\operatorname{Im}\varepsilon=0` / {math}`>0` / {math}`<0`. | lossless / absorbing (lossy) / gain |
| Phase-angle convention | Interpreting real `theta` as the phase-propagation angle and solving for real {math}`k_x`. | phase-angle incidence |
| Formal {math}`n\sin\theta` | Multiplying {math}`\sin\theta` for complex `theta` by the incident medium's full complex refractive index {math}`n` to obtain {math}`k_x`. | formal complex angle |
| Continuity correction | Flipping the principal root to implement path D in closed form, assuming {math}`\operatorname{Im}\varepsilon\ge0`. It activates in absorbing half-spaces but not lossless dielectrics; removed from Wavebit on October 2, 2026. | `_off_principal_branch` |
| Gain medium | {math}`\operatorname{Im}\varepsilon<0`, a medium that amplifies light. | gain / active medium |
| Causal root | The root continued through the upper complex-frequency half-plane. | causal root |
| Root continued from zero gain | The root followed while holding {math}`k_x` fixed and increasing gain from zero; causal for ordinary weak gain. | — |
| Amplified total internal reflection | R > 1 for total internal reflection at a gain medium. | evanescent gain, amplified TIR |

Other terms—outgoing and decaying, principal root, branch point, branch cut, paths D and E, and the Maystre rule—are listed in Appendix A's terminology table.

(ckr-b-confusing)=
## Easily confused statements

- **“In an absorbing incident medium, `theta = 0.4 + 1e-12j` is a slight perturbation of `theta = 0.4`.”** No. Real `theta` uses the phase-angle convention; complex `theta` uses formal {math}`n\sin\theta`, making the imaginary part of {math}`k_x` jump from zero to 0.275 ({ref}`Absorbing incident medium <ckr-absorbing-incidence>`).
- **“An absorbing medium has no root-selection problem.”** Only to the extent of its absorption. Very weak absorption allows lossless-like jumps ({ref}`Absorbing half-space <ckr-lossy>`).
- **“For gain media, just choose the first-quadrant root.”** No. With gain, no first-quadrant root exists even at real {math}`k_x` ({ref}`Gain media <ckr-gain>`).
- **“Wavebit gives R < 1 for total internal reflection at a gain half-space.”** That describes Wavebit before October 2, 2026. Current Wavebit uses Maystre and gives amplified total internal reflection (R > 1) for ordinary weak gain. No rule is trustworthy near the critical angle ({ref}`Gain media <ckr-gain>`).
- **“A sufficiently thick gain layer reveals the half-space answer.”** No. At every angle that limit gives the {math}`\operatorname{Im}k_z>0` rule's answer ({ref}`Gain media <ckr-gain>`).
- **“Matching the angle gives the same illumination in every solver.”** Not with absorbing incidence. At 30° in {math}`\varepsilon=4+3i`, Wavebit gives 1.077700, S4 1.060660, and the fmmax helper 1.000000 ({ref}`Code differences <ckr-b-codes>`).

(ckr-b-references)=
## References and verification scope

Gain media:

- [J. Skaar, “Fresnel equations and the refractive index of active media,” Phys. Rev. E 73, 026605 (2006)](https://doi.org/10.1103/PhysRevE.73.026605). Determining the signs of refractive index and {math}`k_z` in gain media from causality.
- [B. Nistad, J. Skaar, “Causality and electromagnetic properties of active media,” Phys. Rev. E 78, 036603 (2008)](https://doi.org/10.1103/PhysRevE.78.036603). Wavevector selection and absolute instability.
- [J. O. Grepstad, J. Skaar, “Total internal reflection and evanescent gain,” Opt. Express 19, 21404 (2011)](https://doi.org/10.1364/OE.19.021404). Causal roots and amplified total internal reflection at a gain medium.
- [A. E. Siegman, “Fresnel Reflection, Lenserf Reflection and Evanescent Gain,” Opt. Photonics News 21(1), 38 (2010)](https://doi.org/10.1364/OPN.21.1.000038). A view opposing amplified total internal reflection.
- [K. J. Willis, J. B. Schneider, S. C. Hagness, “Amplified total internal reflection: theory, analysis, and demonstration of existence via FDTD,” Opt. Express 16, 1903 (2008)](https://doi.org/10.1364/OE.16.001903). Amplified total internal reflection demonstrated with FDTD.
- [T. S. Mansuripur, M. Mansuripur, “Fresnel reflection from a cavity with net roundtrip gain,” Appl. Phys. Lett. 104, 121106 (2014)](https://doi.org/10.1063/1.4869578). A view opposing amplified total internal reflection at a single interface.
- [M. Mansuripur, P. K. Jakobsen, “Theoretical analysis of Fresnel reflection and transmission in the presence of gain media,” Opt. Rev. 28, 626 (2021)](https://doi.org/10.1007/s10043-021-00690-4). Amplified total internal reflection analyzed using causal wavepackets.
- [A. Lakhtakia, J. B. Geddes III, T. G. Mackay, “When does the choice of the refractive index of a linear, homogeneous, isotropic, active, dielectric medium matter?,” Opt. Express 15, 17709 (2007)](https://doi.org/10.1364/OE.15.017709). The result that root sign does not change the answer for a finite layer.

References for the compared solvers—RETICOLO, S4, and fmmax—appear in {ref}`Appendix A <complex-kx-roots>`.

The original Siegman (2010) article was not available to read; its position was checked only through the papers citing it. Wavebit's conversion of `theta = 0.4` to {math}`k_x`, and its gain-substrate and gain-layer values, are recorded in `docs/validation/learning-records/complex-kx-roots.json`. Pre-October-2, 2026 Wavebit values (path D) remain under `path_d` with their commit. The drawing script computes Wavebit's entry in the “30°” table from the phase-angle equation. Before writing figures, `tools/docs/learning/draw_complex_kx_roots.py` verifies every plotted number: the path-D and path-E closed forms against step-by-step continuation, each branch-cut equation against actual grid discontinuities (including {math}`\varepsilon=4+3i`), and the stated meanings of the two angle conventions.

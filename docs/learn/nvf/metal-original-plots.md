# 13. Finding anomalies in metal calculations by reading plots

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

A peak appears in a calculation. Adding harmonics moves it to another wavelength. How should we interpret that result?

Metal structures can have strong physical resonances, so a sharp curve alone is not proof of error. Conversely, smooth curves or balanced energy do not guarantee accuracy. This chapter uses **power balance, changes with truncation, and independent implementations** in sequence to identify features needing investigation.

The next two chapters pursue that question. [Chapter 14](metal-inverse-rule.md) explains the inverse-rule matrices, and [Chapter 15](metal-regularization.md) describes a small-singular-value floor and its validation.

```{note}
These figures show **pre-correction results** from September 2026 comparisons. A1–A4 select and replot curves from the same recorded CSVs; the original composite figure remains in a dropdown. The original collection was in `studies/results/` on Wavebit branch `claude/modest-pascal-7rrh23`. The English page uses the original English plot bytes; the Korean page preserves the annotated copies. The comparison CSVs are retained unchanged. The record identifies `6fe4b7b` as the pre-correction Wavebit baseline.

The black `E − ΔC` curves test a different assembly order only. **They do not apply Chapter 15's singular-value floor.**
```

## 1. What was calculated?

First distinguish the structures from their observables. All lengths in this table are in nm.

| Case | Structure | Observable |
| --- | --- | --- |
| A | Gold film, thickness 150; air-hole diameter 200; period 500. Air above, glass below | Transmission peak near 820 nm |
| B | Gold disk, diameter 140, thickness 50, period 450; surrounding index 1.5 | Transmission dip near 785 nm |
| D | TiO₂ nanofin rotated 30°, section 250×85, height 600, period 325 | Cross-polarized transmission at 532 nm |
| E | a-Si ellipse rotated 30°, section 300×150, height 715, period 650 | Cross-polarized transmission at 915 nm |
| J | Air hole in a dielectric film of index 3.5 | Transmission dip near 890 nm |

A and B investigate metal anomalies. D, E and J are controls for similar discrepancies without metal.

The `wavelength` axis is optical wavelength. `harmonics` or `number of plane waves` counts retained harmonics. Square truncation at per-axis order {math}`M` uses {math}`(2M+1)^2` functions: 121 means {math}`M=5`, and 441 means {math}`M=10`. Circular truncation's actual count may differ from its request.

Vertical axes differ by panel: distinguish transmittance {math}`T`, reflectance {math}`R`, their sum {math}`R+T`, absorptance {math}`1-R-T`, and peak/dip wavelength. **Peak height and peak position are separate convergence measures.**

The P·F·S·R·L identifiers below locate annotations in the contemporaneous research notes and the preserved Korean figures. The English plots retain their original, unannotated bytes; the text supplies the corresponding panel, curve and wavelength locators. In the annotated copies, red denotes Wavebit, orange fmmax, purple S4, and green Li/Laurent comparisons.

## 2. First check power balance: disk B

With incident power normalized to 1, this passive structure must satisfy

```{math}
A_{\mathrm{abs}}=1-R-T\geq0
```

Gold absorbs light, so usually {math}`R+T<1`. A violation {math}`R+T>1` far beyond numerical error indicates that the calculated values cannot be accepted as physical reflection and transmission efficiencies.

```{figure} ../../figures/learn/nvf/14-B_passivity.png
:alt: Five panels comparing B's power sum at different harmonic counts. The left two are broad wavelength sweeps; the right three show tracked points near the dip. The region above 1 is shaded.
:figclass: learning-figure

Figure 13-1. {math}`R+T` for B on a logarithmic vertical axis. Annotation locators: P4 is the blue 121-harmonic maximum near 1.13; S1 and F1 are the 441-harmonic orange circular-S4 and green square-fmmax maxima near 49 and 16. P5/P8 is the black 441-harmonic assembly-order curve reaching about 1.81; P7 is the blue 1225-harmonic kink reaching about 0.997. Enlarge the figure to read the formulation and truncation legends.
```

Start with **the leftmost, 121-harmonic panel**. The blue Wavebit curve exceeds the reference line 1 around 830–860 nm, with a recorded maximum near 1.13. The calculation outputs more power than enters, so its transmission in that interval cannot be trusted directly.

**The second, 441-harmonic panel** shows larger differences among codes: square-truncated fmmax reaches about 16, and circular S4 Pol about 49. Agreement under other conditions does not qualify a code as a reference for this one.

**The 1225-harmonic panel on the right** marks a Wavebit point approaching 1 from below. It stays within the bound but has a sharp kink at neighbouring wavelengths, warranting further checks. Testing {math}`R+T\leq1` catches obvious failures, not every error.

The three right-hand panels contain only tracked points near the dip. Unsampled wavelength intervals are not thereby validated.

## 3. Read peak shape alongside absorption: nanohole A

For A, comparing power balance with the spectrum makes the issue clearer.

```{figure} ../../figures/learn/nvf/14-A_nvf_closing.png
:alt: Four panels for gold nanohole A. Upper left: transmission with 441 harmonics. Upper right and lower left: absorption. Lower right: close-up of the 1681-harmonic transmission peak.
:figclass: learning-figure

Figure 13-2. Close-ups of suspicious intervals in A. Blue is pre-correction Wavebit; black changes assembly order to {math}`E-\Delta C` using the same material coefficients. P1 locates the blue 806–807 nm peak in panel (a), the nearby 804–805 nm feature of the dotted raster/circular-projector curve, and negative absorption in panel (b). P2 is the roughly −0.27 absorption minimum near 818 nm in panel (c); P3 is the narrow 1681-harmonic peak near 823 nm in panel (d).
```

In **upper-left panel (a)**, the blue transmission curve rises above its surrounding trend around 806–807 nm. Alone, this could be a narrow resonance. At the same wavelengths in **upper-right panel (b)**, absorption falls below zero. Together, these observations rule out interpreting the feature as the ordinary response of this passive structure.

**Figure 13-2(c) shows absorption with 1369 harmonics.** It is the lower-left panel, with wavelength horizontally and absorptance {math}`1-R-T` vertically.

Near 818 nm, the blue curve reaches about −0.27. With incident power set to 1, negative absorptance means a reflected-plus-transmitted power sum near 1.27. That is impossible for this absorbing gold structure. More harmonics than the earlier 441-harmonic calculation have not removed the power-balance violation.

**Lower-right panel (d), with 1681 harmonics**, gives a different lesson. Its blue peak near 823 nm is taller and narrower than comparisons at the same square truncation. The power sum did not exceed 1 here, so passivity alone would have been insufficient to flag it. This motivates checking harmonic count, independent implementations and internal matrices together.

A smoother black curve here does not establish that changing assembly order solves the problem. In the earlier B figure, the same change produced {math}`R+T\approx1.81` under another condition. Improvement in one example must be checked elsewhere.

## 4. Does increasing harmonic count track the same feature?

### A: inspect curve shape, peak position and height separately

Keep the gold nanohole fixed and vary harmonic count. Ask whether **the same peak remains at the same wavelength and height**. Four selected plots from the original CSVs isolate this question from the complex overall comparison.

**A1. First inspect the curve over a broad wavelength range.** Horizontal is wavelength; vertical is transmitted-light fraction. Blue uses 441 harmonics; grey dashed uses 121.

```{figure} ../../figures/learn/nvf/13-reading-spectrum.svg
:alt: A1 overall curves: pre-correction Wavebit NVF transmission with 121 and 441 harmonics. An arrow marks the small feature near 806 nm on the 441-harmonic curve.
:figclass: learning-figure

Figure 13-3a. Stored calculations at 2 nm spacing, joined by lines. The gold nanohole is unchanged; harmonic count differs.
```

The 441-harmonic curve has a broad peak near 820 nm and another small feature to its left near 806 nm. This plot identifies **where that feature is**. Determining whether it is a physical resonance also requires the absorption discussed above and results at other orders.

**A2. Next inspect the peak with more harmonics.** This comparison was extracted from a small inset of the original composite figure.

```{figure} ../../figures/learn/nvf/13-reading-detail.svg
:alt: A2 detailed transmission over narrow wavelength intervals with 961, 1225 and 1681 harmonics. The 1681-harmonic curve rises sharply near 823 nm.
:figclass: learning-figure

Figure 13-3b. This is not simply a zoom of A1: these are **separately computed results with changed harmonic counts**. Curves have different wavelength intervals and sample spacings; only recorded points are connected.
```

The blue 1681-harmonic curve rises into a tall, narrow feature near 823 nm. **Do not interpret this as a magnified view of the same 806 nm point** in A1's 441-harmonic curve. Both harmonic count and the suspicious feature's wavelength have changed.

More harmonics do not automatically imply greater accuracy. Passivity checks, independent calculations and matrix diagnostics are needed to investigate this narrow peak.

**A3. Now extract only the wavelength of each calculation's maximum.** Unlike the previous figures, its horizontal axis is **harmonic count**, not wavelength.

```{figure} ../../figures/learn/nvf/13-reading-position.svg
:alt: A3 peak wavelength versus harmonics for pre-correction Wavebit, square fmmax and circular fmmax. Red outlines mark Wavebit at 1369 and 1681 harmonics.
:figclass: learning-figure

Figure 13-3c. Vertical is peak wavelength. Positions are extremum estimates stored in the CSV. Red outlines identify the two calculations whose anomalies were examined above.
```

Three blue Wavebit points follow roughly **823.5 nm (441) → 818.6 nm (1369) → 822.8 nm (1681)**. Increasing harmonic count moves the peak one way and then back.

Why interpret this cautiously? When selecting a spectrum's highest point, **a narrow spike atop a broad peak wins if it is taller**. A line connecting maxima may therefore fail to track the same physical resonance at every order. That is why A2 examines the underlying curve.

**A4. Finally inspect the height at that maximum.** Even if the position returns near its original value, a different height prevents a claim that the answers have converged.

```{figure} ../../figures/learn/nvf/13-reading-height.svg
:alt: A4 peak heights: Wavebit at 1369 and 1681 harmonics exceeds the fmmax comparison values. All values fit within the vertical range.
:figclass: learning-figure

Figure 13-3d. Vertical is peak transmission, using the CSV's extremum-height estimates unchanged. Values outside the original composite plot's axes are all visible here.
```

Red outlines mark Wavebit at 1369 and 1681 harmonics. At 1681, A3's position returns to about 823 nm, but A4's height still exceeds fmmax: **a difference that position alone can miss**.

At large harmonic counts, circular fmmax peak positions cluster around 822.5–822.8 nm. This helps comparison but does not establish truth from one code's stable curve. Circular and square truncations also use different Fourier bases.

:::{dropdown} View the original comparison with all codes
```{figure} ../../figures/learn/nvf/14-A820_peak.png
:alt: Original A comparison: eight spectra on the left, with peak positions and heights on the right.
:figclass: learning-figure

In the **right panel of the third row within the left two columns**, titled `wavebit NVF (exact circle) — NVF on`, the broad curve corresponds to A1 and its inset to A2. A3/A4 select Wavebit and fmmax from the right-hand summary graphs.

Annotation locators: F3/S4 are the reduced Laurent peaks in the first-column fmmax/S4 441-harmonic spectra; their tracked wavelengths span roughly 847–862 nm in the upper-right plot. L1 is the Wavebit Laurent peak about 5 nm longer and 60% as high as the NVF comparison. R1/L1 locates the Li/Wavebit-Laurent tracked peaks around 826–831 nm. P1 is the Wavebit 441-harmonic feature near 806 nm; P3 is the 1681-harmonic narrow peak in its inset. P2/P3 identifies the power-violating or narrow-peak points in the right-hand summaries, including peak transmissions about 0.221 and 0.129 outside the original bottom-right axis. P9/F2 locates the several-nm, approximately 2–5 nm, spread among square-truncated curves in the middle-right plot.
```
:::

### B: distinguish dip movement from power-balance violations

```{figure} ../../figures/learn/nvf/14-B780_dip.png
:alt: B transmission-dip spectra by code and dip position/depth versus harmonics. Tracked positions differ by formulation; some curves exceed the displayed range.
:figclass: learning-figure

Figure 13-4. Full comparison for B. Broad spectra use 4 nm spacing; some dip-region insets use 0.5 nm spacing. Annotation locators: F3/S4 identifies Laurent dip shifts of roughly 30–50 nm per truncation step in the upper-right summary. S2 locates circular S4 Pol dips about 20–40 nm shorter than other codes; S1 locates its 441-harmonic off-scale transmission near 22.7 ({math}`R+T` about 49). In the Wavebit NVF spectrum, P4 is the 121-harmonic region with {math}`T>1`, and P5 is the 441-harmonic feature in 748–764 nm. P12 marks the Wavebit Laurent dip near 836 nm. P9/F2/S2 locates irregular square-truncation dip tracking in the middle-right summary; its red markers indicate {math}`R+T>1`. F1 denotes square fmmax at 441 harmonics, with {math}`R+T` near 16 outside the bottom-right plot range.
```

**The middle panel on the right** compares NVF calculations. Square fmmax at 441 harmonics gives a point near 768 nm, but the surrounding spectrum includes {math}`R+T>1`. That position cannot simply be treated as an ordinary point on a convergence curve.

At the tested harmonic counts, circular S4 Pol's dip is generally at shorter wavelengths than other NVF results. This is an independent discrepancy; the plot alone cannot attribute it specifically to the inverse rule, normal-field construction or truncation.

Also inspect **the Laurent panels on the left**. At these finite orders, their dip positions differ substantially from NVF. Removing divergence by disabling NVF and converging to the correct position are separate claims.

Circular fmmax is relatively stable here, but its recorded high-order sequence is 785.50 → 784.36 → 784.66 nm. A tendency to cluster is distinct from strict monotone convergence.

## 5. What remains without metal?

Even when the metal inverse rule is suspect, it cannot explain every difference among codes. Dielectric controls help separate issues.

### D/E: energy conservation differs from convergence of one polarization channel

```{figure} ../../figures/learn/nvf/14-DE_P_yx.png
:alt: Top row: TiO2 nanofin D. Bottom row: a-Si ellipse E. Each shows cross-polarized transmission, a close-up, change from the preceding truncation, and power-sum deviation from 1.
:figclass: learning-figure

Figure 13-5. Rotated dielectric structures. Here {math}`P_{yx}` is the **zeroth-order transmitted-power ratio** in x polarization for y-polarized incidence. The two rightmost columns use logarithmic axes. P10 locates Wavebit's nonmonotone E sequence in the bottom-row close-up. S3 locates S4 Pol's E values near 0.21–0.22, about 0.07 below the other NVF curves. F5/S5 locates power-sum deviations in the far-right D and E panels; red rings identify values above 1. For E, the Wavebit symmetric closure is near {math}`10^{-13}` in that panel: good power balance does not settle the channel-convergence issue.
```

In **the second panel of the bottom row**, Wavebit's E curve varies around 0.28–0.30. Yet **the far-right panel of that row** has very small power-balance deviations. Good energy conservation can coexist with order-dependent oscillations of one polarization channel.

**The third column** shows differences from the preceding order. A small value means those two calculations are close, not that their error relative to the exact solution is equally small. Both approximations could be biased in the same direction.

D includes a small material loss, so physical absorption contributes to {math}`|R+T-1|`. Do not interpret every top/bottom-row deviation as purely numerical error. Nevertheless, passive-material points with {math}`R+T>1` deserve separate scrutiny; the original figure marks them with red circles.

The metal correction does not claim to resolve E's normal-field-dependent oscillations or S4 Pol's polarization discrepancies.

### J: read small differences in the context of the magnified scale

```{figure} ../../figures/learn/nvf/14-J_res_4codes.png
:alt: J, an air hole in a dielectric film. Left: spectra by code. Right: overall dip positions, NVF close-up, and changes from the preceding order.
:figclass: learning-figure

Figure 13-6. Dielectric resonance J. Broad spectra use 1 nm spacing; dip insets use 0.25 nm. Tracked wavelengths include estimates between samples. In the upper-right summary, L1 locates Laurent results still approaching the NVF cluster from longer wavelengths; R2 locates Li-rule results approaching from about 2 nm below at the largest plotted truncations. P13 locates the blue 1225-harmonic Wavebit point about 0.025 nm above its neighbours in the middle-right enlargement.
```

**Upper right**, high-order NVF results approach roughly 890 nm, while Laurent and Li are still moving from opposite sides.

In the magnified **middle-right panel**, Wavebit at 1225 harmonics is about 0.025 nm above adjacent orders. Although visually prominent, this differs in scale and character from the several-nm shifts or {math}`R+T=45` seen in metals. Judge it alongside wavelength spacing, extremum estimation and required accuracy. Reporting three decimal places from 0.25 nm-spaced samples does not validate those digits.

## 6. Take the questions from the plots to the matrices

These figures help distinguish three observations.

| Observation | Supported conclusion | Further checks |
| --- | --- | --- |
| Power sum greatly exceeds 1 for a passive structure | The calculation violates physical power balance | Normalization, material conventions, matrix assembly, numerical stability |
| A narrow feature moves or kinks with truncation | Investigate a possible numerical feature | Denser wavelength sweeps, internal matrices, independent calculations |
| Codes still differ as orders increase | No agreed converged value at those settings | Basis sets, material values, geometry representations, formulations |

These observations alone cannot establish one cause. Next ask **which internal operation produces a large response** in the suspicious interval. [Chapter 14](metal-inverse-rule.md) directly examines the inverse of {math}`\left[\!\left[1/\varepsilon\right]\!\right]`.

The original numbered details and the comparison CSVs remain in a separate lab repository that is not published. This chapter shows pre-correction comparisons. Direct before/after comparisons follow in [Chapter 15](metal-regularization.md).

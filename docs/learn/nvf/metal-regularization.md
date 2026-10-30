# 15. Flooring small singular values: correction and validation

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

[Chapter 14](metal-inverse-rule.md) identified a potentially problematic operation: small singular values of {math}`B=\left[\!\left[1/\varepsilon\right]\!\right]` become large amplification in its inverse. This correction floors those small values while retaining the existing NVF assembly.

This chapter follows **what changed → how the recorded results changed → what remains unproved**. First, we define the floor used in those results.

```{note}
The pre-correction baseline is `6fe4b7b`, the implementation is `f90cf1d`, its explanation is `c5267d8`, and backend comparison tests are `56f1cd5`. The original working branch was `claude/determined-clarke-dj4lij`.

This documentation revision cross-checked stored CSVs against code and tests at those commits. It does not report newly rerun optical calculations or a fresh full solver test suite. Conditions for structures A and B are in [Chapter 13](metal-original-plots.md).
```

## 1. Insert the correction immediately before inversion

The selected method floors the small singular values of {math}`B`. Singular values measure how much a matrix stretches or contracts vectors. A smaller minimum singular value permits greater amplification by the inverse.

Let {math}`\varepsilon_{\max}=\max_{\mathbf r}|\varepsilon(\mathbf r)|`, and write

```{math}
B=U\operatorname{diag}(s_j)V^{\dagger},\qquad
s_{\mathrm{floor}}=\frac1{2\varepsilon_{\max}}
```

The corrected matrix is

```{math}
\widetilde B=
U\operatorname{diag}\!\left(\max(s_j,s_{\mathrm{floor}})\right)V^{\dagger},
\qquad \widetilde Q=\widetilde B^{-1}
```

The inverse operator norm is then bounded by

```{math}
\|\widetilde Q\|_2\leq2\varepsilon_{\max}
```

**The singular values receive a lower bound; the resulting inverse amplification receives an upper bound.** This does not clip reflectance or transmittance after calculating them.

For general matrices, singular values do not equal eigenvalue magnitudes. In Chapter 14's two-material model, however, {math}`B` is a linear function of Hermitian {math}`K` and is normal, so they coincide. There the correction can be understood as limiting excessive magnitude while retaining each mode's complex phase.

### Why an upper-bound factor of 2?

This is a **regularization factor** selected by comparisons in the earlier study, not a constant prescribed by a physical law. In one 441-harmonic condition for A, a cap of {math}`\varepsilon_{\max}` corrected 300 modes; caps of {math}`2\varepsilon_{\max}` and {math}`3\varepsilon_{\max}` corrected 6 and 4, respectively. Too low a cap can substantially alter responses away from the pole.

For A's spectrum compared with fmmax at the same square truncation, factor 2 gave a transmission RMS difference of about 0.0015, versus about 0.0021 for factor 3. That comparison motivated factor 2. It does not establish fmmax as ground truth in all conditions or 2 as optimal for every metal.

### When are dielectrics unaffected?

For a layer containing only positive real permittivities,

```{math}
s_{\min}(B)\geq\frac1{\varepsilon_{\max}}
>\frac1{2\varepsilon_{\max}}
```

so the floor is inactive. For lossy materials, the distribution of {math}`1/\varepsilon` in the complex plane also matters. The floor was inactive under the earlier analysis's restricted loss-angle conditions and in the tested dielectric examples. That is not a guarantee for arbitrary complex permittivities.

When no singular value needs correction, the code returns the original {math}`B`, rather than a matrix reconstructed by SVD. This explains the recorded bitwise equality before and after the change in the tested dielectric cases.

Since 2026-10-28 the code first checks the layer's materials. If every permittivity can be read and satisfies {math}`\operatorname{Re}\varepsilon>0` and {math}`\operatorname{Im}\varepsilon\geq 0`, the floor is not evaluated at all, so no singular value of {math}`B` is computed. Otherwise, it computes the singular values alone, and the full SVD only when one of them is below the floor. Skipping the full SVD is not only cheaper. PyTorch's SVD (MKL) once failed to converge on the {math}`B` of a dielectric circle at 961 harmonics, where the floor would not have applied.

## 2. Changes supported by stored data

First compare the same wavelengths. Values below are rounded from the records.

| Case | Harmonics | Wavelength | Before: {math}`R+T` | After: {math}`R+T` |
| --- | --- | --- | --- | --- |
| A hole | 441 | 807 nm | 1.0019 | 0.9404 |
| A hole | 1369 | 818 nm | 1.2726 | 0.8896 |
| B disk | 121 | 852 nm | 1.1259 | 0.8158 |
| B disk | 441 | 776 nm | 0.6745 | 0.5719 |
| B disk | 625 | 760 nm | 45.2584 | 0.7676 |

For B at 441 harmonics, the power sum was already below 1. The concern there was a local spectral kink and disagreement with other calculations. Describing every improvement solely as restored passivity would miss this distinction.

```{figure} ../../figures/learn/nvf/13-b-disk-625-rt.png
:alt: B with 625 harmonics. At 760 nm the power sum decreases from about 45.26 before correction to about 0.77 afterward. The vertical axis is logarithmic.
:figclass: learning-figure

Figure 15-1. B, 745–775 nm at 1 nm spacing. Equal vertical distances on this logarithmic axis represent equal ratios, not equal differences.
```

To avoid drawing conclusions from a selected point, inspect the complete recorded interval. The following compares `sym` and `sym_cap2` rows in `sweep_*.csv`.

| Case · harmonics | Wavelength interval · spacing | Points | Before: points with {math}`R+T>1` | After: such points | After: maximum {math}`R+T` |
| --- | --- | --- | --- | --- | --- |
| A 441 | 796–840 nm · 1 nm | 45 | 1 | 0 | 0.958125 |
| B 121 | 820–870 nm · 1 nm | 51 | 43 | 0 | 0.874622 |
| B 441 | 740–830 nm · 2 nm | 46 | 0 | 0 | 0.870127 |
| B 625 | 745–775 nm · 1 nm | 31 | 15 | 0 | 0.865175 |

These are **checks at stored wavelength samples**. They do not validate every wavelength between samples or other structures.

```{figure} ../../figures/learn/nvf/13-a-hole-441-spectrum.png
:alt: A transmission with 441 harmonics. The narrow feature near 806 nm is reduced by the correction, bringing the curve closer to fmmax at the same square truncation.
:figclass: learning-figure

Figure 15-2. A, 796–840 nm. Displayed points are 2 nm apart; the original Wavebit sweep used 1 nm spacing. Solid fmmax is square truncation; dashed fmmax is circular truncation.
```

In the earlier comparison, the transmission RMS difference from square-truncated fmmax over this interval decreased from about 0.0094 to 0.0015. This supports improved agreement with an independent implementation. That agreement and a smoother curve are useful evidence, but neither gives absolute error.

```{figure} ../../figures/learn/nvf/13-a-hole-1369-spectrum.png
:alt: A transmission with 1369 harmonics: the large pre-correction peak, the corrected curve, and one fmmax comparison point at 818 nm.
:figclass: learning-figure

Figure 15-3. A, 1369 harmonics. The fmmax marker is **one point at 818 nm**; it does not show agreement over the entire curve. Corrected values were retained in the earlier figure-generation code.
```

```{figure} ../../figures/learn/nvf/13-b-disk-1225-reflection.png
:alt: B reflection with 1225 harmonics: a pre-correction kink at 788–792 nm and smoother corrected variation. The compared codes sample different wavelength ranges.
:figclass: learning-figure

Figure 15-4. B, 1225 harmonics. After correction, {math}`R` is about 0.433–0.437 at 786–792 nm. Pre-correction, corrected and comparison-code wavelength samples are not identical.
```

## 3. Interpreting comparisons with other codes

Equal harmonic counts do not ensure equal numerical conditions. Square truncation selects a rectangular set of Fourier-order pairs; circular truncation selects an approximately circular reciprocal-space set. Its actual basis size may differ from the request: here, fmmax's circular request for 441 used 437 basis functions.

Changing the basis also changes the eigenvalues of {math}`K`. Consequently, near-pole modes and changes in the optical spectrum can occur at different wavelengths.

```{figure} ../../figures/learn/nvf/13-b-disk-441-rt-fmmax.png
:alt: Power sums for B across implementations. In this interval square-truncated fmmax reaches about 16 and circular S4 Pol reaches about 49.
:figclass: learning-figure

Figure 15-5. B, with requested harmonic counts around 441. Circular and square truncations use different bases; the vertical axis is logarithmic. This figure cannot evaluate every setting of each code.
```

Square-truncated fmmax, smooth for A, gives {math}`R+T\approx16.1` for B. Circular S4 Pol records about 49. Neither a particular code nor circular truncation can automatically serve as ground truth. These data alone do not identify the exact internal operation responsible for S4's similar anomaly.

There are also no rerun data applying this correction to fmmax or S4. We cannot conclude that the same correction would produce the same improvement there.

## 4. Remaining limitations

**There is no proof of passivity.** After correction, the assembled loss matrix still had minimum eigenvalues around −1.8 to −2.4 under the diagnosed metal conditions. Their magnitudes decreased, but negative directions remained. This limitation belongs alongside the observed {math}`R+T\leq1` in recorded sweeps.

**Autodiff does not compute the exact derivative of the entire correction.** The implementation applies JAX `stop_gradient` or Torch `detach` to the correction {math}`\widetilde B-B`, avoiding derivative instabilities at degenerate singular vectors. When the floor is active, this ignores the correction's parameter dependence and generally need not equal the exact derivative of the corrected forward function. Since 2026-10-02 this no longer describes the code: the floor as it landed on develop (`90bde542`) supplies the floored matrix's first derivative in closed form, exact where the floor binds, and refuses a second derivative there.

Tests at `56f1cd5` compare the {math}`R+T` derivative with respect to shape variable `lx` against central differences for specific gold-disk and lossy-dielectric-disk examples. Tolerances are relative {math}`10^{-4}` and absolute {math}`10^{-9}`. This limited comparison does not guarantee derivatives for every shape, material or floor crossing.

**Convergence with harmonic count remains unresolved.** In the earlier A/B records, peak and dip positions move with truncation shape and harmonic count. Circular fmmax is relatively stable in some intervals, but B's recorded sequence 785.50 → 784.36 → 784.66 nm is not strictly monotone convergence.

**Cross-polarized oscillations in dielectric E are a separate issue.** The a-Si ellipse oscillates with harmonic count even though the metal correction is inactive.

```{figure} ../../figures/learn/nvf/13-e-ellipse-pyx-ladder.png
:alt: Cross-polarized transmission of dielectric ellipse E versus harmonic count: a separate convergence problem outside the metal correction's scope.
:figclass: learning-figure

Figure 15-6. E at 915 nm. Here {math}`P_{yx}` means the zeroth-order transmitted-power ratio in x polarization for y-polarized incidence. Follow this definition rather than inferring an index order from usual matrix notation.
```

The original study also tested alternative assemblies intended to preserve a passive absorption part. Congruence-based assembly encountered the failure of exact {math}`C^2=C` in a finite basis; replacing only the absorption part harmed uniform-layer consistency or comparison spectra. For those conditions, the existing symmetric assembly plus a floor was selected. That does not constitute a complete general solution.

## 5. Following the evidence

| What to check | Source |
| --- | --- |
| Sweep numbers in this chapter | CSVs and figure-generation code in a separate lab repository, not published |
| Original issue labels P·F·S·R·L and detailed comparisons | Contemporaneous research notes in the same unpublished lab repository; the published counterpart is the English record below |
| Floor, symmetric assembly and stopped derivatives | [factorization.py at `f90cf1d`](https://github.com/fieldalgo/wavebit/blob/f90cf1d/wavebit/_core/factorization.py) |
| Design decisions and measurements at the time | [English record at `c5267d8`](https://github.com/fieldalgo/wavebit/blob/c5267d8/docs/nvf-metal-inverse-rule.md) |
| Backend and central-difference comparison conditions | [Tests at `56f1cd5`](https://github.com/fieldalgo/wavebit/blob/56f1cd5/tests/test_nvf_metal_inverse_rule.py) |
| Reading the pre-correction figures | [Chapter 13: finding anomalies in plots](metal-original-plots.md) |

The figure generator stores some earlier measurements as arrays in its code, so not every curve can be reconstructed from CSV alone. In particular, corrected A 1369/1681 and B 1225 points are supported by those arrays. Distinguish original spectral samples from interpolated peak and dip positions.

Solver test-pass counts in the original record belong to that execution environment. This documentation revision checked provenance, agreement between code and explanation, documentation builds and links; it did not rerun those numerical studies.

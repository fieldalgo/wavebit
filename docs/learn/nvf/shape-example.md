# 07. A first calculation: NVF reflection and transmission for a circular inclusion

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

Run the examples in an environment with the Wavebit source checkout installed. The reference version's installation instructions use `python -m pip install .` from the source directory. Wavebit is also published on PyPI (`pip install wavebit`); a release may differ from the checkout you are reading, so compare `wavebit.__version__`. Follow the checkout's [installation guide](../../getting-started/installation.md) for Python and dependency requirements.

```bash
# First install from the Wavebit source directory
python -m pip install .

# Then run from the same directory
python examples/rcwa/learning/nvf/01_shape_nvf.py
```

The code below is the same as the {download}`complete Example 1 file <../../../examples/rcwa/learning/nvf/01_shape_nvf.py>`. All lengths are interpreted in nm. `problem`, `run` and `summarize` are functions defined in this example for explanatory purposes.

```{literalinclude} ../../../examples/rcwa/learning/nvf/01_shape_nvf.py
:language: python
:linenos:
```

## Input conventions to check

**Material numbers mean relative permittivity ε, not refractive index n.** The example's `Material(epsilon=4.)` corresponds to n=2 for a nonmagnetic material. `bottom=Material(epsilon=2.25)` corresponds to n=1.5. Convert an index using `wavebit.eps_from_refractive_index(n)` when necessary. With the `exp(−iωt)` time convention, passive loss ordinarily uses `Im(ε)>0`.

`Circle(radius=160.)` specifies a **radius** of 160 nm. `center=(0., 0.)` is the cell center. `Stack.layers` lists layers from bottom to top; `top` and `bottom` are the semi-infinite media above and below. Use the same length unit for `wavelength`, `period`, `thickness`, and shape dimensions; angles are in radians.

`theta=.2` and `phi=.3` specify incidence direction. The example separately solves `PlaneWave(polarization="p")` and `"s"`; each result supplies its own incident polarization’s `R` and `T`.

## Measured execution results

These values were obtained from the reference code on a NumPy CPU backend in double precision. The complete output and environment are in {download}`measurements.json <../../validation/learning-records/nvf-measurements.json>`. The number of printed digits is not a guarantee of that many digits of physical accuracy.

| Incidence | Total R | Total T | Energy residual abs(R+T−1) |
| --- | --- | --- | --- |
| p | 0.002869503084 | 0.997130496916 | 1.44×10⁻¹⁵ |
| s | 0.005397410043 | 0.994602589957 | 8.88×10⁻¹⁶ |

`R.sum()` and `T.sum()` sum power efficiencies over every diffraction order. They are not simple sums or squared magnitudes of electric-field amplitudes. Because this example is lossless, R+T≈1 is a useful check, but it does not by itself establish accuracy at the chosen Fourier truncation.

## x/y orders and array ordering

```{figure} ../../figures/learn/nvf/09-orders.svg
:alt: Order declarations put x first; result arrays use y rows and x columns
:figclass: learning-figure

Figure 7-1. Order declarations put x first; result arrays use y rows and x columns
```

In `Numerics(orders=(Mu, Mv))`, the half-widths along the two lattice vectors (x and y on this rectangular cell) are independent and always given as a pair. Current arrays use `(wavelength, v, u)`, so `(2,1)` at one wavelength gives `(1,3,5)`. Inspect `result.transmission.orders_u` and `.orders_v`, or use `result.transmission.order(m,n).efficiency` for a particular transmitted order. Not every retained mode propagates into the far field; evanescent modes are also needed to represent near fields and boundary conditions.

```{raw} html
<section class="exercise" aria-labelledby="orders-title"><h3 id="orders-title">Try it · Which Fourier modes am I retaining?</h3><p class="explain">Only the selected integer modes are shown. Dots represent (m,n) Fourier labels, not polarizations.</p><div class="controls"><label for="nx">Nx <output id="nx-value" for="nx">2</output><input id="nx" type="range" min="0" max="4" step="1" value="2"></label><label for="ny">Ny <output id="ny-value" for="ny">1</output><input id="ny" type="range" min="0" max="4" step="1" value="1"></label></div><svg class="mini-svg" viewBox="0 0 400 340" role="img" aria-label="Grid of selected Fourier modes"><rect x="32" y="20" width="312" height="280" fill="white" stroke="#b3cbc2"/><g id="mode-dots"></g><text x="370" y="172" font-size="16" fill="#182b3e">m</text><text x="189" y="14" font-size="16" fill="#182b3e">n</text><text x="187" y="325" font-size="14" fill="#526779">0</text></svg><div class="readout" id="orders-result" aria-live="polite"></div><noscript><p class="no-js">The formula in the text gives the 15 modes retained by (2,1).</p></noscript></section>
```

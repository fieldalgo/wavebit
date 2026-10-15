# 07. A first calculation: NVF reflection and transmission for a circular inclusion

:::{note}
The runnable examples in this imported course explicitly use `wavebit.compat.legacy` to preserve the workflow behind its historical numerical records. For new work, use the [current getting-started guide](../../getting-started/index.md) and [API migration guide](../../migration/index.md). Historical measurements were not regenerated during this documentation import.
:::

Run the examples in an environment with the Wavebit source checkout installed. The reference version's installation instructions use `python -m pip install .` from the source directory. Wavebit is also published on PyPI (`pip install wavebit`); a release may differ from the checkout you are reading, so compare `wavebit.__version__`. Follow the checkout's [installation guide](../../getting-started/installation.md) for Python and dependency requirements.

```bash
# First install from the Wavebit source directory
python -m pip install .

# Then run from the same directory
python examples/rcwa/learning/nvf/01_shape_nvf.py
```

The code below is the same as the {download}`complete Example 1 file <../../../examples/rcwa/learning/nvf/01_shape_nvf.py>`. All lengths are interpreted in nm. `make_solver` and `summarize` are functions defined in this example for explanatory purposes.

```{literalinclude} ../../../examples/rcwa/learning/nvf/01_shape_nvf.py
:language: python
:linenos:
```

## Input conventions to check

**Material numbers mean relative permittivity ε, not refractive index n.** The example's `material=4.` corresponds to n=2 for a nonmagnetic material. `bot=2.25` corresponds to n=1.5. Convert an index using `wavebit.eps_from_refractive_index(n)` when necessary. With the `exp(−iωt)` time convention, passive loss ordinarily uses `Im(ε)>0`.

For `circle`, `lx=320.` is the **diameter**, not the radius. Shape coordinates `cx=0, cy=0` refer to the cell center. `thickness` lists patterned-layer thicknesses from bottom to top. `top` and `bot` are the semi-infinite media above and below. Use the same length unit for `wavelength`, `period`, `thickness`, and shape dimensions; angles are in radians.

`theta=.2` and `phi=.3` specify incidence direction. `psi=0` selects p incidence, so `result.R` and `result.T` in this example are the p-incident results. `result.p_inc` and `result.s_inc` provide explicit access to the two pure incident polarizations.

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

In `truncation_order=(Nx, Ny)`, x and y orders are independent. Arrays put y first, so `(2,1)` produces result shape `(3,5)`. Inspect `result.orders_u` and `result.orders_v` for column and row ordering, or use `result.order(m,n)` to read a particular order. Not every retained mode propagates into the far field; evanescent modes are also needed to represent near fields and boundary conditions.

```{raw} html
<section class="exercise" aria-labelledby="orders-title"><h3 id="orders-title">Try it · Which Fourier modes am I retaining?</h3><p class="explain">Only the selected integer modes are shown. Dots represent (m,n) Fourier labels, not polarizations.</p><div class="controls"><label for="nx">Nx <output id="nx-value" for="nx">2</output><input id="nx" type="range" min="0" max="4" step="1" value="2"></label><label for="ny">Ny <output id="ny-value" for="ny">1</output><input id="ny" type="range" min="0" max="4" step="1" value="1"></label></div><svg class="mini-svg" viewBox="0 0 400 340" role="img" aria-label="Grid of selected Fourier modes"><rect x="32" y="20" width="312" height="280" fill="white" stroke="#b3cbc2"/><g id="mode-dots"></g><text x="370" y="172" font-size="16" fill="#182b3e">m</text><text x="189" y="14" font-size="16" fill="#182b3e">n</text><text x="187" y="325" font-size="14" fill="#526779">0</text></svg><div class="readout" id="orders-result" aria-live="polite"></div><noscript><p class="no-js">The formula in the text gives the 15 modes retained by (2,1).</p></noscript></section>
```

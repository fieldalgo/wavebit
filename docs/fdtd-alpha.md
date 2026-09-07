# FDTD research-alpha scope

The FDTD path is experimental. Its intended use is research on periodic, isotropic raster
stacks, with a NumPy CPU solver and diffraction results comparable to the RCWA path.
It is not yet a general-purpose replacement for an established FDTD package. This page
states what has evidence, what remains uncertain, and how to reproduce the checks.

## Supported interface versus validation coverage

An accepted input is not a guarantee of accuracy for every combination in this table.

| Area | Available interface | Evidence and limits |
| --- | --- | --- |
| Raster geometry | `(layer, y, x)` arrays; y-invariant and rectangular crossed cells | Analytic planar checks, 1D gratings, and selected asymmetric crossed gratings. The crossed external reference covers two dielectric scenes, not arbitrary 3D geometry. |
| Incidence | From above; finite real scalar angles; TE/TM at zero azimuth; linear `psi` at general azimuth | Invalid complex/nonfinite angles and unsupported polarization settings are refused before either dispatch path. Normal, coplanar and conical regression cases; complex per-order amplitudes and both order axes checked. Near-Rayleigh and grazing incidence require separate refinement. |
| Materials | Positive real isotropic epsilon/mu; electric Drude/Lorentz poles | Closed-form dielectric, magnetic and Drude slabs, selected 1D patterned cases. The new installation matrix covers planar magnetic/Drude media through both raster paths. It does not validate general patterned 3D magnetic or dispersive devices. |
| Spectra | Multiple wavelengths in one pulse run | At oblique incidence the in-plane wavevector is fixed, so angle varies across the band. Check the reported angles and incident spectrum. |
| Outputs | R/T and complex s/p amplitudes, labelled diffraction orders | Frozen external-reference tests include complex propagating amplitudes. Evanescent channels are not covered by the convergence table below; deeply evanescent amplitudes can be masked. |
| Time records | Ey/Ex snapshots as available, measured axes, plotting and animation | Component staggering, coherent polarization, recording invariance and plotting contracts are tested. This is not a complete six-component field or frequency-domain near-field API. |
| Execution | NumPy on CPU; optional Matplotlib plotting | Clean wheel installation and calculation checked on macOS arm64/Python 3.12. Other operating systems and Python versions need release CI evidence. |

FDTD does not provide autodiff, JAX/PyTorch execution, GPU/MPI execution, bottom incidence,
skewed lattices, parametric geometry input, anisotropic tensors, elliptical polarization,
or complex-angle incidence. Use the appropriate RCWA capability where available; do not
infer FDTD support from RCWA examples. Unsupported inputs are described in [the solver
notes](fdtd.md) and [API reference](reference/api.md).

## Reproduce the installation check

Build and install a wheel into a fresh environment without system site packages:

```sh
python -m pip wheel . --no-deps --wheel-dir /tmp/wavebit-wheel
python -m venv /tmp/wavebit-clean
/tmp/wavebit-clean/bin/python -m pip install /tmp/wavebit-wheel/wavebit-0.1.0-py3-none-any.whl
cd /tmp
/tmp/wavebit-clean/bin/python /absolute/path/to/checkout/studies/fdtd_install_smoke.py
```

The script uses public APIs and does not add the checkout to the import path. Its JSON
prints the imported package path, dependency versions, and every measured case. It checks
36 planar combinations: two raster paths, three materials, three incidence directions,
and two linear polarizations. It compares complex amplitudes with RCWA, requires a settled
field, and checks that JAX, PyTorch and Matplotlib were not imported by the base calculation.
On 2026-09-07, Python 3.12 with NumPy 2.5.3, SciPy 1.18.1 and array-api-compat
1.15.0 passed all 36 cases: maximum amplitude difference 0.006079 and maximum residual
4.35e-9. The measurements are in `studies/results/fdtd_alpha_install.json`.

The optional plot extra was also installed (Matplotlib 3.11.1); a public `Stack`/`Incidence`
conical solve recorded Ey/Ex and rendered an Ex slice from the installed package.

The 0.02 amplitude bound is an installation smoke-test criterion, not a promised solver
accuracy. Both solvers belong to Wavebit; independent evidence comes from the existing
external-reference suite (`tests/references/README.md` in the checkout).

## Crossed dielectric refinement

Run from the checkout root:

```sh
python studies/fdtd_alpha_convergence.py --output studies/results/fdtd_alpha.json
```

The checked-in JSON records the source digest, dependency versions, actual grid sizes,
PML thicknesses, runtime, residuals and warnings. The case definitions are shared with
`tests/references/_meep_crossed_cases.py`; no external solver is imported or regenerated.
Lengths here are micrometres. Both s and p incident basis solutions are compared on
explicit order labels. Only channels with `n² - |k_parallel|² > 0.05` are included.
Amplitude errors are absolute complex differences, maximized across r_s/r_p/t_s/t_p and
both sources; power errors are absolute per-order R/T differences, not percentages.

The reference is RCWA at order `(13, 13)`, with `(9, 9)` also evaluated to expose its
sensitivity. A small reference change is evidence of stability over that interval, not
an exact solution. FDTD uses 120 optical cycles, 12 PML cells, Courant fraction 0.99,
subpixel averaging, uniform z spacing, and padding scale 1.0. Holding PML *cell count*
fixed changes physical absorber thickness as the grid refines; the sweep therefore
measures the combined numerical setup, not a pure spatial truncation error.

| Scene | Grid (nx × ny × nz) | Cell size | Max amplitude error | Max power error | Residual |
| --- | --- | --- | --- | --- | --- |
| square_18deg | 14 × 14 × 50 | 0.06000 | 0.07492 | 0.02870 | 1.15e-03 |
| square_18deg | 21 × 21 × 54 | 0.04000 | 0.02538 | 0.01348 | 5.99e-04 |
| square_18deg | 28 × 28 × 61 | 0.03000 | 0.01441 | 0.00740 | 5.23e-04 |
| rect_28deg | 16 × 24 × 53 | 0.05000 | 0.06504 | 0.04774 | 5.91e-03 |
| rect_28deg | 24 × 36 × 63 | 0.03333 | 0.02489 | 0.01697 | 4.56e-03 |
| rect_28deg | 32 × 48 × 76 | 0.02500 | 0.01450 | 0.00917 | 3.99e-03 |

RCWA `(9,9)` → `(13,13)` maximum amplitude changes were 0.000215 (square_18deg), 0.000457 (rect_28deg).


These are two selected lossless dielectric scenes. They establish neither a universal
second-order rate nor an error bound for magnetic, lossy, resonant, grazing or near-Rayleigh
structures. Monitor residual is a settling indicator, not an error estimate. A new device
needs separate increases in runtime, padding/PML and spatial resolution until the requested
observable stabilizes. Symmetry and R+T alone cannot establish amplitude or phase accuracy.

## Runtime and boundary controls

The companion study holds the square scene at resolution 25 and the rectangular scene
at resolution 30. These are the middle grids above, not the finest grids. It changes
runtime from 120 to 180 to 240 optical cycles; at 240 cycles it independently increases
PML depth from 12 to 20 cells and padding scale from 1.0 to 1.5. Each boundary control
changes one parameter from the 240-cycle baseline. This separates their sensitivity
from the spatial sweep without claiming the same sensitivity on finer grids.

```sh
python studies/fdtd_alpha_controls.py --output studies/results/fdtd_alpha_controls.json
```

The JSON records complete settings, actual grids, residuals, warnings, reference errors,
and pairwise differences across the same propagating channels. The table reports the
maximum complex amplitude change between each pair, rather than a total R+T change.

| Control | Square amplitude change | Rectangular amplitude change |
| --- | --- | --- |
| 120 → 180 cycles | 0.000145 | 0.001778 |
| 180 → 240 cycles | 0.000040 | 0.000403 |
| PML 12 → 20 cells | 0.000551 | 0.001804 |
| Padding 1.0 → 1.5 | 0.005509 | 0.002415 |

The time sensitivity falls from 120→180 to 180→240 cycles in both scenes. This supports
using longer runs to reduce the remaining transient, but does not make the spatial error
vanish. PML and padding controls must still be inspected separately. A larger padding
region need not monotonically improve the complex coefficient at fixed resolution.
Residuals differ under the boundary controls: these are finite-runtime sensitivities,
not isolated asymptotic absorber errors. Tight error targets require further runtime
refinement for each boundary setting.

An independent analytic check isolates that last point without RCWA: in vacuum,
`t_s = t_p = exp(i k0 cos(theta) d)`. `tests/test_fdtd_boundary_controls.py` uses wavelength
0.8, thickness 0.2, period (0.16, 0.16), theta 18 degrees and phi 40 degrees, with 120 cycles
and 12 PML cells. For padding scales 1.0/1.5, complex transmission errors are respectively
0.006717/0.008513 at resolution 25, and 0.001716/0.002157 at resolution 50 (both source
polarizations agree at these digits). The error and padding sensitivity decrease with
refinement. This is evidence for a numerical phase contribution even in vacuum; it does
not identify every source of padding sensitivity in a patterned device. The test checks
both the analytic phase and the decrease, rather than relying on transmitted power alone.

## Remaining release gates

Before a tagged public alpha, run the configured CI against the final revision, including
clean installs on the claimed platforms; no remote CI result is asserted by this local
report. Select and publish the supported scope with the release, and retain warnings and
known limitations in examples. Before claiming a broader 3D material capability, add
independent patterned magnetic/dispersive references and repeat runtime/absorber refinement
for those scenes. The dielectric controls above do not cover them.
GPU execution, autodiff and a general field API are separate development projects.

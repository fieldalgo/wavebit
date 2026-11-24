# Profiled layers: what the slicing was checked against

Written: 2026-11-18. Targets: `src/wavebit/rcwa/_profile.py`, `src/wavebit/rcwa/problem.py`,
`src/wavebit/rcwa/numerics.py`, `src/wavebit/materials/models.py`, `tests/rcwa/test_profiled_layer.py`,
`tests/rcwa/test_graded_materials.py`, `tests/materials/test_bruggeman.py`,
`tests/references/test_reticolo_taper_golden.py`. Graded materials and the Bruggeman medium added 2026-11-24.

**TL;DR.** A [`ProfiledLayer`](../../guide/rcwa/profiled-layers.md) is solved as a staircase of
ordinary layers. On the same staircase it agrees with RETICOLO V10 to 3.5e-14; its automatic
count lands within 7.3e-5 of RETICOLO's extrapolated profile against a tolerance of 1e-4; the
staircase converges at second order (2.000 on a straight taper, 1.89 to 2.07 on a rounded one
under arc-length slicing, against 1.35 to 1.48 uniformly); automatic derivatives match central
differences to 4e-9; and a vertical wall is the Rectangle layer to the bit. Materials that
change with height reach their closed forms: a linearly graded film is 3.3e-7 from the exact Airy
solution at 512 slices, converging at order 2.0001, and a Bruggeman roughness layer is the
thin-film formula to 3e-16. The criteria were
fixed before the measurements; none was loosened.

Measured on one Linux CPU (x86-64), Python 3.13.17, NumPy 2.5.5, JAX 0.12.0.1, PyTorch
2.15.0+cpu, Octave 8.4.0, RETICOLO V10 (`V10_2025`).

## Pre-registered criteria and results

| | Criterion | Result | Held by |
| --- | --- | --- | --- |
| (a) | a vertical, unrounded `Trapezoid` is the `Rectangle(size=(cd, Ly))` layer, and a `Loft` with equal outlines the `Polygon` layer, bit for bit; with a derivative-carrying angle at `pi/2`, within 1e-12 | bit for bit (every `cd_at`, an off-centre line, Li and default NVF, a scalar and a pair angle); traced: below 1e-12 | `test_profiled_layer.py` |
| (b) | observed order in [1.8, 2.2] on a straight taper, TE and TM; at least 1.8 on a rounded profile under arc-length slicing | 2.000 TE, 1.999 TM; rounded 1.89 to 2.07 at the finest levels | `test_profiled_layer.py` (rounded: slow tier) |
| (c) | automatic derivative against a central difference on the same preparation, within 1e-6 relative | `cd` 2.4e-10, `swa` 8.7e-10, `top_rounding` 4.4e-10, `footing` 3.8e-9, `thickness` 2.5e-10 (PyTorch); JAX `swa` and `thickness` within the bound. Roughness thickness 3.7e-10, volume fraction 2.1e-9, interface thickness 6.4e-10 (PyTorch); JAX fraction within the bound | `test_profiled_layer.py`, `test_graded_materials.py` (JAX: slow tier) |
| (d) | the same uniform staircase against RETICOLO within 1e-10; the automatic count within its tolerance of RETICOLO's extrapolated limit | 3.5e-14 worst over 16 cases; 2.9e-5 to 7.3e-5 against a tolerance of 1e-4 | `test_reticolo_taper_golden.py` |
| (e) | graded and effective-medium films against closed forms in the fine-slice limit, within 1e-6 | linear grading against the Airy solution 3.3e-7 (normal) and 3.8e-7 (30 deg) at 512 slices; roughness layer against the thin-film formula 3e-16; graded interface against 20 000 sublayers 5.7e-8 | `test_graded_materials.py` |
| (f) | NumPy, PyTorch and JAX within 1e-12 | within 1e-12 | `test_profiled_layer.py` (JAX: slow tier) |
| (g) | a raster layer beside a `ProfiledLayer` refused with `layer.profile_raster`, at `run` and at `check` | refused, with the raster layer's path | `test_profiled_layer.py` |

## Convergence of the staircase

Observed order `log2(|R_N - R_N/2| / |R_2N - R_N|)` (maximum over every reflected and
transmitted efficiency) between successive doublings, on the two lines of
`tests/references/_reticolo_taper_cases.py`: silicon-like `eps = 15.2 + 0.16j` on fused silica,
period 0.5 um, wavelength 0.6 um, 31 orders, Li's factorization. `taper` is a straight 80-degree
taper; `rounded` has walls of 82 and 86 degrees, a 0.03 um top rounding and a 0.02 um footing.
Columns are the three-level windows ending at N = 128, 256 and 512.

| profile | rule | light | 128 | 256 | 512 |
| --- | --- | --- | --- | --- | --- |
| taper | uniform or arc length | TE, 0 deg | 1.996 | 1.999 | 2.000 |
| taper | uniform or arc length | TM, 0 deg | 1.989 | 1.997 | 1.999 |
| rounded | uniform | TE, 0 deg | 1.791 | 1.590 | 1.479 |
| rounded | uniform | TE, 20 deg | 1.453 | 1.365 | 1.389 |
| rounded | uniform | TM, 0 deg | 1.191 | 1.284 | 1.354 |
| rounded | uniform | TM, 20 deg | 1.256 | 1.351 | 1.408 |
| rounded | arc length | TE, 0 deg | 1.900 | 2.115 | 2.066 |
| rounded | arc length | TE, 20 deg | 1.787 | 1.852 | 1.995 |
| rounded | arc length | TM, 0 deg | 1.629 | 1.826 | 1.890 |
| rounded | arc length | TM, 20 deg | 1.724 | 1.864 | 1.925 |

A straight taper's arc length is proportional to its height, so the two rules cut it the same.
A circular arc meets the horizontal top with a width that varies as the square root of the
distance from it, which is where uniform slicing loses its order: the observed values drift
toward 1.4, consistent with an order of 1.5 in the limit. Equal arc-length steps put the slices
where the outline turns, and the order returns toward 2. Popov, Nevière, Gralak and Tayeb
(J. Opt. Soc. Am. A **19**, 33 (2002)) describe how a staircase's convergence depends on the
profile; these measurements are this solver's, on these two dielectric lines.

The Richardson estimate the automatic count stops on tracks the actual error: against
RETICOLO's extrapolated limit, the solver's answer at its chosen count was 3.79e-5 from it where
it estimated 3.83e-5 (taper, TE, normal incidence), and the eight cases' estimates and actual
distances agree to within 15%.

## The external reference

`tests/references/gen_goldens_reticolo_taper.py` builds each slice from its own geometry
(`_reticolo_taper_cases.py`: a rounding's centre is the intersection of the two lines offset by
its radius, not the solver's tangent-length formula) and hands RETICOLO an exact 1D texture per
slice, with RETICOLO's `+z` out of the stack. The generator ran with the product's import
blocked. Its geometry and the solver's slices agree to 5.6e-17 over 2001 heights per profile.
RETICOLO's 64, 128 and 256 uniform slices are extrapolated with the observed order, capped at
2. Values and provenance: `tests/references/golden_reticolo_taper.npz` and its
`.reference.json`.

## Materials that change with height

Three closed forms, none of which calls the solver (`tests/rcwa/test_graded_materials.py`):

- A film whose permittivity rises linearly, `eps = 2.25 -> 4.0 + 0.1j` over 0.3 um on fused silica
  under air, TE at 0.6 um. With a linear `eps(z)` the field equation
  `E'' + k0^2 (eps(z) - sin^2(theta) eps_c) E = 0` is Airy's, so the reflectance is exact. The
  solver's uniform slices at 128, 256 and 512 are 5.3e-6, 1.3e-6 and 3.3e-7 from it at normal
  incidence (3.8e-7 at 30 degrees), an observed order of 2.0001; their Richardson extrapolation
  is 6e-12 from it.
- A Bruggeman roughness layer, 4 nm of silicon-like `eps = 15.2 + 0.16j` and void at fractions 0.3
  and 0.5 on 80 nm of the same material: the characteristic-matrix thin-film formula (Born and
  Wolf, sec. 1.6) with the effective permittivity found independently by `numpy.roots`, 3e-16.
- A 10 nm interface layer mixing silicon into fused silica linearly
  (`rcwa.Graded.interface`), at 20 degrees: the characteristic matrix of 20 000 sublayers, each
  with its own independent Bruggeman root, 5.7e-8 at 512 slices.

The Bruggeman model itself (`tests/materials/test_bruggeman.py`) solves its mixing equation to
1e-14, matches `numpy.roots` to 1e-13 of its value, is symmetric under exchanging the phases
(`f <-> 1 - f`), returns its constituents at `f = 0` and `f = 1`, evaluates a Drude constituent at
each wavelength, and its fraction derivative on PyTorch and JAX matches a central difference to
1e-8.

## Cost

Per slice, one wavelength, 31 orders, one CPU core: about 10 ms under `factorization='li'`, and
about 1.1 s under the default NVF, almost all of it the shape normal field
(`parametric._field_tiled`). The two agree to 1e-14 on a line's slices. On JAX, eager execution
costs about 4 s per layer and `jax.jit` compiles in time growing with the slice count; these are
the same for hand-written layers (measured: 16.0 s against 16.0 s for four layers), so they are
not the slicing's. The eigenproblems of the slices are solved one layer at a time, as for every
stack; batching them is a separate change.

## Not checked here

- Lines at oblique azimuth (conical incidence) and two-dimensional `Loft` pillars against an
  external solver: the `Loft` and `Profile` slices are `Polygon`s, whose own external checks are
  in [boundary modeling](../../guide/rcwa/boundary-modeling.md#polygon-regions).
- Metals, where a staircase can converge slowly or not at all in TM (Popov et al.).
- GPU time and memory.

# Fitting planar dispersive layer boundaries

Written: 2026-09-08

Targets:
- `src/wavebit/fdtd/_periodic/scene.py`
- `studies/fdtd/layer_alignment/fdtd_layer_alignment.py`
- `tests/fdtd/test_fdtd_layer_alignment.py`

TL;DR: `align_layers=True` fits every layer boundary to a z node without rounding
the physical thicknesses. It reduces a large coarse-grid TM error in a capped
metal film and gives consistent TE/TM refinement. It is a static mesh option,
not a general dispersive interface model or a guarantee for metal corners.

## Conditions and reproduction

```sh
PYTHONPATH=. python studies/fdtd/layer_alignment/fdtd_layer_alignment.py
```

The default measurement uses JAX 0.11.0, x64 enabled, with NumPy 2.5.1 on an
arm64 CPU. `--backend numpy` uses the same equations without JAX. The structure
has a dielectric cap (epsilon 2.25, thickness 0.113) followed by the synthetic
mixed metal (epsilon-infinity 1.5, thickness 0.213). Its angular-frequency poles
are (omega0, gamma, strength) = (0, 0.08, 160), (10, 0.12, 40), as in the
[mixed-metal study](fdtd-mixed-metal.md). Both half spaces are vacuum; period is
0.05, wavelengths are 0.9/1/1.1, theta is 0.45 radians at center wavelength 1,
and transverse k stays fixed across the band. There are no lateral interfaces.

Unless a control says otherwise, runs use 160 optical periods, Courant fraction
0.8, pulse width 3, padding scale 1, and `round(0.3 * resolution)` PML cells per
end. Each scene evaluates both actual/reference current bases. Diagnostic
windows are the last 200 time steps, so their physical duration varies with dt.
The supplied illumination is TE; `amplitudes_settled` describes this illumination,
whereas `state_decayed` checks both bases and their reference runs.

The reference is an independent tangential E/H characteristic-matrix calculation
with the full complex dispersive epsilon. Each layer propagates its bottom-face
fields to its top face using cos(kz*d) and sin(kz*d), with admittance q for TE and
epsilon/q for TM. Reflection converts to the incidence-relative p basis. Tests
cross-check the reference against a one-layer Fresnel/Airy formula and Wavebit
RCWA at all three wavelengths. Reported errors are maximum absolute complex r/t
errors over the band, separately for TE and TM, not power errors.

## Boundary fitting versus padding grading

The [API guide](../../guide/fdtd/time-domain.md#aligning-every-layer-boundary) describes the
new option. Every layer gets `ceil(thickness * resolution)` equal cells. Uniform
padding retains width `1/resolution`. Optional grading coarsens the padding from
the cell width of its adjacent layer. The old `grading=True` alone still uses a
uniform core spanning the total thickness, and can cut an internal layer boundary.

| Resolution | Align layers | Grade padding | z cells | dt | TE error | TM error |
| --- | --- | --- | --- | --- | --- | --- |
| 40 | False | False | 202 | 0.0115470 | 0.0204954 | 0.252555 |
| 40 | False | True | 124 | 0.0112640 | 0.0100997 | 0.0275787 |
| 40 | True | False | 202 | 0.0111392 | 0.0135473 | 0.0124269 |
| 40 | True | True | 124 | 0.0111392 | 0.0116673 | 0.0108408 |
| 80 | False | False | 399 | 0.00577350 | 0.00380414 | 0.00263503 |
| 80 | False | True | 225 | 0.00570563 | 0.00197114 | 0.111926 |
| 80 | True | False | 400 | 0.00556961 | 0.00340318 | 0.00311745 |
| 80 | True | True | 235 | 0.00556961 | 0.00263229 | 0.00239406 |
| 160 | True | False | 794 | 0.00283781 | 0.000904545 | 0.000830066 |

With aligned layers and uniform padding, refinement 40 to 80 reduces both errors
by about four times; 80 to 160 reduces them by another factor of about 3.76.
The aligned coarse mesh uses the same cell count as the
ordinary coarse mesh, but a 3.5% smaller dt. At resolution 80 it uses one more
z cell. These are geometry and step-count measurements, not timing benchmarks.

At resolution 40, the unaligned uniform mesh misses the internal interface by
0.012; at resolution 80 it misses it by only 0.0005. This accidental near-alignment
explains why the unaligned fine-grid TM result can be slightly better than the
fitted result. It does not establish a convergence order for unaligned meshes.
With old grading at resolution 80, the internal face misses the nearest node by
0.00433333 and TM error grows to 0.111926. The TE amplitude flag passes for this
run, while material decay in the TM basis fails at 160 periods. A TE flag cannot
certify the second source basis.

Aligned runs preserve every layer face within floating-point roundoff. Ez lies
between the boundary nodes, so its control volume does not contain two different
layers. Tangential material values still use arithmetic overlap on the actual
dual cells. This avoids the existing mismatch between harmonic averaging of
epsilon-infinity and arithmetic weighting of normal oscillator strengths at
these z interfaces. No pole formula or averaging model was changed.

Uniform padding has physical PML width 0.3 at both ends. Graded upper PML widths
are 0.547866/0.544736 for aligned resolutions 40/80; they are not equal-thickness
absorber comparisons. Padding grading introduces its own mesh reflections, and
changing the fitted core also changes dt. The guide explicitly leaves these as
separate refinement controls. The PML diagnostic now includes the final closing
cell in its outer bound; the prior bound stopped one cell early in the plot,
although the solver already absorbed through all cells.

## Refinement, bottom illumination and controls

At resolution 80, reversing illumination through the same physical stack gives
TE error 0.00168585 and TM error 0.00197925 against the reversed-layer reference.
Both state and supplied-amplitude criteria pass. The NumPy regression also checks
bottom refinement from 40 to 80, including both polarization bases.

The following controls keep the aligned resolution-80 structure and uniform
padding mesh. Changes are maximum complex r/t differences from the 160-period
baseline, separately for each basis.

| Control | TE change | TM change |
| --- | --- | --- |
| Duration 160 to 320 | 8.75532e-14 | 1.14529e-10 |
| Courant fraction 0.8 to 0.4 | 0.000281009 | 0.000277808 |
| Physical PML width 0.3 to 0.6 | 5.99122e-7 | 4.58294e-7 |
| Padding scale 1 to 1.5 | 7.12851e-5 | 7.13262e-5 |

These runs pass the state and TE amplitude criteria. Doubling duration leaves
the answer effectively unchanged; halving dt changes it less than spatial
refinement does, and doubling PML thickness has a still smaller effect. The
observed refinement is consistent with discretization error in this planar
case, rather than a late pulse or inadequate PML dominating the result. This is
a measured case, not a general error estimator or universal convergence order.

## Scope and verification

Regression checks include top/bottom TE/TM refinement at 40 and 80 against the
characteristic matrix, exact physical layer placement, thin-layer time-step
reduction, point and subpixel sampling, original-coordinate cropped records,
typed conical magnetic layers against RCWA, and JAX/Torch pole derivatives.
Derivative objectives combine a complex TM
transmission component, a user frequency-plane flux and a cropped time record,
with bottom illumination, DFT subsampling and both padding policies. NumPy
central finite differences supply an independent derivative check.
The change passes 126 distinct targeted checks: 21 layer-alignment checks,
37 documentation-contract checks and 68 existing nonuniform-grid, graded/bottom
scene, cropped-record, pulse-width and plotting checks. Ruff and a Sphinx build
with warnings treated as errors also pass. This is not a full-suite total.

Meep's public [subpixel documentation](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/#what-about-dispersive-materials)
explains why averaging the full dispersive response requires a more complicated
frequency-dependent effective material. It supplies context for the limitation,
not an implementation for this change. No external implementation was copied;
no Meep implementation was read, and no fresh Meep/FDTDX or GPU execution is
part of this measurement. Existing stored Meep scene references remain regression
checks for the pre-existing graded and bottom paths.

This option does not align lateral raster boundaries, handle sharp metal corners,
or validate epsilon-near-zero materials. Geometry remains static and cannot be
differentiated. Very thin layers may impose a costly global CFL time step;
preserving a thin layer with one cell is not evidence that its internal field is
resolved. Higher resolution, runtime and absorber controls remain necessary for
new structures. The default mesh and material solver remain unchanged.

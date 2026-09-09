# Rotated rectangle area integration

Written: 2026-09-09

Targets:
- `wavebit/_core/scene_fdtd.py`
- `wavebit/_core/polygon.py`
- `wavebit/main.py`
- `wavebit/manifest.py`
- `tests/test_fdtd_rotated_rectangles.py`
- `studies/fdtd_rotation_gradients.py`
- `tests/references/interface/meep_rotation.py`
- `tests/references/test_meep_rotation.py`

TL;DR: opt-in `shape_integration='area'` integrates a rotated rectangle directly
against each fixed Yee cell, retaining position, size, rotation and material
derivatives. It uses RCWA's shared shape reader and rectangle vertices. The
initial scope is one shape per layer, with periodic wrapping and projected
extents within the periods. Transverse material averaging is arithmetic;
exact geometry does not make this a full effective-tensor interface treatment.

## Why a separate integration option

The default separable method resolves axis-aligned painting into a rectilinear
partition. Its normal harmonic and tangential arithmetic averages depend on
that partition's alignment. A rotated face needs a different integration
method and, for second-order interface accuracy, a more general material tensor.

Area integration provides exact geometric cell fractions without pretending
that a scalar average supplies the missing tensor. It computes the arithmetic
transverse average `background + fraction*(material-background)` separately
at each component's staggered position. Existing arithmetic/harmonic z sampling
then integrates the layer profile. Both epsilon and mu follow this rule. No
off-diagonal effective material coefficient is introduced. The option is
explicit because its interface approximation differs from the default method,
including when the supplied angle is zero.

The [Meep smoothing theory](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/)
explains why an arbitrarily oriented interface generally requires an anisotropic
effective tensor for second-order accuracy. This feature claims exact cell
areas and derivatives of the implemented discretization, not that convergence
rate. The [usage contract](guide/time-domain.md#rotated-rectangle-area-integration)
states the supported geometry and provides a runnable example.

## Live polygon clipping

The shared `shape_geometry` function resolves cell-centred positions and
absolute lengths, including square and `l` aliases. Shared `rect_verts` constructs
four counter-clockwise vertices with rotation measured in radians. Geometry
stays in its backend graph. Each centre folds into the periodic cell before a
3 by 3 image sum. Requiring both projected extents to fit the periods prevents
self-overlapping periodic images and makes that image sum sufficient.

Four half-plane clips intersect each rectangle with the cell. A clipped convex
quadrilateral can have at most eight vertices. Fixed-capacity arrays and stable
packing preserve cyclic order through a trace. The final area uses the signed
shoelace formula in cell-relative coordinates to reduce cancellation. Parallel
inactive edges use safe denominators, avoiding undefined intermediate gradients.
Only out-of-range floating-point fractions are clipped; zero-area intervals
are not assigned a custom derivative. No frozen topology plan, approximate
polygonization or intermediate raster is used.

The actual clipping combinatorics can change. At generic vertex/cell contacts,
the area and its first derivative have their geometric limits. Coincident
parallel edges can instead have different one-sided slopes. There is no claim
of globally C1 sharp geometry or general higher derivatives. The measurements
on this page use zero smoothing. Positive `shape_smoothing` now uses
[exact translated-polygon moments](fdtd-rotation-smoothing.md); its separate
measurements distinguish the resulting graded profile from this sharp target.

An independent oracle integrates vertical slice lengths, splitting at all
vertex x coordinates and intersections with horizontal cell faces. Midpoint
quadrature is exact on each resulting linear interval. It does not clip
polygons or use shoelace area. NumPy, JAX and Torch agree with this oracle for
rotated, axis-aligned and quarter-turn rectangles, including periodic seams
and both unstaggered/half-staggered cells. Position, size and rotation derivatives
agree with two independent finite-difference steps. Total area has exact
parameter derivatives `[0,0,height,width,0]` regardless of angle.

Eight joint variables—two positions, two sizes, rotation, epsilon, mu and
fixed-total internal layer thickness—also pass complete bottom-incidence
conical Jones solves against two finite-difference steps. The mesh, source,
PML, timestep and duration stay fixed. A single JAX trace supports rotation
through zero and periodic translation. Invalid traced geometry reports
`stable=False` and NaN in both real and imaginary scattering components;
the latter avoids a misleading finite zero when an objective selects only the
imaginary component of an invalid result.

## Independent Fourier reference

The test structure has one width-0.3, height-0.18 rectangle at angle 0.37,
centre `(0.071,-0.047)`, epsilon 2.25 and mu 1. Its layer thickness is 0.2,
periods are `(0.6,0.6)`, top epsilon is 1 and substrate epsilon 1.21. Normal
incidence has wavelength 1. Only the central order propagates. All eight
complex r/t channels of the two incident bases are compared, including cross
polarization.

RCWA uses the analytic rectangle transform, orders ±8 on both axes and a
256-point normal-field grid. Reference controls are part of the measurement:

| Reference change | Complex amplitude | Angle derivative | Width derivative |
| --- | ---: | ---: | ---: |
| Orders ±8 to ±10 | 0.000174623 | 0.000286166 | 0.00148383 |
| Normal-field grid 256 to 512 | 5.77e-7 | 0.000702191 | 0.00138708 |
| Difference step 1e-5 to 5e-6 | — | 2.31e-8 | 1.66e-7 |

These are discrete-reference controls, not rigorous continuum error bounds.
In particular, the normal-field sampling change is significant compared with
the final angle-derivative difference. FDTD uses physical PML thickness 1,
pulse width 2.5, duration 60, fitted z layers and directional extraction.
Maximum absolute differences from the stated RCWA reference are:

| Grid resolution | Complex amplitude | Angle derivative | Width derivative | Maximum energy defect |
| --- | ---: | ---: | ---: | ---: |
| 16 | 0.0131202 | 0.00218485 | 0.0295225 | 0.00387795 |
| 24 | 0.00901555 | 0.00152083 | 0.0194761 | 0.00180671 |
| 32 | 0.00587439 | 0.00110452 | 0.00949567 | 0.000993532 |
| 40 | 0.00494553 | 0.00125154 | 0.0152040 | 0.000651761 |

Angle derivatives are per radian and width derivatives per physical length.
The amplitude differences decrease, but the width derivative improves through
32 and worsens at 40. The angle derivative also changes nonmonotonically at
these last two grids. No monotonic derivative convergence, convergence order
or arbitrary-material accuracy guarantee follows. This is a limitation of
using exact geometry with an approximate interface discretization, not a
reason to discard the less favorable grid.

Additional orders-±10 references with normal grids 1024 and 2048 differ by
3.10e-8 in amplitude, 9.36e-5 in angle derivative and 0.000120461 in width
derivative. Against the latter reference, grid-40 FDTD differences are
0.00502439, 0.000700873 and 0.0148728. The normal-field refinement changes the
interpretation of the small angle difference, while the width-gradient error
remains substantial. Extending the grid-40 FDTD duration from 60 to 80 changes
amplitudes by 1.24e-11 and derivatives by 1.85e-10. Longer time integration does
not resolve this interface error.

## Independent three-dimensional Meep measurements

Meep 1.34.0 uses a documented 3D rotated Block with the same shape, periods,
materials and depth, in an eight-unit z cell. The substrate ends at the cell
boundary. A one-unit z PML, Courant 0.4, source plane -2, and DFT planes -1/1.2
are fixed. Full transverse-plane quadrature uses Meep's returned weights, with
sum checked against the period area. Two current components excite the two
incident bases. Incident/outgoing s/p signs and discrete Yee phases reference
reflection at 0 and transmission at 0.2.

The Gaussian-windowed source has width 2.5, peak 12.5 and end 25. Both resolutions
20 and 40 contain integer transverse period and layer cell counts. Independent
runs use durations 80/100, perturbation steps 1e-4/5e-5, and Meep subpixel
tolerance 1e-8. Both archives preserve all eight complex channels, both design
derivatives and the complete duration/step combinations.

| Meep control or difference from orders-±8 RCWA | Grid 20 | Grid 40 |
| --- | ---: | ---: |
| Duration change, amplitude | 6.26e-12 | 1.14e-11 |
| Duration change, derivative | 5.91e-11 | 1.37e-10 |
| Perturbation-size change, derivative | 2.04e-6 | 4.01e-5 |
| Amplitude difference | 0.0102385 | 0.00296640 |
| Angle derivative difference | 0.00291452 | 0.00221176 |
| Width derivative difference | 0.0497478 | 0.0361495 |

Meep's differences from this Fourier reference decrease between these two
resolutions, while its width derivative remains much less accurate than the
amplitude agreement alone suggests. At grid 40, Wavebit and Meep differ by
0.00296135 in amplitude, 0.00267124 in angle derivative and 0.0312323 in width
derivative. The last discrepancy is substantial. Both local derivatives are
repeatable under duration and perturbation changes, but that repeatability
does not establish continuum gradient accuracy. The solvers also use different
interface material averages: Meep's rotated effective tensor and this route's
arithmetic transverse average. No general solver equivalence is claimed.

The regression test checks independently measured reference controls, the
20-to-40 Meep refinement, and Wavebit grids 16/32 against both Fourier and Meep
results. These two Wavebit regression grids bound test cost; the study above
retains the additional 24/40 results and their nonmonotonic changes.

## Reproduce

Measured on an arm64 CPU with Python 3.12.9, NumPy 2.5.1, JAX 0.11.0 x64 and
Torch 2.13.0 float64. No GPU execution is established. From the repository root:

```shell
python -m studies.fdtd_rotation_gradients --output /tmp/rotation-gradients.json
pytest -q tests/test_fdtd_rotated_rectangles.py tests/references/test_meep_rotation.py
```

Generate an independent candidate in a separate Meep installation:

```shell
python tests/references/interface/meep_rotation.py --output /tmp/meep-rotation.npz
python tests/references/interface/meep_rotation.py --resolution 20 --output /tmp/meep-rotation-coarse.npz
```

The wrapper uses only documented public simulation, rotated Block, source and
DFT APIs. No Meep implementation or excluded Meep-derived FDTDX source was
opened or copied.

# Transverse rectangle derivatives

Written: 2026-09-09

Targets:
- `src/wavebit/_core/scene_fdtd.py`
- `src/wavebit/_core/parametric.py`
- `src/wavebit/manifest.py`
- `tests/test_fdtd_rectangles.py`
- `tests/test_manifest.py`
- `studies/fdtd_rectangle_gradients.py`
- `tests/references/interface/meep_rectangle.py`
- `tests/references/test_meep_rectangle.py`

TL;DR: the explicit FDTD route now reads axis-aligned rectangle and square
recipes through RCWA's existing shape convention. Centres, sizes and scalar
epsilon/mu retain NumPy/JAX/Torch graphs through periodic painting and cell
integration. Optional transverse translation averaging smooths cell crossings.
In the measured grating, it reduced width/position derivative errors against
independent Fourier optics. Actual paint contacts can still have physical
kinks, and sharp finite-grid derivatives in both FDTD solvers remain much
less accurate than their scattering amplitudes.

## One geometry convention

The shared `shape_geometry` reader owns shape types, absolute lengths,
centre-relative coordinates, and the `l` shorthand. Common key validators
reject unknown or unused fields. The FDTD adapter accepts rectangles and
squares with static angle zero; it refuses curved/rotated shapes, raster
dispersion labels on recipes and separate raster mu. Magnetic recipe materials
use the existing `Material(epsilon, mu=...)` carrier. The
[usage contract](guide/time-domain.md#transverse-rectangle-shape-derivatives)
includes a runnable JAX example and every supported constraint.

Every periodic face is folded into the rectangular cell and sorted. The
resulting material partition has a fixed number of entries; its boundaries
move, while the Yee mesh does not. Per-shape face events define interval
membership. Applying shapes in input order implements painting rather than
adding overlapping material contrasts. No auxiliary fixed-resolution image
is inserted between the physical recipe and Yee-cell integration.

Zero-width partition intervals matter to differentiation. Two disjoint shapes
can share a projected x or y boundary, or shapes in different layers can do so,
without making physical contact. Classifying their zero-width intervals by a
midpoint would choose an arbitrary material and lose derivatives. Ordered face
events keep a consistent analytic extension through those intervals. The
overlap weights preserve their zero-branch derivative; `max(weight,0)` would
split it at zero on JAX/Torch. Only strictly negative roundoff is clipped.
Independent cell finite differences and exact area derivatives check this case.

Static full-height rectangles use the same one-row y-invariant shortcut as a
raster grating. A live height keeps full y sampling even when its current value
equals the period. Shape count/type, paint order, periods, wavelength, source,
PML and time step are static. Live internal z thicknesses can be combined using
the existing fixed-total `mesh_thickness` contract. No period, lattice or
total-depth derivative is introduced here.

## Cell integration and regularization

Let F(b) be the fraction of a fixed cell [L,R] to the right of face b.
For a sharp interface, F(b)=clip((R-b)/(R-L),0,1). A material interval [a,b]
contributes F(a)-F(b); three periodic images cover every sampled cell.
Tangential coefficients use arithmetic averages and normal coefficients use
harmonic averages, in the same order as the existing raster implementation.
This remains a separable diagonal approximation at corners.

For averaging width h>0, average the complete painted layer over a common
translation uniformly distributed on [-h/2,h/2]. The cell fraction becomes

```text
G_h(x) = 0                       x <= -h/2
         (x+h/2)^2 / (2h)        -h/2 < x < h/2
         x                       x >= h/2

F_h(b) = [G_h(R-b) - G_h(L-b)] / (R-L).
```

Use an independent common translation on each transverse axis. The public
`shape_smoothing` scalar or `(x,y)` pair specifies the full physical widths.
It defaults to zero and each width must lie in [0,period]. A width in physical
units changes the sampled material profile; shrink it with mesh refinement
when checking the intended sharp geometry. It is not a promise of a generally
isotropic graded material for all polarizations.

The geometry oracle independently integrates translated sharp intervals by
splitting the translation integral at every cell/face intersection. Midpoints
integrate each resulting linear piece exactly. All three backends agree on
unequal and repeated boundaries, periodic seams, zero-area intervals and a
full-period averaging width. Area, material painting and periodic translation
provide separate invariants. Aligned magnetic recipes also match exactly
equivalent rasters through complete conical Jones solves from both sides.

At a cell crossing, the sharp derivative has a persistent jump. For a cell of
width 0.1 and h=0.08, the derivative difference at offsets ±delta is exactly
2 delta/(0.08*0.1), and both AD backends reproduce this continuous limit.
Conversely, when a later shape starts covering an earlier shape, the actual
painted area can have distinct one-sided derivatives. The test's explicit
contact has slopes 0 and -0.75. Averaging translations does not remove this
physical topology change. No unique derivative at arbitrary contacts, size
bounds, or general higher-order geometry derivatives is claimed.

## Fourier reference and grid refinement

The measured bar has width 0.43, thickness 0.2, epsilon 2.25 and mu=1, in a
layer background of 1. Top epsilon is 1, substrate epsilon 1.21, periods are
[1.3,0.05] and wavelength is 1. Centres are 0.071, 0.083 and 0.095 relative to
the cell centre. Only orders -1,0,+1 radiate. Both TE and TM complex r/t are
checked, not just their total power.

RCWA uses the analytic rectangle transform and orders ±32; ±48 checks its own
convergence. The maximum complex-amplitude change is 1.07e-5 and the maximum
width-derivative change is 0.000321118. Two width perturbation sizes, 1e-5 and
5e-6, differ by at most 8.63e-9. Position derivatives have a stronger independent
reference: translating the structure by c multiplies diffraction order m by
exp(-2 pi i m c/P), so its derivative is (-2 pi i m/P) times the amplitude.
RCWA's own finite differences also satisfy that phase law.

FDTD uses a fixed grid, fitted z layers, one physical unit of PML, pulse width
3, duration 65 and directional extraction. The common backend uses float64/
complex128. The table reports maximum absolute errors across all three
positions, the four complex amplitude channels and all three orders. Size
and position derivatives are per unit physical length.

| Grid | Averaging width | Amplitude error | Width derivative error | Position derivative error |
| --- | --- | ---: | ---: | ---: |
| 20 | 0 | 0.00886885 | 0.0944707 | 0.171510 |
| 20 | 1 cell | 0.00892673 | 0.0179284 | 0.0361959 |
| 30 | 0 | 0.00375536 | 0.0455949 | 0.130404 |
| 30 | 1 cell | 0.00377353 | 0.0102142 | 0.0137596 |
| 40 | 0 | 0.00204149 | 0.0423246 | 0.110497 |
| 40 | 1 cell | 0.00205371 | 0.00582625 | 0.0128695 |

One cell of translation averaging improves these derivative errors substantially,
while changing the amplitude error only slightly. This is a finite set of
positive-isotropic grating measurements, not a general accuracy guarantee for
corners, touching shapes, arbitrary materials or an optimized design. Separate
JAX/Torch tests jointly vary seven shape/material/z-thickness parameters under
bottom, conical, complex Jones illumination. Two finite-difference steps agree
with AD while mesh coordinates, source/PML placement and timing remain fixed.

## Independent Meep measurements

The generator uses only Meep 1.34.0's documented public API. Its sharp geometry
uses 2D Blocks, with Ez corresponding to Wavebit's TE electric field, periodic
x and y-directed PML. The substrate ends exactly at the computational cell
boundary. The length-10 cell, source at -3, monitor planes -1 and 1.4, Courant
0.4 and one-unit PML are explicit. Quadrature uses the DFT monitor's returned
coordinates and weights. Discrete Yee propagation references r at z=0 and t
at z=0.2. The output includes the outgoing s-frame signs and the conversion
from Meep's cell-centred Fourier origin to Wavebit's corner origin.

The final Gaussian-windowed source has width 3, peak time 15 and end time 30.
Durations 100 and 140 and centered steps 1e-4 and 5e-5 control reference error.
The frozen sharp archives retain both grids, durations and steps.

| Sharp reference check | Grid 40 | Grid 80 |
| --- | ---: | ---: |
| Runtime change, complex amplitudes | 2.81e-6 | 2.67e-6 |
| Runtime change, derivatives | 1.24e-5 | 1.27e-5 |
| Perturbation-size change, derivatives | 2.71e-8 | 2.89e-8 |
| Amplitude error vs RCWA | 0.00319276 | 0.000444645 |
| Width derivative error vs RCWA | 0.0239755 | 0.0192671 |
| Position derivative error vs translation law | 0.0895620 | 0.0599772 |

At grid 40, sharp Wavebit and Meep differ by at most 0.00196359 in complex
amplitude and 0.0883470 in its two-variable Jacobian. The latter is a substantial
finite-grid discrepancy, despite each local derivative being repeatable.
Agreement in scattered amplitude does not establish an accurate shape gradient.
The independent Fourier reference exposes the error in both FDTD methods.

An earlier reference probe with width-1.2 illumination and durations 40/60
changed amplitudes by 0.00113 and derivatives by 0.00539; it was not used as a
settled oracle. At grid 80, steps 1e-3/5e-4 crossed a cell boundary and changed
the sharp derivative by 0.17866. The smaller final steps measure the local
derivative and do not remove its continuum error. These controls are separate
from deciding whether a design gradient is physically accurate.

For the averaged TE case at grid 40, a global periodic material function
implements the same linear transverse epsilon profile with width 0.025.
Meep's documented `do_averaging` option then performs its cell integration.
The comparison gives amplitude and derivative gaps 0.00116054 and 0.00153202,
respectively. Tightening Meep's averaging tolerance from 1e-8 to 1e-9 changes
amplitudes by 1.24e-11 and derivatives by 4.34e-7. Runtime changes are at most
2.81e-6 and 1.36e-5; changing the perturbation size changes derivatives by at
most 6.94e-6. Both averaging tolerances are preserved in separate archives.
This comparison is restricted to TE, whose field is
tangential to both interfaces; it does not assert isotropic-profile equivalence
for the normal TM coefficient.

The [Meep smoothing theory](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/)
explains the tangential/normal averaging distinction and corner limitations.
The [public API](https://meep.readthedocs.io/en/master/Python_User_Interface/)
documents Blocks, material functions and DFT quadrature. No Meep implementation
or previously excluded Meep-derived FDTDX projection file was opened or copied.

## Reproduce

Measured on an arm64 CPU with Python 3.12.9, NumPy 2.5.1, JAX 0.11.0 x64 and
Torch 2.13.0 float64. No GPU execution or arbitrary-shape compatibility is
established by this experiment. Run from the repository root:

```shell
python -m studies.fdtd_rectangle_gradients --output artifacts/rectangle-gradients.json
pytest -q tests/test_fdtd_rectangles.py tests/references/test_meep_rectangle.py
```

In a separate Meep installation, generate candidates outside the repository:

```shell
python tests/references/interface/meep_rectangle.py --output artifacts/meep-rectangle.npz
python tests/references/interface/meep_rectangle.py --resolution 80 --output artifacts/meep-rectangle-fine.npz
python tests/references/interface/meep_rectangle.py --smoothing .025 --output artifacts/meep-rectangle-smoothed.npz
python tests/references/interface/meep_rectangle.py --smoothing .025 --subpixel-tol 1e-9 --output artifacts/meep-rectangle-smoothed-refined.npz
```

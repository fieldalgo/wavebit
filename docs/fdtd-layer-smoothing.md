# Continuous internal-interface sampling

Written: 2026-09-09

Targets:
- `src/wavebit/_core/scene_fdtd.py`
- `tests/test_fdtd_layer_smoothing.py`
- `tests/test_fdtd_layer_gradients.py`
- `studies/fdtd_layer_smoothing.py`
- `tests/references/test_meep_layer_smoothing.py`
- `tests/references/interface/meep_halfspace.py`
- `examples/fdtd_layer_optimization.py`

TL;DR: positive `layer_smoothing` averages internal interface positions over
a static physical width before Yee-cell sampling. The resulting fractions
have continuous first derivatives. One cell of averaging reduced the sampled
sharp-film derivative error in this experiment, but a fixed width also changes
the continuum material profile. The default remains zero. Refinement and an
independent sharp-interface check are still necessary for an optimized design.

## Geometry and derivative contract

This extends [fixed-total layer derivatives](fdtd-layer-gradients.md), using
the same `Stack` and `thickness` spelling as RCWA. Static `mesh_thickness`
chooses all grid geometry; actual lengths preserve its total D. The source,
PML, time step, record coordinates and both outer reference planes stay fixed.
See the [usage contract](guide/time-domain.md#continuous-first-derivatives-at-internal-interfaces)
for input validation and an optimization command.

For a width h > 0, shift every internal interface by the same auxiliary
displacement u, uniformly distributed over [-h/2,h/2], and average the
geometric cell fractions over u. The two outer faces are not shifted.
Internal interfaces must remain at least h/2 from those faces. Neighboring
internal averaging regions can overlap: the shared width and displacement
preserve ordering, nonnegative fractions and their partition of unity.
A single-layer stack has no internal face to average.

This is a regularization of material sampling. It does not differentiate
h, change the fixed mesh, move transverse raster boundaries or vary D.
It does not promise continuous higher derivatives or differentiability of
Boolean validity flags and arbitrary user loss functions.

## Independent derivation

For a fixed Yee cell [L,R] and an internal face b, define

```text
             0                    x <= -h/2
G_h(x) =     (x+h/2)^2 / (2h)     -h/2 < x < h/2
             x                    x >= h/2

C_h(b) = [G_h(R-b) - G_h(L-b)] / (R-L)
```

C is the averaged fraction to the right of b. The weight of a layer between
a and b is C(a)-C(b). Sharp exterior faces use the ordinary clipped step
fraction. Both G and its first derivative match at the two joins, so these
weights are C1 in each internal face position. The independently evaluated
speed transferred from the layer after a moving face to the layer before it is

```text
[clip((R-b)/h + 1/2, 0, 1) - clip((L-b)/h + 1/2, 0, 1)] / (R-L).
```

The common Array API implementation feeds these weights through existing
tangential arithmetic and normal harmonic epsilon/mu averaging, and arithmetic
electric-pole fills. It uses no detached geometry, custom gradient or
straight-through estimator. The default h=0 retains the prior overlap path.

The [Meep theory documentation](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/)
describes tangential/normal averaging and its limits for corners and
dispersive interfaces. The extra interface-position average above is derived
here; it is not an implementation copied from Meep. No Meep implementation or
previously excluded Meep-derived FDTDX projection source was opened.
Differentiating the existing ADE fills does not establish a general
dispersive-interface homogenization theorem.

## Graph and cell-crossing checks

The fraction oracle integrates shifted sharp geometry by splitting the
displacement integral at every cell/face intersection. Its piecewise-linear
integrand is integrated exactly by interval midpoints; it shares no production
primitive. NumPy, JAX and Torch match it on unequal cells and overlapping
internal supports. Analytic derivatives, both sides of each primitive join,
volume conservation, reflection, length-unit scaling and unchanged single-layer
scattering provide separate checks.

Whole-scene JAX/Torch tests jointly differentiate thickness, raster epsilon,
electric-pole strength and exterior epsilon. They include typed crossed
scenes, top/electric and bottom/directional extraction, grading, nominal
alignment, unequal transverse sampling, notched pulses, field monitors,
cropped records and convergence-prefix outputs. Two finite-difference steps
agree with AD. JIT checks also move interfaces on one trace and mask invalid
exterior clearance.

At resolution 20, the magnetic film below crosses the electric-cell face
d=0.175. The maximum difference between complete scattering derivatives at
d-delta and d+delta is:

| Width | delta=1e-4 | delta=1e-6 |
| --- | ---: | ---: |
| 0 | 1.29267 | 1.29182 |
| 0.05 (one cell) | 0.00633085 | 0.0000633094 |

The sharp-overlap jump persists. The averaged response's difference decreases
with the offset, as expected for a continuous first derivative. This is a
finite-computation property, separate from continuum derivative accuracy.

## Sharp and graded continuum references

The normal-incidence magnetic film uses epsilon/mu = 2.25/1.12, with top
1.2/1.1 and bottom 1.6/1.15. Its internal face takes d=0.163, 0.173 and 0.183;
a substrate spacer fixes D=0.4. Wavelengths are 0.98 and 1.02. The scene uses
directional extraction, nominal lengths [0.2,0.2], fitted cells, one physical
unit of CPML, pulse width 3 and duration 65.

The sharp reference is the independent Fresnel/Airy series. For positive h,
the graded reference integrates a linear tangential epsilon/mu ramp over h
using 512 homogeneous transfer-matrix slices. Its translation derivative is
analytic: only the uniform film and substrate lengths around the ramp change.
It uses no AD or FDTD weights. Doubling to 1024 slices changes the largest
reported value or derivative by at most 1.26e-6 across this experiment.
At h=0 it matches Airy to roundoff; two finite-difference steps verify the
analytic derivative, and a uniform-material control has zero translation
sensitivity.

Errors below are maximum absolute component errors across all three positions,
both wavelengths and the vector Re/Im(r), Re/Im(t), R, T. Derivatives are per
unit physical length. h is reduced with the grid when its width in cells is
held fixed.

| Resolution | Width in cells | Value error vs sharp | Derivative error vs sharp | Derivative error vs graded |
| --- | ---: | ---: | ---: | ---: |
| 20 | 0 | 0.02864 | 0.57148 | 0.57148 |
| 20 | 0.5 | 0.02874 | 0.25905 | 0.25655 |
| 20 | 1 | 0.02942 | 0.14189 | 0.08732 |
| 20 | 2 | 0.03130 | 0.27688 | 0.07946 |
| 40 | 0 | 0.006778 | 0.35055 | 0.35055 |
| 40 | 0.5 | 0.006785 | 0.09243 | 0.09034 |
| 40 | 1 | 0.006902 | 0.03265 | 0.01874 |
| 40 | 2 | 0.007379 | 0.07772 | 0.01893 |
| 80 | 0 | 0.001681 | 0.10869 | 0.10869 |
| 80 | 0.5 | 0.001682 | 0.05251 | 0.05286 |
| 80 | 1 | 0.001719 | 0.008700 | 0.005282 |
| 80 | 2 | 0.001832 | 0.01985 | 0.004186 |

One cell gave the smallest sampled derivative error against the sharp
interface at each resolution. Two cells further reduced the fine-grid error
against the graded profile, but increased sharp-interface bias. At resolution
80, the continuum derivative bias is 0.003921 for one cell and 0.015667 for
two cells; the corresponding value biases are 0.0001978 and 0.0007903.
Wider averaging is therefore not automatically more accurate.

This is one planar, lossless magnetic scene. It does not establish the same
error envelope for oblique patterns, thin lossy metals, corners or overlapping
dispersive interfaces. h=0 derivatives remain position-dependent and need not
improve monotonically at every individual interface position.

```shell
python -m studies.fdtd_layer_smoothing --output artifacts/layer-smoothing.json
```

## Independent Meep material-function comparison

The frozen Meep 1.34.0 reference uses its documented
[material-function API](https://meep.readthedocs.io/en/latest/Python_User_Interface/#medium).
A global function returns the sharp top interface at z=0 and the linear
epsilon/mu transition around d. Its `do_averaging` property enables subpixel
integration; `extra_materials` declares the magnetic media. No implementation
source is inspected. A periodic 3D cell carries a uniform transverse plane
wave, with one transverse cell in each direction. This is still a planar
normal-incidence physical experiment.

At resolution 80, d=0.173 and h=0.0125, the generator independently perturbs
d by 1e-3 and 5e-4 at durations 50 and 70. Complex r/t use the same fixed
outer reference planes as Wavebit. The reference uses cell length 10, CPML
thickness 1, Courant 0.4, source z=-3, E monitors z=-1 and 1.2, and a Gaussian
pulse of width 1.2 centered at time 6 and ending at time 12. Metadata in
`golden_meep_layer_smoothing.npz` records these choices; the companion
`golden_meep_layer_smoothing_refined.npz` tightens averaging tolerance from
1e-8 to 1e-9.

| Check | Maximum complex-amplitude or derivative difference |
| --- | ---: |
| Duration 50 to 70, values and derivatives | 0 in the stored arrays |
| Perturbation 1e-3 to 5e-4, derivatives | 1.49e-5 |
| Averaging tolerance 1e-8 to 1e-9, values | 1.58e-11 |
| Averaging tolerance 1e-8 to 1e-9, derivatives | 3.65e-8 |
| One to four cells along each transverse axis, values | 5.32e-16 |
| One to four cells along each transverse axis, derivatives | 1.78e-12 |
| Wavebit vs Meep, r/t | 0.000767 |
| Wavebit AD vs Meep finite differences | 0.000916 |
| Meep vs graded continuum, derivatives | 0.003781 |
| Wavebit vs graded continuum, derivatives | 0.003980 |

The comparison checks the new regularized profile rather than treating the
old sharp-interface golden as its reference. Agreement between finite-grid
derivatives is tighter than either solver's continuum accuracy in this case.
It does not establish general Meep equivalence for patterned or dispersive
shape derivatives. Regenerating the pre-existing sharp-film and half-space
archives with the extended wrapper preserves their data.

```shell
"$MEEP_PYTHON" tests/references/interface/meep_halfspace.py --parameter thickness --smoothing .0125 --output artifacts/meep-layer-smoothing.npz
"$MEEP_PYTHON" tests/references/interface/meep_halfspace.py --parameter thickness --smoothing .0125 --subpixel-tol 1e-9 --output artifacts/meep-layer-smoothing-refined.npz
"$MEEP_PYTHON" tests/references/interface/meep_halfspace.py --parameter thickness --smoothing .0125 --transverse-cells 4 --output artifacts/meep-layer-smoothing-transverse.npz
```

## Coating optimization with a final sharp check

```shell
python -m examples.fdtd_layer_optimization --smoothing-cells 1
```

A coating with epsilon 1.5 lies between air and epsilon 2.25 substrate.
JAX backtracking optimizes its thickness at resolution 40 and h=0.025,
with a compensating substrate spacer. Twenty updates move d from 0.12 to
0.2038164, reducing training reflection from 0.0146115 to 1.484e-7.

At resolution 80, shrinking h to 0.0125 gives reflection 1.465e-7.
The sharp-interface FDTD check on that finer mesh gives 1.446e-7, and
the sharp Airy result is 2.337e-7. The exact quarter-wave thickness is
0.2041241. These distinct checks prevent the finite-grid training objective
from being reported as the exact physical performance.

This remains a bounded single-parameter example. Total-depth changes,
transverse shape derivatives, anisotropic shape homogenization and GPU
execution need separate implementation and evidence.

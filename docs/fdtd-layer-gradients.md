# Internal layer derivatives on a fixed mesh

Written: 2026-09-09

Targets:
- `src/wavebit/_core/scene_fdtd.py`
- `tests/test_fdtd_layer_gradients.py`
- `studies/fdtd_layer_gradients.py`
- `tests/references/test_meep_layer_thickness.py`
- `tests/references/interface/meep_halfspace.py`
- `examples/fdtd_layer_optimization.py`

TL;DR: the explicit-backend scene accepts differentiable layer thicknesses
with static `mesh_thickness`. Actual lengths remain positive and preserve the
nominal total depth. Internal interfaces move through one fixed Yee mesh,
including bottom illumination, records and electric pole masks. JAX/Torch
derivatives match finite differences. Planar derivatives agree closely with
independently perturbed Meep runs, but both discretizations retain appreciable,
position-dependent continuum derivative error.

## Actual geometry and nominal mesh

`thickness` keeps the same meaning as RCWA and the shared `Stack`: the actual
physical length of each layer. There is no separate FDTD shape representation.
`mesh_thickness` selects cell counts, fitted nominal faces and graded padding.
It has one positive static real length per layer. With nominal total depth D,
an admissible two-layer design is `[d, D-d]`, for 0 < d < D. A multilayer design
can use positive fractions normalized to D. Only derivatives along such
constant-total paths are supported; independent changes to the total depth
are outside this API's derivative contract.

Source, PML, extraction planes, monitor coordinates, record selection, time
step and update count stay fixed. `align_layers=True` fits the **nominal**
interfaces. Actual internal interfaces may cross those nodes. The two exterior
faces remain exactly 0 and D, so reflection and transmission keep their stated
interface phase references. Bottom illumination reverses the nominal mesh and
actual layers separately, then restores original-coordinate diagnostics.
`layer_planes` retains the actual backend graph, while `mesh_thickness` in
diagnostics is the original-order static tuple.

JAX/Torch thickness arrays require explicit `mesh_thickness`; eager arrays
are never silently detached to build a new mesh. Plain host thicknesses retain
the existing mesh selection when the option is omitted. `subpixel=True` is
required with the option: point sampling would move interfaces in discrete
jumps. Invalid concrete lengths or total depth raise before stepping. Under
JAX JIT, invalid lengths or totals produce `stable=False` and NaN scattering
and normalized fields. The total-depth check allows 32 machine epsilons times D
for sum roundoff; both outer faces are pinned to the static nominal positions.
Represented boundaries must remain strictly ordered inside those faces, so
sum tolerance cannot silently collapse a positive but unresolvable last layer.

## What is differentiated

For a fixed cell [L,R] and moving layer [a,b], the unnormalized overlap is
`max(0, min(R,b) - max(L,a)) / (R-L)`. These fractions form a nonnegative
partition, including both exterior media. The same backend calculation feeds
electric and magnetic arithmetic/tangential and harmonic/normal averages, and
arithmetic electric-pole filling fractions. No external implementation, detached
geometry, or straight-through estimator is used.

This discretization is continuous and piecewise differentiable. At a cell-face
crossing its one-sided derivatives can differ. Backend subgradients at a kink
are not a unique continuum shape derivative. Check finite differences away
from those crossings, and assess continuum error at several boundary positions.
Pole-mask derivatives correctly differentiate the existing averaged ADE model;
that does not establish a general dispersive-interface homogenization theorem.

The [Meep subpixel-smoothing documentation](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/)
explains tangential arithmetic and normal harmonic averaging, as well as
limitations near sharp corners, interfaces and dispersive materials. It was
consulted as public theory documentation. Meep implementation code was not
opened or copied. The previously excluded Meep-derived FDTDX projection
implementation was not inspected either.

## Discrete graph checks

The overlap test checks independently calculated fractions and interface
derivatives on unequal cells for all three backends. Whole-scene tests vary
thickness, raster epsilon, pole strength and exterior epsilon together in a
crossed typed `Stack`, using JAX x64 and Torch float64 on CPU. They include
top/electric and bottom/directional extraction, unequal nominal layer lengths
and nonuniform fitted meshes,
Jones input, two wavelengths, pulse notches, user field planes, cropped time
records and prefix-convergence outputs. Two central-difference steps, 2e-5
and 1e-5, agree with AD at rtol 5e-5 and atol 4e-8.

JIT tests move the internal interface across cells on one trace. Separate
checks preserve the grid, timing and source/reference locations, reproduce
the previous forward result at nominal thicknesses, and reject invalid inputs.
These checks establish differentiation of the finite computation, not
continuum accuracy for arbitrary shapes or dispersive patterns.

Measurements used macOS arm64, Python 3.12.9, NumPy 2.5.1, JAX 0.11.0
and Torch 2.13.0. GPU execution was not part of these checks.

## Continuum refinement is position dependent

The independent Fresnel/Airy series uses top `(epsilon,mu)=(1.2,1.1)`,
film `(2.25,1.12)` and substrate `(1.6,1.15)`. The film occupies [0,d] and a
substrate spacer continues to the fixed output plane D=0.4. Transmission
includes the spacer phase; omitting its thickness derivative would compare
different reference planes. Wavelengths are 0.98 and 1.02 at normal incidence.
The scene uses directional extraction, physical PML thickness 1, pulse width 3
and duration 65. The oracle's two perturbation sizes agree within 8.4e-9.

These reference coordinates use the historical downward z axis from the
illuminated top face. In current public upward-z coordinates,
`z_public = D - z_report`: the film occupies `[D-d, D]` and the substrate
spacer occupies `[0, D-d]`.

Maximum absolute errors across real/imaginary r/t and power R/T at both wavelengths:

| Resolution | d=0.163 value / derivative | d=0.173 value / derivative | d=0.183 value / derivative |
| --- | --- | --- | --- |
| 20 | 0.02810 / 0.28545 | 0.02763 / 0.57148 | 0.02864 / 0.44209 |
| 40 | 0.006394 / 0.35055 | 0.006778 / 0.04671 | 0.006718 / 0.17227 |
| 80 | 0.001660 / 0.003936 | 0.001681 / 0.04804 | 0.001665 / 0.10869 |

Values improve consistently here. Individual derivative errors do not decrease
monotonically: interface position within a cell changes under refinement.
The maximum sampled derivative error decreases from 0.5715 to 0.3505 to
0.1087. These three positions are a measured sample, not a worst-case bound
over all thicknesses. Do not infer derivative accuracy from small R/T error
or choose only an unusually favorable interface position.

Reproduce the measurement from the repository root:

```shell
python -m studies.fdtd_layer_gradients --output artifacts/layer-gradients.json
```

## Independent Meep perturbations

`golden_meep_layer_thickness.npz` was generated through Meep 1.34.0's
documented public API on a fixed one-dimensional cell, at resolution 80 and
d=0.173. It retains two perturbations (1e-3, 5e-4), two durations (50, 70),
complex r/t and experiment metadata. The cell length is 10, PML thickness 1,
Courant fraction 0.4, source z=-3 and E planes z=-1 and 1.2. A Gaussian
current of width 1.2 peaks at 6 and ends at 12. Each perturbation repeats the
reference and actual simulations, with phases referred to 0 and D=0.4.

Both duration changes are zero at stored precision. Reducing the perturbation
changes the derivative by at most 2.32e-6. Wavebit differs from Meep by
7.67e-4 in complex amplitude and 9.24e-4 in complex thickness derivative.
Meep's own maximum derivative difference from Airy is 0.0510. Thus the close
agreement between finite grids does not demonstrate that their continuum
derivatives are accurate to 0.001. This reference covers one normal-incidence
magnetic film, not patterned, oblique or lossy shape sensitivities.

```shell
"$MEEP_PYTHON" tests/references/interface/meep_halfspace.py --parameter thickness --output artifacts/meep-layer-thickness.npz
```

## Design use and remaining scope

The [coating example](../examples/fdtd_layer_optimization.py) uses a shared
typed `Stack` and JAX gradients to reduce reflection, with a compensating
substrate spacer. Backtracking accepts measured improvements and the final
design is checked at twice the resolution and against quarter-wave optics.
This is a bounded single-parameter example, not evidence for arbitrary
multilayer optimization convergence.

At resolution 40, 20 accepted updates move the coating from 0.12 to
0.203802. Reflection falls from 0.014673 to approximately 3.05e-8 on that mesh.
The same final design gives 1.61e-7 at resolution 80 and 2.57e-7 from Airy;
the exact quarter-wave thickness is 0.204124. The final design is close to
the physical optimum, but the smaller training-grid reflection alone would
overstate its physical performance.

The later [interface-averaging extension](fdtd-layer-smoothing.md) adds optional
C1 sampling with a fixed physical averaging width. The default sharp-overlap
measurements above remain unchanged. Total-depth variation, transverse
shape/raster-boundary derivatives, anisotropic shape homogenization and GPU
execution remain outside this change. Fixed-total layer derivatives can be
combined with existing angle/material derivatives, but each new physical
regime still needs optical and gradient refinement checks.

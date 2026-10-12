# Half-space derivatives without changing the experiment

Written: 2026-09-09

Targets:
- `src/wavebit/_core/scene_fdtd.py`
- `src/wavebit/_core/grid_fdtd.py`
- `src/wavebit/_core/fdtd.py`
- `tests/test_fdtd_halfspace_gradients.py`
- `tests/references/interface/meep_halfspace.py`
- `tests/references/test_meep_halfspace.py`

TL;DR: positive real top/bottom epsilon and mu now retain derivatives through
the explicit-backend scene engine. Static nominal `mesh_indices` select
padding and grading, while actual materials drive the complete physical and
calibration computation, including CPML. Independent coefficient derivatives,
same-grid finite differences, magnetic Airy refinement and public-API Meep
perturbations check different parts of the contract.

## Geometry must be selected separately

Previously, converting the half-space coefficients to Python complex numbers
cut their graphs. It also allowed the resulting refractive indices to select
padding counts and graded cell widths. Removing just the conversions would
leave a mesh depending on differentiated data. Detaching those values to build
a new mesh at each evaluation would compare different discrete experiments.

The [usage contract](guide/time-domain.md#half-space-material-derivatives)
therefore requires an explicit positive static pair `mesh_indices` and a
diffraction window whenever any exterior coefficient is an array. These
nominal indices control only geometry. Scalar host materials retain the
previous default mesh policy. Both nominal and actual background diagnostics
use original top/bottom order, including bottom illumination.

The derivative includes boundary-cell material averaging, Maxwell updates,
material-dependent CPML coefficients, the incident-medium reference runs,
directional admittance projection, interface phase references and power
normalization. At fixed theta/phi the incident refractive index changes the
transverse Bloch vector, preserving the central-wavelength angle. This must
be included when comparing an oblique derivative to another solver.

Duration and CFL checks use actual materials. Increasing the actual optical
travel time can invalidate an otherwise fixed run; the solver does not silently
add updates. Concrete invalid inputs raise. Abstract JAX violations invalidate
`stable` and mask scattering and normalized monitor/record fields.
Changing the outgoing index across its continuum cutoff preserves one JIT
trace and mesh on both sides; an exactly grazing retained channel is refused
or masked. That guard is not a convergence guarantee near the cutoff.

## The absorber is part of the differentiated calculation

The native API now also accepts live `pml_background` pairs. Static physical
geometry and profile strength remain separate. For index
`n = sqrt(epsilon*mu)`, the existing coordinate stretch has
`sigma_max` proportional to `1/n`, so each coefficient obeys
`d sigma / d epsilon = -sigma/(2*epsilon)` and
`d sigma / d mu = -sigma/(2*mu)` with the other variable fixed.
The tests independently differentiate the exponential recurrence and quotient
for both end layers, including unequal physical thicknesses and exact interior
identity coefficients. Values and Jacobians agree to `2e-15` on both AD
backends. No Maxwell equation is duplicated by a backend.

Native full-loop tests vary all four exterior coefficients on uniform and
nonuniform meshes. Two finite-difference steps, `2e-5` and `1e-5`, check powers,
recorded electric fields and final magnetic fields at `rtol=2e-6`,
`atol=2e-9`. The CPML edge and adjacent interior coefficients still have to
match the supplied backgrounds, with no conductivity or nonzero pole strength.

The scene graph test additionally varies incidence theta and Lorentz strength
in a crossed bottom-illuminated scene. It includes graded/aligned z geometry,
unequal transverse sampling, a spectral source notch, independent source and
extraction positions, complex Jones amplitudes, cropped histories, internal
monitors and prefix convergence diagnostics. Electric and directional
extraction both agree with steps `2e-4` and `1e-4` at `rtol=6e-5`, `atol=4e-8`.

## Independent optics and an external derivative

Measurements use macOS arm64 CPU, NumPy, JAX x64 and Torch float64. The planar
case has epsilon/mu `(1.2,1.1)` above, `(1.6,1.15)` below, and `(2.25,1.12)`
inside a thickness-0.2 film. Wavelengths are 0.98 and 1.02 at normal incidence.
The continuum oracle uses the two-interface Airy expression with magnetic
admittance `sqrt(epsilon/mu)`. Its four coefficient derivatives come from
independently converged differences of that analytic expression, rather than
from Wavebit's propagation, calibration or CPML helpers.

The FDTD refinement uses resolutions 40/80, an aligned layer, period 0.05,
physical PML thickness 1, Gaussian width 3 and duration 65. Both complex
amplitudes and R/T are differentiated. Refinement must reduce the maximum
value and derivative errors by more than a factor of two and leave each below
0.005 at resolution 80.

| Resolution | Maximum real-output error | Maximum Jacobian error |
| --- | ---: | ---: |
| 40 | 0.00495837 | 0.00525952 |
| 80 | 0.00123219 | 0.00129845 |

Each maximum includes real/imaginary r/t and R/T at both wavelengths; the
Jacobian maximum also ranges over all four exterior coefficients. Both
errors decrease by approximately four when the cell width is halved.

Meep 1.34.0 was executed through its
[documented public API](https://meep.readthedocs.io/en/latest/Python_User_Interface/).
Its implementation was not read or copied. The project-owned generator runs
one physical slab and one incident-medium reference for every parameter
perturbation in a fixed one-dimensional cell of length 10 at resolution 80.
PML is 1 thick, the source is at z=-3, and Ex measurement planes are at -1 and
1.2. A width-1.2 Gaussian sine current peaks at time 6 and ends at 12.
Homogeneous Yee dispersion de-embeds the planes to the interfaces.

| External-oracle control | Maximum absolute difference |
| --- | ---: |
| Amplitudes, duration 50 versus 70 | 0 at stored precision |
| Derivatives, duration 50 versus 70 | 0 at stored precision |
| Derivatives, perturbations 1e-3 versus 5e-4 | 5.79e-8 |
| Meep amplitudes versus continuum Airy | 0.001293 |
| Meep derivatives versus continuum Airy | 0.001763 |

Ordinary tests read the frozen reference without importing Meep. They require
Wavebit's complex amplitudes to agree within 0.003 and its JAX Jacobian within
0.005. Measured maximum differences are `0.000768690` in amplitude and
`0.00250055` in the Jacobian. Each of the four external parameter directions has a maximum amplitude
slope above 0.1, preventing an absent graph from passing these tolerances.
These are comparisons between different grids/time steps and PML formulations,
not an assertion that the two finite-difference algorithms are identical.

Regenerate the external reference only with a separately installed Meep
interpreter, and preserve the independent controls:

```shell
"$MEEP_PYTHON" tests/references/interface/meep_halfspace.py --output artifacts/meep-halfspace.npz
```

## Limits

A finite-grid derivative can be correct while the physical answer is still
underresolved. The nominal mesh indices do not guarantee accuracy across a
large material range. The
[Meep PML documentation](https://meep.readthedocs.io/en/latest/Perfectly_Matched_Layer/)
also calls for thickness-convergence checks and separation from significant
evanescent fields. Refine resolution, physical PML thickness, padding and
duration where a design will be used.

This extension covers positive real scalar, nondispersive exterior media.
It does not add lossy/dispersive half spaces, geometry derivatives, arbitrary
PML anisotropy or a derivative through changes in order count. The external
case is planar and normally incident; it does not validate general conical
or near-threshold derivatives. CPU differentiation and selected regressions
do not establish GPU behavior, Torch compile support or complete solver parity.

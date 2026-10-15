# Constrained multilayer optimization

Written: 2026-09-09

Targets:
- `src/wavebit/_common/periodic.py`
- `src/wavebit/rcwa/_engine/factorization.py`
- `examples/fdtd/fdtd_multilayer_optimization.py`
- `studies/fdtd/multilayer_optimization/fdtd_multilayer_optimization.py`
- `tests/common/test_layer_thicknesses.py`
- `tests/fdtd/test_fdtd_multilayer_optimization.py`
- `tests/rcwa/test_factorization_core.py`
- `tests/references/test_meep_multilayer.py`
- `tests/references/interface/meep_multilayer.py`

TL;DR: `layer_thicknesses` turns ordinary NumPy/JAX/Torch design weights into
positive lengths with a fixed total and optional individual minima. The same
array feeds RCWA and FDTD. A six-variable dielectric coating example reduced
the independent sharp-stack mean reflectance over 81 sampled wavelengths
from 5.1047% to 0.023705%. Finer FDTD meshes and independently measured Meep
derivatives support this particular design. The optimizer stopped at its
30-iteration limit; this is an improved design, not a convergence certificate.

## Shared geometry coordinates

For N layers, static total D, minimum lengths m and real weights u, define

```text
w_i = exp(u_i - max(u)) / sum_j exp(u_j - max(u))
d_i = m_i + (D - sum(m)) w_i
∂d_i/∂u_j = (D - sum(m)) w_i (δ_ij - w_j).
```

The weights are dimensionless. Their common shift is redundant, so the
example fixes the last weight to zero. The output is an ordinary thickness
array, accepted by the existing `Stack` and both solvers. There is no separate
FDTD geometry representation. Total and minima are static host constraints;
only the weights are differentiated. See the
[input contract](../../migration/autodiff-v1.md#constraining-several-layer-thicknesses).

The analytic Jacobian supplies an independent three-backend test, including
its zero column sums and common-shift null direction. Positive minima prevent
layers from disappearing when the exponential saturates, but cannot prevent
small gradients. Zero-length underflow is refused. Constraints are evaluated
in floating-point arithmetic; use float64 for physical FDTD comparisons.
Invalid traced JAX inputs produce NaN lengths instead of silently clipped
geometry.

Typed RCWA checks preserve the length derivative through `jax.jit` and Torch
and compare with two finite-difference steps. These exposed a JAX CPU FFT
layout failure on a uniform, one-pixel raster. A one-sample DFT is the identity;
the common Fourier builder now uses that identity with complex promotion on
singleton axes. Independent exact pixel integrals check all three backends
on 1-by-1, 1-by-4 and 4-by-1 rasters, including compiled JAX execution.

## Six-variable coating example

The layer arrays recorded here use the historical top-to-bottom order: coatings
first, substrate spacer last. Current public `Stack` arrays use bottom-to-top
order, so both permittivity and thickness arrays must be reversed at that
boundary. The example preserves the original design-parameter order and
performs this conversion explicitly.

The scene has three dielectric coatings followed by a substrate spacer.
Top epsilon is 1 and substrate epsilon is 4, with mu=1 throughout. Total
depth is 0.8 and every layer has minimum length 0.04. All lengths and
wavelengths use the same arbitrary physical unit.

Three free thickness weights plus three real coating permittivities form the
design vector. Weight bounds are [-4,4]; epsilon bounds are [1.1,4]. The
spacer epsilon remains 4, and its variable length keeps both exterior planes
fixed. Initial lengths are [0.08,0.10,0.12,0.50], with coating epsilon
[2.7,1.6,3.2]. Materials are continuous optimization variables, not a discrete
catalog of available or fabricable materials.

The objective is mean s-polarized reflectance at normal incidence and three
wavelengths [0.8,1.0,1.2]. The FDTD setup uses resolution 40, nominal lengths
[0.2,0.2,0.2,0.2], fitted cells, directional extraction, one unit of CPML,
pulse width 3, duration 65 and zeroth-order extraction. Internal averaging
has physical width h=1/resolution; the total, source, mesh and output planes
stay fixed during each optimization. The guide explains why
[interface averaging](fdtd-layer-smoothing.md) gives continuous first
derivatives while changing the sampled physical profile.

The example passes both objective and JAX reverse-mode gradient to SciPy's
[L-BFGS-B interface](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html),
with `ftol=1e-12`, `gtol=1e-7` and a budget of 30 iterations. It reports the
returned status and message. This run used 36 objective evaluations reported
by SciPy, plus callback evaluations for the recorded history. It stopped with
`STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT` and `optimizer_success=False`.

| Quantity | Initial | After 30 iterations |
| --- | ---: | ---: |
| Training mean R, resolution 40 | 0.0522222 | 0.000224965 |
| Independent sharp mean R, 81 wavelengths | 0.0510468 | 0.000237054 |
| Independent sharp largest R, 81 wavelengths | 0.0912548 | 0.000540387 |

The final physical lengths are approximately
[0.0793044472,0.1541410422,0.1383468378,0.4282076728], and coating epsilon
is [1.2308075548,1.5211291788,2.8040978636]. The preserved reference archive
stores the full-precision design vector and its decoded geometry.

| Wavelength | Training R, grid 40, h=0.025 | Grid 80, h=0.0125 | Grid 80, sharp | Sharp continuum |
| --- | ---: | ---: | ---: | ---: |
| 0.8 | 0.0000444441 | 0.0000341833 | 0.0000366224 | 0.0000310290 |
| 1.0 | 0.000421000 | 0.000433841 | 0.000429257 | 0.000437442 |
| 1.2 | 0.000209450 | 0.000215167 | 0.000219167 | 0.000217510 |

The 81-point check samples [0.8,1.2] at spacing 0.005 and includes wavelengths
absent from training. It does not prove a continuous-band maximum or a global
optimum. The run was measured on an arm64 CPU with Python 3.12.9, NumPy 2.5.1,
SciPy 1.18.0, JAX 0.11.0 and Torch 2.13.0, using float64/complex128. GPU execution was not
part of this measurement.

## Independent continuum and derivative checks

The example uses a Fresnel multiple-reflection recursion for its final sharp
check. The study separately propagates the tangential E/H pair through each
layer with 2-by-2 characteristic matrices. The formulations agree with each
other and RCWA to roundoff, conserve energy and preserve amplitudes when a
uniform spacer is split. Neither uses FDTD sampling weights or its gradient.

For positive h, the study integrates the linear tangential epsilon profile
using 128 homogeneous slices per interval between ramp endpoints, allowing
neighboring ramps to overlap. A 256-slice calculation checks this reference's
own convergence. The all-uniform limit has the known Fresnel reflection and
transmission phase. Permittivity is positive, real and isotropic in these
references; these measurements do not establish a general anisotropic or
dispersive design result.

Each row below compares all three reflectances and their full 3-by-6
Jacobian. Reported errors are maximum absolute errors; weight derivatives
and relative-permittivity derivatives are with respect to the coordinates
defined above, not per-unit-thickness sensitivities. The sharp reference
uses centered steps 1e-5 and 5e-6; their maximum disagreement is 1.39e-11.
The largest graded-reference derivative change on doubling its slices is
5.68e-7. Both are smaller than the measured FDTD derivative errors.

| Design | Grid | R error vs sharp | Jacobian error vs sharp | R error vs graded | Jacobian error vs graded |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 20 | 0.0104958 | 0.0203475 | 0.00691242 | 0.0164606 |
| Initial | 40 | 0.00235935 | 0.00381376 | 0.00144959 | 0.00236870 |
| Initial | 80 | 0.000626139 | 0.00127131 | 0.000398677 | 0.000986717 |
| Optimized | 20 | 0.0000854816 | 0.00260481 | 0.0000834278 | 0.00296265 |
| Optimized | 40 | 0.0000164420 | 0.000723824 | 0.0000154745 | 0.000715861 |
| Optimized | 80 | 0.00000360105 | 0.000162521 | 0.00000384282 | 0.000182350 |

Here h decreases with grid spacing. Keeping h fixed instead converges toward
the graded material, so refinement alone would not establish the sharp-stack
derivative. Even at grid 80, the optimized graded and sharp continuum
Jacobians differ by up to 0.000117274. The study records that profile bias
separately from numerical discretization error.

Separate JAX and Torch tests use bottom incidence, nonzero theta/phi and
complex Jones illumination. All six joint variables have nonzero derivatives;
two finite-difference steps agree with AD. Perturbing them preserves grid
coordinates, source/PML positions, time step, step count and nominal geometry.
These graph checks complement the normal-incidence physical references.

## Meep comparison and reference uncertainty

Meep 1.34.0 independently evaluated the final design and all six centered
perturbations. The reference uses its documented global
[material function](https://meep.readthedocs.io/en/latest/Python_User_Interface/#medium)
and [material averaging](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/)
in a 3D periodic cell with one transverse cell in each direction. Grid 80,
h=0.0125, a length-10 cell, one unit of z PML and Courant 0.4 are explicit.
A full-transverse Ex source lies at z=-3, with DFT planes at -1 and 1.4.
Its Gaussian-windowed pulse has width 1.2, peak time 6 and end time 12.
An independently measured uniform-top reference and discrete Yee propagation
phases place r at z=0 and t at z=0.8. Reference metadata stores these settings.

| Check | Maximum absolute change or difference |
| --- | ---: |
| Meep durations 50 vs 70, complex r/t | 5.29e-17 |
| Meep durations 50 vs 70, complex Jacobian | 6.25e-14 |
| Meep perturbation steps 1e-3 vs 5e-4 | 0.000120765 |
| Meep averaging tolerance 1e-8 vs 1e-9, complex r/t | 4.18e-11 |
| Meep averaging tolerance 1e-8 vs 1e-9, complex Jacobian | 2.72e-7 |
| Wavebit vs Meep, complex r/t | 0.00513490 |
| Wavebit vs Meep, complex Jacobian | 0.00239551 |
| Wavebit vs graded continuum, complex r/t | 0.00947685 |
| Meep vs graded continuum, complex r/t | 0.0102665 |
| Wavebit vs graded continuum, complex Jacobian | 0.00471004 |
| Meep vs graded continuum, complex Jacobian | 0.00453652 |

Complex-amplitude errors include phase and are distinct from the reflectance
errors above. The two finite-grid solvers are also checked individually
against the independent continuum profile; mutual agreement is insufficient.
The committed archives keep both averaging tolerances, both durations and
both perturbation sizes. Normal CI reads them without installing Meep.
Only Meep's documentation and black-box public API were used. No Meep
implementation or excluded Meep-derived FDTDX projection source was opened,
copied or linked into Wavebit.

## Reproduce

From the repository root in an environment with Wavebit's JAX extra and SciPy:

```shell
python -m examples.fdtd.fdtd_multilayer_optimization > artifacts/design.json
python -m studies.fdtd.multilayer_optimization.fdtd_multilayer_optimization --design artifacts/design.json --output artifacts/multilayer-check.json
pytest -q tests/common/test_layer_thicknesses.py tests/fdtd/test_fdtd_multilayer_optimization.py tests/references/test_meep_multilayer.py
```

To check the exact preserved design rather than a fresh optimizer trajectory,
first export only its input parameters from the independent reference archive:

```python
import json
from pathlib import Path
import numpy as np

with np.load('tests/references/golden_meep_multilayer.npz') as reference:
    parameters = reference['parameters'].tolist()
Path('artifacts/design.json').write_text(json.dumps({'parameters': parameters}))
```

In a separately installed Meep environment, regenerate candidate reference
files outside the repository and compare them before replacing any archive:

```shell
python tests/references/interface/meep_multilayer.py --design artifacts/design.json --output artifacts/meep-multilayer.npz
python tests/references/interface/meep_multilayer.py --design artifacts/design.json --subpixel-tol 1e-9 --output artifacts/meep-multilayer-refined.npz
```

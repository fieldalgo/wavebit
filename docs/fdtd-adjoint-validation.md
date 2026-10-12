# Adjoint, discrete derivatives and physical accuracy

Written: 2026-09-09

Targets:
- `studies/fdtd_adjoint_comparison.py`
- `studies/fdtd_rotation_smoothing.py`
- `tests/test_fdtd_rotation_smoothing.py`
- `tests/references/interface/meep_adjoint_rotation.py`
- `tests/references/test_meep_adjoint_rotation.py`

TL;DR: the earlier Meep shape derivatives were central differences of forward
simulations. This measurement actually invokes `meep.adjoint`, links its pixel
gradient to an independently differentiated shape profile, and compares it with
matching forward differences. Agreement of forward fields, repeatability of
sensitivities and accuracy against a continuum reference are separate checks.
The measured adjoint differences are retained rather than treating the API as
a guarantee of an accurate reference gradient.

## What is being differentiated

Use the [translation-averaged rotated rectangle](fdtd-rotation-smoothing.md)
with angle 0.37, width 0.3, height 0.18, centre `(0.071,-0.047)`, epsilon
2.25, mu 1 and layer depth 0.2. The periods are `(0.6,0.6)`, wavelength 1,
top epsilon 1 and bottom epsilon 1.21. The fixed translation width is 0.05
on both axes. This page compares the two complex co-polarized transmissions
`t_ss` and `t_pp`, corresponding to channels 2 and 7 of the earlier eight-channel
archive. Cross polarization and reflection are outside this adjoint measurement.

For a real objective `J` and design parameter `p`, the geometry chain is

```text
dJ/dp = sum_i (dJ/d rho_i) (d rho_i/dp).
```

Meep supplies the first factor through its adjoint API. The independent sinc
profile supplies the second analytically, including derivatives of rotation
and width. The real and imaginary parts of each complex transmission are
separate real objectives; the resulting sensitivities are combined afterward.
A complex transmission coefficient is not passed to a real-objective optimizer.

The [Meep adjoint documentation](https://meep.readthedocs.io/en/latest/Python_Tutorials/Adjoint_Solver/)
describes forward/adjoint DFT fields and the shape-parameter chain rule. Its
2D binary-grating example also reports finite-grid differences between adjoint
and finite differences, and a limitation for its in-plane electric polarization.
That is not proof of the cause of the differences in this 3D graded case.
It is a reason to measure them explicitly. Only documentation and public API
execution were used; no Meep implementation or excluded Meep-derived FDTDX
implementation was opened or copied.

## Independently differentiate the material profile

The rotated rectangle's Fourier coefficient is an area factor times two sinc
factors, multiplied by the two translation-box sinc factors and a position
phase. With rotated frequencies `u,v`, rotation gives `du/dangle=v` and
`dv/dangle=-u`. Width differentiation uses
`d/dwidth [width*sin(u*width/2)/(u*width/2)] = cos(u*width/2)`.
A small-argument series evaluates the derivative of `sin(z)/z` safely at zero.

Clipping the truncated profile to `[0,1]` has zero derivative outside that
interval. At a clipping knot this helper chooses zero; finite differences
crossing a knot need not match that local derivative. The tests therefore
compare pixel derivatives where both perturbed profiles retain the same branch.
They include angle zero, a generic rotation and a quarter turn. The unclipped
profile's mean derivatives are exactly `[0, height/period_area]`. Independent
finite differences use two steps, and the Meep sensitivity contracted with two
finite-difference geometry mappings is also retained in each archive.

The optical reference uses this independent profile on a 256 by 256 grid and
RCWA Li factorization at Fourier order 8. The
[existing grid, Fourier-order and difference-step controls](fdtd-rotation-smoothing.md)
bound its numerical uncertainty. This is a controlled approximation to the
graded continuum problem; it is not an exact analytic Maxwell solution or a
reference for the unsmoothed rectangle.

Wavebit's complete complex-transmission graph is independently tested with
JAX forward mode, JAX reverse mode and two NumPy finite-difference steps.
The forward/reverse comparison uses absolute/relative tolerances 3e-11/2e-9;
both match those finite differences. This verifies the derivative of the implemented fixed-grid
calculation. It does not eliminate physical discretization error.

## Keep the forward experiment identifiable

Meep 1.34.0 uses a cell-centred 256 by 256 MaterialGrid, with extra averaging
and projection disabled, and the same geometry as the earlier forward runs.
Public `get_epsilon_grid` probes check axes and coordinates after each design
update. Two separate current components excite the s/p bases. The source plane
is -2, calibration plane -1 and transmission plane 1.2; the cell length is 8
and z PML thickness is 1. Courant is 0.4. Discrete Yee propagation phases
reference transmission to the layer exit at 0.2.

The adjoint wrapper uses a GaussianSource at frequency 1 with width 2.5 and
cutoff 5. Its empty-scene field calibrates the source spectrum and incident
amplitude. Central differences run through the same OptimizationProblem with
`need_gradient=False`, so they share the source, monitors and normalization.
Both are also compared against the earlier CustomSource forward archive.

All FourierFields observations explicitly use `decimation_factor=1` and
`yee_grid=False`, matching the earlier centred-field quadrature. The returned
plane weights integrate to the period area. A preliminary automatic-decimation
probe shifted a complex transmission by 3.15e-7; every-step sampling reduced
the discrepancy to around 1e-11. Native Yee samples have different dimensions
and must not be multiplied by centred-field weights.

The two termination configurations set both minimum and maximum run time to
120 or 140. End times of the last adjoint run and each perturbed forward run
are stored. A preliminary candidate allowed stopping between 80/120 or 100/140;
some perturbed runs stopped at 96.97. Its resolution-40 adjoint change of 4.67e-7
and earlier-forward-derivative discrepancy of 1.33e-7 exceeded the preselected
3e-7 and 1e-7 controls. That candidate was not frozen. Fixed run lengths remove
this varying-stop-time contribution rather than increasing those tolerances.
The retained `decay_by=1e-11` option does not establish that a separate DFT
decay criterion was satisfied: repeatability is tested by extending the fixed
run length.
The forward perturbation steps are 1e-4 and 5e-5. Geometry-only mapping steps
are 1e-5 and 5e-6; the reported adjoint uses the analytic mapping.

## Measured controls and adjoint differences

Both fixed-time candidates pass the original controls. The maximum material
probe error is 4.44e-16. Stored end times are 120.02/140.02 at resolution 20
and 120.01/140.01 at resolution 40, reflecting the discrete time step.

| Control: maximum absolute change | Resolution 20 | Resolution 40 |
| --- | ---: | ---: |
| Transmission, time 120 to 140 | 2.73e-11 | 7.36e-11 |
| Adjoint derivative, time 120 to 140 | 1.14e-9 | 7.11e-10 |
| Central derivative, time 120 to 140 | 5.52e-10 | 9.81e-10 |
| Angle derivative, forward-difference step halved | 2.30e-7 | 5.26e-7 |
| Width derivative, forward-difference step halved | 5.75e-7 | 1.40e-5 |
| Adjoint contraction, analytic versus difference-based geometry mapping | 6.77e-8 | 9.91e-8 |
| Transmission versus earlier forward archive | 5.35e-11 | 5.68e-11 |
| Central derivative versus earlier forward archive | 6.10e-10 | 8.87e-10 |

All derivative differences below are absolute complex-amplitude differences
per radian for angle, or per model length unit for width. They are not
percentages. Compare the analytic-mapping adjoint at time 140 with the matching
central derivative at step 5e-5:

| Resolution | Channel | Angle derivative gap | Width derivative gap |
| ---: | --- | ---: | ---: |
| 20 | `t_ss` | 0.000796833 | 0.000666543 |
| 40 | `t_ss` | 0.000249454 | 0.00108170 |
| 20 | `t_pp` | 0.00960079 | 0.0137699 |
| 40 | `t_pp` | 0.00150153 | 0.0135741 |

These gaps substantially exceed the runtime and geometry-chain controls.
Angle gaps shrink on this refinement, but the `t_ss` width gap grows and the
`t_pp` width gap barely shrinks. Thus this experiment does not establish
monotonic agreement between Meep adjoint and its discrete forward derivative.
It also does not isolate the internal cause of their difference.

## Accuracy against the independently controlled RCWA reference

The following maxima cover only `t_ss` and `t_pp`. Wavebit uses its existing
duration 60 calculation; its 60-to-80 runtime control is documented in the
earlier smoothing study. Both Meep derivative methods share the same forward
transmission at time 140.

| Resolution | Method | Transmission error | Angle derivative error | Width derivative error |
| ---: | --- | ---: | ---: | ---: |
| 20 | Wavebit JAX autodiff | 0.00806968 | 0.00149886 | 0.0127111 |
| 40 | Wavebit JAX autodiff | 0.00210467 | 0.000416159 | 0.00309728 |
| 20 | Meep central differences | 0.0157287 | 0.00528222 | 0.0807183 |
| 40 | Meep central differences | 0.00614268 | 0.00134500 | 0.0518944 |
| 20 | Meep adjoint | 0.0157287 | 0.0148823 | 0.0944307 |
| 40 | Meep adjoint | 0.00614268 | 0.00284622 | 0.0654554 |

At resolution 40, Wavebit's maximum angle/width differences from Meep central
differences are 0.00138489/0.0500235; from Meep adjoint they are
0.00288636/0.0635904. Using adjoint in place of central differences therefore
does not resolve the earlier width discrepancy in this case. Both Meep
methods refine toward the RCWA reference over these two grids, but neither
is a precise continuum width-gradient oracle here.

Wavebit satisfies the preselected resolution-40 budgets of 0.003 for complex
transmission, 0.001 for angle derivatives and 0.005 for width derivatives.
These results apply to this fixed graded profile, two shape parameters and
two transmission channels. They do not establish a general ranking of the
solvers or eliminate the sharp-interface and rotated-material limitations.

## Reproduce

Wavebit measurements use an arm64 CPU, Python 3.12.9, NumPy 2.5.1 and JAX
0.11.0 x64. The separate Meep environment uses Python 3.12.14, NumPy 2.5.2,
Meep 1.34.0, Autograd 1.8.0 and JAX 0.11.0. Autograd's MIT distribution was
installed for this external verification; it is not a new Wavebit dependency.
No GPU execution or performance advantage is established by this comparison.
There are only two shape variables here, so pointwise finite differences remain
practical even though adjoint methods can scale to many material variables.

From the repository root, with a separate Meep environment that includes its
adjoint dependencies:

```shell
python tests/references/interface/meep_adjoint_rotation.py --resolution 20 --output artifacts/meep-adjoint-rotation-coarse.npz
python tests/references/interface/meep_adjoint_rotation.py --resolution 40 --output artifacts/meep-adjoint-rotation.npz
```

Validate candidates before replacing archives. With Wavebit's test dependencies:

```shell
python -m studies.fdtd_adjoint_comparison --output artifacts/adjoint-comparison.json
pytest -q tests/references/test_meep_adjoint_rotation.py
pytest -q tests/test_fdtd_rotation_smoothing.py -k 'sinc_profile or fourier_reference or forward_reverse'
```

The external tests enforce reference provenance, unchanged forward behavior,
repeatability, geometry-chain controls and Wavebit's independent Fourier
accuracy budget. The size of Meep's adjoint-versus-difference gap is a measured
limitation, not a tolerance that is silently assigned to Wavebit.

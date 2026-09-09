# Exact translation averaging for rotated rectangles

Written: 2026-09-09

Targets:
- `wavebit/_core/scene_fdtd.py`
- `wavebit/main.py`
- `wavebit/manifest.py`
- `tests/test_fdtd_rotation_smoothing.py`
- `studies/fdtd_rotation_smoothing.py`
- `tests/references/interface/meep_rotation.py`
- `tests/references/test_meep_rotation_smoothing.py`

TL;DR: positive `shape_smoothing` now works with `shape_integration='area'`.
Exact polygon moments average a rectangle over a transverse translation box
without a quadrature setting. Cell-crossing derivatives become continuous on
averaged axes, while the mesh and computation graph stay fixed. This changes
the physical material profile. A matching independent Fourier profile separates
that change from the FDTD cell error; it does not supply the missing rotated
effective tensor or establish general sharp-interface gradient accuracy.

## Exact integration of the translated cell

For a cell width `d` and translation width `h`, averaging the cell indicator
over `[-h/2,h/2]` gives a trapezoid with knots `±(d+h)/2`, `±(d-h)/2`
and height `min(d,h)/h`. There are at most three linear pieces. The product
of the x and y weights is bilinear on at most nine boxes. Instead of sampling
translated rectangles, the solver clips the live rectangle against each box
and integrates the exact polygon moments of `1`, `u`, `v` and `u*v`.
Coordinates `u,v` are normalized to the box to reduce cancellation.

The zero-width path retains the original exact unweighted area integral.
At `h=d` the middle piece has zero width and is omitted. Very small positive
pieces use safe normalized coordinates. The periodic sum normally uses the
nearest nine rectangle images; when a cell-plus-kernel support exceeds its
period, a second image ring is included on that axis. This is necessary for
near-full-period rectangles at a periodic seam. Widths no larger than their
periods and the existing projected-extent bounds keep that support finite.

All shape and material values remain live through NumPy, JAX or Torch's shared
array operations. Mesh positions and smoothing widths are static. There is no
frozen topology plan, sampled shape raster, numerical quadrature order or
custom derivative in this integration. The [usage contract](guide/time-domain.md#rotated-rectangle-area-integration)
explains the width units, zero components and unchanged one-shape-per-layer
scope. Transverse epsilon/mu are still arithmetic averages; z sampling keeps
its existing normal harmonic and tangential arithmetic rules.

## Independent geometry and complete graphs

The geometry oracle integrates vertical slices. It integrates each y weight
analytically, then splits x at shape vertices, kernel knots and crossings of
horizontal kernel edges. The remaining integrand is cubic, so two-point Gauss
integration is exact. This oracle shares neither clipping nor polygon moments
with the implementation.

| Check | Scope |
| --- | --- |
| Weighted geometry | Three backends; unequal widths, width equal to cell size, a sharp axis, full-period averaging and periodic translation |
| Live geometry derivatives | Position, both sizes and rotation; axis-aligned and rotated shapes; two independent perturbation sizes |
| Full-period limit | Constant fill `width*height/period_area`, exact mean derivatives, second image ring at the seam |
| Cell contact | An axis-aligned face crossing a cell boundary has the known continuous derivative; the difference between slopes at offsets ±delta is `2*delta/(hx*dx)` |
| Complete optical graph | Eight joint shape, epsilon, mu and internal-thickness derivatives; bottom conical Jones incidence; JAX and Torch versus two NumPy differences |
| Uniform magnetic limit | Full-period averaging gives the same six Jones/power outputs as a uniform epsilon/mu slab |
| Fixed trace | Positive smoothing supports a single JAX trace through angle zero and a periodic translation |

The independently integrated fractions agree within 3e-13 in float64.
Full-period mean derivatives agree within 4e-14. General higher derivatives
are not claimed, nor are unsmoothed axes guaranteed continuous at parallel
contacts. Constraints on projected extents still apply at every traced input.

## Keep the continuum target fixed

The structure is the [sharp rotation case](fdtd-rotation-gradients.md): width
0.3, height 0.18, angle 0.37, centre `(0.071,-0.047)`, epsilon 2.25, mu 1,
depth 0.2, periods `(0.6,0.6)`, top epsilon 1, bottom epsilon 1.21 and normal
incidence at wavelength 1. The translation widths are fixed at `(0.05,0.05)`.
All eight complex central-order reflection/transmission channels are retained.
Angle derivatives are per radian and width derivatives per physical length.

The independent profile uses the analytic sinc transform of the rotated
rectangle, multiplied by one translation-box sinc on each axis. An inverse
Fourier series produces a scalar graded epsilon raster. It uses neither the
FDTD cell fractions nor their polygon moments. Independent pointwise box
intersections test the profile. Truncated-series overshoot is clipped to
`[0,1]`, with the approximation explicitly controlled by profile refinement.
RCWA uses this raster with Li factorization, orders ±8 and a 256 by 256 profile.

| Reference control | Complex amplitude change | Angle derivative change | Width derivative change |
| --- | ---: | ---: | ---: |
| Profile grid 256 to 512 | 8.21e-7 | 2.13e-6 | 1.78e-5 |
| Orders ±8 to ±10 | 4.08e-6 | 3.01e-5 | 0.000100583 |
| Difference step 1e-5 to 5e-6 | — | 5.61e-8 | 1.19e-7 |

At profile grids 128/256/512 the largest unclipped overshoot is
0.000505/0.000111/0.0000339, and clipping changes mean fill by
3.15e-6/4.92e-7/8.92e-8. The table measures reference repeatability;
it is not a rigorous continuum error bound.

FDTD uses physical PML thickness 1, duration 60, pulse width 2.5, fitted z
layers and directional extraction. Maximum absolute differences from the
matching graded RCWA reference are:

| Resolution | Complex amplitude | Angle derivative | Width derivative |
| --- | ---: | ---: | ---: |
| 20 | 0.00806968 | 0.00149886 | 0.0127111 |
| 30 | 0.00368232 | 0.000679959 | 0.00525892 |
| 40 | 0.00210467 | 0.000416159 | 0.00309728 |

All three errors decrease across these grids for this fixed graded profile.
Extending resolution-40 FDTD duration from 60 to 80 changes amplitudes by
1.25e-11 and derivatives by 1.83e-10. The remaining differences are not explained
by an insufficient runtime. This does not establish every-grid monotonicity
or a general convergence order.

A different question compares the graded continuum with the sharp one. Against
analytic sharp RCWA with orders ±10 and normal grid 2048, the profile alone
changes amplitudes by 0.00607058, angle derivatives by 0.00175621 and width
derivatives by 0.0152970. These are material-profile biases, not FDTD errors.
Initial sharp-target probes also show why averaging is not a universal accuracy
fix: at resolution 24, widths of half/one/two cells give amplitude differences
0.00957335/0.0107223/0.0137023 and width-derivative differences
0.0186519/0.0218747/0.0304511 from a sharp orders-±10, normal-grid-1024 target.
The smoothing width belongs in an accuracy study alongside the mesh.

## Independent Meep checks

Meep 1.34.0 receives the same independent sinc profile through a documented
MaterialGrid inside a full-period Block. `beta=0` and `do_averaging=False`
retain the already graded scalar profile. The public `get_epsilon_grid` API
checks the prescribed epsilon at nonuniform x/y sample locations for every
perturbed run. Cell-centred profile arrays give agreement within 4.45e-16;
this check catches both axis swaps and an incorrect endpoint interpretation.
The generator records its maximum probe error and sampling convention.

An initial candidate enabling additional MaterialGrid averaging failed the
perturbation control by more than 12 in a complex derivative. It was rejected
as a reference. Disabling that additional averaging passed the controls below;
this observation concerns this graded-profile setup, not a general statement
about Meep subpixel smoothing. Only documented public APIs were used to diagnose
it. No Meep implementation or excluded Meep-derived FDTDX source was inspected.
The public [MaterialGrid and geometry-query documentation](https://meep.readthedocs.io/en/master/Python_User_Interface/#materialgrid)
states the available material interpolation and projection controls.

The cell, source, phase calibration, two incident bases and full-plane DFT
quadrature follow the sharp generator. Resolutions 20/40, durations 80/100
and perturbations 1e-4/5e-5 are retained. A second profile-grid-512 run at
resolution 20 separates profile representation from the FDTD cell refinement.

| Meep control or difference from graded RCWA | Resolution 20 | Resolution 40 |
| --- | ---: | ---: |
| Duration change, amplitude | 7.91e-12 | 1.45e-11 |
| Duration change, derivative | 9.20e-11 | 2.01e-10 |
| Perturbation-size change, derivative | 5.75e-7 | 1.40e-5 |
| Amplitude difference | 0.0157287 | 0.00764161 |
| Angle derivative difference | 0.00528222 | 0.00135572 |
| Width derivative difference | 0.0807183 | 0.0518944 |

The profile-grid-256 to 512 change at resolution 20 is 9.73e-6 in amplitude,
2.62e-5 in angle derivative and 0.000418723 in width derivative. The grid-512
archive also passes duration and perturbation controls (maximum derivative
changes 9.25e-11 and 3.02e-6). Representation of the independent profile is
better controlled than the remaining Meep FDTD cell error in this case.

At resolution 40, Wavebit and Meep differ by 0.00757375 in amplitude,
0.00157206 in angle derivative and 0.0500235 in width derivative. The last gap
is substantial, despite the good repeatability of each calculation. The two
solvers use different interface discretizations, and this MaterialGrid setup
has its additional averaging disabled. No precise continuum width-gradient
oracle can be claimed from these Meep grids. The regression enforces Meep's
independent profile, time, perturbation and refinement controls; Wavebit's
accuracy against the matching graded continuum is checked with the separate
Fourier reference. It does not equate local derivative repeatability with
continuum accuracy or hide the width discrepancy behind the amplitude result.

The Meep derivatives on this page are central differences of forward solves.
The separate [adjoint comparison](fdtd-adjoint-validation.md) executes
`meep.adjoint` and measures its difference from those discrete derivatives.
These methods must not be conflated.

## Reproduce

Measured on an arm64 CPU with Python 3.12.9, NumPy 2.5.1, JAX 0.11.0 x64 and
Torch 2.13.0 float64. No GPU execution is established. From the repository root:

```shell
python -m studies.fdtd_rotation_smoothing --output /tmp/rotation-smoothing.json --meep-reference tests/references/golden_meep_rotation_smoothing.npz
pytest -q tests/test_fdtd_rotation_smoothing.py tests/references/test_meep_rotation_smoothing.py
```

Generate candidates in a separate Meep installation:

```shell
python tests/references/interface/meep_rotation.py --smoothing .05 --resolution 20 --output /tmp/meep-rotation-smoothing-coarse.npz
python tests/references/interface/meep_rotation.py --smoothing .05 --resolution 40 --output /tmp/meep-rotation-smoothing.npz
python tests/references/interface/meep_rotation.py --smoothing .05 --resolution 20 --profile-grid 512 --output /tmp/meep-rotation-smoothing-profile.npz
```

Independent references must pass their own controls before replacing frozen
archives. Close amplitudes alone are insufficient evidence for shape-gradient
accuracy. Rotated effective tensors, multiple shapes per area layer, curved
boundaries, dispersive recipe materials and variable total depth remain outside
this feature.

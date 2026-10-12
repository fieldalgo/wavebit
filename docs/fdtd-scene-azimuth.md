# Azimuth derivatives on one transverse mesh

Written: 2026-09-09

Targets:
- `src/wavebit/_core/scene_fdtd.py`
- `src/wavebit/_core/conventions.py`
- `src/wavebit/main.py`
- `tests/test_fdtd_scene_azimuth.py`

TL;DR: the explicit-backend scene adapter now differentiates phi jointly with
theta and material parameters. Array phi uses full y-period sampling even at
zero, so an angle sweep stays on one discrete experiment. The graph includes
the rotating current bases and incident polarization calibration as well as
the wavevector. Exact discrete dispersion, field rotations, axis exchange and
two finite-difference steps check the result.

## Why the array/scalar distinction matters

The previous y-invariant shortcut used two cells when phi was zero, but sampled
the full declared y period at nonzero phi. Differentiating through that branch
would change the mesh, time step and duration in updates at zero. It would
also make array shapes depend on traced data; the
[JAX transformation documentation](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#dynamic-shapes)
explains that restriction.

The [public contract](guide/time-domain.md#incidence-angle-derivatives) therefore
treats array phi as a request for a fixed full-period mesh. No branch depends
on the array's current value, including in eager NumPy or Torch. Every array
angle requires an explicit diffraction window. Array phi also requires
Jones/psi polarization, since the scalar TE/TM convention fixes the x-z plane.

| One-row scene with declared Ly=0.5 and Ry=10 | y cells |
| --- | ---: |
| Plain scalar phi=0 | 2, preserving the original shortcut |
| Plain scalar phi=0.2 | 5 |
| Array phi=-0.2, 0, or 0.2 | 5 in all three runs |

When only an x period is declared, the auxiliary y period remains `2/Ry` with
two cells. Crossed rasters already use the full declared y period. Scalar
nonzero and corresponding array angles give identical NumPy scattering in
the checked scenes. Array angles preserve mesh nodes, source/PML/measurement
positions, time step and update count when crossing zero. A JIT test passes
through negative, zero and positive azimuth with one trace, and its derivative
agrees with same-mesh finite differences at each point.

The change does not turn y-invariant monitor slices into full-y records or
change their power-per-unit-y convention. These measurements hold half-space
coefficients fixed; the [half-space extension](fdtd-halfspace-gradients.md)
subsequently adds their derivatives. Geometry, mesh targets and wavelength
remain static. No native Maxwell/ADE/CPML update was duplicated or changed
for the azimuth parameter.

## Rotating the source as well as the wavevector

At fixed theta, phi rotates the transverse Bloch vector and the incident s/p
frame. Both current bases must rotate, and so must the conversion of each
uniform-medium reference run to incident s/p amplitudes. Keeping any of those
constant differentiates a different experiment.

At theta=0 the transverse Bloch vector is zero for every phi. The incident
polarization still rotates: a unit s wave has tangential E proportional to
`(-sin(phi), cos(phi))`. Tests use this independent identity on frequency
monitors and carrier-calibrated cropped time records, from both incidence
sides. Their derivatives must rotate as `(-cos(phi), -sin(phi))`, even though
the uniform planar medium's scalar transmission stays constant.

The outgoing real-wavevector frame retains the coordinate-axis treatment from
the [theta extension](fdtd-scene-angles.md). At its origin the explicitly
specified phi rotates the frame; this is not an arbitrary derivative of a
directionless transverse vector.

## Independent discrete and continuum checks

All measurements here use macOS arm64 CPU, NumPy, JAX x64 and Torch float64.
The uniform-medium oracle differentiates the Yee dispersion expression
directly, without calling the solver's propagation or projection helpers.
It includes both theta and phi and the broadband-angle diagnostic. For phi,
`d kx/d phi = -ky` and `d ky/d phi = kx` at fixed theta.
The largest absolute Jacobian discrepancy was `3.99e-12` on JAX and
`3.96e-12` on Torch, for each incidence side.

The vacuum case has theta=0.6, phi=0.6, wavelengths 0.98/1.02, depth 0.2,
periods 0.2/0.3, z resolution 10 and x/y targets 20/5. PML has 10 cells,
Gaussian width is 4 and duration is 80 central periods. Directional extraction
is used. The unequal cells make the exact discrete azimuth derivative large
enough that a missing phi graph cannot pass the comparison tolerance.
The wider envelope and longer run avoid the residual seen in the initial
width-2, 50-period experiment, which missed the `2e-6` phase criterion with
an error of `1.76e-5`. Source-band glancing frequencies and finite duration
must be controlled before interpreting a derivative comparison.

| Check | Criterion |
| --- | --- |
| Uniform discrete propagation, both incidence sides and AD backends | Real/imaginary s/p transmission and angle Jacobians agree with the independent dispersion derivative within `2e-5`. |
| Normal-incidence field rotation, both sides and AD backends | Calibrated monitor/record values agree within `2e-11`, derivatives within `2e-10`. |
| Crossed magnetic scene exchanged across x/y | Diagonal Jones amplitudes are preserved and off-diagonal amplitudes change sign; their phi derivatives obey the same identity with the angle chain rule within `1e-10`. |
| Crossed dispersive bottom scene with grading, source notch, 3x3 order window and records | Joint theta/phi/epsilon/pole Jacobians agree with finite differences at steps `2e-4` and `1e-4`, at relative `5e-5` and absolute `3e-8`. Both electric and directional extraction are checked. |
| Nonfinite phi and azimuth-dependent exact grazing | Eager input guards or JIT `stable=False` prevent valid-looking scattering; invalid normalized fields are also masked. |

An isotropic planar continuum has no azimuth dependence, but unequal Yee cells
break that invariance at finite resolution. A separate magnetic Airy test uses
epsilon=2.25, mu=1.1, depth=0.2, theta=0.55, phi=0.41, wavelengths 0.98/1.02,
x period 0.05, x/y targets `(1.5R, 0.75R)`, layer alignment and directional
extraction. PML thickness is one length unit, pulse width is 3 and duration
is 60 periods. A weighted complex-amplitude objective includes both s/p
reflection and transmission; its continuum phi derivative is exactly zero.

| z resolution R | Objective error against Airy | Computed phi derivative per radian |
| --- | ---: | ---: |
| 40 | 0.0069072082 | -1.8007846e-5 |
| 80 | 0.0017150987 | -4.4806241e-6 |

Both errors fall by about four. A nonzero finite-grid planar phi sensitivity
can therefore be a correctly differentiated discretization error. Independent
grid/time convergence remains necessary; autodiff agreement with finite
differences alone does not establish physical accuracy.

These checks do not establish sensitivities near unresolved diffraction
thresholds, sharp dispersive corners or long-lived resonances. Float32,
Torch compilation and actual GPU execution remain outside this evidence.
No external solver implementation was consulted or incorporated for this
extension. Existing Meep public-API data remain part of the scene regressions.

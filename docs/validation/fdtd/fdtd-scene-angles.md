# Fixed-grid incidence-angle sensitivities

Written: 2026-09-09

Targets:
- `src/wavebit/fdtd/_periodic/scene.py`
- `src/wavebit/_common/conventions.py`
- `src/wavebit/compat/legacy.py`
- `tests/fdtd/test_fdtd_scene_angles.py`

TL;DR: the explicit-backend scene adapter now differentiates its scalar `theta`
at fixed grid and azimuth. The graph includes the source, Bloch boundaries,
outgoing-wave decomposition and incident calibration. Independent uniform-grid
dispersion and differentiated Airy expressions check the resulting sensitivity.
Real outgoing s/p frames also retain their derivatives on either transverse
axis, where an angle-ratio representation previously discarded a derivative.

## What is held fixed

The [usage contract](../../guide/fdtd/time-domain.md#incidence-angle-derivatives) requires
an explicit diffraction window for every array-valued theta. No order count may
be inferred from a differentiable scalar. Plain scalar numbers retain automatic
selection. These theta measurements hold phi, geometry, half spaces, cell
counts, source/monitor locations, PML, time step and run length fixed. The
subsequent [azimuth extension](fdtd-scene-azimuth.md) also differentiates phi,
and the [half-space extension](fdtd-halfspace-gradients.md) differentiates
exterior coefficients with explicit nominal mesh indices. Geometry stays static.
Array-backed geometry does not become differentiable by association.

The transverse wavevector uses the incident index and the midpoint wavelength
of the requested band. Each wavelength therefore has its own physical angle,
and all change together when theta changes. This is the fixed-wavevector pulse
convention described in the [Meep public FAQ](https://meep.readthedocs.io/en/latest/FAQ/).
Only public explanatory documentation was consulted for this change; no Meep
implementation was consulted or incorporated.

## Terms that cannot be frozen

| Term | Consequence of freezing it during an angle derivative |
| --- | --- |
| Staggered source phase and transverse Bloch wrap | The launched field no longer follows the requested transverse momentum. |
| Spatial Fourier transform and Yee interpolation | A changed field is projected onto the old diffraction basis. |
| Discrete E/H admittance and propagation phase | Direction separation or the phase reference responds to a different angle. |
| Incident two-basis calibration | The derivative includes a changing source amplitude instead of unit incident E. |
| Outgoing s/p frame and power factors | Cartesian components or unweighted amplitudes are mistaken for physical scattering sensitivities. |
| Time-record carrier calibration | Recorded fields retain a frozen normalization while the illumination changes. |

All these terms use the same Array API path as the material-dependent fields.
No JAX- or Torch-specific Maxwell implementation was added. Static raster
overlaps, source timing and mesh construction still use host arithmetic.
Invalid abstract angles accumulate into the scene validity flag: scattering,
normalized monitors and time records are masked, even if the underlying
time-update stability check alone would have passed.

## Axis crossings

For real transverse vectors the frame is directly
`(cos(varphi), sin(varphi)) = (kx, ky) / sqrt(kx**2 + ky**2)`.
At `(kx, ky) = (0, 2)`, the derivative of its cosine with respect to kx is
`1/2`. An atan ratio with a constant right-angle fallback returns the correct
value but a zero derivative there. Normalization gives `1/2` on both JAX and
Torch to the test's `1e-15` absolute tolerance, including the negative y axis
and the corresponding sine derivative on the x axis.

At the origin the caller's phi supplies the frame. The theta checks hold phi
fixed; a phi derivative rotates the frame. The implementation
guards the unused zero-norm square root so its undefined derivative cannot
contaminate a valid gradient. This is the derivative along a fixed incident
ray, not a definition for arbitrary two-dimensional perturbations at the
origin. The complex-angle RCWA frame helper is unchanged.

## Validation and scope

The checks run on macOS arm64 CPU with NumPy, JAX x64 and Torch float64. The
uniform-medium oracle differentiates the discrete dispersion relation directly:
with `Omega = 2 sin(omega dt/2)/dt`, `Kx = 2 sin(kx dx/2)/dx` and similarly Ky,
`kz = 2 asin(dz sqrt(Omega**2 - Kx**2 - Ky**2)/2)/dz`. No FDTD extraction or
propagation helper supplies this expected answer.

| Check | Criterion |
| --- | --- |
| Uniform vacuum, two wavelengths, conical illumination, unequal x/y cells, both incidence sides, both AD backends | Real/imaginary s/p transmission derivatives agree with independent Yee dispersion within `2e-5`; the wavelength-angle derivatives obey the fixed-wavevector formula. |
| Magnetic slab, epsilon 2.25, mu 1.1, thickness 0.2, theta 0.3, phi 0.4, wavelengths 0.98/1.02 | The weighted complex-amplitude objective and its angle derivative approach independently differentiated Airy values when z resolution increases 40 to 80. |
| Crossed dispersive bottom scene, graded mesh, independent x/y cells, source notch, retained 3x3 order window | Joint theta/epsilon/mu/pole Jacobians agree with centered finite differences at relative `4e-5`, absolute `2e-8`, including normalized monitors, cropped records and prefix diagnostics. Both extraction methods are checked. |
| Normal incidence in a crossed scene | Nonzero diffraction-axis derivatives agree with a second-order one-sided angle difference; scalar planar power has a finite zero derivative. |
| Invalid ranges, exact physical/discrete cutoffs and nonpropagating broadband incidence | Concrete calls refuse the request; JIT calls cannot report stable scattering. |
| Two angles on one explicit window | Mesh, source, planes, time step, step count and order indices are identical. |

The largest uniform-medium phase-derivative error was `6.02e-9` for each
incidence side and each AD backend. The Airy objective was
`0.1558465179662445`, with derivative `-0.4148107334364167` per radian:

| z resolution | Objective absolute error | Angle derivative absolute error |
| --- | ---: | ---: |
| 40 | 0.0078263190 | 0.0025780961 |
| 80 | 0.0019437343 | 0.0006402016 |

Both errors decrease by about four when the mesh is refined by two. The
derivative comparison is independent of the same-grid finite-difference check;
an implementation can differentiate its own incorrect forward model exactly.

For the Airy check, transverse targets are `(1.5R, 0.75R)`, the x period is
0.05, PML thickness is one length unit, the Gaussian width is 3 and duration
is 60 central periods. Layers are aligned and E/H extraction is directional.
The objective combines both s/p reflection and transmission at both wavelengths
using the complex weights specified in the test. This checks phase sensitivity
as well as amplitude sensitivity.

Finite-difference agreement establishes the derivative of a particular discrete,
finite-duration experiment. It does not establish continuum or infinite-time
convergence. The Airy refinement is a separate bounded check. Near-Rayleigh
thresholds, long-lived modes, sharp metal corners, general dispersive interfaces,
float32 sensitivities and actual GPU execution need their own evidence. Boolean
channel classifications do not define derivatives across a changing passband.

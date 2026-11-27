# Propagating outgoing fields

`wavebit.postprocess.propagation` continues a solver's outgoing complex electric
field through a uniform exterior medium. It does not rerun the structure solver.

```python
from wavebit import Units
from wavebit.postprocess import propagation

view = propagation.propagate(
    result.transmission,
    target=propagation.Plane(x=[0., .1], y=[0., .1], distance=10.,
                             units=Units(length="um")),
    method="orders", chunk_size=256,
)
E = view.field.values  # (wavelength, y, x, xyz), V/m
signal = abs(E)**2     # component-wise V^2/m^2, not power flux
```

The input is an existing `observations.DiffractionOrders`, with Cartesian CW
amplitudes, signed normal and transverse wavevectors, reference plane, lateral
origin, outward normal, units and continuum/time-convention metadata. RCWA
reflection and transmission results supply this contract. Efficiency-only results
and raw pulse spectra cannot supply the missing complex phase or CW calibration.

The target x/y axes can be arbitrary nonempty finite real vectors; they need not
be FFT bins or equally spaced. Distance is nonnegative and measured outward from
the port, so bottom-port propagation decreases public z. Coordinates and
wavelengths in the returned `FieldSamples` use the target length unit. The source
reference and metadata, including power availability, are retained in diagnostics.
Invalid source amplitudes propagate as NaN; they are never replaced by zero.

The synthesis sums each vector amplitude times
`exp(i*(kx*(x-x0) + ky*(y-y0) + kz*delta_z))`. Normal wavevectors already carry
the public Cartesian sign; `delta_z = outward_normal_z * distance`. Evanescent
orders decay in that direction. This path adds neither a finite aperture nor
edge scattering to an infinite periodic solution.

## Supported scope and refusals

This first path supports z-normal ports, real transverse wavevectors and uniform,
isotropic, nonmagnetic, positive lossless exterior media. `method="orders"` consumes continuum diffraction orders. A matching Python
result class alone does not establish compatibility: incomplete metadata and Yee
dispersion are refused on this path. For sampled calibrated fields, use the ASM
path below and explicitly accept its continuum approximation.

For direct orders, NumPy, JAX and PyTorch CPU eager evaluation are supported. Amplitudes, distance
and the upstream RCWA design graph remain differentiable. Abstract JAX JIT
validation is explicitly refused; GPU and batched transforms are not validated.
`chunk_size` is a positive integer bounding each forward observation-point block;
it does not bound the full output or automatic differentiation storage.

Failures use `ValidationError`/`UnsupportedFeatureError` with
`propagation.input_contract`, `.normalization`, `.convention`, `.direction`,
`.sampling` or `.method`. Power flux requires magnetic fields and is not computed.

`tests/postprocess/test_propagation.py` checks coherent interference, outgoing
signs, evanescent decay, units, reference origins, NaNs, RCWA boundary fields,
backend agreement and amplitude/distance/upstream design derivatives against
independent analytic expressions or central differences.

## Sampled-plane angular spectrum (ASM)

Use `method="asm"` for one complex electric component sampled on a uniform
Cartesian plane. This continues a one-way solution of the homogeneous scalar
Helmholtz equation; it does not solve scattering, infer polarization, or reconstruct
magnetic fields. The exterior index must be real and positive. Coordinates,
vacuum wavelength and propagation distance share an explicit length unit.
The time convention is `exp(-i*omega*t)`.

```python
import numpy as np
from wavebit import Units
from wavebit.postprocess import propagation

x = np.linspace(-40., 40., 256, endpoint=False)
source = propagation.PhasorPlane(
    values=np.exp(-(x/10.)**2)[None, :].astype(complex),
    x=x, y=np.array([0.]), z=0., wavelength=.6,
    units=Units(length="um"), medium_n=1.,
    component="Ey", direction="+z", boundary="isolated",
)
view = propagation.propagate(
    source,
    target=propagation.Plane(x=x, y=[0.], distance=100., units=source.units),
    method="asm", numerics=propagation.ASM(),
)
E = view.field.values  # (1, y, x, 1), V/m
```

The target keeps the source x/y grid; interpolation and tilted planes are refused.
Distance is outward and nonnegative: `direction="-z"` decreases public z while
evanescent waves still decay. The unshifted FFT multiplier is
`exp(i*sqrt((2*pi*n/wavelength)**2-kx**2-ky**2)*distance)`, with the shared
outgoing square-root convention. At a cutoff the forward field is defined, but
wavelength/index derivatives are singular; inspect
`diagnostics["parameter_gradient_defined"]` before using those derivatives.

Choose the lateral boundary deliberately:

- `boundary="periodic"`: no zero padding. `carrier_kxy=(kx, ky)`, in radians
  per coordinate length unit, describes a Bloch carrier. The FFT acts on the
  carrier-removed envelope. It does not add the edges of a finite device.
- `boundary="isolated"`: the sampled field is zero outside the input window.
  `ASM(padding=None)` pads ceil(N/2) cells on each side of each non-singleton
  axis. Set `padding=(pad_y, pad_x)` explicitly to check convergence.
- A singleton axis represents an invariant direction and cannot be padded.
  Neither default padding nor fine pixels alone guarantee convergence: enlarge
  the input window and padding, then refine the spacing. Finite FFT windows can
  wrap diffracted light back into the observation region.

## Connecting FDTD phasors

`fdtd.periodic.Result.planes[name]` supplies source-normalized, collocated
phasors. Select one wavelength and electric component explicitly:

```python
source = propagation.from_phasor(
    fdtd_result.planes["outside"], component="Ey", wavelength_index=0,
    medium_n=1., direction="-z", boundary="periodic",
    assume_outgoing=True,
)
view = propagation.plan_asm(source)(10.)
```

The caller must establish that the plane is in a homogeneous exterior and carries
only the selected outgoing wave. An electric field alone cannot separate incident
and reflected waves; `assume_outgoing=True` records that assertion, not a
decomposition. Use the transmitted exterior, beyond sources and interfaces.
Raw pulse Fourier spectra, time series, missing calibration provenance and
nonphysical electric units are refused. No new pulse normalization is inferred.

This adapter uses continuum dispersion after the plane. It preserves FDTD's
calibration provenance but does not correct Yee numerical dispersion or claim
exact continuation of the discrete time-domain solution. Refine the originating
FDTD grid for the intended accuracy. Diagnostics report
`continuum_approximation=True`; `abs(E)**2` remains V²/m², not power flux.

## Repeated execution, JIT and devices

Build `prepared = propagation.plan_asm(source)` eagerly once, then call
`prepared(distance, values=..., wavelength=..., medium_n=...)`. The optional
values must match the prepared shape, backend, dtype and device. Fixed grid,
carrier and padding are not dynamic design variables of this plan. Rebuild the
plan when those change.

NumPy, JAX and PyTorch retain complex arithmetic and eager derivatives. A prepared
JAX plan can be called inside `jax.jit`; use JAX arrays for its source and enable
64-bit JAX before constructing arrays when double precision is needed. Eager
invalid scalar parameters raise. Invalid dynamic parameters under JIT return
NaN fields and `diagnostics["input_valid"]=False`. Nonfinite fields also return
NaNs and a false flag. Always inspect that flag in compiled workflows.

The ASM kernel preserves the source device; the tests include a conditional CUDA
forward/gradient check. A CPU-only CI pass does not establish GPU validation or a
performance benefit. Torch compilation runs this entry point eagerly under the
shared compilation policy. The direct-order path retains its existing CPU/JIT
restrictions.

`tests/postprocess/test_asm.py` checks analytic plane waves and evanescent decay,
zero distance, unit conversion, Gaussian paraxial intensity and grid/padding
convergence, an actual calibrated FDTD plane, backend agreement, JIT admission
and automatic derivatives against central differences. Plane-wave, decay and
backend comparisons use complex128 with 1e-10 tolerances; Gaussian paraxial
intensity uses 1e-3 and normalized gradient error uses 1e-6.

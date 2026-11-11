# Propagating outgoing diffraction orders

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
isotropic, nonmagnetic, positive lossless exterior media. Only `method="orders"`
is implemented. Sampled-plane ASM and FDTD producer normalization/dispersion
adapters are separate future work. A matching Python result class alone does not
establish compatibility: incomplete metadata and Yee dispersion are refused.

NumPy, JAX and PyTorch CPU eager evaluation are supported. Amplitudes, distance
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

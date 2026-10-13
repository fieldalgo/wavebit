# First RCWA calculation

Describe the physical structure and illumination with `rcwa.Problem`. Keep the Fourier truncation and factorization in `rcwa.Numerics`, then check, prepare and run. Both solver modules use the same lifecycle and the same physical material definitions.

This example uses micrometres, a dielectric cylinder in a single patterned layer, and three vacuum wavelengths. It requests diffraction orders and interior fields. The small Fourier order and 64-point NVF grid make a quick demonstration; they are **not a convergence recommendation**.

```python
import numpy as np
from wavebit import Material, Units, Job, Execution, rcwa

problem = rcwa.Problem(
    units=Units(length='um'),
    stack=rcwa.Stack(
        period=(.6, .6),
        top=Material(epsilon=1.),
        bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(
            thickness=.18,
            background=Material(epsilon=1.),
            regions=(rcwa.Region(
                geometry=rcwa.Circle(radius=.12),
                material=Material(epsilon=12.25)),)),)),
    excitation=rcwa.Excitation(
        wavelength=np.array([.85, .9, .95]),
        sources=(rcwa.PlaneWave(side='top', polarization='s', amplitude=1.),)),
    outputs=rcwa.Outputs(fields=True))

# These small orders make this a quick demonstration, not a convergence claim.
numerics = rcwa.Numerics(orders=(1, 1), factorization='nvf', shape_nvf_grid_size=64)
report = rcwa.check(problem, numerics=numerics)
report.raise_if_invalid()
prepared = rcwa.prepare(problem, numerics=numerics)
result = prepared.run(problem)
print(result.R.shape)  # (3, 3, 3): wavelength, order_v, order_u
print(result.transmission.order(0, 0).efficiency)
print(result.transmission.at_wavelength(1).wavelengths)
samples = result.field(x=np.array([0., .1]), y=np.array([0.]), z=np.array([.09]))
print(samples.values.shape)  # (3, 1, 1, 2, 6)
print(samples.units)         # E: V/m, H: A/m, coordinates: um

new_problem = problem.replace(excitation=problem.excitation.replace(
    wavelength=np.array([.86, .91, .96])))
new_result = prepared.run(new_problem)
# result and its retained fields still belong to the first calculation.
job = Job(solver='rcwa', problem=problem, numerics=numerics,
          execution=Execution(backend='numpy'))
assert Job.from_json(job.to_json()).key == job.key
print(new_result.T.sum(axis=(1, 2)))
```

`result.R` and `result.T` have axes `(wavelength, order_v, order_u)`, including a wavelength axis for a single wavelength. `result.transmission.order(0, 0)` selects a physical diffraction order. `result.transmission.at_wavelength(1)` selects the second wavelength without dropping its spectral metadata.

Field values have component order `(Ex, Ey, Ez, Hx, Hy, Hz)`. Public fields use **E in V/m and H in A/m**; the internal normalized magnetic field does not leak into this result. Coordinates retain the problem's length unit. The example's field shape is `(3, 1, 1, 2, 6)` for three wavelengths, one z, one y and two x positions.

A prepared calculation accepts the complete live problem on every run. The second run replaces only the wavelengths, preserving their count and the static structure. Earlier results and their retained fields still belong to the earlier problem. Structural changes require a new preparation; see [lifecycle and reuse](../guide/common/lifecycle.md).

The layers in `rcwa.Stack.layers` are ordered **bottom to top**. `top` and `bottom` mean physical half-spaces. Angles are in radians, x/y are centered in the cell, and z increases upward from the bottom of the finite stack.

Continue with [materials](../guide/common/materials.md), [RCWA problem declarations](../guide/rcwa/index.md), and [the guided curriculum](../learn/fundamentals/index.md).

## Continue with a complete experiment

The [film optimization example](https://github.com/fieldalgo/wavebit/blob/develop/examples/rcwa/film_optimization.py)
uses the current API for a thickness derivative and a bounded optimization. It checks the
JAX derivative against an independent Airy calculation and the optimum against the
quarter-wave thickness. The [shared film and grating comparison](https://github.com/fieldalgo/wavebit/blob/develop/examples/cross_solver/film_and_grating.py)
builds both solver problems from the same physical materials and geometry, aligns amplitude
reference planes, and compares complex diffraction orders under FDTD grid/time refinement.
These are bounded examples with explicit checks, not convergence settings for every device.

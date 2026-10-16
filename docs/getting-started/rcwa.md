# First RCWA calculation

Calculate reflection and transmission at three wavelengths for a periodic layer of dielectric cylinders. Start with NumPy; no automatic-differentiation backend is required.

## 1. Define the structure and run

Lengths are in micrometres. The cylinder layer is between air and a dielectric substrate. `Problem` describes the physics; `Numerics` controls the Fourier approximation. These small Fourier orders and the 64-point NVF grid are for a quick demonstration, **not converged design settings**.

```python
import numpy as np
from wavebit import Material, Units, rcwa

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
```

## 2. Read reflection and transmission

Sum the diffraction orders to get total reflected and transmitted power fractions at each wavelength:

```python
print("R:", result.R.sum(axis=(1, 2)))
print("T:", result.T.sum(axis=(1, 2)))
print("R + T:", (result.R + result.T).sum(axis=(1, 2)))
```

Each array contains one value for each of 0.85, 0.90 and 0.95 µm. For this lossless structure, R + T should be close to 1. Energy balance alone does not establish convergence; increase the Fourier orders and compare the results.

To select one order or wavelength:

```python
print(result.R.shape)  # (3, 3, 3): wavelength, order_v, order_u
print(result.transmission.order(0, 0).efficiency)
print(result.transmission.at_wavelength(1).wavelengths)
```

`R` and `T` have axes `(wavelength, order_v, order_u)`, even for a single wavelength. Order (0, 0) is the undiffracted order.

## 3. Optional: sample the interior field

The run above retains fields with `outputs=rcwa.Outputs(fields=True)`. Sample them at the following coordinates:

```python
samples = result.field(x=np.array([0., .1]), y=np.array([0.]), z=np.array([.09]))
print(samples.values.shape)  # (3, 1, 1, 2, 6)
print(samples.units)         # E: V/m, H: A/m, coordinates: um
```

The axes are `(wavelength, z, y, x, component)` and components are `(Ex, Ey, Ez, Hx, Hy, Hz)`. Electric fields use V/m and magnetic fields use A/m; coordinates use the problem's length unit.

## Continue learning

Layers are ordered **bottom to top**. The cell is centered in x/y, and z increases upward from the bottom of the finite stack. See [coordinate conventions](../learn/fundamentals/conventions.md), [materials](../guide/common/materials.md) and [RCWA fundamentals](../learn/fundamentals/index.md).

### Reuse a prepared calculation

For compatible changes, pass the complete updated problem to the prepared calculation. Here only the wavelengths change, with the same count. Earlier results remain associated with the original problem. Structural changes require a new preparation; see [lifecycle and reuse](../guide/common/lifecycle.md).

```python
new_problem = problem.replace(excitation=problem.excitation.replace(
    wavelength=np.array([.86, .91, .96])))
new_result = prepared.run(new_problem)
# result and its retained fields still belong to the first calculation.
print(new_result.T.sum(axis=(1, 2)))
```

### Save a job description

For later serialization workflows, a `Job` keeps the problem, numerical settings and execution backend together:

```python
from wavebit import Job, Execution

job = Job(solver='rcwa', problem=problem, numerics=numerics,
          execution=Execution(backend='numpy'))
assert Job.from_json(job.to_json()).key == job.key
```

The [film optimization example](https://github.com/fieldalgo/wavebit/blob/main/examples/rcwa/film_optimization.py)
uses the current API for a thickness derivative and a bounded optimization. It checks the
JAX derivative against an independent Airy calculation and the optimum against the
quarter-wave thickness. The [shared film and grating comparison](https://github.com/fieldalgo/wavebit/blob/main/examples/cross_solver/film_and_grating.py)
builds both solver problems from the same physical materials and geometry, aligns amplitude
reference planes, and compares complex diffraction orders under FDTD grid/time refinement.
These are bounded examples with explicit checks, not convergence settings for every device.
